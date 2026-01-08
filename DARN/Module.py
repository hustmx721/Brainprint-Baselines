
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import *


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)




class ICRU(nn.Module):
    def __init__(self, channels, eeg_groups=4, size=None):  # 4
        super(ICRU, self).__init__()
        self.groups = eeg_groups
        self.size = size
        self.gate_treshold = 0.5
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channels // (2 * self.groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channels // (2 * self.groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channels // (2 * self.groups), self.size[-2], self.size[-1]))
        self.sbias = Parameter(torch.ones(1, channels // (2 * self.groups), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channels // (self.groups * 2), channels // (self.groups * 2))
        self.CRU = CRU(op_channel=channels // (2 * self.groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x
        
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xn = self.CRU(xn)[2]
        xs_gn = self.gn(x_1)
        xs = self.sweight * xs_gn + self.sbias
        xs = x_1 * self.sigmoid(xs)
        xs = self.CRU(xs)[2]
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class GlobalEncoder_EEGNet(nn.Module):
    def __init__(self, nfeatl):
        super(GlobalEncoder_EEGNet, self).__init__()
        self.c3 = nn.Conv2d(in_channels=nfeatl, out_channels=nfeatl, kernel_size=(1, 16), stride=1, bias=False,
                            groups=(nfeatl), padding=16 // 2)
        self.b3 = nn.BatchNorm2d(nfeatl)
        self.p3 = nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16))
        self.d3 = nn.Dropout(p=0.5)

    def forward(self, x):
        h3 = self.d3(self.p3(F.elu(self.b3(self.c3(x)))))
        h3_ = torch.flatten(h3, start_dim=1)
        return h3_





class Classifier(nn.Module):
    def __init__(self, nfeatr,num_class):
        super(Classifier, self).__init__()
        self.dense1 = nn.Linear(nfeatr, num_class)

    def forward(self, latent):
        out = self.dense1(latent)
        return out
class ChannelGate_sub(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=True):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm([in_channels // reduction, 1, 1])
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))




    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class CRU(nn.Module):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_kernel_size: int = 1,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=(group_kernel_size, 1), stride=1,
                             groups=up_channel // squeeze_radio)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        Y1, Y2 = torch.split(out, out.size(1) // 2, dim=1)
        return Y1, Y2, Y1 + Y2





class SFRU(nn.Module):
    """Spatial Feature Refinement Unit"""
    
    def __init__(self, channels, size):
        super(SFRU, self).__init__()
        self.layernorm = nn.LayerNorm([size[1], size[2], size[3]])
        self.se = ChannelGate_sub(in_channels=channels)
        self.conv_wise_1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                  kernel_size=(1, 1), stride=1, bias=False)
        self.conv_wise_2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                  kernel_size=(1, 1), stride=1, bias=False)
    
    def forward(self, x):
        x_la = self.layernorm(x)
        x_w = self.layernorm.weight
        x_out = x_la * torch.sigmoid(torch.mul(x_w, x_la))
        x_out_1, x_out_2, _ = self.se(x_out)
        x_out_1 = self.conv_wise_1(x_out_1)
        x_out_2 = self.conv_wise_2(x_out_2)
        x_out_w = torch.sigmoid(torch.mul(x_out_1, x_out_2))
        out = x_out * x_out_w
        return out



class DARN(nn.Module):
    """Dual Attention Refinement Network"""
    def __init__(self, in_channels, out_channels, D, fs, num_ch, eeg_groups, input_size):
        super(DARN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ch = num_ch
        self.eeg_groups = eeg_groups
        self.fs = fs
        self.D = D
        self.input_size = input_size
        self.c1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=(1, int(self.fs / 2)), stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(self.out_channels)
        self.c2 = Conv2dWithConstraint(in_channels=self.out_channels, out_channels=self.out_channels * self.D,
                                     kernel_size=(self.num_ch, 1), stride=1, padding=(0, 0),
                                     bias=False, max_norm=1)
        self.b2 = nn.BatchNorm2d(self.out_channels * self.D)
        size_2 = self.get_size_temporal(self.input_size)
        self.sfru = SFRU(channels=self.out_channels * self.D, size=size_2)
        self.icru = ICRU(channels=self.out_channels * self.D,eeg_groups=self.eeg_groups,size=size_2)
    
    def get_size_temporal(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        h1 = self.c1(data)
        h2 = self.c2(h1)
        size_2 = h2.size()
        return size_2
    
    def forward(self, input):
        h1 = self.b1(self.c1(input))
        h2 = self.b2(self.c2(h1))
        sfru_out = self.sfru(h2)
        icru_out = self.icru(h2)
        out = h2 + sfru_out + icru_out
        return out


