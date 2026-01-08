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
class sa_layer(nn.Module):
    def __init__(self, channel, sa_groups=4,size=None):#4
        super(sa_layer, self).__init__()
        self.groups = sa_groups
        self.size=size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * self.groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * self.groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * self.groups),  self.size[-2], self.size[-1]))
        self.sbias = Parameter(torch.ones(1, channel // (2 * self.groups), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // ( self.groups*2), channel // ( self.groups*2) )
        self.distill =ChannelGate_sub(in_channels=channel // (2 * self.groups),reduction=1)



    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # distill
        x_reid_0=x_0-xn
        x_reid_0_use,_,_=self.distill(x_reid_0)
        xn=xn+x_reid_0_use


        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # distill
        x_reid_1=x_1-xs
        x_reid_1_use,_,_=self.distill(x_reid_1)
        xs=xs+x_reid_1_use

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)


        return out
class gn_layer(nn.Module):
    def __init__(self, channel, sa_groups=4,size=None):#4
        super(gn_layer, self).__init__()
        self.groups = sa_groups
        self.size = size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.GN_weight = nn.Parameter(torch.FloatTensor(size[-2], size[-1]), requires_grad=True)
        nn.init.xavier_uniform_(self.GN_weight)
        self.GN_bias = nn.Parameter(torch.zeros((1, size[-2], 1), dtype=torch.float32), requires_grad=True)

        self.GN_weight_2 = nn.Parameter(torch.FloatTensor(size[-2], size[-1]), requires_grad=True)
        nn.init.xavier_uniform_(self.GN_weight_2)
        self.GN_bias_2 = nn.Parameter(torch.zeros((1, size[-2], 1), dtype=torch.float32), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (self.groups), channel // (self.groups))


    def local_filter_fun(self, x, w, b):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = x.squeeze(-2)
        x = F.relu(torch.mul(x, w) - b)
        return x.unsqueeze(-2)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_filted=self.local_filter_fun(x,self.GN_weight,self.GN_bias)
        x=x+x_filted
        # x=self.gn(x)*self.sigmoid(x)
        # modify on  12.02
        x=self.gn(x)*self.sigmoid(self.local_filter_fun(x,self.GN_weight_2,self.GN_bias_2))
        x = x.reshape(b, -1, h, w)
        return x
class LocalEncoder_EEGNet_log(nn.Module):
    def __init__(self,in_channels,out_channels,D, fs, num_ch, num_time,eeg_groups,sa_groups,input_size):
        super(LocalEncoder_EEGNet_log, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ch = num_ch
        self.num_time = num_time
        self.eeg_groups = eeg_groups
        self.fs = fs
        self.D=D
        self.input_size = input_size
        self.c1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels , kernel_size=(1, int(self.fs / 2)), stride=1, bias=False,
                            padding=(0, (int(fs / 2) // 2) - 1),groups=self.eeg_groups)  # 4[]
        self.b1 = nn.BatchNorm2d(self.out_channels)

        self.c2 = Conv2dWithConstraint(in_channels=self.out_channels, out_channels=self.out_channels * self.D, kernel_size=(self.num_ch, 1), stride=1,
                                       bias=False, groups=self.out_channels, padding=(0, 0), max_norm=1)
        self.b2 = nn.BatchNorm2d(self.out_channels * self.D)
        size = self.get_size_temporal(self.input_size)
        self.power = PowerLayer(dim=-1, length=4, step=int(4))
        self.sa = sa_layer(channel=self.out_channels * self.D, sa_groups=sa_groups, size=size)
        self.gn = gn_layer(channel=self.out_channels * self.D, sa_groups=sa_groups, size=size)
        self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.d2 = nn.Dropout(p=0.5)
        self.conv_wise = nn.Conv2d(in_channels=self.out_channels * self.D, out_channels=self.out_channels * self.D,
                                   kernel_size=(1, 1), stride=1, bias=False, groups=self.out_channels * self.D)

    def get_size_temporal(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        h1 = self.b1(self.c1(data))
        h2 = self.b2(self.c2(h1))

        size = h2.size()
        return size
    def forward(self, input):
        h1 = self.b1(self.c1(input))

        h2 = self.b2(self.c2(h1))


        h4 = self.d2(self.power(h2))
        h5 = self.d2(F.relu(self.p2(self.gn(h2))))
        h_out=torch.cat((h4,h5),dim=-1)
        return h_out


class LocalEncoder_EEGNet_log_v2(nn.Module):
    def __init__(self,in_channels,out_channels,D, fs, num_ch, num_time,eeg_groups,sa_groups,input_size):
        super(LocalEncoder_EEGNet_log_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ch = num_ch
        self.num_time = num_time
        self.eeg_groups = eeg_groups
        self.fs = fs
        self.D=D
        self.input_size = input_size
        self.c1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels , kernel_size=(1, int(self.fs / 2)), stride=1, bias=False,
                            padding=(0, (int(fs / 2) // 2) - 1),groups=self.eeg_groups)  # 4[]
        self.b1 = nn.BatchNorm2d(self.out_channels)

        self.c2 = Conv2dWithConstraint(in_channels=self.out_channels, out_channels=self.out_channels * self.D, kernel_size=(self.num_ch, 1), stride=1,
                                       bias=False, groups=self.out_channels, padding=(0, 0), max_norm=1)
        self.b2 = nn.BatchNorm2d(self.out_channels * self.D)
        size1,size2 = self.get_size_temporal(self.input_size)

        self.BN_weight = nn.Parameter(torch.FloatTensor(size1[-2], size1[-1]), requires_grad=True)
        nn.init.xavier_uniform_(self.BN_weight)
        self.BN_bias = nn.Parameter(torch.zeros((1, size1[-2], 1), dtype=torch.float32), requires_grad=True)

        self.BN_weight_2 = nn.Parameter(torch.FloatTensor(size2[-2], size2[-1]), requires_grad=True)
        nn.init.xavier_uniform_(self.BN_weight_2)
        self.BN_bias_2 = nn.Parameter(torch.zeros((1, size2[-2], 1), dtype=torch.float32), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

        self.power = PowerLayer(dim=-1, length=4, step=int(4))
        self.sa = sa_layer(channel=self.out_channels * self.D, sa_groups=sa_groups, size=size2)
        self.gn = gn_layer(channel=self.out_channels * self.D, sa_groups=sa_groups, size=size2)
        self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.d2 = nn.Dropout(p=0.5)
        self.conv_wise = nn.Conv2d(in_channels=self.out_channels * self.D, out_channels=self.out_channels * self.D,
                                   kernel_size=(1, 1), stride=1, bias=False, groups=self.out_channels * self.D)

    def get_size_temporal(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        h1 = self.c1(data)
        size1=h1.size()
        h2 = self.c2(h1)

        size2 = h2.size()
        return size1,size2
    def local_filter_fun(self, x, w, b):

        if x.size()[-2] == 1:

            w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
            x = x.squeeze(-2)
            x = F.relu(torch.mul(x, w) - b)
            return x.unsqueeze(-2)
        else:

            x = F.relu(torch.mul(x, w) - b)
            return x

    def forward(self, input):
        h1 = self.b1(self.c1(input))



        h2 = self.c2(h1)


        h4 = self.d2(self.power(h2))
        h5 = self.d2(F.relu(self.p2(self.gn(h2))))
        h_out=torch.cat((h4,h5),dim=-1)
        return h_out

class LocalEncoder_EEGNet(nn.Module):
    def __init__(self,in_channels,out_channels,D, fs, num_ch, num_time,eeg_groups):
        super(LocalEncoder_EEGNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ch = num_ch
        self.num_time = num_time
        self.eeg_groups = eeg_groups
        self.fs = fs
        self.D=D
        self.c1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels , kernel_size=(1, int(self.fs / 2)), stride=1, bias=False,
                            padding=(0, (int(fs / 2) // 2) - 1),groups=self.eeg_groups)  # 4[]
        self.b1 = nn.BatchNorm2d(self.out_channels)

        self.c2 = Conv2dWithConstraint(in_channels=self.out_channels, out_channels=self.out_channels * self.D, kernel_size=(self.num_ch, 1), stride=1,
                                       bias=False, groups=self.out_channels, padding=(0, 0), max_norm=1)
        self.b2 = nn.BatchNorm2d(self.out_channels * self.D)

        self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.d2 = nn.Dropout(p=0.5)

    def forward(self, input):
        h1 = self.b1(self.c1(input))
        h2 = self.b2(self.c2(h1))

        return h2
class PowerLayer(nn.Module):
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class GlobalEncoder_EEGNet(nn.Module):
    def __init__(self, num_ch, num_time, nfeatl):
        super(GlobalEncoder_EEGNet, self).__init__()
        self.c3 = nn.Conv2d(in_channels=nfeatl, out_channels=nfeatl, kernel_size=(1, 16), stride=1, bias=False,
                            groups=(nfeatl), padding=(0, 16 // 2))
        self.b3 = nn.BatchNorm2d(nfeatl)
        self.p3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
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



class Decomposer(nn.Module):
    def __init__(self, nfeat):
        super(Decomposer, self).__init__()
        self.nfeat = nfeat
        self.embed_layer = nn.Sequential(nn.Conv2d(nfeat, nfeat*2, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(nfeat*2), nn.ELU(), nn.Dropout())

    def forward(self, x):
        embedded = self.embed_layer(x)
        rele, irre = torch.split(embedded, [int(self.nfeat), int(self.nfeat)], dim=1)

        return rele, irre


class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=True):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm([in_channels//reduction, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
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

