import sys, os
sys.path.append("/mnt/data1/tyl/UserID/baseline/frameworks/DRFNet")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import baseline.frameworks.DRFNet.myModule as md

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, out_channels, dropout_rate, D, eeg_groups, sa_groups):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.D = D
        self.eeg_groups = eeg_groups
        self.sa_groups = sa_groups
        self.local_enc = md.LocalEncoder_EEGNet_log(
            in_channels=self.input_size[0],
            out_channels=self.out_channels,
            D=self.D,
            fs=self.input_size[-1] // 4,
            num_ch=self.input_size[1],
            num_time=self.input_size[-1],
            eeg_groups=self.eeg_groups,
            sa_groups=self.sa_groups,
            input_size=self.input_size

        )

        size = self.get_size_temporal(self.input_size)
        self.local_filter_weight_1 = nn.Parameter(torch.FloatTensor(size[1], size[2]), requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight_1)
        self.local_filter_bias_1 = nn.Parameter(torch.zeros((1, size[1], 1), dtype=torch.float32), requires_grad=True)

        self.local_filter_weight_2 = nn.Parameter(torch.FloatTensor(size[1], size[2]), requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight_2)
        self.local_filter_bias_2 = nn.Parameter(torch.zeros((1, size[1], 1), dtype=torch.float32), requires_grad=True)

        self.OneXOneConv = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(size[1], size[1], kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(size[1]),
            nn.ELU()
        )

        self.BN = nn.BatchNorm2d(size[1])
        self.BN_R = nn.BatchNorm2d(size[1])
        self.BN_weight = nn.Parameter(torch.FloatTensor(size[1], size[2]), requires_grad=True)
        nn.init.xavier_uniform_(self.BN_weight)
        self.BN_bias = nn.Parameter(torch.zeros((1, size[1], 1), dtype=torch.float32), requires_grad=True)
        self.R_plus_weight = nn.Parameter(torch.FloatTensor(size[1], size[2]), requires_grad=True)
        nn.init.xavier_uniform_(self.R_plus_weight)
        self.R_plus_bias = nn.Parameter(torch.zeros((1, size[1], 1), dtype=torch.float32), requires_grad=True)
        self.layernorm1 = nn.LayerNorm([size[1], 1, size[2]])
        self.LA_weight = nn.Parameter(torch.FloatTensor(size[1], size[2]), requires_grad=True)
        nn.init.xavier_uniform_(self.LA_weight)
        self.LA_bias = nn.Parameter(torch.zeros((1, size[1], 1), dtype=torch.float32), requires_grad=True)
        self.reid_layer1 = md.ChannelGate_sub(in_channels=self.out_channels*self.D)
        self.reid_layer2 = md.ChannelGate_sub(in_channels=self.out_channels*self.D)
        self.reid_layer3 = md.ChannelGate_sub(in_channels=self.out_channels*self.D)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(in_channels=self.out_channels*self.D, out_channels=self.out_channels*self.D, kernel_size=1)

    def get_size_temporal(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        output = self.local_enc(data)
        output = output.squeeze(-2)
        size = output.size()
        return size

    def local_filter_fun(self, x, w, b):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = x.squeeze(-2)
        x = F.relu(torch.mul(x, w) - b)
        return x.unsqueeze(-2)



    def forward(self, input_tensor):
            localf = self.local_enc(input_tensor)
            source = localf
            localf_filted = self.local_filter_fun(localf, self.BN_weight, self.BN_bias)
            localf = localf + localf_filted
            localf = self.BN(localf) * self.sigmoid(self.local_filter_fun(localf,self.local_filter_weight_1,self.local_filter_bias_1))
            localf_R,_, _ = self.reid_layer1(localf)
            localf_R_plus_filted = self.local_filter_fun(source, self.local_filter_weight_2,
                                                         self.local_filter_bias_2)
            localf_R_plus = source + localf_R_plus_filted
            localf_R_plus_BN = self.layernorm1(localf_R_plus) * self.sigmoid(
                self.local_filter_fun(localf_R_plus, self.LA_weight, self.LA_bias))
            localf_R_plus_reid = localf_R_plus - localf_R_plus_BN
            localf_R_reid_useful, _, _ = self.reid_layer2(localf_R_plus_reid)
            local_ID = localf_R_plus_BN + localf_R_reid_useful
            local_ID,_,_=self.reid_layer2(local_ID)
            rele =  source+localf_R+local_ID
            return rele
