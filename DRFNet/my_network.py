import torch
import torch.nn as nn
import torch.nn.functional as F
import Module as md
from feature_extractor import FeatureExtractor

class MyModel(nn.Module):
    def __init__(self, device, input_size, num_class, dropout_rate, out_channels, D, eeg_groups, sa_groups):
        super(MyModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        self.out_channels = out_channels
        self.D = D
        self.eeg_groups = eeg_groups
        self.sa_groups = sa_groups
        self.nfeatl = self.out_channels * self.D

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            input_size=self.input_size,
            out_channels=self.out_channels,
            dropout_rate=self.dropout_rate,
            D=self.D,
            eeg_groups=self.eeg_groups,
            sa_groups=self.sa_groups
        ).to(self.device)

        # Global encoder
        self.global_enc = md.GlobalEncoder_EEGNet(
            num_ch=self.input_size[1],
            num_time=self.input_size[-1],
            nfeatl=self.nfeatl
        ).to(self.device)

        # Calculate feature sizes
        self.nfeatl2, self.nfeatg = self.calculate_feature_sizes(self.input_size)

        # Classifier
        self.classifier = md.Classifier(
            self.nfeatg,
            self.num_class
        ).to(self.device)

        # Classification criterion
        self.cls_criterion = nn.CrossEntropyLoss()

    def calculate_feature_sizes(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2]))).to(self.device)
        rele= self.feature_extractor(data)
        size_temporal = rele.size()
        globalf = self.global_enc(rele)
        size_global = globalf.size()
        return size_temporal[-1], size_global[-1]


    def local_filter_fun(self, x, w, b):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = x.squeeze(-2)
        x = F.relu(torch.mul(x, w) - b)
        return x.unsqueeze(-2)

    def forward(self, input_tensor, label_src):
        rele= self.feature_extractor.forward(input_tensor)
        globalf = self.global_enc(rele)
        logits = self.classifier(globalf)
        loss_class = self.cls_criterion(logits, label_src)
        return loss_class
