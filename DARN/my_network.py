import torch
import torch.nn as nn
import torch.optim as optim
from Module import *
from utils import *
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from thop import profile
import time
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from thop import profile
class MyModel(nn.Module):
    def __init__(self, device, input_size, num_class, dropout_rate, out_channels, D, eeg_groups):
        super(MyModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        self.out_channels = out_channels
        self.D = D
        self.eeg_groups = eeg_groups
        self.nfeatl = self.out_channels * self.D
        print("=" * 30, "> Build network")
        self.feature_extractor = DARN(in_channels=self.input_size[0], out_channels=self.out_channels,
                                               D=self.D, fs=self.input_size[-1] // 6,
                                               num_ch=self.input_size[1], eeg_groups=self.eeg_groups,
                                               input_size=self.input_size).to(self.device)
        self.global_enc = GlobalEncoder_EEGNet(nfeatl=self.nfeatl).to(self.device)
        size, self.nfeatg = self.calculate_feature_sizes(self.input_size)
        self.classifier = Classifier(self.nfeatg, self.num_class).to(self.device)
        self.cls_criterion = nn.CrossEntropyLoss()

    def calculate_feature_sizes(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2]))).to(self.device)
        rele = self.feature_extractor(data)
        size_temporal = rele.size()
        globalf = self.global_enc(rele)
        size_global = globalf.size()
        return size_temporal, size_global[-1]


    def forward(self, input, label):
        out = self.feature_extractor(input)
        global_a = self.global_enc(out)
        logits = self.classifier(global_a)
        loss_class = self.cls_criterion(logits, label)
        return loss_class




