import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from my_network import MyModel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef
from thop import profile  # For FLOPs
import time  # For inference time




class Solver:
    def __init__(self, device, batch_size, input_size, dropout_rate, learning_rate, num_class, D, out_channels,
                 eeg_groups, initial_weight_decay=2e-4):
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_class = num_class
        self.D = D
        self.out_channels = out_channels
        self.eeg_groups = eeg_groups
        # new metrics
        self.flops = None
        self.params = None
        self.inference_time = None
        print("=" * 30, "> Build network")
        self.myModel = MyModel(
            device=self.device,
            input_size=self.input_size,
            num_class=self.num_class,
            dropout_rate=self.dropout_rate,
            out_channels=self.out_channels,
            D=self.D,
            eeg_groups=self.eeg_groups
        ).to(self.device)
        self.parameters = list(self.myModel.parameters())
        self.initial_weight_decay = initial_weight_decay
        self.opt = optim.Adam(self.parameters, lr=self.learning_rate, weight_decay=self.initial_weight_decay)
        self.lr_scheduler = lr_scheduler.ExponentialLR(self.opt, gamma=0.99)
        self.cls_criterion = nn.CrossEntropyLoss().cuda(self.device)
        self.xent_loss = nn.CrossEntropyLoss().cuda(self.device)
        self.adv_loss = nn.BCEWithLogitsLoss().cuda(self.device)

    def adjust_weight_decay(self, new_weight_decay):
        for param_group in self.opt.param_groups:
            param_group['weight_decay'] = new_weight_decay

    def set_data_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def estimate_JSD_MI(self, joint, marginal, mean=False):
        joint = (torch.log(torch.tensor(2.0)) - nn.functional.softplus(-joint))
        marginal = (nn.functional.softplus(-marginal) + marginal - torch.log(torch.tensor(2.0)))
        out = joint - marginal
        if mean:
            out = out.mean()
        return out

    def _ring(self, feat, type='geman'):
        x = feat.pow(2).sum(dim=1).pow(0.5)
        radius = x.mean()
        radius = radius.expand_as(x)
        if type == 'geman':
            l2_loss = (x - radius).pow(2).sum(dim=0) / (x.shape[0] * 0.5)
            return l2_loss
        else:
            raise NotImplementedError("Only 'geman' is implemented")



    def compute_model_costs(self, input_size, batch_size=8):  # 增大batch_size
        # 固定随机种子
        torch.manual_seed(42)
        # 固定cuDNN配置
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.myModel.eval()

        # 准备输入数据
        input_tensor = torch.ones(batch_size, *input_size).to(self.device)
        dummy_label = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        # 增加warm-up轮数
        with torch.no_grad():
            for _ in range(20):  # 增加到20次
                _ = self.myModel(input_tensor,dummy_label)

        # 计算 FLOPs 和参数量
        self.flops, self.params = profile(self.myModel, inputs=(input_tensor, dummy_label), verbose=True)
        self.flops = self.flops / batch_size / 1e9  # GFLOPs
        self.params = self.params / 1e6  # M

        # 测量推理时间
        num_trials = 200
        times = []
        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = self.myModel(input_tensor,dummy_label)
            end_event.record()
            torch.cuda.synchronize()  # 确保同步
            times.append(start_event.elapsed_time(end_event))
        times = times[1:]  # 去掉第一次
        self.inference_time = sum(times) / len(times) / batch_size    # ms



        return self.flops, self.params, self.inference_time


    def train_epoch(self):
        self.myModel.train()
        if self.flops is None:  # Compute only once
            flops, params, inference_time = self.compute_model_costs(self.input_size)
            print(f"FLOPs: {flops:.4f}, Params: {params:.4f}, Inference Time: {inference_time:.6f}ms")
        for batch_idx, (data_src, label_src) in enumerate(self.train_loader):
            data_src = data_src.to(self.device)
            label_src = label_src.long().to(self.device)
            self.opt.zero_grad()
            loss_class = self.myModel(data_src, label_src)
            loss_all = loss_class
            loss_all.backward()
            self.opt.step()
            self.opt.zero_grad()
        return
    def validate(self):
        self.myModel.eval()
        loss_all = 0
        pred_all = np.empty(shape=(0), dtype=np.float32)
        real_all = np.empty(shape=(0), dtype=np.float32)
        features_all = [] 
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.val_loader):
                data, label = data.to(self.device), label.to(self.device)
                rele = self.myModel.feature_extractor(data)
                global_a = self.myModel.global_enc(rele)
                logit = self.myModel.classifier(global_a)
                features_all.append(logit.cpu().numpy())
                _, pred = torch.max(nn.functional.softmax(logit, -1).data, -1)
                loss = self.cls_criterion(logit, label)
                loss_all += loss.cpu()
                pred_all = np.concatenate((pred_all, pred.cpu()), axis=-1)
                real_all = np.concatenate((real_all, label.cpu()), axis=-1)
        acc = accuracy_score(real_all, pred_all)
        f1 = f1_score(real_all, pred_all, average='macro')
        precision = precision_score(real_all, pred_all, average='macro', zero_division=0)
        recall = recall_score(real_all, pred_all, average='macro')
        cm = confusion_matrix(real_all, pred_all)
        precision_weighted = precision_score(real_all, pred_all, average='weighted', zero_division=0)
        recall_weighted = recall_score(real_all, pred_all, average='weighted')
        f1_weighted = f1_score(real_all, pred_all, average='weighted')
        kappa = cohen_kappa_score(real_all, pred_all)
        mcc = matthews_corrcoef(real_all, pred_all)
        return acc, f1, precision, recall, cm, precision_weighted, recall_weighted, f1_weighted, kappa, mcc

    def test(self):
        self.myModel.eval()
        loss_all = 0
        pred_all = np.empty(shape=(0), dtype=np.float32)
        real_all = np.empty(shape=(0), dtype=np.float32)
        features_all = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.test_loader):
                data, label = data.to(self.device), label.to(self.device)
                rele = self.myModel.feature_extractor(data)
                global_a = self.myModel.global_enc(rele)
                logit = self.myModel.classifier(global_a)
                features_all.append(logit.cpu().numpy())
                _, pred = torch.max(nn.functional.softmax(logit, -1).data, -1)
                loss = self.cls_criterion(logit, label)
                loss_all += loss.cpu()
                pred_all = np.concatenate((pred_all, pred.cpu()), axis=-1)
                real_all = np.concatenate((real_all, label.cpu()), axis=-1)
        acc = accuracy_score(real_all, pred_all)
        f1 = f1_score(real_all, pred_all, average='macro')
        precision = precision_score(real_all, pred_all, average='macro', zero_division=0)
        recall = recall_score(real_all, pred_all, average='macro')
        cm = confusion_matrix(real_all, pred_all)
        precision_weighted = precision_score(real_all, pred_all, average='weighted', zero_division=0)
        recall_weighted = recall_score(real_all, pred_all, average='weighted')
        f1_weighted = f1_score(real_all, pred_all, average='weighted')
        kappa = cohen_kappa_score(real_all, pred_all)
        mcc = matthews_corrcoef(real_all, pred_all)
        return acc, f1, precision, recall, cm, precision_weighted, recall_weighted, f1_weighted, kappa, mcc#

    def save_model(self, path):
        torch.save(self.myModel.state_dict(), path)

    def load_model(self, path):
        self.myModel.load_state_dict(torch.load(path))
