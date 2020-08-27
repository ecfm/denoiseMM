import os

import torch
import torch.nn.functional as F
from torch import nn, optim

from .modules.mfn import MFN

class Model(nn.Module):
    def __init__(self, device, dataset_class,
                 d_lav, tau, h_l, h_a, h_v, 
                 memsize, windowsize, h_att1, 
                 h_att2, h_gamma1, h_gamma2, h_out,
                 d_fusion, beta):
        """
        Construct a Net model.
        """
        super(Model, self).__init__()
        self.device = device
        ds = dataset_class
        self.ds = ds
        self.best_epoch = -1
        self.proj_l = nn.Linear(ds.dim_l, d_lav)
        self.proj_a = nn.Linear(ds.dim_a, d_lav)
        self.proj_v = nn.Linear(ds.dim_v, d_lav)
        self.tau = tau
        self.proj_l_proxy = nn.Linear(ds.dim_l, ds.dim_l)
        self.proj_a_proxy = nn.Linear(ds.dim_a, ds.dim_a)
        self.proj_v_proxy = nn.Linear(ds.dim_v, ds.dim_v)
        self.mfn = MFN(ds.dim_l, ds.dim_a, ds.dim_v,
                       h_l, h_a, h_v,
                       memsize, windowsize, h_att1,
                       h_att2, h_gamma1, h_gamma2, h_out,
                       d_fusion, beta) 
        self.criterion = self.ds.get_loss()

    def correlation_score(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        corr = torch.sum(vx * vy, (1,2)) / (torch.sqrt(torch.sum(vx ** 2, (1,2))) * torch.sqrt(torch.sum(vy ** 2, (1,2))))
        return corr

    def forward(self, x_l, x_a, x_v, mode="train"):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        words = F.dropout(x_l, p=0.25, training=self.training)
        covarep = x_a
        facet = x_v
        vec_l = self.proj_l(words)
        vec_a = self.proj_a(covarep)
        vec_v = self.proj_v(facet)
        
        corr_la = self.correlation_score(vec_l, vec_a)
        corr_av = self.correlation_score(vec_a, vec_v)
        corr_lv = self.correlation_score(vec_l, vec_v)
        I_l = ((corr_la < self.tau) * (corr_lv < self.tau) == 0).float().reshape((-1, 1, 1)) * torch.ones_like(words)
        I_a = ((corr_la < self.tau) * (corr_av < self.tau) == 0).float().reshape((-1, 1, 1)) * torch.ones_like(covarep)
        I_v = ((corr_av < self.tau) * (corr_lv < self.tau) == 0).float().reshape((-1, 1, 1)) * torch.ones_like(facet)
        proxy_l = torch.zeros_like(words)
        proxy_a = torch.zeros_like(covarep)
        proxy_v = torch.zeros_like(facet)
        if mode == "test":
            proxy_l = self.proj_l_proxy(words) * (I_l == 0).float()
            proxy_a = self.proj_a_proxy(covarep) * (I_a == 0).float()
            proxy_v = self.proj_v_proxy(facet) * (I_v == 0).float()
        f_l = I_l * words + proxy_l
        f_a = I_a * covarep + proxy_a
        f_v = I_v * facet + proxy_v
        x = torch.cat([f_l, f_a, f_v], dim=2)
        output = self.mfn(x)
        return output

    def train_eval(self, instance_dir, train_loader, test_loader,
                   num_epochs, patience_epochs, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.best_epoch = -1
        model_path = os.path.join(instance_dir, 'checkpoint.pytorch')
        best_metrics = None
        logs = []
        all_train_metrics = []
        all_test_metrics = []
        device = self.device
        for epoch in range(num_epochs):
            label_all = []
            output_all = []
            for data in train_loader:
                optimizer.zero_grad()
                words, covarep, facet, inputLen, labels = data
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = self(x_l=words, x_a=covarep, x_v=facet)
                print(outputs)
                loss = self.criterion(outputs, labels)
                # TODO: why retain_graph?
                loss.backward(retain_graph=True)
                optimizer.step()
                output_all.extend(outputs.tolist())
                label_all.extend(labels.tolist())
            train_metrics = self.ds.eval(output_all, label_all)
            test_metrics, test_output_all = self.get_test_results(test_loader)
            all_train_metrics.append(train_metrics)
            all_test_metrics.append(test_metrics)
            logs.append({'epoch': epoch,
                         **{"train." + k: v for k, v in train_metrics.items()},
                         **{"test." + k: v for k, v in test_metrics.items()}})
            print('epoch',  epoch, train_metrics.items(), "test." ,test_metrics.items())
            if self.ds.is_better_metric(test_metrics, best_metrics):
                best_metrics = test_metrics
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state': self.state_dict(),
                    'test_metrics': test_metrics,
                    'test_outputs': test_output_all
                }, model_path)

        best_train_metrics = self.ds.get_best_metrics(all_train_metrics)
        best_test_metrics = self.ds.get_best_metrics(all_test_metrics)
        best_result = {'best_epoch': self.best_epoch,
                       **{"train." + k: v for k, v in best_train_metrics.items()},
                       **{"test." + k: v for k, v in best_test_metrics.items()}}
        return logs, best_result

    def get_test_results(self, test_loader):
        device = self.device
        with torch.no_grad():
            test_output_all = []
            test_label_all = []
            for data in test_loader:
                words, covarep, facet, inputLen, labels = data
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = self(x_l=words, x_a=covarep, x_v=facet, mode="test")
                test_output_all.extend(labels.tolist())
                test_label_all.extend(outputs.tolist())

            test_metrics = self.ds.eval(test_output_all, test_label_all)
            return test_metrics, test_output_all


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
