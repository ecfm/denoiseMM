import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

from models.my_model_av2l_loss_simplified.attention_net import AttentionNet
from models.my_model_av2l_loss_simplified.decision_net import DecisionNet
from models.my_model_av2l_loss_simplified.modules.transformer import TransformerEncoder


class Model(nn.Module):
    def __init__(self, device, dataset_class,
                 d_a, d_v, n_head_av, n_layers_av,
                 dropout):
        """
        Construct a Net model.
        """
        if (d_a + d_v) % n_head_av != 0:
            raise ValueError("SKIP")
        super(Model, self).__init__()
        self.device = device
        ds = dataset_class
        self.ds = ds
        self.best_epoch = -1
        self.proj_a_comp = nn.Conv1d(ds.dim_a, d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v_comp = nn.Conv1d(ds.dim_v, d_v, kernel_size=1, padding=0, bias=False)
        self.enc_av_comp = TransformerEncoder(embed_dim=d_a + d_v,
                                              num_heads=n_head_av,
                                              layers=n_layers_av,
                                              attn_dropout=dropout)
        self.dec_av = DecisionNet(input_dim=d_a + d_v, output_dim=ds.output_dim)
        self.criterion = self.ds.get_loss()
        self.dropout = dropout

    def forward(self, x_l=None, x_a=None, x_v=None, train_l=False, train_av=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        covarep = x_a.transpose(1, 2)
        facet = x_v.transpose(1, 2)

        av_latent_comp = self.enc_av_comp(torch.cat([self.proj_a_comp(covarep).permute(2, 0, 1),
                                                     self.proj_v_comp(facet).permute(2, 0, 1)],
                                                    dim=2))[-1]

        _, outputs = self.dec_av(av_latent_comp)
        return outputs

    def train_eval(self, instance_dir, train_loader, valid_loader, test_loader,
                   num_epochs, patience_epochs, lr, beta):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.best_epoch = -1
        log_path = os.path.join(instance_dir, 'train_eval.log')
        model_path = os.path.join(instance_dir, 'checkpoint.pytorch')
        self.best_metrics = None
        all_train_metrics = []
        all_valid_metrics = []
        all_test_metrics = []
        device = self.device
        logs = pd.DataFrame()
        for epoch in range(num_epochs):
            label_all = []
            output_all = []
            for data in train_loader:
                optimizer.zero_grad()
                words, covarep, facet, inputLen, labels = data
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = self(x_l=words, x_a=covarep, x_v=facet)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                output_all.extend(outputs.tolist())
                label_all.extend(labels.tolist())
            train_metrics = self.ds.eval(output_all, label_all)
            valid_metrics, valid_output_all = self.get_results_no_grad(valid_loader)
            test_metrics, test_output_all = self.get_results_no_grad(test_loader)
            all_train_metrics.append(train_metrics)
            all_valid_metrics.append(valid_metrics)
            logs = logs.append({'epoch': epoch,
                                **{"train." + k: v for k, v in train_metrics.items()},
                                **{"valid." + k: v for k, v in valid_metrics.items()},
                                **{"test." + k: v for k, v in test_metrics.items()}}, ignore_index=True)
            logs.to_csv(log_path, index=False)
            if self.ds.is_better_metric(valid_metrics, self.best_metrics):
                self.best_metrics = valid_metrics
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state': self.state_dict(),
                    'valid_metrics': valid_metrics,
                    'valid_outputs': valid_output_all,
                    'test_metrics': test_metrics,
                    'test_outputs': test_output_all
                }, model_path)
                # Only record the test metrics when valid is the best
                all_test_metrics.append(test_metrics)
            else:
                if epoch - self.best_epoch > patience_epochs:
                    break

        best_train_metrics = self.ds.get_best_metrics(all_train_metrics)
        best_valid_metrics = self.ds.get_best_metrics(all_valid_metrics)
        best_test_metrics = self.ds.get_best_metrics(all_test_metrics)
        best_result = {'best_epoch': self.best_epoch,
                       **{"train." + k: v for k, v in best_train_metrics.items()},
                       **{"valid." + k: v for k, v in best_valid_metrics.items()},
                       **{"test." + k: v for k, v in best_test_metrics.items()}}
        return logs, best_result

    def get_results_no_grad(self, data_loader):
        device = self.device
        with torch.no_grad():
            output_all = []
            label_all = []
            for data in data_loader:
                words, covarep, facet, inputLen, labels = data
                words, covarep, facet, inputLen, labels = words.to(device), covarep.to(device), facet.to(
                    device), inputLen.to(device), labels.to(device)
                outputs = self(x_l=words, x_a=covarep, x_v=facet)
                output_all.extend(outputs.tolist())
                label_all.extend(labels.tolist())
            metrics = self.ds.eval(output_all, label_all)
            return metrics, output_all


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
