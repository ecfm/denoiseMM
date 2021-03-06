import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

from models.my_model_av2l_loss_simplified.attention_net import AttentionNet
from models.my_model_av2l_loss_simplified.decision_net import DecisionNet
from models.my_model_av2l_loss_simplified.modules.transformer import TransformerEncoder

L_MODE = 'L'
AV_MODE = 'AV'
LAV_MODE = 'LAV'
FINAL_MODE = 'FINAL'


class Model(nn.Module):
    def __init__(self, device, dataset_class,
                 d_l, d_a, d_v, n_head_l, n_layers_l, n_head_av2l, n_layers_av2l, n_head_av, n_layers_av,
                 dropout):
        """
        Construct a Net model.
        """
        if d_l % n_head_l != 0 or (d_a + d_v) % n_head_av != 0:
            raise ValueError("SKIP")
        super(Model, self).__init__()
        self.device = device
        ds = dataset_class
        self.ds = ds
        self.best_epoch = -1
        self.proj_l = nn.Conv1d(ds.dim_l, d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(ds.dim_a, d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(ds.dim_v, d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a_comp = nn.Conv1d(ds.dim_a, d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v_comp = nn.Conv1d(ds.dim_v, d_v, kernel_size=1, padding=0, bias=False)

        self.enc_l = TransformerEncoder(embed_dim=d_l,
                                        num_heads=n_head_l,
                                        layers=n_layers_l,
                                        attn_dropout=dropout)
        self.enc_av2l = TransformerEncoder(embed_dim=d_a + d_v,
                                           num_heads=n_head_av2l,
                                           layers=n_layers_av2l,
                                           attn_dropout=dropout)
        self.av2l_l_attn = DecisionNet(d_l + d_l, 2)
        self.proj_avl2l = nn.Linear(d_a + d_v, d_l)
        self.enc_av_comp = TransformerEncoder(embed_dim=d_a + d_v,
                                              num_heads=n_head_av,
                                              layers=n_layers_av,
                                              attn_dropout=dropout)
        self.dec_l = DecisionNet(input_dim=d_l, output_dim=ds.output_dim)
        self.dec_av = DecisionNet(input_dim=d_l, output_dim=ds.output_dim)
        self.dec_lav = DecisionNet(input_dim=d_l + d_a + d_v, output_dim=ds.output_dim)
        self.mode = L_MODE
        self.criterion = self.ds.get_loss()
        self.dropout = dropout

    def forward(self, x_l=None, x_a=None, x_v=None, attn=None, train_l=False, train_av=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        words = F.dropout(x_l.transpose(1, 2), p=0.25, training=self.training)
        words = self.proj_l(words).permute(2, 0, 1)
        l_latent, outputs_l = self.dec_l(self.enc_l(words)[-1])
        if self.mode == L_MODE:
            return outputs_l
        covarep = x_a.transpose(1, 2)
        facet = x_v.transpose(1, 2)
        # [batch_size,  d_a+d_v]
        av2l_intermediate = self.enc_av2l(torch.cat([self.proj_a(covarep).permute(2, 0, 1),
                                                     self.proj_v(facet).permute(2, 0, 1)],
                                                    dim=2))[-1]
        av2l_intermediate = self.proj_avl2l(av2l_intermediate)
        av2l_l_cat = torch.cat([l_latent.detach(), av2l_intermediate.detach()], dim=1)
        av2l_l_stack = torch.stack([l_latent.detach(), av2l_intermediate.detach()], dim=1)

        # [batch_size,  1, 2] * [batch_size,  2, d_l] = [batch_size, d_l]
        av2l_l_weighted = torch.matmul(F.softmax(self.av2l_l_attn(av2l_l_cat)[1], dim=1).unsqueeze(1), av2l_l_stack).squeeze()
        combined_l_latent, outputs_av = self.dec_av(av2l_l_weighted)

        if self.mode == AV_MODE:
            return outputs_l, outputs_av

        av_latent_comp = self.enc_av_comp(torch.cat([self.proj_a_comp(covarep).permute(2, 0, 1),
                                                     self.proj_v_comp(facet).permute(2, 0, 1)],
                                                    dim=2))[-1]

        _, outputs = self.dec_lav(torch.cat([av2l_l_weighted, av_latent_comp], dim=1))
        return outputs

    def change_to_mode(self, to_mode, model_path, epoch):
        self.mode = to_mode
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['state'])
        self.best_epoch = epoch
        self.best_metrics = None

    def train_eval(self, instance_dir, train_loader, valid_loader, test_loader,
                   num_epochs, patience_epochs, lr, beta):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.best_epoch = -1
        self.mode = L_MODE
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
                if self.mode == AV_MODE:
                    l_outputs, outputs = self(x_l=words, x_a=covarep, x_v=facet)
                    l_loss = self.criterion(l_outputs, labels).detach()
                    av_loss = self.criterion(outputs, labels)
                    loss = av_loss * torch.pow(l_loss, beta)

                else:
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
                                'mode': self.mode,
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
                    if self.mode == L_MODE:
                        self.change_to_mode(AV_MODE, model_path, epoch)
                        set_requires_grad(self.proj_l, False)
                        set_requires_grad(self.enc_l, False)
                        set_requires_grad(self.dec_l, False)
                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                               lr=lr)
                    elif self.mode == AV_MODE:
                        self.change_to_mode(LAV_MODE, model_path, epoch)
                        # TODO: verify if l should require grad
                        set_requires_grad(self.proj_l, False)
                        set_requires_grad(self.enc_l, False)
                        set_requires_grad(self.dec_l, False)
                        set_requires_grad(self.proj_a, False)
                        set_requires_grad(self.proj_v, False)
                        set_requires_grad(self.dec_av, False)
                        set_requires_grad(self.enc_av2l, False)
                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                               lr=lr)
                    elif self.mode == LAV_MODE:
                        self.change_to_mode(FINAL_MODE, model_path, epoch)
                        # TODO: verify if l should require grad
                        set_requires_grad(self.proj_l, True)
                        set_requires_grad(self.enc_l, True)
                        set_requires_grad(self.dec_l, False)
                        set_requires_grad(self.proj_a, True)
                        set_requires_grad(self.proj_v, True)
                        set_requires_grad(self.dec_av, False)
                        set_requires_grad(self.enc_av2l, True)
                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                               lr=lr)
                    else:
                        break

        best_train_metrics = self.ds.get_best_metrics(all_train_metrics)
        best_valid_metrics = self.ds.get_best_metrics(all_valid_metrics)
        best_test_metrics = self.ds.get_best_metrics(all_test_metrics)
        best_result = {'best_epoch': self.best_epoch,
                       'final_mode': self.mode,
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
                if self.mode == AV_MODE:
                    _, outputs = self(x_l=words, x_a=covarep, x_v=facet)
                else:
                    outputs = self(x_l=words, x_a=covarep, x_v=facet)
                output_all.extend(outputs.tolist())
                label_all.extend(labels.tolist())
            metrics = self.ds.eval(output_all, label_all)
            return metrics, output_all


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
