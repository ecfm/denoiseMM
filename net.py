import torch
from torch import nn
import torch.nn.functional as F
from consts import global_consts as gc
from decision_net import DecisionNet

from modules.transformer import TransformerEncoder


class Net(nn.Module):
    def __init__(self):
        """
        Construct a Net model.
        """
        super(Net, self).__init__()
        self.proj_l = nn.Conv1d(gc.dim_l, gc.config['d_l'], kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(gc.dim_a, gc.config['d_a'], kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(gc.dim_v, gc.config['d_v'], kernel_size=1, padding=0, bias=False)

        self.enc_l = TransformerEncoder(embed_dim=gc.config['d_l'],
                                   num_heads=gc.config['n_head_l'],
                                   layers=gc.config['n_layers_l'],
                                   attn_dropout=0.1)
        self.enc_av2l = TransformerEncoder(embed_dim=gc.config['d_a'] + gc.config['d_v'],
                                      num_heads=gc.config['n_head_av'],
                                      layers=gc.config['n_layers_av'],
                                      attn_dropout=0.0)
        self.proj_av2l = nn.Linear(gc.config['d_a'] + gc.config['d_v'], gc.config['d_l'])
        self.dec_l = DecisionNet(input_dim=gc.config['d_l'], output_dim=1)

    def forward(self, x_l=None, x_a=None, x_v=None, train_l=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        set_requires_grad(self.proj_l, train_l)
        set_requires_grad(self.enc_l, train_l)
        set_requires_grad(self.dec_l, train_l)
        words = F.dropout(x_l.transpose(1, 2), p=0.25, training=self.training)
        words = self.proj_l(words).permute(2, 0, 1)
        l_latent = self.enc_l(words)[-1]
        outputs_l = self.dec_l(l_latent)
        if train_l:
            return outputs_l
        covarep = x_a.transpose(1, 2)
        facet = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        covarep = self.proj_a(covarep).permute(2, 0, 1)
        facet = self.proj_v(facet).permute(2, 0, 1)

        av2l_intermediate = self.enc_av2l(torch.cat([covarep, facet], dim=2))[-1]
        av2l_latent = self.proj_av2l(av2l_intermediate)
        outputs_av = self.dec_l(av2l_latent)
        return outputs_av, outputs_l, l_latent, av2l_latent

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val