import copy

import torch
import torch.nn as nn

from consts import global_consts as gc
from transformer.layer import EncoderLayer
from transformer.module import PositionalEncoder
from transformer.sub_layer import Norm


def get_clones(d_model, d_ff, heads, dropout, n_layers):
    list = [EncoderLayer(d_model, d_ff, heads, dropout) for _ in range(n_layers)]
    return nn.ModuleList(list)


def get_pad_mask(seq):
    # [batch, seq_len, dim]
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(-1).eq(0)
    return non_pad_mask.unsqueeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        dropout = gc.config["dropout"]
        self.n_layers = gc.config['n_layers']
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        heads = gc.config['n_head']
        self.layers = get_clones(d_model, gc.config['ff_dim_final'], heads, dropout, self.n_layers)
        self.norm = Norm(d_model)

    def forward(self, inputs):
        if gc.config['mod'] == 'all':
            x_l, x_a, x_v = inputs
            x = torch.cat((x_l, x_a, x_v), -1)
        else:
            x = inputs
        mask = get_pad_mask(x)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)
