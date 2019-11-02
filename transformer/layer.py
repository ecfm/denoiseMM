import torch
import torch.nn as nn

from transformer.sub_layer import FeedForward, MultiHeadAttention, Norm
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attns = self.attn(x2, x2, x2, mask)
        attns_mean = torch.stack(attns, dim=0).mean(dim=0)
        x = x + self.dropout_1(attns_mean)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
