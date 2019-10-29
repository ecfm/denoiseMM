import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask, dropout):
    # [bs * N * sl * sl] = [bs * N * sl * d_model] * [bs * N * d_model * sl]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model
        self.h = heads

        for i in range(heads):
            setattr(self, 'k_linear%d' % i, nn.Linear(d_model, d_model))
            setattr(self, 'q_linear%d' % i, nn.Linear(d_model, d_model))
            setattr(self, 'v_linear%d' % i, nn.Linear(d_model, d_model))
            setattr(self, 'out%d' % i, nn.Linear(d_model, d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        multi_head_scores = []
        for i in range(self.h):
            # perform linear operation and split into N heads
            k = getattr(self, 'k_linear%d' % i)(k)
            q = getattr(self, 'q_linear%d' % i)(q)
            v = getattr(self, 'v_linear%d' % i)(v)
            # calculate attention using function we will define next
            attn_score = attention(q, k, v, self.d_k, mask, self.dropout)
            multi_head_scores.append(getattr(self, 'out%d' % i)(attn_score))
        # concatenate heads and put through final linear layer
        # concat = torch.sum(torch.stack(multi_head_scores, -1), -1)
        # model = self.out(concat)
        return multi_head_scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
