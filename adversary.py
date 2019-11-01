import torch.nn as nn
import torch.nn.functional as F


class Adv(nn.Module):
    def __init__(self, d, h, output_dim, dropout):
        super(Adv, self).__init__()
        self.h = h
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def backward(self, grad_output):
        return -grad_output
