import torch.nn.functional as F
from torch import nn


class AttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Construct a MulT model.
        """
        super(AttentionNet, self).__init__()

        # Projection layers
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, input):
        return F.softmax(self.proj2(self.dropout(F.relu(self.proj1(input)))), dim=1)