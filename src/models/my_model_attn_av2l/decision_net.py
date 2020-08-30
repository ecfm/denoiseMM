import torch.nn.functional as F
from torch import nn


class DecisionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Construct a MulT model.
        """
        super(DecisionNet, self).__init__()

        # Projection layers
        self.proj1 = nn.Linear(input_dim, input_dim)
        self.proj2 = nn.Linear(input_dim, input_dim)
        self.out_layer = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        last_hs_proj = self.proj2(F.relu(self.proj1(input)))

        output = self.out_layer(last_hs_proj)
        return output.squeeze()