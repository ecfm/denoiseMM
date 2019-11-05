import torch.nn as nn
import torch.nn.functional as F
from consts import global_consts as gc


class Adv(nn.Module):
    def __init__(self):
        super(Adv, self).__init__()
        h = gc.config['adv_h_dim']
        self.gru = nn.GRU(input_size=gc.dim_l, hidden_size=h)
        self.fc1 = nn.Linear(h, h)
        out_dim = 1
        if gc.dataset == 'mosei_emo':
            out_dim = 6
        self.fc2 = nn.Linear(h, out_dim)

    def forward(self, x):
        _, gru_last_h = self.gru(x.transpose(0, 1))
        x = F.relu(self.fc1(gru_last_h.squeeze())).squeeze()
        x = F.relu(self.fc2(x))
        return x

    def backward(self, grad_output):
        return -grad_output
