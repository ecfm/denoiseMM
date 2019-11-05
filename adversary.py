import torch.nn as nn
import torch.nn.functional as F
from consts import global_consts as gc


class Adv(nn.Module):
    def __init__(self):
        super(Adv, self).__init__()
        h = gc.config['adv_h_dim']
        self.fc1 = nn.Linear(gc.dim_l, h)
        out_dim = 1
        if gc.dataset == 'mosei_emo':
            out_dim = 6
        self.fc2 = nn.Linear(h, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def backward(self, grad_output):
        return -grad_output
