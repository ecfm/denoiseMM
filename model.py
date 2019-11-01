import torch.nn as nn

from consts import global_consts as gc
from transformer import t_model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conf = gc.config
        if gc.config['mod'] == 'all':
            dim = gc.dim_l + gc.dim_a + gc.dim_v
        else:
            dim = getattr(gc, "dim_%s" % gc.config['mod'])
        self.l_transformer_encoder = t_model.TransformerEncoder(gc.dim_l)
        self.a_transformer_encoder = t_model.TransformerEncoder(gc.dim_a)
        self.v_transformer_encoder = t_model.TransformerEncoder(gc.dim_v)
        dim_total_proj = conf['dim_total_proj']
        self.gru = nn.GRU(input_size=dim, hidden_size=dim_total_proj)
        out_dim = 1
        if gc.dataset == 'mosei_emo':
            out_dim = 6
        self.finalW = nn.Linear(dim_total_proj, out_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, words, covarep, facet, inputLens):
        state_w = self.l_transformer_encoder(words)
        state_c = self.a_transformer_encoder(covarep)
        state_f = self.v_transformer_encoder(facet)
        state = torch.concat((state_w,state_c,state_f), -1)
        # convert input to GRU from shape [batch_size, seq_len, input_size] to [seq_len, batch_size, input_size]
        _, gru_last_h = self.gru(state.transpose(0, 1))
        return self.finalW(gru_last_h.squeeze()).squeeze()
