import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelWeightsInit import *
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term2 = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term1)
        pe[:, 1::2] = torch.cos(position * div_term2)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EpilepsyTransformer(nn.Module):

    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)

        ### NETWORK PARAMETERS
        n_nodes = inputmodule_params['n_nodes']
        Lstacks = net_params['Lstacks']
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']
        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']

        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params

        ### NETWORK ARCHITECTURE
        self.pos_enc = PositionalEncoding(21, 0.2, 256)

        enc_layer = nn.TransformerEncoderLayer(d_model = 21, nhead = 7, dim_feedforward = 2048, dropout = 0.2)

        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=4,
        )

        self.fc = nn.Sequential(nn.Linear(n_nodes, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes)
                                )

    def init_weights(self):
        init_weights_xavier_normal(self)

    def forward(self, x):

        ## Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(2, 0, 1)  # transformer [sequence_length, batch, features]

        ## Transformer Processing
        out = self.pos_enc(x)
        out = self.transformer_encoder(out)
        out = out.mean(dim=0)  # Taking the mean over sequence dimension

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x
