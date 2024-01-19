import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelWeightsInit import *

class EpilepsyGRU(nn.Module):

    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)

        ### NETWORK PARAMETERS
        n_nodes = inputmodule_params['n_nodes']
        Gstacks = net_params['Gstacks']  # GRU stacks
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']
        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']

        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params

        ### NETWORK ARCHITECTURE
        self.gru = nn.GRU(input_size=n_nodes,
                          hidden_size=hidden_size,
                          num_layers=Gstacks,
                          batch_first=True,
                          bidirectional=False,
                          dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes))

    def init_weights(self):
        init_weights_xavier_normal(self)

    def forward(self, x):
        ## Reshape input
        x = x.permute(0, 2, 1)  # GRU expects [batch, sequence_length, features]

        ## GRU Processing
        out, hn = self.gru(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn is [num_layers, batch, hidden_size]
        out = out[:, -1, :]  # hT state of length hidden_size

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

        return x
