import torch
import torch.nn as nn
import torch.nn.functional as F

class EpilepsyRNN(nn.Module):
    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)

        ### NETWORK PARAMETERS
        n_nodes = inputmodule_params['n_nodes']
        Rstacks = net_params['Rstacks']  # RNN stacks
        dropout = net_params['dropout']
        hidden_size = net_params['hidden_size']
        n_classes = outmodule_params['n_classes']
        hd = outmodule_params['hd']

        ### NETWORK ARCHITECTURE
        self.rnn = nn.RNN(input_size=n_nodes,
                          hidden_size=hidden_size,
                          num_layers=Rstacks,
                          batch_first=True,
                          nonlinearity='tanh',
                          dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes))
        
    def init_weights(self):
        def init_rnn(m):
            if isinstance(m, nn.RNN):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(init_rnn)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # RNN expects [batch, sequence_length, features]
        out, hn = self.rnn(x)
        out = out[:, -1, :]  # hT state of length hidden_size
        x = self.fc(out)
        return x
