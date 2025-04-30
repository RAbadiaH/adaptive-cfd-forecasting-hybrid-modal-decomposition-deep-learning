import torch
import torch.nn as nn

class ModelLSTM(nn.Module):

    def __init__(self, num_modes, act_fn, hidden_size, p = 3):

        super().__init__()

        self.num_modes      = num_modes
        self.hidden_size    = hidden_size

        self.lstm = nn.LSTM(
            input_size    = self.num_modes, 
            hidden_size   = hidden_size, 
            num_layers    = 1,
            batch_first   = True)

        self.dense_block = nn.Sequential(
            
            nn.Linear(
                in_features   = 100, 
                out_features  = 200),

            act_fn,

            nn.Linear(
                in_features   = 200, 
                out_features  = 200),

            act_fn
        )

        self.last_dense = nn.Linear(
                in_features   = 200, 
                out_features  = p * self.num_modes)

        self.initialize_weights()

        return

    def initialize_weights(self):

        for name, param in self.named_parameters():

            if name.startswith("lstm"):

                if 'weight_ih' in name:
                    
                    nn.init.xavier_uniform_(param.data)

                elif 'weight_hh' in name:

                    nn.init.orthogonal_(param.data)

                elif 'bias_ih' in name:

                    torch.nn.init.zeros_(param.data)

                    # Set forget-gate bias to 1
                    n = param.size(0)
                    param.data[(n // 4):(n // 2)].fill_(1)

                elif 'bias_hh' in name:

                    torch.nn.init.zeros_(param.data)

            elif name.startswith("dense"):

                if name.endswith("weight"):

                    nn.init.kaiming_uniform_(param.data, nonlinearity = 'relu')

                else:

                    torch.nn.init.zeros_(param.data)

            else:

                if name.endswith("weight"):

                    nn.init.xavier_uniform_(param.data)

                else:

                    torch.nn.init.zeros_(param.data)

            # print(f"{name} -> {param}\n")

        return

    def forward(self, x_in, h = None, c = None):

        if h is None or c is None:
            h = torch.zeros(1, x_in.size(0), self.hidden_size)
            c = torch.zeros(1, x_in.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(x_in, (h, c))
        x = out[:, -1, :]

        x = self.dense_block(x)

        x_out = self.last_dense(x)[:, None, :]
        
        # x_out = torch.split(x, self.num_modes, dim = 1)
        # x_out = torch.stack(x_out, dim = 1)

        return x_out, (hn, cn)