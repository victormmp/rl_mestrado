import torch
import torch.nn as nn


class LSTMActor(nn.Module):

    def __init__(
        self,
        features: int = 64,
        seq_length: int = 60, 
        num_layers: int = 20,
        hidden_size: int = 64,
        outputs: int = 3 
    ):

        super().__init__()

        self._features = features
        self._hidden_size = hidden_size
        self._seq_len = seq_length
        self._num_layers = num_layers
        self._outputs = outputs

        self.lstm       = nn.LSTM(input_size=features, hidden_size=hidden_size, num_layers=num_layers, batch_first = False)
        self.relu1      = nn.LeakyReLU()
        self.linear2    = nn.Linear(in_features=self.lstm.hidden_size, out_features=32)
        self.relu2      = nn.LeakyReLU()
        self.output1    = nn.Linear(in_features=32, out_features=outputs)
        self.softmax1   = nn.Softmax(dim=1)


    def forward(self, x, hidden = None):
        # Add batch dimention
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        # Add sequence dimention
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)

        x, (h, c) = self.lstm(x, hidden)
        x = x.view(-1, self._hidden_size)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output1(x)
        x = self.softmax1(x)
        x = torch.squeeze(x)

        return x, (h, c)
    
    def init_hidden(self, batch_size: int = 1):
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, requires_grad=True)
        c0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, requires_grad=True)

        return h0, c0


class SimpleNetworkCritic(nn.Module):

    def __init__(
        self,
        features: int = 64,
        outputs: int = 3 
    ):

        super().__init__()

        self.linear1    = nn.Linear(in_features=features, out_features=64)
        self.relu1      = nn.ReLU()
        self.linear2    = nn.Linear(in_features=64, out_features=32)
        self.relu2      = nn.ReLU()
        self.output1    = nn.Linear(in_features=32, out_features=outputs)


    def forward(self, state, action):
        dim = len(state.shape) - 1
        x = torch.cat([state, action], dim=dim)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output1(x)

        return x