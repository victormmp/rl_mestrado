import torch
import torch.nn as nn


class SimpleNetworkActor(nn.Module):

    def __init__(
        self,
        features: int = 64,
        outputs: int = 3 
    ):

        super().__init__()

        self.linear1    = nn.Linear(in_features=features, out_features=64)
        self.relu1      = nn.LeakyReLU()
        self.linear2    = nn.Linear(in_features=64, out_features=32)
        self.relu2      = nn.LeakyReLU()
        self.output1    = nn.Linear(in_features=32, out_features=outputs)
        self.softmax1   = nn.Softmax(dim=0)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output1(x)
        x = self.softmax1(x)

        return x


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