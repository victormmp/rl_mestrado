import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):

    def __init__(
        self,
        features: int = 64,
        outputs: int = 3 
    ):

        super().__init__()

        self.linear1    = nn.Linear(in_features=features, out_features=64)
        self.batch_n1   = nn.BatchNorm1d(num_features=self.linear1.out_features)
        self.relu1      = nn.LeakyReLU()
        self.linear2    = nn.Linear(in_features=64, out_features=32)
        self.batch_n2   = nn.BatchNorm1d(num_features=self.linear2.out_features)
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