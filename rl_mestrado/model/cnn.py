import torch
import torch.nn as nn


class ConvolutionalActor(nn.Module):

    def __init__(
        self,
        in_shape: list = (17, 60),
        outputs: int = 3 
    ):

        super().__init__()

        features, days = in_shape
        in_channels = 1

        kernel_1 = (5, 5)
        kernel_2 = (5, 5)

        out_c_1 = 8
        out_c_2 = 4

        in_linear_1 = (
            (
                features - kernel_1[0] - kernel_2[0] + 2) 
                * (days - kernel_1[1] - kernel_2[1] + 2)
            ) * out_c_2

        self.conv1      = nn.Conv2d(in_channels=in_channels, out_channels=out_c_1, kernel_size=kernel_1)
        self.batch1     = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.relu1      = nn.LeakyReLU()
        self.conv2      = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=out_c_2, kernel_size=kernel_2)
        self.batch2     = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.relu2      = nn.LeakyReLU()
        self.flat1      = nn.Flatten()
        self.output1    = nn.Linear(in_features=in_linear_1, out_features=int(in_linear_1/2))
        self.relu3      = nn.LeakyReLU()
        self.output2    = nn.Linear(in_features=self.output1.out_features, out_features=outputs)
        self.softmax1   = nn.Softmax(dim=1)

    def forward(self, x):
        # Add batch and channel dimentions
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=0)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.flat1(x)
        x = self.output1(x)
        x = self.relu3(x)
        x = self.output2(x)
        x = self.softmax1(x)

        return torch.squeeze(x)


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