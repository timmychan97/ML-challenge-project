import torch
from torch import nn


class FFNetwork(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(157, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.act2 = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
