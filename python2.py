import torch.cuda
import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
## cuda -> gpu 사용
print("Using {} devise".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        # x1 = self.fc1(x)
        # x1 = nn.ReLU()(x1)
        # x2 = x1 + x
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)