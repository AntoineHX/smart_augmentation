import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## Basic CNN ##
class LeNet(nn.Module):
    """Basic CNN.

    """
    def __init__(self, num_inp, num_out):
        """Init LeNet.

        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_inp, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, num_out)

    def forward(self, x):
        """Main method of LeNet

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __str__(self):
        """ Get name of model

        """
        return "LeNet"
