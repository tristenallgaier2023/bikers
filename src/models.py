import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Bike_Classifier(nn.Module):
    """
    This class creates a neural network for classifying bikers as casual(0) or
    member(1).

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 2 nodes
    - Second hidden layer: fully connected layer of size 3 nodes
    - Output layer: a linear layer with one node per class (so 2 nodes)

    ReLU activation function for both hidden layers
    """
    def __init__(self):
        super(Bike_Classifier, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, 2)

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
