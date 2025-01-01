import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class LogicNet(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim):
        super(LogicNet, self).__init__()
        self.Linear1 = nn.Linear(inputdim, hiddendim)
        self.Linear2 = nn.Linear(hiddendim, outputdim)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.Linear1(x)
        x = torch.tanh(x)
        x = self.Linear2(x)
        return x
    def predict(self, x):
        pred = torch.softmax(self.forward(x), dim=1)
        return torch.argmax(pred, dim=1)
    def getloss(self, x, y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        return loss
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
def plot_losses(losses):
    avgloss = moving_average(losses)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(len(avgloss)), avgloss, 'b--')
    plt.xlabel('step number')
    plt.ylabel('Training loss')
    plt.show()