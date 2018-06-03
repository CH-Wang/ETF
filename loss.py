import torch
import torch.nn as nn

class ETFPriceLoss(nn.Module):

    def __init__(self):
        super(ETFPriceLoss, self).__init__()

    def forward(self, label, output):
        loss = torch.abs(torch.add(label, -1, output))
        loss = torch.add(label, -1, loss)
        loss = torch.div(loss, label)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        loss = torch.mul(loss, weights)
        loss = torch.sum(loss, dim=1)
        loss = torch.mul(loss, -0.5)
        loss = torch.mean(loss)
        return loss

class ETFLoss(nn.Module):

    def __init__(self):
        super(ETFLoss, self).__init__()

    def forward(self, label, output):
        loss = torch.abs(torch.add(label, -1, output))
        loss = torch.add(label, -1, loss)
        loss = torch.div(loss, label)
        weights = torch.tensor([0.1,0.15,0.2,0.25,0.3])
        loss = torch.mul(loss, weights)
        loss = torch.sum(loss, dim=1)
        loss = torch.mul(loss, -0.5)
        loss = torch.mean(loss)
        return loss