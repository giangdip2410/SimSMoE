import numpy as np

import torch
from torch import nn
import copy
from torch.autograd import Variable


class TiCo(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2206.10698v2.pdf
    Adapt from https://github.com/sayannag/TiCo-pytorch/blob/master/TiCo/main.py

    """
    def __init__(self, beta=0.9, rho=20.0, final_dim=256, device='cuda'):
        super().__init__()

        self.beta = beta
        self.rho = rho
        self.C = Variable(torch.zeros(final_dim, final_dim), requires_grad=True).to(device)

    def forward(self, x_1, x_2):
        z_1 = torch.nn.functional.normalize(x_1, dim = -1)
        z_2 = torch.nn.functional.normalize(x_2, dim = -1)
        B = torch.mm(z_1.T, z_1)/z_1.shape[0]
        self.C = self.beta * self.C + (1 - self.beta) * B
        loss = - (z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, self.C) * z_1).sum(dim=1).mean()
        return loss * (-1)
