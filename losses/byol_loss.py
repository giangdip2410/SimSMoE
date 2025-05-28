import numpy as np

import torch
from torch import nn
import copy
from torch.autograd import Variable


class Byol(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2006.07733v3.pdf
    Adapt from https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

    """
    def __init__(self,):
        super().__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return (x * y).sum(dim=-1)
