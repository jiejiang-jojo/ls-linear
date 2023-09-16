"""Common metrics."""

import numpy as np
import torch


class MSE(object):
    def __init__(self):
        self.name = 'MSE'

    def __call__(self, x, y):
        return torch.mean(torch.square(x - y))


class MAE(object):
    def __init__(self):
        self.name = 'MAE'

    def __call__(self, x, y):
        return torch.mean(torch.abs(x - y))
