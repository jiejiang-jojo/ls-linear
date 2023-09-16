"""A simple linear model."""

import torch


class SimpleWindowLinear(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleWindowLinear, self).__init__()
        self.linear = torch.nn.Linear(input_shape[-2] * input_shape[-1], output_shape[-1])

    def forward(self, x):
        x = x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        return self.linear(x).unsqueeze(1)
