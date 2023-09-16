"""A meta model for learning to scale the output of embedded models."""

import torch

from utils.flags import with_flags

@with_flags
class LearnToScale(torch.nn.Module):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--learn_to_scale_model', default='linear',
                            help='The model to use as the embedded model for learning to scale.')

    def __init__(self, input_shape, output_shape):
        super(LearnToScale, self).__init__()
        from .builder import ModelBuilder
        self.model = ModelBuilder[self.args.learn_to_scale_model](input_shape, output_shape)
        self.mstd_linear = torch.nn.Linear(input_shape[-2] * input_shape[-1], 2)

    def forward(self, x):
        y = self.model(x)
        x = x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        t = self.mstd_linear(x).unsqueeze(1)
        y = torch.mul(y, t[:,:,0:1]) +  t[:,:,1:2]
        return y
