from . import simple_linear, tsmixer, lts
from . import lstm
from . import patchtst
from . import dlinear

ModelBuilder = dict([
    ('linear', lambda ishape, oshape: simple_linear.SimpleWindowLinear(ishape, oshape)),
    ('dlinear', lambda ishape, oshape: dlinear.Dlinear(ishape, oshape)),
    ('tsmixer', lambda ishape, oshape: tsmixer.Tsmixer(c_in=ishape[1], seq_len=ishape[0])),
    ('lstm', lambda ishape, oshape: lstm.Lstm(c_in=ishape[1], seq_len=ishape[0])),
    ('patchtst', lambda ishape, oshape: patchtst.PatchTST(c_in=ishape[1], seq_len=ishape[0])),
    ('lts', lambda ishape, oshape: lts.LearnToScale(ishape, oshape)),
    ])
