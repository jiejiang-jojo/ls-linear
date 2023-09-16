import torch
from torch import nn


class ResBlock(nn.Module):
  """Residual block of TSMixer."""

  def __init__(self, c_in, seq_len, norm_type='L', dropout=0.2, ff_dim=32):
    super().__init__()
    self.norm = (
        nn.LayerNorm([seq_len, c_in])
        if norm_type == 'L'
        else nn.BatchNorm1d([seq_len, c_in])
    )
    self.temporal_linear = nn.Linear(seq_len, seq_len)
    self.activation = nn.ReLU()
    self.feature_linear = nn.Linear(c_in, ff_dim)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(ff_dim, c_in)

  def forward(self, x):
    """
    x: tensor [bs x seq_len x n_vars]
    """
    z1 = self.norm(x)
    z1 = z1.transpose(-1, -2) #[bs x n_vars x seq_len]
    z1 = self.temporal_linear(z1)
    z1 = self.activation(z1)
    z1 = z1.transpose(-1, -2) #[bs x seq_len x n_vars]
    z1 = self.dropout(z1)
    res = z1 + x

    z2 = self.norm(res)
    z2 = self.feature_linear(z2)
    z2 = self.activation(z2)
    z2 = self.dropout(z2)
    z2 = self.fc(z2)
    z2 = self.dropout(z2)
    return z2 + res


class Tsmixer(nn.Module):
  """Build TSMixer model."""

  def __init__(
      self,
      c_in,
      seq_len,
      norm_type='L',
      n_block=2,
      dropout=0.2,
      ff_dim=32,
  ):
    super().__init__()
    self.res_blocks = nn.Sequential(*[
        ResBlock(c_in, seq_len, norm_type, dropout, ff_dim) for _ in range(n_block)
    ])
    self.pred_linear = nn.Linear(seq_len*c_in, 1)

  def forward(self, x):
    """x: tensor [bs x seq_len x n_vars]"""
    x = self.res_blocks(x)
    x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
    x = self.pred_linear(x)
    return x.unsqueeze(1)

