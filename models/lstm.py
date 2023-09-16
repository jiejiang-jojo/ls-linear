import torch
from torch import nn

class Lstm(nn.Module):

    def __init__(self, c_in:int, seq_len:int):

        super().__init__()
        f_size =32
        self.lstm = nn.LSTM(input_size=c_in, hidden_size=f_size, num_layers=1, batch_first=True, dropout=0.5)
        self.seq_len = seq_len
        self.linear_rnn = nn.Linear(f_size*self.seq_len, 1)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x, t=None):
        """x: tensor [bs x seq_len x n_vars]"""
        x, _ = self.lstm(x)
        x = self.linear_rnn(self.flatten(x))
        return x.unsqueeze(1)