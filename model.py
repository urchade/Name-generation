import torch
from torch import nn


class NameLSTM(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(n_inputs, hidden_size, num_layers=2,
                           dropout=0.1, batch_first=True)

        self.linear = nn.Linear(hidden_size, n_inputs)

    def forward(self, x, hidden_states=None):
        batch_size, seq_len, _ = x.shape

        outs, hidden_states = self.rnn(x, hidden_states)
        outs = torch.relu(outs)
        outs = outs.contiguous().view(batch_size * seq_len, self.hidden_size)

        return self.linear(outs), hidden_states
