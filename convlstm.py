import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x_seq):
        b, t, c, h, w = x_seq.shape

        h_state = torch.zeros(b, 64, h, w)
        c_state = torch.zeros(b, 64, h, w)

        for i in range(t):
            h_state, c_state = self.cell(x_seq[:, i], h_state, c_state)

        return h_state