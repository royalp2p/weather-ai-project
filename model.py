from convlstm import ConvLSTM
import torch.nn as nn


class WeatherConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvLSTM(input_dim=3, hidden_dim=64)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        # x: [B, 3, T, H, W] → твой dataset должен быть таким
        h = self.backbone(x)
        h = h.mean(dim=[2, 3])  # GAP
        return self.fc(h)