import torch
import torch.nn as nn

# Упрощённый U-Net
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.up = nn.Upsample(scale_factor=2)

        self.out = nn.Conv2d(32, 1, 1)  # маска облаков

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.up(x3)
        return torch.sigmoid(self.out(x4))