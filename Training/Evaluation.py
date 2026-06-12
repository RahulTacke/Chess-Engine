"""CNN evaluation network: 8-channel board tensor → scalar tanh position score."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluation(nn.Module):
    """5-layer CNN with dual pooling and 4-layer MLP; output in (-1, 1) via tanh."""

    def __init__(self):
        super().__init__()

        #Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        #Batch Normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)

        #Dual Pooling — concatenate GAP and GMP for 256-dim features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        #Fully Connected Layers (linear1 takes 256 = 128 GAP + 128 GMP)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """Forward pass; x is (batch, 8, 8, 8). Returns (batch, 1) tanh score."""
        x = F.softsign(self.bn1(self.conv1(x)))
        x = F.softsign(self.bn2(self.conv2(x)))
        x = F.softsign(self.bn3(self.conv3(x)))
        x = F.softsign(self.bn4(self.conv4(x)))
        x = F.softsign(self.bn5(self.conv5(x)))

        avg = self.gap(x).view(x.size(0), -1)
        mx  = self.gmp(x).view(x.size(0), -1)
        x   = torch.cat([avg, mx], dim=1)

        x = F.softsign(self.linear1(x))
        x = self.dropout(x)
        x = F.softsign(self.linear2(x))
        x = self.dropout(x)
        x = F.softsign(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        return x
