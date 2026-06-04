import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluation(nn.Module):
    def __init__(self):
        super().__init__()

        #Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        #Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        #Fully Connected Layers
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = F.softsign(self.conv1(x))
        x = F.softsign(self.conv2(x))
        x = F.softsign(self.conv3(x))
        x = F.softsign(self.conv4(x))
        x = self.gap(F.softsign(self.conv5(x)))

        x = x.view(x.size(0), -1)

        x = F.softsign(self.linear1(x))
        x = self.dropout(x)
        x = F.softsign(self.linear2(x))
        x = self.dropout(x)
        x = F.softsign(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x