import torch
from torch import nn

channel=64

class CovW(nn.Module):
    def __init__(self):
        super(CovW, self).__init__()
        self.covW = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            )
    def forward(self, x):
        return self.covW(x)

class Fusioner_D(nn.Module):
    def __init__(self):
        super(Fusioner_D, self).__init__()
        self.convW = CovW()
    def forward(self, feature_D_I, feature_D_V):
        w = self.convW(feature_D_I)
        add = w*feature_D_I
        out = torch.add(add, feature_D_V)
        return out



class Fusioner_B(nn.Module):
    def __init__(self):
        super(Fusioner_B, self).__init__()
        self.convV1 = CovW()
        self.convV2 = CovW()
        self.convK1 = CovW()
        self.convK2 = CovW()
        self.convQ1 = CovW()
        self.convQ2 = CovW()
        self.conv1_1 = CovW()
        self.conv1_2 = CovW()
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            )
    def forward(self, feature_B_I, feature_B_V):
        V1 = self.convV1(feature_B_I)
        K1 = self.convK1(feature_B_I)
        Q1 = self.convQ1(feature_B_I)
        V2 = self.convV2(feature_B_V)
        K2 = self.convK2(feature_B_V)
        Q2 = self.convQ2(feature_B_V)
        Z1 = V1*(Q2*K1)
        Z2 = V2*(Q1*K2)
        C = self.conv2(torch.cat([Z1,Z2], 1))
        C1 = self.conv1_1(feature_B_I)
        C2 = self.conv1_2(feature_B_V)
        return C*C1*C2






