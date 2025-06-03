# -*- coding: utf-8 -*-

import torch
from torch import nn

channel=64

 
class AE_Encoder(nn.Module):
    def __init__(self):
        super(AE_Encoder, self).__init__()
        self.cov1=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, channel, 3, padding=0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            )
        self.cov2=nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            )
        self.cov3=nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            #nn.PReLU(),
            nn.Tanh(),
            )
        self.cov4=nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            #nn.PReLU(),
            nn.Tanh(),
            )
        
    def forward(self, data_train):
        feature_1=self.cov1(data_train)
        feature_2=self.cov2(feature_1)
        feature_B=self.cov3(feature_2)
        feature_D=self.cov4(feature_2)
        return feature_1,feature_2,feature_B, feature_D
       


class Ind_Decoder(nn.Module):
    def __init__(self):
        super(Ind_Decoder, self).__init__()
        self.cov1=nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            #nn.PReLU(),
            nn.Tanh(),
            )
        self.cov2=nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            )
        self.out_1=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel*2, 1, 3, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            )
        self.out_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel*2, 1, 3, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            )
        self.out_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel*3, 1, 3, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            )
    def forward(self,feature_1,feature_2,feature_B,feature_D):
        output1_1 = self.cov1(feature_B)
        output2_1 = self.cov2(torch.cat([output1_1, feature_2], 1))
        output3 = self.out_1(torch.cat([output2_1,feature_1],1))
        output3 = nn.AvgPool2d(kernel_size=2, stride=2)(output3)
        output1_2 = self.cov1(feature_D)
        output2_2 = self.cov2(torch.cat([output1_2, feature_2], 1))
        output4 = self.out_2(torch.cat([output2_2, feature_1],1))
        output4 = nn.AvgPool2d(kernel_size=2, stride=2)(output4)
        output5 = self.out_3(torch.cat([output2_2, output2_1, feature_1], 1))
        return output3, output4, output5


