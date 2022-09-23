from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class down_conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size=3, stride=1, padding=1):
        super(down_conv_block, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                        padding=padding),
            nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),)
        self.res=nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.conv(x)#+self.res(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,
                            padding=1),
            nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True), )
        self.res = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.conv(x)  # +self.res(x)
        return x
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        psi = self.relu(g1)
        psi = self.psi(psi)
        out = x * psi
        return out,psi
class Features(torch.nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.res1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),)
        self.netOne = down_conv_block(3,64,3,1,1)
        self.netTwo = down_conv_block(64,64,3,1,1)
        self.netThr = down_conv_block(64,64,3,1,1)
        self.netFou = down_conv_block(64,64,3,1,1)
        self.netFive= down_conv_block(64,64,3,1,1)
    def forward(self, tenInput):
        # tenOne = torch.cat([self.netOne(tenInput),self.res1(tenInput)],1)
        # tenTwo = torch.cat([self.netTwo(tenOne),self.res2(tenOne)],1)
        # tenThr = torch.cat([self.netThr(tenTwo),self.res3(tenTwo)],1)
        # tenFou = torch.cat([self.netFou(tenThr),self.res4(tenThr)],1)
        # tenFive= torch.cat([self.netFive(tenFou),self.res5(tenFou)],1)
        tenOne=self.netOne(tenInput)+self.res1(tenInput)
        tenTwo=self.netTwo(tenOne)+tenOne
        tenThr=self.netThr(tenTwo)+tenTwo
        tenFou=self.netFou(tenThr)+ tenThr
        tenFive=self.netFive(tenFou)+tenFou
        return [tenOne, tenTwo, tenThr, tenFou, tenFive]
class Both_Features(torch.nn.Module):
    def __init__(self):
        super(Both_Features, self).__init__()
        self.Input_conv=down_conv_block(3,16,kernel_size=3,stride=1,padding=1)
        self.Both2 = down_conv_block(32, 32,kernel_size=3,stride=1,padding=1)
        self.Both3 = down_conv_block(32, 32, kernel_size=3, stride=1, padding=1)
        self.Both4 = down_conv_block(32, 32, kernel_size=3, stride=1, padding=1)
        self.Both5 = down_conv_block(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self,I1,I2):
        InputFeature1=self.Input_conv(I1)
        InputFeature2=self.Input_conv(I2)
        Both_Feature_1 = torch.cat([InputFeature1, InputFeature2], 1)
        Both_Feature_2 = self.Both2(Both_Feature_1)
        Both_Feature_3 = self.Both3(Both_Feature_2)
        Both_Feature_4 = self.Both4(Both_Feature_3)
        Both_Feature_5 = self.Both5(Both_Feature_4)
        BothFeatures = [Both_Feature_1, Both_Feature_2, Both_Feature_3, Both_Feature_4, Both_Feature_5]
        return BothFeatures
class Up_Net(torch.nn.Module):
    def __init__(self,in_cha=64,middle_cha=64):
        super(Up_Net,self).__init__()
        self.Up5=up_conv(in_cha,64)
        self.Up4=up_conv(middle_cha,64)
        self.Up3=up_conv(middle_cha,64)
        self.Up2=up_conv(middle_cha,64)
        self.Up1=up_conv(middle_cha,64)
        self.Up0=nn.Conv2d(64,1,3,1,1)

    def forward(self,Features,Both_Features,Img):
    #def forward(self, Features):
        Up_f5=  self.Up5 (torch.cat([Features[-1],Both_Features[-1],Img],1))
        Up_f4=  self.Up4 (torch.cat([Up_f5,Both_Features[-2],Img],1))#+Features[-2]
        Up_f3 = self.Up3 (torch.cat([Up_f4,Both_Features[-3],Img],1))#+Features[-3]
        Up_f2 = self.Up2 (torch.cat([Up_f3,Both_Features[-4],Img],1))#+Features[-4]
        Up_f1 = self.Up1 (torch.cat([Up_f2,Both_Features[-5],Img],1))#+Features[-5]
        out=self.Up0(Up_f1)
        return out
class Network_both(torch.nn.Module):
    def __init__(self):
        super(Network_both,self).__init__()
        self.Feature=Features()
        self.Up_net=Up_Net(99,99)
        self.Both_net=Both_Features()
    def forward(self, i1,i2):
        feature1=self.Feature(i1)
        feature2=self.Feature(i2)
        Both_Features=self.Both_net(i1,i2)
        out1=self.Up_net(feature1,Both_Features,i1)
        out2=self.Up_net(feature2,Both_Features,i2)
        return out1,out2
class Up_Net_Res(torch.nn.Module):
    def __init__(self,in_cha=64,middle_cha=64):
        super(Up_Net_Res,self).__init__()
        self.Up5=up_conv(in_cha,64)
        self.Up4=up_conv(middle_cha,64)
        self.Up3=up_conv(middle_cha,64)
        self.Up2=up_conv(middle_cha,64)
        self.Up1=up_conv(middle_cha,64)
        self.Up0=nn.Conv2d(64,1,3,1,1)
    def forward(self,Features,Both_Features,Img):
    #def forward(self, Features):
        Up_f5=  self.Up5 (torch.cat([Features[-1],Both_Features[-1],Img],1))
        Up_f4=  self.Up4 (torch.cat([Up_f5,Both_Features[-2],Features[-2]],1))
        Up_f3 = self.Up3 (torch.cat([Up_f4,Both_Features[-3],Features[-3]],1))
        Up_f2 = self.Up2 (torch.cat([Up_f3,Both_Features[-4],Features[-4]],1))
        Up_f1 = self.Up1 (torch.cat([Up_f2,Both_Features[-5],Features[-5]],1))
        out=self.Up0(Up_f1)
        return out
class Network_both_Res(torch.nn.Module):
    def __init__(self):
        super(Network_both_Res,self).__init__()
        self.Feature=Features()
        self.Up_net2=Up_Net_Res(99,160)
        self.Both_net=Both_Features()
    def forward(self, i1,i2):
        feature1=self.Feature(i1)
        feature2=self.Feature(i2)
        Both_Features=self.Both_net(i1,i2)
        out1=self.Up_net2(feature1,Both_Features,i1)
        out2=self.Up_net2(feature2,Both_Features,i2)
        return out1,out2




















