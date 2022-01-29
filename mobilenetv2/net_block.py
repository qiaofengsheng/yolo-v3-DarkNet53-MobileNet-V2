from torch import nn
import torch
from torch.nn import functional


# 卷积块
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


# 残差块
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
                               padding=0),
            ConvolutionalLayer(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1,
                               padding=1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 下采样
class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 上采样
class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


# 卷积集
class ConvolutionalSet(nn.Module):
    # 一般输入通道大  输出通道小，因为目的就是为了降低通道进行特征提取
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),

            ConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),

            ConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.sub_module(x)


class YoloNet_V3(nn.Module):
    def __init__(self):
        super(YoloNet_V3, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),

            ResidualLayer(64),
            DownSampleLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownSampleLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )
        self.trunk_26 = nn.Sequential(
            DownSampleLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )
        self.trunk_13 = nn.Sequential(
            DownSampleLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024, 512)
        )
        self.detetion_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 45, 1, 1, 0)
        )
        self.up_13_to_26 = nn.Sequential(
            # 原文为1*1的卷积，使用3*3的卷积是为了做特征提取，因为1*1不能进行特征提取
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpSampleLayer()
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalSet(768, 256)
        )
        self.detetion_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 45, 1, 1, 0)
        )

        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpSampleLayer()
        )

        self.convset_52 = nn.Sequential(
            ConvolutionalSet(384, 128)
        )
        self.detetion_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 45, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up_out_13_to_26 = self.up_13_to_26(convset_out_13)
        cat_out_26 = torch.cat((up_out_13_to_26, h_26), dim=1)
        convset_out_26 = self.convset_26(cat_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_26_to_52 = self.up_26_to_52(convset_out_26)
        cat_out_52 = torch.cat((up_out_26_to_52, h_52), dim=1)
        convset_out_52 = self.convset_52(cat_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    yolo = YoloNet_V3()
    x = torch.randn(1, 3, 416, 416)
    y = yolo(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
