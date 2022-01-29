import torch
from torch import nn
from net_block import *

config = [
    [-1, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 4, 2],
    [6, 64, 4, 2],
    [6, 96, 4, 1],
    [6, 160, 4, 2],
    [6, 320, 2, 1],
]


class Bottleneck(nn.Module):
    def __init__(self, c_in, i, t, c, n, s):
        super(Bottleneck, self).__init__()
        self.i = i
        self.n = n
        _s = s if i == n - 1 else 1
        _c = c if i == n - 1 else c_in
        _p_c = c_in * t

        self.sub_module = nn.Sequential(
            nn.Conv2d(c_in, _p_c, 1, 1, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _p_c, 3, _s, 1, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self, x):
        if self.i == self.n - 1:
            return self.sub_module(x)
        else:
            return self.sub_module(x) + x


class MobileNet_v2(nn.Module):
    def __init__(self, config):
        super(MobileNet_v2, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        c_in = config[0][1]
        for t, c, n, s in config[1:4]:
            for i in range(n):
                self.blocks1.append(Bottleneck(c_in, i, t, c, n, s))
            c_in = c
        for t, c, n, s in config[4:5]:
            for i in range(n):
                self.blocks2.append(Bottleneck(c_in, i, t, c, n, s))
            c_in = c
        for t, c, n, s in config[5:]:
            for i in range(n):
                self.blocks3.append(Bottleneck(c_in, i, t, c, n, s))
            c_in = c

        self.hidden_layers1 = nn.Sequential(*self.blocks1)
        self.hidden_layers2 = nn.Sequential(*self.blocks2)
        self.hidden_layers3 = nn.Sequential(*self.blocks3)

        self.convset_13 = ConvolutionalSet(320, 64)
        self.detetion_13 = nn.Sequential(
            ConvolutionalLayer(64, 320, 3, 1, 1),
            nn.Conv2d(320, 24, 1, 1, 0)
        )
        self.up_13_to_26 = nn.Sequential(
            ConvolutionalLayer(64, 32, 3, 1, 1),
            UpSampleLayer()
        )

        self.convset_26 = ConvolutionalSet(96, 32)
        self.detetion_26 = nn.Sequential(
            ConvolutionalLayer(32, 64, 3, 1, 1),
            nn.Conv2d(64, 24, 1, 1, 0)
        )
        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(32, 16, 3, 1, 1),
            UpSampleLayer()
        )

        self.convset_52 = ConvolutionalSet(48, 24)
        self.detetion_52 = nn.Sequential(
            ConvolutionalLayer(24, 48, 3, 1, 1),
            nn.Conv2d(48, 24, 1, 1, 0)
        )

    def forward(self, x):
        out_52 = self.hidden_layers1(self.input_layer(x))
        out_26 = self.hidden_layers2(out_52)
        out_13 = self.hidden_layers3(out_26)

        convset_out_13 = self.convset_13(out_13)
        detetion_out_13 = self.detetion_13(convset_out_13)
        up_13_to_26_out = self.up_13_to_26(convset_out_13)
        cat_out_26 = torch.cat((up_13_to_26_out, out_26), dim=1)

        convset_26 = self.convset_26(cat_out_26)
        detetion_out_26 = self.detetion_26(convset_26)
        up_26_to_52_out = self.up_26_to_52(convset_26)
        cat_out_52 = torch.cat((up_26_to_52_out, out_52), dim=1)

        convset_52 = self.convset_52(cat_out_52)
        detetion_out_52 = self.detetion_52(convset_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    x = torch.randn((1, 3, 416, 416))
    net = MobileNet_v2(config)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
