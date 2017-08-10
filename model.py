import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torch.autograd.variable import Variable


class NetD(nn.Module):
    def __init__(self, de_layer_num, ndf):
        super(NetD, self).__init__()
        self.de_layer_num = de_layer_num
        convs = []
        BNs = []
        for il in np.arange(de_layer_num):
            if il == 0:
                _conv = nn.Conv2d(
                    3,
                    ndf,
                    4, 2, 1, bias=False)
                convs.append(_conv)
            elif il == de_layer_num - 1:
                _conv = nn.Conv2d(
                    ndf * (2 ** (il - 1)),
                    1,
                    8, 1, 0, bias=False)
                convs.append(_conv)
            else:
                _conv = nn.Conv2d(
                    ndf * (2 ** (il - 1)),
                    ndf * (2 ** il),
                    4, 2, 1, bias=False
                )
                _BN = nn.BatchNorm2d(
                    ndf * (2 ** il))
                convs.append(_conv)
                BNs.append(_BN)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        ibl = 0
        for il in range(self.de_layer_num):
            if il == 0:
                x = functional.leaky_relu(
                    self.convs[il](x))
            elif il == self.de_layer_num - 1:
                x = functional.sigmoid(
                    self.convs[il](x))
            else:
                x = functional.leaky_relu(
                    self.BNs[ibl](
                        self.convs[il](x)))
                ibl += 1
        x = x.view(-1)
        return x


class NetG(nn.Module):
    def __init__(self, gen_layer_num, ngf, nz, up_type='deconv'):
        super(NetG, self).__init__()
        self.gen_layer_num = gen_layer_num
        self.up_type = up_type
        self.ngf = ngf
        self.nz = nz
        if up_type == 'shuffle':
            convs = []
            BNs = []
            for il in np.arange(gen_layer_num):
                if il == 0:
                    _conv = nn.Linear(nz, ngf * (2 ** (gen_layer_num - 2 - il)) * 64)
                    _BN = nn.BatchNorm2d(ngf * (2 ** (gen_layer_num - 2 - il)))
                    convs.append(_conv)
                    BNs.append(_BN)
                elif il == gen_layer_num - 1:
                    _conv = nn.Conv2d(
                        ngf * (2 ** (gen_layer_num - 1 - il)),
                        4*3,
                        5, 1, 2)
                    convs.append(_conv)
                else:
                    _conv = nn.Conv2d(
                        ngf * (2 ** (gen_layer_num - 1 - il)),
                        4*ngf * (2 ** (gen_layer_num - 2 - il)),
                        5, 1, 2)
                    _BN = nn.BatchNorm2d(4*ngf * (2 ** (gen_layer_num - 2 - il)))
                    convs.append(_conv)
                    BNs.append(_BN)
        else:
            convs = []
            BNs = []
            for il in np.arange(gen_layer_num):
                if il == 0:
                    _conv = nn.ConvTranspose2d(
                        nz,
                        ngf * (2 ** (gen_layer_num - 2 - il)),
                        8, 1, 0, bias=False)
                    _BN = nn.BatchNorm2d(ngf * (2 ** (gen_layer_num - 2 - il)))
                    convs.append(_conv)
                    BNs.append(_BN)
                elif il == gen_layer_num - 1:
                    _conv = nn.ConvTranspose2d(
                        ngf * (2 ** (gen_layer_num - 1 - il)),
                        3,
                        4, 2, 1)
                    convs.append(_conv)
                else:
                    _conv = nn.ConvTranspose2d(
                        ngf * (2 ** (gen_layer_num - 1 - il)),
                        ngf * (2 ** (gen_layer_num - 2 - il)),
                        4, 2, 1)
                    _BN = nn.BatchNorm2d(ngf * (2 ** (gen_layer_num - 2 - il)))
                    convs.append(_conv)
                    BNs.append(_BN)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, z):
        #
        if self.up_type == 'shuffle':
            for il in np.arange(self.gen_layer_num):
                if il == 0:
                    z = self.convs[il](z.view(-1, self.nz))
                    z = z.view(-1, self.ngf * (2 ** (self.gen_layer_num - 2 - il)), 8, 8)
                    z = functional.relu(
                        self.BNs[il](z))
                elif il == self.gen_layer_num - 1:
                    z = self.convs[il](z)
                    z = functional.tanh(z)
                    z = functional.pixel_shuffle(z, 2)
                else:
                    z = self.convs[il](z)
                    z = functional.relu(self.BNs[il](z))
                    z = functional.pixel_shuffle(z, 2)
        else:
            for il in range(self.gen_layer_num):
                if il == self.gen_layer_num - 1:
                    z = functional.tanh(
                        self.convs[il](z))
                else:
                    z = functional.relu(
                        self.BNs[il](
                            self.convs[il](z)))
        return z
