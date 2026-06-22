"""MiniFASNet — vendored from minivision-ai/Silent-Face-Anti-Spoofing.

Original by zhuying @ Minivision, Apache 2.0. Trimmed to the V2 branch
matching the 2.7_80x80_MiniFASNetV2.pth checkpoint.

Architecture: stem (3x3 stride-2 conv + 3x3 depthwise); body of 3 stages
of depthwise-separable blocks with residuals and stride-2 transitions;
head (separable 1x1 conv + linear block + flatten + FC + BN + dropout +
classifier) emitting logits over num_classes (3 default: live/paper/
replay). `get_kernel(h, w)` gives the conv6_kernel size — (5, 5) at 80x80.
"""

from __future__ import annotations

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)


def get_kernel(height: int, width: int) -> tuple[int, int]:
    """Conv6 depth-wise kernel size as a function of input resolution."""
    return ((height + 15) // 16, (width + 15) // 16)


class _Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _DepthWise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3),
                 stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = _ConvBlock(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = _ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel,
                                  padding=padding, stride=stride)
        self.project = _LinearBlock(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return short_cut + x if self.residual else x


class _Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3),
                 stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = []
        for i in range(num_block):
            modules.append(
                _DepthWise(c1[i], c2[i], c3[i], residual=True,
                           kernel=kernel, padding=padding, stride=stride, groups=groups)
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# Channel configuration for V2 — original name `'1.8M_'` from upstream.
_KEEP_V2 = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
    13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
    52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
    128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
]


class _MiniFASNet(Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),
                 drop_p=0.0, num_classes=3, img_channel=3):
        super().__init__()
        self.embedding_size = embedding_size

        self.conv1 = _ConvBlock(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = _ConvBlock(keep[0], keep[1], kernel=(3, 3), stride=(1, 1),
                                   padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = _Residual(c1, c2, c3, num_block=4, groups=keep[4],
                                kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]),
              (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]),
              (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]),
              (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = _Residual(c1, c2, c3, num_block=6, groups=keep[19],
                                kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = _DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = _Residual(c1, c2, c3, num_block=2, groups=keep[40],
                                kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = _ConvBlock(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = _LinearBlock(keep[47], keep[48], groups=keep[48],
                                      kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = _Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


def MiniFASNetV2(embedding_size: int = 128, conv6_kernel=(7, 7),
                 drop_p: float = 0.2, num_classes: int = 3,
                 img_channel: int = 3) -> _MiniFASNet:
    """Instantiate the V2 variant (0.43M params)."""
    return _MiniFASNet(_KEEP_V2, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)
