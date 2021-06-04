"""
20-Mai-2020
https://github.com/victorca25/BasicSR/blob/14aced7d1049a283761c145f3cf300a94c6ac4b9/codes/models/modules/architectures/SRFlow/glow_arch.py
"""
import torch.nn as nn


def f_conv2d_bias(in_channels, out_channels):
    def padding_same(kernel, stride):
        return [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)]

    padding = padding_same([3, 3], [1, 1])
    assert padding == [1, 1], padding
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[3, 3], stride=1, padding=1,
                  bias=True))
