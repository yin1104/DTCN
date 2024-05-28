# -*- coding: utf-8 -*-
"""
Guney's network proposed in A Deep Neural Network for SSVEP-based Brain-Computer Interfaces.

Modified from https://github.com/osmanberke/Deep-SSVEP-BCI.git
"""
from collections import OrderedDict
# from thop import profile
import torch
import torch.nn as nn
import numpy as np
from scipy import signal


def compute_out_size(input_size: int, kernel_size: int,
                     stride: int = 1, padding: int = 0, dilation: int = 1):
    return int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = ((stride - 1) * input_size - stride + dilation * (kernel_size - 1) + 1)
    return (all_padding // 2, all_padding - all_padding // 2)


def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0])
    lr = compute_same_pad1d(input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1])
    return [*lr, *ud]


def _narrow_normal_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    narrow normal distribution N(0, 0.01).
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def Filterbank(x, sampling, filterIdx):
    # x: time signal, np array with format (electrodes,data)
    # sampling: sampling frequency
    # filterIdx: filter index

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Nq = sampling / 2
    Wp = [passband[filterIdx] / Nq, 90 / Nq]
    Ws = [stopband[filterIdx] / Nq, 100 / Nq]
    [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
    [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')
    y = np.zeros(x.shape)
    channels = x.shape[0]
    for c in range(channels):
        y[c, :] = signal.filtfilt(B, A, x[c, :], padtype='odd', padlen=3 * (max(len(B), len(A)) - 1), axis=-1)
    return y


class GuneyNet(nn.Module):
    """
    Guney's network for decoding SSVEP.
    They used two stages to train the network. 
    
    The first stage is with all training data in the dataset. 
    lr: 1e-4, batch_size: 100, l2_regularization: 1e-3, epochs: 1000
    
    The second stage is a fine-tuning process with each subject's training data.
    lr: 1e-4, batch_size: full size, l2_regularization: 1e-3, epochs: 1000
    spatial_dropout=time1_dropout=0.6
    """
    def __init__(self, n_channels, n_samples, n_classes, n_bands,
        n_spatial_filters=120, spatial_dropout=0.1,
        time1_kernel=2, time1_stride=2, n_time1_filters=120,
        time1_dropout=0.1,
        time2_kernel=10, n_time2_filters=120,
        time2_dropout=0.95):
        # super(GuneyNet, self).__init__()
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_bands = n_bands

        self.model = nn.Sequential(OrderedDict([
            ('band_layer', nn.Conv2d(n_bands, 1, (1, 1), bias=False)),
            ('spatial_layer', nn.Conv2d(1, n_spatial_filters, (n_channels, 1))),
            ('spatial_dropout', nn.Dropout(spatial_dropout)),
            ('time1_layer', 
                nn.Conv2d(n_spatial_filters, n_time1_filters, (1, time1_kernel), 
                    stride=(1, time1_stride))),
            ('time1_dropout', nn.Dropout(time1_dropout)),
            ('relu', nn.ReLU()),
            ('same_padding',
                nn.ConstantPad2d(
                    compute_same_pad2d(
                        (1, compute_out_size(n_samples, time1_kernel, stride=time1_stride)), 
                        (1, time2_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('time2_layer', 
                nn.Conv2d(n_time1_filters, n_time2_filters, (1, time2_kernel), 
                stride=(1, 1))),
            ('time2_dropout', nn.Dropout(time2_dropout)),
            ('flatten', nn.Flatten()),
            ('fc_layer', nn.Linear(
                n_time2_filters*compute_out_size(n_samples, time1_kernel, stride=time1_stride),
                n_classes))
        ]))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        _narrow_normal_weight_zero_bias(self)
        nn.init.ones_(self.model[0].weight)
        # MATLAB uses xavier_uniform_ with varaiance 2/(input+output)
        # perhaps this is a mistake in Help document
        nn.init.xavier_normal_(self.model[-1].weight, gain=1)

    def forward(self, X):
        # X: (n_batch, n_bands, n_channels, n_samples)
        out = self.model(X)
        return out


def test():
    model = GuneyNet(
        n_channels=8,
        n_classes=40,
        n_samples=250,
        n_bands=3,
    )

    x = torch.randn(1, 3, 8, 250)  # 3是输入的band---》我们改成delay，搞成输入输出同大小
    print("Input shape:", x.shape)  # torch.Size([2, 11, 250])
    y = model(x)
    print("Output shape:", y.shape)  # torch.Size([2, 40])
    print(model)
    # flops, params = profile(model, inputs=(x,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))



if __name__ == "__main__":
    test()
