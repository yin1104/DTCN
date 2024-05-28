import torch
from torch import nn
from thop import profile
"""
Compact Convolutional Neural Network (Compact-CNN)
EEGNet variant used for classification of Steady State Visual Evoked Potential (SSVEP) Signals 
Compact Convolutional Neural Networks for Classification of Asynchronous Steady-state Visual Evoked Potentials
https://arxiv.org/pdf/1803.04566.pdf
"""


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        global padding
        super(SeparableConv2d, self).__init__()

        if isinstance(kernel_size, int):
            padding = kernel_size // 2

        if isinstance(kernel_size, tuple):
            padding = (
                kernel_size[0] // 2 if kernel_size[0] - 1 != 0 else 0,
                kernel_size[1] // 2 if kernel_size[1] - 1 != 0 else 0
            )

        self.depthwise = DepthwiseConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                         padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# def EEGNet_SSVEP(nb_classes = 12, Chans = 8, Samples = 256,
#              dropoutRate = 0.5, kernLength = 256, F1 = 96,
#              D = 1, F2 = 96, dropoutType = 'Dropout'):
class CompactEEGNet(nn.Module):
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, f1=96, f2=96, d=1):
        super().__init__()

        self.signal_length = signal_length

        # layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, signal_length), padding=(0, signal_length // 2))
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(f1, d * f1, (num_channel, 1), groups=f1)
        self.bn2 = nn.BatchNorm2d(d * f1)
        self.avgpool1 = nn.AvgPool2d((1, 4))

        # layer 2
        self.separable_conv = SeparableConv2d(
            in_channels=f1,
            out_channels=f2,
            kernel_size=(1, 16)
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.avgpool2 = nn.AvgPool2d((1, 8))

        # layer 3
        self.fc = nn.Linear(in_features=f2 * (signal_length // 32), out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.elu = nn.ELU()

    def forward(self, x):
        # layer 1
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)  # [2, 96, 1, 125]
        x = self.avgpool1(x)
        x = self.dropout(x)

        # layer 2
        x = self.separable_conv(x)  # [2, 96, 1, 32]
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
        #
        # # layer 3
        x = torch.flatten(x, start_dim=1)  # [2, 96, 1, 4]
        x = self.fc(x)  # [2, 384]

        return x


def test():
    model = CompactEEGNet(
        num_channel=8,
        num_classes=40,
        signal_length=int(250*1.4),
    )

    x = torch.randn(1, 8, int(250*1.4))
    print("Input shape:", x.shape)  # torch.Size([2, 11, 250])
    y = model(x)
    print("Output shape:", y.shape)  # torch.Size([2, 40])
    print(model)
    flops, params = profile(model, inputs=(x,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))



if __name__ == "__main__":
    test()