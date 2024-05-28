import torch
from torch import nn
# from thop import profile
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        # 可以整合一下重新写一个
        self.net = nn.Sequential(self.conv1, self.bn1, self.chomp1, self.elu1, self.dropout1,
                                 self.conv2, self.bn1, self.chomp2, self.elu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# def EEGNet_SSVEP(nb_classes = 12, Chans = 8, Samples = 256,
#              dropoutRate = 0.5, kernLength = 256, F1 = 96,
#              D = 1, F2 = 96, dropoutType = 'Dropout'):

class EEGTCNet(nn.Module):
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, f1=8, f2=16, d=2,kernal_tcn=4,kernal_eeg=32):
        super().__init__()

        self.signal_length = kernal_eeg

        # layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, kernal_eeg), padding=(0, kernal_eeg // 2))
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(f1, d * f1, (num_channel, 1), groups=f1)
        self.bn2 = nn.BatchNorm2d(d * f1)
        self.avgpool1 = nn.AvgPool2d((1, 4))

        # layer 2
        self.separable_conv = SeparableConv2d(
            in_channels=d * f1,
            out_channels=f2,
            kernel_size=(1, 16)
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.avgpool2 = nn.AvgPool2d((1, 8))

        # layer 3
        
        self.fc = nn.Linear(in_features=f2 * (signal_length // 32), out_features=num_classes)
        self.tcn = TemporalConvNet(f2, num_channels=[f2] * 2, kernel_size=kernal_tcn, dropout=0.3)

        self.dropout = nn.Dropout(p=0.2)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # layer 1
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout(x)

        # layer 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
        x = torch.squeeze(x)  # [2, 96, 63]
        x = self.tcn(x)  # [2, 96, 63]
        x = torch.unsqueeze(x, 2)  # [2, 96, 1, 63]

        # layer 3
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = self.softmax(x, dim=1)

        return x


def test():
    model = EEGTCNet(
        num_channel=11,
        num_classes=40,
        signal_length=250,
    )

    x = torch.randn(2, 11, 250)
    print("Input shape:", x.shape)  # torch.Size([2, 11, 250])
    y = model(x)
    print("Output shape:", y.shape)  # torch.Size([2, 40])
    print(model)
    # flops, params = profile(model, inputs=(x,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))



if __name__ == "__main__":
    test()