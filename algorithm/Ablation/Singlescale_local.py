import torch
from torch import nn
import math

fs = 250


class Conv2dBlockELU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=(0, 0), dilation=(1, 1), groups=1):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SingleScaleBlock(nn.Module):
    def __init__(self, num_channel=10, filters=8, kernel_window=125):
        super().__init__()
        self.conv_time = Conv2dBlockELU(in_ch=1, out_ch=filters,
                                        kernel_size=(1, kernel_window), padding=(0, int(kernel_window/2)))
        self.conv_chan = Conv2dBlockELU(in_ch=filters, out_ch=filters, kernel_size=(num_channel, 1))

    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv_chan(x)
        return x


class ECA_Block(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super().__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size+1

        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.weight = []

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = x * inputs
        self.weight.append(x.detach())
        return outputs


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


class LocalNet(nn.Module):

    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=8, kernel_window_global=125,
                 kernel_window_local=51, kernel_window=16):
        super(LocalNet, self).__init__()
        filters = [filters_n1, filters_n1]
        out_len = signal_length//8
        self.local_block = SingleScaleBlock(num_channel, filters[1], kernel_window_local)
        # self.atten = ECA_Block(in_channel=filters[0])
        self.tcn = TemporalConvNet(filters[1], num_channels=[filters[1]] * 2, kernel_size=kernel_window, dropout=0.25)
        self.avgpool = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(filters[1] * out_len, out_features=num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.local_block(x)
        # x = self.atten(x)

        x = torch.squeeze(x)
        x = self.tcn(x)
        x = torch.unsqueeze(x, 2)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def test():
    model = LocalNet(
        num_channel=11,
        num_classes=40,
        signal_length=250,
    )
    x = torch.randn(2, 11, 250)
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)
    print(model)


if __name__ == "__main__":
    test()