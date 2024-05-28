import torch
from torch import nn
import math
# from thop import profile
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


class NoTCNet(nn.Module):

    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=8, kernel_window_global=125,
                 kernel_window_local=51, kernel_window=16):
        super(NoTCNet, self).__init__()
        filters = [filters_n1, filters_n1 * 2]
        out_len = signal_length//8
        self.global_block = SingleScaleBlock(num_channel, filters[0], kernel_window_global)
        self.local_block = SingleScaleBlock(num_channel, filters[0], kernel_window_local)
        # self.atten = ECA_Block(in_channel=filters[0]*2)
        self.avgpool = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(filters[1] * out_len, out_features=num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x1 = self.global_block(x)
        x2 = self.local_block(x)
        x = torch.cat((x1, x2), 1)
        # x = self.atten(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def test():
    model = NoTCNet(
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
