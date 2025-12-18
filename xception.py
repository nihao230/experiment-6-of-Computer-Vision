import torch
import torch.nn as nn
import torch.nn.functional as F


def _print_shape(func):
    def print_shape(*args, **kwargs):
        res = func(*args, **kwargs)
        print(func, res.shape)
        return res

    return print_shape


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True):

        super(SeparableConv2d, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding=padding, dilation=dilation,
                               groups=in_channels, bias=bias)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=bias)
        pass

    def forward(self, x):
        return self.pconv(self.dconv(x))  # 先depthwise conv，后pointwise conv
    
    pass


class ResidualConnection(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
		# 默认不下采样，只调整 Channel
        super(ResidualConnection, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        pass

    pass

class _PoolEnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu1=True): # 默认有 relu
        super(_PoolEnBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)
        self.relu1 = None
        if relu1:
            self.relu1 = nn.ReLU(inplace=False) # 特别地，这里要为 Fales
            
        self.sepconv1 = SeparableConv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=True)
        self.sepconv2 = SeparableConv2d(out_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        pass

    def forward(self, x):
        identity = self.project(x)  

        if self.relu1:  
            x = self.relu1(x)
        x = self.sepconv1(x)  # 1th
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.sepconv2(x)  # 2th
        x = self.bn2(x)

        x = self.maxpool(x)  # 下采样2倍
        x = x + identity  # residual connection
        return x

    pass


class _PoolMBlock(nn.Module):
    def __init__(self, in_channels=728):
        super(_PoolMBlock, self).__init__()
        
        mods = [
            nn.ReLU(inplace=False), 
            SeparableConv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        ]
        mods *= 3  # 重复 3 次
        self.convs = nn.Sequential(*mods)

    def forward(self, x):
        return x + self.convs(x)


class _PoolExBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        super(_PoolExBlock, self).__init__()
        self.project = ResidualConnection(in_channels, out_channels, stride=2)

        self.relu1 = nn.ReLU(inplace=False)  
        self.sepconv1 = SeparableConv2d(in_channels, in_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu2 = nn.ReLU(inplace=True)
        self.sepconv2 = SeparableConv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        pass

    def forward(self, x):
        identity = self.project(x) 

        x = self.relu1(x)
        x = self.sepconv1(x)  # 1th
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.sepconv2(x)  # 2th
        x = self.bn2(x)

        x = self.maxpool(x)  # 下采样2倍

        x = x + identity  # plus
        return x

    pass

class Xception(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Xception, self).__init__()
        #Entry flow
        conv1 = [nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),]
        self.entry_conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, 3, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True),]
        self.entry_conv2 = nn.Sequential(*conv2)

        self.entry_block1 = _PoolEnBlock(64, 128, relu1=False)
        self.entry_block2 = _PoolEnBlock(128, 256)
        self.entry_block3 = _PoolEnBlock(256, 728)

        #Middle flow
        self.middle_flow = nn.ModuleList([_PoolMBlock(728) for _ in range(8)])

        #Exit flow
        self.exit_block = _PoolExBlock(728, 1024)

        conv1 = [SeparableConv2d(1024, 1536, 3, padding=1, bias=False),
                 nn.BatchNorm2d(1536),
                 nn.ReLU(inplace=True),]
        self.exit_conv1 = nn.Sequential(*conv1)

        conv2 = [SeparableConv2d(1536, 2048, 3, padding=1, bias=False),
                 nn.BatchNorm2d(2048),
                 nn.ReLU(inplace=True),]
        self.exit_conv2 = nn.Sequential(*conv2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes) 


    def _init_weights(self): # Kaiming Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Entry
        x = self.entry_conv1(x)
        x = self.entry_conv2(x)

        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # Middle
        for block in self.middle_flow:
            x = block(x)

        # Exit
        x = self.exit_block(x)
        x = self.exit_conv1(x)
        x = self.exit_conv2(x)

        # FCnet
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x