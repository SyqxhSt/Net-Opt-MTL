import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 8

'''
========================================================================================================
The feature map X is processed through depthwise separable convolution layers of varying configurations,
generating multi-scale feature maps with channel dimensions of C/2, C/4, C/8, and C/16.
These multi-scale feature maps are then passed through global average pooling, activation functions,
and linear layers to extract channel attention at different channel scales
========================================================================================================
'''

# Depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Deep Convolutional Layer
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # Pointwise convolution layer
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class CACM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CACM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)        # GAP
        
        # If set to False, the nn. int. constanc_ (m.bias, 0) in line 91 of model_degnet_mtan-py needs to be commented out
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,bias=True)
        
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Depthwise separable convolution
        self.fc1dws = DepthwiseSeparableConv(channels, channels // 2)        # C/2
        self.fc2dws = DepthwiseSeparableConv(channels, channels // 4)        # C/4
        self.fc3dws = DepthwiseSeparableConv(channels, channels // 8)        # C/8
        self.fc4dws = DepthwiseSeparableConv(channels, channels // reduction)        # C/16

        # 1 * 1 Conv replaces linear layers
        self.linear1 = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True)
        self.linear2 = nn.Conv2d(channels // 4, channels, kernel_size=1, bias=True)
        self.linear3 = nn.Conv2d(channels // 8, channels, kernel_size=1, bias=True)
        self.linear4 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)

    def forward(self, x):
        module_input = x
        x1 = x
        x2 = x
        x3 = x
        x4 = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.gelu(x)
        x = self.fc2(x)

        x1 = self.fc1dws(x1)
        x1 = self.avg_pool(x1)
        x1 = self.gelu(x1)
        x1 = self.linear1(x1)        # Channel characteristics corresponding to C/2 scale for x1

        x2 = self.fc2dws(x2)
        x2 = self.avg_pool(x2)
        x2 = self.gelu(x2)
        x2 = self.linear2(x2)        # Channel characteristics corresponding to C/4 scale for x1

        x3 = self.fc3dws(x3)
        x3 = self.avg_pool(x3)
        x3 = self.gelu(x3)
        x3 = self.linear3(x3)        # Channel characteristics corresponding to C/8 scale for x1

        x4 = self.fc4dws(x4)
        x4 = self.avg_pool(x4)
        x4 = self.gelu(x4)
        x4 = self.linear4(x4)        # Channel characteristics corresponding to C/16 scale for x1

        x = x + x1 + x2 + x3 + x4
        x = self.sigmoid(x)

        return module_input * x



