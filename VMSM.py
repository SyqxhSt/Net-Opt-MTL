import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
'''
============================================================================================================
The selected expansion rate and step size in this scheme are 2, 4, and 6,
respectively, ensuring that the expansion rate and step size are consistent.
The expansion rate and step size can be determined based on the size of the images in the specific dataset.
For larger images, it is recommended to use a larger expansion rate and step size
============================================================================================================
'''

class VMSM(nn.Module):
    def __init__(self, dim, in_dim):
        super(VMSM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU())        # Expansion rate of 2 (adjustable expansion rate)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU())        # Expansion rate of 4 (adjustable expansion rate)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU())        # Expansion rate of 6 (adjustable expansion rate)
        
        # If batch=1, there will be issues with batchnorm
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        # self.fuse_1 = nn.Sequential(nn.Conv2d(3 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        self.fuse_2 = nn.Sequential(nn.Conv2d(down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        self.softmax = Softmax(dim=-1)

        # Depthwise separable convolution
        self.conv_dws2 = DepthwiseSeparableConv1(down_dim, down_dim)
        self.conv_dws3 = DepthwiseSeparableConv2(down_dim, down_dim)
        self.conv_dws4 = DepthwiseSeparableConv3(down_dim, down_dim)
        # 1*1 Conv
        # self.convcat = nn.Sequential(nn.Conv2d(3 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim),nn.PReLU())

    def forward(self, x):
        S = nn.Sigmoid()

        x = self.down_conv(x)
        conv1 = self.conv1(x)
        
        conv2 = self.conv2(x)
        conv2 = self.conv_dws2(conv2)
        conv2 = F.interpolate(conv2, size=x.size()[2:], mode='bilinear', align_corners=False)

        conv3 = self.conv3(x)
        conv3 = self.conv_dws3(conv3)
        conv3 = F.interpolate(conv3, size=x.size()[2:], mode='bilinear', align_corners=False)

        conv4 = self.conv4(x)
        conv4 = self.conv_dws4(conv4)
        conv4 = F.interpolate(conv4, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        convc = conv2 + conv3 + conv4
        # convc = self.convcat(torch.cat((conv2, conv3, conv4), 1))
        # Bilinear represents upsampling using bilinear interpolation, and x.size() returns a four-dimensional data (B, C, H, W) as the value
        
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=False)
        # return self.fuse_1(torch.cat((conv1, out2, out3,out4, conv5), 1))

        # return self.fuse_1(torch.cat((conv1, convc, conv5), 1))
        return self.fuse_2(conv1 + conv2 + conv5)


class DepthwiseSeparableConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):        # Step size 2 corresponds to an expansion rate of 2 (the expansion rate can be adjusted)
        super(DepthwiseSeparableConv1, self).__init__()

        # Deep convolutional layer
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # Pointwise convolutional layer
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=4):        # Step size 4 corresponds to an expansion rate of 4 (the expansion rate can be adjusted)
        super(DepthwiseSeparableConv2, self).__init__()

        # Deep convolutional layer
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # Pointwise convolutional layer
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=6):        # Step size 6 corresponds to an expansion rate of 6 (the expansion rate can be adjusted)
        super(DepthwiseSeparableConv3, self).__init__()

        # Deep convolutional layer
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # Pointwise convolutional layer
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
