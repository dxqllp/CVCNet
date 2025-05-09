# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock (with optional skip connection)"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        self.has_skip = True  # 动态判断是否启用 skip
        self.in_channels2 = in_channels2

        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling in [1, 2, 3]:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            if mode_upsampling == 1:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            elif mode_upsampling == 2:
                self.up = nn.Upsample(scale_factor=2, mode='nearest')
            else:
                self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        # 延迟构建 conv block，在 forward 时判断是否有 skip
        self.conv_block_with_skip = ConvBlock(in_channels2 * 2, out_channels, dropout_p)
        self.conv_block_no_skip = ConvBlock(in_channels2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            x = self.conv_block_with_skip(x)
        else:
            x = self.conv_block_no_skip(x1)

        return x

class LocalFeatureExtractionModule2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalFeatureExtractionModule2D, self).__init__()

        # 深度可分离卷积模块（使用 groups=in_channels 实现 depthwise）
        self.depthwise_conv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.depthwise_conv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.depthwise_conv_7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # 点卷积
        self.pointwise_conv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pointwise_conv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pointwise_conv_7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 最终融合
        self.conv_1x1 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x3 = self.pointwise_conv_3x3(self.depthwise_conv_3x3(x))
        x5 = self.pointwise_conv_5x5(self.depthwise_conv_5x5(x))
        x7 = self.pointwise_conv_7x7(self.depthwise_conv_7x7(x))

        x_fused = torch.cat([x3, x5, x7], dim=1)
        out = self.conv_1x1(x_fused)

        return out

class GAM_Attention2D(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention2D, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, in_channels, kernel_size=1)
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 通道注意力（可进一步用于加权，但此处未加权输出）
        x_channel_att = self.channel_attention(x)

        # 空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()

        # 加权
        out = x * x_spatial_att

        return out


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Encoder1(nn.Module):
    def __init__(self, params):
        super(Encoder1, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        # 插入Local Feature Extraction模块（LFE）
        self.lfe2 = LocalFeatureExtractionModule2d(self.ft_chns[1], self.ft_chns[1])
        self.lfe3 = LocalFeatureExtractionModule2d(self.ft_chns[2], self.ft_chns[2])
        self.lfe4 = LocalFeatureExtractionModule2d(self.ft_chns[3], self.ft_chns[3])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x1 = self.lfe2(x1)

        x2 = self.down2(x1)
        x2 = self.lfe3(x2)

        x3 = self.down3(x2)
        x3 = self.lfe4(x3)

        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]

class Encoder2(nn.Module):
    def __init__(self, params, rate=4):
        super(Encoder2, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        # 添加 GAM 注意力模块（2D 版本）
        self.gam_attention_1 = GAM_Attention_2d(self.ft_chns[1], self.ft_chns[1], rate)
        self.gam_attention_2 = GAM_Attention_2d(self.ft_chns[2], self.ft_chns[2], rate)
        self.gam_attention_3 = GAM_Attention_2d(self.ft_chns[3], self.ft_chns[3], rate)

    def forward(self, x):
        x0 = self.in_conv(x)           # 输出通道 ft_chns[0]
        x1 = self.down1(x0)            # 输出通道 ft_chns[1]
        x1 = self.gam_attention_1(x1)  # 加 GAM 模块

        x2 = self.down2(x1)            # 输出通道 ft_chns[2]
        x2 = self.gam_attention_2(x2)

        x3 = self.down3(x2)            # 输出通道 ft_chns[3]
        x3 = self.gam_attention_3(x3)

        x4 = self.down4(x3)            # 输出通道 ft_chns[4]

        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder1(nn.Module):
    def __init__(self, params):
        super(Decoder1, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)  # 保留 skip
        x = self.up2(x, None)  # 去除 x2
        x = self.up3(x, x1)  # 保留 skip
        x = self.up4(x, None)  # 去除 x0
        output = self.out_conv(x)
        return output


class Decoder2(nn.Module):
    def __init__(self, params):
        super(Decoder2, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, None)  # 保留 skip
        x = self.up2(x, x2)  # 去除 x2
        x = self.up3(x, None)  # 保留 skip
        x = self.up4(x, x0)  # 去除 x0
        output = self.out_conv(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1


class MCNet2d(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}
        self.encoder1 = Encoder(params1)
        self.encoder2 = Encoder1(params1)
        self.encoder3 = Encoder2(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder1(params2)
        self.decoder3 = Decoder2(params3)

    def forward(self, x):
        feature1 = self.encoder1(x)
        feature2 = self.encoder2(x)
        feature3 = self.encoder3(x)
        output1 = self.decoder1(feature1)
        output2 = self.decoder2(feature2)
        output3 = self.decoder3(feature3)
        return output1, output2, output3

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info

    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb;

    ipdb.set_trace()
