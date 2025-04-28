
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import numpy as np
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 normalization='batch', activation='relu', dropout=0.0):
        super().__init__()
        # Flexible normalization
        norm_layer = {
            'batch': nn.BatchNorm2d,
            'instance': nn.InstanceNorm2d,
            'layer': nn.LayerNorm
        }.get(normalization, nn.BatchNorm2d)

        # Flexible activation
        act_layer = {
            'relu': nn.ReLU,
            'leaky': nn.LeakyReLU,
            'gelu': nn.GELU
        }.get(activation, nn.ReLU)

        mid_channels = out_channels if mid_channels is None else mid_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            act_layer(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_mode='max',
                 normalization='batch', activation='relu', dropout=0.0):
        super().__init__()
        # Flexible downsampling
        downsample_layers = {
            'max': nn.MaxPool2d(2),
            'avg': nn.AvgPool2d(2),
            'conv': nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        }.get(downsample_mode, nn.MaxPool2d(2))

        self.maxpool_conv = nn.Sequential(
            downsample_layers,
            DoubleConv(in_channels, out_channels,
                               normalization=normalization,
                               activation=activation,
                               dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False,
                 normalization='batch', activation='relu', dropout=0.0):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels,
                mid_channels=in_channels // 2,
                normalization=normalization,
                activation=activation,
                dropout=dropout
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels,
                normalization=normalization,
                activation=activation,
                dropout=dropout
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding to match dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SCFA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, reduction_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = self.head_dim ** -0.5

        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
            for k in [3, 5, 7]
        ])

        # Cross-modal projections
        self.x1_qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1),
            nn.BatchNorm2d(dim * 3),
            nn.GELU()
        )
        self.x2_qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1),
            nn.BatchNorm2d(dim * 3),
            nn.GELU()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Output projections
        self.project_out_x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out_x2 = nn.Conv2d(dim, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_x1 = nn.LayerNorm(dim)
        self.layer_norm_x2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        # Multi-scale feature extraction
        x1_features = torch.stack([conv(x1) for conv in self.multi_scale_conv])
        x2_features = torch.stack([conv(x2) for conv in self.multi_scale_conv])
        x1_features = x1_features.mean(0)
        x2_features = x2_features.mean(0)

        # Query-Key-Value projections
        x1_qkv = self.x1_qkv(x1_features)
        x2_qkv = self.x2_qkv(x2_features)

        x1_q, x1_k, x1_v = x1_qkv.chunk(3, dim=1)
        x2_q, x2_k, x2_v = x2_qkv.chunk(3, dim=1)

        # Multi-head attention preparation
        x1_q = rearrange(x1_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_q = rearrange(x2_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x1_k = rearrange(x1_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_k = rearrange(x2_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x1_v = rearrange(x1_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_v = rearrange(x2_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalized cross-attention
        q1 = F.normalize(x1_q, dim=-1)
        k2 = F.normalize(x2_k, dim=-1)
        q2 = F.normalize(x2_q, dim=-1)
        k1 = F.normalize(x1_k, dim=-1)

        # Attention computation
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature

        attn1 = self.dropout(attn1.softmax(dim=-1))
        attn2 = self.dropout(attn2.softmax(dim=-1))

        # Value aggregation
        x1_out = rearrange((attn1 @ x1_v), 'b head c (h w) -> b (head c) h w',
                           head=self.num_heads, h=h, w=w)
        x2_out = rearrange((attn2 @ x2_v), 'b head c (h w) -> b (head c) h w',
                           head=self.num_heads, h=h, w=w)

        # Channel-wise attention
        x1_channel_attn = self.channel_attention(x1_out)
        x2_channel_attn = self.channel_attention(x2_out)

        x1_out = x1_out * x1_channel_attn
        x2_out = x2_out * x2_channel_attn

        # Final projections with residual and layer norm
        x1_out = self.project_out_x1(x1_out)
        x2_out = self.project_out_x2(x2_out)

        # Layer normalization and residual connection
        x1_out = x1_out + x1
        x2_out = x2_out + x2

        return x1_out, x2_out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, x_n_channels, n_classes, bilinear=False, type = 'seg'):
        super(UNet, self).__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = (DoubleConv(n_channels, 32))
        self.inc_x_data = (DoubleConv(x_n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512 // factor))
        self.scfa0 = SCFA(dim=32, num_heads=1)
        self.scfa1 = SCFA(dim=64, num_heads=1)
        self.scfa2 = SCFA(dim=128, num_heads=1)
        self.scfa3 = SCFA(dim=256, num_heads=1)
        self.scfa4 = SCFA(dim=512 // factor, num_heads=1)

        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.scfa_up1 = SCFA(dim=256 // factor, num_heads=1)
        self.scfa_up2 = SCFA(dim=128 // factor, num_heads=1)
        self.scfa_up3 = SCFA(dim=64 // factor, num_heads=1)
        self.scfa_up4 = SCFA(dim=32, num_heads=1)

        self.outc = (OutConv(32, n_classes))
        self.nn1 = nn.Linear(32, n_classes, bias=True)

    def forward(self, x, x_data):
        x = x.squeeze(1)
        x_data = x_data.squeeze(1)
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=True)
        x_data = F.interpolate(x_data, size=(96, 96), mode='bilinear', align_corners=True)
        x = x + torch.randn_like(x) * 0.02  # Add noise for regularization
        x_data = x_data + torch.randn_like(x_data) * 0.02

        x1 = self.inc(x)
        x_data1 = self.inc_x_data(x_data)
        x1_c, x_data1_c = self.scfa0(F.relu(x1), F.relu(x_data1))

        x2 = self.down1(F.relu(x1) + F.relu(x1_c))
        x_data2 = self.down1(F.relu(x_data1) + F.relu(x_data1_c))
        x2_c, x_data2_c = self.scfa1(F.relu(x2), F.relu(x_data2))

        x3 = self.down2(F.relu(x2) + F.relu(x2_c))
        x_data3 = self.down2(F.relu(x_data2) + F.relu(x_data2_c))
        x3_c, x_data3_c = self.scfa2(F.relu(x3), F.relu(x_data3))

        x4 = self.down3(F.relu(x3) + F.relu(x3_c))
        x_data4 = self.down3(F.relu(x_data3) + F.relu(x_data3_c))
        x4_c, x_data4_c = self.scfa3(F.relu(x4), F.relu(x_data4))

        x5 = self.down4(F.relu(x4) + F.relu(x4_c))
        x_data5 = self.down4(F.relu(x_data4) + F.relu(x_data4_c))
        x5_c, x_data5_c = self.scfa4(F.relu(x5), F.relu(x_data5))

        x = self.up1(F.relu(x5) + F.relu(x5_c), F.relu(x4))
        x_data = self.up1(F.relu(x_data5) + F.relu(x_data5_c), F.relu(x_data4))
        x_c, x_data_c = self.scfa_up1(F.relu(x), F.relu(x_data))

        x = self.up2(F.relu(x) + F.relu(x_c), F.relu(x3))
        x_data = self.up2(F.relu(x_data) + F.relu(x_data_c), F.relu(x_data3))
        x_c, x_data_c = self.scfa_up2(F.relu(x), F.relu(x_data))

        x = self.up3(F.relu(x) + F.relu(x_c), F.relu(x2))
        x_data = self.up3(F.relu(x_data) + F.relu(x_data_c), F.relu(x_data2))
        x_c, x_data_c = self.scfa_up3(F.relu(x), F.relu(x_data))

        x = self.up4(F.relu(x) + F.relu(x_c), F.relu(x1))
        x_data = self.up4(F.relu(x_data) + F.relu(x_data_c), F.relu(x_data1))
        x, x_data = self.scfa_up4(F.relu(x), F.relu(x_data))

        final = F.relu(x) + F.relu(x_data)
        final = F.interpolate(final, size=(h, w), mode='bilinear', align_corners=True)
        out_cls = final[:, :, final.shape[2]//2, final.shape[3]//2]
        out_seg = self.outc(final)
        if self.type == 'seg':
            out_seg = out_seg
            out_cls = self.nn1(out_cls)
            return out_seg, out_cls
        elif self.type == 'cls':
            out_cls = self.nn1(out_cls)
            return out_cls
        else:
            raise ValueError('type must be either seg or cls')
