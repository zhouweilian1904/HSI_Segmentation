""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
import math
from attentions.cbam import CBAM


class CrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = in_channels // num_heads
        self.scale = self.d_k ** -0.5
        self.qkv_conv = nn.Conv2d(in_channels, num_heads * self.d_k * 3, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(num_heads * self.d_k, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, img1, img2):
        # Compute QKV for both inputs
        Q1, K1, V1 = self.extract_qkv(self.qkv_conv(img1))
        Q2, K2, V2 = self.extract_qkv(self.qkv_conv(img2))

        # Cross attention
        attn_output1 = self.attention(Q1, K2, V2, img1)
        attn_output2 = self.attention(Q2, K1, V1, img2)

        return attn_output1, attn_output2

    def extract_qkv(self, qkv):
        batch_size, _, height, width = qkv.size()
        qkv = qkv.view(batch_size, self.num_heads, 3 * self.d_k, height * width).permute(0, 1, 3, 2)
        Q, K, V = qkv.split(self.d_k, dim=-1)
        return Q, K, V

    def attention(self, Q, K, V, img):
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V).permute(0, 1, 3, 2).contiguous()
        attn_output = attn_output.view(img.size())
        return self.norm(self.dropout(self.out_conv(attn_output)) + img)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=5, stride=1)
            self.conv = DoubleConv((in_channels // 2) * 3, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x2 = self.up(x2)
        x = torch.cat([x3, x2, x1], dim=1)
        return self.norm(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, x_n_channels, n_classes, base_dim=16, bilinear=False):
        super(UNet, self).__init__()
        self.inc_img = DoubleConv(n_channels, base_dim)
        self.inc_aux = DoubleConv(x_n_channels, base_dim)

        self.down1 = nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=5, stride=1)
        self.down2 = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=5, stride=1)
        self.down3 = nn.Conv2d(base_dim * 8, base_dim * 8, kernel_size=5, stride=1)
        self.down4 = nn.Conv2d(base_dim * 16, base_dim * 16, kernel_size=5, stride=1)

        self.crossattn1 = CrossAttention(base_dim, num_heads=2)
        self.crossattn2 = CrossAttention(base_dim * 2, num_heads=2)
        self.crossattn3 = CrossAttention(base_dim * 4, num_heads=2)
        self.crossattn4 = CrossAttention(base_dim * 8, num_heads=2)
        self.crossattn5 = CrossAttention(base_dim * 16, num_heads=2)

        self.up1 = Up(base_dim * 16, base_dim * 8, bilinear)
        self.up2 = Up(base_dim * 8, base_dim * 4, bilinear)
        self.up3 = Up(base_dim * 4, base_dim * 2, bilinear)
        self.up4 = Up(base_dim * 2, base_dim, bilinear)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.outc_img = OutConv(base_dim, n_classes)
        self.outc_aux = OutConv(base_dim, n_classes)
        self.mlp_head = nn.Sequential(
            nn.AdaptiveMaxPool1d(base_dim),
            nn.Linear(base_dim, base_dim // 2),
            nn.LayerNorm(base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, n_classes)
        )

    def forward(self, x, x_data):
        b, f, c, h, w = x.shape
        x1 = self.inc_img(x.squeeze(1))
        x_data1 = self.inc_aux(x_data.squeeze(1))

        x_fea_1, x_data_fea_1 = self.crossattn1(x1, x_data1)

        x2 = self.down1(torch.cat([F.relu(x1), F.relu(x_fea_1)], dim=1))
        x_data2 = self.down1(torch.cat([F.relu(x_data1), F.relu(x_data_fea_1)], dim=1))

        x_fea_2, x_data_fea_2 = self.crossattn2(x2, x_data2)

        x3 = self.down2(torch.cat([F.relu(x2), F.relu(x_fea_2)], dim=1))
        x_data3 = self.down2(torch.cat([F.relu(x_data2), F.relu(x_data_fea_2)], dim=1))

        x_fea_3, x_data_fea_3 = self.crossattn3(x3, x_data3)

        x4 = self.down3(torch.cat([F.relu(x3), F.relu(x_fea_3)], dim=1))
        x_data4 = self.down3(torch.cat([F.relu(x_data3), F.relu(x_data_fea_3)], dim=1))

        x_fea_4, x_data_fea_4 = self.crossattn4(x4, x_data4)

        x5 = self.down4(torch.cat([F.relu(x4), F.relu(x_fea_4)], dim=1))
        x_data5 = self.down4(torch.cat([F.relu(x_data4), F.relu(x_data_fea_4)], dim=1))

        #----------------------------------Bottleneck--------------------------------------------

        # x_fea_5, x_data_fea_5 = self.crossattn5(x5, x_data5)
        combine = F.relu(x5) + F.relu(x_data5)

        #------------------------------------UP-------------------------------------------------------
        x4_ = self.up1(F.relu(combine), F.relu(x5), F.relu(x4))
        x_data4_ = self.up1(F.relu(combine), F.relu(x_data5), F.relu(x_data4))

        x_fea_4_, x_data_fea_4_ = self.crossattn4(x_fea_4, x_data4_)

        x3_ = self.up2(F.relu(x4_), F.relu(x_fea_4_), F.relu(x3) )
        x_data3_ = self.up2(F.relu(x_data_fea_4_), F.relu(x_data4_), F.relu(x_data3))

        x_fea_3_, x_data_fea_3_ = self.crossattn3(x_fea_3, x_data3_)

        x2_ = self.up3(F.relu(x3_), F.relu(x_fea_3_), F.relu(x2))
        x_data2_ = self.up3(F.relu(x_data_fea_3_), F.relu(x_data3_), F.relu(x_data2))

        x_fea_2_, x_data_fea_2_ = self.crossattn2(x_fea_2, x_data2_)

        x1_ = self.up4(F.relu(x2_), F.relu(x_fea_2_), F.relu(x1))
        x_data1_ = self.up4(F.relu(x_data_fea_2_), F.relu(x_data2_), F.relu(x_data1))

        x_fea_1_, x_data_fea_1_ = self.crossattn1(x_fea_1, x_data1_)

        # Auxiliary MLP output
        # x_global = combine.mean(dim=[2, 3])
        # x_center = combine[:, :, combine.shape[2] // 2, combine.shape[3] // 2]
        # out_cls = self.mlp_head(x_center)

        out_seg = self.outc_img(F.relu(x1_) + F.relu(x_fea_1_))
        out_cls = out_seg[:, :, out_seg.shape[2] // 2, out_seg.shape[3] // 2]

        # aux_seg = self.outc_aux(F.relu(x_data1_) + F.relu(x_data_fea_1_))

        # aux_rec = self.aux_rec(F.relu(x_data1_) + F.relu(x_data_fea_1_))

        return out_seg, out_cls
