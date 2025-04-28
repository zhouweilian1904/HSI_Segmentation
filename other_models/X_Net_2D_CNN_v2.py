""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
import math
from attentions.cbam import CBAM


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.kernel_size = 3
        self.padding = (self.kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=self.padding, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / fan_out))

    def get_attention(self, input_tensor):
        input_mean = input_tensor.mean(dim=3)
        input_transposed = input_mean.transpose(1, 3)
        conv_result = self.conv(input_transposed.squeeze(-1).transpose(-1, -2))
        conv_transposed = conv_result.transpose(-1, -2).unsqueeze(-1).transpose(1, 3)
        attention_map = torch.mean(input_tensor * conv_transposed.unsqueeze(3), dim=-1)
        attention_normalized = F.softmax(attention_map / 0.025, dim=-1) + 1
        return attention_normalized

    def forward(self, feature_map_1, feature_map_2):
        batch_size, num_channels_1, num_features, height, width = feature_map_1.size()
        num_channels_2 = feature_map_2.size(1)

        feature_map_1 = feature_map_1.view(batch_size, num_channels_1, num_features, -1)
        feature_map_2 = feature_map_2.view(batch_size, num_channels_2, num_features, -1)

        feature_map_1_normalized = F.normalize(feature_map_1, p=2, dim=2, eps=1e-12)
        feature_map_2_normalized = F.normalize(feature_map_2, p=2, dim=2, eps=1e-12)

        f1_transposed = feature_map_1_normalized.transpose(2, 3).unsqueeze(2)
        f2_expanded = feature_map_2_normalized.unsqueeze(1)

        attention_matrix_1 = torch.matmul(f1_transposed, f2_expanded)
        attention_matrix_2 = attention_matrix_1.transpose(3, 4)

        attention_map_1 = self.get_attention(attention_matrix_1)
        attention_map_2 = self.get_attention(attention_matrix_2)

        enhanced_feature_map_1 = feature_map_1 * attention_map_1
        enhanced_feature_map_2 = feature_map_2 * attention_map_2

        output_map_1 = enhanced_feature_map_1.view(batch_size, num_features, height, width)
        output_map_2 = enhanced_feature_map_2.view(batch_size, num_features, height, width)

        return output_map_1, output_map_2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.se_attn = CBAM(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv_block(x)
        x = self.se_attn(x)
        x += residual  # Residual connection
        return nn.ReLU(inplace=True)(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.se_attn = CBAM(in_channels)

    def forward(self, x):
        return self.conv(self.se_attn(x))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=1)
            self.conv = DoubleConv((in_channels // 2) * 3, out_channels)
        self.se_attn = CBAM(out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x2 = self.up(x2)
        x = torch.cat([x3, x2, x1], dim=1)
        return self.se_attn(self.conv(x))



class UNet(nn.Module):
    def __init__(self, n_channels, x_n_channels, n_classes, dim=16, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc_1 = (DoubleConv(n_channels, dim))
        self.inc_2 = (DoubleConv(x_n_channels, dim))

        self.down1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, 1),
            CBAM(dim * 2))
        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, 3, 1),
            CBAM(dim * 4))
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, 3, 1),
            CBAM(dim * 8))
        self.down4 = nn.Sequential(
            nn.Conv2d(dim * 16, dim * 16, 3, 1),
            CBAM(dim * 16))

        self.down1_aux = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, 1),
            CBAM(dim * 2))
        self.down2_aux = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, 3, 1),
            CBAM(dim * 4))
        self.down3_aux = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, 3, 1),
            CBAM(dim * 8))
        self.down4_aux = nn.Sequential(
            nn.Conv2d(dim * 16, dim * 16, 3, 1),
            CBAM(dim * 16))

        self.cam = CAM()

        factor = 2 if bilinear else 1

        self.up1 = (Up(dim * 16, dim * 8 // factor, bilinear))
        self.up2 = (Up(dim * 8, dim * 4 // factor, bilinear))
        self.up3 = (Up(dim * 4, dim * 2 // factor, bilinear))
        self.up4 = (Up(dim * 2, dim, bilinear))

        self.outc_img = (OutConv(dim, n_classes))
        self.outc_aux = (OutConv(dim, n_classes))
        self.aux_rec = (OutConv(dim, n_channels))
        self.out_rec = OutConv(dim, n_channels)

        self.mlp_head = nn.Sequential(
            nn.AdaptiveMaxPool1d(dim),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_classes)
        )
        self.proj_norm = nn.BatchNorm2d(dim * 16)
        self.fc1 = nn.Linear(dim * 16, dim * 16)

    def forward(self, x, x_data):
        b, f, c, h, w = x.shape
        x1 = self.inc_1(x.squeeze(1))
        x_data1 = self.inc_2(x_data.squeeze(1))

        x_fea_1, x_data_fea_1 = self.cam(x1.unsqueeze(1), x_data1.unsqueeze(1))

        x2 = self.down1(torch.cat([F.relu(x1), F.relu(x_fea_1)], dim=1))
        x_data2 = self.down1_aux(torch.cat([F.relu(x_data1), F.relu(x_data_fea_1)], dim=1))

        x_fea_2, x_data_fea_2 = self.cam(x2.unsqueeze(1), x_data2.unsqueeze(1))

        x3 = self.down2(torch.cat([F.relu(x2), F.relu(x_fea_2)], dim=1))
        x_data3 = self.down2_aux(torch.cat([F.relu(x_data2), F.relu(x_data_fea_2)], dim=1))

        x_fea_3, x_data_fea_3 = self.cam(x3.unsqueeze(1), x_data3.unsqueeze(1))

        x4 = self.down3(torch.cat([F.relu(x3), F.relu(x_fea_3)], dim=1))
        x_data4 = self.down3_aux(torch.cat([F.relu(x_data3), F.relu(x_data_fea_3)], dim=1))

        x_fea_4, x_data_fea_4 = self.cam(x4.unsqueeze(1), x_data4.unsqueeze(1))

        x5 = self.down4(torch.cat([F.relu(x4), F.relu(x_fea_4)], dim=1))
        x_data5 = self.down4_aux(torch.cat([F.relu(x_data4), F.relu(x_data_fea_4)], dim=1))

        # ----------------------------------Bottleneck--------------------------------------------

        x_fea_5, x_data_fea_5 = self.cam(x5.unsqueeze(1), x_data5.unsqueeze(1))
        combine = F.relu(self.proj_norm((x_fea_5 + x_data_fea_5)))

        # ------------------------------------UP-------------------------------------------------------
        x4_ = self.up1(F.relu(x_fea_5), F.relu(x5), F.relu(x4))
        x_data4_ = self.up1(F.relu(x_data_fea_5), F.relu(x_data5), F.relu(x_data4))

        x_fea_4_, x_data_fea_4_ = self.cam(x_fea_4.unsqueeze(1), x_data4_.unsqueeze(1))

        x3_ = self.up2(F.relu(x4_), F.relu(x_fea_4_), F.relu(x3))
        x_data3_ = self.up2(F.relu(x_data_fea_4_), F.relu(x_data4_), F.relu(x_data3))

        x_fea_3_, x_data_fea_3_ = self.cam(x_fea_3.unsqueeze(1), x_data3_.unsqueeze(1))

        x2_ = self.up3(F.relu(x3_), F.relu(x_fea_3_), F.relu(x2))
        x_data2_ = self.up3(F.relu(x_data_fea_3_), F.relu(x_data3_), F.relu(x_data2))

        x_fea_2_, x_data_fea_2_ = self.cam(x_fea_2.unsqueeze(1), x_data2_.unsqueeze(1))

        x1_ = self.up4(F.relu(x2_), F.relu(x_fea_2_), F.relu(x1))
        x_data1_ = self.up4(F.relu(x_data_fea_2_), F.relu(x_data2_), F.relu(x_data1))

        x_fea_1_, x_data_fea_1_ = self.cam(x_fea_1.unsqueeze(1), x_data1_.unsqueeze(1))

        # Auxiliary MLP output
        # x_global = combine.mean(dim=[2, 3])
        x_center = combine[:, :, combine.shape[2] // 2, combine.shape[3] // 2]
        out_cls = self.mlp_head(x_center)

        out_seg = self.outc_img(F.relu(x1_) + F.relu(x_fea_1_))

        out_rec = self.out_rec(F.relu(x1_) + F.relu(x_fea_1_))

        aux_seg = self.outc_aux(F.relu(x_data1_) + F.relu(x_data_fea_1_))

        aux_rec = self.aux_rec(F.relu(x_data1_) + F.relu(x_data_fea_1_))

        return out_seg, out_cls