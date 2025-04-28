#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the CNN models
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSampler(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut - nIn, 3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.act(output)
        return output


class BasicResidualBlock(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        self.c1 = CBR(nIn, nOut, 3, 1)
        self.c2 = CB(nOut, nOut, 3, 1)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        output = input + output
        # output = self.drop(output)
        output = self.act(output)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class CDilated1(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)
        self.br = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.br(output)


class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        # output = self.drop(output)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB1(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 3, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        # d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        # add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        # output = self.drop(output)
        output = self.act(output)
        return output


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize=48):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(downSize),
            nn.Conv2d(nIn, nOut, 1, bias=False),
            nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(nOut),
            nn.Upsample(size=upSize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)


class ResNetC1(nn.Module):
    '''
        Segmentation model with ESP as the encoding block.
        This is the same as in stage 1
    '''

    def __init__(self, in_channels, classes):
        super().__init__()
        self.level1 = CBR(in_channels, 16, 7, 2)  # 384 x 384

        self.p01 = PSPDec(16 + classes, classes, 160, 192)
        self.p02 = PSPDec(16 + classes, classes, 128, 192)
        self.p03 = PSPDec(16 + classes, classes, 96, 192)
        self.p04 = PSPDec(16 + classes, classes, 72, 192)

        self.class_0 = nn.Sequential(
            nn.Conv2d(16 + 5 * classes, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = DilatedParllelResidualBlockB1(128, 128)
        self.level2_1 = DilatedParllelResidualBlockB1(128, 128)  # 512 x 256

        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)  # 256 x 128

        self.level4_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_3 = DilatedParllelResidualBlockB1(256, 256, 0.3)  # 128 x 64

        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 4 * 128, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )
        # C(320, classes, 7, 1)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input1):
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)

        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)

        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))
        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1))

        output3 = self.upsample_3(output3)

        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23), self.p30(combine_up_23),
             self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        combine_up = torch.cat([output0, output23_hook], 1)

        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up), self.p03(combine_up), self.p04(combine_up)], 1))

        #        output3 = output2_0 + output3

        #        classifier = self.classifier(output3)
        classifier = self.upsample_1(output0_hook)

        return classifier


class ResNetC1_YNet(nn.Module):
    '''
    Jointly learns segmentation and classification with a simplified, fully-preserved structure.
    '''

    def __init__(self, H, W, in_channels, classes, diagClasses, segNetFile=None):
        super().__init__()

        self.H = H  # Output height
        self.W = W  # Output width

        # Level 1: Initial feature extraction layer
        self.level1 = CBR(in_channels, 16, 7, 2)

        # Level 2: Downsampling and residual layers
        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = DilatedParllelResidualBlockB1(128, 128)
        self.level2_1 = DilatedParllelResidualBlockB1(128, 128)

        # Batch normalization and ReLU
        self.br_2 = BR(256)

        # Level 3: More downsampling and residual layers
        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)

        # PSP modules for multi-scale feature aggregation
        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        # Final classifier for segmentation map generation
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 4 * 128, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        # Upsample layers for progressively restoring resolution
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Additional levels and layers for classification branch
        self.level4_0 = DownSamplerA(512, 128)
        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)

        self.br_con_4 = BR(256)

        self.level5_0 = DownSamplerA(256, 64)
        self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.br_con_5 = BR(128)

        # Global pooling and fully connected layers for classification output
        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, diagClasses)

        # Load pretrained weights if specified
        if segNetFile is not None:
            print('Loading pre-trained segmentation model')
            self.load_state_dict(torch.load(segNetFile), strict=False)

    def forward(self, input1, x_data=None):
        # Input pre-processing and resizing
        input1 = input1.squeeze(1)
        input1 = F.interpolate(input1, size=(384, 384), mode='bilinear', align_corners=False)
        # Level 1 forward pass
        enc1 = self.level1(input1)

        # Level 2 forward pass
        enc2 = self.level2(enc1)
        enc2_0 = self.level2_0(enc2)
        enc2_1 = self.level2_1(enc2_0)
        enc2_combined = self.br_2(torch.cat([enc2, enc2_1], dim=1))

        # Level 3 forward pass
        enc3_0 = self.level3_0(enc2_combined)
        enc3_1 = self.level3_1(enc3_0)
        enc3_2 = self.level3_2(enc3_1)
        enc3_combined = self.br_4(torch.cat([enc3_0, enc3_2], dim=1))

        # Apply PSP modules
        psp_out = torch.cat([self.p1(enc3_combined), self.p2(enc3_combined),
                             self.p3(enc3_combined), self.p4(enc3_combined), enc3_combined], dim=1)

        # Generate segmentation output
        seg_out = self.classifier(psp_out)

        seg_out = self.upsample_3(seg_out)
        seg_out = F.interpolate(seg_out, size=(self.H, self.W), mode='bilinear', align_corners=False)

        # Classification branch forward pass
        class_enc4_0 = self.level4_0(enc3_combined)
        class_enc4_1 = self.level4_1(class_enc4_0)
        class_enc4_2 = self.level4_2(class_enc4_1)
        class_out = self.br_con_4(torch.cat([class_enc4_0, class_enc4_2], dim=1))

        class_enc5_0 = self.level5_0(class_out)
        class_enc5_1 = self.level5_1(class_enc5_0)
        class_enc5_2 = self.level5_2(class_enc5_1)
        class_out_combined = self.br_con_5(torch.cat([class_enc5_0, class_enc5_2], dim=1))

        # Global pooling and fully connected layers for diagnostic classification
        class_out_avg = self.global_Avg(class_out_combined)
        class_out_flat = class_out_avg.view(class_out_avg.size(0), -1)
        diag_class = self.fc2(F.relu(self.fc1(class_out_flat)))

        return seg_out, diag_class




class ResNetD1(nn.Module):
    '''
    Segmentation model with BasicResidualBlock (RCB) as encoding blocks.
    '''

    def __init__(self, in_channels, classes):
        super().__init__()

        # Level 1: Initial feature extraction
        self.level1 = CBR(in_channels, 16, 7, 2)  # Input 3 channels (RGB), output 16 channels

        # PSP modules for multi-scale feature aggregation
        self.p01 = PSPDec(16 + classes, classes, 160, 192)
        self.p02 = PSPDec(16 + classes, classes, 128, 192)
        self.p03 = PSPDec(16 + classes, classes, 96, 192)
        self.p04 = PSPDec(16 + classes, classes, 72, 192)

        # Segmentation head for first level
        self.class_0 = nn.Sequential(
            nn.Conv2d(16 + 5 * classes, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        # Level 2: Additional downsampling and residual blocks
        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = BasicResidualBlock(128, 128)
        self.level2_1 = BasicResidualBlock(128, 128)

        # PSP modules for second level
        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        # Segmentation head for second level
        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        # Level 3: Additional downsampling and residual blocks
        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = BasicResidualBlock(256, 256, 0.3)
        self.level3_2 = BasicResidualBlock(256, 256, 0.3)

        # Level 4: More residual blocks for deeper features
        self.level4_1 = BasicResidualBlock(256, 256, 0.3)
        self.level4_2 = BasicResidualBlock(256, 256, 0.3)
        self.level4_3 = BasicResidualBlock(256, 256, 0.3)

        # PSP modules for third level
        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        # Final classifier for segmentation map generation
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128 * 4, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        # Upsample layers for progressive restoration of resolution
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input1):
        # Forward pass through each level
        output0 = self.level1(input1)
        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)
        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)
        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))

        # Apply PSP modules and classifier
        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1)
        )
        output3 = self.upsample_3(output3)

        # Second upsampling and feature aggregation
        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23),
             self.p30(combine_up_23), self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        # Final upsampling and output generation
        combine_up = torch.cat([output23_hook, output0], 1)
        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up),
             self.p03(combine_up), self.p04(combine_up)], 1))
        classifier = self.upsample_1(output0_hook)

        return classifier  # Segmentation output


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetD1_YNet(nn.Module):
    '''
    Joint segmentation and classification model with integrated layers for simplicity.
    '''

    def __init__(self, H, W, in_channels, classes, diagClasses, segNetFile=None):
        super().__init__()

        self.H = H  # Output height
        self.W = W  # Output width

        # Level 1: Initial feature extraction
        self.level1 = CBR(in_channels, 16, 7, 2)

        # PSP modules for multi-scale feature aggregation at Level 1
        self.p01 = PSPDec(16 + classes, classes, 160, 192)
        self.p02 = PSPDec(16 + classes, classes, 128, 192)
        self.p03 = PSPDec(16 + classes, classes, 96, 192)
        self.p04 = PSPDec(16 + classes, classes, 72, 192)

        # Segmentation head for Level 1
        self.class_0 = nn.Sequential(
            nn.Conv2d(16 + 5 * classes, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        # Level 2: Downsampling and residual blocks
        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = BasicResidualBlock(128, 128)
        self.level2_1 = BasicResidualBlock(128, 128)

        # PSP modules for Level 2
        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        # Segmentation head for Level 2
        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        # Level 3: Downsampling and residual blocks
        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = BasicResidualBlock(256, 256, 0.3)
        self.level3_2 = BasicResidualBlock(256, 256, 0.3)

        # Level 4: More residual blocks for deeper features (adjusted to 256 input and output channels)
        self.level4_1 = BasicResidualBlock(256, 256, 0.3)
        self.level4_2 = BasicResidualBlock(256, 256, 0.3)
        self.level4_3 = BasicResidualBlock(256, 256, 0.3)

        # PSP modules for Level 4
        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        # Final classifier for segmentation map generation
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128 * 4, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        # Upsample layers for progressive resolution restoration
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Classification branch additional layers
        self.level4_0 = DownSamplerA(512, 128)
        self.level4_1_class = BasicResidualBlock(128, 128, 0.3)
        self.level4_2_class = BasicResidualBlock(128, 128, 0.3)
        self.br_con_4 = BR(256)

        self.level5_0 = DownSamplerA(256, 64)
        self.level5_1 = BasicResidualBlock(64, 64, 0.3)
        self.level5_2 = BasicResidualBlock(64, 64, 0.3)
        self.br_con_5 = BR(128)

        # Global average pooling and classification fully connected layers
        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, diagClasses)

    def forward(self, input1, x_data=None):
        # Input pre-processing and resizing
        input1 = input1.squeeze(1)
        input1 = F.interpolate(input1, size=(384, 384), mode='bilinear', align_corners=False)

        # Forward pass through each level
        output0 = self.level1(input1)
        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)
        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)
        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))

        # Apply PSP modules and classifier
        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1)
        )
        output3 = self.upsample_3(output3)

        # Second upsampling and feature aggregation
        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23),
             self.p30(combine_up_23), self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        # Final upsampling and output generation
        combine_up = torch.cat([output23_hook, output0], 1)
        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up),
             self.p03(combine_up), self.p04(combine_up)], 1))
        classifier = self.upsample_1(output0_hook)

        # Classification branch processing
        enc_out = output3  # Use the output from the deepest level of segmentation
        enc_out = F.interpolate(enc_out, size=(self.H, self.W), mode='bilinear', align_corners=False)
        l5_0 = self.level4_0(enc_out)
        l5_1 = self.level4_1_class(l5_0)
        l5_2 = self.level4_2_class(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], dim=1))

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        l6_con = self.br_con_5(torch.cat([l6_0, l6_2], dim=1))

        # Global pooling and fully connected layers for diagnostic classification
        glbAvg = self.global_Avg(l6_con)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        diag_class = self.fc2(F.relu(self.fc1(flatten)))

        return classifier, diag_class  # Returns both segmentation and classification output


