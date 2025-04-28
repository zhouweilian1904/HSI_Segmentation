import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import random
from other_models import FreqFusion

from torch.ao.nn.quantized.functional import threshold

from attentions.cbam import CBAM


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.scale = dim_head ** -0.5

        self.qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, groups=dim // 8, bias=False)  # Grouped convolution
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, kernel_size=1, groups=dim // 8, bias=False),  # Grouped convolution
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.qkv(x.transpose(1, 2))  # (B, D, N)
        q, k, v = qkv.chunk(3, dim=1)

        attn = torch.matmul(q.transpose(1, 2), k) * self.scale
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v.transpose(1, 2))
        out = out.transpose(1, 2)
        out = self.to_out(out)
        return rearrange(out, 'b d n -> b n d')


class RegionAggregator(nn.Module):
    """Region Aggregator to dynamically correlate regions."""

    def __init__(self, dim, num_regions, heads=8):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.query_projector = nn.Linear(dim, dim, bias=False)
        self.key_projector = nn.Linear(dim, dim, bias=False)
        self.value_projector = nn.Linear(dim, dim, bias=False)
        self.out_projector = nn.Linear(dim, dim)

    def forward(self, queries, regions):
        # queries: (batch, heads, length, dim)
        # regions: (batch, heads, length, regions, dim)
        b, h, l, d = queries.shape
        _, _, _, r, _ = regions.shape

        q = self.query_projector(queries)  # (batch, heads, length, dim)
        k = self.key_projector(regions)  # (batch, heads, length, regions, dim)
        v = self.value_projector(regions)  # (batch, heads, length, regions, dim)

        scores = torch.einsum('bhld,bhlrd->bhlr', q, k) * self.scale  # (batch, heads, length, regions)
        attention = scores.softmax(dim=-1)  # (batch, heads, length, regions)

        aggregated = torch.einsum('bhlr,bhlrd->bhld', attention, v)  # (batch, heads, length, dim)
        return self.out_projector(aggregated)


class FourRegionAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.region_dim = dim

        # Layers for projection
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.region_projector = nn.Linear(dim, self.region_dim, bias=True)  # Project regions to a fixed size
        self.region_aggregator = RegionAggregator(dim=dim, num_regions=4, heads=heads)  # Use RegionAggregator

        self.to_out = nn.Sequential(
            nn.Linear(heads * dim, dim),
            nn.Dropout(dropout)
        )

    def get_regions(self, x, h, w, H, W):
        """Extract four regions for position (h, w)"""
        b, N, dim = x.shape
        x = x.view(b, H, W, dim)  # Reshape into spatial dimensions

        regions = [
            x[:, :h + 1, :w + 1],  # top-left
            x[:, :h + 1, w:],  # top-right
            x[:, h:, :w + 1],  # bottom-left
            x[:, h:, w:],  # bottom-right
        ]

        # Project each region into a fixed-size vector
        projected_regions = []
        for region in regions:
            if region.numel() > 0:  # Avoid empty regions
                region = region.reshape(b, -1, dim)  # Flatten spatial dimensions
                region_pooled = torch.mean(region, dim=1)  # Pool over spatial dimensions
                projected_regions.append(region_pooled)

        return torch.stack(projected_regions, dim=-1)  # (batch, dim, regions)

    def forward(self, x):
        x = self.norm(x)
        b, n, dim = x.shape
        H, W = int(n ** 0.5), int(n ** 0.5)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Prepare region features for all positions
        region_features = []

        for pos in range(H * W):
            h, w = pos // W, pos % W

            # Extract regions for the current position
            region_vectors = self.get_regions(x, h, w, H, W)  # (batch, dim, regions)
            region_vectors = region_vectors.unsqueeze(1).repeat(1, self.heads, 1, 1)  # (batch, heads, dim, regions)
            region_features.append(region_vectors)

        # Stack all region features
        region_features = torch.stack(region_features, dim=2)  # (batch, heads, length, dim, regions)
        region_features = rearrange(region_features, 'b h l d r -> b h l r d')  # Align regions with `q`

        # Correlate query with region features using RegionAggregator
        out = self.region_aggregator(q,
                                     region_features)  # Pass (batch, heads, length, dim) and (batch, heads, length, regions, dim)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # FourRegionAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_residual = x
            x = attn(x)  # Ensure attention output matches input shape
            x = self.norm(x)
            assert x.shape == x_residual.shape, f"Shape mismatch: {x.shape} vs {x_residual.shape}"
            x = x + x_residual
            x = ff(x) + x
        return self.norm(x)


class General_ViT_2(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, heads, dropout=0.1, type='seg'):
        super().__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        self.image_size = image_size

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Dynamically set groups for grouped convolution
        groups = math.gcd(patch_dim, dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # Normalize patch dimensions
            nn.Linear(patch_dim, dim, bias=True),  # Map patch_dim to transformer dimension
            Rearrange('b n d -> b d n'),  # Prepare for Conv1d (channels-first)
            nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=True),  # Grouped convolution
            Rearrange('b d n -> b n d'),  # Restore transformer input shape
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.to_latent = nn.Identity()
        self.out_seg = nn.Conv2d(dim, dim, kernel_size=1, bias=True)  # Output for segmentation
        self.norm = nn.LayerNorm([dim, int(num_patches**0.5), int(num_patches**0.5)])

    def forward(self, img):
        # Apply patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add positional embeddings
        # x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer(x)

        # Rearrange to image format
        x_seg = rearrange(x, 'b (h w) d -> b d h w', h=int(n ** 0.5), w=int(n ** 0.5))
        x_seg = self.to_latent(x_seg)

        # Segmentation output
        out_seg = self.out_seg(x_seg)
        return out_seg


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, patch_size, depth, heads):
        super().__init__()
        self.process = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            General_ViT_2(channels=out_channels, dim=out_channels, image_size=image_size, patch_size=patch_size,
                          depth=depth, heads=heads),
        )

    def forward(self, x):
        return self.process(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x1_out, _ = self.attn1(x1, x2, x2)
        x2_out, _ = self.attn2(x2, x1, x1)
        return x1_out, x2_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size, depth, heads):
        super().__init__()
        self.init = DoubleConv(in_channels - out_channels, out_channels)
        self.conv = DoubleConv(out_channels * 2, out_channels)
        self.vit = General_ViT_2(channels=out_channels, dim=out_channels, image_size=image_size,
                                 patch_size=patch_size, depth=depth, heads=heads)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x_lr, x_hr):
        b, c, h, w = x_hr.shape
        x_lr = self.init(x_lr)
        x_lr = F.adaptive_max_pool2d(x_lr, output_size=(h, w))
        x = torch.cat([x_lr, x_hr], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        return self.vit(x)


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d(downSize),
            nn.Conv2d(nIn, nOut * 2, 1, bias=True),
            nn.BatchNorm2d(nOut * 2),
            nn.GELU(),
            nn.Conv2d(nOut * 2, nOut, 1, bias=True),
            nn.AdaptiveMaxPool2d(upSize),
            nn.BatchNorm2d(nOut),
            nn.GELU(),
        )

    def forward(self, x):
        return self.features(x)


class ViT_UNet(nn.Module):
    def __init__(self, channels=20, x_data_channel=1, num_classes=20, image_size=49, dim=32, depth=4, heads=1,
                 dropout=0.):
        super().__init__()
        self.inc = (DoubleConv(channels, dim * 1))
        self.inc_x_data = (DoubleConv(x_data_channel, dim * 1))
        self.dropout = nn.Dropout(dropout)
        self.image_size = image_size

        self.down1 = Down(in_channels=dim * 4, out_channels=dim, image_size=128, patch_size=16, depth=depth,
                          heads=heads)
        self.down2 = Down(in_channels=dim * 4, out_channels=dim, image_size=64, patch_size=8, depth=depth,
                          heads=heads)
        self.down3 = Down(in_channels=dim * 4, out_channels=dim, image_size=32, patch_size=4, depth=depth,
                          heads=heads)
        self.down4 = Down(in_channels=dim * 4, out_channels=dim, image_size=16, patch_size=2, depth=depth,
                          heads=heads)
        self.down5 = Down(in_channels=dim * 4, out_channels=dim, image_size=8, patch_size=2, depth=depth,
                          heads=heads)
        self.down6 = Down(in_channels=dim * 4, out_channels=dim, image_size=4, patch_size=1, depth=depth,
                          heads=heads)

        # PSP modules for multi-scale feature aggregation
        self.p1 = PSPDec(dim, dim, upSize=8, downSize=8 // 2)
        self.p1_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p2 = PSPDec(dim, dim, upSize=16, downSize=16 // 2)
        self.p2_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p3 = PSPDec(dim, dim, upSize=32, downSize=32 // 2)
        self.p3_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p4 = PSPDec(dim, dim, upSize=64, downSize=64 // 2)
        self.p4_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p5 = PSPDec(dim, dim, upSize=128, downSize=128 // 2)
        self.p5_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p6 = PSPDec(dim, dim, upSize=256, downSize=256 // 2)
        self.p6_prob = nn.Conv2d(dim, num_classes, kernel_size=1)

        self.up1 = Up(in_channels=dim + dim, out_channels=dim, image_size=4, patch_size=1, depth=depth,
                      heads=heads)
        self.up2 = Up(in_channels=dim + dim, out_channels=dim, image_size=8, patch_size=2, depth=depth,
                      heads=heads)
        self.up3 = Up(in_channels=dim + dim, out_channels=dim, image_size=16, patch_size=2, depth=depth,
                      heads=heads)
        self.up4 = Up(in_channels=dim + dim, out_channels=dim, image_size=32, patch_size=4, depth=depth,
                      heads=heads)
        self.up5 = Up(in_channels=dim + dim, out_channels=dim, image_size=64, patch_size=8, depth=depth,
                      heads=heads)
        self.up6 = Up(in_channels=dim + dim, out_channels=dim, image_size=128, patch_size=16, depth=depth,
                      heads=heads)

        self.out_seg = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        self.out_cls = nn.Linear(dim, num_classes, bias=True)
        self.out_rec = nn.Conv2d(dim, channels, kernel_size=1, bias=True)

    def forward(self, img, x_data):
        x = img.squeeze(1)
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x + torch.randn_like(x) * 0.02
        x1 = self.inc(x)
        # -------------------------------------------------------------
        # print(f'x1 {x1.shape}')
        x1_ = F.pixel_unshuffle(x1, 2)
        x2 = self.down1(x1_)
        x2 = F.adaptive_max_pool2d(x2, (128, 128))
        x2 = self.dropout(x2)

        # print(f'x2 {x2.shape}')
        x2_ = F.pixel_unshuffle(x2, 2)
        x3 = self.down2(x2_)
        x3 = F.adaptive_max_pool2d(x3, (64, 64))
        x3 = self.dropout(x3)

        # print(f'x3 {x3.shape}')
        x3_ = F.pixel_unshuffle(x3, 2)
        x4 = self.down3(x3_)
        x4 = F.adaptive_max_pool2d(x4, (32, 32))
        x4 = self.dropout(x4)

        # print(f'x4 {x4.shape}')
        x4_ = F.pixel_unshuffle(x4, 2)
        x5 = self.down4(x4_)
        x5 = F.adaptive_max_pool2d(x5, (16, 16))
        x5 = self.dropout(x5)

        # print(f'x5 {x5.shape}')
        x5_ = F.pixel_unshuffle(x5, 2)
        x6 = self.down5(x5_)  # BOTTLENECK
        x6 = F.adaptive_max_pool2d(x6, (8, 8))
        x6 = self.dropout(x6)

        # print(f'x6 {x6.shape}')
        x6_ = F.pixel_unshuffle(x6, 2)
        x7 = self.down6(x6_)  # BOTTLENECK
        x7 = self.dropout(x7)
        # print(f'x7 {x7.shape}')

        #-------------------------------------------------------------------------------------------
        # Forward pass with refactored logic
        p1 = self.p1(x7)
        p1_prob, p1 = apply_mask_and_threshold(p1, self.p1_prob,)
        # print(f'p1 {p1.shape} | p1_prob {p1_prob.shape}')

        p2 = self.p2(p1)
        p2_prob, p2 = apply_mask_and_threshold(p2, self.p2_prob,)
        # print(f'p2 {p2.shape} | p2_prob {p2_prob.shape}')

        p3 = self.p3(p2)
        p3_prob, p3 = apply_mask_and_threshold(p3, self.p3_prob,)
        # print(f'p3 {p3.shape} | p3_prob {p3_prob.shape}')

        p4 = self.p4(p3)
        p4_prob, p4 = apply_mask_and_threshold(p4, self.p4_prob,)
        # print(f'p4 {p4.shape} | p4_prob {p4_prob.shape}')

        p5 = self.p5(p4)
        p5_prob, p5 = apply_mask_and_threshold(p5, self.p5_prob,)
        # print(f'p5 {p5.shape} | p5_prob {p5_prob.shape}')

        p6 = self.p6(p5)
        p6_prob, p6 = apply_mask_and_threshold(p6, self.p6_prob,)
        # print(f'p6 {p6.shape} | p6_prob {p6_prob.shape}')

        #-----------------------------------------------------------------------------------

        x6_up = self.up1(x7, x6)
        # print(f'x6_up {x6_up.shape}')
        x6_up = F.interpolate(x6_up, size=(8, 8), mode='bilinear', align_corners=False)
        x6_up = x6_up * p1_prob.unsqueeze(1)
        x6_up = self.dropout(x6_up)
        # print(f'x6_up {x6_up.shape}')

        x5_up = self.up2(x6_up, x5)
        # print(f'x5_up {x5_up.shape}')
        x5_up = F.interpolate(x5_up, size=(16, 16), mode='bilinear', align_corners=False)
        x5_up = x5_up * p2_prob.unsqueeze(1)
        x5_up = self.dropout(x5_up)
        # print(f'x5_up {x5_up.shape}')

        x4_up = self.up3(x5_up, x4)
        # print(f'x4_up {x4_up.shape}')
        x4_up = F.interpolate(x4_up, size=(32, 32), mode='bilinear', align_corners=False)
        x4_up = x4_up * p3_prob.unsqueeze(1)
        x4_up = self.dropout(x4_up)
        # print(f'x4_up {x4_up.shape}')

        x3_up = self.up4(x4_up, x3)
        # print(f'x3_up {x3_up.shape}')
        x3_up = F.interpolate(x3_up, size=(64, 64), mode='bilinear', align_corners=False)
        x3_up = x3_up * p4_prob.unsqueeze(1)
        x3_up = self.dropout(x3_up)
        # print(f'x3_up {x3_up.shape}')

        x2_up = self.up5(x3_up, x2)
        # print(f'x2_up {x2_up.shape}')
        x2_up = F.interpolate(x2_up, size=(128, 128), mode='bilinear', align_corners=False)
        x2_up = x2_up * p5_prob.unsqueeze(1)
        x2_up = self.dropout(x2_up)
        # print(f'x2_up {x2_up.shape}')

        x1_up = self.up6(x2_up, x1)
        # print(f'x1_up {x1_up.shape}')
        x1_up = F.interpolate(x1_up, size=(256, 256), mode='bilinear', align_corners=False)
        x1_up = x1_up * p6_prob.unsqueeze(1)
        # x1_up = self.dropout(x1_up)
        # print(f'x1_up {x1_up.shape}')
        cls = x1_up[:, :, x1_up.shape[2] // 2, x1_up.shape[3] // 2]

        # b, c, h, w = x1_up.shape
        out_back = F.adaptive_avg_pool2d(x1_up, (h, w))
        out_seg = self.out_seg(out_back)
        out_cls = self.out_cls(cls)
        # out_rec = self.out_rec(out_back)

        return out_seg, out_cls


def apply_mask_and_threshold(x, prob_layer):
    # Project probabilities using the prob_layer and apply softmax
    prob = prob_layer(x)  # Expected shape: (batch_size, num_classes, height, width)
    prob = F.softmax(prob, dim=1)  # Softmax across class dimension

    # Get the maximum probability across classes
    prob_max, _ = torch.max(prob, dim=1)  # Shape: (batch_size, height, width)

    # Calculate mean and standard deviation for the probabilities
    mean_prob = torch.mean(prob_max)  # Mean across all elements
    std_prob = torch.std(prob_max)  # Std across all elements

    # Compute threshold dynamically
    threshold = mean_prob
    # print(f'threshold {threshold}')

    # Apply the mask: Set probabilities below the threshold to 0
    mask = prob_max < threshold
    prob_max = prob_max.clone()  # Clone to avoid modifying in-place
    prob_max[mask] = 0.0

    return prob_max, x

