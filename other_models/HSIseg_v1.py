import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from attentions.cbam import CBAM
from attentions.se_module import SELayer

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


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) module with a learnable class token.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))

        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.class_token, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features of shape (batch_size, num_tokens, channels)
            mask: Optional attention mask of shape (num_windows, tokens, tokens)
        """
        B, N, C = x.shape

        # Expand the class token to batch size and concatenate
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention calculation
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N + 1, N + 1) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N + 1, N + 1)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Separate the class token from the rest of the output
        cls_token_output, tokens_output = x[:, 0], x[:, 1:]

        return cls_token_output, tokens_output



# class RegionAggregator(nn.Module):
#     """Region Aggregator to dynamically correlate regions."""
#
#     def __init__(self, dim, num_regions, heads=8):
#         super().__init__()
#         self.scale = (dim // heads) ** -0.5
#         self.query_projector = nn.Linear(dim, dim, bias=False)
#         self.key_projector = nn.Linear(dim, dim, bias=False)
#         self.value_projector = nn.Linear(dim, dim, bias=False)
#         self.out_projector = nn.Linear(dim, dim)
#
#     def forward(self, queries, regions):
#         # queries: (batch, heads, length, dim)
#         # regions: (batch, heads, length, regions, dim)
#         b, h, l, d = queries.shape
#         _, _, _, r, _ = regions.shape
#
#         q = self.query_projector(queries)  # (batch, heads, length, dim)
#         k = self.key_projector(regions)  # (batch, heads, length, regions, dim)
#         v = self.value_projector(regions)  # (batch, heads, length, regions, dim)
#
#         scores = torch.einsum('bhld,bhlrd->bhlr', q, k) * self.scale  # (batch, heads, length, regions)
#         attention = scores.softmax(dim=-1)  # (batch, heads, length, regions)
#
#         aggregated = torch.einsum('bhlr,bhlrd->bhld', attention, v)  # (batch, heads, length, dim)
#         return self.out_projector(aggregated)
#
#
# class FourRegionAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.region_dim = dim
#
#         # Layers for projection
#         self.norm = nn.LayerNorm(dim)
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
#         self.region_projector = nn.Linear(dim, self.region_dim, bias=True)  # Project regions to a fixed size
#         self.region_aggregator = RegionAggregator(dim=dim, num_regions=4, heads=heads)  # Use RegionAggregator
#
#         self.to_out = nn.Sequential(
#             nn.Linear(heads * dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def get_regions(self, x, h, w, H, W):
#         """Extract four regions for position (h, w)"""
#         b, N, dim = x.shape
#         x = x.view(b, H, W, dim)  # Reshape into spatial dimensions
#
#         regions = [
#             x[:, :h + 1, :w + 1],  # top-left
#             x[:, :h + 1, w:],  # top-right
#             x[:, h:, :w + 1],  # bottom-left
#             x[:, h:, w:],  # bottom-right
#         ]
#
#         # Project each region into a fixed-size vector
#         projected_regions = []
#         for region in regions:
#             if region.numel() > 0:  # Avoid empty regions
#                 region = region.reshape(b, -1, dim)  # Flatten spatial dimensions
#                 region_pooled = torch.mean(region, dim=1)  # Pool over spatial dimensions
#                 projected_regions.append(region_pooled)
#
#         return torch.stack(projected_regions, dim=-1)  # (batch, dim, regions)
#
#     def forward(self, x):
#         x = self.norm(x)
#         b, n, dim = x.shape
#         H, W = int(n ** 0.5), int(n ** 0.5)
#
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#
#         # Prepare region features for all positions
#         region_features = []
#
#         for pos in range(H * W):
#             h, w = pos // W, pos % W
#
#             # Extract regions for the current position
#             region_vectors = self.get_regions(x, h, w, H, W)  # (batch, dim, regions)
#             region_vectors = region_vectors.unsqueeze(1).repeat(1, self.heads, 1, 1)  # (batch, heads, dim, regions)
#             region_features.append(region_vectors)
#
#         # Stack all region features
#         region_features = torch.stack(region_features, dim=2)  # (batch, heads, length, dim, regions)
#         region_features = rearrange(region_features, 'b h l d r -> b h l r d')  # Align regions with `q`
#
#         # Correlate query with region features using RegionAggregator
#         out = self.region_aggregator(q,
#                                      region_features)  # Pass (batch, heads, length, dim) and (batch, heads, length, regions, dim)
#
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class RegionAggregator(nn.Module):
    """Region Aggregator to dynamically correlate regions."""

    def __init__(self, dim, num_regions, heads=8):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.query_projector = nn.Linear(dim, dim, bias=False)
        # self.query_projector = FeedForward(dim, dim * 2, dim, dropout=0.)
        self.key_projector = nn.Linear(dim, dim, bias=False)
        # self.key_projector = FeedForward(dim, dim * 2, dim, dropout=0.)
        self.value_projector = nn.Linear(dim, dim, bias=False)
        # self.value_projector = FeedForward(dim, dim * 2, dim, dropout=0.)
        self.out_projector = nn.Linear(dim, dim)
        # self.out_projector = FeedForward(dim, dim * 2, dim, dropout=0.)

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
        # self.to_qkv = FeedForward(dim, inner_dim * 3 * 2, inner_dim * 3, dropout=0.)
        self.DWA = WindowAttention(dim=dim, num_heads=heads)

        self.region_aggregator = RegionAggregator(dim=dim, num_regions=4, heads=heads)  # Use RegionAggregator

        self.to_out = nn.Sequential(
            nn.Linear(heads * dim, dim),
            # FeedForward(heads * dim, dim * 2, dim, dropout=0.),
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

                region_pooled = self.DWA(region)[0]

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
        region_features = rearrange(region_features, 'b h l d r -> b h l r d')  # Align regions with q

        # Correlate query with region features using RegionAggregator
        out = self.region_aggregator(q,
                                     region_features)  # Pass (batch, heads, length, dim) and (batch, heads, length, regions, dim)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# class Deformable_Window_Attention(nn.Module):
#     def __init__(self, dim=64, depth=1, heads=8, mlp_dim=64,
#                  pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         self.to_patch_embedding = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim),
#             # FeedForward(dim, dim * 2, dim, dropout=0.),
#             nn.LayerNorm(dim),
#         )
#         self.pos_embedding = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer2(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Linear(dim, dim)
#         # self.mlp_head = FeedForward(dim, dim * 2, dim, dropout=0.)
#
#     def forward(self, seq):
#         x = self.to_patch_embedding(seq)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embedding(rearrange(x, 'b n d -> b d n')).permute(0, 2, 1)
#         x = self.dropout(x)
#
#         x = self.transformer(x)
#
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)


class General_ViT_2(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, heads, dropout=0.1, type='seg'):
        super().__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')

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

    def forward(self, img):
        # Apply patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add positional embeddings
        x += self.pos_embedding[:, :n]
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
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            # nn.ReLU(),
            General_ViT_2(channels=out_channels, dim=out_channels, image_size=image_size, patch_size=patch_size,
                          depth=depth, heads=heads),
            SELayer(out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, skip):
        x = self.norm(x)
        skip = self.norm(skip)
        x, _ = self.attn(x, skip, skip)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size, depth, heads):
        super().__init__()
        self.init = DoubleConv(in_channels - out_channels, out_channels)
        self.conv = DoubleConv(out_channels * 2, out_channels)
        self.vit = General_ViT_2(channels=out_channels, dim=out_channels, image_size=image_size,
                                 patch_size=patch_size, depth=depth, heads=heads)
        self.norm = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x_lr, x_hr):
        b, c, h, w = x_hr.shape
        x_lr = self.init(x_lr)
        x_lr = F.adaptive_max_pool2d(x_lr, output_size=(h, w))
        x = torch.cat([x_lr, x_hr], dim=1)
        x = self.conv(x)
        x = self.se(x)
        x = self.norm(x)
        return self.vit(x)


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d(downSize),
            nn.Conv2d(nIn, nOut * 2, 3, bias=True),
            nn.BatchNorm2d(nOut * 2),
            nn.GELU(),
            nn.ConvTranspose2d(nOut * 2, nOut, 3, bias=True),
            nn.AdaptiveMaxPool2d(upSize),
            nn.BatchNorm2d(nOut),
            nn.GELU(),
        )

    def forward(self, x):
        return self.features(x)


class ViT_UNet(nn.Module):
    def __init__(self, channels, x_data_channel=1, num_classes=20, image_size=49, dim=64, depth=1, heads=2,
                 dropout=0.1):
        super().__init__()
        self.inc = (DoubleConv(channels, dim))
        self.inc_data = (DoubleConv(x_data_channel, dim))
        self.dropout = nn.Dropout(dropout)
        self.inital_patch_size = image_size ** 0.5

        self.down1 = Down(in_channels=dim, out_channels=dim, image_size=49, patch_size=7, depth=depth, heads=heads)
        self.down2 = Down(in_channels=dim, out_channels=dim, image_size=36, patch_size=6, depth=depth, heads=heads)
        self.down3 = Down(in_channels=dim, out_channels=dim, image_size=25, patch_size=5, depth=depth, heads=heads)
        self.down4 = Down(in_channels=dim, out_channels=dim, image_size=16, patch_size=4, depth=depth, heads=heads)
        self.down5 = Down(in_channels=dim, out_channels=dim, image_size=9, patch_size=3, depth=depth, heads=heads)

        self.up1 = Up(in_channels=dim + dim, out_channels=dim, image_size=9, patch_size=3, depth=depth, heads=heads)
        self.up2 = Up(in_channels=dim + dim, out_channels=dim, image_size=16, patch_size=4, depth=depth, heads=heads)
        self.up3 = Up(in_channels=dim + dim, out_channels=dim, image_size=25, patch_size=5, depth=depth, heads=heads)
        self.up4 = Up(in_channels=dim + dim, out_channels=dim, image_size=36, patch_size=6, depth=depth, heads=heads)
        self.up5 = Up(in_channels=dim + dim, out_channels=dim, image_size=49, patch_size=7, depth=depth, heads=heads)

        # PSP modules for multi-scale feature aggregation
        self.p1 = PSPDec(dim, dim, upSize=9, downSize=9 // 2)
        self.p1_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p2 = PSPDec(dim, dim, upSize=16, downSize=16 // 2)
        self.p2_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p3 = PSPDec(dim, dim, upSize=25, downSize=25 // 2)
        self.p3_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p4 = PSPDec(dim, dim, upSize=36, downSize=36 // 2)
        self.p4_prob = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.p5 = PSPDec(dim, dim, upSize=49, downSize=49 // 2)
        self.p5_prob = nn.Conv2d(dim, num_classes, kernel_size=1)

        self.out_seg = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        self.out_cls = nn.Linear(dim, num_classes, bias=True)
        self.out_rec = nn.Conv2d(dim, channels, kernel_size=1, bias=True)

    def forward(self, img, x_data=None):
        x = img.squeeze(1)
        x_data = x_data.squeeze(1)
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(49, 49), mode='bilinear', align_corners=False)
        x_data = F.interpolate(x_data, size=(49, 49), mode='bilinear', align_corners=False)
        x = x + torch.randn_like(x) * 0.02
        x_data = x_data + torch.randn_like(x_data) * 0.02
        x = self.dropout(x)
        x_data = self.dropout(x_data)

        x1 = self.inc(x)
        x_data1 = self.inc_data(x_data)

        x2 = self.down1(x1 + x_data1)
        x2 = F.adaptive_max_pool2d(x2, (36, 36))
        x2 = self.dropout(x2)

        x3 = self.down2(x2)
        x3 = F.adaptive_max_pool2d(x3, (25, 25))
        x3 = self.dropout(x3)

        x4 = self.down3(x3)
        x4 = F.adaptive_max_pool2d(x4, (16, 16))
        x4 = self.dropout(x4)

        x5 = self.down4(x4)
        x5 = F.adaptive_max_pool2d(x5, (9, 9))
        x5 = self.dropout(x5)

        x6 = self.down5(x5)  # BOTTLENECK
        x6 = F.adaptive_max_pool2d(x6, (4, 4))
        x6 = self.dropout(x6)
        #-------------------------------------------------------------------
        p1 = self.p1(x6)
        p1_prob, p1 = apply_mask_and_threshold(p1, self.p1_prob, )

        p2 = self.p2(p1)
        p2_prob, p2 = apply_mask_and_threshold(p2, self.p2_prob, )

        p3 = self.p3(p2)
        p3_prob, p3 = apply_mask_and_threshold(p3, self.p3_prob, )

        p4 = self.p4(p3)
        p4_prob, p4 = apply_mask_and_threshold(p4, self.p4_prob, )

        p5 = self.p5(p4)
        p5_prob, p5 = apply_mask_and_threshold(p5, self.p5_prob, )

        #---------------------------------------------------------
        x = self.up1(x6, x5)
        x = self.dropout(x)
        x = F.relu(F.adaptive_max_pool2d(x, (9, 9))) * p1_prob.unsqueeze(1)

        x = self.up2(x, x4)
        x = self.dropout(x)
        x = F.relu(F.adaptive_max_pool2d(x, (16, 16))) * p2_prob.unsqueeze(1)

        x = self.up3(x, x3)
        x = self.dropout(x)
        x = F.relu(F.adaptive_max_pool2d(x, (25, 25))) * p3_prob.unsqueeze(1)

        x = self.up4(x, x2)
        x = self.dropout(x)
        x = F.relu(F.adaptive_max_pool2d(x, (36, 36))) * p4_prob.unsqueeze(1)

        x = self.up5(x, x1)
        x = self.dropout(x)
        x = F.relu(F.adaptive_max_pool2d(x, (49, 49))) * p5_prob.unsqueeze(1)

        out_cls = self.out_cls(x[:, :, x.shape[2] // 2, x.shape[3] // 2])
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        out_seg = self.out_seg(x)
        out_rec = self.out_rec(x)

        return out_seg, out_cls, p5, out_rec


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
    threshold = mean_prob + std_prob
    if threshold > 0.8:
        threshold = 0.8
    # print(f'threshold {threshold}')

    # Apply the mask: Set probabilities below the threshold to 0
    mask = prob_max < threshold
    prob_max = prob_max.clone()  # Clone to avoid modifying in-place
    # position = prob_max.clone()  # Clone to avoid modifying in-place
    prob_max[mask] = 0.0
    # plt.imshow(prob_max[0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()

    # position[mask] = 0.0
    # plt.imshow(position[0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()

    # position[~mask] = 1.0
    # plt.imshow(position[0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()
    return prob_max, x
