import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FourRegionAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def get_regions(self, x, h, w, H, W):
        """Get four regions for position (h, w)"""
        # Original sequence shape: (b, H*W, dim)
        b, N, dim = x.shape
        x = x.view(b, H, W, dim)  # Reshape to get spatial dimensions

        # Define the four regions
        region1 = x[:, :h + 1, :w + 1]  # top-left
        region2 = x[:, :h + 1, w:]  # top-right
        region3 = x[:, h:, :w + 1]  # bottom-left
        region4 = x[:, h:, w:]  # bottom-right

        # Reshape regions back to sequence form
        region1 = region1.reshape(b, -1, dim)
        region2 = region2.reshape(b, -1, dim)
        region3 = region3.reshape(b, -1, dim)
        region4 = region4.reshape(b, -1, dim)
        # print(f'region1 {region1.shape} | region2 {region2.shape} | region3 {region3.shape} | region4 {region4.shape}')
        return [region1, region2, region3, region4]

    def forward(self, x):
        # x shape: (b, n, dim)
        x = self.norm(x)
        b, n, dim = x.shape
        H, W = int(n ** 0.5), int(n ** 0.5)

        # Get QKV for all positions
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Initialize output tensor
        out = torch.zeros_like(q)

        # Process each position
        for pos in range(H * W):
            h, w = pos // W, pos % W

            # Get regions for current position
            regions = self.get_regions(x, h, w, H, W)

            # Process each region
            region_outputs = []
            curr_q = q[:, :, pos:pos + 1]  # Query for current position

            for region in regions:
                if region.size(1) == 0:  # Skip empty regions
                    continue

                # Get k, v for current region
                region_kv = self.to_qkv(region).chunk(3, dim=-1)[1:]  # Only need k, v
                region_k, region_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), region_kv)

                # Compute attention
                dots = torch.matmul(curr_q, region_k.transpose(-1, -2)) * self.scale
                attn = self.attend(dots)
                attn = F.relu(attn)

                # Apply attention to values
                region_out = torch.matmul(attn, region_v)
                region_outputs.append(region_out)

            # Combine outputs from all regions
            if region_outputs:
                out[:, :, pos:pos + 1] = torch.mean(torch.stack(region_outputs, dim=0), dim=0)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer class
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., global_attention=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        # Optionally add global self-attention
        self.global_attention = global_attention
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, bias=True) if global_attention else None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FourRegionAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

            # Optionally apply global attention for global context
            if self.global_attention:
                x = rearrange(x, 'b n d -> n b d')  # Reshape for MultiheadAttention
                x, _ = self.global_attn(x, x, x)
                x = rearrange(x, 'n b d -> b n d')

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size=25, patch_size=5, num_classes=10, dim=64, depth=1, heads=8, mlp_dim=64, pool='mean', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1, global_attention=True, type = 'seg'):
        super().__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.H = image_height // patch_height
        self.W = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(1, out_channels=24, kernel_size=(3, 3, 3), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            Rearrange('b c h w y -> b (c h) w y'),
            nn.Conv2d(in_channels=24 * (channels - 2), out_channels=channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim, bias=True),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, global_attention)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(num_classes, num_classes, bias=True),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(num_classes, num_classes, bias=True)
        )
        self.out_seg = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, img, x_data):
        # img = img.squeeze(1)
        b, _, c, h, w = img.shape
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.type == 'seg':
            b, n, d = x.shape
            x_seg = x.view(b, d, self.H, self.W)
            x_seg = F.interpolate(x_seg, size=(h, w), mode='bilinear', align_corners=False)
            x_seg = self.to_latent(x_seg)
            out_seg = self.out_seg(x_seg)
            b, c, h, w = out_seg.shape
            out_cls = out_seg[:, :, h//2, w//2 ]
            out_cls = self.mlp_head(out_cls)
            return out_seg, out_cls
        elif self.type == 'cls':
            x = x.mean(dim=[2, 3])
            out_cls = self.mlp_head(x)
            return out_cls