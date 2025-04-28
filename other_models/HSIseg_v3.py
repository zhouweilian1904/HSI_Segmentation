import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import math
from matplotlib import pyplot as plt


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_types=Attention):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attn_types(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


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
        self.DWA = Deformable_Window_Attention(dim=dim, depth=1, heads=heads, mlp_dim=dim,
                                               pool='cls', dim_head=dim, dropout=dropout, emb_dropout=dropout,
                                               attn_types=Attention)
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

                region_pooled = self.DWA(region)

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


class Deformable_Window_Attention(nn.Module):
    def __init__(self, dim=64, depth=1, heads=8, mlp_dim=64,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0., attn_types=Attention):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            # FeedForward(dim, dim * 2, dim, dropout=0.),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_types=attn_types)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, dim)
        # self.mlp_head = FeedForward(dim, dim * 2, dim, dropout=0.)

    def forward(self, seq):
        x = self.to_patch_embedding(seq)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding(rearrange(x, 'b n d -> b d n')).permute(0, 2, 1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class DSRT_down(nn.Module):
    def __init__(self, channels=64, image_size_in=128, image_size_out=64, patch_size=2, shuffle_factor=2, dim=64,
                 depth=4, heads=4, dropout=0.1, use_pooling=False, attn_types=Attention, ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size_in = image_size_in
        self.image_size_out = image_size_out
        self.shuffle_factor = shuffle_factor
        self.use_pooling = use_pooling

        num_patches = (self.image_size_in ** 2) // (self.patch_size ** 2) // (self.shuffle_factor ** 2)
        shuffle_dim = channels * (self.shuffle_factor ** 2)
        patch_dim = shuffle_dim * (self.patch_size ** 2)

        self.to_patch_embedding = nn.Sequential(
            nn.PixelUnshuffle(self.shuffle_factor),
            nn.Conv2d(shuffle_dim, patch_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # FeedForward(patch_dim, dim * 2, dim, dropout=0.),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout, attn_types=attn_types)

        self.to_latent = nn.Identity()

        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.out_seg = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, img):
        # print(img.shape)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # print(x.shape)

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=int(n ** 0.5), w=int(n ** 0.5))
        # print(x.shape)
        x = self.to_latent(x)
        if self.use_pooling:
            x = F.adaptive_avg_pool2d(x, output_size=(self.image_size_out, self.image_size_out))
        else:
            x = self.upsample(x)
            x = F.interpolate(x, size=(self.image_size_out, self.image_size_out), mode='bilinear', align_corners=False)
        x = self.out_seg(x)
        # print(x.shape)
        return x


class DSRT_up(nn.Module):
    def __init__(self, channels=64, image_size_in=64, image_size_out=128, patch_size=2, shuffle_factor=2, dim=64,
                 depth=4, heads=4, dropout=0.1, use_pooling=False, attn_types=Attention, ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size_in = image_size_in
        self.image_size_out = image_size_out
        self.shuffle_factor = shuffle_factor
        self.use_pooling = use_pooling

        num_patches = ((self.image_size_in * self.shuffle_factor) ** 2) // (self.patch_size ** 2)
        shuffle_dim = channels // (self.shuffle_factor ** 2)
        patch_dim = shuffle_dim * (self.patch_size ** 2)

        self.to_patch_embedding = nn.Sequential(
            nn.PixelShuffle(self.shuffle_factor),
            nn.Conv2d(shuffle_dim, patch_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # FeedForward(patch_dim, dim * 2, dim, dropout=0.),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim, dim, dropout, attn_types=attn_types)

        self.to_latent = nn.Identity()

        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.out_seg = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, img):
        # print(img.shape)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # print(x.shape)

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=int(n ** 0.5), w=int(n ** 0.5))

        x = self.to_latent(x)
        if self.use_pooling:
            x = F.adaptive_avg_pool2d(x, output_size=(self.image_size_out, self.image_size_out))
        else:
            x = self.upsample(x)
            x = F.interpolate(x, size=(self.image_size_out, self.image_size_out), mode='bilinear', align_corners=False)
        x = self.out_seg(x)
        # print(x.shape)

        return x


class Encoder_Down(nn.Module):
    """Downscaling with maxpool, depthwise separable convolution, and transformer-based processing."""

    def __init__(self, channels, image_size_in, image_size_out, patch_size, shuffle_factor,
                 dim, depth, heads, dropout, activation=nn.GELU, attn_types=Attention):
        super().__init__()
        self.image_size_in = image_size_in
        self.image_size_out = image_size_out

        # Submodules
        # self.depthwise_separable_conv = Depthwise_Separable_Conv(channels, 2, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(channels, dim, kernel_size=1, bias=True)
        self.vit_down = DSRT_down(
            channels=channels, image_size_in=image_size_in, image_size_out=image_size_out,
            patch_size=patch_size, shuffle_factor=shuffle_factor, dim=dim, depth=depth,
            heads=heads, dropout=dropout, attn_types=attn_types, use_pooling=False
        )
        self.norm = nn.BatchNorm2d(dim)
        self.activation = activation()  # Configurable activation function

    def forward(self, x):
        # Residual path (downsample input to target size)
        residual = F.interpolate(x, size=(self.image_size_out, self.image_size_out), mode='bilinear',
                                 align_corners=False)
        residual = self.conv2(residual)

        # Depthwise separable convolution
        x = self.conv(x)

        # Activation
        x = self.activation(x)

        # ViT-based downsampling
        x = self.vit_down(x)
        # print(x.shape)

        # Add residual and normalize
        x += residual
        return self.norm(x)


class Spectral_CFI(nn.Module):
    def __init__(self, dim, heads, input_size, bias=True, dropout=0.1):
        super().__init__()
        H, W = input_size, input_size
        self.heads = heads
        self.scale = (H * W) ** -0.5

        # Normalization layers for spatial features
        self.norm1 = nn.LayerNorm(H * W)
        self.norm2 = nn.LayerNorm(H * W)

        # Attention mechanism components
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Linear layers for query, key, value projections
        # self.to_qkv_1 = nn.Linear(H * W, H * W * 3, bias=bias)
        self.to_qkv_1 = FeedForward(H * W, H * W * 3 * 2, H * W * 3, dropout=dropout)
        # self.to_qkv_2 = nn.Linear(H * W, H * W * 3, bias=bias)
        self.to_qkv_2 = FeedForward(H * W, H * W * 3 * 2, H * W * 3, dropout=dropout)

        # Output projections
        self.to_out_1 = nn.Sequential(
            nn.Linear(H * W * heads, H * W),
            # FeedForward(H * W * heads, H * W * 2, H * W, dropout=dropout),
            nn.Dropout(dropout)
        )
        self.to_out_2 = nn.Sequential(
            nn.Linear(H * W * heads, H * W),
            # FeedForward(H * W * heads, H * W * 2, H * W, dropout=dropout),
            nn.Dropout(dropout)
        )

    @staticmethod
    def smoother(x):
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def sharper(x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1_1 = self.smoother(x1)
        x2_2 = self.sharper(x2)

        x1_1 = rearrange(x1_1, 'b c h w -> b c (h w)')
        x2_2 = rearrange(x2_2, 'b c h w -> b c (h w)')

        x1_1 = self.norm1(x1_1)
        x2_2 = self.norm2(x2_2)

        qkv_1 = self.to_qkv_1(x1_1).chunk(3, dim=-1)
        qkv_2 = self.to_qkv_2(x2_2).chunk(3, dim=-1)

        q1, k1, v1 = map(lambda t: rearrange(t, 'b c (head hw) -> b head c hw', head=self.heads), qkv_1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b c (head hw) -> b head c hw', head=self.heads), qkv_2)

        dots1 = torch.matmul(q1, k2.transpose(-1, -2)) * self.scale
        dots2 = torch.matmul(q2, k1.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots1)
        attn2 = self.attend(dots2)

        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)

        out1 = torch.matmul(attn1, v2)
        out2 = torch.matmul(attn2, v1)

        out1 = rearrange(out1, 'b head c hw -> b c (head hw)')
        out2 = rearrange(out2, 'b head c hw -> b c (head hw)')
        out1 = self.to_out_1(out1)
        out2 = self.to_out_2(out2)
        out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h)
        out2 = rearrange(out2, 'b c (h w) -> b c h w', h=h)
        return out1 + x1, out2 + x2


class Spatial_CFI(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = dim ** -0.5

        # Query-value and Key-value layers
        self.x1_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.x2_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # Depthwise separable convolutions
        self.x1_dw_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.x2_dw_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        # Output projection layers
        self.project_out_x1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_x2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Projection layer for final output
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def smoother(x):
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def sharper(x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        # Get spatial sizes
        b, c, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape

        # x1_1 = self.smoother(x1)
        # x2_2 = self.sharper(x2)

        # Query and value for x1
        x1_qkv = self.x1_dw_conv(self.x1_qkv(x1))
        x1_q, x1_k, x1_v = x1_qkv.chunk(3, dim=1)

        # Query and value for x2
        x2_qkv = self.x2_dw_conv(self.x2_qkv(x2))
        x2_q, x2_k, x2_v = x2_qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        x1_q = rearrange(x1_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_q = rearrange(x2_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x1_v = rearrange(x1_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_v = rearrange(x2_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x1_k = rearrange(x1_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x2_k = rearrange(x2_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize query and key
        q1 = F.normalize(x1_q, dim=-1)  # Query from x1
        k2 = F.normalize(x2_k, dim=-1)  # Key from x2

        q2 = F.normalize(x2_q, dim=-1)  # Query from x2
        k1 = F.normalize(x1_k, dim=-1)  # Key from x1

        # Compute attention
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature  # Attention: x1 -> x2
        attn1 = self.dropout(attn1.softmax(dim=-1))  # Shape: (b, num_heads, h1*w1, h2*w2)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature  # Attention: x2 -> x1
        attn2 = self.dropout(attn2.softmax(dim=-1))  # Shape: (b, num_heads, h2*w2, h1*w1)

        # Aggregate values
        x1_out = rearrange((attn1 @ x1_v), 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)
        x2_out = rearrange((attn2 @ x2_v), 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h2, w=w2)

        # Combine with residual connections
        x1_out = F.gelu(self.dropout(self.project_out_x1(x1_out)) + x1)
        x2_out = F.gelu(self.dropout(self.project_out_x2(x2_out)) + x2)

        return x1_out, x2_out


class Decoder_Up(nn.Module):
    """Upscaling with feature concatenation, depthwise separable convolution, and transformer-based processing."""

    def __init__(self, channels_x1, channels_x2, image_size_in, image_size_out, patch_size, shuffle_factor,
                 dim, depth, heads, dropout, activation=nn.GELU, attn_types=Attention):
        super().__init__()
        self.image_size_in = image_size_in
        self.image_size_out = image_size_out
        self.inial_conv_lr = nn.Conv2d(channels_x1, channels_x2, kernel_size=1, bias=True)
        # Submodules
        self.conv = nn.Conv2d(channels_x2 * 2, dim, kernel_size=1, bias=True)
        # self.depthwise_separable_conv = Depthwise_Separable_Conv(channels, 2, channels)
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(dim)
        self.vit_up = DSRT_up(
            channels=dim, image_size_in=image_size_in, image_size_out=image_size_out,
            patch_size=patch_size, shuffle_factor=shuffle_factor, dim=dim, depth=depth,
            heads=heads, dropout=dropout, attn_types=attn_types, use_pooling=False,
        )
        self.activation = activation()  # Configurable activation function
        self.Spatial_CFI_1 = Spatial_CFI(dim, heads)
        self.Spatial_CFI_2 = Spatial_CFI(dim, heads)

    def forward(self, x_lr, x_hr):
        # Resize high-resolution input to match low-resolution spatial dimensions
        x_lr = self.inial_conv_lr(x_lr)
        b, c, h, w = x_lr.shape
        x_hr = F.interpolate(x_hr, size=(h, w), mode='bilinear', align_corners=False)
        x_lr1, x_hr1 = self.Spatial_CFI_1(x_lr, x_hr)
        x_hr2, x_lr2 = self.Spatial_CFI_2(x_hr, x_lr)

        # Concatenate low-resolution and high-resolution features
        x = torch.cat([x_lr1 + x_lr2, x_hr1 + x_hr2], dim=1)

        # Reduce channels with a 1x1 convolution
        x = self.conv(x)

        # Create residual for output refinement
        residual = F.interpolate(x, size=(self.image_size_out, self.image_size_out), mode='bilinear',
                                 align_corners=False)

        # Process with depthwise separable convolution and activation
        x = self.conv_1(x)
        x = self.activation(x)

        # Process with ViT-based upsampling
        x = self.vit_up(x)

        # Add residual and normalize
        x += residual
        return self.norm(x)


class DFS(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d(downSize),
            nn.Conv2d(nIn, nOut * 2, 1, bias=True),
            nn.BatchNorm2d(nOut * 2),
            # nn.GELU(),
            nn.ConvTranspose2d(nOut * 2, nOut, 1, bias=True),
            nn.AdaptiveMaxPool2d(upSize),
            nn.BatchNorm2d(nOut),
            # nn.GELU(),
        )

    def forward(self, x):
        return self.features(x)


class ViT_UNet(nn.Module):

    def __init__(self, channels=20, x_data_channel=1, num_classes=20, image_size=49, dim=64, depth=2, heads=2,
                 dropout=0., attn_type=Attention):
        super().__init__()
        self.initialize = nn.Conv2d(channels, dim * 1, kernel_size=1, bias=True)
        self.inc_x_data = nn.Conv2d(x_data_channel, dim, kernel_size=1, bias=True)
        self.image_size = image_size

        # Encoder (Downsampling)
        self.down1 = Encoder_Down(channels=dim * 1, image_size_in=49, image_size_out=36, patch_size=7, shuffle_factor=1,
                                  dim=dim * 2, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.down2 = Encoder_Down(channels=dim * 2, image_size_in=36, image_size_out=25, patch_size=6, shuffle_factor=1,
                                  dim=dim * 3, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.down3 = Encoder_Down(channels=dim * 3, image_size_in=25, image_size_out=16, patch_size=5, shuffle_factor=1,
                                  dim=dim * 4, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.down4 = Encoder_Down(channels=dim * 4, image_size_in=16, image_size_out=9, patch_size=4, shuffle_factor=1,
                                  dim=dim * 5, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.down5 = Encoder_Down(channels=dim * 5, image_size_in=9, image_size_out=4, patch_size=3, shuffle_factor=1,
                                  dim=dim * 6, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)

        # PSP Modules (Pyramid Scene Parsing)
        self.p1 = DFS(dim * 6, dim * 5, upSize=9, downSize=9 // 2)
        self.p1_prob = nn.Conv2d(dim * 5, num_classes, kernel_size=1, bias=True)
        self.p2 = DFS(dim * 5, dim * 4, upSize=16, downSize=16 // 2)
        self.p2_prob = nn.Conv2d(dim * 4, num_classes, kernel_size=1, bias=True)
        self.p3 = DFS(dim * 4, dim * 3, upSize=25, downSize=25 // 2)
        self.p3_prob = nn.Conv2d(dim * 3, num_classes, kernel_size=1, bias=True)
        self.p4 = DFS(dim * 3, dim * 2, upSize=36, downSize=36 // 2)
        self.p4_prob = nn.Conv2d(dim * 2, num_classes, kernel_size=1, bias=True)
        self.p5 = DFS(dim * 2, dim * 1, upSize=49, downSize=49 // 2)
        self.p5_prob = nn.Conv2d(dim * 1, num_classes, kernel_size=1, bias=True)

        # Decoder (Upsampling)
        self.up1 = Decoder_Up(channels_x1=dim * 6, channels_x2=dim * 5, image_size_in=4, image_size_out=9,
                              patch_size=2, shuffle_factor=1,
                              dim=dim * 5, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.up2 = Decoder_Up(channels_x1=dim * 5, channels_x2=dim * 4, image_size_in=9, image_size_out=16,
                              patch_size=3, shuffle_factor=1,
                              dim=dim * 4, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.up3 = Decoder_Up(channels_x1=dim * 4, channels_x2=dim * 3, image_size_in=16, image_size_out=25,
                              patch_size=4, shuffle_factor=1,
                              dim=dim * 3, depth=depth, heads=heads, dropout=dropout)
        self.up4 = Decoder_Up(channels_x1=dim * 3, channels_x2=dim * 2, image_size_in=25, image_size_out=36,
                              patch_size=5, shuffle_factor=1,
                              dim=dim * 2, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)
        self.up5 = Decoder_Up(channels_x1=dim * 2, channels_x2=dim * 1, image_size_in=36, image_size_out=49,
                              patch_size=6, shuffle_factor=1,
                              dim=dim * 1, depth=depth, heads=heads, dropout=dropout, attn_types=attn_type)

        # Final layers
        self.out_seg = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        self.out_rec = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        self.out_cls = nn.Linear(dim, num_classes, bias=True)

    def forward(self, img, x_data):
        x = img.squeeze(1)  # Remove single channel dimension
        x_data = x_data.squeeze(1)
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(49, 49), mode='bilinear', align_corners=False)
        x_data = F.interpolate(x_data, size=(49, 49), mode='bilinear', align_corners=False)
        x = x + torch.randn_like(x) * 0.02  # Add noise for regularization
        x_data = x_data + torch.randn_like(x_data) * 0.02
        x0 = self.initialize(x)
        x0_x = self.inc_x_data(x_data)

        # Encoder
        x1 = self.down1(x0 + x0_x)
        # print(f'x1: {x1.shape}')
        x2 = self.down2(x1)
        # print(f'x2: {x2.shape}')
        x3 = self.down3(x2)
        # print(f'x3: {x3.shape}')
        x4 = self.down4(x3)
        # print(f'x4: {x4.shape}')
        x5 = self.down5(x4)
        # print(f'x5: {x5.shape}')

        # PSP Modules
        p1 = self.p1(x5)
        p1_prob, p1 = apply_mask_and_threshold(p1, self.p1_prob)
        # print(f'p1: {p1.shape} | p1_prob: {p1_prob.shape}')

        p2 = self.p2(p1)
        p2_prob, p2 = apply_mask_and_threshold(p2, self.p2_prob)
        # print(f'p2: {p2.shape} | p2_prob: {p2_prob.shape}')

        p3 = self.p3(p2)
        p3_prob, p3 = apply_mask_and_threshold(p3, self.p3_prob)
        # print(f'p3: {p3.shape} | p3_prob: {p3_prob.shape}')

        p4 = self.p4(p3)
        p4_prob, p4 = apply_mask_and_threshold(p4, self.p4_prob)
        # print(f'p4: {p4.shape} | p4_prob: {p4_prob.shape}')

        p5 = self.p5(p4)
        p5_prob, p5 = apply_mask_and_threshold(p5, self.p5_prob)
        # print(f'p5: {p5.shape} | p5_prob: {p5_prob.shape}')

        # Decoder
        x4_up = self.up1(x5, x4) * p1_prob.unsqueeze(1)
        x3_up = self.up2(x4_up, x3) * p2_prob.unsqueeze(1)
        x2_up = self.up3(x3_up, x2) * p3_prob.unsqueeze(1)
        x1_up = self.up4(x2_up, x1) * p4_prob.unsqueeze(1)
        x0_up = self.up5(x1_up, x0) * p5_prob.unsqueeze(1)

        # Final interpolation and outputs
        x0_up = F.interpolate(x0_up, size=(h, w), mode='bilinear', align_corners=False)
        out_cls = self.out_cls(x0_up[:, :, x0_up.size(2) // 2, x0_up.size(2) // 2])  # Classification output
        out_seg = self.out_seg(x0_up)  # Segmentation output
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        # p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        # p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        # p2 = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        # p1 = F.interpolate(p1, size=(h, w), mode='bilinear', align_corners=False)
        out_rec = self.out_rec(p5)  # Reconstruction output

        return out_seg, out_cls, out_rec


def apply_mask_and_threshold(x, prob_layer, quantile=0.7, debug=False):
    """
    Compute probabilities and mask pixels below a specified quantile threshold.

    Args:
        x (Tensor): Input tensor.
        prob_layer (nn.Module): Layer projecting `x` to class probabilities.
        quantile (float): Threshold quantile (e.g., 0.5 for median).
        debug (bool): Whether to visualize the masked probability map.

    Returns:
        masked_prob (Tensor): Masked probability map.
        x (Tensor): Original input tensor (unchanged).
    """
    prob = prob_layer(x)  # Project to class probabilities
    prob = F.softmax(prob, dim=1)  # Apply softmax across class dimension
    prob_max, _ = torch.max(prob, dim=1)  # Maximum probability per pixel

    # Compute quantile threshold
    threshold = torch.quantile(prob_max, quantile)  # Quantile value across all pixels

    # Create the mask
    mask = prob_max >= threshold
    masked_prob = prob_max.clone()  # Clone to avoid in-place modifications
    masked_prob[~mask] = 0.0  # Mask out pixels below the threshold

    # Optional debugging visualization
    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(masked_prob[0].detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title("Masked Probability Map")
        plt.show()

    return masked_prob, x
