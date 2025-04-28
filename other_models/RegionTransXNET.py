import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from other_models.FreqFusion import FreqFusion

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

    def forward(self, x, H, W):
        # x shape: (b, n, dim)
        x = self.norm(x)
        b, n, dim = x.shape

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

# Transformer class
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., global_attention=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        # Optionally add global self-attention
        self.global_attention = global_attention
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, bias=True) if global_attention else None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # FourRegionAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.H = None
        self.W = None

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
                 dim_head=64, dropout=0.1, emb_dropout=0.1, global_attention=True, **kwargs):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.H = image_height // patch_height
        self.W = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
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
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(dim, dim, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(emb_dropout),
        #     nn.Linear(dim, num_classes, bias=True)
        # )
        # self.out_seg = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(emb_dropout),
        #     nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        # )

    def forward(self, img):
        # img = img.squeeze(1)
        b, c, h, w = img.shape
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        # Set H and W in transformer
        self.transformer.H = self.H
        self.transformer.W = self.W

        x = self.transformer(x)
        b, n, d = x.shape
        x_seg = x.view(b, d, self.H, self.W)
        x_seg = F.interpolate(x_seg, size=(h, w), mode='bilinear', align_corners=False)
        x_seg = self.to_latent(x_seg)


        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x = x[:, int(n//2), :]
        # x = self.to_latent(x)
        # return self.out_seg(x_seg), self.mlp_head(x)
        return x_seg, x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, scale_factor):
        x = self.up(x)
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return x

class XNET(nn.Module):
    def __init__(self, *, image_size, num_classes=10, dim=32, depth=1, heads=8, mlp_dim=64, pool='mean', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1, global_attention=True, **kwargs):
        super().__init__()
        
        image_size1 = image_size
        image_size2 = 25
        image_size3 = 16

        self.initiate = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.initiate_data = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.vit1 = ViT(image_size=35, patch_size=5, num_classes=num_classes, dim=32, channels=32)
        self.vit2 = ViT(image_size=25, patch_size=5, num_classes=num_classes, dim=128, channels=128)
        self.vit3 = ViT(image_size=16, patch_size=4, num_classes=num_classes, dim=512, channels=512)

        self.xdown1 = Downsample(64, 128)
        self.xdown2 = Downsample(256, 512)
        self.xup3 = Upsample(1024, 128)
        self.xup4 = Upsample(256, 32)

        self.ff1 = FreqFusion(hr_channels=32, lr_channels=32)
        self.ff1down = Downsample(32, 32, kernel_size=2, stride=2)
        self.ff2 = FreqFusion(hr_channels=128, lr_channels=128)
        self.ff2down = Downsample(128, 128, kernel_size=2, stride=2)
        self.ff3 = FreqFusion(hr_channels=512, lr_channels=512)
        self.ff3down = Downsample(512, 512, kernel_size=2, stride=2)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes, bias=True)
        )
        self.out_seg = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        )
    
    def forward(self, x, x_data):
        # branch x
        x = x.squeeze(1)
        x_data = x_data.squeeze(1)
        # print(x.shape)
        # print(x_data.shape)
        x = self.initiate(x)
        x_data = self.initiate_data(x_data)
        # ----------------- XNET -----------------
        # layer 1 49*49*32-->25*25*128
        x_v1,_ = self.vit1(x) # 49*49*32-->49*49*32
        x_data_v1,_ = self.vit1(x_data) # 49*49*32-->49*49*32
        x_data_v1up = F.interpolate(x_data_v1, scale_factor=2, mode='bilinear', align_corners=False) # 49*49*32-->98*98*32
        _,x_data_ff1,x_ff1 = self.ff1(hr_feat=x_data_v1up, lr_feat=x_v1) # 98*98*32-->98*98*32
        x_data_ff1 = self.ff1down(x_data_ff1) # 98*98*32-->49*49*32
        x_ff1 = self.ff1down(x_ff1) # 98*98*32-->49*49*32
        x_v1 = torch.cat([x_v1, x_ff1], dim=1)  # 49*49*64
        x_data_v1 = torch.cat([x_data_v1, x_data_ff1], dim=1) # 49*49*64

        x_d1 = self.xdown1(x_v1) # 49*49*64-->25*25*128
        x_d1 = F.interpolate(x_d1, size=(25, 25), mode='bilinear', align_corners=False) # 25*25*128-->25*25*128
        x_data_d1 = self.xdown1(x_data_v1) # 49*49*64-->25*25*128
        x_data_d1 = F.interpolate(x_data_d1, size=(25, 25), mode='bilinear', align_corners=False) # 25*25*128-->25*25*128
        torch.cuda.empty_cache()
        # print('layer 1 done')
        # layer 2 25*25*128-->16*16*512
        x_v2,_ = self.vit2(x_d1) # 25*25*128-->25*25*128
        x_data_v2,_ = self.vit2(x_data_d1) # 25*25*128-->25*25*128
        x_data_v2up = F.interpolate(x_data_v2, size=(50, 50), mode='bilinear', align_corners=False) # 25*25*128-->50*50*128
        _,x_data_ff2,x_ff2 = self.ff2(hr_feat=x_data_v2up, lr_feat=x_v2) # 50*50*128-->50*50*128
        x_data_ff2 = self.ff2down(x_data_ff2) # 50*50*128-->25*25*128
        x_ff2 = self.ff2down(x_ff2) # 50*50*128-->25*25*128
        x_v2 = torch.cat([x_v2, x_ff2], dim=1)  # 25*25*256
        x_data_v2 = torch.cat([x_data_v2, x_data_ff2], dim=1) # 25*25*256

        x_d2 = self.xdown2(x_v2) # 25*25*256-->12*12*512
        x_d2 = F.interpolate(x_d2, size=(16, 16), mode='bilinear', align_corners=False) # 12*12*512-->16*16*512
        x_data_d2 = self.xdown2(x_data_v2) # 25*25*256-->12*12*512
        x_data_d2 = F.interpolate(x_data_d2, size=(16, 16), mode='bilinear', align_corners=False) # 12*12*512-->16*16*512
        torch.cuda.empty_cache()
        # print('layer 2 done')
        # layer 3 bottom 16*16*512-->16*16*512
        x_v3,_ = self.vit3(x_d2) # 16*16*512-->16*16*512
        x_data_v3,_ = self.vit3(x_data_d2) # 16*16*512-->16*16*512
        torch.cuda.empty_cache()
        # print('layer 3 done')
        # layer 4 16*16*512-->25*25*128
        x_data_v3up = F.interpolate(x_data_v3, size=(32, 32), mode='bilinear', align_corners=False) # 16*16*512-->32*32*512
        _,x_data_ff3,x_ff3 = self.ff3(hr_feat=x_data_v3up, lr_feat=x_v3) # 32*32*512-->32*32*512
        x_data_ff3 = self.ff3down(x_data_ff3) # 32*32*512-->16*16*512
        x_ff3 = self.ff3down(x_ff3) # 32*32*512-->16*16*512
        x_v3 = torch.cat([x_v3, x_ff3], dim=1) # 16*16*1024
        x_data_v3 = torch.cat([x_data_v3, x_data_ff3], dim=1) # 16*16*1024

        x_u3 = self.xup3(x_v3, scale_factor=1.5625) # 16*16*1024-->25*25*256
        x_data_u3 = self.xup3(x_data_v3, scale_factor=1.5625) # 16*16*1024-->25*25*256
        torch.cuda.empty_cache()
        # print('layer 4 done')
        # layer 5 25*25*128-->36*36*64
        x_v4,_ = self.vit2(x_u3) # 25*25*128-->25*25*128
        x_data_v4,_ = self.vit2(x_data_u3) # 25*25*128-->25*25*128
        x_data_v4up = F.interpolate(x_data_v4, size=(50, 50), mode='bilinear', align_corners=False) # 25*25*128-->50*50*128
        _,x_data_ff4,x_ff4 = self.ff2(hr_feat=x_data_v4up, lr_feat=x_v4) # 50*50*128-->50*50*128
        x_data_ff4 = self.ff2down(x_data_ff4) # 50*50*128-->25*25*128
        x_ff4 = self.ff2down(x_ff4) # 50*50*128-->25*25*128
        x_v4 = torch.cat([x_v4, x_ff4], dim=1) # 25*25*256
        x_data_v4 = torch.cat([x_data_v4, x_data_ff4], dim=1) # 25*25*256

        x_u4 = self.xup4(x_v4, scale_factor=1.4) # 25*25*256-->49*49*32
        x_data_u4 = self.xup4(x_data_v4, scale_factor=1.4) # 25*25*256-->49*49*32
        torch.cuda.empty_cache()
        seg_out, x_out = self.vit1(x_u4) # 49*49*32-->49*49*32
        seg_data_out, x_data_out = self.vit1(x_data_u4) # 49*49*32-->49*49*32
        # print('layer 5 done')
        #----------------------------------------
        x_seg = self.out_seg(seg_out)
        x_seg_data = self.out_seg(seg_data_out)
        cls_x = self.mlp_head(x_out)
        cls_x_data = self.mlp_head(x_data_out)
        # print(f'x_seg {x_seg.shape} | x_seg_data {x_seg_data.shape} | cls_x {cls_x.shape} | cls_x_data {cls_x_data.shape}')

        return x_seg, cls_x[:, cls_x.shape[1]//2, :]