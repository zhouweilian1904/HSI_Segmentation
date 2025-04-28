import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class General_ViT_2(nn.Module):
    def __init__(self, channels, x_data_channel, num_classes, image_size=27, patch_size=3, dim=128, depth=4, heads=4, mlp_dim=128,
                 pool='cls', dim_head=128, dropout=0., emb_dropout=0., type = 'seg'):
        super(General_ViT_2, self).__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.x_data_proj = nn.Conv2d(x_data_channel, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.x_proj = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim, bias=True),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.out_seg = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)
        self.regression = nn.Linear(dim, channels)
        self.aux_loss_weight = 1
        self.nn1 = nn.Linear(dim, num_classes, bias=True)

    def forward(self, img, x_data):
        img = img.squeeze(1)
        x_data = x_data.squeeze(1)
        # print(f'img shape: {img.shape} | x_data shape: {x_data.shape}')
        b, c, h, w = img.shape
        img = self.x_proj(img)
        x_data = self.x_data_proj(x_data)
        # print(f'img shape: {img.shape} | x_data shape: {x_data.shape}')
        # x = torch.concat([F.relu(img), F.relu(x_data)], dim=1)
        x = self.proj_out(img)
        x = self.to_patch_embedding(x)
        # x = self.spt(torch.concat([F.relu(img), F.relu(x_data)], dim=1))
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.type == 'seg':
            x_seg = x[:, 1:, :]
            x_seg = rearrange(x_seg, 'b (h w) d -> b d h w', h=int(n**0.5), w=int(n**0.5))
            x_seg = F.adaptive_avg_pool2d(x_seg, (h, w))
            x_seg = self.to_latent(x_seg)
            b, c, h, w = x_seg.shape
            out_cls = x_seg[:, :, h // 2, w // 2]
            out_seg = self.out_seg(x_seg)
            out_cls = self.nn1(out_cls)
            return out_seg, out_cls
        elif self.type == 'cls':
            x = self.to_latent(x[:, 0])
            out_cls = self.nn1(x)
            return out_cls
        else:
            raise ValueError('type must be either seg or cls')

