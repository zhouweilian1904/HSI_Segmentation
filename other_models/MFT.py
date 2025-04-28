from torch.nn import LayerNorm, Linear, Dropout, Softmax
import copy
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g, padding=kernel_size // 3,
                             stride=stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
        #         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        #         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        #         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        #         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)

        encoded = self.encoder_norm(x)

        return encoded[:, 0]


class MFT(nn.Module):
    def __init__(self, FM=16, NC=20, NCLidar=1, Classes=16, HSIOnly=False, patchsize=16, type = 'seg'):
        super(MFT, self).__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        self.HSIOnly = HSIOnly
        self.patchsize = patchsize
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                    ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )

        self.last_BandSize = NC // 2 // 2 // 2

        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidar, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.ca = TransformerEncoder(FM * 4)
        self.out3 = nn.Linear(FM * 4, Classes)
        self.out_seg = nn.Conv2d(FM * 4, Classes, 1, 1, 0)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, FM * 4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64),
                                       requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64),
                                       requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV_L)

    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0], -1, self.patchsize, self.patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)

        x1 = self.conv6(x1)
        x2 = self.lidarConv(x2)
        x2 = x2.reshape(x2.shape[0], -1, self.patchsize ** 2)
        x2 = x2.transpose(-1, -2)
        wa_L = self.token_wA_L.expand(x1.shape[0], -1, -1)
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose
        A_L = A_L.softmax(dim=-1)
        wv_L = self.token_wV_L.expand(x2.shape[0], -1, -1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        x2 = torch.einsum('bij,bjk->bik', A_L, VV_L)
        x1 = x1.flatten(2)

        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0], -1, -1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0], -1, -1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((x2, T), dim=1)  # [b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings)
        # print(f'x shape: {x.shape}')
        b, c = x.shape
        out_seg = x.reshape(b, c, 1, 1)
        out_seg = F.interpolate(out_seg, size=(self.patchsize, self.patchsize), mode='bilinear',)
        out_seg = self.out_seg(out_seg)
        x = x.reshape(x.shape[0], -1)
        out3 = self.out3(x)
        if self.type == 'seg':
            return out_seg, out3
        else:
            return out3