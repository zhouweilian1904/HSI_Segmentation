
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
# from TPPI.models.utils import *



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)#用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0的正态分布
#残差网络
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
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
# 等于 FeedForward
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        # head_dim = dim // heads
        # self.scale =  head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        #print("x", x.size())
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        #print("q", q.size())
        # split into multi head attentions分成多个头的注意力

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale#einsum求和
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')
        # concat heads into one matrix, ready for next encoder blockConcat头进入一个矩阵，准备为下一个编码器块
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class GWTransformer(nn.Module):

    def __init__(self, num_tokens=4, dim=128, channelsNum=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(GWTransformer, self).__init__()

        # Tokenization
        self.L = num_tokens
        self.cT = dim

        self.token_wA = nn.Parameter(torch.empty(1, self.L, channelsNum),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, channelsNum, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()


    def forward(self, x):

        # 参考SSFTT
        #print("x", x.size())
        x = rearrange(x, 'b c h w -> b (h w) c')
        #print("x", x.size())
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        #print("wa", wa.size())
        A = torch.einsum('bij,bjk->bik', x, wa)
        #print("A", A.size())
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)#矩阵乘法
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        trans = self.transformer(x)  # main game
        x = trans + x
        # x = trans
        return x


class TransHSI(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, in_channels=20, patch_size=13, num_classes=16, depth=1, heads=8, mlp_dim=8, dropout=0.1,  type = 'seg'):
        super(TransHSI, self).__init__()
        self.type = type
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')
        self.patch_size = patch_size
        channelsNum = [32, 64, 128]
        self.cT = channelsNum[2]

        self.spe_conv3d01= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channelsNum[0], kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            # nn.Dropout3d(p=0.1),
        )
        self.spe_conv3d02 = nn.Sequential(
            nn.Conv3d(in_channels=channelsNum[0], out_channels=channelsNum[1], kernel_size=(5, 3, 3), stride=(1, 1, 1),
                      padding=(2, 1, 1)),
            nn.BatchNorm3d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=channelsNum[1], out_channels=channelsNum[0], kernel_size=(7, 3, 3), stride=(1, 1, 1),
                      padding=(3, 1, 1)),
            nn.BatchNorm3d(channelsNum[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
        )
        # in_channels=channelsNum[0]*Band15
        self.spa_conv2d01 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[0]*in_channels, out_channels=channelsNum[1], kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        # Transformer
        self.transformer01 = Transformer(channelsNum[1],depth, heads,mlp_dim, dropout)

        self.spa_conv2d02 = nn.Sequential(

            nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),#消融实验区
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(in_channels=channelsNum[2], out_channels=channelsNum[1], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        #消融实验
        # self.spa_conv2d02 = nn.Sequential(
        #     # nn.Conv2d(in_channels=channelsNum[1], out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
        #     #          padding=(1, 1)),
        #     nn.Conv2d(in_channels=15, out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
        #               padding=(1, 1)),  # 消融实验区
        #     nn.BatchNorm2d(channelsNum[2]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.1),
        #
        #     nn.Conv2d(in_channels=channelsNum[2], out_channels=channelsNum[1], kernel_size=(3, 3), stride=(1, 1),
        #               padding=(1, 1)),
        #     nn.BatchNorm2d(channelsNum[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.1),
        # )
        # Transformer
        # self.dim = channelsNum[1]
        self.transformer02 = Transformer(channelsNum[1],depth, heads,mlp_dim, dropout)

        self.spa_conv2d03 = nn.Sequential(
            nn.Conv2d(in_channels=channelsNum[1]*2, out_channels=channelsNum[2], kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(channelsNum[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )



        # Tokenization

        self.transformer03 = GWTransformer(channelsNum=channelsNum[2])

        self.to_cls_token = nn.Identity()


        self.nn1 = nn.Sequential(
            nn.Linear(channelsNum[2], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.nn2 = nn.Linear(64, num_classes)

        self.out_seg = nn.Conv2d(channelsNum[2], num_classes, kernel_size=1)

        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)


    def forward(self, x, mask=None):

        # local and global spectral feature extraction局部和全局光谱特征提取
        # x=self.spe_conv00(x)
        raw_x = x

        # x = torch.unsqueeze(x, dim=1)  #
        x = self.spe_conv3d01(x)

        for iterNum in range(2):
        #     # print("X1", x.size())
            spe_conv3d = self.spe_conv3d02(x)
            x = spe_conv3d + x
            # x = spe_conv3d
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.spa_conv2d01(x)
        #
        trans01 = rearrange(x, 'b c h w -> b (h w) c')#维度转换  flatten
        trans01 = self.transformer01(trans01)  # main game
        trans01 = rearrange(trans01, 'b (h w) c -> b c h w', h=self.patch_size, w=self.patch_size)
        # print("spe_conv3d2", spe_conv3d.size())
        x = trans01  + x
        # x = trans01
        spectral_x = x

        # print("X2", x.size())

        # local and global spatial feature extraction局部和全局空间特征提取
        for iterNum in range(2):
            spa_conv2d= self.spa_conv2d02(x)
            x = spa_conv2d + x
            # x = spa_conv2d
        trans02 = rearrange(x, 'b c h w -> b (h w) c')
        trans02 = self.transformer02(trans02)  # main game
        # print("X2", spa_conv2d.size())
        trans02 = rearrange(trans02, 'b (h w) c -> b c h w',h=self.patch_size, w=self.patch_size)
        x = trans02 + x
        # x = trans02
        Spatial_x = x

        x = torch.cat([spectral_x,Spatial_x], dim=1)
        # x = torch.cat([spectral_x, raw_x], dim=1)
        x = self.spa_conv2d03(x)
        # x=rearrange(x, 'b c h w -> b (h w c)')
        # print("x", x.size())
        # 参考SSFTT
        trans03 = self.transformer03(x)  # main game
        if self.type == 'cls':
            x = self.to_cls_token(trans03[:, 0])
            x = self.nn1(x)
            x = self.nn2(x)
            return x
        elif self.type == 'seg':
            out_seg = trans03[:, 1:, :]
            b, n, d = out_seg.shape
            out_cls = out_seg[:, n//2, :]
            out_seg = out_seg.reshape(-1, self.cT, n // 2, n // 2)
            out_seg = F.interpolate(out_seg, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
            out_seg = self.out_seg(out_seg)
            out_cls = self.nn1(out_cls)
            out_cls = self.nn2(out_cls)
            return out_seg, out_cls
        else:
            raise ValueError('type must be either cls or seg')
        #
        # x = self.to_cls_token(trans03[:, 0])
        # # print("x", x.size())
        #
        # x = self.nn1(x)
        # x = self.nn2(x)
        # # print("X2", x.size())
        # return x


if __name__ == '__main__':
    model = TransHSI()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 15, 11, 11)
    y = model(input)
    print(y.size())
