
from einops import rearrange
import os
import random
from typing import List, Tuple
import numpy as np
import torch
from einops import repeat
from torch import Tensor, nn
import torch.nn.functional as F


def init_random_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True


# from https://huggingface.co/transformers/_modules/transformers/modeling_utils.html
def get_module_device(parameter: nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=3,  # Change patch_dim to 3 as per your requirement
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        super().__init__()
        # Calculate padded dimensions to make the image divisible by patch_dim
        padded_dim = (img_dim + patch_dim - 1) // patch_dim * patch_dim
        self.p = patch_dim
        self.classification = classification
        self.padded_dim = padded_dim
        # tokens = number of patches
        tokens = (padded_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))

        if self.classification:
            self.mlp_head = nn.Linear(self.dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Pad image to the nearest multiple of patch size
        pad_h = (self.padded_dim - img.shape[2])
        pad_w = (self.padded_dim - img.shape[3])
        img = nn.functional.pad(img, (0, pad_w, 0, pad_h))

        # Create patches
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)
        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]


class SignleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SignleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SignleConv(in_ch, out_ch, norm_layer),
                                  SignleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv1 = DoubleConv(in_channels * 2, out_channels)
        self.conv2 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2 = None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            out = self.conv1(x)
        else:
            out = self.conv2(x1)
        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=64,
                 patch_size=3,
                 vit_transformer_dim=64,
                 vit_transformer=None,
                 vit_dim=64,
                 type = 'seg'
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.vit_transformer_dim = vit_transformer_dim
        self.vit_dim = vit_dim
        self.type = type #seg or cls
        print('Doing segmentation' if self.type == 'seg' else 'Doing classification')

        # Encoder with padding adjustment
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, vit_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.vit_dim),
            nn.ReLU(inplace=True)
        )
        self.conv1 = Bottleneck(self.vit_dim, self.vit_dim, stride=1)
        self.conv2 = Bottleneck(self.vit_dim, self.vit_dim, stride=1)

        # 计算ViT的输入尺寸
        self.img_dim_vit = img_dim

        # Initialize ViT
        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=self.vit_dim,
                       patch_dim=patch_size,
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        # Decoder path
        self.project_patches_back = nn.Linear(vit_transformer_dim, self.vit_dim * (patch_size ** 2))
        self.vit_conv = SignleConv(in_ch=self.vit_dim, out_ch=self.vit_dim)
        self.dec1 = Up(self.vit_dim, self.vit_dim)
        self.dec2 = Up(self.vit_dim, self.vit_dim)
        self.dec3 = Up(self.vit_dim, 32)
        self.conv1x1 = nn.Conv2d(in_channels=vit_dim, out_channels=classes, kernel_size=1, bias=True)
        self.nn = nn.Linear(classes, classes, bias=True)
        self.conv1x1_rec = nn.Conv2d(in_channels=vit_dim, out_channels=in_channels, kernel_size=1)

    def forward(self, x, x_data = None):
        # Encoder
        x1 = self.init_conv(x.squeeze(1))
        # print(f'x1 shape {x1.shape}')
        x2 = self.conv1(x1)
        # print(f'x2 shape {x2.shape}')
        x3 = self.conv2(x2)
        # print(f'x3 shape {x3.shape}')

        # Vision Transformer
        y = self.vit(x3)
        # print(f'y shape {y.shape}')

        # Reshape and project back
        y = self.project_patches_back(y)
        # print(f'y shape {y.shape}')
        y = rearrange(y,
                      'b (num_patches_x num_patches_y) (patch_x patch_y c) -> b c (patch_x num_patches_x) (patch_y num_patches_y)',
                      num_patches_x=self.img_dim_vit // self.patch_size,
                      num_patches_y=self.img_dim_vit // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size, c=self.vit_dim)

        # Decoder
        # print(f'y shape {y.shape}')
        y = self.vit_conv(y)
        # print(f'y shape {y.shape}')
        y = self.dec1(y, x2)
        # print(f'y shape {y.shape}')
        y = self.dec2(y, x1)
        # print(f'y shape {y.shape}')
        # y = self.dec3(y)
        # print(f'y shape {y.shape}')
        if self.type == 'seg':
            out_seg = self.conv1x1(y)
            b, c, h, w = out_seg.shape
            out_cls = out_seg[:, :, h//2, w//2]
            out_cls = self.nn(out_cls)
            return out_seg, out_cls
        elif self.type == 'cls':
            out_cls = self.conv1x1(y)
            out_cls = out_cls.mean(dim=(2, 3))
            return out_cls[:, :]
        else:
            raise ValueError('type must be seg or cls')
