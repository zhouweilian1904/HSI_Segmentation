from einops import rearrange, repeat
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
import numpy as np
import torch.nn.functional as F


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class VisionEncoderMambaBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.forward_ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.backward_ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward conv1d
        x1 = self.process(
            x,
            self.forward_conv1d,
            self.forward_ssm,
        )
        x1 = self.silu(x1)

        # backward conv1d
        x2 = self.process(
            torch.flip(x, dims=[1]),
            self.backward_conv1d,
            self.backward_ssm,
        )
        x2 = torch.flip(x2, dims=[1])
        x2 = self.silu(x2)

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

    def process(
            self,
            x: Tensor,
            conv1d: nn.Conv1d,
            ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


class Custom3DRecurrentCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Custom3DRecurrentCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size * 3, hidden_size)
        self.activation = nn.Tanh()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, h_prev1, h_prev2, h_prev3):
        input_processed = self.input_layer(input)
        hidden_concat = torch.cat((h_prev1, h_prev2, h_prev3), dim=-1)
        hidden_processed = self.hidden_layer(hidden_concat)
        hidden_state = self.activation(input_processed + hidden_processed)
        return hidden_state

class MultiDimensionalRecurrentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiDimensionalRecurrentNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.cell = Custom3DRecurrentCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, h, w, c, d = x.shape
        hidden = torch.zeros(batch_size, h, w, c, self.hidden_size, device=x.device)

        # Precompute zeros for previous hidden states
        zero_hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    h_prev1 = hidden[:, i - 1, j, k, :] if i > 0 else zero_hidden
                    h_prev2 = hidden[:, i, j - 1, k, :] if j > 0 else zero_hidden
                    h_prev3 = hidden[:, i, j, k - 1, :] if k > 0 else zero_hidden

                    hidden[:, i, j, k, :] = self.cell(x[:, i, j, k, :], h_prev1, h_prev2, h_prev3)

        return hidden


def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):
    batch_size, f, h, w, dim = patches.shape
    device = patches.device

    z, y, x = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe_z = torch.cat((z.sin(), z.cos()), dim=1)
    pe_y = torch.cat((y.sin(), y.cos()), dim=1)
    pe_x = torch.cat((x.sin(), x.cos()), dim=1)

    pe = torch.cat((pe_x, pe_y, pe_z), dim=1)

    # Ensure the feature dimension matches the original dimension by padding if necessary
    if pe.size(1) < dim:
        pe = F.pad(pe, (0, dim - pe.size(1)))

    # Reshape to (f, h, w, dim)
    pe = pe.view(f, h, w, -1)

    return pe.type(dtype)


class Mamba3D(nn.Module):

    def __init__(self, channels, num_classes, image_size, spatial_patch_size=3, channel_patch_size=10, dim=128, depth=2,
                 emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(spatial_patch_size)
        numh = image_height // patch_height
        numw = image_width // patch_width
        numc = channels // channel_patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert channels % channel_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = numh * numw * numc
        patch_dim = patch_height * patch_width * channel_patch_size
        print('num_w:', numh, 'num_w:', numw, 'num_f:', numc, 'num_patches:', num_patches, 'patch_dim:', patch_dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (c pc) (h p1) (w p2) -> b c h w (p1 p2 pc)', p1=patch_height, p2=patch_width,
                      pc=channel_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.vim = VisionEncoderMambaBlock(dim=dim, dt_rank=dim, dim_inner=dim, d_state=dim)
        self.rnn3d = MultiDimensionalRecurrentNetwork(dim, dim)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                self.rnn3d
            )

        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, hsi):
        x = hsi.squeeze(1)
        x = self.to_patch_embedding(x)
        b, c, h, w, d = x.shape
        pe = posemb_sincos_3d(x)
        # print('pe', pe.shape)

        # x = rearrange(x, 'b c h w d -> b (c h w) d') + pe
        x = x + pe
        # x = x.view(b, -1, d)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        x = self.tanh(x)
        # print('x', x.shape)

        x = x[:, c-1,h-1,w-1,:]

        x = self.to_latent(x)
        return self.mlp_head(x)
