from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PScan2D(torch.autograd.Function):
    @staticmethod
    def pscan2d(A, X):
        # A : (B, D, H, W, N)
        # X : (B, D, H, W, N)

        B, D, H, W, _ = A.size()
        num_steps_h = int(math.log2(H))
        num_steps_w = int(math.log2(W))

        # up sweep or reduction step
        Aa = A  # ([100, 64, 4, 4, 32])
        Xa = X  # ([100, 64, 4, 4, 32])

        # Horizontal scan h
        for k in range(num_steps_w):
            T = 2 * (Xa.size(3) // 2)

            Aa = Aa[:, :, :, :T, :].reshape(B, D, H, T // 2, 2, -1)  # ([100, 64, 4, 4, 2, 32])
            Xa = Xa[:, :, :, :T, :].reshape(B, D, H, T // 2, 2, -1)  # ([100, 64, 4, 4, 2, 32])

            Xa[:, :, :, :, 1, :].add_(Aa[:, :, :, :, 1, :].mul(Xa[:, :, :, :, 0, :]))  # ([100, 64, 4, 4, 2, 32])
            Aa[:, :, :, :, 1, :].mul_(Aa[:, :, :, :, 0, :])  # ([100, 64, 4, 4, 2, 32])

            Aa = Aa[:, :, :, :, 1, :]  # ([100, 64, 4, 4, 32])
            Xa = Xa[:, :, :, :, 1, :]  # ([100, 64, 4, 4, 32])

        # Vertical scan w
        for k in range(num_steps_h):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T, :, :].reshape(B, D, T // 2, W, 2, -1)  # ([100, 64, 4, 4, 2, 32])
            Xa = Xa[:, :, :T, :, :].reshape(B, D, T // 2, W, 2, -1)  # ([100, 64, 4, 4, 2, 32])

            Xa[:, :, :, :, 1, :].add_(Aa[:, :, :, :, 1, :].mul(Xa[:, :, :, :, 0, :]))
            Aa[:, :, :, :, 1, :].mul_(Aa[:, :, :, :, 0, :])

            Aa = Aa[:, :, :, :, 1, :]  # ([100, 64, 4, 4, 32])
            Xa = Xa[:, :, :, :, 1, :]  # ([100, 64, 4, 4, 32])

        # down sweep h
        for k in range(num_steps_h - 1, -1, -1):
            Aa = A[:, :, 2 ** k - 1: H: 2 ** k, :, :]  # ([100, 64, 4, 4, 32])
            Xa = X[:, :, 2 ** k - 1: H: 2 ** k, :, :]  # ([100, 64, 4, 4, 32])

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1, :, :].add_(Aa[:, :, -1, :, :].mul(Xa[:, :, -2, :, :]))
                Aa[:, :, -1, :, :].mul_(Aa[:, :, -2, :, :])

            Aa = Aa[:, :, :T, :, :].reshape(B, D, T // 2, W, 2, -1)  # ([100, 64, 4, 4, 2, 32])
            Xa = Xa[:, :, :T, :, :].reshape(B, D, T // 2, W, 2, -1)  # ([100, 64, 4, 4, 2, 32])

            Xa[:, :, 1:, :, 0, :].add_(Aa[:, :, 1:, :, 0, :].mul(Xa[:, :, :-1, :, 1, :]))  # ([100, 64, 4, 4, 2, 32])
            Aa[:, :, 1:, :, 0, :].mul_(Aa[:, :, :-1, :, 1, :])  # ([100, 64, 4, 4, 2, 32])

        # down sweep w
        for k in range(num_steps_w - 1, -1, -1):
            Aa = A[:, :, :, 2 ** k - 1: W: 2 ** k, :]  # ([100, 64, 4, 4, 32])
            Xa = X[:, :, :, 2 ** k - 1: W: 2 ** k, :]  # ([100, 64, 4, 4, 32])

            T = 2 * (Xa.size(3) // 2)

            if T < Xa.size(3):
                Xa[:, :, :, -1, :].add_(Aa[:, :, :, -1, :].mul(Xa[:, :, :, -2, :]))  # ([100, 64, 4, 4, 32])
                Aa[:, :, :, -1, :].mul_(Aa[:, :, :, -2, :])  # ([100, 64, 4, 4, 32])

            Aa = Aa[:, :, :, :T, :].reshape(B, D, H, T // 2, 2, -1)  # ([100, 64, 4, 4, 2, 32])
            Xa = Xa[:, :, :, :T, :].reshape(B, D, H, T // 2, 2, -1)  # ([100, 64, 4, 4, 2, 32])

            Xa[:, :, :, 1:, 0, :].add_(Aa[:, :, :, 1:, 0, :].mul(Xa[:, :, :, :-1, 1, :]))  # ([100, 64, 4, 4, 2, 32])
            Aa[:, :, :, 1:, 0, :].mul_(Aa[:, :, :, :-1, 1, :])  # ([100, 64, 4, 4, 2, 32])

    @staticmethod
    def forward(ctx, A_in, X_in):
        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, H, W, D, N)
        X = X_in.clone()  # (B, H, W, D, N)

        # prepare tensors
        A = A.permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)
        X = X.permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)

        # parallel scan
        PScan2D.pscan2d(A, X)

        ctx.save_for_backward(A_in, X)

        return X.permute(0, 2, 3, 1, 4)  # (B, H, W, D, N)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors  # Ain (B, H, W, D, N ) X # (B, D, H, W, N)

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)

        A = torch.cat((A[:, :, :1, :, :], A[:, :, 1:, :, :].flip(2)), dim=2)

        A = torch.cat((A[:, :, :, :1, :], A[:, :, :, 1:, :].flip(3)), dim=3)

        grad_output_b = grad_output_in.permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2).flip(3)

        PScan2D.pscan2d(A, grad_output_b)  # (B, D, H, W, N) * (B, D, H, W, N)

        grad_output_b = grad_output_b.flip(2).flip(3)

        Q = torch.zeros_like(X)  # Q (B, D, H, W, N)

        Q[:, :, 1:, :, :].add_(X[:, :, :-1, :, :] * grad_output_b[:, :, 1:, :, :])
        Q[:, :, :, 1:, :].add_(X[:, :, :, :-1, :] * grad_output_b[:, :, :, 1:, :])

        return Q.permute(0, 2, 3, 1, 4), grad_output_b.permute(0, 2, 3, 1, 4)


pscan2d = PScan2D.apply


class SSM2D(nn.Module):
    def __init__(self, in_channels, dt_rank: int, dim_inner: int, d_state: int):
        super(SSM2D, self).__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.deltaBC_layer = nn.Conv2d(in_channels, dt_rank + 2 * d_state, kernel_size=1, bias=True)
        self.dt_proj_layer = nn.Conv2d(dt_rank, dim_inner, kernel_size=1, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )

        self.D = nn.Parameter(torch.ones(dim_inner), requires_grad=True)

    def forward(self, x, pscan: bool = True):
        A = -torch.exp(self.A_log.float())  # (64, 64)
        D = self.D.float()  # (64)

        deltaBC = self.deltaBC_layer(x)  # (b, 64*3, 7, 7)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=1)
        B = rearrange(B, 'b d h w -> b h w d')
        C = rearrange(C, 'b d h w -> b h w d')
        delta = F.softplus(self.dt_proj_layer(delta))
        delta = rearrange(delta, 'b d h w -> b h w d')  # (100, 7, 7, 64)

        if pscan:
            y = selective_scan_2d(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq_2d(x, delta, A, B, C, D, self.dim_inner, self.d_state)

        return y


def selective_scan_seq_2d(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    x = rearrange(x, 'b d h w -> b h w d')
    b, H, W, d = x.shape

    deltaA = delta.unsqueeze(-1) * A  # (100, 7, 7, 64, 1) * (64, 64)
    A_bar = torch.exp(deltaA)  # (100, 7, 7, 64, 64)

    deltaB = delta.unsqueeze(-1) * B.unsqueeze(3)  # (100, 7, 7, 64, 64)
    BX = deltaB * x.unsqueeze(-1)  # (100, 7, 7, 64, 64)

    # # Initialize h without requires_grad=True
    # h_i_stack = []
    # h_j_stack = []
    # h_i = nn.Parameter(torch.zeros((b, H, dim_inner, d_state), device=A_bar.device), requires_grad=True)
    # h_j = nn.Parameter(torch.zeros((b, W, dim_inner, d_state), device=A_bar.device), requires_grad=True)

    h_ij = nn.Parameter(torch.zeros((b, H, W, dim_inner, d_state), device=A_bar.device))
    h_curij = torch.zeros((b, H, W, dim_inner, d_state), device=A_bar.device)
    h_update = []
    for i in range(H):
        for j in range(W):
            h_pre_i = h_ij[:, i - 1, j, :, :] if i > 0 else torch.zeros_like(h_ij[:, 0, j, :, :])
            h_pre_j = h_ij[:, i, j - 1, :, :] if j > 0 else torch.zeros_like(h_ij[:, i, 0, :, :])
            h_pre_ij = h_ij[:, i - 1, j - 1, :, :] if j > 0 and i > 0 else torch.zeros_like(h_ij[:, 0, 0, :, :])
            h_cur_ij = h_ij[:, i, j, :, :] if j > 0 and i > 0 else torch.zeros_like(h_ij[:, 0, 0, :, :])

            # Use temporary variables to avoid in-place operations
            temp_1 = A_bar[:, i - 1, j, :, :] * h_pre_i + BX[:, i - 1, j, :, :]
            temp_2 = A_bar[:, i, j - 1, :, :] * h_pre_j + BX[:, i, j - 1, :, :]
            temp_3 = A_bar[:, i - 1, j - 1, :, :] * h_pre_ij + BX[:, i - 1, j - 1, :, :]
            temp_4 = A_bar[:, i, j, :, :] * h_cur_ij + BX[:, i, j, :, :]
            # h_ij[:, i, j, :, :] = F.relu(temp_1 + temp_2 + temp_3 + temp_4)

            # Assign the result to h_ij without in-place modification
            h_update.append(F.relu(temp_1 + temp_2 + temp_3 + temp_4))

            h_curij[:, i, j, :, :] = F.relu(temp_1 + temp_2 + temp_3 + temp_4)
    # h_update = torch.stack(h_update, dim=1)
    # h_ij = (rearrange(h_update, 'b (h w) d1 d2 -> b h w d1 d2', h=H, w=W) + h_ij) / 2
    #

    # for i in range(H):
    #     h_prei = A_bar[:, i-1, :, :, :] * h_i + BX[:, i-1, :, :, :]
    #     h_curi = A_bar[:, i, :, :, :] * h_prei + BX[:, i, :, :, :]
    #     h_i_stack.append(F.relu(h_curi))
    #     h_i = F.relu(h_curi)
    #     h_ij[:, i, :, :, :] = h_i
    #
    # for j in range(W):
    #     h_prej = A_bar[:, :, j-1, :, :] * h_j + BX[:, :, j-1, :, :]
    #     h_curj = A_bar[:, :, j, :, :] * h_prej + BX[:, :, j, :, :]
    #     h_j_stack.append(F.relu(h_curj))
    #     h_j = F.relu(h_curj)
    #     h_ij[:, :, j, :, :] = h_j

    # h_i_stack = torch.stack(h_i_stack, dim=1)  # (B, H*W, D, state)
    # h_j_stack = torch.stack(h_j_stack, dim=1)  # (B, H*W, D, state)
    # hs = h_i_stack + h_j_stack

    hs = h_curij
    # Compute the output
    y = (hs @ C.unsqueeze(-1)).squeeze(-1)  # (100, 7, 7, 64, 1) -> (100, 7, 7, 64)
    y = y + D * x  # (100, 7, 7, 64)

    return y  # (100, 7, 7, 64)


def selective_scan_2d(x, delta, A, B, C, D):
    x = rearrange(x, 'b d h w -> b h w d')
    b, H, W, d = x.shape

    deltaA = delta.unsqueeze(-1) * A  # (100, 7, 7, 64, 1) * (64, 64)
    A_bar = torch.exp(deltaA)  # (100, 7, 7, 64, 64)

    deltaB = delta.unsqueeze(-1) * B.unsqueeze(3)  # (100, 7, 7, 64, 64)
    BX = deltaB * x.unsqueeze(-1)  # (100, 7, 7, 64, 64)

    hs = pscan2d(A_bar, BX)

    y = (hs @ C.unsqueeze(-1)).squeeze(-1)  # (100, 7, 7, 64, 1) -> (100, 7, 7, 64)
    y = y + D * x  # (100, 7, 7, 64)

    return y  # (B, H, W, ED)


class VisionEncoderMambaBlock2D(nn.Module):
    def __init__(
            self,
            dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            num_tokens_sqrt: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_tokens_sqrt = num_tokens_sqrt

        self.forward_conv2d = nn.Conv2d(
            dim, dim, kernel_size=1
        )

        self.norm = nn.LayerNorm([dim, num_tokens_sqrt, num_tokens_sqrt])
        self.silu = nn.SiLU()
        self.forward_ssm = SSM2D(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=1)  # point-wise convolution
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1)

        # Softplus
        self.softplus = nn.Softplus()
        # pooling sets
        self.adapool = nn.AdaptiveAvgPool2d((num_tokens_sqrt, num_tokens_sqrt))

    def forward(self, x: torch.Tensor):
        b, d, h, w = x.shape
        # print('vim x input', x.shape)  # (100, 64, 7, 7)
        # Skip connection
        skip = x
        skip = self.adapool(skip)  # b d h w (100, 64, 7, 7)
        # print('vim skip shape', skip.shape)

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z = self.proj1(x)
        x = self.proj2(x)

        # forward conv1d
        x1 = self.process_direction(
            x,
            self.forward_conv2d,
            self.forward_ssm,
        )

        x1 = self.adapool(x1)
        x1 = self.silu(x1)

        # Activation
        z = self.adapool(z)
        z = self.silu(z)

        x1 = z * x1
        # x = z * x

        # Residual connection
        return x1 + skip

    def process_direction(
            self,
            x: Tensor,
            conv2d: nn.Conv2d,
            ssm2d: SSM2D,
    ):
        x = self.softplus(conv2d(x))
        # x = rearrange(x, 'b d h w -> b h w d')
        # print('x after cnn', x.shape)
        x = ssm2d(x)
        x = rearrange(x, 'b h w d -> b d h w')
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class T_Mamaba(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout
                 , seq_length, num_tokens):

        super(T_Mamaba, self).__init__()
        seq_length_sqrt = int(seq_length ** 0.5)
        if seq_length_sqrt ** 2 != seq_length:
            raise ValueError("seq_length must be a perfect square.")
        num_tokens_sqrt = int(num_tokens ** 0.5)
        if num_tokens_sqrt ** 2 != num_tokens:
            raise ValueError("num_tokens must be a perfect square.")

        self.num_patches = seq_length
        self.patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, seq_length_sqrt, seq_length_sqrt))
        self.dropout = nn.Dropout(emb_dropout)
        self.vim2d = VisionEncoderMambaBlock2D(dim=dim, dt_rank=dim, dim_inner=dim, d_state=dim,
                                               num_tokens_sqrt=num_tokens_sqrt)
        # self.rnn2d = MultiDimensionalRecurrentNetwork2D(dim, dim)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                self.vim2d
            )
        self.norm = nn.LayerNorm([dim, num_tokens_sqrt, num_tokens_sqrt])

    def forward(self, img):
        # b d h w = x.shape
        # print('Tmamba img input', img.shape)
        skip = img

        x = img + self.pos_embedding
        # print('Tmamba x input', x.shape)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        # print('Tmamba x layer', x.shape)

        x = self.norm(x + skip)

        x = F.relu(x)

        return x  # (100, 64, 7, 7)


class Mamba2D_block(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout,
                 seq_length, num_tokens):
        super(Mamba2D_block, self).__init__()

        # Initialize Mamba_2 models
        self.T_mamba1 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                 emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)

    def forward(self, x1, x2, x3, x4):
        # Get outputs and regressions from each transformed input
        out_1 = self.T_mamba1(x1)
        out_4 = self.T_mamba1(x4)
        out_2 = self.T_mamba1(x2)
        out_3 = self.T_mamba1(x3)

        return out_1, out_2, out_3, out_4


class Mamba2D_v2(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Parameter, nn.BatchNorm1d, Mamba2D_block,
                          Mamba2D_v2, T_Mamaba, VisionEncoderMambaBlock2D)):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=64, depth=1, emb_dropout=0.):
        super(Mamba2D_v2, self).__init__()
        self.mamba2d = Mamba2D_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                     emb_dropout=emb_dropout, seq_length=((image_size + 1) // 2) ** 2,
                                     num_tokens=((image_size + 1) // 2) ** 2)
        # self.mim_2 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=7 ** 2, num_tokens=5 ** 2)
        # self.mim_3 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=5 ** 2, num_tokens=3 ** 2)
        # self.mim_4 = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                        emb_dropout=emb_dropout, seq_length=3 ** 2, num_tokens=1 ** 2)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            Rearrange("b h w d -> b d h w"),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )

        self.to_latent = nn.Identity()
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)  # Start with equal weights
        self.back = nn.Conv2d(dim, num_classes, 1)

    def WMF(self, *o):
        # Normalize weights to ensure they sum to 1
        k_weights = torch.softmax(self.k_weights, dim=0)

        # Check if the number of outputs matches the number of weights
        assert len(o) == len(k_weights), "The number of outputs and weights must match."

        # Weighted sum of outputs
        O = sum(w * out for w, out in zip(k_weights, o))

        # Average over features
        if O.dim() == 3:
            O_mean = torch.mean(O, dim=1)
        else:
            # O_mean = O[:,:,-1,-1]
            O_mean = reduce(O, 'b d h w -> b d', reduction='mean')
            # O_mean = O.view(O.size(0), -1)
            # O_mean = self.back(O)
        return O_mean

    def forward(self, x):
        x = self.to_patch_embedding(x.squeeze(1))
        # patch split
        b, c, H, W = x.shape
        h_mid = H // 2  # 7//2 = 3
        w_mid = W // 2  # 7//2 = 3
        # print('x input', x.shape)

        # Define the regions using the dynamically calculated coordinates
        part1 = x[:, :, 0:h_mid + 1, 0:w_mid + 1]  # From (0,0) to (h_mid,h_mid)
        part2 = x[:, :, 0:h_mid + 1, w_mid:]  # From (0,w_mid-1) to (h_mid,W-1)
        part3 = x[:, :, h_mid:, 0:w_mid + 1]  # From (h_mid-1,0) to (H-1,w_mid)
        part4 = x[:, :, h_mid:, w_mid:]  # From (h_mid-1,w_mid-1) to (H-1,W-1)
        # print('part n', part1.shape, part2.shape, part3.shape, part4.shape, )

        # Apply the rotations
        part2 = torch.rot90(part2, 1, [2, 3])  # 90 degrees counterclockwise
        part3 = torch.rot90(part3, -1, [2, 3])  # 90 degrees clockwise
        part4 = torch.rot90(part4, 2, [2, 3])  # 180 degrees counterclockwise
        # print('part_rot n', part1.shape, part2.shape, part3.shape, part4.shape, )

        tm1_1, tm2_1, tm3_1, tm4_1 = self.mamba2d(part1, part2, part3, part4)
        # print('tm', tm1_1.shape, tm2_1.shape, tm3_1.shape, tm4_1.shape, )

        O_1 = self.WMF(tm1_1, tm2_1, tm3_1, tm4_1)
        # print('O', O_1.shape)
        # O_1 = self.tanh(O_1)
        # print('tm', O_1.shape)

        # tm1_2, tm2_2, tm3_2, tm4_2 = self.mim_2(tm1_1, tm2_1, tm3_1, tm4_1)

        # O_2 = self.WMF(tm1_2, tm2_2, tm3_2, tm4_2)
        # O_2 = self.tanh(O_2)

        # tm1_3, tm2_3, tm3_3, tm4_3 = self.mim_3(tm1_2, tm2_2, tm3_2, tm4_2)
        #
        # O_3 = self.WMF(tm1_3, tm2_3, tm3_3, tm4_3)
        # O_3 = self.tanh(O_3)
        #
        # tm1_4, tm2_4, tm3_4, tm4_4 = self.mim_4(tm1_3, tm2_3, tm3_3, tm4_3)
        #
        # O_4 = self.WMF(tm1_4, tm2_4, tm3_4, tm4_4)
        # O_4 = self.tanh(O_4)

        return self.mlp_head(self.to_latent(O_1))
