from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any

@torch.jit.script
def pscan2d_optimized(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B, D, H, W, N = A.size()
    num_steps_h = int(torch.log2(torch.tensor(H)).item())
    num_steps_w = int(torch.log2(torch.tensor(W)).item())

    # 上扫描或归约步骤
    Aa, Xa = A, X

    # 组合水平和垂直扫描
    for k in range(max(num_steps_h, num_steps_w)):
        T_h, T_w = 2 * (Xa.size(2) // 2), 2 * (Xa.size(3) // 2)

        if T_h > 0 and T_w > 0:
            Aa = Aa[:, :, :T_h, :T_w, :].reshape(B, D, T_h // 2, T_w // 2, 2, 2, N)
            Xa = Xa[:, :, :T_h, :T_w, :].reshape(B, D, T_h // 2, T_w // 2, 2, 2, N)

            Xa[:, :, :, :, 1, 1].add_(Aa[:, :, :, :, 1, 1].mul(Xa[:, :, :, :, 0, 0]))
            Aa[:, :, :, :, 1, 1].mul_(Aa[:, :, :, :, 0, 0])

            Aa = Aa[:, :, :, :, 1, 1]
            Xa = Xa[:, :, :, :, 1, 1]

    # 组合下扫描
    for k in range(max(num_steps_h, num_steps_w) - 1, -1, -1):
        step = 2 ** k
        h_indices = torch.arange(step - 1, H, step, device=A.device)
        w_indices = torch.arange(step - 1, W, step, device=A.device)

        Aa = A.index_select(2, h_indices).index_select(3, w_indices)
        Xa = X.index_select(2, h_indices).index_select(3, w_indices)

        T_h, T_w = 2 * (Xa.size(2) // 2), 2 * (Xa.size(3) // 2)

        if T_h > 0 and T_w > 0:
            if T_h < Xa.size(2) or T_w < Xa.size(3):
                Xa[:, :, -1, -1].add_(Aa[:, :, -1, -1].mul(Xa[:, :, -2, -2]))
                Aa[:, :, -1, -1].mul_(Aa[:, :, -2, -2])

            Aa = Aa[:, :, :T_h, :T_w].reshape(B, D, T_h // 2, T_w // 2, 2, 2, N)
            Xa = Xa[:, :, :T_h, :T_w].reshape(B, D, T_h // 2, T_w // 2, 2, 2, N)

            Xa[:, :, 1:, 1:, 0, 0].add_(Aa[:, :, 1:, 1:, 0, 0].mul(Xa[:, :, :-1, :-1, 1, 1]))
            Aa[:, :, 1:, 1:, 0, 0].mul_(Aa[:, :, :-1, :-1, 1, 1])

    return X


class PScan2DUnified(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in, X_in):
        A = A_in.clone().permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)
        X = X_in.clone().permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)

        X = pscan2d_optimized(A, X)

        ctx.save_for_backward(A_in, X)

        return X.permute(0, 2, 3, 1, 4)  # (B, H, W, D, N)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors

        A = A_in.clone().permute(0, 3, 1, 2, 4)  # (B, D, H, W, N)
        A = F.pad(A.flip(2).flip(3), (0, 0, 1, 0, 1, 0), mode='reflect')

        grad_output_b = grad_output_in.permute(0, 3, 1, 2, 4).flip(2).flip(3)  # (B, D, H, W, N)

        grad_output_b = pscan2d_optimized(A, grad_output_b)

        grad_output_b = grad_output_b.flip(2).flip(3)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])
        Q[:, :, :, 1:].add_(X[:, :, :, :-1] * grad_output_b[:, :, :, 1:])

        return Q.permute(0, 2, 3, 1, 4), grad_output_b.permute(0, 2, 3, 1, 4)


pscan2d = PScan2DUnified.apply

# 如果使用PyTorch 2.0或更高版本，可以添加以下代码：
# pscan2d = torch.compile(pscan2d)


class SSM2D(nn.Module):
    def __init__(self, in_channels: int, dt_rank: int, dim_inner: int, d_state: int):
        super(SSM2D, self).__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.deltaBC_layer = nn.Conv2d(in_channels, dt_rank + 2 * d_state, kernel_size=1, bias=True)
        self.dt_proj_layer = nn.Conv2d(dt_rank, dim_inner, kernel_size=1, bias=True)

        # 使用更好的初始化方法
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 2, d_state)).repeat(dim_inner, 1))
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x: torch.Tensor, pscan: bool = True) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=1)
        B = B.permute(0, 2, 3, 1)
        C = C.permute(0, 2, 3, 1)
        delta = F.softplus(self.dt_proj_layer(delta))
        delta = delta.permute(0, 2, 3, 1)

        if pscan:
            y = selective_scan_2d(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq_2d(x, delta, A, B, C, D, self.dim_inner, self.d_state)

        return y


def selective_scan_seq_2d(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                          D: torch.Tensor, dim_inner: int, d_state: int) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    b, H, W, d = x.shape

    deltaA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(deltaA)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(3)
    BX = deltaB * x.unsqueeze(-1)

    h_ij = torch.zeros((b, H, W, dim_inner, d_state), device=A_bar.device)

    # 使用向量化操作替代显式循环
    h_pre_i = F.pad(h_ij[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
    h_pre_j = F.pad(h_ij[:, :, :-1], (0, 0, 0, 0, 1, 0, 0, 0))
    h_pre_ij = F.pad(h_ij[:, :-1, :-1], (0, 0, 0, 0, 1, 0, 1, 0))

    temp = (A_bar * h_pre_i +
            A_bar * h_pre_j +
            A_bar * h_pre_ij +
            BX)

    h_ij = F.relu(temp)

    y = (h_ij @ C.unsqueeze(-1)).squeeze(-1)
    y = y + D * x

    return y


def selective_scan_2d(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                      D: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    b, H, W, d = x.shape

    deltaA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(deltaA)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(3)
    BX = deltaB * x.unsqueeze(-1)

    hs = pscan2d(A_bar, BX)

    y = (hs @ C.unsqueeze(-1)).squeeze(-1)
    y.add_(D * x)

    return y


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

        self.forward_conv2d = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm = nn.LayerNorm([dim, num_tokens_sqrt, num_tokens_sqrt])
        self.silu = nn.SiLU()
        self.forward_ssm = SSM2D(dim, dt_rank, dim_inner, d_state)

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        )

        self.softplus = nn.Softplus()
        self.adapool = nn.AdaptiveMaxPool2d((num_tokens_sqrt, num_tokens_sqrt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        skip = self.adapool(x)

        x = self.norm(x)
        proj_out = self.proj(x)
        z, x = proj_out.chunk(2, dim=1)

        x1 = self.process_direction(x, self.forward_conv2d, self.forward_ssm)
        x1 = self.adapool(x1)

        z = self.adapool(z)
        z = self.silu(z)

        x = z * x1

        return x.add_(skip)

    def process_direction(
            self,
            x: torch.Tensor,
            conv2d: Any,
            ssm2d: Any,
    ) -> torch.Tensor:
        x = self.softplus(conv2d(x))
        x = ssm2d(x)
        return x.permute(0, 3, 1, 2)  # 替代 rearrange


class T_Mamba(nn.Module):
    def __init__(self, dim: int, depth: int, emb_dropout: float, image_token_size: int):
        super(T_Mamba, self).__init__()

        self.dim = dim
        self.pos_embedding = nn.Conv2d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(emb_dropout)
        self.vim2d = VisionEncoderMambaBlock2D(dim=dim, dt_rank=dim, dim_inner=dim, d_state=dim,
                                               num_tokens_sqrt=image_token_size)
        self.vim2d_layers = nn.ModuleList([self.vim2d for _ in range(depth)])
        self.norm = nn.LayerNorm([dim, image_token_size, image_token_size])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img + self.pos_embedding(img)
        x = self.dropout(x)

        for vim2d in self.vim2d_layers:
            x = x + vim2d(x)  # 简化残差连接

        x = self.norm(x)
        return F.relu(x)


class Mamba_2d(nn.Module):
    def __init__(self, dim: int, depth: int, emb_dropout: float, image_token_size: int):
        super(Mamba_2d, self).__init__()
        self.mamba = T_Mamba(depth=depth, dim=dim, emb_dropout=emb_dropout, image_token_size=image_token_size)
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 使用批处理版本的 rot90
        rotations = [img, torch.rot90(img, 1, [2, 3]), torch.rot90(img, -1, [2, 3]), torch.rot90(img, 2, [2, 3])]
        outputs = [self.mamba(rot) for rot in rotations]
        return self.WMF(outputs, self.k_weights)

    @staticmethod
    def WMF(outputs: list, k_weights: torch.Tensor) -> torch.Tensor:
        k_weights = F.softmax(k_weights, dim=0)
        assert len(outputs) == k_weights.size(0), "输出和权重的数量必须匹配。"
        O = sum(w * out for w, out in zip(k_weights, outputs))
        return O.mean(dim=1) if O.dim() == 3 else O


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv_block(x)
        x += residual  # Residual connection
        return nn.ReLU(inplace=True)(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, x_n_channels: int, n_classes: int, patch_size: int, dim: int = 64, dropout_rate: float = 0.1):
        super(UNet, self).__init__()

        self.dconv_down1 = ResidualDoubleConv(n_channels, dim)
        self.dconv_down2 = ResidualDoubleConv(x_n_channels, dim)
        self.mamba_down1 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 0)
        self.mamba_down2 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 2)
        self.mamba_down3 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 4)
        self.mamba_down4 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 6)

        self.maxpool = nn.MaxPool2d(3, stride=1)

        self.mamba_up4 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 6)
        self.mamba_up3 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 4)
        self.mamba_up2 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 2)
        self.mamba_up1 = Mamba_2d(depth=1, dim=dim, emb_dropout=dropout_rate, image_token_size=patch_size - 0)

        self.upconv1 = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1)
        self.dconv_up1 = ResidualDoubleConv(dim + dim, dim)

        self.conv_last = nn.Conv2d(dim, n_classes, kernel_size=1)
        self.conv_last_2 = nn.Conv2d(dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

        # Auxiliary MLP head for global features
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes)
        )
        self.aux_loss_weight = 1

    def forward(self, x, x_data = None):
        x = x.squeeze(1)
        x_data = x_data.squeeze(1)
        x_data = self.dconv_down2(x_data)

        # Encoder path
        conv1 = self.dconv_down1(x)

        conv1 = self.mamba_down1(conv1 + x_data)
        x = self.maxpool(conv1)

        conv2 = self.mamba_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.mamba_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.mamba_down4(x)
        x = self.maxpool(conv4)

        # Decoder path
        x = F.relu(self.mamba_up4(self.upconv1(x)))
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up1(x)

        x = F.relu(self.mamba_up3(self.upconv1(x)))
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up1(x)

        x = F.relu(self.mamba_up2(self.upconv1(x)))
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up1(x)

        x = F.relu(self.mamba_up1(self.upconv1(x)))
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        x = self.dropout(x)

        out_seg = self.conv_last(x)
        aux_rec = self.conv_last_2(x)

        # Auxiliary MLP output
        x_global = x.mean(dim=[2, 3])
        # x_global = x[:, :, x.shape[2] // 2, x.shape[3] // 2]
        out_cls = self.mlp_head(x_global)

        return out_seg, out_cls,
