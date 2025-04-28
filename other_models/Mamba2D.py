import torch
from multiscan_v3 import snake_flatten
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
import torch.nn.functional as F
import math


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2 ** k - 1: L: 2 ** k]
            Xa = X[:, :, 2 ** k - 1: L: 2 ** k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


def selective_scan(x, delta, A, B, C, D):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (
            hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
            hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):

        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y


class VisionEncoderMambaBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            num_tokens: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_tokens = num_tokens

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
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(2 * dim, dim)

        # Softplus
        self.softplus = nn.Softplus()
        # pooling sets
        self.adapool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x
        skip = self.adapool(rearrange(skip, 'b n d -> b d n'))
        skip = rearrange(skip, 'b d n -> b n d')

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z = self.proj1(x)
        x = self.proj2(x)

        # forward conv1d
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.forward_ssm,
        )
        x1 = self.adapool(rearrange(x1, 'b n d -> b d n'))
        x1 = rearrange(x1, 'b d n -> b n d')
        # x1 = x1 * self.gaussian_decay_mask(x1, 'center', 'index').unsqueeze(-1)
        x1 = self.silu(x1)

        # backward conv1d
        x2 = self.process_direction(
            torch.flip(x, dims=[1]),
            self.backward_conv1d,
            self.backward_ssm,
        )
        x2 = torch.flip(x2, dims=[1])
        x2 = self.adapool(rearrange(x2, 'b n d -> b d n'))
        x2 = rearrange(x2, 'b d n -> b n d')
        # x2 = x2 * self.gaussian_decay_mask(x2, 'center', 'index').unsqueeze(-1)
        x2 = self.silu(x2)

        # Activation
        z = self.adapool(rearrange(z, 'b n d -> b d n'))
        z = rearrange(z, 'b d n -> b n d')
        z = self.silu(z)

        # Matmul
        # x = torch.cat([x1, x2], dim=-1)
        # x = self.proj3(x)
        # x = x * self.gaussian_decay_mask(x, 'center', 'index').unsqueeze(-1)
        # x = self.silu(x)

        x1 = z * x1
        x2 = z * x2
        # x = z * x

        # Residual connection
        return x1 + x2 + skip

    def process_direction(
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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class T_Mamaba(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout
                 , seq_length, num_tokens):
        super(T_Mamaba, self).__init__()
        self.num_patches = seq_length
        self.patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.vim = VisionEncoderMambaBlock(dim=dim, dt_rank=dim, dim_inner=dim, d_state=dim, num_tokens=num_tokens)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                self.vim
            )
        self.norm = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()

    def forward(self, img):
        # print('img', img.shape)
        if img.dim() == 4:
            img = snake_flatten(img)
        else:
            img = img

        if img.size(2) == self.patch_dim:
            x = self.to_patch_embedding(img)
        else:
            x = img

        # print('x00', x.shape)
        x = x + self.pos_embedding  # (100, seq_length, 64)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        x = self.tanh(x)

        return x


class MiM_block(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout,
                 seq_length, num_tokens):
        super(MiM_block, self).__init__()

        # Initialize Mamba_2 models
        self.T_mamba1 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                 emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba2 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba3 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)
        # self.T_mamba4 = T_Mamaba(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
        #                          emb_dropout=emb_dropout, seq_length=seq_length, num_tokens=num_tokens)

    def forward(self, x1, x2, x3, x4):
        # Get outputs and regressions from each transformed input
        out_1 = self.T_mamba1(x1)
        out_4 = self.T_mamba1(x4)
        out_2 = self.T_mamba1(x2)
        out_3 = self.T_mamba1(x3)

        return out_1, out_2, out_3, out_4


class Mamba2D(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Parameter, nn.BatchNorm1d, MiM_block,
                          Mamba2D, T_Mamaba, VisionEncoderMambaBlock)):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def __init__(self, channels, num_classes, image_size, patch_size=1, dim=64, depth=1, emb_dropout=0.):
        super(Mamba2D, self).__init__()
        self.mamba2d = MiM_block(channels, image_size, patch_size=patch_size, dim=dim, depth=depth,
                                 emb_dropout=emb_dropout, seq_length=16,
                                 num_tokens=16)
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
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )

        self.to_latent = nn.Identity()
        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)  # Start with equal weights

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
            O_mean = O
        return O_mean

    def forward(self, x):
        x = x.squeeze(1)
        # patch split
        b, c, H, W = x.shape
        h_mid = H // 2 #7//2 = 3
        w_mid = W // 2 #7//2 = 3
        # print('h w mid', h_mid, w_mid)

        # Define the regions using the dynamically calculated coordinates
        part1 = x[:, :, 0:h_mid + 1, 0:w_mid + 1]  # From (0,0) to (h_mid,h_mid)
        part2 = x[:, :, 0:h_mid + 1, w_mid:W]  # From (0,w_mid-1) to (h_mid,W-1)
        part3 = x[:, :, h_mid:H, 0:w_mid + 1]  # From (h_mid-1,0) to (H-1,w_mid)
        part4 = x[:, :, h_mid:H, w_mid:W]  # From (h_mid-1,w_mid-1) to (H-1,W-1)

        # Apply the rotations
        part2_rot = torch.rot90(part2, 1, [2, 3])  # 90 degrees counterclockwise
        part3_rot = torch.rot90(part3, -1, [2, 3])  # 90 degrees clockwise
        part4_rot = torch.rot90(part4, 2, [2, 3])  # 180 degrees counterclockwise

        # Verify the shape of each part
        # print("Part 1 shape:", part1.shape)
        # print("Part 2 rotated shape:", part2_rot.shape)
        # print("Part 3 rotated shape:", part3_rot.shape)
        # print("Part 4 rotated shape:", part4_rot.shape)

        x_1 = snake_flatten(part1)
        x_2 = snake_flatten(part2_rot)
        x_3 = snake_flatten(part3_rot)
        x_4 = snake_flatten(part4_rot)
        # print('x1', x_1.shape)
        # print('x4', x_4.shape)

        tm1_1, tm2_1, tm3_1, tm4_1 = self.mamba2d(x_1, x_2, x_3, x_4)
        # print('tm',tm1_1.shape)

        O_1 = self.WMF(tm1_1, tm2_1, tm3_1, tm4_1)
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
