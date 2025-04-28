from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


class MAE(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            decoder_dim=64,
            masking_ratio=0.75,
            decoder_depth=2,
            decoder_heads=2,
            decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_in_channels = encoder.num_in_channels
        self.num_in_height = encoder.num_in_height
        self.num_in_width = encoder.num_in_width

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.out_seg = nn.Conv2d(decoder_dim * self.num_in_channels, decoder_dim, kernel_size=1, bias=True)

    def forward(self, img):
        b, c, h, w = img.shape
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens += self.encoder.pos_embedding[:, :(num_patches)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        out_seg = rearrange(decoded_tokens, 'b (nc nh nw) d -> b (nc d) nh nw', nc=self.num_in_channels,
                            nh=self.num_in_height, nw=self.num_in_width)
        out_seg = F.interpolate(out_seg, size=(h, w), mode='bilinear', align_corners=False)
        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches)
        return self.out_seg(out_seg), recon_loss


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
        return x


class ViT3D(nn.Module):
    def __init__(self, *, image_size=49, image_patch_size=7, channel_patch_size=4, num_classes=20, dim=128, depth=1,
                 heads=4,
                 mlp_dim=128, channels=20, x_data_channel=1, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        patch_channel = channel_patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (channels // patch_channel)
        patch_dim = patch_height * patch_width * patch_channel
        self.num_in_channels = channels // patch_channel
        self.num_in_height = image_height // patch_height
        self.num_in_width = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (c pc) (h p1) (w p2) -> b (c h w) (pc p1 p2)', p1=patch_height, p2=patch_width,
                      pc=patch_channel),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.conv3d = nn.Conv3d(self.num_in_channels, 1, kernel_size=1, bias=True)
        self.to_latent = nn.Identity()
        self.bn = nn.LayerNorm([dim, image_height, image_width], eps=1e-6)
        self.out_seg = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.out_rec = nn.Conv2d(dim, channels, kernel_size=1, bias=True)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, num_classes, bias=True)
        )

    def forward(self, hsi):
        img = hsi.squeeze(1)
        b, c, h, w = img.shape
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)

        out_seg = rearrange(x, 'b (nc nh nw) d -> b nc d nh nw', nc=self.num_in_channels, nh=self.num_in_height,
                            nw=self.num_in_width)
        out_seg = self.conv3d(out_seg)
        out_seg = out_seg.squeeze(1)
        out_seg = F.adaptive_avg_pool2d(out_seg, (h, w))
        out_seg = self.to_latent(out_seg)
        out_seg = self.bn(out_seg)
        out_seg = self.out_seg(out_seg)
        # out_cls = out_seg[:, :, h // 2, w // 2]
        # out_cls = self.mlp_head(out_cls)
        return out_seg


class X_Net_3D(nn.Module):
    def __init__(self, *, in_channels = 20, x_in_channels = 1, image_size = 49, dim = 64, num_classes = 20, spa_patch_size = 7, chn_patch_size = 4):
        super().__init__()

        self.initial_proj = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=True)
        # self.initial_proj_x = nn.Conv2d(x_in_channels, dim, kernel_size=1, bias=True)

        self.encoder_down_1 = ViT3D(channels=dim, image_size=image_size, dim=dim, image_patch_size=spa_patch_size, channel_patch_size=chn_patch_size)
        # self.encoder_down_2 = ViT3D(channels=dim, image_size=image_size, dim=dim, image_patch_size=spa_patch_size, channel_patch_size=chn_patch_size)
        # self.encoder_down_3 = ViT3D(channels=dim, image_size=image_size, dim=dim, image_patch_size=spa_patch_size, channel_patch_size=chn_patch_size)
        # self.encoder_down_4 = ViT3D(channels=dim, image_size=image_size, dim=dim, image_patch_size=spa_patch_size, channel_patch_size=chn_patch_size)
        # self.bottleneck = ViT3D(channels=dim, image_size=image_size, dim=dim, image_patch_size=spa_patch_size, channel_patch_size=chn_patch_size)

        self.out_proj = nn.Conv2d(dim, num_classes, kernel_size=1, bias=True)

    def forward(self, hsi, x_data=None):
        hsi_0 = hsi.squeeze(1)
        hsi_0 = self.initial_proj(hsi_0)

        # x_data_0 = x_data.squeeze(1)
        # x_data_0 = self.initial_proj_x(x_data_0)

        hsi_down_1 = self.encoder_down_1(hsi_0)
        # x_data_down_1 = self.encoder_down_1(x_data_0)

        hsi_down_2 = self.encoder_down_1(hsi_down_1)
        # x_data_down_2 = self.encoder_down_2(x_data_down_1)

        hsi_down_3 = self.encoder_down_1(hsi_down_2)
        # x_data_down_3 = self.encoder_down_3(x_data_down_2)

        hsi_down_4 = self.encoder_down_1(hsi_down_3)
        # x_data_down_4 = self.encoder_down_4(x_data_down_3)

        bottleneck_hsi = self.encoder_down_1(hsi_down_4)  # bottleneck
        # bottleneck_x = self.bottleneck(x_data_down_4)

        hsi_up_0 = self.out_proj(bottleneck_hsi)

        out_seg = hsi_up_0
        out_cls = out_seg[:, :, hsi_0.shape[2] // 2, hsi_0.shape[3] // 2]

        return out_seg, out_cls
