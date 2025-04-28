
import torch
import torch.nn as nn
from einops import rearrange


# borrowed from
# https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21
# i will try to reimplement the function
# as soon as i understand how it works
# not clear to me how it works yet
def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    """
    Same functionality as RelPosEmb1D

    Args:
        q: a 4d tensor of shape [batch, heads, tokens, dim]
        rel_emb: a 2D or 3D tensor
        of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    if shared_heads:
        emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    else:
        emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        """
        Based on Bottleneck transformer paper
        paper: https://arxiv.org/abs/2101.11605 . Figure 4
        Output: qr^T [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        self.h, self.w = feat_map_size  # height , width
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads)

    def expand_emb(self, r, dim_size):
        # Decompose and unsqueeze dimension
        r = rearrange(r, 'b (h x) i j -> b h x () i j', x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1, -1]  # -1 indicates no expansion
        r = r.expand(expand_index)
        return rearrange(r, 'b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)')

    def forward(self, q):
        """
        Args:
            q: [batch, heads, tokens, dim_head]
        Returns: [ batch, heads, tokens, tokens]
        """
        assert self.total_tokens == q.shape[2], f'Tokens {q.shape[2]} of q must \
        be equal to the product of the feat map size {self.total_tokens} '

        # out: [batch head*w h h]
        r_h = self.emb_w(rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.h)
        return q_r  # q_r transpose in figure 4 of the paper