import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# feedforward

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm1d(dim, affine = True)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Conv1d(dim * mult, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x, mask = None):
        n, device, h, causal = x.shape[2], x.device, self.heads, self.causal

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_all((q, k, v), 'b (h d) n -> (b h) n d', h = h)

        q = q * self.scale

        sim = einsum('b i d, b j d -> b i j', q, k)

        if causal:
            mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b (h d) n', h = h)
        return self.to_out(out)

# main class

class MultistreamTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens,
        max_seq_len,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult))
            ]))

        self.to_logits = nn.Sequential(
            Rearrange('b d n -> b n d'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')

        x = x + pos_emb
        x = rearrange(x, 'b n d -> b d n')

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.to_logits(x)
