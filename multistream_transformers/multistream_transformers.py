import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# feedforward

class GroupLayerNorm(nn.Module):
    def __init__(self, dim, groups = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.g = nn.Parameter(torch.ones(1, groups, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, groups, dim, 1))

    def forward(self, x):
        x = rearrange(x, 'b (g d) n -> b g d n', g = self.groups)
        std = torch.var(x, dim = 2, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 2, keepdim = True)
        out = (x - mean) / (std + self.eps) * self.g + self.b
        return rearrange(out, 'b g d n -> b (g d) n')

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
        groups = 1
    ):
        super().__init__()
        self.norm = GroupLayerNorm(dim, groups = groups)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        groups = 1
    ):
        super().__init__()
        input_dim = dim * groups
        hidden_dim = dim * mult * groups

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1, groups = groups)
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
        causal = False,
        groups = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.groups = groups
        self.heads = heads
        self.causal = causal
        input_dim = dim * groups
        inner_dim = dim_head * heads * groups

        self.to_qkv = nn.Conv1d(input_dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, input_dim, 1)

    def forward(self, x, mask = None):
        n, device, h, g, causal = x.shape[2], x.device, self.heads, self.groups, self.causal

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_all((q, k, v), 'b (g h d) n -> (b g h) n d', g = g, h = h)

        q = q * self.scale

        sim = einsum('b i d, b j d -> b i j', q, k)

        if causal:
            mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b g h) n d -> b (g h d) n', h = h, g = g)
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
        ff_mult = 4,
        num_streams = 1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_streams = num_streams
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, groups = num_streams), groups = num_streams),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, groups = num_streams), groups = num_streams)
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

        if self.num_streams > 1:
            x = repeat(x, 'b n d -> b n (s d)', s = self.num_streams)

        x = rearrange(x, 'b n d -> b d n')

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        if self.num_streams > 1:
            x = reduce(x, 'b (s d) n -> b d n', 'mean', s = self.num_streams)

        return self.to_logits(x)
