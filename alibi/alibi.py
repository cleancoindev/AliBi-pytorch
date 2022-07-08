import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from math import log2, floor

def exists(val):
    return val is not None

# residual wrapper

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# pre-normalization wrapper

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward layer with GELU activation function

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout), # optional dropout
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# AliBi Positional Bias

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return bias

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.alibi_pos_biases = AlibiPositionalBias(heads = self.heads)

        # for caching causal mask

        self.register_buffer("mask", None, persistent=False) 

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), 1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, x):
        n, h, device = x.shape[1], self.heads, x.device

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # ALiBi positional bias
        sim = sim + self.alibi_pos_biases(sim)

        # causal mask
        mask_value = -torch.finfo(sim.dtype).max
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn) # Optional dropout

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim = dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# ALiBi Model

class ALiBi(nn.Module):
    def __init__(self, *, num_tokens, dim, depth, dim_head, heads):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(dim, depth, dim_head, heads)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        x = self.token_emb(x)
        x = self.transformer(x)
        logits = self.to_logits(x)
        return logits

if __name__ == "__main__":
    alibi = ALiBi(
        num_tokens = 20000,
        dim = 512,
        depth = 12,
        heads = 8,
        dim_head = 64,
    )

    tokens = torch.randint(0, 20000, (1, 512))
    logits = alibi(tokens) # (1, 512, 20000)
    print(logits.shape)