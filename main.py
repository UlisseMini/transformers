import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int

    n_embed: int = 64
    n_heads: int = 4


class SelfAttention(nn.Module):
    """
    Multi-headed attention
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_heads == 0

        # Query, key, value matrices. all in one.
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        bs = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))

        self.n_head = config.n_heads
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.n_embed

        # only place x is used in attention
        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1)

        # split outputs to each head, move hs in front of batch dimension
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # hs = n_embed per head, nh = n_heads

        # q: (B, nh, T, hs)
        # k: (B, nh, hs, T)
        # karpathy wrote (-2, -1) I wrote (-1,-2). they're the same.
        attn = (q @ k.transpose(-1,-2)) / math.sqrt(k.size(-1))
        attn.masked_fill(self.bias[:, :, :T, :T] == 1, float('-inf'))
        attn = attn.softmax(dim=-1)

        # attn: (B, nh, T, T)
        # v:    (B, nh, T, hs)
        # out:  (B, nh, T, hs)
        out = attn @ v
        # reassemble (B, nh, T, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass



if __name__ == '__main__':
    config = ModelConfig(vocab_size=27, block_size=100)
    attend = SelfAttention(config)
    x = torch.randn((1, 50, config.n_embed))
    y = attend(x)
    print(f'{tuple(x.shape)} -> {tuple(y.shape)}')