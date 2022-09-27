import math
import string
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
    n_layer: int = 4


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
        # karpathy wrote (-2, -1) I wrote (-1,-2). they work the same
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

        self.attn = SelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.GELU(),
            nn.Linear(4*config.n_embed, config.n_embed),
        )

    def forward(self, x):
        """
        In http://peterbloem.nl/blog/transformers the layer norm is applied after attention,
        but makemore applies it before attention. I'm not sure which is correct. Maybe it doesn't matter.
        x = self.ln_1(self.attn(x) + x)
        x = self.ln_2(self.mlp(x) + x)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """GPT-2"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embed), # positional embeddings
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        # bias=False because final layer norm has a bias. avoid duplicates.
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)


    def forward(self, idx):
        # idx is a LongTensor of shape (B, T) with values in [0, vocab_size).
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        z = tok_emb + pos_emb
        for block in self.transformer.blocks:
            z = block(z)
        z = self.transformer.ln_f(z)
        logits = self.lm_head(z)
        return logits


if __name__ == '__main__':
    vocab = string.printable[:-2]
    in_text = "Hello transformer!"
    encoded = torch.tensor([vocab.index(c) for c in in_text], dtype=torch.long).unsqueeze(0)
    config = ModelConfig(vocab_size=len(vocab), block_size=100)

    print('INPUT', in_text)
    transformer = Transformer(config)
    logits = transformer(encoded)
    print(logits.shape)
    logits = logits.softmax(dim=-1)
    print(logits)
    idx_next = torch.multinomial(logits[0], num_samples=1)
    text_next = ''.join(vocab[i] for i in idx_next)
    print('OUTPUT', text_next)
