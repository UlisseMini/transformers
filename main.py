import math
import string
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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


    def forward(self, idx, targets=None):
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

        loss = None
        if targets is not None:
            # flatten stuff
            # logits   (B, T, vocab_size)
            # logits'  (B*T, vocab_size)
            # targets' (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss



# ================================

# Shamelessly stolen from makemore, becuase data wrangling is boring. Thanks karpathy!

class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations (see ignore_index=-1)
        return x, y


def create_datasets(input_file, sep='\n'):
    # read and clean up data
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.split(sep)
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)

    # create test and train set
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[test_set_size:]]
    test_words = [words[i] for i in rp[:test_set_size]]

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    return train_dataset, test_dataset


def str_sample(x: list):
    x = x[1:] # remove <START> token
    crop_index = x.index(0) if 0 in x else len(x)
    x = x[:crop_index]
    return train_set.decode(x)


# ================================



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set, test_set = create_datasets('data/names.txt')
    vocab = train_set.chars

    config = ModelConfig(vocab_size=train_set.get_vocab_size(), block_size=train_set.get_output_length())
    print(config)
    transformer = Transformer(config).to(device)
    optim = torch.optim.Adam(transformer.parameters())

    for x,y in train_set:
        in_text = str_sample(x.tolist())
        x,y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        logits, loss = transformer(x, targets=y)
        transformer.zero_grad()
        loss.backward()
        optim.step()

        logits = logits.softmax(dim=-1)
        idx_next = torch.multinomial(logits[0], num_samples=1)
        # TODO: Remove hack; <START> token is not in model outputs, but is in samples.
        text_next = str_sample([0] + idx_next.T[0].tolist())

        print(f'loss {loss:.3f} input {in_text} output {str(text_next)}')