import math
import string
import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

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




# ================================
# more stuff stolen from makemore (I do understand it all though!)
# (theoretically these helpers would be builtin to something like pytorch lightning)


@torch.no_grad()
def generate(model, idx, max_new_tokens, tempature=0.5, top_k=None, do_sample=False):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """

    block_size = model.block_size
    for _ in range(max_new_tokens):
        # if the sequence is too long, cut it off
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired tempature
        logits = logits[:, -1, :] / tempature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < v[:, [-1]]] = -float('inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
    


def print_samples(num):
    from tqdm import tqdm

    X_init = torch.zeros(num, 1, dtype=torch.long).to(device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_set.get_output_length() - 1 # -1 bc <START> token
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    samples = []
    for i in tqdm(range(X_samp.size(0))): # iter over batches
        row = X_samp[i, 1:].tolist() # skip <START> token
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_set.decode(row)
        samples.append(word_samp)
    
    print('=' * 80)
    for sample in samples:
        print(sample)
        print('-' * 80)
    print('=' * 80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


# ================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformers")
    parser.add_argument('--input-file', type=str, default='data/names.txt', help='path to input file')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--sample-only', action='store_true', help='only sample from a pretrained model')
    parser.add_argument('--work-dir', '-o', type=str, default='out', help='working directory')
    parser.add_argument('--device', type=str, default='auto', help='device to use')
    parser.add_argument('--top-k', type=int, default=-1, help='top k sampling')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    train_set, test_set = create_datasets(args.input_file)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    config = ModelConfig(vocab_size=train_set.get_vocab_size(), block_size=train_set.get_output_length())
    print(config)
    model = Transformer(config).to(device)

    if args.resume or args.sample_only:
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=10)
        exit()

    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    # training loop
    optim = torch.optim.Adam(model.parameters())
    best_loss = None
    step = 0
    for epoch in range(10000):
        for batch, (x,y) in enumerate(train_loader):
            step += 1
            x,y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if batch % 10 == 0:
                print(f'[{epoch=} {batch=}/{len(train_loader)}] loss {loss:.3f}')

            if step % 100 == 0:
               train_loss = evaluate(model, train_set, batch_size=100, max_batches=10)
               test_loss  = evaluate(model, test_set,  batch_size=100, max_batches=10)
               writer.add_scalar("Loss/train", train_loss, step)
               writer.add_scalar("Loss/test", test_loss, step)
               writer.flush()
               print(f"[{epoch=} {batch=}] train loss: {train_loss} test loss: {test_loss}")
               # save the model to disk if it has improved
               if best_loss is None or test_loss < best_loss:
                   out_path = os.path.join(args.work_dir, "model.pt")
                   print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                   torch.save(model.state_dict(), out_path)
                   best_loss = test_loss