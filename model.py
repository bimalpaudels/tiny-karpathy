import torch
import torch.nn as nn
import torch.nn.functional as F

# Reading the data from the file
with open('tiny-input.txt', 'r', encoding='utf-8-sig') as f:
    text = f.read()

# Unique possible characters in the text
chars = sorted(list(set(text)))

# Functions to encode and decode the characters to and from indices
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda txt: [stoi[c] for c in txt]
decode = lambda idx: "".join([itos[i] for i in idx])

# Splitting into training and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
training_data = data[:n]
val_data = data[n:]

# Hyper-parameters for the model
vocab_size = len(chars)  # Number of classes
batch_size = 64  # No. of batches
block_size = 256  # No. of tokens in each context
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 384  # No. of embedding dimensions for each token
learning_rate = 3e-4  # Rate of stepping

max_iters = 5000  # Number of loops in training
eval_iters = 200  # No. of indices to be used to estimate the losses
eval_interval = 500  # No. of iters before calling the estimator function

n_head = 6  # No. of heads in the multi-head attention block
n_layer = 6  # No. of blocks
dropout = 0.2  # Regularization to avoid over-fitting


# Data Loader
def get_batch(split):
    data = training_data if split == 'train' else val_data

    idx = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i + block_size] for i in idx])
    yb = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
    return xb.to(device), yb.to(device)


# Function to estimate the overall loss in both the training and val set
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.mean().item()
        out[split] = losses.mean()
    model.train()
    return out


# Transformers implementation
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)  # B,T,head_size
        k = self.key(x)  # B,T,head_size

        affinity = q @ k.transpose(-2, -1)  # B,T,head_size @ B,head_size,T -> B,T,T
        wei = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)  # B,T,head_size
        out = wei @ v  # B,T,T @ B,T,head_size -> B,T,head_size
        return out


class MultiHead(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # MLP
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)  # Each vocab will have a row of n_embd dim. vector
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)  # Each index is embedded based on the context
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)  # Conveting the embeddings back to vocab_size at the end

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.embedding_table(idx)  # (B,T) embedded returns with B,T,C
        pos_embd = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.ln_head(x)  # (B,T,C) @ (C,vocab_size) -> (B,T,vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens=10):
        for _ in range(max_tokens):
            B, T = idx.shape
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # B,T,C
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=-1)

        return idx


# Model initialization
model = LanguageModel()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for itr in range(max_iters):

    if itr % eval_interval == 0 or itr == max_iters - 1:
        loss = estimate_loss()
        print(f"Iteration: {itr} Training Loss: {loss.get('train'):.4f}, Validation Loss: {loss.get('val'):.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss = loss.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=1000)[0].tolist()
print(decode(context))
