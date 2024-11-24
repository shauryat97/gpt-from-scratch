import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

#hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
torch.manual_seed(1337)

input_text_path = '/home/admin/development/swin/gpt-from-scratch/input.txt'

with open(input_text_path,'r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from character to integer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[i] for i in s])

# split the data
data = torch.tensor(encode(text),dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# loading data
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # avergares loss over multiple batches -> less noisy ; for both splits
    model.train()
    return out
    
class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False) #  (C,16)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer ('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] ==0,float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention runnning in parallel"""
    
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out  = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non-linearity"""

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # multiplied 4 (from the paper)
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), # projection
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """ One transformer block : sa + ffwd"""

    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size  = n_embd// n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # layer norm applied before as opposed to vanilla paper implementation apply after
        x = x + self.ffwd(self.ln2(x)) # layer norm applied before as opposed to vanilla paper implementation apply after
        return x

# simple bigram model :  # This model is a simple bigram language model where the embedding vectors are used to predict the next token in a sequence based on the current token.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        # self.sa_head = Head(n_embd) # for now head_size = C , single head self attention
        # self.sa_head = MultiHeadAttention(4, n_embd//4) # 4 self-attention heads of dimension 32/4 = 8.
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head = n_head )])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.lm_head = nn.Linear(n_embd,vocab_size)
    
    def forward(self,idx, targets=None):
        B,T = idx.shape
        # idx and target are (B,T) shaped tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) - batch_size, sequence_length/time(block_size), emb_size = C
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))  #(T,C)
        x = tok_emb+ pos_emb
        # x = self.sa_head(x) # apply one head of self-attention # (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self,idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:,-1,:] # (B,C)
            # apply softmax to get the probability
            probs  = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append the sample index to the running sequence
            idx = torch.cat([idx, idx_next], dim =1 ) # (B,T+1)
        return idx

model = BigramLanguageModel()
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M params')
sys.exit()

# optimizer
optimizer  = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    if iter % eval_interval ==0 :
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb,yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context  = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))