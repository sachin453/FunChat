import torch
from torch import nn
import torch.nn.functional as F
import config


class BiagramLanguageModel(nn.Module):
    def __init__(self , vocab_size):
        super().__init__()
        # each toekn will directly take logits for the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size , config.n_emb)  
        self.pos_embedding_table = nn.Embedding(config.block_size , config.n_emb)
        # self.sa_head = MultiHeadAttention(config.num_heads , int(config.n_emb / config.num_heads))
        # self.feedforward = FeedForward(config.n_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(config.num_blocks)])
        self.ln1 = nn.LayerNorm(config.n_emb)   # final layer norm
        self.lm_head = nn.Linear(config.n_emb , vocab_size)

    def forward(self , idx , targets = None):
        B , T = idx.shape
        """idx and targets are both tensors of (B,T) integers"""
        token_emb = self.token_embedding_table(idx)  # B , T , n_emb
        pos_emb = self.pos_embedding_table(torch.arange(T , device = config.device)) # T , n_emb
        x = token_emb + pos_emb # (B , T , n_emb)
        # x = self.sa_head(x) # applies one head of self attention # (B , T , head_size)
        # x = self.feedforward(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B , T , vocab_size)

        if targets is None:
            loss = None
        else:
            B , T , C = logits.shape
            logits = logits.view(B*T , C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits , targets)

        return logits , loss

    def generate(self, idx , max_new_tokens):
        """idx in (B,T) array of indices in current context"""
        for _ in range(max_new_tokens):
            # crop to get last block_size tokens
            idx_cond = idx[:,-config.block_size:] # (B , T )
            # get the predictions
            logits , loss = self(idx_cond) # (B*T , vocab_size)
            # focus only on the last time step
            logits = logits[:,-1,:] # (B , vocab_size)
            # apply softmax to get probabilities
            probs = nn.functional.softmax(logits , dim=-1) # (B , vocab_size)
            # sample to get the next predicted token
            idx_next = torch.multinomial(probs , num_samples = 1) # (B , 1)
            # add the token to idx so that next token can also be generated
            idx = torch.cat((idx , idx_next) , dim = 1) # (B , T + 1)
        return idx


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self):
        super().__init__()
        head_size = config.n_emb // config.num_heads
        self.key = nn.Linear(config.n_emb , head_size , bias  = False)
        self.query = nn.Linear(config.n_emb , head_size , bias  = False)
        self.value = nn.Linear(config.n_emb , head_size , bias  = False)
        self.register_buffer('tril' , torch.tril(torch.ones((config.block_size , config.block_size))))

    def forward(self , x):
        B , T , C = x.shape
        k = self.key(x)      # (B , T , head_size)
        q = self.query(x)    # (B , T , head_size)
        """compute attention weights (affinities) """
        wei = q @ k.transpose(-2,-1) * C** -0.5   # (B , T , T)
        wei = wei.masked_fill(self.tril[:T,:T]==0 , -float(config.inf))  # (B , T , T)
        wei = torch.nn.functional.softmax(wei , dim = -1)  # (B , T , T)
        v = self.value(x)    # (B , T , head_size)
        out = wei @ v # (B , T , T) @ (B , T , head_size) = (B , T , head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention running in parallel"""
    def __init__(self , num_heads ):
        super().__init__()
        head_size = config.n_emb // num_heads
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_emb , config.n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self , x):
        x = torch.cat([h(x) for h in self.heads] , dim = -1)   # (B , T , num_heads*head_size)  == (B , T , n_emb)
        x = self.proj(x)     # (B , T , n_emb)
        x = self.dropout(x)   # (B , T , n_emb)
        return x              # (B , T , n_emb)


class FeedForward(nn.Module):
    """multi-layer preceptron layer"""
    def __init__(self , n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(config.dropout),
        )

    def forward(self , x):
        return self.net(x)




class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config.num_heads)
        self.feedforward = FeedForward(config.n_emb)
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)

    def forward(self , x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x
        



    