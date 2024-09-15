import torch
batch_size = 512 # how many sequences will be processed in parallel
n_emb = 128  # number of embeddings
num_heads = 8 # number of self attention heads
inf = 1e9
block_size = 64 # max context length to make predictions
# head_size = 32
dropout = 0.10
num_blocks = 4





learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
