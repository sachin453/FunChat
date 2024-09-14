import torch
batch_size = 64 # how many sequences will be processed in parallel
n_emb = 64  # number of embeddings
inf = 1e9
block_size = 16 # max context length to make predictions
# head_size = 32





learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
