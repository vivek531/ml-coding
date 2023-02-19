import torch
from torch import nn

block_size = 8
batch_size = 4
num_heads = 6
embed_dim = 30

key = torch.randn(batch_size, block_size, embed_dim)
query = torch.randn(batch_size, block_size, embed_dim)
value = torch.randn(batch_size, block_size, embed_dim)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
multihead_attn
attn_output, attn_output_weights = multihead_attn(query, key, value)

print(attn_output[0, 1, :])
print("Shape of the attention output: ", attn_output.shape)
for head_index in range(0, num_heads, 5):
    print("head number {} : ", attn_output[0, 3, head_index*5: (head_index+1)*5])





