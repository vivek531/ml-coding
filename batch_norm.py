import torch
from torch import nn

torch.manual_seed(531)

batch_size = 5
feature_size = 10
m = nn.BatchNorm1d(feature_size, eps=1e-03, affine=False)
input = torch.randn(batch_size, feature_size)
print("mean of all values of tensor ->", torch.mean(input))
output = m(input)
print("mean of output tensor across rows is expected to be zero -> ", torch.mean(output, 0))
print("mean of output tensor across cols is not expected to be zero -> ", torch.mean(output, 1))

def custom_batch_norm(x):
    eps=1e-03
    mean = torch.mean(x, 0, keepdims=True)
    var = torch.var(x, 0, keepdims=True)
    x_normalized = torch.div(x - mean, torch.sqrt(var + eps))
    return x_normalized

normalized_output = custom_batch_norm(input)
print("mean of output tensor across rows is expected to be zero -> ", torch.mean(normalized_output, 0))
print("mean of output tensor across cols is not expected to be zero -> ", torch.mean(normalized_output, 1))
print("Is the two batch norm outputs equal:  ", torch.equal(output, normalized_output))

