import torch
from torch import nn


batch_size = 20
input_dim = 30
output_dim = 100
m = nn.Linear(input_dim, output_dim)
input = torch.randn(batch_size, input_dim)
output = m(input)
print(output.size())

weight_matrix = m.weight
biases = m.bias

custom_output = input@weight_matrix.T + biases
print("Both output should be equal: ", torch.equal(custom_output, output))