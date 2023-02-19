import torch
from torch import nn
import matplotlib.pyplot as plt

batch_size = 4
block_size = 8
emb_size = 10
x = torch.randn(batch_size, block_size, emb_size)
y = x.view(batch_size, block_size//2, -1)

print("shape of input: {}", x.shape)
print("shape of output: {}", y.shape)
print("two consecutive inputs before concatanation ", x[0,0:2,:])
print("two consecutive inputs after concatanation ", y[0,0,:])

print("two consecutive inputs before concatanation ", x[0,2:4,:])
print("two consecutive inputs after concatanation ", y[0,1,:])

class WaveNetModel(nn.Module):
    def __init__(self, block_size, emb_size, channel_size):
        super(WaveNetModel, self).__init__()
        self.block_size = block_size
        self.emb_size = emb_size

    def forward(self, x):
        B, T, C = x.shape
        i = 0
        while T != 1:
            print(B, T, C)
            x = x.reshape(B, T//2, C*2)
            lm = nn.Linear(C*2, self.emb_size)
            x = lm(x)
            x = x.permute(0, 2, 1)
            bn = nn.BatchNorm1d(self.emb_size)
            x = bn(x)
            x = torch.tanh(x)
            x = x.permute(0, 2, 1)
            B, T, C = x.shape
            i += 1

        return x


model = WaveNetModel(8, 20, 10)
batch_size = 4
block_size = 8
channel_size = 10
x = torch.randn(batch_size, block_size, channel_size)
model = WaveNetModel(8, 100, channel_size)
y = model(x)
print(y.shape)
plt.hist(x.view(1, -1).tolist())
plt.show()




