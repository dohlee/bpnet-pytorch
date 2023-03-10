import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Reduce, Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

class BPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 25, padding='same'),
            nn.ReLU(),
            *[Residual(nn.Sequential(nn.Conv1d(64, 64, 3, padding='same', dilation=2**i), nn.ReLU())) for i in range(1, 10)]
        )

        self.profile_head = nn.Sequential(
            nn.ConvTranspose1d(64, 2, 25, padding=12),  # padding=12 works like padding='same' here.
            Rearrange('b c l -> b l c'),
        )

        self.total_count_head = nn.Sequential(
            Reduce('b c l -> b c', 'mean'),  # Global average pooling.
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)

        return {
            'x': x,
            'profile': self.profile_head(x),
            'total_count': self.total_count_head(x),
        }

if __name__ == '__main__':
    model = BPNet()
    
    x = torch.randn(1, 4, 5000, requires_grad=True)
    out = model(x)

    print(out['x'].shape)  # (1, 64, 5000)
    print(out['profile'].shape)  # (1, 5000, 2)
    print(out['total_count'].shape)  # (1, 2)

    loss = out['x'][:, 0, 2500].sum()
    loss.backward()

    print((x.grad != 0.0).sum() / 4)  # (1, 4, 3000)