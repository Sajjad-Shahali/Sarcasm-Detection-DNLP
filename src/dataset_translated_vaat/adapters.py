import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()

        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # residual adapter
        return x + self.dropout(self.up(self.activation(self.down(x))))
