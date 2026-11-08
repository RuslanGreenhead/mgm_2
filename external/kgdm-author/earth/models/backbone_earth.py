from torch import nn
import torch.nn.functional as F


class EarthModel(nn.Module):
    """Smallish residual-free MLP backbone mapping noise -> R^3."""

    def __init__(self, input_dim=3, hidden_dim=128, n_blocks=4, output_dim=3):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.silu(self.input(x))
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.output(x)
