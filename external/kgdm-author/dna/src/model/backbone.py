"""
Non-temporal 20-block dilated 1D CNN for promoter DNA.

Architecturally identical to FFM / RMF / KGD's "PromoterModel" backbone
(20 dilated residual conv blocks, channels=256, kernel=9, dilation pattern
[1,1,4,16,64] repeated 4 times) with every time-conditioning piece removed:
no Gaussian-Fourier time embedding, no per-block `dense(t)` injection, no
time_dependent_weights, no tangent-space projection at the head.

The plus-strand CAGE conditioning signal enters the network via channel
concatenation at the stem — input width = 4 (noise / data) + 1 (signal) = 5.
"""

import torch
import torch.nn as nn

# 4 cycles of the FFM/RMF dilation pattern → 20 residual blocks.
DILATION_PATTERN: tuple[int, ...] = (1, 1, 4, 16, 64) * 4


class ResBlock(nn.Module):
    """`x → x + SiLU(Conv1d(GroupNorm(x)))` with same-length conv."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # "same" padding for odd kernel: (k-1)/2 * dilation
        padding = (kernel_size - 1) // 2 * dilation
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.act = nn.SiLU()

    def forward(self, x, gamma=None, beta=None):
        h = self.norm(x)
        if gamma is not None:  # FiLM: per-channel scale/shift from a global latent
            h = (1.0 + gamma).unsqueeze(-1) * h + beta.unsqueeze(-1)
        return x + self.act(self.conv(h))


class PromoterBackbone(nn.Module):
    """
    Input:
        x      (B, L, 4): noise tensor in train, or any per-position features.
        signal (B, L, 1): plus-strand CAGE conditioning.
    Output:
        (B, L, 4) raw real-valued logits — NO softmax, NO argmax. The drifting
        loss operates in this raw space; one-hot discretization happens only
        at sampling time via argmax.
    """

    def __init__(
        self,
        n_channels: int = 256,
        kernel_size: int = 9,
        in_features: int = 4,
        signal_features: int = 1,
        out_features: int = 4,
        dilation_pattern: tuple[int, ...] = DILATION_PATTERN,
        film_dim: int = 0,
    ):
        super().__init__()
        in_width = in_features + signal_features
        self.stem = nn.Conv1d(
            in_width, n_channels, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.blocks = nn.ModuleList(
            [ResBlock(n_channels, kernel_size, d) for d in dilation_pattern]
        )
        # FiLM: a GLOBAL latent z (B, film_dim) -> per-block per-channel (gamma,beta).
        # Built only when film_dim>0. Last layer: SMALL-std weight + zero bias -> mild,
        # NEAR-identity modulation at init that is still z-DEPENDENT (so z gets gradient
        # from step 1 and is actually used for across-seq variety; a full zero-init
        # would decouple z and risk it being ignored). z modulates globally; the
        # per-position input is signal-only (in_features=0).
        self.film = None
        if film_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(film_dim, 4 * n_channels), nn.SiLU(),
                nn.Linear(4 * n_channels, 2 * n_channels * len(dilation_pattern)),
            )
            nn.init.normal_(self.film[-1].weight, std=0.01)
            nn.init.zeros_(self.film[-1].bias)
        self.head = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n_channels, out_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, signal: torch.Tensor,
                z: torch.Tensor = None) -> torch.Tensor:
        # per-position input: [x, signal]; in global-latent (FiLM) mode x is None
        # so the only per-position input is the signal.
        h = signal if x is None else torch.cat([x, signal], dim=-1)
        h = self.stem(h.transpose(1, 2))
        if z is not None and self.film is not None:
            fp = self.film(z).view(z.shape[0], len(self.blocks), 2, -1)
            for i, block in enumerate(self.blocks):
                h = block(h, fp[:, i, 0], fp[:, i, 1])
        else:
            for block in self.blocks:
                h = block(h)
        h = self.head(h)
        return h.transpose(1, 2)  # back to (B, L, out_features)
