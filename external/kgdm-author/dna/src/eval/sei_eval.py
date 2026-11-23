# Port of FFM's SeiEval (fisher-flow/src/data/components/promoter_eval.py).
# Changes vs upstream:
#   - `data_dir` is a constructor arg instead of hardcoded "data/promoter_design/".
#   - dropped the Singleton pattern (just instantiate where needed).
#   - `torch.load(..., weights_only=True)` for modern torch.
import os
import re

import pandas as pd
import torch
from torch import Tensor

# tap -> (submodule path, hook kind). reshape: 15360-d pre-classifier conv
# latent (encoder features); chrom: 21907-d post classifier[0]+ReLU (pre-sigmoid).

from .sei import Sei


class NonStrandSpecific(torch.nn.Module):
    """Average Sei predictions for a sequence and its reverse complement.

    This is the 1D, mean-mode behavior of Selene's wrapper. Keeping the
    ``model`` attribute preserves the published checkpoint's state-dict keys
    without importing Selene's unrelated training and genome dependencies.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sequence: Tensor) -> Tensor:
        # Input is [batch, A/C/G/T, positions]. Reverse both DNA axes.
        reverse_complement = torch.flip(sequence, dims=(1, 2))
        return (self.model(sequence) + self.model(reverse_complement)) / 2


def _upgrade_state_dict(state_dict, prefixes=("module.",)):
    """Strip a leading prefix from every state-dict key."""
    pattern = re.compile("^" + "|".join(prefixes))
    return {pattern.sub("", name): param for name, param in state_dict.items()}


class SeiEval:
    """
    Wrap a pretrained Sei (FunctionLab) and expose two operations:
      - get_sei_profile(seq_one_hot): scalar H3K4me3-mean per sequence.
      - eval_sp_mse(pred, target, b_index): per-sample squared error of
        the H3K4me3 mean profile, with batch-indexed target caching.

    Input shape contract:
      seq_one_hot: (B, L=1024, 4) float on any device.
      Sei expects (B, 4, 4096); we pad 1024 → 4096 with 0.25 on both sides.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._sei = NonStrandSpecific(Sei(4096, 21907))
        self._sei_features = pd.read_csv(
            os.path.join(data_dir, "target.sei.names"), sep="|", header=None
        )
        self._sei_loaded = False
        self._sei_cache: dict[int, Tensor] = {}

    def _ensure_loaded(self, device):
        if self._sei_loaded:
            return
        ckpt_path = os.path.join(self.data_dir, "best.sei.model.pth.tar")
        # weights_only=False: the FunctionLab Sei checkpoint contains non-tensor
        # objects (selene training metadata) and trips torch>=2.6's safe loader.
        # The file is trusted (pretrained model from the original Sei release).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._sei.load_state_dict(_upgrade_state_dict(ckpt["state_dict"]))
        self._sei.to(device)
        self._sei.eval()
        self._sei_loaded = True

    @torch.no_grad()
    def get_sei_profile(self, seq_one_hot: Tensor):
        """
        Returns numpy array of shape (B,): mean H3K4me3 activity per sequence.
        """
        self._ensure_loaded(seq_one_hot.device)
        B, _, _ = seq_one_hot.shape
        pad = torch.full(
            (B, 4, 1536), 0.25, device=seq_one_hot.device, dtype=seq_one_hot.dtype
        )
        sei_inp = torch.cat([pad, seq_one_hot.transpose(1, 2), pad], dim=2)
        sei_out = self._sei(sei_inp).cpu().detach().numpy()  # (B, 21907)
        mask = self._sei_features[1].str.strip().values == "H3K4me3"
        return sei_out[:, mask].mean(axis=1)  # (B,)

    @torch.no_grad()
    def get_sei_latent(self, seq_one_hot: Tensor, tap: str = "reshape") -> Tensor:
        """L2-normalized Sei latent used as the drift LOCALITY feature map phi
        (decides which positives are neighbors; see compute_V_sphere_multi's
        loc_dist_*). Frozen, stop-gradient, forward strand only (single pass),
        returned as a torch tensor ON DEVICE for cdist in the train loop.

          tap='reshape': 15360-d pre-classifier conv latent (encoder features).
          tap='chrom':   21907-d post classifier[0]+ReLU (pre-sigmoid chromatin).

        Input contract matches get_sei_profile: (B, L=1024, 4) one-hot float;
        padded 1024 -> 4096 with 0.25 on both sides.
        """
        self._ensure_loaded(seq_one_hot.device)
        model = self._sei.model  # inner Sei (unwrap NonStrandSpecific)
        B = seq_one_hot.shape[0]
        pad = torch.full(
            (B, 4, 1536), 0.25, device=seq_one_hot.device, dtype=seq_one_hot.dtype
        )
        sei_inp = torch.cat([pad, seq_one_hot.transpose(1, 2), pad], dim=2)

        captured: dict[str, Tensor] = {}
        if tap == "reshape":  # input to classifier == reshape_out [B, 15360]
            handle = model.classifier.register_forward_pre_hook(
                lambda m, a: captured.__setitem__("z", a[0])
            )
        elif tap == "chrom":  # output of classifier[1] (ReLU) == [B, 21907]
            handle = model.classifier[1].register_forward_hook(
                lambda m, i, o: captured.__setitem__("z", o)
            )
        else:
            raise ValueError(f"unknown sei tap: {tap!r}")
        try:
            model.forward(sei_inp)  # forward strand only; discard the prediction
        finally:
            handle.remove()

        z = captured["z"]  # [B, d]
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def eval_sp_mse(
        self,
        seq_one_hot: Tensor,
        target: Tensor,
        b_index: int | None = None,
    ):
        """
        Per-sample squared error of the H3K4me3 mean profile.
        Caches the target profile keyed on `b_index` to avoid recomputation.
        """
        if b_index is not None and b_index in self._sei_cache:
            target_prof = self._sei_cache[b_index]
        else:
            target_prof = self.get_sei_profile(target)
            if b_index is not None:
                self._sei_cache[b_index] = target_prof
        pred_prof = self.get_sei_profile(seq_one_hot)
        return (pred_prof - target_prof) ** 2
