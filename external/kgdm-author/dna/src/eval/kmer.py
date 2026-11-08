# Port of RMF's kmer_freq.py (Riemannian-MeanFlow/src/eval/kmer_freq.py).
# Pure numpy + scipy; the only change is dropping flax.nnx.metrics.Metric
# in favor of a plain class with the same update/compute/reset interface.

import numpy as np
from scipy.stats import pearsonr


def compute_kmer_freq(seq, k: int = 6):
    """
    K-mer counts of one-hot or label-encoded sequences.

    Args:
        seq: (B, L, 4) one-hot or (B, L) integer labels in {0,1,2,3}.
        k:   k-mer size.

    Returns:
        np.ndarray of shape (4^k,) of int64 counts.
    """
    seq = np.asarray(seq)
    if seq.ndim == 3:
        seq = np.argmax(seq, axis=-1)

    windows = np.lib.stride_tricks.sliding_window_view(seq, window_shape=k, axis=-1)
    windows = windows.reshape(-1, k)

    # base-4 encoding: e.g. [0,1,2,3,0,1] -> 0*4^5 + 1*4^4 + ... + 1
    powers = 4 ** np.arange(k - 1, -1, -1)
    kmer_ids = windows @ powers
    return np.bincount(kmer_ids, minlength=4**k)


class KmerFreq:
    """
    Streaming k-mer Pearson correlation between a "true" corpus and a
    "predicted" corpus, both built up batch-by-batch.

    Usage:
        m = KmerFreq(k=6)
        for true_batch, pred_batch in loader:
            m.update(true_batch, pred_batch)
        corr = m.compute()   # scalar Pearson r
    """

    def __init__(self, k: int = 6):
        self.k = k
        self.n_kmers = 4**k
        self.reset()

    def reset(self):
        self.true_counts = np.zeros(self.n_kmers, dtype=np.int64)
        self.pred_counts = np.zeros(self.n_kmers, dtype=np.int64)

    def update(self, true, pred):
        self.true_counts += compute_kmer_freq(true, k=self.k)
        self.pred_counts += compute_kmer_freq(pred, k=self.k)

    def update_with_true_freq(self, true_freq, pred):
        """Update when the true k-mer counts have already been computed."""
        self.true_counts += true_freq
        self.pred_counts += compute_kmer_freq(pred, k=self.k)

    def compute(self) -> float:
        true_total = self.true_counts.sum()
        pred_total = self.pred_counts.sum()
        if true_total == 0 or pred_total == 0:
            return 0.0

        true_freqs = self.true_counts / true_total
        pred_freqs = self.pred_counts / pred_total

        # Only consider k-mers observed in at least one corpus.
        mask = (self.true_counts > 0) | (self.pred_counts > 0)
        if mask.sum() < 2:
            return 0.0

        corr, _ = pearsonr(true_freqs[mask], pred_freqs[mask])
        return float(corr)
