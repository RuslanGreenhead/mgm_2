"""Aggregation regressions for a paused run, without GPU or training imports."""
import unittest

import numpy as np

from .partial_report import independent_mmd, summarize, SCORES


class PartialReportTests(unittest.TestCase):
    def row(self, seed, value):
        return {"data": "volcano", "method": "mgm", "seed": seed, **{key: value for key in SCORES}}

    def test_no_imputation_of_missing_seeds(self):
        cell = summarize([self.row(0, .1)])[("volcano", "mgm")]
        self.assertFalse(cell["complete_three_seed"])
        self.assertIsNone(cell["kernel_mmd2_v_seed_sd"])

    def test_sd_is_across_training_seeds(self):
        cell = summarize([self.row(i, v) for i, v in enumerate((.1, .2, .3))])[("volcano", "mgm")]
        self.assertTrue(cell["complete_three_seed"])
        self.assertAlmostEqual(cell["kernel_mmd2_v_mean"], .2)
        self.assertAlmostEqual(cell["kernel_mmd2_v_seed_sd"], .1)

    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):
            summarize([self.row(0, .1), self.row(0, .2)])

    def test_periodic_embedding(self):
        x = np.array([[.1, .4], [1.2, 2.5]])
        self.assertAlmostEqual(independent_mmd(x, x + 2 * np.pi, "torus", 1.5), 0, places=12)

    def test_chunked_mmd_matches_direct(self):
        rng = np.random.default_rng(17)
        x, y = rng.normal(size=(257, 3)), rng.normal(size=(129, 3))
        kernel = lambda a, b: np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / 2).mean()
        expected = kernel(x, x) + kernel(y, y) - 2 * kernel(x, y)
        self.assertAlmostEqual(independent_mmd(x, y, "sphere", 2), expected, places=12)


if __name__ == "__main__":
    unittest.main()
