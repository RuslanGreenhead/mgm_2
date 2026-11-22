"""Small real-format fixture for the full preparation and training path."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.dna.prepare import FILENAMES, prepare
from experiments.dna.train import main as train_main
from rmgm.dna.data import PromoterArrays


ROOT = Path(__file__).resolve().parents[1]


def create_source(directory: Path) -> None:
    import pyBigWig

    rng = np.random.default_rng(4)
    chromosomes = ("chr1", "chr8", "chr10")
    with (directory / FILENAMES["fasta"]).open("w") as handle:
        for chrom in chromosomes:
            bases = "".join(rng.choice(np.array(list("ACGT")), size=3000))
            handle.write(f">{chrom}\n{bases}\n")
    with gzip.open(directory / FILENAMES["tss"], "wt") as handle:
        handle.write("chr\tTSS\tstrand\n")
        for chrom in chromosomes:
            for strand in ("+", "-"):
                handle.write(f"{chrom}\t1500\t{strand}\n")
    for key, value in (("plus", 2.0), ("minus", 3.0)):
        bw = pyBigWig.open(str(directory / FILENAMES[key]), "w")
        bw.addHeader([(chrom, 3000) for chrom in chromosomes])
        for chrom in chromosomes:
            if key == "plus":
                bw.addEntries([chrom], [0], ends=[3000], values=[value])
            else:
                bw.addEntries([chrom, chrom], [0, 1500], ends=[1500, 3000], values=[3.0, 4.0])
        bw.close()
    for key in ("blacklist_plus", "blacklist_minus"):
        with gzip.open(directory / FILENAMES[key], "wt") as handle:
            handle.write("chr1\t1500\t1501\n")


class PipelineTests(unittest.TestCase):
    def test_preparation_and_cli_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, processed, output = root / "raw", root / "processed", root / "run"
            source.mkdir()
            create_source(source)
            manifest = prepare(source, processed, limit=6)
            self.assertEqual({key: value["count"] for key, value in manifest["splits"].items()},
                             {"train": 2, "valid": 2, "test": 2})
            train = PromoterArrays(processed, "train", shift=10)
            valid = PromoterArrays(processed, "valid")
            self.assertEqual(len(train), 2)
            prepared_signal = np.load(processed / "train_signal.npy", mmap_mode="r")
            self.assertEqual(int((prepared_signal[0] == 0).sum()), 1)
            self.assertEqual(int((prepared_signal[1] == 0).sum()), 1)
            _, plus = valid[0]
            _, minus = valid[1]
            self.assertEqual(float(plus[0]), 2.0)
            self.assertEqual(float(minus[0]), 4.0)
            self.assertEqual(float(minus[-1]), 3.0)
            config = {"seed": 1, "epochs": 2, "batch_size": 2, "num_workers": 0,
                      "width": 8, "critic_width": 8, "blocks": 1, "critic_warmup": 1,
                      "critic_steps": 1, "train_shift": 10, "log_every": 1,
                      "checkpoint_every_steps": 1, "eval_every_epochs": 1}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            arguments = ["train.py", "--config", str(config_path), "--data", str(processed),
                         "--output", str(output), "--device", "cpu", "--max-steps", "2", "--test"]
            with patch("sys.argv", arguments):
                train_main()
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            self.assertIn("kmer_pearson", json.loads((output / "test.json").read_text()))


if __name__ == "__main__":
    unittest.main()
