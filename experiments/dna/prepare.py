"""Prepare compact, auditable promoter arrays from the Zenodo source files."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from rmgm.dna.data import FLANK, LENGTH


ROOT = Path(__file__).resolve().parents[2]
FILENAMES = {
    "fasta": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    "tss": "FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv.gz",
    "plus": "agg.plus.bw.bedgraph.bw",
    "minus": "agg.minus.bw.bedgraph.bw",
    "blacklist_plus": "fantom.blacklist8.plus.bed.gz",
    "blacklist_minus": "fantom.blacklist8.minus.bed.gz",
}
ENCODE = np.full(256, 4, dtype=np.uint8)
for code, base in enumerate(b"ACGT"):
    ENCODE[base] = code
    ENCODE[base + 32] = code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate(directory: Path) -> dict[str, Path]:
    result = {}
    for key, name in FILENAMES.items():
        matches = list(directory.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {name} under {directory}; found {len(matches)}")
        result[key] = matches[0]
    return result


def read_tsses(path: Path, limit: int) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", newline="") as handle:
        rows = list(itertools.islice(csv.DictReader(handle, delimiter="\t"), limit))
    if not rows or not {"chr", "TSS", "strand"}.issubset(rows[0]):
        raise ValueError("Unexpected TSS table columns")
    return rows


def read_blacklist(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            chrom, start, end, *_ = line.split()
            intervals.setdefault(chrom, []).append((int(start), int(end)))
    merged = {}
    for chrom, items in intervals.items():
        runs: list[list[int]] = []
        for start, end in sorted(items):
            if runs and start <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], end)
            else:
                runs.append([start, end])
        merged[chrom] = ([a for a, _ in runs], [b for _, b in runs])
    return merged


def apply_blacklists(signal: np.ndarray, chrom: str, start: int, blacklists: list[dict]) -> None:
    """RMF's loader zeros both signal tracks for either blacklist."""
    end = start + len(signal)
    for blacklist in blacklists:
        starts, ends = blacklist.get(chrom, ([], []))
        left = bisect.bisect_right(ends, start)
        right = bisect.bisect_left(starts, end)
        for i in range(left, right):
            signal[max(starts[i] - start, 0):min(ends[i] - start, len(signal))] = 0


def prepare(source: Path, output: Path, limit: int = 100000, limit_per_split: int | None = None) -> dict:
    try:
        import pyBigWig
        from pyfaidx import Fasta
    except ImportError as exc:
        raise RuntimeError("Preparation requires pyBigWig and pyfaidx") from exc
    files = locate(source)
    rows = read_tsses(files["tss"], limit)
    groups = {"train": [], "valid": [], "test": []}
    for number, row in enumerate(rows):
        chrom = row["chr"]
        split = "test" if chrom in ("chr8", "chr9") else "valid" if chrom == "chr10" else "train"
        if limit_per_split is None or len(groups[split]) < limit_per_split:
            groups[split].append((number, row))

    genome = Fasta(str(files["fasta"]), sequence_always_upper=True)
    plus, minus = pyBigWig.open(str(files["plus"])), pyBigWig.open(str(files["minus"]))
    blacklists = [read_blacklist(files["blacklist_plus"]), read_blacklist(files["blacklist_minus"])]
    output.mkdir(parents=True, exist_ok=True)
    length = LENGTH + 2 * FLANK
    manifest = {"source": "https://zenodo.org/records/7943307", "assembly": "GRCh38",
                "selection": f"first {limit} rows of sorted FANTOM-CAT TSS table",
                "length": LENGTH, "flank": FLANK, "raw_rows": len(rows), "files": {}, "splits": {},
                "signal": "strand-matched aggregate CAGE BigWig; both blacklist regions zeroed"}
    for split, selected in groups.items():
        labels = np.lib.format.open_memmap(output / f"{split}_labels.npy", mode="w+", dtype=np.uint8,
                                            shape=(len(selected), length))
        signal = np.lib.format.open_memmap(output / f"{split}_signal.npy", mode="w+", dtype=np.float32,
                                            shape=(len(selected), length))
        unknown = 0
        with (output / f"{split}_records.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("source_row", "chr", "TSS", "strand"))
            for j, (number, row) in enumerate(selected):
                chrom, pos, strand = row["chr"], int(row["TSS"]), row["strand"]
                if strand not in ("+", "-"):
                    raise ValueError(f"Bad strand at row {number}: {strand}")
                # The +1 on minus strand matches the official RMF loader.
                start = pos - LENGTH // 2 + (1 if strand == "-" else 0) - FLANK
                end = start + length
                if start < 0 or end > len(genome[chrom]):
                    raise ValueError(f"Window beyond chromosome at row {number}: {chrom}:{start}-{end}")
                bases = genome[chrom][start:end].seq.encode("ascii")
                encoded = ENCODE[np.frombuffer(bases, dtype=np.uint8)]
                if strand == "-":
                    encoded = encoded[::-1].copy()
                    complement = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
                    encoded = complement[encoded]
                track = plus if strand == "+" else minus
                values = np.asarray(track.values(chrom, start, end, numpy=True), dtype=np.float32)
                values = np.nan_to_num(values, nan=0.0)
                apply_blacklists(values, chrom, start, blacklists)
                if strand == "-":
                    values = values[::-1]
                labels[j] = encoded
                signal[j] = values
                unknown += int((encoded == 4).sum())
                writer.writerow((number, chrom, pos, strand))
        labels.flush()
        signal.flush()
        manifest["splits"][split] = {"count": len(selected), "unknown_bases": unknown}
    plus.close()
    minus.close()
    for key, path in files.items():
        manifest["files"][key] = {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    for path in output.iterdir():
        if path.is_file() and path.name != "manifest.json":
            manifest.setdefault("processed_files", {})[path.name] = sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/promoters_rmf10")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.limit, args.limit_per_split)["splits"], indent=2))


if __name__ == "__main__":
    main()
