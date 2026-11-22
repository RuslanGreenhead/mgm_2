"""Download and verify the original DDSM promoter data archive.

Run with: PYTHONPATH=src python experiments/dna/download.py
The transfer can resume after interruption. Extract only the promoter subtree.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
from pathlib import Path
import tarfile
import urllib.request


URL = "https://zenodo.org/api/records/7943307/files/data.tar.gz/content"
MD5 = "aa53f13ddcf1474962548bc02cbb9b93"
SIZE = 10162295808
ROOT = Path(__file__).resolve().parents[2]
EXTRACT_NAMES = {
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai",
    "FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv.gz",
    "agg.plus.bw.bedgraph.bw.gz",
    "agg.minus.bw.bedgraph.bw.gz",
    "fantom.blacklist8.plus.bed.gz",
    "fantom.blacklist8.minus.bed.gz",
    "best.sei.model.pth.tar.gz",
    "target.sei.names",
}


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60) as response:
        status = response.status
        if offset and status != 206:
            raise RuntimeError("Server did not honor Range; refusing to overwrite partial archive")
        with destination.open("ab" if offset else "wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                print(f"\rDownloaded {output.tell() / 1e9:.2f} GB", end="", flush=True)
    print()


def parallel_download(destination: Path, workers: int = 16, chunks: int = 64) -> None:
    """Resumable byte ranges; each part is checked before assembly."""
    part_directory = destination.with_suffix(destination.suffix + ".parts")
    part_directory.mkdir(parents=True, exist_ok=True)
    chunk_size = (SIZE + chunks - 1) // chunks

    def fetch(index: int) -> int:
        start = index * chunk_size
        end = min(SIZE, start + chunk_size) - 1
        path = part_directory / f"{index:04d}.part"
        have = path.stat().st_size if path.exists() else 0
        if have > end - start + 1:
            raise ValueError(f"Oversized part {path}")
        if have == end - start + 1:
            return index
        requested = start + have
        request = urllib.request.Request(URL, headers={"Range": f"bytes={requested}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {requested}-{end}/"
            ):
                raise RuntimeError(f"Unexpected range response for part {index}")
            with path.open("ab") as output:
                while block := response.read(64 * 1024):
                    output.write(block)
        if path.stat().st_size != end - start + 1:
            raise IOError(f"Incomplete part {index}")
        return index

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(chunks)]
        for count, future in enumerate(as_completed(futures), 1):
            print(f"\rCompleted {count}/{chunks} ranges", end="", flush=True)
            future.result()
    print()
    staging = destination.with_suffix(destination.suffix + ".assembling")
    with staging.open("wb") as output:
        for index in range(chunks):
            with (part_directory / f"{index:04d}.part").open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if staging.stat().st_size != SIZE:
        raise IOError("Assembled archive has unexpected size")
    verify(staging)
    staging.replace(destination)
    for path in part_directory.iterdir():
        path.unlink()
    part_directory.rmdir()


def verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != SIZE:
        raise ValueError(
            f"Archive is incomplete: found {actual_size:,} bytes; expected {SIZE:,} bytes. "
            "Download the complete data.tar.gz before extraction."
        )
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError(f"Archive MD5 mismatch: {digest.hexdigest()}")


def extract_promoter(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive if member.isfile()
                   and "promoter_design/" in member.name
                   and Path(member.name).name in EXTRACT_NAMES]
        if not members:
            raise ValueError("No promoter_design subtree in archive")
        for member in members:
            parts = Path(member.name).parts
            relative = Path(*parts[parts.index("promoter_design"):])
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if target.is_file():
                    print(f"Already extracted {relative}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(member.name)
                with source, target.open("wb") as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                print(f"Extracted {relative}")
    # pyfaidx and pyBigWig need seekable, uncompressed inputs.
    promoter = destination / "promoter_design"
    for compressed in (
        promoter / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        promoter / "agg.plus.bw.bedgraph.bw.gz",
        promoter / "agg.minus.bw.bedgraph.bw.gz",
        promoter / "best.sei.model.pth.tar.gz",
    ):
        target = compressed.with_suffix("")
        if compressed.is_file() and not target.is_file():
            staging = target.with_suffix(target.suffix + ".tmp")
            with gzip.open(compressed, "rb") as source, staging.open("wb") as output:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
            staging.replace(target)
            print(f"Decompressed {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/data.tar.gz")
    parser.add_argument("--extract-to", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="Verify and extract an archive already placed on disk")
    parser.add_argument("--parallel", type=int, default=0, metavar="WORKERS")
    parser.add_argument("--chunks", type=int, default=64)
    args = parser.parse_args()
    if args.verify_only and args.extract_only:
        parser.error("Choose either --verify-only or --extract-only")
    verified_by_parallel = False
    if not (args.verify_only or args.extract_only):
        if args.parallel:
            parallel_download(args.archive, args.parallel, args.chunks)
            verified_by_parallel = True
        else:
            download(args.archive)
    if not verified_by_parallel:
        verify(args.archive)
    if not args.verify_only:
        extract_promoter(args.archive, args.extract_to)


if __name__ == "__main__":
    main()
