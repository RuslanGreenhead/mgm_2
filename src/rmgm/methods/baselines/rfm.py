"""Location of the preserved RFM implementations.

The historical trainer has its own manifm package and protocol. It is kept in
the external tree rather than silently exposed as a compatible 2026 benchmark
method. Experiment runners can resolve its source through this module.
"""

from pathlib import Path


def reference_source(official=False):
    root = Path(__file__).resolve().parents[4]
    dirname = "riemannian-fm-official-clean" if official else "riemannian-fm"
    source = root / "external" / dirname
    if not (source / "train.py").is_file():
        raise FileNotFoundError(source)
    return source
