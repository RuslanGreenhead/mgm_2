"""Real ACRONYM grasp experiments; optional dependencies stay in the data subtree."""

from pathlib import Path
import os
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[3]
DEPS = ROOT / "external" / "acronym-data" / "python-deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))

OUTPUT = ROOT / "outputs" / "comparison2026" / "grasp"
