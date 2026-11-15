"""Run available CPU tests, explicitly skipping omitted Git/data prerequisites."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("modern", "controlled", "all"), default="all")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    loader = unittest.TestLoader()
    suites = []
    if args.suite in ("modern", "all"):
        suites.append(loader.discover(str(ROOT / "legacy/paper/benchmarks2026"), pattern="test_*.py", top_level_dir=str(ROOT / "legacy")))
    if args.suite in ("controlled", "all"):
        sys.path.insert(0, str(ROOT / "legacy/paper/experiments"))
        suites.append(loader.loadTestsFromName("test_controlled_suite"))
    replacements = []
    prepared = unittest.TestSuite()
    for test in flatten(unittest.TestSuite(suites)):
        identity, reason = test.id(), None
        if identity.endswith(".test_author_checkout_commit_is_pinned_without_processes") and not (ROOT / "external/riemannian-meanflow-author/.git/HEAD").exists():
            reason = "Git checkout metadata intentionally omitted; use MANIFEST source hashes and UPSTREAM_PROVENANCE"
        if identity.endswith(".test_split_matches_author") and not (ROOT / "external/riemannian-fm/data/earth_data/fire.csv").is_file():
            reason = "Raw Fire CSV intentionally not included in the code archive"
        if identity.endswith((".test_saved_float32_sphere_duplicates_are_exact_zero", ".test_saved_volcano_radius_matches_normalized_reference")) and not (ROOT / "legacy/outputs/paper_icml2026/datasets/volcano/split_indices_and_targets.npz").is_file():
            reason = "Saved Volcano binary reference arrays intentionally omitted"
        if reason:
            replacements.append({"test": identity, "reason": reason})

            def skip(reason=reason):
                raise unittest.SkipTest(reason)

            prepared.addTest(unittest.FunctionTestCase(skip, description=identity))
        else:
            prepared.addTest(test)
    result = unittest.TextTestRunner(verbosity=2).run(prepared)
    assert not torch.cuda.is_initialized(), "CPU runner unexpectedly initialized CUDA"
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "passed_tests": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
              "skipped_tests": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors),
              "archive_prerequisite_skips": replacements,
              "all_skips": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
              "cuda_initialized": torch.cuda.is_initialized(), "scope": "unit tests only, no experimental training or dataset reproduction"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
