"""Presentation-only CPU progress render from frozen seed0 final test files."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from matplotlib.figure import Figure
from paper.benchmarks2026.grasp import OUTPUT
from paper.benchmarks2026.grasp.data import sha256, write_json
from paper.benchmarks2026.grasp.runner import source_hashes
from paper.benchmarks2026.grasp.visualization import visualize

torch.set_num_threads(1)
destination = Path(__file__).resolve().parent
plan_path = OUTPUT / "pilot_plan_gpu_v2.json"
plan = json.loads(plan_path.read_text())
assert plan["source_sha256"] == source_hashes()
paths = [OUTPUT / "public_within_pilot" / method / "seed0" / f"eval_test_nfe{steps}" / "metrics.json"
         for method, steps in (("mgm", 1), ("rmf_mt", 1), ("rfm", 50), ("rfm", 100))]
reports = [json.loads(path.read_text()) for path in paths]
assert all(r["seed"] == 0 and r["split"] == "test" and r["training_iteration"] == 10000 for r in reports)
label = "PROVISIONAL SEED 0 ONLY; NOT A THREE-SEED AGGREGATE"
original_suptitle = Figure.suptitle


def labeled_title(figure, text, *args, **kwargs):
    return original_suptitle(figure, text + "\n" + label, *args, **kwargs)


figures = []
with patch.object(Figure, "suptitle", labeled_title):
    for record in plan["objects"]:
        category = record["id"].split("_")[0].lower()
        image = visualize(paths, destination / f"{category}_seed0_provisional.png", object_id=record["id"])
        companion = Path(image).with_suffix(".json")
        receipt = json.loads(companion.read_text())
        receipt["presentation_only_label"] = label
        receipt["seed"] = 0
        receipt["aggregate_seed_count"] = 1
        write_json(companion, receipt)
        figures.append({"object_id": record["id"], "png": image, "pdf": str(Path(image).with_suffix(".pdf"))})

rows = []
for report in reports:
    for obj in report["objects"]:
        proxy = obj["static_geometry"]
        reference = obj["heldout_reference_static_geometry"]
        rows.append({"object": obj["object_id"].split("_")[0], "method": report["method"], "nfe": report["nfe"],
            "mmd2": obj["distribution"]["mmd2_unbiased_fixed_chordal_product_kernel"],
            "static_proxy_pose_percent": 100 * proxy["author_static_feasible_pose_fraction"],
            "official_franka_collision_percent": 100 * proxy["official_franka_mesh_collision_fraction"],
            "heldout_reference_static_proxy_percent": 100 * reference["author_static_feasible_pose_fraction"],
            "generated_draws": obj["distribution"]["n_generated"],
            "reference_draws": obj["distribution"]["n_reference"],
            "proxy_draws": len(proxy["per_pose_proxy_pass"]),
            "reference_proxy_source_indices": obj["heldout_reference_proxy_source_indices"]})
assert not torch.cuda.is_initialized()
summary = {"stage": label, "seed": 0, "aggregate_seed_count": 1, "cpu_only": True,
    "cuda_initialized": False, "frozen_source_unchanged": plan["source_sha256"] == source_hashes(),
    "physical_success_rate": None, "selection": "First 12 saved unfiltered draws, no quality selection",
    "reports": [{"path": str(p), "sha256": sha256(p)} for p in paths], "figures": figures, "rows": rows}
write_json(destination / "seed0_metrics.json", summary)
print(json.dumps({"figures": figures, "rows": [{k: v for k, v in row.items() if k != "reference_proxy_source_indices"} for row in rows]}))
