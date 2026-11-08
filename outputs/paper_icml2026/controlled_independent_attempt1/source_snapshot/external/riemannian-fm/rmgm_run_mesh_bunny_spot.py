"""Run bunny/spot mesh MGM pilot experiments.

The runner keeps the output roots for projection and sphere-map variants
separate so their R-MGM checkpoint directories cannot collide.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ("bunny10", "bunny50", "bunny100", "spot10", "spot50", "spot100")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("projection", "sphere_map", "sphere_map_gmm"),
        default=["projection", "sphere_map"],
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--mesh-numeigs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=400)
    parser.add_argument("--output-base", type=Path, default=Path("../../outputs/rmgm_mesh_bunny_spot_pilot"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    output_base = (SCRIPT_DIR / args.output_base).resolve()
    log_dir = output_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for data_name in args.datasets:
        for variant in args.variants:
            cmd = build_command(args, data_name, variant, output_base)
            label = f"{data_name}_{variant}"
            log_path = log_dir / f"{label}.log"
            started = time.time()
            event = {
                "event": "started",
                "label": label,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "log": str(log_path),
            }
            print(json.dumps(event), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=SCRIPT_DIR,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            elapsed = time.time() - started
            done = {
                "event": "completed" if proc.returncode == 0 else "failed",
                "label": label,
                "returncode": proc.returncode,
                "elapsed_sec": round(elapsed, 1),
                "log": str(log_path),
            }
            print(json.dumps(done), flush=True)
            summary.append(done)
            if proc.returncode != 0 and not args.keep_going:
                break
        else:
            continue
        break

    summary_path = output_base / "queue_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", "path": str(summary_path)}), flush=True)


def build_command(args: argparse.Namespace, data_name: str, variant: str, output_base: Path) -> list[str]:
    common = [
        sys.executable,
        "-B",
        "rmgm_train_real.py",
        "--data",
        data_name,
        "--method",
        "rmgm",
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--latent-dim",
        str(args.latent_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--critic-steps",
        str(args.critic_steps),
        "--lr-g",
        "1e-4",
        "--lr-c",
        "2e-4",
        "--log-every",
        str(args.log_every),
        "--plot-every",
        str(args.plot_every),
        "--eval-samples",
        str(args.eval_samples),
        "--mesh-datadir",
        "data/mesh",
        "--mesh-numeigs",
        str(args.mesh_numeigs),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if variant == "projection":
        return [
            *common,
            "--generator-type",
            "local_gmm",
            "--generator-components",
            "16",
            "--generator-residual-scale",
            "0.10",
            "--generator-kmeans-init",
            "--generator-kmeans-samples",
            "16384",
            "--generator-kmeans-iters",
            "30",
            "--output-dir",
            str(output_base / "projection"),
        ]
    if variant == "sphere_map":
        return [
            *common,
            "--generator-type",
            "sphere_map",
            "--sphere-map-temperature",
            "80",
            "--sphere-map-vertex-stride",
            "1",
            "--output-dir",
            str(output_base / "sphere_map"),
        ]
    if variant == "sphere_map_gmm":
        return [
            *common,
            "--generator-type",
            "sphere_map_gmm",
            "--generator-components",
            "16",
            "--generator-residual-scale",
            "0.18",
            "--generator-kmeans-init",
            "--generator-kmeans-samples",
            "16384",
            "--generator-kmeans-iters",
            "30",
            "--sphere-map-temperature",
            "80",
            "--sphere-map-vertex-stride",
            "1",
            "--output-dir",
            str(output_base / "sphere_map_gmm"),
        ]
    raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    main()
