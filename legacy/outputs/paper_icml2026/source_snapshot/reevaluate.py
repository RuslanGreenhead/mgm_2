"""Reevaluate a supplement checkpoint with independent draws on CPU or CUDA."""
import argparse
import json
import time

import numpy as np
import torch

import controlled_suite as s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=s.Path, required=True)
    parser.add_argument("--output-dir", type=s.Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=200000)
    parser.add_argument("--nfe", type=int, default=128)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(s.OUT.resolve()):
        raise ValueError("All generated outputs must stay under outputs/paper_icml2026")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = s.RealTrainConfig(**ck["config"])
    method = ck["method"]
    manifold, dim, data, indices = s.load_data(cfg.data)
    manifest_path = s.OUT / "datasets" / cfg.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for split in ("train", "val", "test"):
        if s.tensor_sha(data[split]) != manifest["data_tensor_sha256"][split]:
            raise ValueError("Local source data no longer matches the run manifest")
    ref = {k: manifest[k] for k in ("bandwidth2", "coverage_radius")}
    targets = np.load(manifest_path.with_name("split_indices_and_targets.npz"))
    if method == "rfm":
        model = s.field_model(manifold, dim)
    else:
        c = s.asdict(cfg)
        if method in ("mgm_mlp", "mmd_mlp"):
            c.update(generator_type="mlp", torus_generator="mlp")
        model = s.build_generator({"generator": c}, manifold, dim)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(device).eval()
    arrays, rows = {}, []
    n = min(1024, len(targets["test_target"]), len(targets["val_target"]))
    with torch.no_grad():
        for i in range(args.draws):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            x = s.draw(model, manifold, dim, cfg, method, n, args.seed_base + i, device,
                       args.nfe if method == "rfm" else 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            arrays[f"generated_draw_{i}"] = x.cpu().numpy()
            for split in ("val", "test"):
                row = s.metrics(manifold, x, torch.from_numpy(targets[split + "_target"]), ref)
                row.update(data=cfg.data, method=method, step=ck["step"], training_seed=cfg.seed,
                           split=split, draw=i, draw_seed=args.seed_base + i, device=args.device,
                           sampler_nfe=args.nfe if method == "rfm" else 1,
                           sample_seconds=elapsed, checkpoint_sha256=s.sha(args.checkpoint))
                rows.append(row)
    np.savez_compressed(output / "samples.npz", **arrays)
    s.write_json(output / "results.json", rows)
    print(json.dumps({"output": str(output), "rows": len(rows), "checkpoint": str(args.checkpoint)}))


if __name__ == "__main__":
    main()
