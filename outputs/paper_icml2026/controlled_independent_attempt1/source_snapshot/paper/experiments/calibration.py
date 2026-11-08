"""CPU-only empirical training-resampling reference, with no fitted generator."""
import numpy as np
import torch

import controlled_suite as s


def main():
    torch.set_num_threads(1)
    rows, summary = [], []
    for name in s.REAL:
        manifold, dim, data, ref = s.prepare(name)
        n = min(1024, len(data["val_target"]), len(data["test_target"]))
        arrays, case = {}, []
        for draw in range(3):
            seed = 300000 + draw
            g = torch.Generator().manual_seed(seed)
            x = data["train"][torch.randint(len(data["train"]), (n,), generator=g)]
            arrays[f"generated_draw_{draw}"] = x.numpy()
            for split in ("val", "test"):
                row = s.metrics(manifold, x, data[split + "_target"], ref)
                row.update(data=name, method="empirical_train_resampling", draw=draw, draw_seed=seed,
                           split=split, n_training_seeds=0, training_steps=0, training_seconds=0.0)
                rows.append(row)
                case.append(row)
        dest = s.OUT / "calibration" / name
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "samples.npz", **arrays)
        for split in ("val", "test"):
            subset = [r for r in case if r["split"] == split]
            item = {"data": name, "method": "empirical_train_resampling", "split": split,
                    "n_training_seeds": 0, "mc_draws": len(subset)}
            for field in ("mmd2_biased", "symmetric_nn", "coverage"):
                values = [r[field] for r in subset]
                item[field] = {"mean": float(np.mean(values)), "sd_across_mc_draws": float(np.std(values, ddof=1))}
            summary.append(item)
    s.write_json(s.OUT / "calibration_results.json", rows)
    s.write_json(s.OUT / "calibration_summary.json", summary)
    print(f"CPU calibration complete: {len(s.REAL)} datasets, {len(rows)} rows; no training or GPU work")


if __name__ == "__main__":
    main()
