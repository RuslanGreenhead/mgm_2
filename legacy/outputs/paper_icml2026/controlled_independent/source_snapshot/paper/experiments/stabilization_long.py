"""Two fixed 10k-step validation-only follow-ons, after the independent rerun."""
import argparse
import copy
import json

import torch

import controlled_suite as s
import stabilization as st


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=s.Path, required=True)
    parser.add_argument("--output-root", type=s.Path, required=True)
    args = parser.parse_args()
    root, destination = args.root.resolve(), args.output_root.resolve()
    if not destination.is_relative_to(root / "outputs/paper_icml2026"):
        raise ValueError("Output outside authorized directory")
    s.ROOT, s.REPO, s.OUT = root, root / "external/riemannian-fm", root / "outputs/paper_icml2026"
    st.DEST = destination
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = json.loads(s.Path(__file__).with_name("stabilization_general.json").read_text())
    base.update(generator_steps=10000, lr_g=2e-4,
                evaluation_generator_steps=[0, 100, 500, 1000, 2000, 5000, 10000])
    trials = [
        {"id": "periodic_random_mlp_c5_lr2e4_g10000", "method": "mgm", "initialization": "random_mlp", "periodic": True, "critic_steps": 5, "candidate": True},
        {"id": "periodic_random_gmm_c5_lr2e4_g10000", "method": "mgm", "initialization": "random_gmm", "periodic": True, "critic_steps": 5, "candidate": True},
    ]
    base["trials"] = trials
    destination.mkdir(parents=True, exist_ok=True)
    s.write_json(destination / "protocol.json", {"created_at": s.utc(), "config": base,
                 "source_sha256": s.sha(s.Path(__file__)), "base_source_sha256": s.sha(s.Path(st.__file__)),
                 "test_policy": "Exactly two validation-only seed-0 follow-ons; no test evaluation or stochastic-path sweep"})
    results = []
    for trial in trials:
        print(json.dumps({"event": "long_trial_start", "trial": trial["id"], "finished": len(results), "planned": 2, "at": s.utc()}), flush=True)
        results.append(st.run_trial(copy.deepcopy(base), trial))
        s.write_json(destination / "results.json", results)
        print(json.dumps({"event": "long_trial_finished", "finished": len(results), "planned": 2, **results[-1]}), flush=True)
    selected = min(results, key=lambda r: r["final_validation_mmd2_mean"])
    s.write_json(destination / "selection.json", {"selected_at": s.utc(), "selected_trial": selected["trial"],
                 "rule": "Best fixed-final-step validation MMD2 among these two predefined long trials", "validation_summary": selected,
                 "new_test_evaluations": 0, "note": "Exploratory single seed, not an efficacy or superiority claim"})


if __name__ == "__main__":
    main()
