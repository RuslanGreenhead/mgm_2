"""Two validation-only LR refinements, explicitly separate from the initial sweep."""
import copy
import json

import torch

import controlled_suite as s
import stabilization as st


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = json.loads(s.Path(__file__).with_name("stabilization_general.json").read_text())
    previous = json.loads((st.DEST / "results.json").read_text())
    destination = s.OUT / "stabilization_general_refinement"
    destination.mkdir(parents=True, exist_ok=True)
    trials = [
        {"id": "periodic_random_mlp_c5_lr1e4", "method": "mgm", "initialization": "random_mlp", "periodic": True, "critic_steps": 5, "candidate": True, "lr_g": 1e-4},
        {"id": "periodic_random_mlp_c5_lr2e4", "method": "mgm", "initialization": "random_mlp", "periodic": True, "critic_steps": 5, "candidate": True, "lr_g": 2e-4},
    ]
    s.write_json(destination / "protocol.json", {"created_at": s.utc(), "trials": trials, "base": base,
                 "reason": "Validation-only initial sweep shows slow cold-start MLP convergence; test two faster G rates, no other changes.",
                 "source_sha256": s.sha(s.Path(__file__)), "base_source_sha256": s.sha(s.Path(st.__file__)),
                 "previous_results_sha256": s.sha(st.DEST / "results.json"), "test_policy": "No test evaluation",
                 "cap": "Exactly two additional seed-0 trials, final step 2000; no iterative expansion in this script"})
    st.DEST = destination
    results = []
    for trial in trials:
        protocol = copy.deepcopy(base)
        protocol["lr_g"] = trial["lr_g"]
        protocol["trials"] = trials
        print(json.dumps({"event": "refinement_start", "trial": trial["id"], "finished": len(results), "planned": 2, "at": s.utc()}), flush=True)
        results.append(st.run_trial(protocol, trial))
        s.write_json(destination / "results.json", results)
        print(json.dumps({"event": "refinement_finished", "finished": len(results), "planned": 2, **results[-1]}), flush=True)
    eligible = {t["id"] for t in base["trials"] if t["candidate"]} | {t["id"] for t in trials}
    selected = min((r for r in previous + results if r["trial"] in eligible), key=lambda r: r["final_validation_mmd2_mean"])
    s.write_json(destination / "selection.json", {"selected_at": s.utc(), "selected_trial": selected["trial"],
                 "rule": "Lowest fixed-final-step validation mean MMD2 across all five periodic random-initialized MGM candidates",
                 "validation_summary": selected, "new_test_evaluations": 0,
                 "scope": "Exploratory seed-0 selection; independent-seed confirmation and locked test assessment remain future work"})
    print(json.dumps({"event": "refined_selection_locked", "trial": selected["trial"], "test_evaluations": 0}), flush=True)


if __name__ == "__main__":
    main()
