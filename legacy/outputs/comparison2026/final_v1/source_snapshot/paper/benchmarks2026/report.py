"""Build a factual final report from completed, audited comparisons only."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data_metrics import ROOT,OUT,DATASETS,read,write,sha

METHODS=("mgm","rwflow","kgdm","meanflow")
LABELS={"mgm":"MGM","rwflow":"RW-Flow-C","kgdm":"KGDM","meanflow":"Riemannian MeanFlow"}
SCORES=("kernel_mmd2_v","symmetric_nn","minimum_matching_distance","radius_coverage",
        "nn_occupancy_coverage","one_nna","constraint_mean")


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",default="final_v1");args=p.parse_args()
    phase=OUT/args.phase;protocol=read(phase/"protocol.json")
    assert read(phase/"completed.json")["all_completed"] and read(phase/"audit.json")["passed"]
    seeds=[];groups=defaultdict(list)
    for job in protocol["plan"]["jobs"]:
        folder=ROOT/job["output"];done=read(folder/"completed.json");meta=read(folder/"job.json")
        records=read(folder/f"test_step{done['selected_step']}_metrics.json")
        assert len(records)==3 and all(r["model_forward_calls"]==1 for r in records)
        latency=read(folder/"isolated_latency.json");assert latency["isolated"]
        row={"data":job["data"],"method":job["method"],"seed":job["seed"],
            "variant":job["config"].get("variant"),"steps":done["steps"],"selected_step":done["selected_step"],
            "train_seconds":done["train_seconds"],"critic_updates":done["critic_updates"],
            "training_gpu_max_workers":protocol["plan"].get("max_workers",1),
            "model_parameters":meta["model_parameters"],"critic_parameters":meta["critic_parameters"],
            "sampling_batch":latency["batch"],"sampling_seconds_median":float(np.median(latency["gpu_seconds"]))}
        row.update({score:float(np.mean([r[score] for r in records])) for score in SCORES})
        seeds.append(row);groups[(row["data"],row["method"])].append(row)
    summary=[]
    for (data,method),rows in groups.items():
        assert sorted(r["seed"] for r in rows)==[0,1,2]
        item={"data":data,"method":method,"n_seeds":3}
        for key in (*SCORES,"train_seconds","sampling_seconds_median","selected_step"):
            values=[r[key] for r in rows]
            item[key+"_mean"]=float(np.mean(values));item[key+"_sd"]=float(np.std(values,ddof=1))
        summary.append(item)
    dest=OUT/"report";dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [("per_seed.csv",seeds),("summary.csv",summary)]:
        with (dest/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={(r["data"],r["method"]):r for r in summary}
    lines=["# MGM and Recent One-Step Manifold Baselines", "",
        "Completed controlled-scale experiment, not a reproduction of the authors' model sizes or training budgets.","",
        "## Protocol", "",
        f"{len(seeds)} CUDA training jobs on RTX 4090; nine real datasets, four methods, three training seeds.",
        "Every job completed 50,000 generator/field updates. MGM additionally uses 500 critic warmup updates and five or ten critic updates per generator step, chosen on validation; exact counts and critic sizes are in per_seed.csv.",
        "Configurations were chosen on representative validation sets (Fire, General, RNA), then frozen before final tests. Checkpoints, including initialization, were selected only by validation.",
        "Three Monte Carlo test draws are averaged within each training seed; reported SD is sample SD across three seeds, not across the nine draws.",
        "The saved historical row splits do not measure transfer to unseen proteins, times or source groups.","",
        "## Primary Test Score", "",
        "Biased Gaussian embedded kernel MMD squared, fixed train-median bandwidth; lower is better. Values below are multiplied by 1,000. This is not the nearest-neighbor minimum matching distance sometimes also abbreviated MMD.","",
        "| Dataset | MGM | RW-Flow-C | KGDM | Riemannian MeanFlow |", "| --- | ---: | ---: | ---: | ---: |"]
    wins=defaultdict(int)
    for data in DATASETS:
        winner=min(METHODS,key=lambda method:lookup[(data,method)]["kernel_mmd2_v_mean"]);wins[winner]+=1
        entries=[]
        for method in METHODS:
            row=lookup[(data,method)]
            cell=f"{1000*row['kernel_mmd2_v_mean']:.3f} +/- {1000*row['kernel_mmd2_v_sd']:.3f}"
            entries.append("**"+cell+"**" if method==winner else cell)
        lines.append("| "+" | ".join([data,*entries])+" |")
    lines += ["", "Bold identifies the smallest observed mean, not a significance test. A finite-sample biased MMD need not be zero under a correct model.",
        "Observed lowest-mean counts: "+", ".join(f"{LABELS[m]} {wins[m]}/9" for m in METHODS)+".","",
        "## Locked Recipes", "",
        "| Geometry | MGM critic width/steps/LR; path noise | RW epsilon/L/solver | KGDM kernel/step | RMF variant |",
        "| --- | --- | --- | --- | --- |"]
    recipes=read(OUT/"selected_config_v1.json")["groups"]
    for group in ["sphere","torus2","torus7"]:
        c=recipes[group];g=c["mgm"];r=c["rwflow"];k=c["kgdm"]
        lines.append(f"| {group} | {g.get('critic_width',g['width'])}/{g['critic_steps']}/{g['lr_c']}; {g['noise']} | {r['epsilon']}/{r['sinkhorn_iters']}/{r['sinkhorn_solver']} | {k['kernel']}/{k['step_size']} | {c['meanflow']['variant']} |")
    lines += ["", "The tuning grids differ by method and MGM received a declared stronger-critic refinement. These tuning and optimization costs are not compute matched.","",
        "## Additional Metrics and Cost", "",
        "CSV files contain intrinsic symmetric nearest-neighbor distance, directed target-to-generated minimum distance, radius coverage, occupancy coverage, 1-NNA and constraint residuals. For 1-NNA, the reference is near 0.5, not monotonically lower.",
        "Training seconds are per-job wall times under the declared bounded shared-GPU queue; they include warmup and may include contention, but exclude validation. They are not clean per-method training-efficiency measurements. Equal optimizer updates are not equal compute. Parameters and critic-update counts are listed separately.",
        "Sampling times include RNG, one network evaluation and the geometric output map for batches of 1,024. These are measured separately and sequentially only after all training workers exit, with five warmups and 30 measured draws.",
        "See per_seed.csv for actual selected steps, wall times, parameters, critic counts and isolated sampling measurements. Old GPU-contended calibration times are not used for any efficiency claim.","",
        "## Fidelity", "",
        "- MGM, KGDM and RW-Flow-C use the same unfitted, uniform-prior neural point generator (128 width, three affine layers). No train-fitted GMM is used.",
        "- RW-Flow-C is an equation-tested paper reimplementation with an explicitly finite Sinkhorn solver, not verified author code. Equivalent solver implementations were checked on training-only GPU probes. The final iteration count is numerically refined on independent training banks to a relative marginal residual of at most 1e-3; this is not test-based entropy or model tuning. Training logs retain residuals throughout optimization.",
        "- KGDM on the sphere executes the checksum-pinned author's released field; its wrapped-log torus field is our adaptation. Its EMA is a sampling copy, not an extra sampling NFE or distillation teacher.",
        "- Riemannian MeanFlow uses the source-derived native time network, stopped directional derivative target and the validation-selected core/PCGrad variant. Periodic torus features and small-network budgets are explicit adaptations. Its parameter count differs from the common point generator.",
        "- The comparison does not establish convergence at the original authors' much larger scales. Validation curves and solver residuals should accompany any paper claim.","",
        "## SE(3) Public Pilot", "",
        "The separate grasp experiment uses two actual public ACRONYM H5/mesh pairs, with within-object held-out poses. It is not a full unseen-object ACRONYM evaluation. Static mesh feasibility is not physical grasp success, and no dynamic lifting success is inferred from pose MMD or historical FleX labels.",
        "See ../grasp/ for the frozen pilot protocol, completed GPU results and mesh/gripper figures. If that queue is incomplete, its pilot must remain explicitly marked incomplete.","",
        "## Verification and Sources", "",
        "audit.json verifies source/data hashes, counts, selected-score arithmetic, unique final test files, 1 NFE, manifold validity and an independent NumPy recomputation of MMD. Legacy calibration workers eagerly loaded test arrays without using them for loss, scoring or selection; revised refinement/final workers enforce lazy test access. The historical splits were already exposed in earlier work, so this is not a newly blinded benchmark.",
        "- [RW-Flow](https://arxiv.org/html/2609.39271v2)",
        "- [KGDM](https://arxiv.org/html/2605.10727)",
        "- [Riemannian MeanFlow](https://arxiv.org/html/2603.10718v3)",
        "- [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660)","",
        "Method-specific source commits, exact equations and tests are documented in paper/benchmarks2026/*NOTES.md. Figures live in ../visualizations/final_v1/."]
    (dest/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write(dest/"manifest.json",{"phase":args.phase,"source_protocol_sha256":sha(phase/"protocol.json"),
        "audit_sha256":sha(phase/"audit.json"),"per_seed":seeds,"summary":summary,"lowest_mean_counts":dict(wins)})
    print(dest/"comparison.md")


if __name__=="__main__":main()
