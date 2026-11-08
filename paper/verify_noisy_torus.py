"""Read-only statistical/provenance checks for the completed validation-only sweep."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent
BASE=ROOT/"outputs/paper_icml2026/noisy_torus_study"


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    protocol=read(BASE/"protocol.json")
    results=read(BASE/"results.json")
    assert len(results)==3
    assert {r["noise_amplitude"] for r in results}=={0.,.3,.7}
    assert protocol["config"]["generator_steps"]==10000
    interrupted=BASE.parent/"noisy_torus_study_interrupted_20261023/protocol.json"
    if interrupted.exists():
        earlier=read(interrupted)
        assert earlier["config"]==protocol["config"]
        assert earlier["source_sha256"]==protocol["source_sha256"]
    assert max(protocol["cpu_checks"].values())<1e-10
    for relative,digest in protocol["source_sha256"].items():assert sha(ROOT/relative)==digest,relative
    files={};rows_count=checkpoints=0
    target_hashes=set()
    for result in results:
        folder=BASE/result["trial"]
        assert result["steps"]==10000 and result["critic_updates"]==50500
        assert result["total_optimizer_updates"]==60500 and result["test_evaluations"]==0
        cfg=read(folder/"config.json")
        target_hashes.add(cfg["validation_target_sha256"])
        assert cfg["initial_seam_max_absolute"]<1e-4
        final=read(folder/"final_step10000_metrics.json")
        val=[r["mmd2_biased"] for r in final if r["split"]=="val"]
        assert len(val)==3
        assert math.isclose(np.mean(val),result["final_validation_mmd2_mean"],rel_tol=1e-12)
        assert math.isclose(np.std(val,ddof=1),result["final_validation_mmd2_sd_mc"],rel_tol=1e-12)
        for path in folder.glob("*_metrics.json"):
            rows=read(path);rows_count+=len(rows)
            assert len(rows)==6 and {r["split"] for r in rows}=={"train","val"}
            assert all(r["training_seed"]==0 and np.isfinite(r["mmd2_biased"]) for r in rows)
            files[path.relative_to(BASE).as_posix()]=sha(path)
        for path in folder.glob("*_samples.npz"):
            with np.load(path) as arrays:
                assert set(arrays.files)=={f"generated_draw_{i}" for i in range(3)}
                assert all(np.isfinite(arrays[key]).all() and arrays[key].shape==(1024,2) for key in arrays.files)
            files[path.relative_to(BASE).as_posix()]=sha(path)
        for path in folder.glob("*.pt"):
            state=torch.load(path,map_location="cpu",weights_only=True)
            assert state["optimizer"] and state["critic_optimizer"]
            assert all(k in state for k in ["cuda_rng","cpu_rng","target_rng","latent_rng"])
            assert all(torch.isfinite(t).all() for t in state["model"].values() if t.is_floating_point())
            assert all(torch.isfinite(t).all() for t in state["critic"].values() if t.is_floating_point())
            checkpoints+=1;files[path.relative_to(BASE).as_posix()]=sha(path)
    assert len(target_hashes)==1
    report={"all_checks_passed":True,"completed_trials":3,"training_seeds_per_trial":1,
        "new_test_evaluations":0,"checkpoints_checked":checkpoints,"train_validation_rows":rows_count,
        "amplitudes":[0.,.3,.7],"all_recorded_sources_unchanged":True,"files_sha256":files,
        "cpu_path_checks":protocol["cpu_checks"],"verification_script_sha256":sha(Path(__file__)),
        "interruption_note":"An earlier incomplete deterministic trial was preserved separately and excluded. All three final trials were restarted under the same fixed settings, not selected by scores."}
    (BASE/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="files_sha256"}))


if __name__=="__main__":main()
