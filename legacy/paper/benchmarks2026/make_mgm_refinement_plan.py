"""Two prospective two-timescale MGM settings, validation only before tests."""
from .data_metrics import OUT,read,write


def main():
    phase="mgm_refinement_v1";jobs=[]
    old=read(OUT/"mgm_calibration_v1/completed.json");assert old["all_completed"]
    for name in ["fire","general","rna"]:
        rows=[r for r in old["jobs"] if r["data"]==name]
        chosen=min(rows,key=lambda r:r["selected_validation_mmd2"])
        candidates=[]
        for folder in (OUT/"mgm_calibration_v1").iterdir():
            if (folder/"completed.json").exists() and (folder/"job.json").exists():
                done=read(folder/"completed.json")
                if done==chosen:candidates.append(read(folder/"job.json")["config"])
        assert len(candidates)==1
        base=candidates[0]
        for i,change in enumerate([{"critic_width":256,"critic_steps":5,"lr_c":8e-4},
                                   {"critic_width":256,"critic_steps":10,"lr_c":4e-4}]):
            jobs.append({"data":name,"method":"mgm","seed":0,"steps":4000,"eval_every":1000,
                "validation_draws":1,"score_test":False,"config":{**base,**change},
                "output":f"outputs/comparison2026/{phase}/{name}_c{i}_s0"})
    write(OUT/"plans"/(phase+".json"),{"phase":phase,"jobs":jobs,
        "rule":"Before final tests: two declared stronger-critic/timescale settings per geometry; preserve common point generator, inherit validation-selected noisy path only. No target-test scoring or inspection."})


if __name__=="__main__":main()
