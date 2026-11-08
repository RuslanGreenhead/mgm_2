"""Freeze per-geometry recipes using only completed validation-only jobs."""
from pathlib import Path
from .data_metrics import OUT,read,write,sha


def main():
    solver_probe=read(OUT/"solver_probe.json")
    phases=[OUT/"mgm_calibration_v1",OUT/"calibration_v2",OUT/"mgm_refinement_v1"]
    candidates={};sources=[]
    for phase in phases:
        assert read(phase/"completed.json")["all_completed"]
        protocol=read(phase/"protocol.json")
        sources.append({"phase":phase.name,"protocol_sha256":sha(phase/"protocol.json")})
        for planned in protocol["plan"]["jobs"]:
            folder=Path(planned["output"])
            if not folder.is_absolute():folder=OUT.parent.parent/folder
            row=read(folder/"completed.json");job=read(folder/"job.json")
            assert all(job[key]==value for key,value in planned.items()),f"Job differs from frozen plan: {folder}"
            assert not job["score_test"] and row["test_evaluations"]==0
            group={"fire":"sphere","general":"torus2","rna":"torus7"}[row["data"]]
            candidates.setdefault((group,row["method"]),[]).append((row["selected_validation_mmd2"],folder,job,row))
    groups={};selection=[]
    for (group,method),options in sorted(candidates.items()):
        score,folder,job,row=min(options,key=lambda x:x[0])
        config=job["config"].copy()
        if method=="rwflow":config["sinkhorn_solver"]=solver_probe["group_solvers"][group]
        groups.setdefault(group,{})[method]=config
        selection.append({"group":group,"method":method,"selected_folder":str(folder.relative_to(OUT)),
            "validation_mmd2":score,"selected_step":row["selected_step"],"config":config,
            "candidates":[{"folder":str(f.relative_to(OUT)),"validation_mmd2":s} for s,f,_,_ in options]})
    assert set(groups)=={"sphere","torus2","torus7"}
    assert all(set(v)=={"mgm","rwflow","kgdm","meanflow"} for v in groups.values())
    output=OUT/"selected_config_v1.json"
    assert not output.exists(),"Never silently replace a frozen test protocol"
    write(output,{"generator_steps":50000,"seeds":[0,1,2],"groups":groups,"selection":selection,"sources":sources,
        "solver_probe_sha256":sha(OUT/"solver_probe.json"),
        "rule":"Lowest fixed-probe validation kernel MMD squared; representative fire/T2-general/T7-rna; initial checkpoint eligible; no test inspected.",
        "scope":"Controlled small-backbone comparison; not author-budget reproduction or a convergence guarantee. All final jobs get 50000 generator updates; MGM critic updates counted separately."})
    print(output)


if __name__=="__main__":main()
