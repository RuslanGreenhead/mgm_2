"""Predeclare validation probes, then lock the final comparison before testing."""
import argparse
from .data_metrics import read,write,OUT,DATASETS


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["integration_v1","calibration_v2","final_v1"])
    args=p.parse_args();phase=args.phase;jobs=[];workers=1
    common={"width":128,"depth":3,"latent_dim":16,"batch_size":256,"lr":2e-4}
    if phase=="integration_v1":
        candidates={"kgdm":[{}],"meanflow":[{"variant":"rmf_mt"}],"rwflow":[{"epsilon":1.,"sinkhorn_iters":64}]}
        datasets=["fire","general","rna"];steps=100;seeds=[0];test=False;interval=100
    elif phase=="calibration_v2":
        candidates={"kgdm":[{"kernel":"laplace","step_size":.5},{"kernel":"laplace","step_size":1.},
                            {"kernel":"gauss","step_size":1.}],
                    "meanflow":[{"variant":"rmf"},{"variant":"rmf_mt"}],
                    "rwflow":[{"epsilon":e,"flow_step":s,"sinkhorn_iters":128} for e in [.3,1.] for s in [.05,.2]]}
        datasets=["fire","general","rna"];steps=4000;seeds=[0];test=False;interval=1000
    else:
        frozen=read(OUT/"selected_config_v1.json")
        throughput=read(OUT/"throughput_probe.json")
        workers=throughput["recommended_concurrency"]
        datasets=DATASETS;steps=frozen["generator_steps"];seeds=[0,1,2];test=True;interval=5000
        candidates=None
    for name in datasets:
        group="sphere" if name in DATASETS[:4] else "torus2" if name!="rna" else "torus7"
        recipes=candidates if candidates is not None else {method:[cfg] for method,cfg in frozen["groups"][group].items()}
        for method,options in recipes.items():
            for i,cfg in enumerate(options):
                for seed in seeds:
                    ident=f"{name}_{method}_c{i}_s{seed}"
                    job={"data":name,"method":method,"seed":seed,"steps":steps,"eval_every":interval,
                        "validation_draws":1 if not test else 3,"score_test":test,
                        "output":f"outputs/comparison2026/{phase}/{ident}","config":{**common,**cfg}}
                    if test:job["concurrent_training_workers"]=workers
                    jobs.append(job)
    write(OUT/"plans"/(phase+".json"),{"phase":phase,"jobs":jobs,"max_workers":workers,
        "protocol":"Shared 128x3 neural map for MGM/KGDM/RWFlow; native RMF time network. Validation-only grid; no test tuning. No author-budget reproduction claim."})


if __name__=="__main__":main()
