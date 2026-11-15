"""Prospective validation-only MGM calibration, not final test evidence."""
from .data_metrics import write,OUT


def main():
    phase="mgm_calibration_v1"
    jobs=[]
    for name in ["fire","general","rna"]:
        for noise in ([0.] if name=="fire" else [0.,.7]):
            ident=f"{name}_noise{noise:g}"
            jobs.append({"data":name,"method":"mgm","seed":0,"steps":4000,"eval_every":1000,"validation_draws":1,"score_test":False,
                "output":f"outputs/comparison2026/{phase}/{ident}",
                "config":{"width":128,"depth":3,"batch_size":256,"latent_dim":16,"lr":2e-4,"lr_c":2e-4,
                    "critic_steps":5,"critic_warmup":500,"harmonics":4,"noise":noise}})
    write(OUT/"plans"/(phase+".json"),{"phase":phase,"jobs":jobs,"precommitted_candidates":"Deterministic vs true-velocity noise0.7 only on tori; fixed common neural generator, no fitted mixture."})


if __name__=="__main__":main()
