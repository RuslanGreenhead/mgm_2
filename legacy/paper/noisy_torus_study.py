"""Validation-only true-velocity randomized-torus diagnostic; never read test metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent
sys.path.insert(0,str(PAPER/"experiments"))
import stabilization as st
from manifm.rmgm.geometry import sample_midpoint_flip_time
from manifm.rmgm.losses import metric_inner

AMPLITUDE=0.


def noisy_path(manifold,x0,x1):
    if manifold.__class__.__name__!="FlatTorus":raise ValueError("This diagnostic is torus-only")
    time=sample_midpoint_flip_time(len(x0),device=x0.device,dtype=x0.dtype)
    delta=manifold.logmap(x0,x1)
    xi=torch.randn_like(delta)
    s=time.s
    x=manifold.projx(x0+s*delta+AMPLITUDE*torch.sin(torch.pi*s)*xi)
    v=delta+AMPLITUDE*torch.pi*torch.cos(torch.pi*s)*xi
    return x,v,time


def critic_loss(manifold,generator,critic,z,x1,normalize_dim=True):
    with torch.no_grad():x,a,time=noisy_path(manifold,generator(z),x1)
    pred=critic(time.t,x)
    loss=metric_inner(manifold,x,pred-a,pred-a).mean()
    if normalize_dim:loss=loss/x1.shape[-1]
    return loss,{"critic_loss":loss.detach()}


def generator_loss(manifold,generator,critic,z,x1,normalize_dim=True):
    x,a,time=noisy_path(manifold,generator(z),x1)
    pred=critic(time.t,x)
    loss=(2*metric_inner(manifold,x,pred,a)-metric_inner(manifold,x,pred,pred)).mean()
    if normalize_dim:loss=loss/x1.shape[-1]
    return loss,{"generator_loss":loss.detach()}


def check():
    from manifm.manifolds.torus import FlatTorus
    torch.set_num_threads(1);torch.manual_seed(5701)
    m=FlatTorus();x=torch.rand(64,2,dtype=torch.float64)*6.28;y=torch.rand_like(x)*6.28
    t=torch.rand(64,1,dtype=torch.float64)*.8+.1;xi=torch.randn_like(x);a=.7
    delta=m.logmap(x,y)
    def path(s):return m.projx(x+s*delta+a*torch.sin(torch.pi*s)*xi)
    p,v=torch.func.jvp(path,(t,),(torch.ones_like(t),))
    exact=delta+a*torch.pi*torch.cos(torch.pi*t)*xi
    reverse=m.projx(y+(1-t)*m.logmap(y,x)+a*torch.sin(torch.pi*(1-t))*xi)
    u=m.logmap(y,x)+a*torch.pi*torch.cos(torch.pi*(1-t))*xi
    errors={"velocity_jvp":float((v-exact).abs().max()),"point_reversal":float(m.logmap(p,reverse).abs().max()),
            "velocity_reversal":float((v+u).abs().max())}
    assert max(errors.values())<1e-10,errors
    return errors


def main():
    global AMPLITUDE
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true")
    parser.add_argument("--steps",type=int,default=10000);args=parser.parse_args()
    checks=check()
    if args.check:print(json.dumps(checks));return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    original_evaluate=st.evaluate
    def isolated_evaluate(*args,**kwargs):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            return original_evaluate(*args,**kwargs)
    st.evaluate=isolated_evaluate
    st.s.critic_loss=critic_loss;st.s.generator_loss=generator_loss
    st.DEST=st.s.OUT/"noisy_torus_study"
    st.DEST.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((PAPER/"experiments/stabilization_general.json").read_text())
    protocol.update(generator_steps=args.steps,lr_g=2e-4,lr_c=2e-4,critic_warmup=500,
                    evaluation_generator_steps=sorted(set([0,1000,2000,5000,args.steps])),
                    selection="Final validation mean only; exploratory seed0 diagnostic; no new test evaluation")
    trials=[{"id":f"noise{amplitude:g}_random_mlp_c5","method":"mgm","initialization":"random_mlp",
             "periodic":True,"critic_steps":5,"candidate":True,"noise_amplitude":amplitude}
            for amplitude in (0.,.3,.7)]
    protocol["trials"]=trials
    sources=[Path(__file__),Path(st.__file__),Path(st.s.__file__)]
    st.s.write_json(st.DEST/"protocol.json",{"created_at":st.s.utc(),"config":protocol,"cpu_checks":checks,
                      "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      "test_policy":"Validation only; historical benchmark exposure disclosed; no efficacy ranking"})
    results=[]
    for trial in trials:
        AMPLITUDE=trial["noise_amplitude"]
        print(json.dumps({"event":"noisy_trial_start","amplitude":AMPLITUDE,"steps":args.steps}),flush=True)
        result=st.run_trial(protocol,trial);result["noise_amplitude"]=AMPLITUDE
        results.append(result);st.s.write_json(st.DEST/"results.json",results)
        print(json.dumps({"event":"noisy_trial_finished",**result}),flush=True)
    best=min(results,key=lambda r:r["final_validation_mmd2_mean"])
    st.s.write_json(st.DEST/"selection.json",{"selected_at":st.s.utc(),"selection":best,
                    "new_test_evaluations":0,"interpretation":"Single-seed exploratory validation diagnostic, not confirmed efficacy."})


if __name__=="__main__":main()
