"""Numerical audits of the exact/approximate geometric interpolation conventions."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"external/riemannian-fm"))
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus
from manifm.manifolds.spd import SPD
from manifm.manifolds.hyperbolic import PoincareBall
from manifm.rmgm.geometry import midpoint_path
from manifm.rmgm.losses import metric_inner


def run():
    torch.set_num_threads(1)
    torch.manual_seed(5801)
    dtype=torch.float64
    n=64
    cases=[]
    sphere=Sphere()
    x=sphere.projx(torch.randn(n,3,dtype=dtype))
    y=sphere.projx(torch.randn(n,3,dtype=dtype))
    cases.append(("sphere",sphere,x,y,True))
    torus=FlatTorus()
    cases.append(("torus",torus,torch.rand(n,3,dtype=dtype)*6.28,
                  torch.rand(n,3,dtype=dtype)*6.28,True))
    spd=SPD()
    a=torch.randn(n,2,2,dtype=dtype)*.3
    b=torch.randn(n,2,2,dtype=dtype)*.3
    x=spd.vectorize(a@a.transpose(-1,-2)+torch.eye(2,dtype=dtype)*1.5)
    y=spd.vectorize(b@b.transpose(-1,-2)+torch.eye(2,dtype=dtype)*2)
    cases.append(("spd",spd,x,y,True))
    ball=PoincareBall()
    cases.append(("hyperbolic",ball,torch.randn(n,2,dtype=dtype)*.12,
                  torch.randn(n,2,dtype=dtype)*.12,False))
    rows=[]
    for name,manifold,x,y,is_exact in cases:
        s=.1+.8*torch.rand(n,1,dtype=dtype)
        p,v=midpoint_path(manifold,x,y,s)
        q,w=midpoint_path(manifold,y,x,1-s)
        eps=1e-6
        lo,_=midpoint_path(manifold,x,y,s-eps)
        hi,_=midpoint_path(manifold,x,y,s+eps)
        numerical=manifold.proju(p,(hi-lo)/(2*eps))
        # Periodic coordinates require local differences across their seam.
        if name=="torus": numerical=manifold.logmap(lo,hi)/(2*eps)
        vel_error=(numerical-v).norm(dim=-1).max().item()
        point_error=(p-q).norm(dim=-1).max().item()
        reverse_error=metric_inner(manifold,p,v+w,v+w).clamp_min(0).sqrt().max().item()
        row={"geometry":name,"dtype":"float64","pairs":n,
             "max_point_reversal_ambient":point_error,
             "max_velocity_reversal_metric":reverse_error,
             "max_velocity_vs_fd_ambient":vel_error,
             "target_is_exact_derivative":is_exact}
        assert point_error<1e-7,(name,row)
        assert reverse_error<1e-5,(name,row)
        assert vel_error<(1e-5 if is_exact else 1e-3),(name,row)
        rows.append(row)
    a=torch.logspace(-4,-.001,200,dtype=dtype)
    d=2*a*a/(1+a*a)
    assert torch.all(a*a<=d+1e-14)
    assert torch.all(d<=(1+a*a)/2+1e-14)
    output={"seed":5801,"geometric_audits":rows,
            "two_point_transport_bound_cases":len(a),
            "note":"Numerical implementation checks, not a replacement for the proofs."}
    destination=ROOT/"outputs/paper_icml2026/geometry_audit.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__=="__main__":run()
