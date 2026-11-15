"""Shared sphere/torus metrics from the 2026 locked evaluator."""
import numpy as np
import torch

def direction_distances(ops,x,y):
    if ops.kind=="sphere":
        x=ops.proj(x);y=ops.proj(y)
        return 2*torch.atan2((x[:,None]-y[None]).norm(dim=-1),(x[:,None]+y[None]).norm(dim=-1))
    delta=x[:,None]-y[None]
    return torch.atan2(delta.sin(),delta.cos()).norm(dim=-1)


@torch.no_grad()
def metrics(ops,x,y,reference):
    x=x.detach().cpu().double();y=y.detach().cpu().double()
    ex,ey=ops.embed(x),ops.embed(y);h2=reference["bandwidth2"]
    kxx=torch.exp(-torch.cdist(ex,ex).square()/h2).mean()
    kyy=torch.exp(-torch.cdist(ey,ey).square()/h2).mean()
    kxy=torch.exp(-torch.cdist(ex,ey).square()/h2).mean()
    d=direction_distances(ops,x,y);a,b=d.min(1),d.min(0)
    within_x=direction_distances(ops,x,x);within_y=direction_distances(ops,y,y)
    within_x.fill_diagonal_(float("inf"));within_y.fill_diagonal_(float("inf"))
    wx,wy=within_x.min(1).values,within_y.min(1).values
    correct_x=(wx<a.values).double()+.5*(wx==a.values).double()
    correct_y=(wy<b.values).double()+.5*(wy==b.values).double()
    nna=.5*(correct_x.mean()+correct_y.mean())
    validity=(x.norm(dim=-1)-1).abs().mean() if ops.kind=="sphere" else ((x<0)|(x>=2*np.pi)).any(-1).double().mean()
    return {"kernel_mmd2_v":float((kxx+kyy-2*kxy).clamp_min(0)),"symmetric_nn":float((a.values.mean()+b.values.mean())/2),
        "minimum_matching_distance":float(b.values.mean()),"radius_coverage":float((b.values<=reference["coverage_radius"]).double().mean()),
        "nn_occupancy_coverage":len(a.indices.unique())/len(y),"one_nna":float(nna),"constraint_mean":float(validity),
        "generated_n":len(x),"target_n":len(y)}
