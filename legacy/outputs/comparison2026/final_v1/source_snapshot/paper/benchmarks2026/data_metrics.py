"""Locked real-data splits and a single explicitly named evaluator for all methods."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from .common import ManifoldOps

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/comparison2026"
DATASETS=["volcano","earthquake","flood","fire","general","glycine","proline","prepro","rna"]


def read(path):return json.loads(Path(path).read_text())


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def prepare():
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/"paper/experiments"))
    import controlled_suite as old
    for name in DATASETS:
        dest=OUT/"data"/name
        if (dest/"manifest.json").exists():continue
        manifold,dim,data,indices=old.load_data(name)
        source=ROOT/"outputs/paper_icml2026/controlled_independent/datasets"/name
        locked=np.load(source/"split_indices_and_targets.npz")
        ref=read(source/"manifest.json")
        arrays={k:v.numpy() for k,v in data.items()}
        arrays.update({k+"_indices":v.numpy() for k,v in indices.items()})
        arrays.update({k:locked[k] for k in ["val_target","test_target","kernel_reference_train"]})
        dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/"data.npz",**arrays)
        write(dest/"manifest.json",{"name":name,"kind":"sphere" if name in DATASETS[:4] else "torus","dim":dim,
            "counts":{k:len(v) for k,v in data.items()},"data_seed":0,"reference":{k:ref[k] for k in ["bandwidth2","coverage_radius"]},
            "source_manifest_sha256":sha(source/"manifest.json"),"source_arrays_sha256":sha(source/"split_indices_and_targets.npz"),
            "arrays_sha256":sha(dest/"data.npz"),"split_policy":"Existing historically exposed row split, no source-group transfer claim"})
        print(json.dumps({"prepared":name,"dim":dim,"counts":{k:len(v) for k,v in data.items()}}),flush=True)


if __name__=="__main__":prepare()
