"""Execute pinned author RMF-MT loss algebra and PCGrad, with declared adapters.

Only the loss tail and PCGrad class are loaded from the read-only author clone.
We do NOT run its CUDA-default initializer, SDF encoder, Lightning, or its
apparently inconsistent ascending-time grasp sampler. No source is rewritten.
"""

import ast
import copy
from functools import lru_cache
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.func import jvp
from torch.nn import functional as F

from . import ROOT
from .data import AUTHOR_COMMIT, sha256
from .geometry import hat, path, prior


@lru_cache(maxsize=1)
def load_source():
    root = ROOT / "external/riemannian-meanflow-author/grasp/src/models"
    loss_file, pc_file = root / "lightning.py", root / "pcgrad.py"
    if sha256(loss_file) != "bd14b702da8bf97a62bb9960147811b30c303ec21dbe8584fc9e31ce45106eea":
        raise ValueError("RMF-MT author source differs from the inspected pin")
    if sha256(pc_file) != "908542515fd4b1657a729642d77ae8976f0cbee0a6664695fcccefa0728ee354":
        raise ValueError("RMF-MT PCGrad differs from the inspected pin")
    tree = ast.parse(loss_file.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Lightning")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss_pc")
    # Start at the first angular loss expression, after the author's JVP unpack.
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "rie" for t in n.targets))
    template = ast.parse("def native_parts(self, xt_so3, ut_so3, vt_so3, dudt_so3, ut_r3, vt_r3, dudt_r3, t_expanded, r_expanded, prefix='train'):\n    pass\n").body[0]
    template.body = method.body[start:]
    module = ast.fix_missing_locations(ast.Module(body=[template], type_ignores=[]))
    namespace = {"torch": torch, "F": F}
    exec(compile(module, str(loss_file), "exec"), namespace)
    pc_tree = ast.parse(pc_file.read_text(encoding="utf-8"))
    pc_class = next(n for n in pc_tree.body if isinstance(n, ast.ClassDef) and n.name == "PCGrad")
    for node in pc_class.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [d for d in node.decorator_list if not
                                   (isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and
                                    d.value.id == "torch" and d.attr == "compile")]
    pc_namespace = {"torch": torch, "np": np, "copy": copy, "random": random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pc_class], type_ignores=[])), str(pc_file), "exec"), pc_namespace)
    provenance = {"author_commit": AUTHOR_COMMIT, "loss_file_sha256": sha256(loss_file),
                  "pcgrad_file_sha256": sha256(pc_file), "native_loss_algebra": "compute_loss_pc angular/R3 loss tail, executed unchanged",
                  "native_pcgrad": "PCGrad class executed unchanged except removing the torch.compile decorator; no compilation/child workers",
                  "architecture": "Common local PointNet encoder and six-vector field, NOT author SDF/12-output network",
                  "time_path": "Data(0)->Haar/Gaussian(1), sample backwards; fixes source grasp training/sampler time-pair mismatch",
                  "diagonal_fraction": 0.0, "sigma_min": 0.0,
                  "loss_weights": {"so3_loss_weight": 1., "r3_loss_weight": 3.},
                  "optimizer_hyperparameters": "Common pilot Adam/LR/gradient cap, not native optimizer schedule",
                  "classification": "RMF-MT source-loss adaptation, NOT native-author experiment reproduction"}
    return namespace["native_parts"], pc_namespace["PCGrad"], provenance


def objective_parts(field, data, rng, batch):
    native, _, _ = load_source()
    ids, rd, pd = data.sample(batch, rng)
    rn, pn = prior(batch, rng, data.device)
    t = torch.rand(batch, 1, generator=rng, device=data.device)
    s = torch.rand(batch, 1, generator=rng, device=data.device) * t
    with torch.no_grad():
        rt, pt, v = path(rd, pd, rn, pn, t)
    condition = field.encode(data, ids)

    def ambient(r, p, end):
        u = field(r, p, s, end, condition)
        return r @ hat(u[:, :3]), u[:, 3:]

    (ur, up), (dur, dup) = jvp(ambient, (rt, pt, t),
                              (rt @ hat(v[:, :3]), v[:, 3:], torch.ones_like(t)))
    context = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(so3_loss_weight=1., r3_loss_weight=3.)))
    l1, l2, _ = native(context, rt, ur, rt @ hat(v[:, :3]), dur, up, v[:, 3:], dup, t, s)
    return l1, l2
