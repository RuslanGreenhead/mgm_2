"""Training loop for the promoter-DNA kernel-gradient-drift model.

One backbone, one drift loss, multi-temperature sum, coherent (activity-
stratified) batches. Six geometry cells

    ambient | simplex | sphere   x   no-grad | grad-form

plus the spectral Matern kernel-score drift on (S^3_+)^L, are all selected by a
handful of booleans on `cfg` (see `src.main`). The forward pass is identical
across cells:

    noise (B,L,4) + signal (B,L,1) --backbone--> logits (B,L,4)
    x = (sphere-embed | softmax | raw)
    loss = drift_loss(x, real_onehot, temps, dist_norm, grad_form)   # multi-T

Model selection is on the chromosome-10 validation split via three in-memory
EMA snapshots (best paper-score, lowest Sei-MSE, highest k-mer); each is given
ONE final eval on the held-out test chromosomes (chr8/9). Nothing is selected on
test. See the README for the objective and the reported numbers.
"""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PromoterDataset
from src.data.cluster_sampler import CoherentBatchSampler
from src.eval.kmer import KmerFreq
from src.eval.sei_eval import SeiEval
from src.model.backbone import PromoterBackbone
from src.model.drift_core import (drift_loss, drift_loss_sphere,
                                   drift_loss_sphere_spectral)
from src.model.spectral_kernel import matern_weights_s3
from src.utils.logger import JsonlWriter, make_run_dir, setup_logger

BASES = ("A", "C", "G", "T")
SEQ_LEN, ALPHA = 1024, 4


# ----------------------------- utilities -----------------------------

def seed_everything(seed: int, strict: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cudnn.deterministic makes dilated 1D conv ~100x slower on H100; opt-in only.
    torch.backends.cudnn.deterministic = bool(strict)
    torch.backends.cudnn.benchmark = not strict


def worker_init(worker_id: int):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def lr_lambda(step: int, warmup: int) -> float:
    return 1.0 if warmup <= 0 else min(step / warmup, 1.0)


def total_grad_norm(model) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().norm()) ** 2
    return s ** 0.5


def pairwise_hamming_labels(labels, k_pairs: int = 256, rng=None) -> float:
    rng = rng or np.random.default_rng(123)
    n = len(labels)
    if n < 2:
        return float("nan")
    vals = []
    for _ in range(min(k_pairs, n * (n - 1) // 2)):
        i, j = rng.choice(n, size=2, replace=False)
        vals.append(float((labels[i] != labels[j]).mean()))
    return float(np.mean(vals))


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.state_dict().items()
                       if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.state_dict().items():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, target):
        sd = target.state_dict()
        for n, v in self.shadow.items():
            sd[n].copy_(v)


# ------------------------------- eval --------------------------------

@torch.no_grad()
def evaluate(backbone, ds, sei, device, val_batch, num_workers):
    """Sample on the split's real conditioning signals; pooled 6-mer correlation,
    per-position accuracy, pairwise Hamming, Sei-MSE (paired vs permuted floor)
    and activity stats, plus a signal/noise sensitivity probe. Always evaluates
    the FULL split (the RMF/SOTA-native eval set)."""
    backbone.eval()
    loader = DataLoader(ds, batch_size=val_batch, shuffle=False,
                        num_workers=num_workers, drop_last=False)

    kmer = KmerFreq(k=6)
    gen_labels_all, gt_labels_all = [], []
    gen_prof_all, real_prof_all = [], []
    pos_acc_sum, pos_n = 0.0, 0
    first_batch = None

    for seq, sig in loader:
        seq = seq.to(device).float()
        sig = sig[..., :1].to(device).float()
        if first_batch is None:
            first_batch = (seq, sig)
        B, L, C = seq.shape
        x_in = torch.randn(B, L, C, device=device)
        idx = backbone(x_in, sig).argmax(-1)            # (B, L)
        samp = F.one_hot(idx, num_classes=4).to(seq.dtype)

        kmer.update(seq.cpu().numpy(), samp.cpu().numpy())
        gen_labels_all.append(idx.cpu().numpy())
        gt_labels_all.append(seq.argmax(-1).cpu().numpy())
        pos_acc_sum += float((idx == seq.argmax(-1)).float().sum())
        pos_n += B * L
        if sei is not None:
            gen_prof_all.append(sei.get_sei_profile(samp))
            real_prof_all.append(sei.get_sei_profile(seq))

    gen_labels = np.concatenate(gen_labels_all, 0)
    gt_labels = np.concatenate(gt_labels_all, 0)
    base = np.bincount(gen_labels.reshape(-1), minlength=4).astype(float)
    base /= max(base.sum(), 1)

    out = {
        "val_kmer_corr": kmer.compute(),
        "val_pos_acc": pos_acc_sum / max(pos_n, 1),
        "val_hamming": pairwise_hamming_labels(gen_labels),
        "val_gt_hamming": pairwise_hamming_labels(gt_labels),
        "n_eval": int(len(gen_labels)),
        **{f"val_base_{b}": float(base[i]) for i, b in enumerate(BASES)},
    }

    if sei is not None:
        gen_prof = np.concatenate(gen_prof_all, 0)
        real_prof = np.concatenate(real_prof_all, 0)
        correct = float(np.mean((gen_prof - real_prof) ** 2))
        rng = np.random.default_rng(0)
        rand = float(np.mean([
            np.mean((gen_prof - real_prof[rng.permutation(len(real_prof))]) ** 2)
            for _ in range(20)
        ]))
        out.update({
            "val_sei_mse": correct,
            "val_sei_random": rand,
            "val_sei_ratio": correct / rand if rand > 0 else 0.0,
            "val_gen_act_mean": float(gen_prof.mean()),
            "val_gen_act_std": float(gen_prof.std()),
            "val_real_act_mean": float(real_prof.mean()),
            "val_real_act_std": float(real_prof.std()),
        })

    # signal/noise sensitivity probe on the first batch
    if first_batch is not None:
        seq0, sig0 = first_batch
        B, L, C = seq0.shape
        x1 = torch.randn(B, L, C, device=device)
        x2 = torch.randn(B, L, C, device=device)
        perm = torch.randperm(B, device=device)
        idx_a = backbone(x1, sig0).argmax(-1)
        idx_b = backbone(x1, sig0[perm]).argmax(-1)     # same noise, permuted signal
        idx_c = backbone(x2, sig0).argmax(-1)           # diff noise, same signal
        out["val_sig_sens"] = float((idx_a != idx_b).float().mean())
        out["val_noise_sens"] = float((idx_a != idx_c).float().mean())

    backbone.train()
    return out


# ------------------------------- train -------------------------------

def run(cfg):
    """Train one geometry cell. `cfg` is a flat namespace assembled by src.main
    from configs/config.yaml + (optionally) a configs/winners.yaml formulation +
    CLI overrides."""
    seed_everything(cfg.seed, strict=cfg.strict_determinism)

    run_dir = make_run_dir(cfg.logs_root, cfg.run_name)
    log = setup_logger(run_dir / "run.log")
    metrics = JsonlWriter(run_dir / "metrics.jsonl")
    evals = JsonlWriter(run_dir / "eval.jsonl")
    test_evals = JsonlWriter(run_dir / "test_eval.jsonl")
    log.info(f"run_dir = {run_dir}")
    log.info(f"cfg     = {vars(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device  = {device}")
    if device.type == "cuda":
        log.info(f"gpu     = {torch.cuda.get_device_name(0)}")

    temps = tuple(float(x) for x in str(cfg.t_list).split(","))
    if cfg.grad_form and not cfg.sphere and not cfg.simplex:
        log.warning("--grad-form in PURE AMBIENT space: the unit-direction grad is "
                    "O(1) and never self-damps; only the bounded one-hot targets keep "
                    "it from diverging. Watch V_norm/x_norm.")

    spectral_weights = None
    if cfg.spectral:
        kappas = ([float(k) for k in str(cfg.kappa_list).split(",")]
                  if cfg.kappa_list else [cfg.kappa])
        spectral_weights = [matern_weights_s3(cfg.nu, k, cfg.lmax) for k in kappas]
        log.info(f"SPECTRAL kernel score on (S^3_+)^L  kappas={kappas}  "
                 f"nu={cfg.nu}  lmax={cfg.lmax}  eta={cfg.eta}")
    else:
        log.info(f"temps={temps}  dist_norm={cfg.dist_norm}  simplex={cfg.simplex}  "
                 f"sphere={cfg.sphere}  grad_form={cfg.grad_form}")

    train_ds = PromoterDataset(data_dir=cfg.data_dir, split="train", rand_offset=1)
    val_ds = PromoterDataset(data_dir=cfg.data_dir, split="valid", rand_offset=0)
    test_ds = PromoterDataset(data_dir=cfg.data_dir, split="test", rand_offset=0)
    log.info(f"train = {len(train_ds):,}   val = {len(val_ds):,}   test = {len(test_ds):,}")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
        generator=g, worker_init_fn=worker_init,
    )

    backbone = PromoterBackbone().to(device)
    n_params = sum(p.numel() for p in backbone.parameters())
    log.info(f"params  = {n_params:,}  ({n_params/1e6:.2f} M)")

    betas = tuple(float(x) for x in str(cfg.betas).split(","))
    opt = torch.optim.AdamW(backbone.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=betas)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, cfg.warmup_steps))

    ema = EMA(backbone, decay=cfg.ema_decay)
    ema_backbone = copy.deepcopy(backbone)
    for p in ema_backbone.parameters():
        p.requires_grad_(False)
    log.info(f"ema     = decay {cfg.ema_decay}")

    log.info("loading Sei...")
    sei = SeiEval(data_dir=cfg.data_dir)
    seq0, _ = next(iter(DataLoader(val_ds, batch_size=4, num_workers=0)))
    _ = sei.get_sei_profile(seq0.to(device).float())
    log.info("Sei loaded")

    # Coherent (activity-stratified) batches: precompute a per-example Sei
    # H3K4me3-activity scalar once, then sample each batch from a tau-softmax
    # neighborhood around a random anchor. The drift/generator are unchanged --
    # only batch composition differs.
    if cfg.sei_batch_tau > 0:
        log.info(f"precomputing coherent-batch key [sei-activity] (tau={cfg.sei_batch_tau})...")
        key_chunks = []
        pre_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)
        for bseq, _bsig in pre_loader:
            k = torch.as_tensor(sei.get_sei_profile(bseq.to(device).float()),
                                dtype=torch.float32)
            key_chunks.append(k)
        key_bank = torch.cat(key_chunks)
        sampler = CoherentBatchSampler(key_bank, cfg.batch,
                                       tau=cfg.sei_batch_tau, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=device.type == "cuda", worker_init_fn=worker_init,
        )
        log.info(f"coherent batches: key {key_bank.mean():.3f}+-{key_bank.std():.3f} "
                 f"(n={key_bank.numel()})")
    steps_per_epoch = len(train_loader)
    log.info(f"steps/epoch = {steps_per_epoch}  total = {steps_per_epoch*cfg.epochs:,}")

    import wandb
    run_name = cfg.wandb_run_name or run_dir.name
    wandb_run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                           name=run_name, dir=str(run_dir),
                           mode=cfg.wandb_mode, config=vars(cfg))
    log.info(f"wandb   = {cfg.wandb_mode}  {cfg.wandb_project}/{run_name}")

    def do_eval(epoch, step):
        ema.copy_to(ema_backbone)
        e = evaluate(ema_backbone, val_ds, sei, device, cfg.val_batch, cfg.num_workers)
        e["epoch"] = epoch
        e["step"] = step
        # satisficing objective (val only): reach the k-mer floor, then minimize sei.
        if "val_sei_ratio" in e:
            f = cfg.score_kmer_floor
            k = e["val_kmer_corr"]
            e["val_paper_score"] = (k if k < f
                                    else f + (1.0 - e["val_sei_ratio"]) + 0.1 * (k - f))
        evals.write(**e)
        wandb_run.log({f"eval/{k}": v for k, v in e.items()
                       if k not in ("epoch", "step")} | {"epoch": epoch, "step": step})
        sei_s = (f"sei {e['val_sei_mse']:.4f}/rand {e['val_sei_random']:.4f} "
                 f"(ratio {e['val_sei_ratio']:.2f})  "
                 f"act {e['val_gen_act_mean']:.3f}+-{e['val_gen_act_std']:.3f}"
                 f"/{e['val_real_act_mean']:.3f}+-{e['val_real_act_std']:.3f}  "
                 if "val_sei_mse" in e else "")
        log.info(f"EVAL ep {epoch:3d}  kmer {e['val_kmer_corr']:.4f}  "
                 f"pos_acc {e['val_pos_acc']:.4f}  "
                 f"hamm {e['val_hamming']:.3f}/{e['val_gt_hamming']:.3f}  "
                 f"sigD {e.get('val_sig_sens', 0):.3f} noiseD {e.get('val_noise_sens', 0):.3f}  "
                 f"{sei_s}n={e['n_eval']}")
        return e

    best = {"sei": float("inf"), "kmer": 0.0, "score": float("-inf"),
            "score_epoch": 0, "sei_epoch": 0, "kmer_epoch": 0}
    best_state, best_sei_state, best_kmer_state = {}, {}, {}

    def _snap(dst):
        dst.clear()
        dst.update({k: v.detach().cpu().clone()
                    for k, v in ema_backbone.state_dict().items()})

    def do_test_eval(epoch, step, tag):
        """Test-set (chr8/9) eval -- REPORTING ONLY, never used for selection.
        Assumes ema_backbone already holds the weights to evaluate."""
        te = evaluate(ema_backbone, test_ds, sei, device, cfg.val_batch, cfg.num_workers)
        te = {("test_" + k[4:]) if k.startswith("val_") else k: v for k, v in te.items()}
        te["epoch"] = epoch
        te["step"] = step
        te["tag"] = tag
        test_evals.write(**te)
        wandb_run.log({f"test/{k}": v for k, v in te.items()
                       if k not in ("epoch", "step", "tag")}
                      | {"epoch": epoch, "step": step})
        return te

    def track_best(e):
        if "val_sei_mse" in e and e["val_sei_mse"] < best["sei"]:
            best["sei"] = e["val_sei_mse"]
            best["sei_epoch"] = e["epoch"]
            wandb_run.summary["best_val_sei_mse"] = best["sei"]
            wandb_run.summary["best_sei_epoch"] = e["epoch"]
            _snap(best_sei_state)
        if e["epoch"] > cfg.best_kmer_min_epoch and e["val_kmer_corr"] > best["kmer"]:
            best["kmer"] = e["val_kmer_corr"]
            best["kmer_epoch"] = e["epoch"]
            wandb_run.summary["best_kmer_corr"] = best["kmer"]
            wandb_run.summary["best_kmer_epoch"] = e["epoch"]
            _snap(best_kmer_state)
        if e.get("val_paper_score", float("-inf")) > best["score"]:
            best["score"] = e["val_paper_score"]
            best["score_epoch"] = e["epoch"]
            wandb_run.summary["paper_score"] = best["score"]
            wandb_run.summary["paper_score_epoch"] = e["epoch"]
            wandb_run.summary["kmer_at_best_score"] = e["val_kmer_corr"]
            wandb_run.summary["sei_at_best_score"] = e.get("val_sei_mse")
            wandb_run.summary["sei_ratio_at_best_score"] = e.get("val_sei_ratio")
            _snap(best_state)

    global_step = 0
    t0 = time.time()
    do_eval(0, 0)

    try:
        for epoch in range(1, cfg.epochs + 1):
            backbone.train()
            for seq, sig in train_loader:
                seq = seq.to(device, non_blocking=True).float()
                sig = sig[..., :1].to(device, non_blocking=True).float()
                B = seq.shape[0]
                x_in = torch.randn(B, SEQ_LEN, ALPHA, device=device)

                logits = backbone(x_in, sig)
                if cfg.spectral:
                    loss, info = drift_loss_sphere_spectral(
                        logits, seq, spectral_weights, eta=cfg.eta)
                elif cfg.sphere:
                    loss, info = drift_loss_sphere(
                        logits, seq, temps, dist_norm=cfg.dist_norm,
                        grad_form=cfg.grad_form)
                else:
                    x = F.softmax(logits, -1) if cfg.simplex else logits
                    loss, info = drift_loss(
                        x.reshape(B, SEQ_LEN * ALPHA), seq.reshape(B, SEQ_LEN * ALPHA),
                        temps, dist_norm=cfg.dist_norm, grad_form=cfg.grad_form)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = total_grad_norm(backbone)
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                ema.update(backbone)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    rec = {"step": global_step, "epoch": epoch,
                           "loss": info["loss"], "V_norm": info["V_norm"],
                           "x_norm": info["x_norm"], "grad_norm": gn,
                           "lr": opt.param_groups[0]["lr"], "wall_s": time.time() - t0}
                    metrics.write(**rec)
                    wandb_run.log({f"train/{k}": v for k, v in rec.items()
                                   if k not in ("epoch", "step")}
                                  | {"step": global_step, "epoch": epoch})
                    log.info(f"ep {epoch:3d} step {global_step:6d}  "
                             f"loss {rec['loss']:.4e}  V {rec['V_norm']:.3f}  "
                             f"grad {gn:.2e}  lr {rec['lr']:.2e}  ({rec['wall_s']:.0f}s)")

            if epoch % cfg.eval_every_epochs == 0 or epoch == cfg.epochs:
                track_best(do_eval(epoch, global_step))
                if cfg.test_eval_every_epochs > 0 and epoch % cfg.test_eval_every_epochs == 0:
                    do_test_eval(epoch, global_step, tag="periodic")

        # FINAL test-set eval on the three selected checkpoints (in memory, not
        # disk): best val paper-score, lowest val Sei-MSE, highest val k-mer.
        def _final_test(state, ep, tag, suffix):
            if not state:
                return
            ema_backbone.load_state_dict({k: v.to(device) for k, v in state.items()})
            te = do_test_eval(ep, global_step, tag=tag)
            for k in ("kmer_corr", "sei_mse", "sei_ratio"):
                if f"test_{k}" in te:
                    wandb_run.summary[f"test_{k}_at_{suffix}"] = te[f"test_{k}"]
            log.info(f"TEST@{suffix} (val ep {ep}): "
                     f"kmer {te.get('test_kmer_corr', float('nan')):.4f}  "
                     f"sei {te.get('test_sei_mse', float('nan')):.4f}  "
                     f"ratio {te.get('test_sei_ratio', float('nan')):.3f}")

        _final_test(best_state, best["score_epoch"], "best_val", "best_val")
        _final_test(best_sei_state, best["sei_epoch"], "best_sei", "best_sei")
        _final_test(best_kmer_state, best["kmer_epoch"], "best_kmer", "best_kmer")
    finally:
        metrics.close()
        evals.close()
        test_evals.close()
        wandb_run.finish()
        log.info(f"DONE  best kmer {best['kmer']:.4f}  best sei {best['sei']:.4f}")
