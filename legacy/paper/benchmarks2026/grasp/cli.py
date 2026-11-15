"""Run from workspace: python -m paper.benchmarks2026.grasp.cli --help."""

import argparse
import json

from . import OUTPUT, ROOT


def parser():
    p = argparse.ArgumentParser(description="Real ACRONYM SE(3) preparation and conditional generation. No implicit GPU jobs.")
    sub = p.add_subparsers(dest="command", required=True)
    public = sub.add_parser("fetch-public", help="Fetch only official, pinned, licensed public examples")
    public.add_argument("--destination", default=str(ROOT / "external/acronym-data/public"))
    full = sub.add_parser("fetch-full-grasps", help="Optional official ~1.6GB grasp archive, not gated meshes")
    full.add_argument("--destination", default=str(ROOT / "external/acronym-data/full"))
    full.add_argument("--expected-sha256")
    sub.add_parser("audit-access", help="Probe full-mesh gate and record released evaluator semantics")
    sub.add_parser("audit-author-data", help="Record author/upstream data-link search without downloading meshes")
    prep = sub.add_parser("prepare")
    prep.add_argument("--grasps-root", required=True)
    prep.add_argument("--mesh-root", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--license-record")
    prep.add_argument("--seed", type=int, default=2026)
    prep.add_argument("--points", type=int, default=256)
    prep.add_argument("--smoke-only", action="store_true")
    prep.add_argument("--protocol", choices=("object_heldout", "within_object_heldout_pose"), default="object_heldout")
    train = sub.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--method", choices=("mgm", "rfm", "rmf_ccadapt", "rmf_mt", "glsd"), required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--iterations", type=int, default=10000)
    train.add_argument("--batch", type=int, default=256)
    train.add_argument("--width", type=int, default=128)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--translation-weight", type=float, default=1.)
    train.add_argument("--critic-steps", type=int, default=5)
    train.add_argument("--warmup", type=int, default=100)
    train.add_argument("--save-every", type=int, default=500)
    train.add_argument("--out")
    train.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--split", choices=("val", "test"), default="test")
    ev.add_argument("--samples", type=int, default=512)
    ev.add_argument("--steps", type=int, default=1)
    ev.add_argument("--eval-seed", type=int, default=96231)
    ev.add_argument("--collision-samples", type=int, default=100)
    ev.add_argument("--skip-collision", action="store_true")
    ev.add_argument("--official-gripper", default=str(ROOT / "external/acronym-data/public/data/franka_gripper_collision_mesh.stl"))
    ev.add_argument("--out")
    for command in (train, ev):
        command.add_argument("--device", default="cpu")
        command.add_argument("--cpu-threads", type=int, choices=(1, 2), default=2)
        command.add_argument("--gpu-release", action="store_true", help="Only after explicit parent shared-queue release")
        command.add_argument("--allow-smoke-data", action="store_true")
    vis = sub.add_parser("visualize")
    vis.add_argument("--metrics", nargs="+", required=True)
    vis.add_argument("--out", default=str(OUTPUT / "visualizations/grasp_comparison.png"))
    vis.add_argument("--count", type=int, default=12)
    vis.add_argument("--object-id")
    test = sub.add_parser("test", help="CPU-only geometry, raw source, native loss, resume and queue-guard tests")
    test.add_argument("--report", default=str(OUTPUT/"verification_ready/test_results.json"))
    smoke = sub.add_parser("cpu-smoke", help="Five real-H5 CPU runs, exact state resumes, validation proxies, PNG+PDF")
    smoke.add_argument("--manifest", default=str(OUTPUT / "data/public_within_object/manifest.json"))
    smoke.add_argument("--out-root", default=str(OUTPUT / "verification_ready/cpu_integration"))
    smoke.add_argument("--iterations", type=int, default=30)
    plan = sub.add_parser("pilot-plan", help="Write nine fixed 10k-step GPU job specifications without launching")
    plan.add_argument("--out")
    profile = sub.add_parser("profile-cpu", help="Bounded CPU timing at the exact pilot architecture, no test metrics")
    profile.add_argument("--out-root")
    profile.add_argument("--iterations", type=int, default=100)
    for name in ("pilot-check", "pilot-run"):
        command = sub.add_parser(name, help="Verify the frozen pilot" if name == "pilot-check" else "Sequential resumable nine-job pilot; explicit GPU release required")
        command.add_argument("--plan", default=str(OUTPUT / "pilot_plan_ready.json"))
        if name == "pilot-run":
            command.add_argument("--gpu-release", action="store_true")
            command.add_argument("--calibrate-only", action="store_true", help="Only first 200 updates of seed0 for each method; no quality evaluation")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == "fetch-public":
        from .data import fetch_public
        result = fetch_public(args.destination)
        print(json.dumps(result, indent=2))
    elif args.command == "audit-access":
        from .data import audit_access
        print(json.dumps(audit_access(), indent=2))
    elif args.command == "audit-author-data":
        from .data import audit_author_data
        print(json.dumps(audit_author_data(), indent=2))
    elif args.command == "fetch-full-grasps":
        from .data import fetch_full_grasps
        result = fetch_full_grasps(args.destination, args.expected_sha256)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        print("Grasp H5 files:", len(result["files"]))
    elif args.command == "prepare":
        from .data import prepare
        result = prepare(args.grasps_root, args.mesh_root, args.out, args.seed, args.points, args.smoke_only, args.license_record, args.protocol)
        print(json.dumps({k: v for k, v in result.items() if k != "objects"}, indent=2))
        print("Objects:", [(o["object_id"], o["split"], o["n_positive"]) for o in result["objects"]])
    elif args.command == "train":
        if min(args.iterations, args.batch, args.width, args.save_every, args.critic_steps) < 1 or args.warmup < 0:
            p.error("Positive training sizes and nonnegative warmup required")
        from .runner import train
        print(train(args))
    elif args.command == "evaluate":
        if args.samples < 2 or args.steps < 1 or args.collision_samples < 20:
            p.error("Need samples >=2, steps >=1, collision-samples >=20")
        from .runner import evaluate
        print(evaluate(args))
    elif args.command == "visualize":
        from .visualization import visualize
        print(visualize(args.metrics, args.out, args.count, args.object_id))
    elif args.command == "test":
        import unittest
        from . import tests
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        from .data import write_json
        write_json(args.report, {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                   "skipped": len(result.skipped), "successful": result.wasSuccessful(), "gpu_used": False,
                   "failure_details": [(str(t), details) for t, details in result.failures+result.errors]})
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.command == "cpu-smoke":
        from .smoke import smoke
        print(smoke(args.manifest, args.out_root, args.iterations))
    elif args.command == "pilot-plan":
        from .pilot_plan import build_plan
        print(build_plan(args.out))
    elif args.command == "profile-cpu":
        from .pilot_plan import profile_cpu
        print(profile_cpu(args.out_root, args.iterations))
    elif args.command in ("pilot-check", "pilot-run"):
        from .pilot import preflight, run_pilot
        print(preflight(args.plan) if args.command == "pilot-check" else run_pilot(args.plan, args.gpu_release, args.calibrate_only))


if __name__ == "__main__":
    main()
