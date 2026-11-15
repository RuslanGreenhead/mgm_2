"""Dry-run or launch helper for extra R-MGM/RFM geometry runs.

Default behavior is intentionally conservative: print PowerShell commands and
do not start training. Add ``--run`` only when you really want to launch the
selected command block.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONDA = r"E:\anaconda\Scripts\conda.exe"
DEFAULT_ENV = r"..\..\.conda\rmgm"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    section: str
    argv: tuple[str, ...]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare commands for appendix geometries and high-D scaling."
    )
    parser.add_argument(
        "--section",
        choices=("all", "appendix", "official-rfm-highd", "metrics"),
        default="all",
        help="Command block to print or run.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "full"),
        default="pilot",
        help="Scale of generated commands. Default prints pilot commands only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conda", default=DEFAULT_CONDA)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument(
        "--format", choices=("text", "markdown", "json"), default="text"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute selected commands. Without this flag, this is a dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run or print only the first N selected commands.",
    )
    args = parser.parse_args()

    specs = list(build_plan(args.conda, args.env, args.profile, args.seed))
    if args.section != "all":
        specs = [spec for spec in specs if spec.section == args.section]
    if args.limit is not None:
        specs = specs[: args.limit]

    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "name": spec.name,
                        "section": spec.section,
                        "cwd": str(SCRIPT_DIR),
                        "command": powershell_command(spec.argv),
                        "argv": list(spec.argv),
                        "note": spec.note,
                    }
                    for spec in specs
                ],
                indent=2,
            )
        )
    elif args.format == "markdown":
        print_markdown(specs)
    else:
        print_text(specs)

    if args.run:
        for spec in specs:
            print(f"\n=== running {spec.name} ===", flush=True)
            subprocess.run(spec.argv, cwd=SCRIPT_DIR, check=True)


def build_plan(
    conda: str, env: str, profile: str, seed: int
) -> Iterable[CommandSpec]:
    yield from appendix_commands(conda, env, profile, seed)
    yield from official_rfm_highd_commands(conda, env, profile, seed)
    yield metrics_command(conda, env)


def appendix_commands(
    conda: str, env: str, profile: str, seed: int
) -> Iterable[CommandSpec]:
    profile_suffix = "" if profile == "pilot" else f"_{profile}"
    rmgm_root = f"../../outputs/rmgm_extra_appendix{profile_suffix}"
    rfm_root = f"../../outputs/rfm_extra_appendix{profile_suffix}"
    rfm_1nfe_root = f"../../outputs/rfm_extra_appendix_1nfe{profile_suffix}"

    rmgm_defaults = {
        "torus2d": [
            "--steps",
            "2000",
            "--batch-size",
            "512",
            "--latent-dim",
            "16",
            "--hidden-dim",
            "256",
            "--num-layers",
            "4",
            "--critic-steps",
            "3",
            "--lr-g",
            "3e-5",
            "--lr-c",
            "2e-4",
            "--log-every",
            "100",
            "--plot-every",
            "250",
            "--eval-samples",
            "4096",
            "--torus-generator",
            "gmm",
            "--torus-components",
            "4",
            "--torus-residual-scale",
            "0.25",
            "--torus-kmeans-init",
        ],
        "spd": [
            "--steps",
            "1500",
            "--batch-size",
            "256",
            "--latent-dim",
            "12",
            "--hidden-dim",
            "192",
            "--num-layers",
            "4",
            "--critic-steps",
            "3",
            "--lr-g",
            "5e-5",
            "--lr-c",
            "2e-4",
            "--log-every",
            "100",
            "--plot-every",
            "250",
            "--eval-samples",
            "4096",
            "--metric-normalize",
        ],
        "hyperbolic": [
            "--steps",
            "2000",
            "--batch-size",
            "512",
            "--latent-dim",
            "16",
            "--hidden-dim",
            "256",
            "--num-layers",
            "4",
            "--critic-steps",
            "3",
            "--lr-g",
            "3e-5",
            "--lr-c",
            "2e-4",
            "--log-every",
            "100",
            "--plot-every",
            "250",
            "--eval-samples",
            "4096",
            "--generator-type",
            "local_gmm",
            "--generator-components",
            "4",
            "--generator-residual-scale",
            "0.20",
            "--generator-kmeans-init",
        ],
    }

    rfm_defaults = {
        "torus2d": ["2500", "512", "256", "4", "250", "500", "4096"],
        "spd": ["1500", "256", "192", "4", "150", "300", "4096"],
        "hyperbolic": ["1500", "384", "192", "4", "150", "300", "4096"],
    }

    for experiment, extra in rmgm_defaults.items():
        args = scale_standalone_args(extra, profile)
        yield CommandSpec(
            name=f"rmgm-{experiment}",
            section="appendix",
            argv=conda_python(
                conda,
                env,
                "rmgm_train.py",
                [
                    "--experiment",
                    experiment,
                    "--seed",
                    str(seed),
                    "--device",
                    "cuda",
                    "--output-dir",
                    rmgm_root,
                    *args,
                ],
            ),
            note=(
                "Standalone R-MGM appendix geometry. Current rmgm_train.py uses "
                "hard-coded dims: torus2d=2, spd=3 vectorized SPD(2), "
                "hyperbolic=2."
            ),
        )

    for experiment, values in rfm_defaults.items():
        steps, batch, hidden, layers, log_every, plot_every, eval_samples = values
        args = [
            "--experiment",
            experiment,
            "--steps",
            steps,
            "--batch-size",
            batch,
            "--hidden-dim",
            hidden,
            "--num-layers",
            layers,
            "--lr",
            "2e-4",
            "--ode-steps",
            "128",
            "--log-every",
            log_every,
            "--plot-every",
            plot_every,
            "--eval-samples",
            eval_samples,
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--output-dir",
            rfm_root,
        ]
        if experiment == "spd":
            args.append("--metric-normalize")
        yield CommandSpec(
            name=f"rfm-{experiment}-128nfe",
            section="appendix",
            argv=conda_python(
                conda,
                env,
                "rfm_baseline_train.py",
                scale_rfm_args(args, profile, ode_steps="128"),
            ),
            note="Lightweight sample-quality RFM baseline with 128 Euler steps.",
        )
        yield CommandSpec(
            name=f"rfm-{experiment}-1nfe",
            section="appendix",
            argv=conda_python(
                conda,
                env,
                "rfm_baseline_train.py",
                scale_rfm_args(
                    replace_flags(args, {"--output-dir": rfm_1nfe_root}),
                    profile,
                    ode_steps="1",
                ),
            ),
            note="Same-NFE RFM baseline for one-step comparison.",
        )


def official_rfm_highd_commands(
    conda: str, env: str, profile: str, seed: int
) -> Iterable[CommandSpec]:
    cases = [
        ("wrapped_torus_d5", "wrapped_torus", {"wrapped.dim": "5"}),
        ("wrapped_torus_d10", "wrapped_torus", {"wrapped.dim": "10"}),
        ("wrapped_torus_d20", "wrapped_torus", {"wrapped.dim": "20"}),
        ("wrapped_spd_vec3", "wrapped_spd", {"wrapped.dim": "3"}),
        ("wrapped_spd_vec6", "wrapped_spd", {"wrapped.dim": "6"}),
        ("wrapped_spd_vec10", "wrapped_spd", {"wrapped.dim": "10"}),
        ("hyperbolic_d2", "hyperbolic", {}),
    ]
    for slug, experiment, overrides in cases:
        args = [
            "experiment=" + experiment,
            "seed=" + str(seed),
            "use_wandb=False",
            "hydra/launcher=basic",
            f"hydra.run.dir=../../outputs/rfm_official_extra/{profile}/{slug}",
            *hydra_profile_overrides(profile, experiment),
        ]
        args.extend(f"{key}={value}" for key, value in overrides.items())
        yield CommandSpec(
            name=f"official-rfm-{slug}",
            section="official-rfm-highd",
            argv=conda_python(conda, env, "train.py", args),
            note=(
                "Official Hydra RFM path. This is the only checked path here "
                "that honors wrapped.dim for high-dimensional wrapped data."
            ),
        )


def metrics_command(conda: str, env: str) -> CommandSpec:
    return CommandSpec(
        name="collect-standalone-extra-metrics",
        section="metrics",
        argv=conda_python(
            conda,
            env,
            "rmgm_compare_results.py",
            [
                "../../outputs/rmgm_extra_appendix",
                "../../outputs/rfm_extra_appendix",
                "../../outputs/rfm_extra_appendix_1nfe",
                "--json",
                "../../outputs/rmgm_summary/extra_geometry_metrics.json",
            ],
        ),
        note=(
            "Collects standalone metrics.jsonl runs. Official Hydra RFM runs "
            "write metrics.json directly and should be summarized separately."
        ),
    )


def hydra_profile_overrides(profile: str, experiment: str) -> list[str]:
    if profile == "smoke":
        return [
            "optim.num_iterations=20",
            "optim.batch_size=64",
            "optim.val_batch_size=64",
            "val_every=20",
            "model.d_model=256",
            "model.num_layers=2",
            "visualize=False",
        ]
    if profile == "pilot":
        model_size = "1024" if experiment == "wrapped_torus" else "512"
        return [
            "optim.num_iterations=5000",
            "optim.batch_size=512",
            "optim.val_batch_size=512",
            "val_every=1000",
            f"model.d_model={model_size}",
            "visualize=True",
        ]
    return ["visualize=True"]


def scale_standalone_args(args: Sequence[str], profile: str) -> list[str]:
    if profile != "smoke":
        return list(args)
    return replace_flags(
        args,
        {
            "--steps": "20",
            "--batch-size": "64",
            "--log-every": "10",
            "--plot-every": "20",
            "--eval-samples": "256",
            "--hidden-dim": "64",
            "--num-layers": "2",
        },
    )


def scale_rfm_args(args: Sequence[str], profile: str, ode_steps: str) -> list[str]:
    out = replace_flags(args, {"--ode-steps": ode_steps})
    if profile != "smoke":
        return out
    return replace_flags(
        out,
        {
            "--steps": "20",
            "--batch-size": "64",
            "--log-every": "10",
            "--plot-every": "20",
            "--eval-samples": "256",
            "--hidden-dim": "64",
            "--num-layers": "2",
        },
    )


def replace_flags(args: Sequence[str], replacements: dict[str, str]) -> list[str]:
    out = list(args)
    for flag, value in replacements.items():
        if flag not in out:
            continue
        index = out.index(flag)
        if index + 1 >= len(out):
            raise ValueError(f"Flag {flag} has no value.")
        out[index + 1] = value
    return out


def conda_python(
    conda: str, env: str, script: str, args: Sequence[str]
) -> tuple[str, ...]:
    return (conda, "run", "-p", env, "python", "-B", script, *args)


def print_text(specs: Sequence[CommandSpec]) -> None:
    print(f"cwd: {SCRIPT_DIR}")
    print("dry-run: add --run to execute selected commands")
    for spec in specs:
        print(f"\n[{spec.section}] {spec.name}")
        print(f"# {spec.note}")
        print(powershell_command(spec.argv))


def print_markdown(specs: Sequence[CommandSpec]) -> None:
    print(f"Run from `{SCRIPT_DIR}`. Default helper mode is dry-run.\n")
    for spec in specs:
        print(f"### {spec.name}")
        print(f"Section: `{spec.section}`")
        print(f"{spec.note}\n")
        print("```powershell")
        print(powershell_command(spec.argv))
        print("```\n")


def powershell_command(argv: Sequence[str]) -> str:
    return "& " + " ".join(quote_powershell(arg) for arg in argv)


def quote_powershell(arg: str) -> str:
    if arg == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=./:\\")
    if all(ch in safe for ch in arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
