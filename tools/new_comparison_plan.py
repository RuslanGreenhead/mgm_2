"""Write a fresh-name reproduction plan without executing training or CUDA."""
import argparse
import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"final_[a-z0-9_]{1,56}", args.phase):
        parser.error("Use a new phase name starting with final_, followed by lowercase letters/digits/underscores")
    base = ROOT / "outputs/comparison2026"
    destination = base / "plans" / f"{args.phase}.json"
    if destination.exists() or (base / args.phase).exists():
        parser.error("The phase or plan already exists; no files were changed")
    plan = copy.deepcopy(json.loads((base / "plans/final_v1.json").read_text(encoding="utf-8")))
    plan["phase"] = args.phase
    for job in plan["jobs"]:
        old = Path(job["output"].replace("\\", "/"))
        job["output"] = f"outputs/comparison2026/{args.phase}/{old.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"plan": destination.relative_to(ROOT).as_posix(), "jobs": len(plan["jobs"]),
                      "training_launched": False, "requires_datasets_before_training": True}))


if __name__ == "__main__":
    main()
