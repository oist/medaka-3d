"""Copy only the frozen original Stage A/B inputs needed by the comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.local.toml"))
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path(
            "work/multiview_node_review_v1/human_review_followup/"
            "stage_a_review_audit_v2.provenance.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("work/custom2d_comparison_v1/inputs"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["paths"]
    shared = Path(config["work_dir"])
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with provenance("stage_custom2d_comparison_inputs", args.output) as rec:
        rec.input("prior_source_audit", args.previous, sha256=sha256(args.previous))
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.param("shared_work_root", str(shared))
        entries = {}
        for name, old in previous["inputs"].items():
            if name not in {"C28", "c27_summary", "c27_provenance"} and not name.startswith("C27_"):
                continue
            original = Path(old["path"])
            # Read the configured root, never an implicit alternative mount.
            relative = original.relative_to(shared)
            source = shared / relative
            digest = sha256(source)
            if digest != old["sha256"]:
                raise ValueError(f"Frozen input changed since audit: {name}")
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256(destination) != digest:
                    raise ValueError(f"Refusing to overwrite different staged input: {name}")
            else:
                with source.open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if sha256(destination) != digest:
                raise ValueError(f"Staged copy differs: {name}")
            rec.input(name, source, sha256=digest)
            rec.output(name, destination, sha256=digest)
            entries[name] = {"path": str(destination), "sha256": digest}
        required = {"C28", "C27_top_cam", "C27_side1_cam", "C27_side2_cam"}
        if not required.issubset(entries):
            raise ValueError(f"Missing audit inputs: {required - entries.keys()}")
        path = args.output / "manifest.json"
        path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        rec.output("manifest", path, sha256=sha256(path))
        print(json.dumps({"staged": len(entries), "manifest": str(path)}))


if __name__ == "__main__":
    main()
