"""Copy the staged appearance runs of this analysis to the shared data parent, with a manifest.

Same procedure as ``analyses/20261109_unreviewed_identity_candidates/archive_runs.py``
for the runs of ``config_appearance_g2o1_v1.toml``. Each
``<staging>/<output>/<run id>`` is copied to ``work/appearance/<recording>/<run id>/``
under the data parent, leaving out the intermediate crop arrays, every copied
file is verified by SHA-256 and ``staged_copy_manifest.json`` is written into
the destination. Existing destinations are never overwritten.

Usage: archive_runs.py [--dry-run] [--keys g21 g22 ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import tomllib
from pathlib import Path

from medaka3d import load_paths

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "20261031_p11_identity_evaluation" / "config_appearance_g2o1_v1.toml"
CROP_DIRS = ("crops_all", "crops", "warning_crops", "masked", "warning_masked")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keys", nargs="*", help="recording keys of the configuration; default all"
    )
    args = parser.parse_args(argv)
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    paths = load_paths()
    root = paths.raw_data_dir.parent
    if paths.appearance_staging_dir is None:
        raise ValueError("Set appearance_staging_dir in configs/paths.local.toml")
    keys = args.keys or list(cfg["run_ids"])
    for key in keys:
        rc = cfg[key]
        src = paths.appearance_staging_dir / rc["output"] / cfg["run_ids"][key]
        if not src.is_dir():
            raise FileNotFoundError(f"Staged run is missing: {src}")
        dest = root / "work" / "appearance" / rc["experiment"] / src.name
        if dest.exists():
            print(f"skip {dest}: exists")
            continue
        files = [p for p in src.rglob("*") if p.is_file()]
        skipped = [p for p in files if p.relative_to(src).parts[0] in CROP_DIRS]
        files = [p for p in files if p not in skipped]
        total = sum(p.stat().st_size for p in files)
        print(
            f"{key} {rc['experiment']} {src.name}: {len(files)} files, {total / 1e9:.2f} GB,"
            f" skipping {len(skipped)} crop files",
            flush=True,
        )
        if args.dry_run:
            continue
        started = time.time()
        manifest = {
            "source": str(src),
            "destination": str(dest),
            "files": {},
            "skipped": [str(p.relative_to(src)) for p in skipped],
        }
        dest.mkdir(parents=True, exist_ok=False)
        for p in files:
            rel = p.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            digest = sha256(p)
            if sha256(target) != digest:
                raise RuntimeError(f"Copy verification failed for {rel}")
            manifest["files"][str(rel)] = {"size": p.stat().st_size, "sha256": digest}
        manifest["seconds"] = round(time.time() - started, 1)
        (dest / "staged_copy_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"   copied in {manifest['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
