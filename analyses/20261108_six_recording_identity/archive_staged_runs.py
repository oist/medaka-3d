"""Copy appearance runs staged on local disk to the shared data parent, with a manifest.

New appearance runs of the six-recording study were written to a local
directory because the network share runs at about 10 MB/s. For recordings
marked ``staged = true``, the source is the configured local
``appearance_staging_dir/<recording>/<run id>``. This copies each source to
``work/appearance/<recording>/<run id>/`` under the data parent, verifies
every file by SHA-256 after the copy, and writes ``staged_copy_manifest.json``
(source, destination, size and hash per file) into the destination. Large
intermediate crop arrays can be left out with ``--skip-crops``; the manifest
records what was skipped. Existing destinations are never overwritten.

Usage: archive_staged_runs.py [--skip-crops] [--dry-run]
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
CROP_DIRS = ("crops_all", "crops", "warning_crops", "masked", "warning_masked")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-crops", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    paths = load_paths()
    root = paths.raw_data_dir.parent
    if paths.appearance_staging_dir is None:
        raise ValueError("Set appearance_staging_dir in configs/paths.local.toml")
    for name, rc in cfg["recordings"].items():
        if not rc.get("staged", False):
            continue
        for key in ("appearance_run", "appearance_run_group"):
            src = paths.appearance_staging_dir / name / Path(rc[key]).name
            if not src.is_dir():
                raise FileNotFoundError(f"Declared staged run is missing: {src}")
            dest = root / "work" / "appearance" / name / src.name
            if dest.exists():
                print(f"skip {dest}: exists")
                continue
            files = [p for p in src.rglob("*") if p.is_file()]
            skipped = [
                p for p in files if args.skip_crops and p.relative_to(src).parts[0] in CROP_DIRS
            ]
            files = [p for p in files if p not in skipped]
            total = sum(p.stat().st_size for p in files)
            print(
                f"{name} {src.name}: {len(files)} files, {total / 1e9:.2f} GB"
                + (f", skipping {len(skipped)} crop files" if skipped else ""),
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
