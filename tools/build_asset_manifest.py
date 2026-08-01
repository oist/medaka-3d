"""Generate MANIFEST.json for the inherited asset root.

Re-runnable. Walks a declarative spec of asset groups, hashes each file, and
writes the manifest that `medaka3d.assets.load_catalog` validates against.

Large canonical artifacts (SLEAP checkpoints, training labels, Detectron2
weights) are referenced by absolute path where they already live, so the asset
root does not duplicate tens of gigabytes. Assets with no other home — ground
truth, COCO training sets, calibration — were copied under the root and are
referenced relatively.

Usage:
    python tools/build_asset_manifest.py --asset-root Z:/StephensU/medaka/assets
    python tools/build_asset_manifest.py --asset-root ... --no-hash   # quick
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medaka3d.assets import sha256_file

BUCKET = Path("Z:/StephensU/medaka")

SLEAP_ROUNDS = {
    "round7": ("retrain_20260704", "round7_260704", {"top": 1295, "side": 2590}),
    "round8": ("retrain_20260722_round8", "round8_260723_strictmouth", {"top": 1382, "side": 2764}),
}

# Which round is production for which view. The retired repo never stated this
# in one place, and the split is the single most confusing thing it left behind.
PRODUCTION = {"top": "round7", "side": "round8"}


def sleap_models() -> dict[str, dict]:
    """The 8 keeper checkpoints: 2 rounds x 2 views x 2 model types."""
    out: dict[str, dict] = {}
    for round_key, (round_dir, tag, counts) in SLEAP_ROUNDS.items():
        for view, n in counts.items():
            for kind in ("centroid", "centered_instance"):
                run = BUCKET / "sleap_models" / round_dir / view / "runs" / f"{tag}.{kind}_n{n}"
                for filename, suffix in (("best.ckpt", ""), ("training_config.yaml", ".config")):
                    path = run / filename
                    if not path.is_file():
                        print(f"  ! missing {path}", file=sys.stderr)
                        continue
                    production = PRODUCTION[view] == round_key
                    out[f"sleap.{round_key}.{view}.{kind}{suffix}"] = {
                        "path": path,
                        "kind": "sleap_model" if not suffix else "sleap_model_config",
                        "provenance": (
                            f"{round_dir} n={n}"
                            + (" [PRODUCTION]" if production else " [fallback]")
                        ),
                    }
    return out


def sleap_labels() -> dict[str, dict]:
    """Hand-annotated keypoint labels — the most expensive assets to recreate."""
    out: dict[str, dict] = {}
    r8 = BUCKET / "sleap_models" / "retrain_20260722_round8"
    r7 = BUCKET / "sleap_models" / "retrain_20260704"
    candidates = {
        "labels.round8.combined": (r8 / "combined_labels.pkg.slp", "round8 cumulative superset"),
        "labels.round8.top.train": (r8 / "top" / "train_labels.pkg.slp", "round8 top train split"),
        "labels.round8.top.val": (r8 / "top" / "val_labels.pkg.slp", "round8 top val split"),
        "labels.round8.side.train": (
            r8 / "side" / "train_labels.pkg.slp",
            "round8 side train split",
        ),
        "labels.round8.side.val": (r8 / "side" / "val_labels.pkg.slp", "round8 side val split"),
        "labels.round7.top": (r7 / "top" / "labels.pkg.slp", "round7 top n=1295"),
        "labels.round7.side": (r7 / "side" / "labels.pkg.slp", "round7 side n=2590"),
    }
    for key, (path, provenance) in candidates.items():
        if path.is_file():
            out[key] = {"path": path, "kind": "training_labels", "provenance": provenance}
        else:
            print(f"  ! missing {path}", file=sys.stderr)
    return out


def detectron2_models() -> dict[str, dict]:
    root = BUCKET / "medaka_202512" / "detectron2_models"
    pairs = {
        "detectron2.top": root / "top_detectron2_model",
        "detectron2.side": root / "side_detectron2_model_firstprompt",
    }
    out: dict[str, dict] = {}
    for key, directory in pairs.items():
        for filename, suffix in (("model_final.pth", ""), ("detectron2_config.yaml", ".config")):
            path = directory / filename
            if path.is_file():
                out[f"{key}{suffix}"] = {
                    "path": path,
                    "kind": "detectron2_model" if not suffix else "detectron2_config",
                    "provenance": directory.name,
                }
    return out


def appearance_models() -> dict[str, dict]:
    """Per-experiment F/M classifiers. Archived for a later phase, NOT wired in.

    Naming is inconsistent across experiments and only K_1_2 has a full set —
    recorded here so that irregularity is visible rather than surprising.
    """
    out: dict[str, dict] = {}
    root = BUCKET / "runs" / "udmt_validation" / "segid"
    if not root.is_dir():
        return out
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for model_dir in sorted(experiment_dir.glob("model_*")):
            name = model_dir.name
            keep = ("_clean" in name or name.startswith("model_top_")) and "smoke" not in name
            path = model_dir / "model.pt"
            if keep and path.is_file():
                out[f"appearance.{experiment_dir.name}.{name}"] = {
                    "path": path,
                    "kind": "appearance_classifier",
                    "provenance": f"per-experiment, {experiment_dir.name}; phase-2 archive",
                }
    return out


def local_tree(asset_root: Path, subdir: str, kind: str, provenance: str) -> dict[str, dict]:
    """Every file under an asset-root subdirectory, keyed by relative path."""
    base = asset_root / subdir
    out: dict[str, dict] = {}
    if not base.is_dir():
        return out
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(asset_root).as_posix()
        out[f"{kind}:{rel}"] = {"path": path, "kind": kind, "provenance": provenance}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=BUCKET / "assets")
    parser.add_argument("--no-hash", action="store_true", help="skip sha256 (fast, weaker)")
    args = parser.parse_args()

    root: Path = args.asset_root
    if not root.is_dir():
        parser.error(f"asset root does not exist: {root}")

    groups: dict[str, dict] = {}
    print("collecting...")
    for label, collect in (
        ("sleap models", sleap_models),
        ("sleap labels", sleap_labels),
        ("detectron2", detectron2_models),
        ("appearance", appearance_models),
    ):
        found = collect()
        print(f"  {label}: {len(found)}")
        groups.update(found)

    for subdir, kind, provenance in (
        ("models/calibration", "calibration", "retired repo calibration_results/ + models/"),
        ("groundtruth", "ground_truth", "hand-made; rescued from the retired repo and Drive"),
        ("training/detectron2", "detectron2_coco", "K_1_2 COCO training sets"),
        ("training/sleap/legacy", "training_labels", "labels rescued from deleted model rounds"),
        ("catalog", "catalog", "snapshot of medaka_202512 catalog + experiment list"),
    ):
        found = local_tree(root, subdir, kind, provenance)
        print(f"  {subdir}: {len(found)}")
        groups.update(found)

    print(f"hashing {len(groups)} assets..." if not args.no_hash else "sizing...")
    assets: dict[str, dict] = {}
    total = 0
    for key, entry in groups.items():
        path: Path = entry["path"]
        size = path.stat().st_size
        total += size
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()  # canonical location outside the root
        assets[key] = {
            "path": relative,
            "size": size,
            "sha256": "" if args.no_hash else sha256_file(path),
            "kind": entry["kind"],
            "provenance": entry["provenance"],
        }

    manifest = {
        "generated": datetime.now(UTC).isoformat(),
        "generator": "tools/build_asset_manifest.py",
        "asset_root": str(root),
        "hashed": not args.no_hash,
        "production_models": {
            "sleap_top": "round7 (retrain_20260704)",
            "sleap_side": "round8 (retrain_20260722_round8)",
            "note": "Production is SPLIT across two rounds. The retired repo never said so.",
        },
        "asset_count": len(assets),
        "total_bytes": total,
        "assets": assets,
    }
    out_path = root / "MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}\n  {len(assets)} assets, {total / 1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
