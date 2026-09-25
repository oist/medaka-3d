"""Plot observed, excluded and accepted output without treating coverage as ID accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmark import BASELINE_DIR, EXPERIMENT, sha256
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def read_candidate(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("frame_idx", "xyz", "accepted_xyz", "review")}


def availability_state(xyz, accepted):
    observed = np.isfinite(xyz).all(axis=-1)
    accepted_mask = np.isfinite(accepted).all(axis=-1)
    if np.any(accepted_mask & ~observed):
        raise ValueError("an accepted coordinate must be observed")
    return np.where(accepted_mask, 2, np.where(observed, 1, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--post-run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    raw_path, post_path = args.raw_run / "candidate.npz", args.post_run / "candidate.npz"
    raw, post = read_candidate(raw_path), read_candidate(post_path)
    frames = raw["frame_idx"]
    np.testing.assert_array_equal(frames, post["frame_idx"])
    if not len(frames) or not np.all(np.diff(frames) == 1):
        raise ValueError("timeline requires a contiguous native-rate run")
    baseline_path = args.input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        baseline_xyz = np.where((baseline["slot_track"][frames] >= 0)[..., None],
                                baseline["xyz"][frames], np.nan)
    methods = [
        ("Frozen C34 (lambda 0.767)", availability_state(baseline_xyz, baseline_xyz)),
        ("3D recovery v2", availability_state(raw["xyz"], raw["accepted_xyz"])),
        ("Global ID postpass", availability_state(post["xyz"], post["accepted_xyz"])),
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    duration = len(frames) / 166
    figure, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True,
                                gridspec_kw={"hspace": 0.65})
    palette = ListedColormap(["#d3d9e0", "#edb04b", "#268c83"])
    metrics = {}
    for axis, (label, states) in zip(axes, methods, strict=True):
        axis.imshow(states.T, origin="upper", aspect="auto", interpolation="nearest",
                    extent=[0, duration, 2.5, -0.5], cmap=palette, vmin=0, vmax=2)
        coverage = (states == 2).mean(axis=0)
        axis.set_yticks(range(3), [f"Seed ID {i}: {100 * coverage[i]:.2f}% accepted"
                                  for i in range(3)])
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlim(0, duration)
        axis.set_xticks(np.linspace(0, duration, 7))
        axis.grid(axis="x", color="white", alpha=0.65, linewidth=0.7)
        available_name = ("provided_output_animal_frames" if label.startswith("Frozen")
                          else "observed_animal_frames")
        metrics[label] = {
            available_name: int((states > 0).sum()),
            "accepted_animal_frames": int((states == 2).sum()),
            "accepted_per_seed_identity": (states == 2).sum(axis=0).tolist(),
            "denominator_animal_frames": int(states.size),
        }
    axes[-1].set_xlabel("Seconds since the sole initial seed")
    figure.suptitle(f"One initial seed | {len(frames):,} native frames | {duration:.3f} seconds",
                   x=0.28, ha="left", fontsize=13, fontweight="bold")
    figure.legend(handles=[Patch(color=palette.colors[i], label=label) for i, label in enumerate(
        ["Missing pectoral position", "Observed but excluded", "Accepted output"])],
        loc="lower center", bbox_to_anchor=(0.6, 0.045), ncol=3, frameon=False)
    figure.text(0.28, 0.015, "Accepted output is not verified identity accuracy. "
                "C34 shows its supplied output; review policies differ. "
                "Seed IDs may be wrong or permuted.",
                fontsize=8, color="#444444")
    figure.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=0.19)
    figure.savefig(args.out / "review_timeline.png", dpi=180)
    plt.close(figure)
    inputs = [raw_path, post_path, baseline_path, Path(__file__)]
    metadata = {
        "frames": [int(frames[0]), int(frames[-1]) + 1], "fps": 166,
        "input_hashes": {str(path): sha256(path) for path in inputs},
        "metrics": metrics, "interpretation": "Availability, not identity ground truth.",
    }
    (args.out / "timeline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
