"""Descriptive v2/v3 body and availability comparison, without rerunning inference."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
runs = {"v2": root / "transient_333519-333719",
        "v3": root / "anatomy_v3_transient_333519-333719"}
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
audit = {}
first_frames = first_seed = None
for (name, folder), color in zip(runs.items(), ("#cd8c1d", "#198f94"), strict=True):
    provenance = json.loads((folder / "benchmark.provenance.json").read_text())
    seed = np.array(provenance["params"]["seed"]["pose_xyz"])
    source = folder / "candidate.npz"
    with np.load(source) as arrays:
        frames, pose = arrays["frame_idx"], arrays["pose_xyz"]
        xyz, review = arrays["xyz"], arrays["review"]
        accepted = (arrays["accepted_xyz"] if "accepted_xyz" in arrays.files
                    else np.where(review[..., None], np.nan, xyz))
    if first_frames is None:
        first_frames, first_seed = frames, seed
    else:
        np.testing.assert_array_equal(frames, first_frames)
        np.testing.assert_array_equal(seed, first_seed)
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    observed_fraction = np.isfinite(xyz).all(axis=-1).mean(axis=1)
    accepted_fraction = np.isfinite(accepted).all(axis=-1).mean(axis=1)
    axes[0].plot(frames, lengths[:, 0, 1], color=color, lw=2, label=name)
    axes[1].plot(frames, observed_fraction * 100, color=color, lw=1.5, label=name)
    axes[2].plot(frames, accepted_fraction * 100, color=color, lw=1.5, label=name)
    snapshot_checks = {}
    for path, expected in provenance["params"]["code_hashes"].items():
        snapshot = folder / "code_snapshot" / Path(path).name
        snapshot_checks[Path(path).name] = hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
    if not all(snapshot_checks.values()):
        raise ValueError(f"stored source snapshots do not match recorded hashes: {name}")
    audit[name] = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "snapshots_match_recorded_hashes": snapshot_checks,
                   "max_segment_length_mm": lengths.max(axis=0).tolist(),
                   "observed_animal_fraction": float(observed_fraction.mean()),
                   "accepted_animal_fraction": float(accepted_fraction.mean())}
axes[0].set_title("Anatomy correction on the same initialization and raw frames", loc="left",
                  weight="bold", fontsize=13)
axes[0].set_ylabel("S0 pectoral-tail (mm)")
axes[0].legend(frameon=False)
axes[1].set_ylabel("Observed pectoral (%)")
axes[2].set_ylabel("Accepted pectoral (%)")
for ax in axes[1:]:
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 33.33, 66.67, 100], ["0", "33", "67", "100"])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.axvline(333619, color="#888", lw=1, ls=":")
axes[2].set_xlabel("Video frame (166 frames/s)")
axes[2].ticklabel_format(axis="x", style="plain", useOffset=False)
fig.text(.01, .01, "One trio, one development window. Slots are anonymous. Availability and anatomy "
         "do not establish correct identity; only one female anchor is in this window.",
         fontsize=8, color="#555")
fig.tight_layout(rect=(0, .045, 1, 1))
fig.savefig(root / "transient_anatomy_v2_v3.png", dpi=170)
fig.savefig(root / "transient_anatomy_v2_v3.svg")
(root / "transient_anatomy_v2_v3_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
