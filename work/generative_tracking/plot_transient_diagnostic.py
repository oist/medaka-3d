"""Post-hoc visualization of the preserved v2 transient run; no refitting."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

root = Path(__file__).resolve().parent
source = root / "transient_333519-333719/candidate.npz"
with np.load(source) as arrays:
    frames = arrays["frame_idx"]
    poses = arrays["pose_xyz"]
    observed = arrays["observed"]
    review = arrays["review"]
lengths = np.linalg.norm(np.diff(poses, axis=2), axis=-1)
colors = ("#cd8c1d", "#008cb1", "#c34b8a")
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 2.2, 1]})
for fish, color in enumerate(colors):
    axes[0].plot(frames, lengths[:, fish, 1], color=color, label=f"S{fish}", lw=1.8)
axes[0].axhline(lengths[0, 0, 1], color="#656565", ls="--", lw=1,
                label="S0 initial pectoral-tail length")
axes[0].set_ylabel("Pectoral-tail length (mm)")
axes[0].legend(loc="upper left", frameon=False, ncol=2)
axes[0].grid(alpha=0.2)
axes[0].set_title("A numerically converged fit can produce an impossible body", loc="left",
                  weight="bold", fontsize=13)
extent = (frames[0] - .5, frames[-1] + .5, 8.5, -.5)
axes[1].imshow(observed.reshape(len(frames), 9).T, aspect="auto", interpolation="nearest",
               cmap=ListedColormap(["#ededed", "#299c94"]), vmin=0, vmax=1, extent=extent)
axes[1].set_yticks(range(9), [f"S{i} {node}" for i in range(3)
                             for node in ("mouth", "pectoral", "tail")])
axes[1].set_title("Node observation support: teal = at least two views; gray = prediction",
                  loc="left", fontsize=10)
axes[2].imshow(review.T, aspect="auto", interpolation="nearest",
               cmap=ListedColormap(["#ededed", "#d48b32"]), vmin=0, vmax=1,
               extent=(frames[0] - .5, frames[-1] + .5, 2.5, -.5))
axes[2].set_yticks(range(3), ["S0", "S1", "S2"])
axes[2].set_title("Identity review: amber = withheld", loc="left", fontsize=10)
axes[2].set_xlabel("Video frame (166 frames/s)")
axes[2].ticklabel_format(axis="x", style="plain", useOffset=False)
for ax in axes:
    ax.axvline(333619, color="#4c4c4c", ls=":", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "Preserved v2 run, K_1_2, one trio. Dashed vertical: prior noted event. "
         "Slots are anonymous; this does not validate male identity.", fontsize=9, color="#555")
fig.tight_layout(rect=(0, .035, 1, 1))
figure = root / "transient_v2_diagnostic.png"
fig.savefig(figure, dpi=170)
fig.savefig(root / "transient_v2_diagnostic.svg")
summary = {
    "analysis": "post-hoc visualization; no tracker changes or parameter selection",
    "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "node_pairs": ["mouth-pectoral", "pectoral-tail"],
    "initial_lengths_mm": lengths[0].tolist(), "end_lengths_mm": lengths[-1].tolist(),
    "maximum_lengths_mm": lengths.max(axis=0).tolist(),
    "observed_node_frames_by_slot": observed.sum(axis=0).tolist(),
    "review_frames_by_slot": review.sum(axis=0).tolist(),
    "total_frames_per_slot": len(frames), "figure": str(figure),
}
(root / "transient_v2_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(figure)
