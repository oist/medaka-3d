"""A compact trajectory overview; not a native-frame identity certification."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "labs/20260909_three_body"))
import run_experiment as base

from medaka3d.provenance import provenance

root = repo / "work/three_body"
run = root / "identity_memory_continuous_one_minute_v1"
evaluation = root / "identity_memory_continuous_one_minute_comparison_v1"
output = root / "identity_memory_minute_overview_v1"
if output.exists():
    raise FileExistsError(output)
hashes = {str(Path(__file__)): base.sha256(__file__)}
for folder, name in ((run, "experiment"), (evaluation, "evaluation")):
    path = folder / f"{name}.provenance.json"
    record = json.loads(path.read_text())
    if record["status"] != "ok":
        raise ValueError("both inference and evaluation must be complete")
    hashes[str(path)] = base.sha256(path)
    for item in record["outputs"].values():
        if base.sha256(item["path"]) != item["sha256"]:
            raise ValueError("completed plot evidence changed")
        hashes[item["path"]] = item["sha256"]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


identity = load(run / "identity.npz")
dp = load(run / "dp.npz")
motion = load(evaluation / "motion_only_identity.npz")
np.testing.assert_array_equal(identity["frame_idx"], motion["frame_idx"])
frames = identity["frame_idx"]
if len(frames) != 9960 or frames[0] != 76620 or frames[-1] != 86579:
    raise ValueError("the overview requires the registered complete minute")
time = (frames - frames[0]) / 166
colors = ["#0072B2", "#E69F00", "#CC79A7"]
fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, .8, 1.5]})
extent = [0, len(frames) / 166, 2.5, -.5]
for ax, values, title in (
    (axes[0], identity["identity_to_raw_slot"], "A. Image color + observation-memory motion"),
    (axes[1], motion["identity_to_raw_slot"], "B. Observation-memory motion only"),
):
    ax.imshow(values.T, aspect="auto", interpolation="nearest", extent=extent,
              cmap=ListedColormap(colors), vmin=-.5, vmax=2.5)
    changes = np.flatnonzero(np.any(np.diff(values, axis=0), axis=1)) + 1
    ax.scatter(time[changes], np.full(len(changes), -.42), marker="v", color="black", s=12)
    ax.set_title(f"{title}  |  mapping changes: {len(changes)}", loc="left", fontsize=11)
    ax.set_yticks([0, 1, 2], ["ID 0", "ID 1", "ID 2"])
axes[2].imshow(identity["proposed_observed"][:, :, 1].T, aspect="auto",
               interpolation="nearest", extent=extent, cmap=ListedColormap(["#eeeeee", "#333333"]),
               vmin=0, vmax=1)
axes[2].set_title("C. Direct pectoral support in A (dark: observed; pale: missing)",
                  loc="left", fontsize=11)
axes[2].set_yticks([0, 1, 2], ["ID 0", "ID 1", "ID 2"])
for i, color in enumerate(colors):
    gap = dp["identity_alternative_gap"][:, i].copy()
    gap[~np.isfinite(gap)] = np.nan
    axes[3].plot(time, gap, color=color, lw=.8, label=f"ID {i}")
axes[3].set_yscale("symlog", linthresh=.01)
axes[3].set_ylim(bottom=0)
axes[3].set_ylabel("Alternative cost gap\n(symlog; not probability)")
axes[3].set_title("D. Best competing whole-path cost for a different raw slot in A",
                  loc="left", fontsize=11)
axes[3].legend(loc="upper right", ncol=3, fontsize=9)
axes[3].grid(alpha=.2)
axes[3].set_xlabel("Seconds from source frame 76620 (166 fps)")
axes[3].set_xlim(0, 60)
fig.suptitle("One continuous minute: proposed identity assignments on frozen 3D geometry",
             fontsize=14, y=.99)
fig.legend(handles=[Patch(color=color, label=f"Raw slot S{i}") for i, color in enumerate(colors)],
           loc="upper center", bbox_to_anchor=(.5, .962), ncol=3, frameon=False, fontsize=9)
fig.text(.06, .012,
         "Anonymous model identities; no independent all-three ID validation. Original review flags "
         "are retained in the data.\nOverview compresses native frames; use the saved full arrays "
         "and raw video to inspect brief events. Geometry fitting time is separate.", fontsize=9)
fig.tight_layout(rect=(.02, .065, 1, .94))
with provenance("three_body_identity_memory_overview", output, name="plot") as rec:
    for path, digest in hashes.items():
        rec.input(path, path, sha256=digest)
    for extension in ("png", "pdf"):
        path = output / f"identity_memory_minute.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        rec.output(extension, path, sha256=base.sha256(path))
    if any(base.sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("plot input evidence changed")
    rec.param("native_frame_count", len(frames))
    rec.param("independent_identity_truth", False)
plt.close(fig)
print(json.dumps({"output": str(output)}))
