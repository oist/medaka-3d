"""Plot the completed first-chunk diagnosis without refitting or accepting IDs."""

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "work/three_body/matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import summarize_chest_candidate_replay as checks  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def main():
    root = REPO / "work/three_body/directional_mouth_first_chunk_diagnosis_v1"
    output = REPO / "work/three_body/directional_mouth_depth_figure_v1"
    checks.require(not output.exists(), "new figure output required")
    hashes = {str(Path(__file__).resolve()): checks.sha(__file__)}
    expected = "e87d4c2c137ca09ced694efd8a96381b8e7b1d95555b539546575cd9dd3ee609"
    checks.require(checks.sha(root / "diagnosis.provenance.json") == expected, "wrong diagnosis")
    seal = checks.read(root / "diagnosis.provenance.json")
    checks.verify(root, "diagnosis.provenance.json", seal["stage"], hashes)
    values = {}
    for name in ("original", "directional"):
        path = root / f"{name}_same_horizon.npz"
        checks.require(str(path) in hashes, "unsealed trajectory")
        with np.load(path, allow_pickle=False) as archive:
            values[name] = {
                key: archive[key].copy() for key in ("frame_idx", "pose_xyz", "sources")
            }
    old, new = values["original"], values["directional"]
    checks.require(np.array_equal(old["frame_idx"], new["frame_idx"]), "different native frames")
    keep = (old["frame_idx"] >= 67630) & (old["frame_idx"] < 67785)
    frames = old["frame_idx"][keep]
    z = np.stack([v["pose_xyz"][keep, 0, :, 2] for v in (old, new)])
    views = np.stack([(v["sources"][keep, 0, :, 0] >= 0).sum(-1) for v in (old, new)])
    loss = (frames >= 67663) & (frames < 67696)
    checks.require(
        loss.sum() == 33 and (views[0, loss] >= 2).all() and (views[1, loss] < 2).all(),
        "fixed lost33 diagnosis differs",
    )
    output.mkdir(parents=True)
    with provenance("directional_mouth_depth_figure", output, name="figure") as record:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
        colors = ("#555555", "#0077BB")
        for i, label in enumerate(("Original", "Directional")):
            axes[0].plot(frames, z[i, :, 0], color=colors[i], label=label)
            axes[2].step(frames, views[i], where="mid", color=colors[i], label=label)
        for node, label, color in (
            (0, "Mouth", "#0077BB"),
            (1, "Chest", "#228833"),
            (2, "Tail", "#AA3377"),
        ):
            axes[1].plot(frames, z[1, :, node] - z[0, :, node], label=label, color=color)
        axes[0].set_ylabel("Mouth world z (mm)")
        axes[1].set_ylabel("Directional - original\nworld z (mm)")
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[2].set_ylabel("Supported mouth views")
        axes[2].set_yticks([0, 1, 2, 3])
        axes[2].set_ylim(-0.15, 3.2)
        axes[2].axhline(2, color="black", linewidth=0.6, linestyle=":")
        axes[2].set_xlabel("Native frame (166 fps)")
        for ax in axes:
            ax.axvspan(67663 - 0.5, 67696 - 0.5, color="#EE7733", alpha=0.12)
            ax.axvline(67647, color="#777777", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", alpha=0.18)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.legend(loc="upper right", ncols=3)
        axes[0].set_title(
            "S0: mouth depth and lost support at the same 68060-frame horizon\n"
            "Orange: 33 differential-loss frames. Dashed line: chest reacquisition."
        )
        fig.savefig(output / "mouth_depth.png", dpi=160)
        fig.savefig(output / "mouth_depth.svg")
        plt.close(fig)
        np.savez_compressed(
            output / "plotted_values.npz",
            frame_idx=frames,
            world_z=z,
            supported_mouth_views=views,
            differential_loss=loss,
        )
        summary = dict(
            status="ok",
            native_interval=[67630, 67785],
            same_horizon=68060,
            plotted_slot=0,
            independently_verified_ID=False,
            pose_refit=False,
            source_diagnosis_sha256=expected,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "figure input changed")
            record.input(path, path, sha256=digest)
        for path in output.iterdir():
            if path.is_file() and not path.name.endswith("provenance.json"):
                record.output(path.name, path, sha256=checks.sha(path))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
