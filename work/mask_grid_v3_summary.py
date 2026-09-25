"""Plot saved development mask-grid diagnostics, without re-running inference."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    root = REPO / "work/mask_grid_v3"
    out = root / "summary"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    with provenance("mask_grid_saved_data_summary", out, name="summary") as record:
        source = root / "diagnostic.json"
        rows = json.loads(source.read_text())["rows"]
        record.input("diagnostic", source, sha256=sha256(source))
        record.input("script", Path(__file__), sha256=sha256(Path(__file__)))
        frames = [row["frame"] for row in rows]
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, name in zip(axes[0], ("forward_S2", "refined_S2"), strict=True):
            for key, label, color in (
                ("two_view_voxels", ">=2 views (third may contradict)", "#d58a24"),
                ("strict_three_voxels", "strict 3 views", "#288957"),
            ):
                ax.plot(
                    frames,
                    [r["S2_context_regions"][name][key] for r in rows],
                    label=label,
                    color=color,
                )
            ax.set_title(f"Within 15 mm of {name}: diagnostic region only")
            ax.set_ylabel("2 mm grid cells")
            ax.legend(fontsize=8)
        axes[1, 0].plot(
            frames, [r["pool_strict_three"] for r in rows], label="strict 3", color="#288957"
        )
        axes[1, 0].plot(
            frames,
            [r["pool_known_third_contradiction"] for r in rows],
            label="2 views, known third contradiction",
            color="#d58a24",
        )
        axes[1, 0].set_title("Composition of the bounded 48-candidate pool")
        axes[1, 0].set_ylabel("candidate count")
        axes[1, 0].legend(fontsize=8)
        for name, color in (("forward_S2", "#cc4343"), ("refined_S2", "#3863c6")):
            axes[1, 1].plot(
                frames,
                [r["S2_context_regions"][name]["nearest_pool_distance_mm"] for r in rows],
                label=name,
                color=color,
            )
        axes[1, 1].set_title("Distance to nearest global pool candidate")
        axes[1, 1].set_ylabel("mm; centres are not pectoral landmarks")
        axes[1, 1].legend(fontsize=8)
        for ax in axes.ravel():
            ax.set_xlabel("native source frame")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(alpha=0.2)
        figure.suptitle("Known development episode: image-derived support, no identity verdict")
        path = out / "support_timeline.png"
        figure.savefig(path, dpi=145)
        plt.close(figure)
        record.output("support_timeline", path, sha256=sha256(path))
        data_path = root / "f333544_support.npz"
        record.input("frame_support", data_path, sha256=sha256(data_path))
        with np.load(data_path, allow_pickle=False) as data:
            pool, count = data["pool_xyz"], data["pool_count_supported"]
            queries, segments = data["query_pectoral_xyz"], data["healthy_segments_xyz"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, dims, names in ((axes[0], (0, 1), ("X", "Y")), (axes[1], (0, 2), ("X", "Z"))):
            for threshold, color, label in (
                (3, "#288957", "strict3 candidate"),
                (2, "#d58a24", "2-view candidate"),
            ):
                pts = pool[count == threshold]
                ax.scatter(pts[:, dims[0]], pts[:, dims[1]], s=20, c=color, alpha=0.75, label=label)
            for index, color, label in (
                (2, "#cc4343", "v1 S2 pectoral estimate"),
                (5, "#3863c6", "v2 H1 S2 pectoral estimate"),
            ):
                ax.scatter(
                    queries[index, dims[0]],
                    queries[index, dims[1]],
                    marker="x",
                    s=90,
                    linewidths=2,
                    c=color,
                    label=label,
                )
            for index, segment in enumerate(segments):
                ax.plot(
                    segment[:, dims[0]],
                    segment[:, dims[1]],
                    c="#444444",
                    lw=2,
                    label="supported S0/S1 segments" if index == 0 else None,
                )
            ax.set_xlabel(f"{names[0]} (mm)")
            ax.set_ylabel(f"{names[1]} (mm)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        figure.suptitle("333544: nonlocal spatial alternatives; possible silhouette ghosts remain")
        path = out / "f333544_spatial_pool.png"
        figure.savefig(path, dpi=145)
        plt.close(figure)
        record.output("spatial_pool", path, sha256=sha256(path))


if __name__ == "__main__":
    run()
