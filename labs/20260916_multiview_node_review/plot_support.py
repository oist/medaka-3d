"""Plot tail fit residuals alongside the actual supporting camera count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geometry import geometry_metrics
from PIL import Image

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

COLORS = {"baseline": "#2166ac", "proposal": "#c45b16"}
EVENT = 76988
SLOTS = (1, 2)


def finite_value(value):
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("work/multiview_node_review_v1/analysis/cg20s/evaluation.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("work/multiview_node_review_v1/review"))
    parser.add_argument(
        "--replace", action="store_true", help="Regenerate this named support figure"
    )
    args = parser.parse_args()
    stem = f"tail_support_{EVENT}"
    png, data_path = args.output / f"{stem}.png", args.output / f"{stem}.json"
    sidecar = args.output / f"{stem}.provenance.json"
    if not args.replace and any(path.exists() for path in (png, data_path, sidecar)):
        raise FileExistsError("The named support figure already exists; preserve prior artifacts")
    parent_path = args.evaluation.parent.parent / "multiview_node_reassessment.provenance.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    digest = sha256(args.evaluation)
    if digest != parent["outputs"]["cg20s/evaluation.npz"]["sha256"]:
        raise ValueError("Saved evaluation hash differs from the completed experiment")

    with provenance("tail_support_evidence", args.output, name=stem) as record:
        record.params_from(args)
        record.param("frame_interval_half_open", [76972, 77005])
        record.param("slots_zero_based", list(SLOTS))
        record.param("accepted_identity_corrections", 0)
        record.input("evaluation", args.evaluation, sha256=digest)
        record.input("parent_provenance", parent_path, sha256=sha256(parent_path))
        record.input("plot_code", __file__, sha256=sha256(Path(__file__)))
        for key in ("geometry.py", "calibration", "cam0", "cam1", "cam2"):
            path = Path(parent["inputs"][key]["path"])
            if sha256(path) != parent["inputs"][key]["sha256"]:
                raise ValueError(f"Calibration or geometry implementation changed: {key}")
            record.input(key, path, sha256=sha256(path))
        rig = load_rig(Path(parent["inputs"]["calibration"]["path"]).parent)
        with np.load(args.evaluation, allow_pickle=False) as data:
            selected = (data["frames"] >= 76972) & (data["frames"] < 77005)
            frames = data["frames"][selected]
            source = {key: data[key][selected] for key in ("original_source", "candidate_source")}
            originals = data["original_uv"][selected]
            proposals = data["candidate_uv"][selected]
            saved_ray = {
                "baseline": data["baseline_ray_mm"][selected],
                "proposal": data["candidate_ray_mm"][selected],
            }
        if not np.array_equal(frames, np.arange(76972, 77005)):
            raise ValueError("Support plot requires every native frame in the requested interval")
        curves = {}
        for label, uv in (("baseline", originals), ("proposal", proposals)):
            geometry = geometry_metrics(rig, uv)
            np.testing.assert_allclose(
                geometry["ray_rms_mm"], saved_ray[label], rtol=1e-8, atol=1e-8, equal_nan=True
            )
            curves[label] = {
                "ray_rms_mm": geometry["ray_rms_mm"][:, SLOTS, 2],
                "view_count": geometry["view_counts"][:, SLOTS, 2],
                # np.max preserves NaN whenever any held-view score is absent;
                # do not turn an unavailable three-view check into a partial one.
                "held_view_max_px": np.max(
                    geometry["leave_one_out_px"][:, list(SLOTS)][..., 2], axis=-1
                ),
            }
        event_row = int(np.flatnonzero(frames == EVENT)[0])
        np.testing.assert_array_equal(curves["baseline"]["view_count"][event_row], [3, 2])
        np.testing.assert_array_equal(curves["proposal"]["view_count"][event_row], [2, 3])
        if not np.isnan(curves["proposal"]["held_view_max_px"][event_row, 0]):
            raise AssertionError("S1 proposal must have no three-view held-out score")

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 12,
                "axes.labelsize": 12,
                "axes.titlesize": 15,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            }
        )
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey="row")
        fig.subplots_adjust(left=0.085, right=0.98, top=0.86, bottom=0.20, hspace=0.26, wspace=0.12)
        fig.suptitle(
            "Tail correspondence and camera support at frame 76988",
            x=0.085,
            y=0.977,
            ha="left",
            fontsize=19,
            fontweight="bold",
        )
        fig.text(
            0.085,
            0.94,
            "K_1_2  |  Native frames 76972-77004 at 166 fps  |  Unaccepted proposal",
            fontsize=12,
            color="#444444",
        )
        metrics = ("ray_rms_mm", "view_count", "held_view_max_px")
        labels = ("Tail ray RMS (mm)", "Tail supporting views", "Tail held-view max (px)")
        limits = ((0, 9.5), (-0.1, 3.45), (0, 135))
        for row, (metric, ylabel, ylim) in enumerate(zip(metrics, labels, limits, strict=True)):
            for column, slot in enumerate(SLOTS):
                ax = axes[row, column]
                ax.axvspan(EVENT - 0.45, EVENT + 0.45, color="#e8e8e8", zorder=0)
                ax.axvline(EVENT, color="#555555", linestyle=":", linewidth=1.1, zorder=1)
                for label, marker, style in (("baseline", "o", "-"), ("proposal", "D", "--")):
                    ax.plot(
                        frames,
                        curves[label][metric][:, column],
                        color=COLORS[label],
                        linestyle=style,
                        marker=marker,
                        markersize=3.8,
                        linewidth=1.5,
                        markerfacecolor="white" if label == "baseline" else COLORS[label],
                        label="Baseline" if label == "baseline" else "Proposal (unaccepted)",
                        drawstyle="steps-mid" if metric == "view_count" else "default",
                    )
                ax.set_ylim(*ylim)
                ax.set_xlim(frames[0], frames[-1])
                ax.set_xticks([76972, 76980, 76988, 76996, 77004])
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                ax.grid(axis="y", color="#dddddd", linewidth=0.65)
                ax.spines[["top", "right"]].set_visible(False)
                if row == 0:
                    ax.set_title(f"Slot S{slot}: tail", pad=12, fontweight="bold")
                if column == 0:
                    ax.set_ylabel(ylabel)
                if metric == "view_count":
                    ax.set_yticks([0, 1, 2, 3])
                if row == 2:
                    ax.set_xlabel("Native source frame")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.08, 0.925),
            frameon=False,
            ncol=2,
            fontsize=12,
        )
        boxes = dict(facecolor="white", edgecolor="none", alpha=0.94, pad=3)
        axes[0, 0].text(
            0.04,
            0.58,
            "8.29 → 0.69 mm\n3-view fit → 2-view fit",
            transform=axes[0, 0].transAxes,
            fontsize=11,
            bbox=boxes,
        )
        axes[0, 1].text(
            0.05,
            0.65,
            "0.32 → 1.21 mm\n2-view fit → 3-view fit",
            transform=axes[0, 1].transAxes,
            fontsize=11,
            bbox=boxes,
        )
        axes[1, 0].text(
            0.54,
            0.19,
            "At 76988: 3 → 2 views\nSide2 tail is absent",
            transform=axes[1, 0].transAxes,
            fontsize=11,
            bbox=boxes,
        )
        axes[1, 1].text(
            0.54,
            0.19,
            "At 76988: 2 → 3 views\nSame Side2 tail is added",
            transform=axes[1, 1].transAxes,
            fontsize=11,
            bbox=boxes,
        )
        axes[2, 0].text(
            0.04,
            0.62,
            "At 76988: proposal unavailable\nNo three-view held-out check",
            transform=axes[2, 0].transAxes,
            fontsize=11,
            color=COLORS["proposal"],
            bbox=boxes,
        )
        axes[2, 1].text(
            0.04,
            0.66,
            "At 76988: baseline unavailable\nProposal max: 17.08 px",
            transform=axes[2, 1].transAxes,
            fontsize=11,
            bbox=boxes,
        )
        fig.text(
            0.085,
            0.083,
            "At 76988 the evidence set changes: S1 loses a tail view while S2 gains it.\n"
            "The lower S1 fit error does not demonstrate recovered three-view agreement.",
            fontsize=12,
            fontweight="bold",
            linespacing=1.55,
        )
        fig.text(
            0.085,
            0.031,
            "Held-view scores require all 3 views; gaps remain NaN, with no interpolation.  "
            "S1/S2 are slots, not confirmed biological identities.\n"
            "Source: cg20s/evaluation.npz. Geometry and source-row changes remain unaccepted.",
            fontsize=10.5,
            color="#444444",
            linespacing=1.4,
        )
        fig.savefig(png, dpi=170, facecolor="white")
        plt.close(fig)
        with Image.open(png) as image:
            image.load()
            dimensions = list(image.size)
            if min(dimensions) < 1500:
                raise AssertionError("Figure resolution is too small for the six-panel plot")
        rows = []
        for i, frame in enumerate(frames):
            for column, slot in enumerate(SLOTS):
                rows.append(
                    {
                        "frame": int(frame),
                        "slot": slot,
                        **{
                            f"{label}_{metric}": finite_value(values[metric][i, column])
                            for label, values in curves.items()
                            for metric in metrics
                        },
                        "original_tail_owners": source["original_source"][i, slot, :, 2].tolist(),
                        "proposal_tail_owners": source["candidate_source"][i, slot, :, 2].tolist(),
                    }
                )
        artifact = {
            "event_frame": EVENT,
            "frame_interval_half_open": [76972, 77005],
            "png": str(png),
            "png_dimensions": dimensions,
            "source": str(args.evaluation),
            "source_sha256": digest,
            "view_count_definition": "Finite physically valid camera rays for this tail node",
            "held_view_definition": (
                "Maximum of all three leave-one-view-out errors; NaN if any unavailable"
            ),
            "interpretation": (
                "Evidence set changes; lower S1 fit is not recovered three-view agreement"
            ),
            "accepted_identity_corrections": 0,
            "event_rows": [row for row in rows if row["frame"] == EVENT],
            "rows": rows,
        }
        data_path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
        reread = json.loads(data_path.read_text(encoding="utf-8"))
        assert len(reread["rows"]) == 66
        assert reread["event_rows"][0]["proposal_held_view_max_px"] is None
        for path in (png, data_path):
            record.output(path.name, path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    "png": str(png),
                    "json": str(data_path),
                    "dimensions": dimensions,
                    "event_rows": artifact["event_rows"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
