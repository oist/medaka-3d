"""Compare preserved generative runs on identical frames, without refitting.

Chart contract: static Matplotlib small multiples answer whether anatomy and
observational support differ between v2 and v3, at native-frame resolution.
The takeaway is measured after loading completed artifacts, never prewritten.
Columns are versions; rows are segment ratios, node support, and ID review.
Three fish colors use blue/gold/pink, with circle/triangle/square markers as a
non-color distinction. Shared scales, dashed latent and solid observed segments.

Example (from repository root):
  .venv/Scripts/python.exe work/generative_tracking/plot_version_diagnostics.py \
    --v2 work/generative_tracking/transient_333519-333719 \
    --v3 work/generative_tracking/anatomy_v3_transient_333519-333719 \
    --out work/generative_tracking/transient_v2_v3_diagnostic --event 333619
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

COLORS = ("#2878A8", "#B88618", "#AA4F84")
MARKERS = ("o", "^", "s")
SEGMENTS = ("Mouth-pectoral", "Pectoral-tail")
NODES = ("mouth", "pectoral", "tail")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Run:
    label: str
    directory: Path
    frames: np.ndarray
    pose: np.ndarray
    observed: np.ndarray
    review: np.ndarray
    seed: np.ndarray
    provenance: dict

    @property
    def lengths(self):
        return np.linalg.norm(np.diff(self.pose, axis=2), axis=-1)

    @property
    def seed_lengths(self):
        return np.linalg.norm(np.diff(self.seed, axis=1), axis=-1)

    @property
    def ratios(self):
        return self.lengths / self.seed_lengths

    @property
    def measured_segments(self):
        return self.observed[:, :, :-1] & self.observed[:, :, 1:]

    @property
    def accepted_segments(self):
        return self.measured_segments & ~self.review[:, :, None]


def read_run(path: Path, label: str) -> Run:
    provenance_path = path / "benchmark.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "ok":
        raise ValueError(f"{label}: benchmark has not completed successfully")
    seed = np.asarray(provenance["params"]["seed"]["pose_xyz"], dtype=float)
    with np.load(path / "candidate.npz", allow_pickle=False) as data:
        result = Run(label, path, data["frame_idx"].copy(), data["pose_xyz"].copy(),
                     data["observed"].copy(), data["review"].copy(), seed, provenance)
    expected = np.arange(provenance["params"]["start"], provenance["params"]["stop"])
    if not np.array_equal(result.frames, expected):
        raise ValueError(f"{label}: result is not the complete declared native-frame window")
    if result.pose.shape != (len(expected), 3, 3, 3):
        raise ValueError(f"{label}: unexpected pose shape")
    if result.observed.shape != (len(expected), 3, 3) or result.review.shape != (len(expected), 3):
        raise ValueError(f"{label}: unexpected observation/review shape")
    if not np.isfinite(result.seed).all() or np.any(result.seed_lengths <= 0):
        raise ValueError(f"{label}: a valid seed is needed for length ratios")
    return result


def masked_maximum(values, mask):
    out = []
    for fish in range(3):
        per_segment = []
        for segment in range(2):
            selected = values[:, fish, segment][mask[:, fish, segment]]
            per_segment.append(float(selected.max()) if selected.size else None)
        out.append(per_segment)
    return out


def run_summary(run):
    return {
        "directory": str(run.directory.resolve()),
        "candidate_sha256": sha256(run.directory / "candidate.npz"),
        "provenance_sha256": sha256(run.directory / "benchmark.provenance.json"),
        "seed_lengths_mm_by_slot_segment": run.seed_lengths.tolist(),
        "maximum_latent_length_mm_by_slot_segment": run.lengths.max(axis=0).tolist(),
        "maximum_latent_ratio_by_slot_segment": run.ratios.max(axis=0).tolist(),
        "maximum_observed_ratio_by_slot_segment": masked_maximum(run.ratios, run.measured_segments),
        "maximum_accepted_ratio_by_slot_segment": masked_maximum(run.ratios, run.accepted_segments),
        "observed_node_frames_by_slot_node": run.observed.sum(axis=0).tolist(),
        "review_frames_by_slot": run.review.sum(axis=0).tolist(),
        "accepted_pectoral_frames_by_slot": (
            run.observed[:, :, 1] & ~run.review).sum(axis=0).tolist(),
        "latent_segment_frames_above_2x": int((run.ratios > 2.0 + 1e-8).sum()),
        "observed_segment_frames_above_2x": int(((run.ratios > 2.0 + 1e-8)
                                                  & run.measured_segments).sum()),
        "accepted_segment_frames_above_2x": int(((run.ratios > 2.0 + 1e-8)
                                                  & run.accepted_segments).sum()),
        "tracking_seconds": json.loads((run.directory / "comparison.json").read_text(
            encoding="utf-8"))["tracking_seconds"],
    }


def write_csv(path: Path, runs):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["version", "frame", "seconds_from_window_start", "anonymous_slot",
                         "segment", "length_mm", "seed_length_mm", "length_to_seed_ratio",
                         "both_nodes_observed", "identity_review", "accepted_segment",
                         "mouth_observed", "pectoral_observed", "tail_observed"])
        for run in runs:
            lengths, ratios = run.lengths, run.ratios
            measured, accepted = run.measured_segments, run.accepted_segments
            for t, frame in enumerate(run.frames):
                for fish in range(3):
                    for segment in range(2):
                        writer.writerow([
                            run.label, int(frame), (frame - run.frames[0]) / 166, f"S{fish}",
                            SEGMENTS[segment], lengths[t, fish, segment],
                            run.seed_lengths[fish, segment], ratios[t, fish, segment],
                            int(measured[t, fish, segment]), int(run.review[t, fish]),
                            int(accepted[t, fish, segment]),
                            *run.observed[t, fish].astype(int).tolist(),
                        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output basename, no extension")
    parser.add_argument("--event", type=int, action="append", default=[])
    args = parser.parse_args()
    runs = [read_run(args.v2, "v2"), read_run(args.v3, "v3")]
    if not np.array_equal(runs[0].frames, runs[1].frames):
        raise ValueError("versions must contain exactly the same frames")
    np.testing.assert_allclose(runs[0].seed, runs[1].seed, atol=1e-10, rtol=0)
    frames = runs[0].frames
    figsize = (14, 11)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": "#30343B", "text.color": "#30343B"})
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.8, 1.8, 2.1, 0.9]})
    for column, run in enumerate(runs):
        for segment in range(2):
            ax = axes[segment, column]
            for fish, color in enumerate(COLORS):
                values = run.ratios[:, fish, segment]
                ax.plot(frames, values, color=color, lw=1.1, ls="--", alpha=0.6,
                        marker=MARKERS[fish], markersize=3, markerfacecolor="none",
                        markevery=max(1, len(frames) // 15))
                measured = np.where(run.measured_segments[:, fish, segment], values, np.nan)
                ax.plot(frames, measured, color=color, lw=1.65,
                        marker=MARKERS[fish], markersize=3, markevery=max(1, len(frames) // 15))
            ax.axhline(1, color="#7A7F86", lw=0.8, alpha=0.5)
            ax.axhline(2, color="#4A4E54", lw=1.0, ls="-.")
            maximum = max(float(other.ratios[:, :, segment].max()) for other in runs)
            ax.set_ylim(0, max(2.25, maximum * 1.06))
            ax.grid(axis="y", color="#DFE3E7", lw=0.6)
            ax.set_title(f"{run.label}: {SEGMENTS[segment]}", loc="left", fontsize=11)
            if column == 0:
                ax.set_ylabel("Length / initialization length")
        extent = (frames[0] - 0.5, frames[-1] + 0.5, 8.5, -0.5)
        axes[2, column].imshow(run.observed.reshape(len(frames), 9).T, aspect="auto",
                               interpolation="nearest", cmap=ListedColormap(["#E4E7EB", COLORS[0]]),
                               vmin=0, vmax=1, extent=extent)
        axes[2, column].set_yticks(
            range(9), [f"S{fish} {node}" for fish in range(3) for node in NODES]
        )
        axes[2, column].set_title("Node support: blue = at least 2 views; gray = latent only",
                                  loc="left", fontsize=9)
        for separator in [2.5, 5.5]:
            axes[2, column].axhline(separator, color="white", lw=2)
        axes[3, column].imshow(run.review.T, aspect="auto", interpolation="nearest",
                               cmap=ListedColormap(["#E4E7EB", COLORS[1]]), vmin=0, vmax=1,
                               extent=(frames[0] - 0.5, frames[-1] + 0.5, 2.5, -0.5))
        axes[3, column].set_yticks(range(3), ["S0", "S1", "S2"])
        axes[3, column].set_title("Identity review: gold = withheld; gray = no review flag",
                                  loc="left", fontsize=9)
        axes[3, column].set_xlabel("Native video frame (166 frames/s)")
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            for event in args.event:
                ax.axvline(event, color="#50545A", lw=0.9, ls=":")
    fig.suptitle(f"Generative tracking diagnostics: v2 and v3, frames {frames[0]}-{frames[-1]}",
                 x=0.08, y=0.985, ha="left", fontsize=15, weight="bold")
    subtitle = (f"K_1_2; one trio; {len(frames)} matched frames ({len(frames)/166:.2f} s). "
                "Shared initialization and plot scales; no refitting.")
    fig.text(0.08, 0.956, subtitle, fontsize=10)
    handles = [Line2D([], [], color=color, marker=MARKERS[i], lw=1.5, label=f"S{i}")
               for i, color in enumerate(COLORS)]
    handles += [Line2D([], [], color="#50545A", lw=1.5, label="Both segment nodes observed"),
                Line2D([], [], color="#50545A", lw=1.1, ls="--", label="Latent segment"),
                Line2D([], [], color="#50545A", lw=1.0, ls="-.", label="v3 engineering guard: 2x")]
    if args.event:
        handles.append(Line2D([], [], color="#50545A", lw=0.9, ls=":", label="Prior noted event"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.946),
               ncol=len(handles), frameon=False, fontsize=8)
    fig.text(0.08, 0.014,
             "Solid lines establish node support, not correct identity. Accepted data also require "
             "no review flag.\nSlots S0-S2 are anonymous; these diagnostics do not validate male "
             "identity. The 2x cap is an engineering guard relative to the seed, "
             "not a species anatomy model.",
             fontsize=8, color="#555B64")
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.92), h_pad=1.5, w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.out.with_suffix(f".{extension}"), dpi=170)
    write_csv(args.out.with_suffix(".csv"), runs)
    config_v2, config_v3 = [run.provenance["params"]["tracker_config"] for run in runs]
    summary = {
        "analysis": "post-hoc artifact comparison, no refitting or tuning",
        "frame_start": int(frames[0]), "frame_stop_exclusive": int(frames[-1] + 1),
        "native_fps": 166, "frames_per_slot": len(frames),
        "n_groups": 1, "slots_are_anonymous": True, "node_order": NODES,
        "previously_noted_event_frames": args.event,
        "segment_order": SEGMENTS, "seed_is_common": True,
        "observation_definition": "each node supported by at least two camera rays",
        "acceptance_definition": "both segment nodes observed and no identity review flag",
        "identity_validation": "not established by shape, continuity, or review flags",
        "code_sha256": sha256(Path(__file__)),
        "runs": {run.label: run_summary(run) for run in runs},
        "configuration_differences": {key: {"v2": config_v2.get(key), "v3": config_v3.get(key)}
            for key in sorted(config_v2.keys() | config_v3.keys())
            if config_v2.get(key) != config_v3.get(key)},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.out.with_suffix(".png")),
                      "summary": str(args.out.with_suffix(".json"))}))


if __name__ == "__main__":
    main()
