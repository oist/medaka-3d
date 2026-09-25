"""Plot native-frame diagnostics for all eight reviewed comparison clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

COLORS = {"A": "#4b5563", "B": "#d97706", "C": "#0284c7"}


def finite_max(values):
    live = np.isfinite(values)
    result = np.max(np.where(live, values, -np.inf), axis=1)
    return np.where(live.any(axis=1), result, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/custom2d_comparison_v1/event_figures")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    review_path = args.analysis / "review_windows.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    protocol_path = args.analysis / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    c_label = "C: custom + deferred" if "deferred_config" in protocol else "C: conservative custom"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    entries = []
    with provenance("plot_custom2d_review_events", args.output) as rec:
        rec.input("review_windows", review_path, sha256=sha256(review_path))
        rec.input("protocol", protocol_path, sha256=sha256(protocol_path))
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        for event in review["rows"]:
            center = event["frame"]
            first, stop = event["native_frame_range_half_open"]
            fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
            loaded, hashes = {}, {}
            for arm in "ABC":
                path = args.analysis / event["case"] / f"{arm}.npz"
                hashes[arm] = sha256(path)
                rec.input(event["case"] + "_" + arm, path, sha256=hashes[arm])
                with np.load(path, allow_pickle=False) as data:
                    mask = (data["frames"] >= first) & (data["frames"] < stop)
                    loaded[arm] = {
                        k: data[k][mask]
                        for k in ("frames", "final__xyz", "final__source", "final__pose")
                    }
                held = loaded[arm]
                frames = held["frames"]
                xyz = held["final__xyz"].astype(float)
                steps = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
                pose = held["final__pose"].astype(float)
                length = np.linalg.norm(pose[:, :, 0] - pose[:, :, 2], axis=-1)
                label = {
                    "A": "A: saved SLEAP",
                    "B": "B: cut suspect links",
                    "C": c_label,
                }[arm]
                style = {"A": "-", "B": "--", "C": ":"}[arm]
                axes[0].plot(
                    frames[1:],
                    finite_max(steps),
                    color=COLORS[arm],
                    linestyle=style,
                    linewidth=1.9,
                    label=label,
                )
                axes[1].plot(
                    frames, finite_max(length), color=COLORS[arm], linestyle=style, linewidth=1.8
                )
                perm = event["arms"][arm]["comparison_permutation_only"]
                changed = np.count_nonzero(
                    loaded["A"]["final__source"] != held["final__source"][:, perm],
                    axis=(1, 2),
                )
                axes[2].step(
                    frames, changed, where="mid", color=COLORS[arm], linestyle=style, linewidth=1.8
                )
            axes[0].axhline(
                10,
                color="#b91c1c",
                linestyle="--",
                linewidth=0.8,
                label="10 mm diagnostic threshold",
            )
            axes[0].set_ylabel("Available-fish maximum step\n(mm / native frame)")
            axes[0].set_ylim(bottom=0)
            axes[0].legend(loc="upper right", fontsize=8, ncol=2)
            axes[1].set_ylabel("Available-fish maximum\nmouth-tail length (mm)")
            axes[1].set_ylim(bottom=0)
            axes[2].set_ylabel("Changed view-slot entries\nrelative to A (0-9)")
            axes[2].set_ylim(-0.3, 9.3)
            axes[2].set_yticks([0, 3, 6, 9])
            axes[2].set_xlabel("Native frame (166 fps); vertical line = review anchor")
            for axis in axes:
                axis.axvline(center, color="#64748b", linewidth=0.8, alpha=0.8)
                axis.grid(alpha=0.2)
                axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axes[2].set_xlim(first, stop - 1)
            fig.suptitle(
                f"f{center} | {event['human_kind']} | final stage",
                x=0.075,
                ha="left",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.075,
                0.022,
                "Native coordinates; maxima use finite fish only, without imputing missing fish.\n"
                "One fixed renaming per clip for source comparison. "
                "A is not ID truth; the anchor is not an exact swap time.",
                fontsize=9,
                color="#475569",
            )
            fig.tight_layout(rect=(0.03, 0.075, 1, 0.95), h_pad=1.4)
            output = args.output / f"f{center}_diagnostics.png"
            fig.savefig(output, dpi=140)
            plt.close(fig)
            digest = sha256(output)
            rec.output(output.name, output, sha256=digest)
            entries.append(
                dict(
                    path=output.name,
                    sha256=digest,
                    center_frame=center,
                    case=event["case"],
                    input_npz_sha256=hashes,
                    native_frame_range_half_open=[first, stop],
                )
            )
        manifest = args.output / "manifest.json"
        manifest.write_text(
            json.dumps(
                dict(
                    analysis_directory=args.analysis.as_posix(),
                    entries=entries,
                    coordinates_interpolated=False,
                    biological_identity_accuracy=False,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rec.output("manifest", manifest, sha256=sha256(manifest))
        print(manifest)


if __name__ == "__main__":
    main()
