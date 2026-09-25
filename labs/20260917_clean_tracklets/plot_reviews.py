"""Native-frame diagnostics for the eight human-review anchors (not ID ground truth)."""

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

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean_tracklets_v1"
PREVIOUS = ROOT / "work/custom2d_comparison_v1/analysis_v3"
REVIEWS = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
COLORS = {"saved_C34": "#44546a", "previous_custom_C_v3": "#b17616", "new": "#158875"}
LABELS = {"saved_C34": "Saved C34", "previous_custom_C_v3": "Previous C v3", "new": "New F -> 3D"}
SLOTS = ["#ffbe53", "#53d8ea", "#f784ba"]


def read_stage(path, prefix=""):
    with np.load(path, allow_pickle=False) as z:
        return {k.removeprefix(prefix): z[k].copy() for k in z.files if k.startswith(prefix)}


def finite_max(values):
    good = np.isfinite(values)
    return np.where(good.any(-1), np.max(np.where(good, values, -np.inf), axis=-1), np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    args = parser.parse_args()
    base, dest = args.root.resolve(), args.root.resolve() / "event_figures"
    dest.mkdir(parents=True, exist_ok=True)
    events = json.loads(REVIEWS.read_text(encoding="utf-8"))["events"]
    entries, window_metrics = [], []
    with provenance("clean_tracklets_review_figures", dest, name="figures") as rec:
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("human_review_anchors_only", REVIEWS, sha256=sha256(REVIEWS))
        for event in events:
            frame = event["frame"]
            case = "cg20s" if 75270 <= frame < 78590 else f"known_{frame}"
            target = base / case / "reconstruction.npz"
            summary_path = base / case / "summary.json"
            if not target.is_file() or not summary_path.is_file():
                continue
            new = read_stage(target)
            frames = new.pop("frames")
            stages = {
                "saved_C34": read_stage(PREVIOUS / case / "saved_baseline.npz", "final__"),
                "previous_custom_C_v3": read_stage(PREVIOUS / case / "C.npz", "final__"),
                "new": new,
            }
            input_paths = [
                target,
                summary_path,
                PREVIOUS / case / "saved_baseline.npz",
                PREVIOUS / case / "C.npz",
            ]
            hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in input_paths}
            for p in input_paths:
                rec.input(f"{frame}_{p.name}", p, sha256=sha256(p))
            selected = (frames >= frame - 83) & (frames < frame + 83)
            f = frames[selected]
            if len(f) != 166 or np.any(np.diff(f) != 1):
                raise ValueError(f"Review clip {frame} must preserve all 166 native frames")
            fig, ax = plt.subplots(5, 1, figsize=(13.2, 11.0), sharex=True, layout="constrained")
            for key, data in stages.items():
                xyz, pose = data["xyz"][selected], data["pose"][selected]
                present = np.isfinite(xyz).all(-1)
                step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
                held = data.get("identity_review")
                review = held[selected] if held is not None else None
                ax[0].plot(
                    f[1:],
                    finite_max(step),
                    color=COLORS[key],
                    label=LABELS[key],
                    lw=1.5,
                    alpha=0.88,
                )
                ax[1].step(f, present.sum(-1), color=COLORS[key], where="mid", lw=1.5, alpha=0.8)
                finite = step[np.isfinite(step)]
                window_metrics.append(
                    dict(
                        frame=frame,
                        case=case,
                        condition=LABELS[key],
                        condition_id=key,
                        native_frame_start=int(f[0]),
                        native_frame_stop_exclusive=int(f[-1] + 1),
                        steps_over_4mm=int((step > 4).sum()),
                        steps_over_10mm=int((step > 10).sum()),
                        maximum_step_mm=float(finite.max()) if len(finite) else None,
                        observed_native_steps=int(np.isfinite(step).sum()),
                        observed_fish_frames=int(present.sum()),
                        possible_fish_frames=int(present.size),
                        missing_fish_frames=int((~present).sum()),
                        identity_review_fish_frames=int(review.sum())
                        if review is not None
                        else None,
                        measured_and_review_fish_frames=int((review & present).sum())
                        if review is not None
                        else None,
                    )
                )
            ax[0].axhline(10, color="#ad3030", ls=":", lw=1, label="10 mm diagnostic")
            ax[0].set_ylabel("Max 3D step\n(mm / native frame)")
            ax[0].legend(loc="upper right", ncols=4, fontsize=8)
            ax[1].set_ylabel("Observed 3D fish")
            ax[1].set_yticks([0, 1, 2, 3])
            review = stages["new"]["identity_review"][selected]
            present = np.isfinite(stages["new"]["xyz"][selected]).all(-1)
            ax[2].step(
                f,
                review.sum(-1),
                where="mid",
                color="#b83136",
                label="New: unresolved identity (includes missing)",
            )
            ax[2].step(
                f,
                (review & present).sum(-1),
                where="mid",
                color="#e47b53",
                ls="--",
                label="New: measured but unresolved identity",
            )
            ax[2].set_ylabel("New: ID review\n(fish)")
            ax[2].set_yticks([0, 1, 2, 3])
            ax[2].legend(loc="upper right", ncols=2, fontsize=8)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            alignment = summary["display_alignment"]
            order = alignment["permutation"] if alignment.get("valid") else [0, 1, 2]
            pose = stages["new"]["pose"][selected][:, order]
            for i, segment in enumerate(((0, 1), (1, 2))):
                length = np.linalg.norm(pose[:, :, segment[0]] - pose[:, :, segment[1]], axis=-1)
                for slot in range(3):
                    ax[3 + i].plot(
                        f, length[:, slot], color=SLOTS[slot], lw=1.4, label=f"New S{slot}"
                    )
                ax[3 + i].set_ylabel(
                    ("Mouth-pectoral" if i == 0 else "Pectoral-tail") + "\nlength (mm)"
                )
                ax[3 + i].legend(loc="upper right", ncols=3, fontsize=8)
            for a in ax:
                a.axvline(frame, color="#b83136", lw=0.8, ls="--")
                a.grid(alpha=0.18)
                a.set_xlim(f[0], f[-1])
                a.ticklabel_format(axis="x", style="plain", useOffset=False)
                a.spines[["top", "right"]].set_visible(False)
            ax[-1].set_xlabel("Native frame (166 fps; no interpolation)")
            fig.suptitle(
                f"{case} / human-review anchor f{frame} / all 166 native frames\n"
                "ID review is an abstention flag, not a confirmed swap; "
                "historical outputs have no comparable flag",
                fontsize=12,
            )
            path = dest / f"f{frame}_diagnostics.png"
            fig.savefig(path, dpi=150, facecolor="white")
            plt.close(fig)
            rec.output(str(frame), path, sha256=sha256(path))
            entries.append(
                dict(
                    case=case,
                    center_frame=frame,
                    kind="image",
                    path=path.name,
                    sha256=sha256(path),
                    native_frame_range_half_open=[int(f[0]), int(f[-1] + 1)],
                    input_sha256=hashes,
                    display_alignment=alignment,
                    scope=(
                        "Mechanical diagnostics around a historical review anchor; "
                        "not new human ground truth"
                    ),
                )
            )
        out = dest / "manifest.json"
        out.write_text(
            json.dumps(
                dict(
                    analysis_directory=str(base.relative_to(ROOT)).replace("\\", "/"),
                    entries=entries,
                    window_metrics=window_metrics,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rec.output("manifest", out, sha256=sha256(out))
    print(json.dumps(dict(figures=len(entries), window_rows=len(window_metrics))))


if __name__ == "__main__":
    main()
