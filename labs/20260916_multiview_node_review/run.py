"""Reevaluate the previously frozen bidirectional windows; preserve C34 inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from candidates import fit_flank, predictions, propose
from geometry import geometry_metrics

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig
from medaka3d.eval.generative import observed_motion_summary
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


class DetectionCache:
    def __init__(self, root, record):
        self.data = []
        for view in VIEWS:
            path = root / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
            record.input(view, path, sha256=sha256(path))
            with np.load(path, allow_pickle=False) as a:
                if (
                    str(a["experiment"]) != EXPERIMENT
                    or str(a["view"]) != view
                    or tuple(a["nodes"]) != ("mouth", "pectoral", "tail")
                ):
                    raise ValueError("Detection cache metadata mismatch")
                f, points = a["frame_idx"], a["points"]
                if np.any(np.diff(f) < 0):
                    raise ValueError("Unsorted detection cache")
                counts = np.bincount(f, minlength=348600)
                self.data.append((points, np.r_[0, np.cumsum(counts)]))

    def frame(self, frame):
        return [p[offset[frame] : offset[frame + 1]] for p, offset in self.data]


def suspect_mask(frames, episodes):
    mask = np.zeros(len(frames), bool)
    for group in episodes:
        mask |= (frames >= group["start_frame"]) & (frames < group["stop_frame_exclusive"])
    return mask


def spans(mask):
    borders = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(borders[::2], borders[1::2], strict=True))


def body_reference(metrics, tested, fallback=None):
    good = (
        ~tested[:, None]
        & np.isfinite(metrics["pose"]).all(axis=(-2, -1))
        & (metrics["ray_rms_mm"] <= 1).all(-1)
    )
    lengths = np.stack([metrics["mp_mm"], metrics["pt_mm"]], axis=-1)[good]
    if len(lengths) < 30:
        if fallback is None:
            raise ValueError("Primary outside-window body reference has fewer than30samples")
        return dict(**fallback, fallback_from_primary=True)
    median = np.median(lengths, axis=0)
    scale = np.maximum(3 * np.median(np.abs(lengths - median), axis=0), 0.75)
    return dict(
        median=median, scale=scale, sample_fish_frames=len(lengths), fallback_from_primary=False
    )


def paired_summary(original, candidate, mask):
    a, b = original[mask], candidate[mask]
    valid = np.isfinite(a) & np.isfinite(b)
    return dict(
        count=int(valid.sum()),
        original_median=float(np.median(a[valid])) if valid.any() else None,
        candidate_median=float(np.median(b[valid])) if valid.any() else None,
        original_p95=float(np.percentile(a[valid], 95)) if valid.any() else None,
        candidate_p95=float(np.percentile(b[valid], 95)) if valid.any() else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous", type=Path, default=Path("work/bidirectional_reconnect_v1/analysis_v2")
    )
    parser.add_argument("--inputs", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument(
        "--output", type=Path, default=Path("work/multiview_node_review_v1/analysis")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("multiview_node_reassessment", args.output) as record:
        record.params_from(args)
        for name in ("README.md", "run.py", "candidates.py", "geometry.py"):
            path = Path(__file__).with_name(name)
            record.input(name, path, sha256=sha256(path))
        old_protocol = args.previous / "protocol.json"
        record.input("selected_cases", old_protocol, sha256=sha256(old_protocol))
        protocol = json.loads(old_protocol.read_text(encoding="utf-8"))
        write_json(
            args.output / "protocol.json",
            dict(
                parent_protocol=protocol,
                decision_rule=(Path(__file__).with_name("README.md")).read_text(encoding="utf-8"),
                accepted_identity_corrections=0,
            ),
        )
        cache = DetectionCache(args.inputs, record)
        calib_dir = args.inputs / CALIBRATION
        manifest_path = calib_dir / "calibration_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record.input("calibration", manifest_path, sha256=sha256(manifest_path))
        for entry in manifest["model_files"]:
            path = calib_dir / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Calibration hash mismatch")
            record.input(entry["camera"], path, sha256=sha256(path))
        rig = load_rig(calib_dir)
        previous_provenance = json.loads(
            (args.previous / "bidirectional_development.provenance.json").read_text(
                encoding="utf-8"
            )
        )
        report = dict(
            experiment=EXPERIMENT,
            cases=[],
            accepted_identity_corrections=0,
            scope="local unaccepted reconstruction hypotheses; no identity accuracy claim",
        )
        primary_prior = None
        for case in protocol["cases"]:
            name = case["name"]
            print(f"Reevaluating {name}", flush=True)
            directory = args.previous / name
            for filename in ("poses.npz", "analysis.json"):
                path = directory / filename
                digest = sha256(path)
                if digest != previous_provenance["outputs"][f"{name}/{filename}"]["sha256"]:
                    raise ValueError(f"Parent output hash mismatch: {path}")
                record.input(f"{name}/{filename}", path, sha256=digest)
            with np.load(directory / "poses.npz", allow_pickle=False) as a:
                old = {key: a[key] for key in a.files}
            diagnosis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
            frames, uv, original_pose = old["frames"], old["uv"], old["pose"]
            tested = suspect_mask(frames, diagnosis["episodes"])
            baseline = geometry_metrics(rig, uv, frozen_pose=original_pose)
            prior = body_reference(baseline, tested, primary_prior)
            if name == "cg20s":
                primary_prior = prior
            candidate_uv = uv.copy()
            original_source = np.repeat(old["source"][..., None], 3, axis=-1)
            candidate_source = original_source.copy()
            candidate_pose = original_pose.copy()
            score_original = np.full(len(frames), np.nan)
            score_candidate = score_original.copy()
            changed = np.zeros(len(frames), bool)
            supported = changed.copy()
            frame_details = []
            window_details = []
            for lo, hi in spans(tested):
                left, right = (
                    original_pose[max(0, lo - 17) : lo, :, 1].copy(),
                    original_pose[hi : hi + 17, :, 1].copy(),
                )
                left[tested[max(0, lo - 17) : lo]] = np.nan
                right[tested[hi : hi + 17]] = np.nan
                forward = fit_flank(left, frames[max(0, lo - 17) : lo])
                backward = fit_flank(right, frames[hi : hi + 17])
                window_details.append(
                    dict(
                        first=int(frames[lo]),
                        stop=int(frames[hi - 1]) + 1,
                        forward=forward,
                        backward=backward,
                    )
                )
                for t in range(lo, hi):
                    owners, proposed_uv, metrics, detail = propose(
                        rig,
                        old["source"][t],
                        cache.frame(frames[t]),
                        prior,
                        predictions(forward, frames[t]),
                        predictions(backward, frames[t]),
                    )
                    candidate_uv[t] = proposed_uv
                    candidate_source[t] = owners
                    candidate_pose[t] = metrics["pose"]
                    changed[t] = np.any(owners != original_source[t])
                    if not changed[t]:
                        candidate_pose[t] = original_pose[t]
                    supported[t] = detail["review_worthy"]
                    score_original[t], score_candidate[t] = (
                        detail["original_cost"],
                        detail["candidate_cost"],
                    )
                    changes = np.argwhere(owners != original_source[t])
                    detail.update(
                        frame=int(frames[t]),
                        changed=bool(changed[t]),
                        forward_valid=[m["valid"] for m in forward],
                        backward_valid=[m["valid"] for m in backward],
                        source_changes=[
                            dict(
                                slot=int(s),
                                view=VIEWS[v],
                                node=("mouth", "pectoral", "tail")[n],
                                original_row=int(original_source[t, s, v, n]),
                                candidate_row=int(owners[s, v, n]),
                            )
                            for s, v, n in changes
                        ],
                    )
                    frame_details.append(detail)
            candidate = geometry_metrics(rig, candidate_uv, frozen_pose=candidate_pose)
            core = old["core"]
            tested_core = tested & core
            ray_before, ray_after = baseline["ray_rms_mm"], candidate["ray_rms_mm"]
            summary = dict(
                case=case,
                tested_frames=int(tested_core.sum()),
                changed_frames=int((changed & core).sum()),
                review_worthy_frames=int((supported & core).sum()),
                original_episode_count=len(diagnosis["episodes"]),
                merged_windows=len(window_details),
                families=dict(Counter(d["family"] for d in frame_details if d["changed"])),
                ray_mm=paired_summary(ray_before, ray_after, tested_core),
                reprojection_px=paired_summary(
                    baseline["frozen_reprojection_px"],
                    candidate["frozen_reprojection_px"],
                    tested_core,
                ),
                loo_px=paired_summary(
                    baseline["leave_one_out_px"], candidate["leave_one_out_px"], tested_core
                ),
                baseline_observed_nodes=int(np.isfinite(original_pose[tested_core]).all(-1).sum()),
                candidate_observed_nodes=int(
                    np.isfinite(candidate_pose[tested_core]).all(-1).sum()
                ),
                baseline_motion=observed_motion_summary(original_pose[core, :, 1], frames[core]),
                candidate_motion=observed_motion_summary(candidate_pose[core, :, 1], frames[core]),
                accepted_identity_corrections=0,
            )
            report["cases"].append(summary)
            out = args.output / name
            out.mkdir()
            np.savez_compressed(
                out / "evaluation.npz",
                frames=frames,
                core=core,
                tested=tested,
                baseline_pose=original_pose,
                candidate_pose=candidate_pose,
                original_uv=uv,
                candidate_uv=candidate_uv,
                original_source=original_source,
                candidate_source=candidate_source,
                baseline_reprojection_px=baseline["frozen_reprojection_px"],
                candidate_reprojection_px=candidate["frozen_reprojection_px"],
                baseline_loo_px=baseline["leave_one_out_px"],
                candidate_loo_px=candidate["leave_one_out_px"],
                baseline_ray_mm=ray_before,
                candidate_ray_mm=ray_after,
                baseline_lengths=np.stack([baseline["mp_mm"], baseline["pt_mm"]], axis=-1),
                candidate_lengths=np.stack([candidate["mp_mm"], candidate["pt_mm"]], axis=-1),
                baseline_bend_deg=baseline["bend_deg"],
                candidate_bend_deg=candidate["bend_deg"],
                score_original=score_original,
                score_candidate=score_candidate,
                changed=changed,
                supported=supported,
                body_reference_median=prior["median"],
                body_reference_scale=prior["scale"],
            )
            write_json(
                out / "evaluation.json",
                dict(
                    summary=summary,
                    body_reference=prior,
                    windows=window_details,
                    frames=frame_details,
                    parent_episodes=diagnosis["episodes"],
                ),
            )
            for filename in ("evaluation.npz", "evaluation.json"):
                path = out / filename
                record.output(f"{name}/{filename}", path, sha256=sha256(path))
            print(
                f"  tested={summary['tested_frames']}, changed={summary['changed_frames']}, "
                f"review-worthy={summary['review_worthy_frames']}",
                flush=True,
            )
        write_json(args.output / "summary.json", report)
        for filename in ("summary.json", "protocol.json"):
            path = args.output / filename
            record.output(filename, path, sha256=sha256(path))


if __name__ == "__main__":
    main()
