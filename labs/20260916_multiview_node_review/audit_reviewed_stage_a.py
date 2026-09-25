"""Audit original C27 labels at eight reviewed scenes; change no tracking data.

This measures where label selection and image coordinates change. It does not
infer biological identity from label equality or the size of a displacement.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
NODES = ("mouth", "pectoral", "tail")
STEM = "stage_a_review_audit"


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


def node_steps(before, after):
    return np.linalg.norm(after.astype(float) - before.astype(float), axis=-1)


def finite_max(values):
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def selected(cache, source, frames):
    labels = np.full(source.shape, -1, np.int64)
    uv = np.full((*source.shape, 3, 2), np.nan)
    for v, data in enumerate(cache):
        rows = source[:, :, v]
        valid = rows >= 0
        if np.any(rows[valid] >= np.broadcast_to(data["counts"][frames, None], rows.shape)[valid]):
            raise ValueError("Source refers outside its original detection frame")
        absolute = data["start"][frames, None] + np.maximum(rows, 0)
        labels[:, :, v][valid] = data["labels"][absolute[valid]]
        uv[:, :, v][valid] = data["points"][absolute[valid]]
    return labels, uv


def raw_same_label_edges(cache, frames):
    """All same-ID adjacent-frame motions, independent of the C34 selection."""
    records = []
    for v, data in enumerate(cache):
        for frame in frames[1:]:
            previous = np.arange(data["start"][frame - 1], data["stop"][frame - 1])
            current = np.arange(data["start"][frame], data["stop"][frame])
            before = {int(data["labels"][i]): int(i) for i in previous if data["labels"][i] >= 0}
            after = {int(data["labels"][i]): int(i) for i in current if data["labels"][i] >= 0}
            if len(before) != np.count_nonzero(data["labels"][previous] >= 0) or len(
                after
            ) != np.count_nonzero(data["labels"][current] >= 0):
                raise ValueError("A camera has duplicate live C27 labels in one frame")
            for label in sorted(before.keys() & after.keys()):
                i, j = before[label], after[label]
                steps = node_steps(data["points"][i], data["points"][j])
                records.append(
                    dict(
                        edge_to_frame=int(frame),
                        view=VIEWS[v],
                        c27_label=label,
                        detection_rows=[
                            int(i - data["start"][frame - 1]),
                            int(j - data["start"][frame]),
                        ],
                        node_step_px=steps,
                        max_mp_step_px=finite_max(steps[:2]),
                    )
                )
    return records


def summarize_selection(labels, uv, source, frames, xyz):
    steps = node_steps(uv[:-1], uv[1:])
    present = (labels[:-1] >= 0) & (labels[1:] >= 0)
    changed = present & (labels[:-1] != labels[1:])
    same = present & ~changed
    availability_changed = (labels[:-1] >= 0) != (labels[1:] >= 0)
    events, summary = [], []
    xyz_steps = node_steps(xyz[:-1], xyz[1:])
    for slot in range(3):
        for v, view in enumerate(VIEWS):
            gate = 60 if v == 0 else 50  # Existing C24b diagnostic; not tuned here.
            mp = steps[:, slot, v, :2]
            large = (np.isfinite(mp) & (mp > gate)).any(axis=-1)
            group = dict(
                slot=slot,
                view=view,
                existing_c24b_gate_px=gate,
                c27_label_selection_changes=int(changed[:, slot, v].sum()),
                same_label_edges=int(same[:, slot, v].sum()),
                large_same_label_edges=int((large & same[:, slot, v]).sum()),
                large_changed_label_edges=int((large & changed[:, slot, v]).sum()),
                missing_endpoint_edges=int((~present[:, slot, v]).sum()),
                availability_changed_edges=int(availability_changed[:, slot, v].sum()),
                max_mp_step_same_label_px=finite_max(mp[same[:, slot, v]]),
                max_mp_step_changed_label_px=finite_max(mp[changed[:, slot, v]]),
            )
            summary.append(group)
            # Keep every selection change, every legacy-gate exceedance, and the
            # three largest MP steps even when below that deliberately loose gate.
            ranked = np.max(np.where(np.isfinite(mp), mp, -np.inf), axis=-1)
            top = set(np.argsort(ranked)[-3:].tolist())
            for i in range(len(frames) - 1):
                if not (
                    changed[i, slot, v]
                    or availability_changed[i, slot, v]
                    or large[i]
                    or (i in top and np.isfinite(ranked[i]))
                ):
                    continue
                relation = (
                    "different_c27_labels"
                    if changed[i, slot, v]
                    else "same_c27_label"
                    if same[i, slot, v]
                    else "observed_to_missing"
                    if labels[i, slot, v] >= 0
                    else "missing_to_observed"
                    if labels[i + 1, slot, v] >= 0
                    else "both_missing"
                )
                events.append(
                    dict(
                        edge_to_frame=int(frames[i + 1]),
                        slot=slot,
                        view=view,
                        c27_labels=[int(labels[i, slot, v]), int(labels[i + 1, slot, v])],
                        source_rows=[int(source[i, slot, v]), int(source[i + 1, slot, v])],
                        label_relation=relation,
                        node_step_px=steps[i, slot, v],
                        selected_3d_step_mm=xyz_steps[i, slot],
                        exceeds_existing_c24b_gate=bool(large[i]),
                    )
                )
    return dict(
        summary_by_slot_view=summary,
        selected_edges=sorted(events, key=lambda x: (x["edge_to_frame"], x["slot"], x["view"])),
        selected_node_step_px=steps,
        selected_label_changed=changed,
        selected_availability_changed=availability_changed,
        selected_3d_step_mm=xyz_steps,
    )


def edge_snapshot(case, frame):
    """Explicitly tabulate one edge without treating a label as identity truth."""
    i = int(np.flatnonzero(case["frames"] == frame)[0])
    raw = [r for r in case["raw_c27_same_label_edges"] if r["edge_to_frame"] == frame]
    raw_max = {}
    for view in VIEWS:
        values = [r["max_mp_step_px"] for r in raw if r["view"] == view]
        raw_max[view] = max((x for x in values if x is not None), default=None)
    return dict(
        edge_from_frame=frame - 1,
        edge_to_frame=frame,
        raw_c27_same_label_max_mp_step_px_by_view=raw_max,
        stages={
            stage: dict(
                xyz_step_mm_by_slot=d["selected_3d_step_mm"][i - 1],
                c27_labels_before=d["selected_c27_labels"][i - 1],
                c27_labels_after=d["selected_c27_labels"][i],
                source_rows_before=d["source_rows"][i - 1],
                source_rows_after=d["source_rows"][i],
                node_step_px_by_slot_view_node=d["selected_node_step_px"][i - 1],
            )
            for stage, d in case["stages"].items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=Path("configs/paths.local.toml"))
    parser.add_argument("--shared-work", type=Path, default=None)
    parser.add_argument("--name", default=STEM)
    parser.add_argument("--inputs", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("docs/reviews/2026-09-16_multiview_identity_review.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/multiview_node_review_v1/human_review_followup")
    )
    parser.add_argument("--radius", type=int, default=40)
    args = parser.parse_args()
    if args.shared_work is None:
        with args.paths_config.open("rb") as handle:
            args.shared_work = Path(tomllib.load(handle)["paths"]["work_dir"])
    if Path(args.name).name != args.name:
        raise ValueError("Name must be a filename stem")
    destination = args.output / f"{args.name}.json"
    if destination.exists() or (args.output / f"{args.name}.provenance.json").exists():
        raise FileExistsError("Preserve existing named stage-A audit")
    if args.radius < 20:
        raise ValueError("Radius must include the documented 31497 exit")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    c34_path = Path(review["baseline"]["path"])
    c27_dir = (
        args.shared_work / "c27_uncapped_tracklets_any" / EXPERIMENT / "mouth-pectoral_oks_serial"
    )
    c28_path = (
        args.shared_work
        / "c28_lambda_grid_transfer"
        / EXPERIMENT
        / f"C28-gated__{EXPERIMENT}__lam0.767.npz"
    )
    with provenance("reviewed_original_stage_a_attribution", args.output, name=args.name) as record:
        record.params_from(args)
        for name, path in (
            ("code", Path(__file__)),
            ("human_review", args.review),
            ("C34", c34_path),
            ("C28", c28_path),
            ("c27_summary", c27_dir / "c27_summary.json"),
            ("c27_provenance", c27_dir / "c27.provenance.json"),
            ("review_clip_builder", Path(__file__).with_name("build_visuals.py")),
        ):
            digest = sha256(path)
            if name == "C34" and digest != review["baseline"]["sha256"]:
                raise ValueError("Reviewed C34 baseline changed")
            record.input(name, path, sha256=digest)
        c27_provenance = json.loads((c27_dir / "c27.provenance.json").read_text(encoding="utf-8"))
        with np.load(c34_path, allow_pickle=False) as data:
            c34_source, c34_xyz = data["source"], data["xyz"]
        with np.load(c28_path, allow_pickle=False) as data:
            c28_source, c28_xyz = data["source"], data["xyz"]
        if c34_source.shape != c28_source.shape or c34_source.shape[1:] != (3, 3):
            raise ValueError("Unexpected reconstruction shape")
        n_frames = c34_source.shape[0]
        cache, alignment = [], []
        for view in VIEWS:
            cache_path = (
                args.inputs / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
            )
            label_path = c27_dir / f"track2d__{view}.npy"
            for name, path in ((f"C01_{view}", cache_path), (f"C27_{view}", label_path)):
                record.input(name, path, sha256=sha256(path))
            shared_cache = args.shared_work / "c01_s16_predictions/bottomup" / cache_path.name
            shared_digest = sha256(shared_cache)
            record.input(f"current_shared_C01_{view}", shared_cache, sha256=shared_digest)
            if shared_digest != record.inputs[f"C01_{view}"]["sha256"]:
                raise ValueError("Local and current shared C01 caches differ")
            historical = [
                entry
                for entry in c27_provenance.get("inputs", {}).values()
                if Path(entry["path"]).name == cache_path.name
            ]
            historical_hashes = [entry["sha256"] for entry in historical if entry.get("sha256")]
            if any(digest != shared_digest for digest in historical_hashes):
                raise ValueError("C27 historical input hash differs from current C01")
            labels = np.load(label_path, allow_pickle=False)
            with np.load(cache_path, allow_pickle=False) as data:
                if (
                    str(data["experiment"]) != EXPERIMENT
                    or str(data["view"]) != view
                    or tuple(data["nodes"]) != NODES
                ):
                    raise ValueError("Detection cache metadata mismatch")
                f, points = data["frame_idx"], data["points"]
                if (
                    int(data["n_frames"]) != n_frames
                    or len(f) != len(labels)
                    or np.any(np.diff(f) < 0)
                ):
                    raise ValueError("Detection/label length or frame ordering mismatch")
            start = np.searchsorted(f, np.arange(n_frames), side="left")
            stop = np.searchsorted(f, np.arange(n_frames), side="right")
            cache.append(
                dict(points=points, labels=labels, start=start, stop=stop, counts=stop - start)
            )
            alignment.append(
                dict(
                    view=view,
                    detection_rows=len(f),
                    label_rows=len(labels),
                    unique_c27_labels=np.unique(labels),
                    unlabelled_rows=int((labels < 0).sum()),
                    local_matches_current_shared_C01_sha256=True,
                    historical_C27_input_hash_available=bool(historical_hashes),
                    historical_C27_input_hashes=historical_hashes,
                )
            )
        report = dict(
            schema_version=1,
            experiment=EXPERIMENT,
            views=VIEWS,
            nodes=NODES,
            native_fps=166,
            context_radius_frames=args.radius,
            evidence_scope=(
                "Stage attribution only; labels are not biological identity ground truth"
            ),
            interpretation_limits=[
                "Original C34 baseline only; the human-reviewed panel remains unspecified.",
                "A same-label jump measures coordinate discontinuity inside the original C27 ID; "
                "it does not prove an identity swap.",
                "A selected label change measures downstream source selection; "
                "it may be a correct reassignment.",
                "C28 and C34 slot numbers need not represent the same animal "
                "after downstream permutations.",
                "Local and current shared C01 hashes must match. Original C27 input hashes "
                "are checked if recorded; their absence limits historical row provenance.",
                "60/50 px gates are inherited C24b diagnostic thresholds; "
                "they are not new acceptance criteria.",
                "No detections, tracks, human annotations, "
                "or candidate acceptance decisions were changed.",
                "Review anchors are not exact onset/offset annotations. "
                "The f76930 and f76988 videos overlap and cannot count as independent swaps.",
            ],
            alignment=alignment,
            c27_summary=json.loads((c27_dir / "c27_summary.json").read_text(encoding="utf-8")),
            original_c27_provenance_inputs=c27_provenance.get("inputs", {}),
            review_clip_overlap_context=dict(
                source="build_visuals.py: native slow clips use [center-83, center+83)",
                anchors=[76930, 76988],
                clip_ranges_half_open=[[76930 - 83, 76930 + 83], [76988 - 83, 76988 + 83]],
                overlap_range_half_open=[76988 - 83, 76930 + 83],
                overlap_native_frames=(76930 + 83) - (76988 - 83),
                independent_swap_count="not_established",
            ),
            key_edges=[],
            cases=[],
        )
        for event in review["events"]:
            center = event["frame"]
            frames = np.arange(
                max(0, center - args.radius), min(n_frames, center + args.radius + 1)
            )
            case = dict(review_anchor_frame=center, human_review=event, frames=frames, stages={})
            raw_edges = raw_same_label_edges(cache, frames)
            case["raw_c27_same_label_edges"] = raw_edges
            case["raw_c27_largest_mp_edges_by_view"] = {
                view: sorted(
                    (r for r in raw_edges if r["view"] == view and r["max_mp_step_px"] is not None),
                    key=lambda r: r["max_mp_step_px"],
                    reverse=True,
                )[:6]
                for view in VIEWS
            }
            for stage, all_source, all_xyz in (
                ("C34", c34_source, c34_xyz),
                ("C28", c28_source, c28_xyz),
            ):
                source, xyz = all_source[frames], all_xyz[frames]
                labels, uv = selected(cache, source, frames)
                data = dict(
                    source_rows=source, selected_c27_labels=labels, selected_uv_px=uv, xyz_mm=xyz
                )
                data.update(summarize_selection(labels, uv, source, frames, xyz))
                case["stages"][stage] = data
            report["cases"].append(case)
            report["key_edges"].append(edge_snapshot(case, center))
            if center == 31479:
                # Include C28's later change and C34's documented return boundary.
                report["key_edges"].extend(edge_snapshot(case, f) for f in (31481, 31497))
            print(
                f"Audited f{center}: {len(frames)} frames, {len(raw_edges)} raw same-label edges",
                flush=True,
            )
        destination.write_text(
            json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        record.output("stage_a_review_audit", destination, sha256=sha256(destination))
        print(destination, flush=True)


if __name__ == "__main__":
    main()
