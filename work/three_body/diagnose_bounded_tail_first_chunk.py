"""Sealed same-68060 history chronology and old-tail-row/new-pose diagnostics.

No live progress, future candidate, raw decode, fit, GPU or ground truth is read.
All-six rows are compared by saved beam rank, not by hypothesized identity.
Saved source omission cannot identify the unavailable pre-fit assignment cause.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from collections import Counter
from pathlib import Path

import audit_bounded_first_chunk as interim
import long_baseline_support as baseline
import numpy as np
from continuous_tracker import RECORD_FIELDS

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

comparison, checks, REPO = interim.comparison, interim.checks, interim.REPO
START, STOP = interim.START, interim.STOP
AUDIT = REPO / "work/three_body/bounded_mouth_first_chunk_interim_audit_v1"
AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
AUDITOR_SHA = "7be951a77f39e8f2dea5e6cf2aeae1a10663036ede0e131a1462beb67c45759e"
BASELINE_SHA = "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4"
MANIFEST_SHA = "603490402bbaa6ea99a1694ae3e81ade77ba7b4dfa2f1ff7dec5b16605ba9048"
BRANCH = REPO / "work/three_body/bounded_mouth_one_minute_66400-76360_v1"
STAGE = "bounded_mouth_first_chunk_tail_diagnosis_v1"
DETAIL = (66620, 66920)


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def first_native(mask, frames):
    indices = np.flatnonzero(mask)
    return None if not len(indices) else int(frames[indices[0]])


def row_differences(a, b):
    """First dimension is row; Unicode storage width is reported, not a value change."""
    checks.require(a.shape == b.shape and a.ndim >= 1, "row shape differs")
    if a.dtype.kind == b.dtype.kind == "U":
        return (a != b).reshape(len(a), -1).any(1)
    checks.require(a.dtype == b.dtype, "numeric/discrete dtype differs")
    return (
        np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
        != np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)
    ).any(1)


def history_analysis(old, new, *, start=START, stop=STOP, first=interim.FIRST):
    frames = np.arange(start, stop, dtype=np.int64)
    begin, end = start - first, stop - first
    offsets = [a["history_offsets"] for a in (old, new)]
    for a, off in zip((old, new), offsets, strict=True):
        checks.require(
            off.shape == (stop - first + 1,)
            and off[0] == 0
            and np.all(np.diff(off)[begin:end] == 6),
            "six rows/native interval required",
        )
        native = np.repeat(np.arange(first, stop), np.diff(off))
        checks.require(np.array_equal(a["history_frame_idx"], native), "history native differs")
    checks.require(
        np.array_equal(offsets[0][: begin + 1], offsets[1][: begin + 1]),
        "pre-interval offsets differ",
    )
    changed = np.zeros((len(frames), 6, len(RECORD_FIELDS)), bool)
    numeric = np.zeros_like(changed)
    fields, pools = {}, [{}, {}]
    for fi, key in enumerate(RECORD_FIELDS):
        a, b = old["history_" + key], new["history_" + key]
        prefix = int(offsets[0][begin])
        if prefix:
            checks.require(
                not row_differences(a[:prefix], b[:prefix]).any(),
                "history differs before common parent: " + key,
            )
        slices = [
            x[int(off[begin]) : int(off[end])] for x, off in zip((a, b), offsets, strict=True)
        ]
        changed[:, :, fi] = row_differences(*slices).reshape(len(frames), 6)
        for pool, x in zip(pools, slices, strict=True):
            pool[key] = x.reshape(len(frames), 6, *x.shape[1:]).copy()
        if a.dtype.kind == "f":
            x, y = slices
            different = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            numeric[:, :, fi] = different.reshape(len(frames), 6, -1).any(2)
        else:
            numeric[:, :, fi] = changed[:, :, fi]
        first_diff = first_native(changed[:, :, fi].any(1), frames)
        first_t = None if first_diff is None else first_diff - start
        fields[key] = dict(
            category="float" if a.dtype.kind == "f" else "discrete",
            old_dtype=str(a.dtype),
            new_dtype=str(b.dtype),
            unicode_storage_width_only_ignored=a.dtype.kind == "U",
            first_byte_or_discrete_value_difference=first_diff,
            first_numerical_difference=first_native(numeric[:, :, fi].any(1), frames),
            changed_rows=int(changed[:, :, fi].sum()),
            changed_native_frames=int(changed[:, :, fi].any(1).sum()),
            first_different_beam_ranks=(
                [] if first_t is None else np.flatnonzero(changed[first_t, :, fi]).tolist()
            ),
            old_first_values=None if first_t is None else pools[0][key][first_t],
            new_first_values=None if first_t is None else pools[1][key][first_t],
        )
    float_idx = [i for i, k in enumerate(RECORD_FIELDS) if old["history_" + k].dtype.kind == "f"]
    discrete_idx = [i for i in range(len(RECORD_FIELDS)) if i not in float_idx]
    bounded = pools[1]["proposal_kind"] == "bounded_mouth_recovery"
    summary = dict(
        comparison="same native and retained beam rank; rank is not hypothesis identity",
        pre_interval_history_value_exact=True,
        first_any_difference=first_native(changed.any((1, 2)), frames),
        first_float_byte_difference=first_native(changed[:, :, float_idx].any((1, 2)), frames),
        first_float_numerical_difference=first_native(numeric[:, :, float_idx].any((1, 2)), frames),
        first_discrete_value_difference=first_native(
            changed[:, :, discrete_idx].any((1, 2)), frames
        ),
        first_bounded_retained=first_native(bounded.any(1), frames),
        first_hypothesis_count_difference=fields["hypotheses"]["first_numerical_difference"],
        bounded_retained_rows=int(bounded.sum()),
        bounded_retained_frames=int(bounded.any(1).sum()),
        fields=fields,
    )
    arrays = dict(
        frame_idx=frames,
        record_fields=np.asarray(RECORD_FIELDS),
        changed=changed,
        numerically_changed=numeric,
    )
    for label, pool in zip(("directional", "bounded"), pools, strict=True):
        arrays.update({label + "_" + key: value for key, value in pool.items()})
    return finite_json(summary), arrays


def classify_tail_rows(old, new, cross):
    """Classify old tail rows without mutating either saved strict assignment."""
    rows, current = old["sources"][..., 2], new["sources"][..., 2]
    valid = cross["valid_source"][..., 2]
    occupied = np.zeros((*rows.shape, 3), bool)
    for slot in range(3):
        for other in range(3):
            if slot != other:
                occupied[:, slot, :, other] = (rows[:, slot] >= 0) & (
                    current[:, other] == rows[:, slot]
                )
    collision = occupied.any(-1)
    same = (rows >= 0) & (rows == current)
    unassigned = current < 0
    free = valid & ~collision
    augmented = (current >= 0) | free
    ignoring = (current >= 0) | valid
    lost = old["observed"][..., 2] & ~new["observed"][..., 2]
    gained = ~old["observed"][..., 2] & new["observed"][..., 2]
    source_rows, cases = [], []
    for t, slot in np.argwhere(lost):
        category = (
            "geometrically_valid_available_rows_omitted_from_saved_sources"
            if augmented[t, slot].sum() >= 2
            else "support_possible_only_ignoring_other_slot_occupancy"
            if ignoring[t, slot].sum() >= 2
            else "old_and_current_rows_geometrically_insufficient_at_new_pose"
        )
        cases.append(
            dict(
                frame_idx=int(old["frame_idx"][t]),
                slot=int(slot),
                category=category,
                old_sources=rows[t, slot],
                new_sources=current[t, slot],
                old_rows_valid_views=int(valid[t, slot].sum()),
                augmented_views=int(augmented[t, slot].sum()),
                augmented_views_ignoring_occupancy=int(ignoring[t, slot].sum()),
                new_unassigned_valid_unoccupied_views=int((unassigned & free)[t, slot].sum()),
                identity_review=bool(new["identity_review"][t, slot]),
                old_tail_xyz=old["pose_xyz"][t, slot, 2],
                new_tail_xyz=new["pose_xyz"][t, slot, 2],
                tail_displacement_mm=float(
                    np.linalg.norm(old["pose_xyz"][t, slot, 2] - new["pose_xyz"][t, slot, 2])
                ),
            )
        )
        for vi, view in enumerate(VIEWS):
            present = rows[t, slot, vi] >= 0
            row_class = (
                "no_old_source"
                if not present
                else "retained_same_slot"
                if same[t, slot, vi]
                else "geometrically_incompatible"
                if not valid[t, slot, vi]
                else "occupied_by_other_slot"
                if collision[t, slot, vi]
                else "valid_unoccupied_but_absent_from_saved_sources"
            )
            source_rows.append(
                dict(
                    frame_idx=int(old["frame_idx"][t]),
                    slot=int(slot),
                    view=view,
                    old_source=int(rows[t, slot, vi]),
                    new_source=int(current[t, slot, vi]),
                    category=row_class,
                    geometry_pass=bool(valid[t, slot, vi]),
                    other_slot_occupants=np.flatnonzero(occupied[t, slot, vi]),
                    new_slot_view_unassigned=bool(unassigned[t, slot, vi]),
                    ray_distance_mm=cross["ray_distance_mm"][t, slot, vi, 2],
                    native_pixel_error=cross["native_pixel_error"][t, slot, vi, 2],
                )
            )
    flags = dict(
        frame_idx=old["frame_idx"].copy(),
        lost=lost,
        gained=gained,
        old_rows=rows.copy(),
        new_rows=current.copy(),
        old_row_geometry_valid=valid.copy(),
        other_slot_occupancy=occupied,
        old_row_same_slot_retained=same,
        new_view_unassigned=unassigned,
        augmented_supported_views=augmented,
        augmented_supported_views_ignoring_occupancy=ignoring,
    )
    summary = dict(
        lost_by_slot=lost.sum(0).tolist(),
        gained_by_slot=gained.sum(0).tolist(),
        lost_fish_frames=int(lost.sum()),
        gained_fish_frames=int(gained.sum()),
        lost_case_categories=dict(Counter(r["category"] for r in cases)),
        old_source_categories=dict(Counter(r["category"] for r in source_rows)),
        lost_with_two_valid_old_rows=int((lost & (valid.sum(2) >= 2)).sum()),
        lost_with_two_new_unassigned_valid_unoccupied_rows=int(
            (lost & ((unassigned & free).sum(2) >= 2)).sum()
        ),
        lost_support_recoverable_by_existing_plus_unassigned_old_rows=int(
            (lost & (((current >= 0) | (unassigned & free)).sum(2) >= 2)).sum()
        ),
        occupied_old_source_rows_in_lost=int((collision & lost[:, :, None]).sum()),
        occupied_and_geometrically_valid_old_rows_in_lost=int(
            (collision & valid & lost[:, :, None]).sum()
        ),
    )
    return summary, finite_json(cases), finite_json(source_rows), flags


def load_inputs(hashes):
    interim.bind(AUDIT / "audit.provenance.json", AUDIT_SHA, hashes)
    audit = checks.verify(AUDIT, "audit.provenance.json", interim.STAGE, hashes)
    for source, digest in list(hashes.items()):
        if source in audit["inputs"]:
            checks.require(digest == audit["inputs"][source]["sha256"], "audited code changed")
    interim.bind(interim.__file__, AUDITOR_SHA, hashes)
    interim.bind(baseline.__file__, BASELINE_SHA, hashes)
    summary = checks.read(AUDIT / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["complete_minute"] is False
        and summary["native_interval"] == [START, STOP]
        and summary["all39_arrays_semantically_decoded"] is True
        and summary["all21_arrays_backtraced"] is True
        and summary["expected_manifest_sha256"] == MANIFEST_SHA,
        "interim scope",
    )
    pointer = checks.read(AUDIT / "committed_checkpoint_record.json")
    checks.require(
        Path(pointer["path"]).resolve() == BRANCH / "checkpoints/through_068060"
        and pointer["manifest_sha256"] == MANIFEST_SHA,
        "committed checkpoint differs",
    )
    states, traces, contexts = {}, {}, []
    for name, root, digest in (
        ("directional", interim.OLD / "checkpoints/through_068060", interim.OLD_MANIFEST_SHA),
        ("bounded", BRANCH / "checkpoints/through_068060", MANIFEST_SHA),
    ):
        for filename in ("manifest.json", "metadata.json", "arrays.npz"):
            interim.input_member(audit, root / filename, hashes)
        checks.require(hashes[str(root / "manifest.json")] == digest, "checkpoint SHA differs")
        binding = checks.read(root / "manifest.json")["bindings"]
        checks.require(
            binding["original_binding"] == summary["shared_input_binding"],
            "original input binding differs",
        )
        state, metadata, _ = comparison.load_bound_checkpoint(root, binding, hashes)
        comparison.validate_lifecycle(metadata, STOP, 4)
        states[name] = state
        trace = comparison.short.backtrace(state)
        traces[name] = {k: v[START - interim.FIRST :].copy() for k, v in trace.items()}
        with np.load(AUDIT / f"{name}_same68060_candidate.npz", allow_pickle=False) as z:
            checks.require(
                set(z.files) == set(traces[name]) and len(z.files) == 21,
                "all21 audited trace fields required",
            )
            for key in z.files:
                checks.require(
                    checks.equal_bytes(z[key], traces[name][key]),
                    "backtrace differs from interim: " + key,
                )
        checks.require(
            np.array_equal(traces[name]["frame_idx"], np.arange(START, STOP)),
            "backtrace native interval differs",
        )
        contexts.append(metadata["context"])
    shared = summary["shared_input_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        interim.input_member(audit, path, hashes)
        checks.require(hashes[str(Path(path).resolve())] == digest, "shared SHA differs")
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == baseline.EXPERIMENT + "__" + v + ".npz")
        for v in VIEWS
    }
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig = load_rig(calibration)
    normalized = comparison.support_audit.COMMON.normalize(
        [dataclasses.asdict(c) for c in rig.cameras]
    )
    for context in contexts:
        checks.require(
            normalized == context["cameras"]
            and context["views"] == list(rig.views)
            and context["world_units"] == rig.world_units == "mm",
            "camera context differs",
        )
    return states, traces, baseline.RegisteredNodes(paths, np.arange(START, STOP)), rig, summary


def write_csv(path, rows):
    checks.require(bool(rows), "nonempty table required")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
            for r in rows
        )


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new output required"
    )
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="diagnosis") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(
                    json.dumps(finite_json(value), indent=2, allow_nan=False), encoding="utf-8"
                )
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for source, digest in code.items():
            destination = output / "code_snapshot" / Path(source).relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(source).read_bytes())
            checks.require(checks.sha(destination) == digest, "code snapshot differs")
            emit(str(destination.relative_to(output)))
        emit(
            "plan.json",
            dict(
                native_interval=[START, STOP],
                horizon_stop=STOP,
                detail_interval=list(DETAIL),
                all19_fields=list(RECORD_FIELDS),
                all_lost_tail_cases=True,
                audit_sha256=AUDIT_SHA,
                ray_mm=2.0,
                native_pixel_gate=12.0,
                raw_decode=False,
                fit=False,
                GPU=False,
            ),
        )
        states, traces, nodes, rig, prior = load_inputs(hashes)
        history, arrays = history_analysis(states["directional"], states["bounded"])
        np.savez_compressed(output / "all6_history_comparison.npz", **arrays)
        emit("all6_history_comparison.npz")
        emit("history_first_differences.json", history)
        for name, candidate in traces.items():
            actual = baseline.classify_strict_support(candidate, nodes, rig)
            for key in ("sources", "observed"):
                checks.require(
                    checks.equal_bytes(actual[key], candidate[key]),
                    "saved support differs: " + name + "/" + key,
                )
            np.savez_compressed(output / f"{name}_strict_support.npz", **actual)
            emit(f"{name}_strict_support.npz")
        cross_input = {k: traces["bounded"][k].copy() for k in ("frame_idx", "pose_xyz", "sources")}
        cross_input["sources"][..., 2] = traces["directional"]["sources"][..., 2]
        cross = baseline.classify_strict_support(cross_input, nodes, rig)
        np.savez_compressed(output / "old_tail_rows_at_new_pose.npz", **cross)
        emit("old_tail_rows_at_new_pose.npz")
        counts, cases, source_rows, flags = classify_tail_rows(
            traces["directional"], traces["bounded"], cross
        )
        checks.require(
            counts["lost_fish_frames"] == prior["support_delta"]["lost_by_node"][2]
            and counts["gained_fish_frames"] == prior["support_delta"]["gained_by_node"][2],
            "tail gain/loss differs from interim",
        )
        np.savez_compressed(output / "tail_classification.npz", **flags)
        emit("tail_classification.npz")
        emit("lost_tail_cases.json", cases)
        emit("lost_tail_source_rows.json", source_rows)
        write_csv(output / "lost_tail_cases.csv", cases)
        write_csv(output / "lost_tail_source_rows.csv", source_rows)
        emit("lost_tail_cases.csv")
        emit("lost_tail_source_rows.csv")
        frames = arrays["frame_idx"]
        selected = traces["bounded"]["proposal_kind"] == "bounded_mouth_recovery"
        chronology = {k: v for k, v in history.items() if k.startswith("first_")}
        chronology.update(
            first_bounded_selected_same68060=first_native(selected, frames),
            selected_bounded_frames=frames[selected].tolist(),
            first_tail_loss_any_slot=first_native(flags["lost"].any(1), frames),
            first_tail_loss_S0=first_native(flags["lost"][:, 0], frames),
        )
        details = []
        for t in np.flatnonzero((frames >= DETAIL[0]) & (frames < DETAIL[1])):
            details.append(
                dict(
                    frame_idx=int(frames[t]),
                    old_hypotheses=arrays["directional_hypotheses"][t],
                    new_hypotheses=arrays["bounded_hypotheses"][t],
                    old_retained_kind=arrays["directional_proposal_kind"][t],
                    new_retained_kind=arrays["bounded_proposal_kind"][t],
                    old_retained_parent=arrays["directional_parent"][t],
                    new_retained_parent=arrays["bounded_parent"][t],
                    old_selected_kind=traces["directional"]["proposal_kind"][t],
                    new_selected_kind=traces["bounded"]["proposal_kind"][t],
                    old_observed=traces["directional"]["observed"][t],
                    new_observed=traces["bounded"]["observed"][t],
                    old_pose=traces["directional"]["pose_xyz"][t],
                    new_pose=traces["bounded"]["pose_xyz"][t],
                    old_sources=traces["directional"]["sources"][t],
                    new_sources=traces["bounded"]["sources"][t],
                    old_tail_rows_at_new_pose_ray_mm=cross["ray_distance_mm"][t, :, :, 2],
                    old_tail_rows_at_new_pose_pixel_error=cross["native_pixel_error"][t, :, :, 2],
                    other_slot_tail_occupancy=flags["other_slot_occupancy"][t],
                )
            )
        emit("fixed_context_66620-66920.json", details)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            horizon_stop=STOP,
            chronology=chronology,
            tail=counts,
            all39_payloads_codec_verified=True,
            all21_backtraces_match_interim=True,
            all_saved_support_recomputed=True,
            ray_mm=2.0,
            native_pixel_gate=12.0,
            input_audit_sha256=AUDIT_SHA,
            same_SL3_calibration4=True,
            original_inputs_not_redecoded=True,
            complete_minute=False,
            independent_ID_truth=False,
            fits=0,
            GPU=False,
            live_files_modified=False,
            limit="Post-fit saved sources only; no pre-fit plan or unretained pool evidence. "
            "Old rows and slot numbers are not independent truth. "
            "Same-68060 backtraces may change with the full-minute future.",
        )
        emit("summary.json", summary)
        report = (
            "# Bounded 最初10秒・尾支持の保存履歴診断\n\n"
            f"同68060終端の全1660時刻。尾支持喪失 {counts['lost_fish_frames']}、"
            f"獲得 {counts['gained_fish_frames']}。\n\n"
            f"最初の全6履歴差 {chronology['first_any_difference']}、浮動数値差 "
            f"{chronology['first_float_numerical_difference']}、離散差 "
            f"{chronology['first_discrete_value_difference']}。初bounded保持 "
            f"{chronology['first_bounded_retained']}、同終端逆追跡で初採用 "
            f"{chronology['first_bounded_selected_same68060']}。最初のS0尾喪失 "
            f"{chronology['first_tail_loss_S0']}。これは時間順序であり原因の確定ではない。\n\n"
            f"旧尾行を新poseで2mm/12px再検査した分類: {counts['lost_case_categories']}。"
            f"新sourceが未割当の2視点以上に有効かつ非競合の旧行がある喪失ケース "
            f"{counts['lost_with_two_new_unassigned_valid_unoccupied_rows']}。\n\n"
            "全19fieldsの最初の差と全6履歴を保存した。全6はbeam順位比較で仮説identityの対応ではない。"
            "同view/tailの他slot占有は幾何適合と独立に全行保存した。旧sourceもS0も独立正解とはしない。\n\n"
            "保存sourceはfit後の支持filter結果であり、欠けた行が生成・fit・選択のどこで落ちたかは"
            "未記録のprefit plan/全候補からは判定できない。追加fit/原画decode/GPU/GTは使用しない。"
            "これは最初10秒の同終端診断で、1分完走・生物ID正解を認定せず、将来backtraceは変わり得る。\n"
        )
        (output / "RESULT.md").write_text(report, encoding="utf-8")
        emit("RESULT.md")
        checks.require(code == comparison.source_registry.local_code(), "loaded code changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "diagnostic input changed")
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "diagnostic output changed")
        record.param("all_used_inputs_sources_snapshots_outputs_SHA_unchanged", True)
    print(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
