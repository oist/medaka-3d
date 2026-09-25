"""Fixed, completed 68060-horizon mouth-support diagnosis; no fit or ID inference.

Read the sealed interim audit and its committed checkpoint, never the running
minute candidate/progress. Counterfactual old source rows are scored at the new
saved pose with the unchanged float32-ray 2 mm / 12 pixel gates. Saved sources
are post-filter support, not the unavailable pre-fit assignment plan.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import audit_directional_first_chunk as interim
import long_baseline_support as baseline
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

comparison, checks = interim.comparison, interim.checks
REPO, START, STOP = interim.REPO, interim.START, interim.STOP
AUDIT = REPO / "work/three_body/directional_first_chunk_interim_audit_v1"
AUDIT_SHA = "9ad75c5e7613b10032bae127645984ae056e89254f564ebfba52c945350ea152"
BASELINE_SHA = "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4"
BRANCH = REPO / "work/three_body/directional_one_minute_66400-76360_v1"
ORIGINAL = REPO / "work/three_body/long_remaining_supervisor_v1/phases/f066400-076360/tracking"
STAGE = "directional_mouth_first_chunk_diagnosis_v1"


def bind(path, digest, hashes):
    path = Path(path).resolve(strict=True)
    checks.require(checks.sha(path) == digest, "bound input changed: " + str(path))
    hashes[str(path)] = digest
    return path


def audit_input(audit, path, hashes):
    path = Path(path).resolve(strict=True)
    item = audit["inputs"].get(str(path))
    checks.require(item is not None and Path(item["path"]).resolve() == path, "unbound audit input")
    return bind(path, item["sha256"], hashes)


def load_inputs(hashes):
    bind(AUDIT / "audit.provenance.json", AUDIT_SHA, hashes)
    audit = checks.verify(
        AUDIT, "audit.provenance.json", "directional_first_chunk_interim_audit", hashes
    )
    comparison.require_bound(
        AUDIT,
        ("summary.json", "committed_checkpoint_record.json", "short_history_comparison.json"),
        hashes,
    )
    summary = checks.read(AUDIT / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["complete_minute"] is False
        and summary["completed_checkpoint_interval"] == [START, STOP]
        and summary["short_retained_history_all_arrays_exact"] is True,
        "wrong completed interim audit",
    )
    for source, digest in list(hashes.items()):
        if source in audit["inputs"]:
            checks.require(
                digest == audit["inputs"][source]["sha256"], "audited helper source changed"
            )
    bind(baseline.__file__, BASELINE_SHA, hashes)
    checkpoint = BRANCH / "checkpoints/through_068060"
    pointer = checks.read(AUDIT / "committed_checkpoint_record.json")
    checks.require(Path(pointer["path"]).resolve() == checkpoint, "different committed checkpoint")
    for root in (checkpoint, ORIGINAL / "checkpoint"):
        for name in ("manifest.json", "metadata.json", "arrays.npz"):
            audit_input(audit, root / name, hashes)
    checks.require(
        pointer["manifest_sha256"] == hashes[str(checkpoint / "manifest.json")],
        "commit SHA differs",
    )
    old_seal_path = audit_input(audit, ORIGINAL / "continuation.provenance.json", hashes)
    old_seal = checks.read(old_seal_path)
    checks.require(
        old_seal["status"] == "ok"
        and old_seal["stage"] == "three_body_long_continuation"
        and old_seal["params"]["through_frame"] == comparison.STOP,
        "original successful 76360 checkpoint required",
    )
    binding = old_seal["params"]["bindings"]
    new_binding = checks.read(checkpoint / "manifest.json")["bindings"]
    checks.require(new_binding["original_binding"] == binding, "different original context")
    values = {}
    contexts = []
    for method, root, bound, stop in (
        ("original", ORIGINAL / "checkpoint", binding, comparison.STOP),
        ("directional", checkpoint, new_binding, STOP),
    ):
        state, metadata, _ = comparison.load_bound_checkpoint(root, bound, hashes)
        comparison.validate_lifecycle(metadata, stop, 4)
        traced = interim.prefix_trace(state, STOP)
        checks.require(
            np.array_equal(traced["frame_idx"], np.arange(interim.FIRST, STOP)),
            "native prefix differs",
        )
        values[method] = {
            key: value[START - interim.FIRST :].copy() for key, value in traced.items()
        }
        checks.require(
            comparison.short.stats(traced, START, STOP)
            == summary["methods"][method]["ten_seconds"],
            "same-horizon trace differs from sealed interim metrics",
        )
        contexts.append(metadata["context"])
        del state, traced
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "three SL and four calibration inputs required")
    for path, digest in shared.items():
        bind(path, digest, hashes)
    paths = {
        v: next(Path(p) for p in shared if Path(p).name == f"{baseline.EXPERIMENT}__{v}.npz")
        for v in VIEWS
    }
    manifests = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    checks.require(len(manifests) == 1, "one calibration manifest required")
    rig = load_rig(manifests[0].parent)
    normalized = comparison.support_audit.COMMON.normalize(
        [dataclasses.asdict(c) for c in rig.cameras]
    )
    for context in contexts:
        checks.require(
            normalized == context["cameras"]
            and list(rig.views) == context["views"] == list(VIEWS)
            and rig.world_units == context["world_units"] == "mm",
            "actual camera context differs",
        )
    nodes = baseline.RegisteredNodes(paths, np.arange(START, STOP, dtype=np.int64))
    return values, nodes, rig, summary


def runs(mask):
    """All maximal native runs per existing logical slot, with no identity matching."""
    result = []
    for slot in range(3):
        transitions = np.diff(np.r_[False, mask[:, slot], False].astype(np.int8))
        for first, stop in zip(
            np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True
        ):
            result.append(
                dict(
                    slot=slot,
                    start=START + int(first),
                    stop_exclusive=START + int(stop),
                    frames=int(stop - first),
                )
            )
    return sorted(result, key=lambda row: (-row["frames"], row["start"], row["slot"]))


def available_old_sources(old, new, cross):
    """An old row may be geometrically valid yet occupied by another current slot."""
    rows = old["sources"][..., 0]
    current = new["sources"][..., 0]
    collision = np.zeros(rows.shape, bool)
    for slot in range(3):
        others = [i for i in range(3) if i != slot]
        collision[:, slot] = (rows[:, slot] >= 0) & (current[:, others] == rows[:, slot, None]).any(
            axis=1
        )
    valid = cross["valid_source"][..., 0]
    free = valid & ~collision
    augment = (current >= 0) | free
    possible_ignoring_collision = (current >= 0) | valid
    return dict(
        old_source_taken_by_other_new_slot=collision,
        old_source_valid_at_new_pose=valid,
        old_source_valid_and_unoccupied=free,
        augmented_supported_views=augment,
        augmented_supported_views_ignoring_collision=possible_ignoring_collision,
    )


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def case_rows(values, checked, cross, availability):
    old, new = values["original"], values["directional"]
    lost = old["observed"][..., 0] & ~new["observed"][..., 0]
    gained = ~old["observed"][..., 0] & new["observed"][..., 0]
    result = []
    for t, slot in np.argwhere(lost | gained):
        available = int(availability["augmented_supported_views"][t, slot].sum())
        any_rows = int(availability["augmented_supported_views_ignoring_collision"][t, slot].sum())
        category = (
            (
                "old_rows_can_augment_saved_sources_without_collision"
                if available >= 2
                else "old_rows_can_augment_only_ignoring_current_collision"
                if any_rows >= 2
                else "old_and_current_rows_geometrically_insufficient_at_new_pose"
            )
            if lost[t, slot]
            else "gained_strict_support"
        )
        row = dict(
            frame_idx=START + int(t),
            slot=int(slot),
            change="lost" if lost[t, slot] else "gained",
            category=category,
            old_rows_supported_views_at_new_pose=int(cross["valid_source"][t, slot, :, 0].sum()),
            augmented_supported_views=available,
            old_row_collision=availability["old_source_taken_by_other_new_slot"][t, slot],
            old_rows_at_new_pose_ray_mm=cross["ray_distance_mm"][t, slot, :, 0],
            old_rows_at_new_pose_pixel_error=cross["native_pixel_error"][t, slot, :, 0],
            old_rows_at_new_pose_gate_pass=cross["valid_source"][t, slot, :, 0],
            mouth_displacement_mm=float(
                np.linalg.norm(new["pose_xyz"][t, slot, 0] - old["pose_xyz"][t, slot, 0])
            ),
            biological_identity_confirmed=False,
        )
        for method, candidate in values.items():
            row[method] = dict(
                mouth_xyz=candidate["pose_xyz"][t, slot, 0],
                mouth_sources=candidate["sources"][t, slot, :, 0],
                mouth_ray_mm=checked[method]["ray_distance_mm"][t, slot, :, 0],
                mouth_pixel_error=checked[method]["native_pixel_error"][t, slot, :, 0],
                observed_nodes=candidate["observed"][t, slot],
                body_supported=bool(candidate["body_supported"][t, slot]),
                identity_review=bool(candidate["identity_review"][t, slot]),
                node_missing=candidate["node_missing"][t, slot],
                body_missing=int(candidate["missing"][t, slot]),
                proposal_kind=str(candidate["proposal_kind"][t]),
                hypotheses=candidate["hypotheses"][t],
                local_cost=float(candidate["costs"][t]),
                loss_parts=candidate["loss_parts"][t],
            )
        result.append(finite_json(row))
    return result, lost, gained


def report(summary):
    lines = [
        "# 最初1660フレームの口支持変化",
        "",
        "同じ終了時点68060の軌跡を用いた [66400,68060) のCPU診断。"
        f"口支持は {summary['mouth_counts']['original']} → "
        f"{summary['mouth_counts']['directional']} / 4980。",
        f"失った口支持は {summary['lost_fish_frames']}、"
        f"得た口支持は {summary['gained_fish_frames']} "
        f"(差 {summary['gained_fish_frames'] - summary['lost_fish_frames']:+d})。"
        "全変化行は cases.jsonl、全連続区間は runs.json に保存した。",
        "",
        "新しい保存姿勢に元の口sourceを当てた、lost行の分類:",
        "",
    ]
    labels = {
        "old_rows_can_augment_saved_sources_without_collision": (
            "現在の別slotとの排他を守って2視点へ補える"
        ),
        "old_rows_can_augment_only_ignoring_current_collision": (
            "2視点へ補うには現在の別slotとのsource衝突を解消する必要がある"
        ),
        "old_and_current_rows_geometrically_insufficient_at_new_pose": (
            "元と現在のsourceだけでは新poseを2視点で支持できない"
        ),
    }
    lines += [f"- {labels[k]}: {v}" for k, v in summary["lost_categories"].items()]
    lines += ["", "|変化|slot|開始|終了(排他)|フレーム数|", "|---|---:|---:|---:|---:|"]
    for change in ("lost", "gained"):
        for item in summary["runs"][change][:8]:
            lines.append(
                f"|{change}|{item['slot']}|{item['start']}|{item['stop_exclusive']}|{item['frames']}|"
            )
    lines += [
        "",
        "全1660フレーム・全3node・全保存sourceについて、"
        "両方式とも元のfloat32 ray手順と2mm/12pxを再計算し、"
        "保存sources/observedとの完全一致と同view/node排他を確認した。"
        "原画の再decode、マスク検証、fit、GPU、GTは使用していない。",
        "",
        "sourcesはfit後の厳密支持配列であり、fit前planではない。補える行でも、fit前に割り当てられなかったのか、その後の選択や幾何で除外されたのかはこの保存データだけでは特定できない。旧sourceを新poseへ当てる算術は対応候補の検査であり、永久IDや生物学的正解の移植ではない。",
        "",
        "slotは同じ初期状態から継承された配列番号として列挙した。全3魚のID真値はなく、口支持の増減・胸部再取得・小さい画像誤差はID正解の証明にならない。今回の終了68060でのbacktraceは、後続フレームを含む最終1分のbacktraceとは変わり得る。親1分の完走を主張しない。",
        "",
    ]
    return "\n".join(lines)


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new output required"
    )
    codes = comparison.source_registry.local_code()
    hashes = dict(codes)
    started = time.perf_counter()
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="diagnosis") as record:
        try:
            for source, digest in codes.items():
                source = Path(source)
                dest = output / "code_snapshot" / source.relative_to(REPO)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
                checks.require(checks.sha(dest) == digest, "source snapshot changed")
                record.output(str(dest.relative_to(output)), dest, sha256=digest)
            values, nodes, rig, prior = load_inputs(hashes)
            checked, support_summary = {}, {}
            for method, candidate in values.items():
                actual = baseline.classify_strict_support(candidate, nodes, rig)
                for key in ("sources", "observed"):
                    checks.require(
                        np.array_equal(actual[key], candidate[key]),
                        "saved strict support differs: " + method + "/" + key,
                    )
                held = actual["valid_source"]
                support_summary[method] = dict(
                    native_frames=STOP - START,
                    source_rows=int(held.sum()),
                    observed_nodes=int(actual["observed"].sum()),
                    maximum_ray_mm=float(np.max(actual["ray_distance_mm"][held])),
                    maximum_pixel_error=float(np.max(actual["native_pixel_error"][held])),
                    all_saved_sources_and_observed_exact=True,
                    per_view_node_exclusivity=True,
                )
                checked[method] = actual
                np.savez_compressed(output / f"{method}_same_horizon.npz", **candidate)
                np.savez_compressed(output / f"{method}_support.npz", **actual)
            cross_input = {
                key: value.copy()
                for key, value in values["directional"].items()
                if key in ("frame_idx", "pose_xyz", "sources")
            }
            cross_input["sources"][..., 0] = values["original"]["sources"][..., 0]
            cross = baseline.classify_strict_support(cross_input, nodes, rig)
            np.savez_compressed(output / "old_mouth_rows_at_new_pose.npz", **cross)
            availability = available_old_sources(values["original"], values["directional"], cross)
            rows, lost, gained = case_rows(values, checked, cross, availability)
            np.savez_compressed(
                output / "changes_and_exclusivity.npz",
                frame_idx=values["original"]["frame_idx"],
                lost=lost,
                gained=gained,
                **availability,
            )
            categories = {
                name: sum(row["change"] == "lost" and row["category"] == name for row in rows)
                for name in (
                    "old_rows_can_augment_saved_sources_without_collision",
                    "old_rows_can_augment_only_ignoring_current_collision",
                    "old_and_current_rows_geometrically_insufficient_at_new_pose",
                )
            }
            all_runs = {"lost": runs(lost), "gained": runs(gained)}
            summary = dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                same_backtrace_horizon=STOP,
                interim_audit_sha256=AUDIT_SHA,
                parent_minute_completed_claim=False,
                mouth_counts={
                    name: int(value["observed"][..., 0].sum()) for name, value in values.items()
                },
                mouth_by_slot={
                    name: value["observed"][..., 0].sum(axis=0).tolist()
                    for name, value in values.items()
                },
                lost_fish_frames=int(lost.sum()),
                gained_fish_frames=int(gained.sum()),
                lost_native_frames=int(lost.any(1).sum()),
                gained_native_frames=int(gained.any(1).sum()),
                lost_by_slot=lost.sum(0).tolist(),
                gained_by_slot=gained.sum(0).tolist(),
                lost_categories=categories,
                runs=all_runs,
                full_source_support=support_summary,
                old_rows_alone_support_new_pose_in_lost=int(
                    (lost & cross["observed"][..., 0]).sum()
                ),
                method_metrics={name: prior["methods"][name]["ten_seconds"] for name in values},
                requested_chest_reacquisition_67647=prior["methods"]["directional"]["gap_metrics"][
                    "reacquisitions"
                ],
                no_fit=True,
                no_GPU=True,
                no_raw_decode=True,
                no_GT=True,
                no_permanent_ID_matching=True,
                saved_sources_are_post_filter_not_fit_plan=True,
                historical_input_SHA_scope=(
                    "Referenced successful interim seal; used checkpoints/source/SL/calibration "
                    "are rehashed here. Original videos/masks are not reread."
                ),
                elapsed_seconds_before_final_guards=time.perf_counter() - started,
            )
            # Keep only the specifically requested recovery event, not unrelated events.
            summary["requested_chest_reacquisition_67647"] = [
                r for r in summary["requested_chest_reacquisition_67647"] if r["frame_idx"] == 67647
            ]
            for name, value in (("summary.json", summary), ("runs.json", all_runs)):
                (output / name).write_text(
                    json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
                )
            (output / "cases.jsonl").write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
            )
            with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
                fields = [
                    "frame_idx",
                    "slot",
                    "change",
                    "category",
                    "old_rows_supported_views_at_new_pose",
                    "augmented_supported_views",
                    "mouth_displacement_mm",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: row[k] for k in fields} for row in rows)
            (output / "RESULT.md").write_text(report(summary), encoding="utf-8")
            for path in output.iterdir():
                if path.is_file():
                    record.output(path.name, path, sha256=checks.sha(path))
        finally:
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "diagnostic input changed")
            for item in record.outputs.values():
                checks.require(
                    checks.sha(item["path"]) == item["sha256"], "diagnostic output changed"
                )
        record.param("all_opened_inputs_sources_snapshots_outputs_unchanged", True)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "mouth_counts",
                    "lost_fish_frames",
                    "gained_fish_frames",
                    "lost_categories",
                    "full_source_support",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
