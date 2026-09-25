"""Six-frame projection/SL diagnostics from one sealed renderer JSON only.

No video, mask pixels, candidate NPZ, calibration projection, fit or ID solve.
SL rows are local to each frame/view and are fitting inputs, not identity truth.
12 native pixels is an existing support scale, not a new adoption threshold.
"""

# Japanese report prose intentionally uses native punctuation and characters.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
SOURCE_ROOT = WORK / "fast_tail_five_minute_id_transition_raw_review_v2"
SEAL_SHA = "0063b4be2dd9b37200d279c9042dff4862e38d84f9c01446f6559beb146a1924"
EVIDENCE_SHA = "5405ce02b34cd4c1673487f7427d101ccf333cfef869dc740a9acb426ea70d41"
STAGE = "saved_id_transition_body_consistency_v1"
FRAMES = (90141, 90147, 90258, 90259, 90269, 90290)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODES = ("mouth", "chest", "tail")
PIXELS = 12.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def finite(value):
    return float(value) if np.isfinite(value) else None


def best_rows(cost):
    if not np.isfinite(cost).any():
        return []
    return np.flatnonzero(cost == np.min(cost)).tolist()


def view_diagnostics(projected, points, sources, observed):
    projected, points = np.asarray(projected, float), np.asarray(points, float)
    require(projected.shape == (3, 3, 2), "three projected M/C/T bodies required")
    if points.size == 0:
        points = points.reshape(0, 3, 2)
    require(points.ndim == 3 and points.shape[1:] == (3, 2), "SL row shape differs")
    require(sources.shape == observed.shape == (3, 3), "view source/support shape differs")
    require(sources.dtype.kind in "iu" and observed.dtype.kind == "b", "source/support dtype")
    delta = projected[:, None] - points[None]
    distances = np.linalg.norm(delta, axis=-1)
    distances[~np.isfinite(distances)] = np.inf  # (raw slot, SL row, node)
    slots, third_view_examples, assigned_outside = [], [], []
    for slot in range(3):
        nearest, grouped = {}, []
        for node, name in enumerate(NODES):
            values = distances[slot, :, node]
            best = best_rows(values)
            error = finite(values[best[0]]) if best else None
            source = int(sources[slot, node])
            require(-1 <= source < len(points), "saved source row out of range")
            assigned_error = finite(values[source]) if source >= 0 else None
            info = dict(
                SL_rows=best,
                nearest_distance_px=error,
                nearest_within_12px=error is not None and error <= PIXELS,
                assigned_SL_row=source,
                assigned_distance_px=assigned_error,
                global_observed=bool(observed[slot, node]),
            )
            nearest[name] = info
            if observed[slot, node] and source == -1 and error is not None and error > PIXELS:
                third_view_examples.append(dict(slot=slot, node=name, **info))
            if source >= 0 and (assigned_error is None or assigned_error > PIXELS):
                assigned_outside.append(dict(slot=slot, node=name, **info))
        complete_cost = np.full(len(points), np.inf)
        for row in range(len(points)):
            values = distances[slot, row]
            valid = np.isfinite(values)
            count = int(valid.sum())
            rms = float(np.sqrt(np.mean(values[valid] ** 2))) if count else None
            if count == 3:
                complete_cost[row] = rms
            grouped.append(
                dict(
                    SL_row=row,
                    node_distances_px=[finite(v) for v in values],
                    comparable_nodes=count,
                    RMS_of_available_nodes_px=rms,
                    complete_MCT_RMS_px=rms if count == 3 else None,
                    complete_MCT_max_px=finite(values.max()) if count == 3 else None,
                    nodes_within_12px=int((values <= PIXELS).sum()),
                    all_three_nodes_within_12px=bool((values <= PIXELS).all()),
                )
            )
        slots.append(
            dict(
                raw_slot=slot,
                nearest_by_node=nearest,
                same_SL_row_bodies=grouped,
                best_complete_body_SL_rows=best_rows(complete_cost),
                best_complete_body_RMS_px=finite(complete_cost.min()) if len(points) else None,
            )
        )
    chest = distances[:, :, 1]
    nearest = [best_rows(row) for row in chest]
    unique = [rows[0] if len(rows) == 1 else None for rows in nearest]
    duplicate_pairs = [
        [a, b]
        for a, b in itertools.combinations(range(3), 2)
        if unique[a] is not None and unique[a] == unique[b]
    ]
    assignments = []
    for rows in itertools.permutations(range(len(points)), 3):
        values = chest[np.arange(3), rows]
        if np.isfinite(values).all():
            assignments.append((float(values @ values), rows, values))
    assignments.sort(key=lambda row: (row[0], row[1]))
    independent_cost = sum(float(row.min() ** 2) for row in chest) if len(points) else np.inf
    assignment = dict(
        objective="sum of squared native-pixel chest distances; distinct SL rows",
        independently_nearest_SL_rows=nearest,
        duplicated_unique_nearest_slot_pairs=duplicate_pairs,
        independent_nearest_sum_squared_px=finite(independent_cost),
        feasible_assignment_count=len(assignments),
        feasible=bool(assignments),
        reason=None if assignments else "fewer than three finite assignable chest rows",
        best_distinct_SL_rows=None,
        best_node_distances_px=None,
        best_sum_squared_px=None,
        excess_sum_squared_px_over_independent=None,
        equal_best_assignments=[],
    )
    if assignments:
        cost, rows, values = assignments[0]
        assignment.update(
            best_distinct_SL_rows=list(rows),
            best_node_distances_px=values.tolist(),
            best_sum_squared_px=cost,
            excess_sum_squared_px_over_independent=cost - independent_cost,
            equal_best_assignments=[list(r) for c, r, _ in assignments if c == cost],
        )
    return dict(
        SL_rows=len(points),
        slots=slots,
        chest_one_to_one=assignment,
        globally_observed_unassigned_view_over_12px=third_view_examples,
        assigned_source_over_12px=assigned_outside,
    )


def analyze(records):
    require(tuple(r["frame"] for r in records) == FRAMES, "fixed six records required")
    output, examples = [], []
    for record in records:
        raw = record["raw_candidate"]
        source = np.asarray(raw["sources"])
        observed = np.asarray(raw["observed"])
        require(source.shape == (3, 3, 3) and observed.shape == (3, 3), "candidate shape")
        views = {}
        for vi, view in enumerate(VIEWS):
            obs = record["observations"][view]
            result = view_diagnostics(
                record["raw_projections"][view], obs["SLEAP_points"], source[:, vi], observed
            )
            result.update(mask_count=obs["mask_count"], mask_pixels_available_here=False)
            views[view] = result
            examples.extend(
                dict(frame=record["frame"], view=view, **example)
                for example in result["globally_observed_unassigned_view_over_12px"]
            )
        output.append(dict(frame=record["frame"], global_observed=observed.tolist(), views=views))
    return dict(records=output, globally_observed_unassigned_view_over_12px=examples)


def number(value):
    return "なし" if value is None else f"{value:.2f}"


def report_text(result):
    lines = [
        "# 保存6時点の投影とSLEAP群の整合性",
        "",
        "6時点×3視点を保存JSONだけから診断した。12 pxは既存の観測支持尺度。"
        "SLはfit入力であり真値ではない。",
        "`observed=True` は2視点以上の支持であり、全3視点が対応するという意味ではない。"
        "2 mmのray距離は再計算していない。",
        "",
        "|frame|view|slot/node|最寄りSL距離 px|このviewのsource|全体observed|",
        "|---:|---|---|---:|---:|---|",
    ]
    for item in result["globally_observed_unassigned_view_over_12px"]:
        lines.append(
            f"|{item['frame']}|{item['view']}|S{item['slot']} {item['node']}|"
            f"{number(item['nearest_distance_px'])}|-1|True|"
        )
    lines.extend(
        [
            "",
            "各視点内で胸部の最寄りSL行を独立に選んだ場合と、3スロットを異なるSL行へ割り当てた場合：",
            "",
            "|frame|view|独立最寄り行 S0/S1/S2|重複slot対|一対一の行 S0/S1/S2|一対一距離 px|",
            "|---:|---|---|---|---|---|",
        ]
    )
    for record in result["records"]:
        for view, value in record["views"].items():
            assignment = value["chest_one_to_one"]
            errors = assignment["best_node_distances_px"]
            error_text = (
                ",".join(number(v) for v in errors) if errors is not None else "3行へ割当不能"
            )
            lines.append(
                f"|{record['frame']}|{view}|{assignment['independently_nearest_SL_rows']}|"
                f"{assignment['duplicated_unique_nearest_slot_pairs']}|"
                f"{assignment['best_distinct_SL_rows']}|{error_text}|"
            )
    lines.extend(
        [
            "",
            "SL行番号はframe/viewごとに独立であり、topとsideの行番号の違いは異個体対応の証拠にならない。",
            "同一SL行へのM/C/T距離、比較可能ノード数、完全3ノードRMSと最大値はdiagnostic.jsonに保存した。欠測を0誤差として扱っていない。",
            "マスクの枚数だけを参照した。マスク画素がないためIoU、輪郭との一致、画像背景への逸脱をここでは判定していない。",
            "一対一割当は不足する検出や誤検出を含めて距離最小の候補を示すだけで、3個体の正解対応やID連続性を確定しない。",
            "90290についても、視点間の番号差だけから誤対応とは断言しない。全18視点の保存距離を優先して判断する。",
            "",
        ]
    )
    return "\n".join(lines)


def run(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    input_bytes = {}
    for name, sha in (
        ("render.provenance.json", SEAL_SHA),
        ("source_and_frame_evidence.json", EVIDENCE_SHA),
    ):
        value = (SOURCE_ROOT / name).read_bytes()
        require(digest(value) == sha, "fixed saved evidence changed")
        input_bytes[name] = value
    seal = json.loads(input_bytes["render.provenance.json"])
    evidence = json.loads(input_bytes["source_and_frame_evidence.json"])
    require(seal["status"] == evidence["status"] == "ok", "successful renderer required")
    require(
        seal["stage"] == evidence["schema"] == "fast_tail_five_minute_id_transition_raw_review_v1",
        "source stage",
    )
    member = seal["outputs"]["source_and_frame_evidence.json"]
    require(
        member["sha256"] == EVIDENCE_SHA
        and member["size_bytes"] == len(input_bytes["source_and_frame_evidence.json"])
        and Path(member["path"]).resolve() == SOURCE_ROOT / "source_and_frame_evidence.json",
        "evidence output binding",
    )
    require(
        evidence["terminal_horizon"] == 116200 and evidence["frames"] == list(FRAMES),
        "same horizon/six frames",
    )
    result = analyze(evidence["records"])
    result.update(
        schema=STAGE,
        status="ok",
        frames=list(FRAMES),
        source_renderer_seal_sha256=SEAL_SHA,
        source_evidence_sha256=EVIDENCE_SHA,
        terminal_horizon=116200,
        reference_pixels=PIXELS,
        SL_is_fitting_input_not_truth=True,
        SL_row_indices_are_frame_view_local=True,
        ray_distance_recomputed=False,
        mask_pixels_available=False,
        IoU_computed=False,
        physical_ID_truth=False,
        physical_swap_count_estimated=False,
        candidate_SL_mapping_archives_read=0,
        video_mask_checkpoint_reads=0,
        fit_solver_calls=0,
        GPU_used=False,
    )
    outputs = {"diagnostic.json": json_bytes(result), "REPORT.md": report_text(result).encode()}
    source_path = Path(__file__).resolve()
    source = source_path.read_bytes()
    output.mkdir()
    descriptors = {}
    for name, value in outputs.items():
        path = output / name
        path.write_bytes(value)
        require(path.read_bytes() == value, "diagnostic output roundtrip")
        descriptors[name] = dict(path=str(path), sha256=digest(value), size_bytes=len(value))
    for name, value in input_bytes.items():
        require((SOURCE_ROOT / name).read_bytes() == value, "small source input changed")
    require(source_path.read_bytes() == source, "diagnostic source changed")
    provenance = dict(
        stage=STAGE,
        status="ok",
        finished_at=datetime.now(UTC).isoformat(),
        inputs={str(SOURCE_ROOT / name): digest(value) for name, value in input_bytes.items()},
        source=dict(path=str(source_path), sha256=digest(source)),
        outputs=descriptors,
    )
    (output / "diagnosis.provenance.json").write_bytes(json_bytes(provenance))
    print(
        json.dumps(
            dict(
                status="ok",
                views=18,
                counterexamples=len(result["globally_observed_unassigned_view_over_12px"]),
                output=str(output),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
