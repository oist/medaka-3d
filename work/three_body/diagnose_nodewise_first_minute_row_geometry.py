"""Triangulate one visually selected SLEAP row tuple from saved small JSON only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))

from body_row_proposals import DEFAULT_CONFIG  # noqa: E402
from node_geometry import RayObservations  # noqa: E402

from medaka3d.calib import load_rig, triangulate  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

INPUT = WORK / "nodewise_first_minute_raw_detection_review_v1"
SEAL_SHA = "846aa2c99a22b2ac3d4a8b9f87d7291a0fef12356f31e76271571cf8b602871e"
PLAN = WORK / "nodewise_body_row_five_minute_continuation_v1/plan.json"
PLAN_SHA = "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
INDEX = PLAN.parent / "input_index.json"
INDEX_SHA = "f5840d96f166f04516c72326aee157e78fac01a290296653aabdec01dd8ae568"
FRAMES = (75053, 75054, 75055)
ROWS = (0, 1, 1)
VIEW_SETS = ((0, 1, 2), (0, 1), (0, 2), (1, 2))
NODES = ("mouth", "pectoral", "tail")


def signature(value):
    a = np.ascontiguousarray(value)
    return dict(
        dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def analyze(points, rig, lower, upper, config=DEFAULT_CONFIG):
    """Evaluate the row tuple in isolation; no pool enumeration or scene ranking."""
    rays = RayObservations.from_points(rig, points)
    origins = np.stack([rays.origins[v][ROWS[v]] for v in range(3)], axis=1)
    directions = np.stack([rays.directions[v][ROWS[v]] for v in range(3)], axis=1)
    selected = np.stack([points[v][ROWS[i]] for i, v in enumerate(VIEWS)], axis=1)
    if not np.isfinite([origins, directions]).all() or not np.isfinite(selected).all():
        raise ValueError("the fixed selected row tuple requires all nine finite rays and pixels")
    results = []
    for views in VIEW_SETS:
        held = np.broadcast_to(np.isin(np.arange(3), views), (3, 3)).copy()
        xyz, rms = triangulate(origins, directions, held)
        offset = xyz[:, None, :] - origins
        forward = np.sum(offset * directions, axis=-1)
        distance = np.linalg.norm(offset - forward[..., None] * directions, axis=-1)
        projected = np.stack([rig.camera(v).project(xyz) for v in VIEWS], axis=1)
        error = np.linalg.norm(projected - selected, axis=-1)
        finite = np.isfinite(xyz).all(-1) & np.isfinite(rms)
        gate = dict(
            finite=finite,
            minimum_two_rays=held.sum(-1) >= 2,
            ray_rms=rms <= config.maximum_ray_rms_mm,
            lower_box=(xyz >= lower).all(-1),
            upper_box=(xyz <= upper).all(-1),
            forward_used_views=np.all(~held | (forward >= 0), axis=-1),
            reprojection_used_views=np.all(~held | (error <= config.maximum_pixel_error), axis=-1),
        )
        good = np.logical_and.reduce(list(gate.values()))
        # Independent numerical check of the returned RMS definition.
        rms_from_distances = np.sqrt(np.where(held, distance**2, 0).sum(-1) / held.sum(-1))
        if not np.allclose(rms, rms_from_distances, atol=1e-12, rtol=1e-12, equal_nan=True):
            raise ValueError("triangulation RMS disagrees with explicit perpendicular distances")
        normal = np.eye(3) - directions[..., :, None] * directions[..., None, :]
        eigenvalues = np.linalg.eigvalsh(np.where(held[..., None, None], normal, 0).sum(1))
        results.append(
            dict(
                used_view_indices=list(views),
                used_views=[VIEWS[v] for v in views],
                pose_xyz=xyz,
                ray_rms_mm=rms,
                perpendicular_ray_mm_by_node_view=distance,
                forward_ray_mm_by_node_view=forward,
                projected_xy_by_node_view=projected,
                reprojection_error_px_by_node_view=error,
                least_squares_normal_eigenvalues=eigenvalues,
                node_gates=gate,
                node_passes_all_body_pool_gates=good,
                failed_gates_by_node={
                    n: [k for k, v in gate.items() if not v[i]] for i, n in enumerate(NODES)
                },
                all_three_nodes_pass=bool(good.all()),
                segment_lengths_mm=np.linalg.norm(np.diff(xyz, axis=0), axis=-1),
                unused_view_errors_are_diagnostic_not_gated=True,
            )
        )
    preferred = {}
    for i, node in enumerate(NODES):
        options = (
            [0]
            if results[0]["node_passes_all_body_pool_gates"][i]
            else [j for j in (1, 2, 3) if results[j]["node_passes_all_body_pool_gates"][i]]
        )
        preferred[node] = [list(VIEW_SETS[j]) for j in options]
    return dict(
        source_rows_by_view=dict(zip(VIEWS, ROWS, strict=True)),
        selected_points_by_node_view=selected,
        origins_by_node_view=origins,
        directions_by_node_view=directions,
        reconstructions=results,
        nodewise_allowed_view_options_for_this_row=preferred,
        all_nodes_have_geometric_option=all(preferred.values()),
        actual_body_pool_membership_checked=False,
        joint_three_body_scene_or_cap_checked=False,
        state0_proposal_gate_or_retained_history_checked=False,
    )


def run(output):
    if output.exists():
        raise FileExistsError(output)
    inputs = {}

    def capture(path, expected=None):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"input/source SHA differs: {path}")
        inputs[str(path)] = digest
        return data

    seal = json.loads(capture(INPUT / "review.provenance.json", SEAL_SHA))
    if seal["status"] != "ok" or seal["stage"] != INPUT.name:
        raise ValueError("successful saved raw-detection review required")
    ref = seal["outputs"]["diagnostics.json"]
    if Path(ref["path"]) != INPUT / "diagnostics.json":
        raise ValueError("foreign saved observations")
    data = capture(Path(ref["path"]), ref["sha256"])
    if len(data) != ref["size_bytes"]:
        raise ValueError("observation size differs")
    doc = json.loads(data)
    plan = json.loads(capture(PLAN, PLAN_SHA))
    index = json.loads(capture(INDEX, INDEX_SHA))
    lower, upper = np.array([0.0, 0.0, -6.0]), np.array([192.0, 192.0, 186.0])
    for name, value in (("lower", lower), ("upper", upper)):
        if signature(value) != plan["tracker_context"][name]:
            raise ValueError("live engineering bound differs")
    policy = plan["tracker_context"]["body_row_policy"]
    if policy["config"] != asdict(DEFAULT_CONFIG):
        raise ValueError("body pool gate config differs from live producer")
    for path in (
        WORK / "body_row_proposals.py",
        WORK / "nodewise_body_row_proposals.py",
        REPO / "labs/20260909_three_body/node_geometry.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
    ):
        capture(path, plan["source_sha256"][str(path)])
    capture(Path(__file__).resolve())
    shared = index["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    rig_paths = {Path(p): h for p, h in shared.items() if Path(p).suffix == ".json"}
    if len(rig_paths) != 4:
        raise ValueError("registered calibration manifest and three camera models required")
    for path, digest in rig_paths.items():
        capture(path, digest)
    manifests = [p for p in rig_paths if p.name == "calibration_manifest.json"]
    if len(manifests) != 1:
        raise ValueError("one registered rig required")
    rig_doc = json.loads(manifests[0].read_bytes())
    if {manifests[0].parent / x["path"] for x in rig_doc["model_files"]} != set(rig_paths) - set(
        manifests
    ):
        raise ValueError("rig would consume unregistered camera files")
    rig = load_rig(manifests[0].parent)
    if tuple(doc["frames"]) != FRAMES or tuple(doc["views"]) != tuple(VIEWS):
        raise ValueError("saved native/view scope differs")
    records = {(x["frame_idx"], x["view"]): x for x in doc["observations"]}
    if len(records) != 9 or len(doc["observations"]) != 9:
        raise ValueError("duplicate or incomplete observations")
    cases = []
    for frame in FRAMES:
        points = {}
        for view in VIEWS:
            row = records[frame, view]
            points[view] = np.asarray(row["raw_SL_points"], dtype=np.float32)
            expected = doc["raw_SL_selected_signatures"][f"f{frame}_{view}"]
            if (
                signature(points[view]) != row["raw_SL_signature"]
                or signature(points[view]) != expected
            ):
                raise ValueError("saved JSON-to-float32 observation signature differs")
        cases.append(dict(frame_idx=frame, **analyze(points, rig, lower, upper)))
    report = dict(
        schema="nodewise_first_minute_row011_geometry_v1",
        status="ok",
        input_sha256=inputs,
        scope=dict(frames=FRAMES, nodes=NODES, source_rows=ROWS, view_sets=VIEW_SETS),
        config=asdict(DEFAULT_CONFIG),
        engineering_lower_mm=lower,
        engineering_upper_mm=upper,
        initial_length_ratio_gate=[
            DEFAULT_CONFIG.minimum_length_ratio,
            DEFAULT_CONFIG.maximum_length_ratio,
        ],
        actual_initial_length_ratio_not_evaluated=True,
        cases=cases,
        limitations=[
            "Root visually selected this row tuple; it is not anatomical or physical ID truth.",
            "Equal raw-row indices across three frames do not establish identity.",
            "Body-node gates here evaluate unfit triangulations, not optimized candidates.",
            "No joint scene, initial-length ratio, quota, state0 gate or beam/cost reconstruction.",
            "Passing geometry does not prove actual proposal generation, admission or selection.",
            "Post-fit support uses a separate 2mm/12px gate; body-pool RMS is 1.5mm here.",
            "The sixth term is neutral at 75053/54/55; its live activation begins 76360.",
        ],
        original_SL_mask_video_candidate_checkpoint_reads=0,
        fit_grid_tracking_GPU_calls=0,
        physical_ID_truth=False,
    )
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source/small input changed during diagnosis: {path}")
    output.mkdir(parents=True, exist_ok=False)
    result = output / "diagnostics.json"
    result.write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# 75053-75055: SL行 (0,1,1) の屈折三角測量",
        "",
        "| native | 使用view | M/C/T ray RMS (mm) | 使用view最大再投影誤差 (px) | M/C/T gate |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        for item in case["reconstructions"]:
            views = item["used_view_indices"]
            rms = "/".join(f"{x:.4f}" for x in item["ray_rms_mm"])
            pixels = "/".join(
                f"{x:.3f}" for x in item["reprojection_error_px_by_node_view"][:, views].max(1)
            )
            passed = "/".join(
                "PASS" if x else "FAIL" for x in item["node_passes_all_body_pool_gates"]
            )
            rows.append(f"| {case['frame_idx']} | {views} | {rms} | {pixels} | {passed} |")
    rows += [
        "",
        "view順はtop/side1/side2。既存の1.5mm RMS、12px、前向き光線、"
        "探索box [0,0,-6]..[192,192,186] を使用。"
        "全視点の誤差・不通過理由・3D座標は diagnostics.json に保存。",
        "",
        "この3D照合は局所的な幾何条件の検査であり、"
        "当時の候補生成・容量・親状態・費用・beam・最終選択を再現していない。"
        "初期長に対する比0.5-1.5のscene gateも今回未評価。"
        "raw行番号は個体IDではない。75053-75055の第6項はneutral。",
        "",
        "元SL/mask archive・動画・candidate・39状態の読込、fit、追跡、GPUは0。"
        "保存JSONのfloat32署名と登録rig4JSONを確認した。",
    ]
    (output / "RESULT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                "diagnostics_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                "all_three_view_pass": [
                    bool(c["reconstructions"][0]["all_three_nodes_pass"]) for c in cases
                ],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
