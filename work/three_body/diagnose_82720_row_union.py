"""Saved-SL-only ray diagnosis for the accepted P6/P9 row union at 82720."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from medaka3d import calib
from medaka3d.types import VIEWS

WORK = Path(__file__).resolve().parent
GEOMETRY = WORK / "whole_row_body_geometry_diagnostic_v1"
VISUAL = WORK / "whole_row_body_geometry_visual_review_v1"
OBSERVATIONS = WORK / "depth_identity_transition_raw_review_v1"
PINS = {
    GEOMETRY
    / "geometry.provenance.json": "39208ffcd2314098fcf89d944ef9bedbe949fd6c6dbc4c19bc88238acf4a27fb",
    VISUAL
    / "render.provenance.json": "ace63cdac8f9730cb5e58e00a1de23cacb9fc8da2e8e43c39d099e70125246bd",
    OBSERVATIONS
    / "render.provenance.json": "9f3721806523e5b656f317684ffffa19cc5c08b94341a5e353d1bd5af383deea",
}
SUBSETS = ((0, 1, 2), (0, 1), (0, 2), (1, 2))
NODES = ("mouth", "chest", "tail")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    value = np.ascontiguousarray(value)
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run():
    output = WORK / "ROW_UNION_82720_RAY_DIAGNOSIS_V1.json"
    note = WORK / "ROW_UNION_82720_RAY_DIAGNOSIS_V1.md"
    require(not output.exists() and not note.exists(), "fresh diagnostic outputs required")
    captured = {}

    def capture(path, digest=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        require(digest is None or actual == digest, "input SHA differs: " + str(path))
        captured[str(path)] = dict(sha256=actual, size_bytes=len(data))
        return data

    def read(path, digest=None):
        return json.loads(capture(path, digest))

    def member(root, seal, role):
        ref = seal["outputs"][role]
        path = (root / role).resolve()
        require(str(path) == ref["path"], "output path differs")
        value = read(path, ref["sha256"])
        require(captured[str(path)]["size_bytes"] == ref["size_bytes"], "output size differs")
        return value

    seals = {p: read(p, h) for p, h in PINS.items()}
    for seal in seals.values():
        require(seal["status"] == "ok", "successful upstream required")
    geometry = seals[GEOMETRY / "geometry.provenance.json"]
    visual = seals[VISUAL / "render.provenance.json"]
    observations = seals[OBSERVATIONS / "render.provenance.json"]
    plan = member(GEOMETRY, geometry, "plan.json")
    saved = member(VISUAL, visual, "summary.json")
    observed = member(OBSERVATIONS, observations, "source_and_frame_evidence.json")
    observed_path = str((OBSERVATIONS / "source_and_frame_evidence.json").resolve())
    require(
        geometry["inputs"][observed_path] == captured[observed_path]["sha256"],
        "geometry's consumed SL evidence differs",
    )
    require(
        visual["inputs"][str(GEOMETRY / "geometry.provenance.json")]
        == PINS[GEOMETRY / "geometry.provenance.json"],
        "visual geometry edge differs",
    )
    calib_path = Path(calib.__file__).resolve()
    capture(calib_path, plan["source_code"][str(calib_path)])
    require(
        geometry["inputs"][str(calib_path)] == captured[str(calib_path)]["sha256"],
        "calibration source edge differs",
    )
    capture(Path(__file__))
    rig_paths = {
        Path(p): h
        for p, h in observed["shared_SL_calibration_sha256"].items()
        if Path(p).suffix == ".json"
    }
    require(len(rig_paths) == 4, "exact registered rig4 required")
    for path, digest in rig_paths.items():
        require(geometry["inputs"][str(path)] == digest, "rig consumed edge differs")
        capture(path, digest)
    manifest = next(p for p in rig_paths if p.name == "calibration_manifest.json")
    rig = calib.load_rig(manifest.parent)
    require(tuple(rig.views) == tuple(VIEWS), "camera order differs")
    record = next(r for r in observed["records"] if r["frame"] == 82720)
    points = {}
    for view in VIEWS:
        row = record["observations"][view]
        all_points = np.asarray(row["SLEAP_points"], np.float32)
        require(signature(all_points) == row["SL_points_signature"], "SL JSON roundtrip differs")
        points[view] = all_points[2].copy()
    origins, directions = rig.rays(points)  # node,view,xyz
    finite = np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
    results = []
    raw_results = []
    for subset in SUBSETS:
        held = finite & np.isin(np.arange(3), subset)[None]
        xyz, rms = calib.triangulate(origins, directions, held)
        projections = {v: rig.camera(v).project(xyz) for v in VIEWS}
        uv_error = np.stack(
            [np.linalg.norm(projections[v] - points[v], axis=-1) for v in VIEWS], axis=-1
        )
        offset = xyz[:, None] - origins
        along = np.sum(offset * directions, axis=-1)
        perpendicular = np.linalg.norm(offset - along[..., None] * directions, axis=-1)
        good = np.isfinite(xyz).all(-1) & np.isfinite(rms)
        good &= rms <= 1.5
        good &= (xyz >= [0, 0, -6]).all(-1) & (xyz <= [192, 192, 186]).all(-1)
        good &= np.all(~held | (along >= 0), axis=-1)
        good &= np.all(~held | (uv_error <= 12), axis=-1)
        nodes = []
        for n, node in enumerate(NODES):
            nodes.append(
                dict(
                    node=node,
                    used_views=[VIEWS[v] for v in np.flatnonzero(held[n])],
                    used_ray_count=int(held[n].sum()),
                    pose_xyz=xyz[n],
                    ray_rms_mm=rms[n],
                    projection_by_view={v: projections[v][n] for v in VIEWS},
                    pixel_error_by_view=dict(zip(VIEWS, uv_error[n], strict=True)),
                    perpendicular_ray_mm_by_view=dict(zip(VIEWS, perpendicular[n], strict=True)),
                    forward_mm_by_view=dict(zip(VIEWS, along[n], strict=True)),
                    passes_existing_node_gates=bool(good[n]),
                    failed_used_view_pixel_gate=[
                        VIEWS[v] for v in range(3) if held[n, v] and not uv_error[n, v] <= 12
                    ],
                    omitted_finite_views_over_12px=[
                        VIEWS[v]
                        for v in range(3)
                        if finite[n, v] and not held[n, v] and uv_error[n, v] > 12
                    ],
                    fails_rms=bool(held[n].sum() >= 2 and not rms[n] <= 1.5),
                )
            )
        measured = held.sum(-1) >= 2
        results.append(
            dict(
                selected_views=[VIEWS[v] for v in subset],
                nodes=nodes,
                accepted_partial_body=bool(measured.sum() >= 2 and np.all(~measured | good)),
                measured_nodes=measured,
            )
        )
        raw_results.append((xyz, rms, measured, good))
    pool_checks = []
    for arm in ("sparse_instance_rescore", "expanded_depth_instance_rescore"):
        case = next(c for c in saved["cases"] if c["arm"] == arm and c["frame_idx"] == 82720)
        metadata = member(GEOMETRY, geometry, f"cases/{arm}_82720.json")
        require(case["pool_input"]["archive"] == metadata["archive"], "pool archive edge differs")
        for index, subset_index in ((6, 2), (9, 3)):
            saved_body = next(r for r in case["geometry_rows"] if r["index"] == index)
            pose = np.asarray(saved_body["pose_xyz"], np.float64)[0]
            pool_body = metadata["tree"]["pool"][index]
            require(
                signature(pose) == pool_body["pose_xyz"]["signature"],
                "saved visual pose differs from accepted geometry signature",
            )
            xyz, rms, measured, _ = raw_results[subset_index]
            require(np.array_equal(xyz, pose, equal_nan=True), "retriangulated P6/P9 pose differs")
            require(
                np.array_equal(measured, np.asarray(saved_body["triangulated_nodes"])[0]),
                "P6/P9 measured nodes differ",
            )
            require(
                np.array_equal(rms, np.asarray(saved_body["ray_rms_mm"], float)[0], equal_nan=True),
                "P6/P9 RMS differs",
            )
            pool_checks.append(
                dict(
                    arm=arm,
                    pool_index=index,
                    source_rows=saved_body["source_rows"][0],
                    exact_equal_nan_pose=True,
                    accepted_pool_signature=signature(pose),
                    geometry_archive_reference=metadata["archive"],
                )
            )
    m = raw_results[0][0][0]
    t = raw_results[0][0][2]
    alternatives = []
    for i, (xyz, rms, _, good) in enumerate(raw_results):
        pose = np.stack((m, xyz[1], t))
        alternatives.append(
            dict(
                chest_selected_views=results[i]["selected_views"],
                pose_xyz=pose,
                segment_lengths_mm=np.linalg.norm(np.diff(pose, axis=0), axis=-1),
                all_three_nodes_individually_pass=bool(
                    raw_results[0][3][0] and good[1] and raw_results[0][3][2]
                ),
                chest_gate_pass=bool(good[1]),
                chest_rms_mm=rms[1],
            )
        )
    shift = raw_results[2][0][1] - raw_results[3][0][1]
    result = dict(
        schema="row_union_82720_ray_diagnosis_v1",
        status="ok",
        frame_idx=82720,
        union_rows=[2, 2, 2],
        view_order=list(VIEWS),
        node_order=list(NODES),
        row_pixels=points,
        available_views_by_node={
            n: [VIEWS[v] for v in np.flatnonzero(finite[i])] for i, n in enumerate(NODES)
        },
        gates=dict(
            ray_rms_mm=1.5,
            assigned_view_pixel_error=12,
            lower=[0, 0, -6],
            upper=[192, 192, 186],
            forward_rays_required=True,
        ),
        subset_results=results,
        accepted_pool_checks=pool_checks,
        p6_minus_p9_chest_xyz_mm=shift,
        p6_p9_chest_distance_mm=float(np.linalg.norm(shift)),
        node_specific_view_combinations=alternatives,
        interpretation=dict(
            same_local_row_is_not_physical_identity=True,
            per_node_view_selection_is_new_policy_not_executed=True,
            template_completion_not_applied=True,
            segment_lengths_are_diagnostics_not_verified_anatomy=True,
            omitted_finite_views_remain_diagnostic_evidence=True,
        ),
        scope=dict(
            saved_json_only=True,
            calibration_JSON_files=4,
            original_SL_cache_reads=0,
            NPZ_reads=0,
            video_or_image_reads=0,
            mask_reads=0,
            tracker_calls=0,
            GPU_used=False,
            calibration_cause_or_detector_cause_not_identified=True,
        ),
        inputs=captured,
    )
    # Final immutability checks cover only the already captured small JSON/source/rig files.
    for path, ref in captured.items():
        data = Path(path).read_bytes()
        require(
            len(data) == ref["size_bytes"] and hashlib.sha256(data).hexdigest() == ref["sha256"],
            "input changed during diagnostic",
        )
    output.write_text(
        json.dumps(safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            safe(
                dict(
                    output=str(output),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    chest_distance_mm=result["p6_p9_chest_distance_mm"],
                    subsets=results,
                    node_specific=alternatives,
                )
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run()
