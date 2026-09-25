"""Apply one-slot reservation to six saved candidate banks, without tracking."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))

from node_geometry import RayObservations, supported_nodes  # noqa: E402
from observation_reserve_selection import (  # noqa: E402
    observation_eligibility,
    select_with_observation_reserve,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

SAMPLES = WORK / "nodewise_first_minute_candidate_reacquisition_v1"
OUT = WORK / "saved_observation_reserve_six_cases_v1"
FRAMES = (75053, 75054, 75055)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=digest(array.tobytes()))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    inputs = {}
    source_bytes = {
        p: p.read_bytes() for p in (Path(__file__), WORK / "observation_reserve_selection.py")
    }
    for path, data in source_bytes.items():
        inputs[str(path)] = dict(sha256=digest(data), size_bytes=len(data))

    def read(path, expected):
        path = Path(path).resolve()
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if digest(data) != expected or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"changed input: {path}")
        inputs[str(path)] = dict(sha256=expected, size_bytes=len(data))
        return data

    seal = json.loads(
        read(
            SAMPLES / "diagnosis.provenance.json",
            "b518608186f3471e044c36b98c31289e124a909d8d8a658931d6d22cc1f9394a",
        )
    )
    if seal["status"] != "ok" or seal["stage"] != SAMPLES.name:
        raise ValueError("completed candidate diagnosis required")

    def sample(name):
        ref = seal["outputs"][name]
        data = read(SAMPLES / name, ref["sha256"])
        if len(data) != ref["size_bytes"]:
            raise ValueError("sample size differs")
        return data

    candidates = json.loads(sample("diagnostics.json"))
    geometry = json.loads(
        read(
            WORK / "nodewise_first_minute_row011_geometry_v1/diagnostics.json",
            "797af289bd0d108aaf8eb70eeb2485e89b7cbd0245408772fb829a824902857c",
        )
    )
    observations = json.loads(
        read(
            WORK / "nodewise_first_minute_raw_detection_review_v1/diagnostics.json",
            "457df7379a599ae8dacd7b28a3410b5bc922c8cf7421d88ba3ef77d1c6063573",
        )
    )
    # Reuse the registered geometric helper and four calibration JSON inputs.
    for relative in (
        "labs/20260909_three_body/node_geometry.py",
        "src/medaka3d/calib.py",
        "src/medaka3d/types.py",
    ):
        path = REPO / relative
        read(path, geometry["input_sha256"][str(path)])
    manifests = [
        Path(p) for p in geometry["input_sha256"] if Path(p).name == "calibration_manifest.json"
    ]
    if len(manifests) != 1:
        raise ValueError("one registered rig manifest required")
    manifest = manifests[0]
    data = json.loads(read(manifest, geometry["input_sha256"][str(manifest)]))
    paths = [manifest] + [manifest.parent / ref["path"] for ref in data["model_files"]]
    rig_refs = {path: geometry["input_sha256"][str(path)] for path in paths}
    if len(rig_refs) != 4:
        raise ValueError("four registered rig files required")
    for path, sha in rig_refs.items():
        read(path, sha)
    manifest = next(path for path in rig_refs if path.name == "calibration_manifest.json")
    rig = load_rig(manifest.parent)
    if tuple(rig.views) != tuple(VIEWS):
        raise ValueError("camera order differs")
    rays = {}
    for frame in FRAMES:
        points = {}
        for view in VIEWS:
            rows = [
                r
                for r in observations["observations"]
                if r["frame_idx"] == frame and r["view"] == view
            ]
            if len(rows) != 1:
                raise ValueError("one observation record per native/view required")
            points[view] = np.asarray(rows[0]["raw_SL_points"], dtype=np.float32)
            if signature(points[view]) != rows[0]["raw_SL_signature"]:
                raise ValueError("saved point signature differs")
        rays[frame] = RayObservations.from_points(rig, points)

    results, sources_saved = [], {}
    banks = sample("candidate_samples.npz")
    with np.load(io.BytesIO(banks), allow_pickle=False) as archive:
        expected_keys = {
            f"{arm}_{frame}_{field}"
            for arm in ("tail", "nodewise")
            for frame in FRAMES
            for field in ("fitted_pose", "proposal_plan", "initial_pose", "initial_pose_available")
        }
        if len(archive.files) != 24 or set(archive.files) != expected_keys:
            raise ValueError("saved six-case sample members differ")
        if {(c["frame_idx"], c["arm"]) for c in candidates["cases"]} != {
            (frame, arm) for frame in FRAMES for arm in ("tail", "nodewise")
        } or len(candidates["cases"]) != 6:
            raise ValueError("exactly six candidate cases required")
        for case in candidates["cases"]:
            frame, arm = case["frame_idx"], case["arm"]
            key = f"{arm}_{frame}"
            pose, plan = (archive[f"{key}_{field}"] for field in ("fitted_pose", "proposal_plan"))
            rows = case["candidates"]
            if [r["index"] for r in rows] != list(range(len(pose))):
                raise ValueError("candidate indices differ")
            path = np.asarray([r["path_cost"] for r in rows])
            parent = np.asarray([r["parent"] for r in rows])
            permutation = np.asarray([r["permutation"] for r in rows])
            parts = np.asarray([r["parts"] for r in rows])
            baseline, _ = select_with_observation_reserve(
                pose, path, parent, permutation, np.zeros(len(pose), bool)
            )
            np.testing.assert_array_equal(baseline, case["retained_indices"])
            postfit = [
                supported_nodes(p, a, rays[frame], rig) for p, a in zip(pose, plan, strict=True)
            ]
            sources = np.stack([value[0] for value in postfit])
            observed = np.stack([value[1] for value in postfit])
            if arm == "tail":
                np.testing.assert_array_equal(
                    observed, [r["fitted_geometric_observed"] for r in rows]
                )
            eligible, checks = observation_eligibility(sources, observed, parts, int(baseline[0]))
            selected, selection = select_with_observation_reserve(
                pose, path, parent, permutation, eligible
            )
            results.append(
                dict(
                    frame_idx=frame,
                    arm=arm,
                    candidates=len(pose),
                    **selection,
                    eligible_indices=np.flatnonzero(eligible),
                    eligibility_checks=checks,
                    baseline_observed=observed[baseline],
                    selected_observed=observed[selected],
                    fitted_sources_signature=signature(sources),
                    fitted_observed_signature=signature(observed),
                    final76360_original_match_indices=case["final76360_match_indices"],
                    same_frame_true_costs=path,
                    candidate_kind=[r["kind"] for r in rows],
                )
            )
            sources_saved[f"{key}_sources"] = sources
            sources_saved[f"{key}_observed"] = observed
    for path, data in source_bytes.items():
        if path.read_bytes() != data:
            raise ValueError("probe source changed during execution")
    OUT.mkdir()
    report = dict(
        schema=OUT.name,
        status="ok",
        inputs=inputs,
        cases=results,
        total_candidates=sum(case["candidates"] for case in results),
        changed_cases=sum(case["changed"] for case in results),
        all_current_best_preserved=all(case["current_best_preserved"] for case in results),
        original_candidate_archive_reads=0,
        original_video_SL_mask_checkpoint_reads=0,
        fitted_support_recomputed=True,
        fitting_or_tracking_or_GPU_calls=0,
        limitations=[
            "One-step candidate membership only: future inputs/parents/fit/path are not rerun.",
            "Supported SLEAP rows are not independent physical identity truth.",
            "Image criterion reuses the original sampled D2 union loss, not per-fish coverage.",
            "This module is not connected to the live five-minute tracker.",
        ],
    )
    (OUT / "diagnostics.json").write_text(
        json.dumps(report, default=plain, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(OUT / "postfit_support.npz", **sources_saved)
    for file, data in source_bytes.items():
        (OUT / file.name).write_bytes(data)
    outputs = {
        p.name: dict(sha256=digest(p.read_bytes()), size_bytes=p.stat().st_size)
        for p in OUT.iterdir()
    }
    (OUT / "probe.provenance.json").write_text(
        json.dumps(dict(stage=OUT.name, status="ok", inputs=inputs, outputs=outputs), indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status="ok",
                output=str(OUT),
                total_candidates=report["total_candidates"],
                changed_cases=report["changed_cases"],
                cases=[
                    {
                        "frame": c["frame_idx"],
                        "arm": c["arm"],
                        "baseline": c["baseline_indices"],
                        "selected": c["selected_indices"],
                        "eligible": c["eligible_indices"].tolist(),
                        "reason": c["reason"],
                        "replacement": c["replacement"],
                    }
                    for c in results
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
