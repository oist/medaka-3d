"""Seal the audited frame-49800 seed after independent native SL/ray verification.

Only the selected first source/availability rows are read from the historical C34
archive. No baseline coordinate member or human annotation is opened. The output
is an optimistic initialization registration, not a biological identity label.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from node_geometry import RayObservations, supported_nodes
from observations import CachedNodes
from run_experiment import (
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    sha256,
    skeleton_from_sources,
    staging_catalog,
    verified_input,
)

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
START, STOP = 49800, 348600
LOWER = np.array([0.0, 0.0, -6.0])
UPPER = np.array([192.0, 192.0, 186.0])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_baseline_row(path, member, row_shape):
    """Decode only one fixed native row; skip prior bytes without array loading."""
    with zipfile.ZipFile(path) as archive, archive.open(member + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "unexpected baseline NPY header version")
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0) else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(stream)
        require(
            shape == (STOP, *row_shape) and not fortran and dtype.kind in "iu",
            "baseline native array shape/order/dtype differs from the first-frame audit",
        )
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        stream.seek(START * row_bytes, 1)
        data = stream.read(row_bytes)
        require(len(data) == row_bytes, "incomplete first baseline source row")
        return np.frombuffer(data, dtype=dtype).reshape(row_shape).copy()


def code_paths():
    paths = {Path(__file__).resolve(), LAB / "tracker.py"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                paths.add(path)
    return sorted(paths)


def run(audit_path, input_root, output):
    audit_path, input_root, output = (
        Path(p).resolve() for p in (audit_path, input_root, output)
    )
    if output.exists():
        raise FileExistsError(f"no overwrite: {output}")
    codes = {p: sha256(p) for p in code_paths()}
    audit_sha = sha256(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(
        audit["start"] == START and audit["stop_exclusive"] == STOP
        and audit["frames"] == STOP - START and audit["fps"] == 166,
        "prior audit does not name the registered exact 30-minute native interval",
    )
    require(
        audit["human_labels_opened"] is False
        and audit["later_baseline_frames_used_for_seeding"] is False,
        "prior audit does not declare first-frame-only seeding",
    )
    catalog, catalog_path = staging_catalog(input_root)
    baseline_relative = f"{BASELINE_DIR}/C34-label__{EXPERIMENT}__lam0.767.npz"
    baseline = verified_input(input_root, baseline_relative, catalog)
    old_script = REPO / "work/three_body/audit_long_start.py"
    expected = {baseline, catalog_path, old_script}
    for key in catalog:
        if key.startswith(CALIBRATION_DIR + "/") or key.startswith("work/c01_s16_predictions/"):
            expected.add(verified_input(input_root, key, catalog))
    inherited = {Path(p).resolve(): h for p, h in audit["input_sha256"].items()}
    require(len(inherited) == 10 and set(inherited) == expected, "prior ten input roles changed")
    require(all(sha256(p) == h for p, h in inherited.items()), "prior audited input SHA changed")
    source = first_baseline_row(baseline, "source", (3, 3))
    available = first_baseline_row(baseline, "slot_track", (3,)) >= 0
    require(np.array_equal(source, audit["seed_source"]), "first source row differs from audit")
    require(
        available.all() and np.array_equal(available, audit["baseline_slots_available"]),
        "first baseline slot availability differs from audit",
    )
    frames = np.array([START], np.int64)
    nodes = CachedNodes(input_root, EXPERIMENT, frames)
    rig = load_rig(input_root / CALIBRATION_DIR)
    require(tuple(rig.views) == tuple(VIEWS), "calibration view order changed")
    points = {view: nodes.at(view, START) for view in VIEWS}
    for v, view in enumerate(VIEWS):
        require(
            np.all(source[:, v] >= 0) and np.all(source[:, v] < len(points[view]))
            and len(np.unique(source[:, v])) == 3,
            f"nonexclusive or unavailable first-frame SL source for {view}",
        )
        require(np.isfinite(points[view][source[:, v]]).all(), f"nonfinite SL source for {view}")
    reconstructed, triangulated = skeleton_from_sources(rig, nodes, frames, source[None])
    pose = np.asarray(audit["seed_pose_xyz"], np.float64)
    require(pose.shape == (3, 3, 3), "invalid prior pose shape")
    require(np.array_equal(reconstructed[0], pose), "recomputed seed pose is not bitwise equal")
    require(
        np.isfinite(pose).all() and np.all(pose >= LOWER) and np.all(pose <= UPPER),
        "seed is outside the unchanged tracker engineering bounds",
    )
    lengths = np.linalg.norm(np.diff(pose, axis=1), axis=-1)
    require(np.all(lengths >= 0.1), "seed has a degenerate tracker body segment")
    rays = RayObservations.from_points(rig, points)
    plan = np.repeat(source[..., None], 3, axis=-1).astype(np.int32)
    strict_sources, observed, ray_distance = supported_nodes(
        pose, plan, rays, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0
    )
    projected = np.stack([rig.camera(view).project(pose) for view in VIEWS], axis=1)
    selected_points = np.stack([points[view][source[:, v]] for v, view in enumerate(VIEWS)], axis=1)
    pixel_error = np.linalg.norm(projected - selected_points, axis=-1)
    direct = np.isfinite(pixel_error) & (pixel_error <= 12.0) & (ray_distance <= 2.0)
    require(np.isfinite(projected).all(), "first-pose refractive projection is nonfinite")
    require(np.array_equal(strict_sources, audit["strict_sources"]), "source support changed")
    require(
        triangulated.all() and observed.all() and np.all(direct.sum(axis=1) >= 2)
        and np.array_equal(observed, audit["strict_direct_support"])
        and np.array_equal(triangulated[0], audit["triangulated_nodes"]),
        "seed no longer has direct support for all nine nodes",
    )
    inputs = {**inherited, audit_path: audit_sha}
    output.mkdir(parents=True)
    with provenance("three_body_long_seed_registration", output, name="registration") as rec:
        for path, digest in inputs.items():
            rec.input(str(path), path, sha256=digest)
        for path, digest in codes.items():
            rec.input("code:" + path.relative_to(REPO).as_posix(), path, sha256=digest)
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            require(sha256(target) == digest, "code changed while snapshotting")
            rec.output(target.relative_to(output).as_posix(), target, sha256=digest)
        # Full baseline/c01 arrays are deliberately not copied to the registration.
        snapshots = {
            audit_path: "prior_audit.json", old_script: "audit_long_start.py",
            catalog_path: "staging_manifest.json",
        }
        for path in inherited:
            if path.parent == input_root / CALIBRATION_DIR:
                snapshots[path] = "calibration/" + path.name
        for path, name in snapshots.items():
            target = output / "input_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            require(sha256(target) == inputs[path], "input changed while snapshotting")
            rec.output(target.relative_to(output).as_posix(), target, sha256=inputs[path])
        initial_path, seed_path = output / "initial_pose_xyz.npy", output / "seed.npz"
        np.save(initial_path, pose, allow_pickle=False)
        np.savez_compressed(
            seed_path, frame_idx=frames, pose_xyz=pose, source_rows=source.astype(np.int64),
            sources=strict_sources, observed=observed, projected_uv=projected,
            selected_points_uv=selected_points, ray_distance_mm=ray_distance,
            reprojection_error_px=pixel_error, direct_view=direct,
            segment_lengths_mm=lengths,
            **{f"sleap_points_{view}": points[view] for view in VIEWS},
        )
        for path in (initial_path, seed_path):
            rec.output(path.name, path, sha256=sha256(path))
        registration = {
            "schema": "three_body_first_frame_seed_registration_v1",
            "experiment": EXPERIMENT, "frame": START, "source_frame_range": [START, STOP],
            "fps": 166, "views": list(VIEWS), "nodes": ["mouth", "pectoral", "tail"],
            "scope": "one initial pose only; no tracking or identity certification",
            "initial_pose": {"path": initial_path.name, "sha256": sha256(initial_path)},
            "seed": {"path": seed_path.name, "sha256": sha256(seed_path)},
            "pose_xyz": pose.tolist(), "source_rows": source.tolist(),
            "sources": strict_sources.tolist(), "observed": observed.tolist(),
            "direct_views_per_node": direct.sum(axis=1).tolist(),
            "maximum_ray_distance_mm": float(ray_distance.max()),
            "maximum_reprojection_error_px": float(pixel_error.max()),
            "recomputed_pose_bitwise_equal_to_prior_audit": True,
            "support_gate": {"minimum_views": 2, "ray_mm": 2.0, "pixel_error": 12.0},
            "engineering_bounds_mm": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
            "physical_floor_z0_violations": np.argwhere(pose[..., 2] < 0).tolist(),
            "initialization_count_allowed": 1, "external_resets_allowed": 0,
            "baseline_relative": baseline_relative, "baseline_sha256": inherited[baseline],
            "baseline_members_opened": ["source.npy", "slot_track.npy"],
            "baseline_rows_deserialized": [START], "baseline_coordinates_opened": False,
            "human_ground_truth_opened": False, "later_baseline_arrays_exported": False,
            "initial_identity_limitation": (
                "Optimistic initial source correspondence inherited from the whole-recording "
                "offline C34 baseline. Direct ray support validates geometric consistency, "
                "not biological identity or independent initialization. Raw slots are arbitrary."
            ),
            "bounds_limitation": (
                "Engineering z extends to -6 mm; a seed mouth below physical z=0 is retained "
                "without clipping. The physical-floor constraint is not certified."
            ),
            "background_protocol": "Not supplied by this seed registration.",
            "prior_audit_sha256": audit_sha,
            "recorded_prior_ten_inputs_reverified": True,
            "input_hashes": {str(p): h for p, h in inputs.items()},
            "code_hashes": {p.relative_to(REPO).as_posix(): h for p, h in codes.items()},
            "inputs_unchanged_during_registration": True,
            "code_unchanged_during_registration": True,
        }
        target = output / "registration.json"
        target.write_text(json.dumps(registration, indent=2, allow_nan=False), encoding="utf-8")
        rec.output(target.name, target, sha256=sha256(target))
        rec.param("scope", registration["scope"])
        rec.param("frame", START)
        rec.param("source_frame_range", [START, STOP])
        rec.param("initial_pose_optimistic", True)
        for path, digest in {**inputs, **codes}.items():
            require(sha256(path) == digest, f"registration input or source changed: {path}")
        for info in rec.outputs.values():
            require(sha256(info["path"]) == info["sha256"], "registration output changed")
    print(json.dumps({
        "output": str(output), "registration_sha256": sha256(target),
        "direct_views_per_node": registration["direct_views_per_node"],
        "maximum_ray_distance_mm": registration["maximum_ray_distance_mm"],
        "maximum_reprojection_error_px": registration["maximum_reprojection_error_px"],
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=REPO / "work/three_body/long_window_first_frame_audit/result.json")
    parser.add_argument("--input-root", type=Path, default=REPO / "work/generative_tracking_inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.input_root, args.output)


if __name__ == "__main__":
    main()
