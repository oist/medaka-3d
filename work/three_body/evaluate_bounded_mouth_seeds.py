"""Evaluate all 202 fixed mouth ray combinations with the original support gates.

The bounded initializer receives only current rays and the frozen engineering
box. No reference winner, human point, pixel mask, fit or temporal update enters
the initializer. This assesses proposals, not their eventual fit/beam retention.
"""

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import bounded_ray_seed as solver
import explain_directional_mouth_intersections as prior
import numpy as np
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig, _injective

from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance

checks = prior.checks
REPO = prior.prior.REPO
ROOT = REPO / "work/three_body/directional_mouth_202_intersections_v1"
SEAL_SHA = "3976a27574edfc6e8ac5d15c078a35856caa7cee4009beac6c7625c9a5416c98"
STAGE = "bounded_mouth_seed_fixed202_evaluation"


def member(root, name, hashes):
    path = (root / name).resolve()
    checks.require(str(path) in hashes, "unsealed required input: " + str(path))
    return path


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new work output required"
    )
    code = prior.generated.comparison.source_registry.local_code()
    hashes = dict(code)
    prior.prior.bind(ROOT / "intersections.provenance.json", SEAL_SHA, hashes)
    checks.verify(ROOT, "intersections.provenance.json", prior.STAGE, hashes)
    combinations = [
        json.loads(line)
        for line in member(ROOT, "all202.jsonl", hashes).read_text(encoding="utf-8").splitlines()
    ]
    checks.require(len(combinations) == 202, "fixed202 required")
    gen_root = prior.ROOT
    prior.prior.bind(gen_root / "generation.provenance.json", prior.SHA, hashes)
    generation = checks.verify(
        gen_root, "generation.provenance.json", prior.generated.STAGE, hashes
    )
    checkpoint = prior.prior.BRANCH / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json"):
        prior.prior.audit_input(generation, checkpoint / name, hashes)
    context = checks.read(checkpoint / "metadata.json")["context"]
    config = NodeRecoveryConfig(**context["recovery_config"])
    binding = checks.read(checkpoint / "manifest.json")["bindings"]["original_binding"]
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    calibration = None
    for path, digest in shared.items():
        if Path(path).suffix == ".json":
            prior.prior.bind(path, digest, hashes)
            if Path(path).name == "calibration_manifest.json":
                calibration = Path(path).parent
    checks.require(calibration is not None, "calibration missing")
    rig = load_rig(calibration)
    normalize = prior.generated.comparison.support_audit.COMMON.normalize
    checks.require(
        normalize([dataclasses.asdict(c) for c in rig.cameras]) == context["cameras"],
        "rig context differs",
    )
    rows, frames = [], []
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="evaluation") as record:
        for frame in range(67663, 67696):
            path = member(gen_root, f"f{frame}_generation.npz", hashes)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            scene, plan = data["predicted_pose"], data["predicted_plan"]
            rays = RayObservations(
                {v: data[v + "_points"] for v in prior.prior.VIEWS},
                tuple(data[v + "_ray_origins"] for v in prior.prior.VIEWS),
                tuple(data[v + "_ray_directions"] for v in prior.prior.VIEWS),
            )
            combos = [c for c in combinations if c["frame_idx"] == frame]
            checks.require(
                [c["index_in_frame"] for c in combos] == list(range(len(combos))),
                "combination order differs",
            )
            admitted = []
            for combo in combos:
                source = np.asarray(combo["source_rows"], np.int32)
                views = np.flatnonzero(source >= 0)
                origins = np.array([rays.origins[v][source[v], 0] for v in views])
                directions = np.array([rays.directions[v][source[v], 0] for v in views])
                xyz, error = triangulate(origins[None], directions[None])
                checks.require(
                    np.allclose(xyz[0], combo["xyz"], atol=1e-10, rtol=0)
                    and abs(float(error[0]) - combo["ray_rms_mm"]) <= 1e-10,
                    "original unconstrained intersection differs",
                )
                before = prior.facts(
                    xyz[0],
                    error[0],
                    origins,
                    directions,
                    scene[0, 1],
                    data["initial_lengths"][0, 0],
                    config,
                )
                checks.require(before["first_gate"] == combo["first_gate"], "original gate differs")
                started = time.perf_counter()
                solution = solver.solve_bounded_ray_seed(
                    origins, directions, config.lower_mm, config.upper_mm
                )
                seconds = time.perf_counter() - started
                point = solution.point_xyz
                delta = point - origins
                perp = delta - (delta * directions).sum(-1, keepdims=True) * directions
                rms = float(np.sqrt(np.sum(perp * perp) / len(views)))
                checks.require(
                    abs(rms - solution.rms_mm) <= 1e-10, "normalized/raw ray RMS differs"
                )
                facts = prior.facts(
                    point,
                    rms,
                    origins,
                    directions,
                    scene[0, 1],
                    data["initial_lengths"][0, 0],
                    config,
                )
                candidate, candidate_plan = scene.copy(), plan.copy()
                candidate[0, 0], candidate_plan[0, :, 0] = point, source
                checks.require(_injective(candidate_plan), "bounded point stole protected row")
                strict_sources, observed, distance = supported_nodes(
                    candidate, candidate_plan, rays, rig
                )
                mouth_chest_direct = bool(observed[0, 0] and observed[0, 1])
                accepted = facts["first_gate"] == "reaches_strict_support" and mouth_chest_direct
                if facts["first_gate"] == "reaches_strict_support":
                    facts["first_gate"] = "accepted" if accepted else "strict_support"
                item = dict(
                    frame_idx=frame,
                    index_in_frame=combo["index_in_frame"],
                    source_rows=source,
                    original_first_gate=combo["first_gate"],
                    **facts,
                    initializer=dataclasses.asdict(solution),
                    solve_seconds=seconds,
                    strict_mouth_and_chest=mouth_chest_direct,
                    strict_mouth_views=int((strict_sources[0, :, 0] >= 0).sum()),
                    strict_sources=strict_sources,
                    observed=observed,
                    strict_float32_ray_distance_mm=distance,
                    admitted=bool(accepted),
                    selected_for_fitting=False,
                )
                rows.append(item)
                if accepted:
                    admitted.append((item, candidate, candidate_plan))
            admitted.sort(
                key=lambda a: (
                    -a[0]["strict_mouth_views"],
                    a[0]["ray_rms_mm"],
                    float(np.linalg.norm(a[1][0, 0] - scene[0, 0])),
                    tuple(a[2][0, :, 0]),
                )
            )
            selected = admitted[: config.proposals_per_fish]
            for item, _, _ in selected:
                item["selected_for_fitting"] = True
            selected_ids = [item["index_in_frame"] for item, _, _ in selected]
            frame_row = dict(
                frame_idx=frame,
                combinations=len(combos),
                admitted=len(admitted),
                returned=len(selected),
                selected_indices=selected_ids,
            )
            frames.append(frame_row)
            np.savez_compressed(
                output / f"f{frame}_bounded.npz",
                admitted_pose=np.stack([a[1] for a in admitted])
                if admitted
                else np.empty((0, 3, 3, 3)),
                admitted_plan=np.stack([a[2] for a in admitted])
                if admitted
                else np.empty((0, 3, 3, 3), np.int32),
                admitted_combination_indices=np.array(
                    [a[0]["index_in_frame"] for a in admitted], np.int64
                ),
                selected_for_fitting=np.arange(len(admitted)) < len(selected),
                predicted_pose=scene,
                predicted_plan=plan,
                initial_lengths=data["initial_lengths"],
            )
        checks.require(len(rows) == 202 and len(frames) == 33, "evaluation scope changed")
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[67663, 67696],
            combinations=202,
            frame_count=33,
            original_gate_counts=dict(Counter(r["original_first_gate"] for r in rows)),
            bounded_gate_counts=dict(Counter(r["first_gate"] for r in rows)),
            admitted=sum(f["admitted"] for f in frames),
            returned=sum(f["returned"] for f in frames),
            frames_with_candidates=sum(f["returned"] > 0 for f in frames),
            total_solver_seconds=sum(r["solve_seconds"] for r in rows),
            initializers_use_only_current_rays_and_box=True,
            old_winner_coordinates_used=False,
            thresholds_changed=False,
            original_recovery_config=dataclasses.asdict(config),
            biological_ID_truth=False,
            full_scene_fit_performed=False,
            continuous_tracking_performed=False,
            GPU_used=False,
            adopted=False,
            scope=(
                "Fixed post-hoc lost33 proposal assessment, not final fit/beam selection "
                "or held-out validation."
            ),
        )
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        (output / "all202.jsonl").write_text(
            "".join(json.dumps(prior.prior.finite_json(r), allow_nan=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        for source, digest in code.items():
            path = output / "code_snapshot" / Path(source).relative_to(REPO)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(source).read_bytes())
            checks.require(checks.sha(path) == digest, "source snapshot changed")
        checks.require(
            code == prior.generated.comparison.source_registry.local_code(), "source set changed"
        )
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "evaluation input changed")
            record.input(path, path, sha256=digest)
        for path in output.rglob("*"):
            if path.is_file() and not path.name.endswith("provenance.json"):
                record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
        record.param("all_bound_files_unchanged", True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
