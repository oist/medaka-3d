"""Pure CPU node proposals on four saved scenes; no fitter, GPU or ID inference."""

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from node_geometry import RayObservations, assignment_plans, supported_nodes  # noqa: E402
from node_recovery import NodeRecoveryConfig, node_recovery_proposals  # noqa: E402
from observations import CachedNodes  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

RUN = ROOT / "work/three_body/v1_continuous_one_minute_detectron2"
OUT = ROOT / "work/three_body/node_recovery_geometry_v2"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    OUT.mkdir(exist_ok=False)
    seal = json.loads((RUN / "experiment.provenance.json").read_text())
    candidate = RUN / "candidate.npz"
    assert seal["status"] == "ok"
    assert sha256(candidate) == seal["outputs"]["candidate"]["sha256"]
    with np.load(candidate, allow_pickle=False) as arrays:
        frames, poses = arrays["frame_idx"], arrays["pose_xyz"]
    initial_path = RUN / "initial_pose_xyz.npy"
    initial = np.load(initial_path, allow_pickle=False)
    lengths = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    rig = load_rig(RUN / "calibration_snapshot")
    nodes = CachedNodes(ROOT / "work/generative_tracking_inputs", "20251225_K_1_2_FCMC", frames)
    paths = [
        Path(__file__),
        candidate,
        initial_path,
        RUN / "experiment.provenance.json",
        *(RUN / "calibration_snapshot").glob("*.json"),
        *nodes.paths.values(),
        LAB / "node_recovery.py",
        LAB / "node_geometry.py",
        LAB / "observations.py",
        ROOT / "tests/test_three_body_node_recovery.py",
    ]
    hashes = {str(p): sha256(p) for p in paths}
    cfg = NodeRecoveryConfig()
    with provenance("current_frame_mouth_recovery_proposal_check", OUT, name="geometry") as record:
        record.param("config", dataclasses.asdict(cfg))
        record.param("cpu_only_no_tracking_no_fitter", True)
        record.param("base_plan", "fresh assignment_plans at the saved scene, not an old row index")
        record.param("scope", "geometry eligibility only, no identity verdict or recovery claim")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        results = []
        for frame, target_fish, expected in (
            (80261, 1, True),
            (80431, 2, True),
            (80344, 2, False),
            (80370, 2, False),
        ):
            pose = poses[frame - int(frames[0])].copy()
            rays = RayObservations.from_points(rig, {v: nodes.at(v, frame) for v in rig.views})
            plan = assignment_plans(pose, rays, keep=1)[0][1]
            original_pose, original_plan = pose.copy(), plan.copy()
            started = time.perf_counter()
            output = node_recovery_proposals(pose, plan, rays, rig, lengths, cfg)
            elapsed = time.perf_counter() - started
            assert np.array_equal(pose, original_pose) and np.array_equal(plan, original_plan)
            selected = [p for p in output.proposals if p.fish == target_fish]
            assert bool(selected) is expected
            for proposal in output.proposals:
                other_pose = np.ones((3, 3), bool)
                other_pose[proposal.fish, 0] = False
                assert np.array_equal(proposal.pose_xyz[other_pose], pose[other_pose])
                other_plan = np.ones(plan.shape, bool)
                other_plan[proposal.fish, :, 0] = False
                assert np.array_equal(proposal.plan[other_plan], plan[other_plan])
                _, direct, _ = supported_nodes(proposal.pose_xyz, proposal.plan, rays, rig)
                assert direct[proposal.fish, :2].all()
            results.append(
                dict(
                    frame=frame,
                    target_fish=target_fish,
                    expected_target_eligibility=expected,
                    base_scene_xyz=pose,
                    fresh_base_plan=plan,
                    diagnostics=output.diagnostics,
                    proposals=[dataclasses.asdict(p) for p in output.proposals],
                    cpu_seconds=elapsed,
                )
            )
        assert {path: sha256(path) for path in hashes} == hashes
        output = dict(
            status="passed",
            config=dataclasses.asdict(cfg),
            cases=results,
            inputs=hashes,
            candidate_node_displacement_is_not_an_accepted_optimizer_step=True,
            limitations=[
                "Saved scene is historical; online predicted states may differ.",
                "Current-frame fresh plans may differ from unsaved original pre-fit plans.",
                "Neither the existing 14-step fitter nor a temporal tracker was run.",
                "Geometry eligibility does not certify a fish identity or eliminate ghosts.",
            ],
        )
        path = OUT / "summary.json"
        path.write_text(json.dumps(clean(output), indent=2, allow_nan=False))
        record.output("summary", path, sha256=sha256(path))
        for source in (Path(__file__), LAB / "node_recovery.py"):
            target = OUT / source.name
            target.write_bytes(source.read_bytes())
            record.output(source.name, target, sha256=sha256(target))
        print(
            json.dumps(
                [
                    dict(
                        frame=r["frame"],
                        target=r["target_fish"],
                        target_proposals=sum(p["fish"] == r["target_fish"] for p in r["proposals"]),
                        elapsed_ms=1000 * r["cpu_seconds"],
                    )
                    for r in results
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
