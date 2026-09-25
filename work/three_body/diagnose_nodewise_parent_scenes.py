"""Compare body proposals using the actual saved 82720 parent, without fitting."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import run_nodewise_body_row_geometry as geometry
from body_row_proposals import BodyRowConfig, joint_body_row_scenes

proof, require, WORK = geometry.proof, geometry.require, geometry.WORK


def run():
    refs = {}

    def read(path, expected=None):
        digest = proof.sha(path)
        if expected is not None:
            require(digest == expected, "registered input changed")
        refs[str(path)] = digest
        return proof.read_json(path)

    geo_root = WORK / "nodewise_body_row_geometry_v1"
    seal = read(
        geo_root / "geometry.provenance.json",
        "c8eab931e843ce15c7646104b6c8346b33f7430edca81411872a9e9ab5c86cff",
    )
    require(seal["status"] == "ok", "successful geometry required")
    name = "cases/sparse_instance_rescore_82720.json"
    case = read(geo_root / name, seal["outputs"][name]["sha256"])
    before = read(
        WORK / "BODY_ROW_82720_FIT_DIAGNOSIS_V1.json",
        "adba218510ea7c5c12d904aac402a9673bbc551dc3ad6bd6e74d4ebc68c28831",
    )
    seed = read(
        WORK / "long_seed_registration_49800_v1" / "registration.json",
        "34da64c0c61a4c4e589b3e38a01627baff0e934b1d1a4325a280645e8a132b40",
    )
    root = WORK / "whole_row_body_ten_second_continuation_v1"
    tracking = read(
        root / "tracking.provenance.json",
        "562acdab952d46cd915ac15ffc16d0e26ed14b2aeba1096cd022f98ae04c9b96",
    )
    plan = read(root / "plan.json", tracking["outputs"]["plan.json"]["sha256"])
    predicted = np.asarray(before["parent_prediction"], float)
    lengths = np.linalg.norm(np.diff(np.asarray(seed["pose_xyz"], float), axis=1), axis=-1)
    require(
        proof.signature(lengths) == plan["tracker_context"]["initial_lengths"],
        "fixed initial lengths differ",
    )
    config = BodyRowConfig(**plan["tracker_context"]["body_row_policy"]["config"])
    pool = case["pool"]
    for body in pool:
        for key, dtype in (
            ("pose_xyz", float),
            ("ray_rms_mm", float),
            ("triangulated_nodes", bool),
            ("view_rows", np.int32),
            ("proposed_sources", np.int32),
        ):
            body[key] = np.asarray(body[key], dtype=dtype)
    count = case["pool_stats"]["legacy_candidate_count"]
    results = {}
    for name, bodies in (("legacy", pool[:count]), ("nodewise", pool)):
        scenes, stats = joint_body_row_scenes(
            bodies, predicted, lengths, case["lower"], case["upper"], config
        )
        if name == "legacy":
            require(len(scenes) == len(before["scenes"]), "old scene count differs")
            for new, old in zip(scenes, before["scenes"], strict=True):
                require(
                    np.array_equal(new["pose_xyz"], np.asarray(old["pre_fit_pose_xyz"]))
                    and np.array_equal(new["pool_indices"], old["pool_indices"])
                    and new["proposal_score"] == old["proposal_heuristic_score"],
                    "legacy parent scene did not reproduce exactly",
                )
        for index, scene in enumerate(scenes):
            scene["rank"] = index
            scene["would_fit_existing_headroom"] = index < before["counts"]["headroom"]
            scene["new_body_indices"] = [int(i) for i in scene["pool_indices"] if i >= count]
        results[name] = dict(scenes=scenes, stats=stats)
    refs[str(Path(__file__).resolve())] = proof.sha(Path(__file__))
    report = dict(
        schema="nodewise_82720_parent_scene_diagnosis_v1",
        status="ok",
        frame_idx=82720,
        parent_prediction=predicted,
        fixed_initial_lengths=lengths,
        config=asdict(config),
        legacy_scenes_reproduced_exactly=True,
        results=results,
        inputs=refs,
        image_fitting=False,
        new_trajectory_created=False,
        physical_ID_truth=False,
    )
    output = WORK / "NODEWISE_82720_PARENT_SCENE_DIAGNOSIS_V1.json"
    require(not output.exists(), "fresh diagnostic output required")
    output.write_bytes(proof.json_bytes(geometry.json_value(report)))
    for name, row in results.items():
        print(
            name,
            [
                (s["pool_indices"].tolist(), int(s["triangulated_nodes"].sum()))
                for s in row["scenes"]
            ],
        )


if __name__ == "__main__":
    run()
