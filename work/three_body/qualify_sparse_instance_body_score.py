"""Paired CPU distance/scoring equivalence on nine sealed recording mask frames.

No fitting, tracking, raw-video decode, GPU or checkpoint access. This qualifies
only the fixed three-frame workload; timings are one paired measurement/view.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import instance_body_score as dense
import numpy as np
import score_saved_capacity_island as source
import sparse_instance_body_score as sparse

proof, WORK, require = source.proof, source.WORK, source.require
STAGE = "sparse_instance_body_score_fixed_three_qualification_v1"
SOURCE_SHA = "a64107b63347565135daaaa070a2d0abfde7a72d4024343ebc56d0a72a27821f"


def exact_tree(first, second, path="result"):
    """Strict array bytes and nonarray structure; no tolerance or inf subtraction."""
    if isinstance(first, np.ndarray):
        require(isinstance(second, np.ndarray), "array type differs: " + path)
        require(proof.signature(first) == proof.signature(second), "array bytes differ: " + path)
        return 1
    require(type(first) is type(second), "type differs: " + path)
    if isinstance(first, dict):
        require(first.keys() == second.keys(), "keys differ: " + path)
        return sum(exact_tree(first[k], second[k], path + "/" + k) for k in first)
    if isinstance(first, (list, tuple)):
        require(len(first) == len(second), "length differs: " + path)
        return sum(
            exact_tree(a, b, f"{path}/{i}")
            for i, (a, b) in enumerate(zip(first, second, strict=True))
        )
    require(first == second, "value differs: " + path)
    return 0


def execute(args):
    require(
        args.execute and proof.valid_sha(args.sparse_sha),
        "execution and reviewed sparse SHA required",
    )
    output = args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence = proof.Output(output), proof.Evidence()
    started, records = proof.utc(), []
    try:
        for module, digest in (
            (dense, source.CORE_SHA),
            (source, SOURCE_SHA),
            (sparse, args.sparse_sha),
        ):
            evidence.add(Path(module.__file__), digest)
        evidence.add(Path(__file__))
        evidence.add(Path(proof.__file__))
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(source.FRAMES),
                views=list(source.VIEWS),
                arms=list(source.ARMS),
                dense_source_sha256=source.CORE_SHA,
                sparse_source_sha256=args.sparse_sha,
                source_diagnosis_sha256=source.SOURCE_SHA,
                frozen_config=dataclasses.asdict(dense.DEFAULT_CONFIG),
                numerical_requirement=(
                    "all returned array bytes, gate evidence and result fields exact"
                ),
                retuning=False,
                paired_timing_repetitions_per_view=1,
                timing_order=["dense", "sparse"],
                timing_scope="Single paired CPU measurements, not end-to-end tracking speed",
                masks_to_decode=9,
                raw_video_decodes=0,
                checkpoint_payload_decodes=0,
                GPU_used=False,
                tracking_modified=False,
                adoption=False,
            ),
        )
        data, selected, rig, packed = source.load(evidence)
        for t, frame in enumerate(source.FRAMES):
            bank = np.stack([selected[arm]["pose_xyz"][t] for arm in source.ARMS])
            bank_before = proof.signature(bank)
            for view in source.VIEWS:
                observation = data["records"][t]["observations"][view]
                index = frame - data["source_chunk"]["start"]
                decoded = source.prepare_sequence.decode_frame_masks(packed[view], index)
                mask_sha = source.prepare_sequence.array_sha256(decoded)
                require(
                    mask_sha
                    == observation["mask_sha256"]
                    == bytes(packed[view]["frame_mask_sha256"][index]).decode()
                    and len(decoded) == observation["mask_count"]
                    and bool(packed[view]["view_available"][index])
                    == observation["mask_available"],
                    "sealed mask frame differs",
                )
                masks = decoded if observation["mask_available"] else None
                points = np.asarray(observation["SLEAP_points"], dtype=np.float32)
                points_before = proof.signature(points)
                require(
                    np.where(np.isfinite(points), points.astype(np.float64), None).tolist()
                    == observation["SLEAP_points"],
                    "SLEAP float32 roundtrip differs",
                )
                observations, results, timing = {}, {}, {}
                for name, prepare, score in (
                    ("dense", dense.prepare_instance_observation, dense.score_instance_bodies),
                    (
                        "sparse",
                        sparse.prepare_sparse_instance_observation,
                        sparse.score_sparse_instance_bodies,
                    ),
                ):
                    tick = time.perf_counter()
                    prepared = prepare(masks, points, dense.DEFAULT_CONFIG)
                    prep_seconds = time.perf_counter() - tick
                    tick = time.perf_counter()
                    result = score(bank, rig.camera(view).project, prepared)
                    timing[name] = dict(
                        prepare_seconds=prep_seconds, score_seconds=time.perf_counter() - tick
                    )
                    observations[name] = dict(
                        eligible=prepared.eligible,
                        reason=prepared.reason,
                        image_shape_hw=prepared.image_shape_hw,
                        sleap_row_by_instance=prepared.sleap_row_by_instance,
                        evidence=prepared.evidence,
                        config=dataclasses.asdict(prepared.config),
                    )
                    results[name] = result
                exact_tree(observations["dense"], observations["sparse"], "observation")
                arrays_count = exact_tree(results["dense"], results["sparse"])
                require(proof.signature(bank) == bank_before, "candidate pose bank changed")
                require(proof.signature(points) == points_before, "SLEAP points changed")
                require(
                    source.prepare_sequence.array_sha256(decoded) == mask_sha, "mask array changed"
                )
                arrays = {
                    key: value
                    for key, value in results["sparse"].items()
                    if isinstance(value, np.ndarray)
                }
                out.write(f"arrays/f{frame}_{view}.npz", arrays)
                record = dict(
                    frame=frame,
                    view=view,
                    eligible=results["sparse"]["eligible"],
                    reason=results["sparse"]["reason"],
                    all_result_arrays_BYTEEXACT=True,
                    result_array_count=arrays_count,
                    all_gate_evidence_exact=True,
                    all_result_metadata_exact=True,
                    array_signatures={key: proof.signature(value) for key, value in arrays.items()},
                    best_cost=results["sparse"]["best_cost"].tolist(),
                    timings=timing,
                )
                records.append(record)
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in ("frame", "view", "eligible", "all_result_arrays_BYTEEXACT")
                        }
                    ),
                    flush=True,
                )
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "qualification initialized CUDA",
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(source.FRAMES),
                records=records,
                all_nine_views_exact=True,
                mask_frames_decoded=9,
                raw_video_decodes=0,
                checkpoint_payload_decodes=0,
                GPU_used=False,
                tracking_modified=False,
                execution_equivalence_qualified_only_on_fixed_workload=True,
                continuous_rollout_equivalence=False,
                end_to_end_speedup_established=False,
                physical_ID_truth=False,
                adoption=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            completed_views=records,
            error=repr(error),
        )
        (output / "qualification.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "qualification.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sparse-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
