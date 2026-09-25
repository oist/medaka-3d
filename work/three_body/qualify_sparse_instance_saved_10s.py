"""CPU sparse parity against all saved dense diagnostics for native76360..78019.

Reuse saved candidates, the exact consumed mask chunk and frozen CachedNodes.at.
No video reader, checkpoint, tracker step, GPU or dense EDT is executed. Only
fields actually retained by the dense companion are reference observations:
unsaved per-sample distances and endpoint/segment components cannot be compared.
Sparse-only timings do not establish a paired or end-to-end speed ratio.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import instance_body_score as dense
import numpy as np
import qualify_sparse_instance_body_score as fixed
import run_instance_rescore_10s as codec
import sparse_instance_body_score as sparse

proof, WORK, REPO, require = fixed.proof, codec.WORK, codec.REPO, fixed.require

# The frozen imports above establish the laboratory/source import roots.
import chunk_masks  # noqa: E402
import observations  # noqa: E402
import prepare_sequence  # noqa: E402

from medaka3d import calib  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

START, STOP = 76360, 78020
STAGE = "sparse_instance_saved_ten_second_qualification_v1"
PRODUCER = WORK / "instance_rescore_ten_second_continuation_v1"
PRODUCER_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
SPARSE_SHA = "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
CODEC_SHA = "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
FIXED_SHA = "b183ca5c537c6cbb85b7d6f5839972b11a5e7c26f7cc0f551724f8c56418ddc1"
FIXED_SEAL_SHA = "7d9c6eb462570497b006969bee4fc83fc66c888548c7a0545c7d62cccd87d3ce"
UNSAVED = [
    "worst_endpoint_costs",
    "segment_leakage_costs",
    "projected_samples",
    "distances_by_body_sample_instance",
]


def file_stat(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def member(seal, name, evidence, *, archive=False):
    path = fixed.source.relative(PRODUCER, name)
    row = seal["outputs"][name]
    require(
        path == fixed.source.local(row["path"]) and path.stat().st_size == row["size_bytes"],
        "saved member path/size differs: " + name,
    )
    evidence.add(path, row["sha256"])
    if archive:
        with np.load(path, allow_pickle=False) as z:
            require(len(z.files) == len(set(z.files)), "duplicate companion array")
            return {key: z[key] for key in z.files}
    return json.loads(path.read_text(encoding="utf8"))


def compact_view(result, cap):
    """Exactly the subset/normalization the frozen tracker saved, without EDT."""
    row = {
        k: result[k]
        for k in (
            "eligible",
            "reason",
            "neutral",
            "observation_evidence",
            "best_assignment",
            "assignment_costs",
            "optimal_assignment_mask",
            "body_instance_costs",
        )
    }
    raw = result["best_cost"]
    normalized = raw / 3
    invalid = ~np.isfinite(raw)
    clipped = np.clip(normalized, 0, cap)
    clipped[invalid] = cap
    if not result["eligible"]:
        clipped[:] = 0
    row.update(
        raw_uncapped_scene_cost=raw,
        uncapped_mean_body_cost=normalized,
        capped_view_contribution=clipped,
        nonfinite_score=invalid,
        candidate_projection_finite=result.get("candidate_projection_finite"),
        cap_applied=(normalized > cap) | invalid,
    )
    return row


def load_observations(seal, input_index, evidence, mask_records):
    """Only three mask NPZ payload reads; all thirteen sidecars retained and SHA-bound."""
    binding = input_index["original_binding"]["inputs"]
    chunks = input_index["prefix_chunks"]
    require(
        len(chunks) == 17 and [c["index"] for c in chunks] == list(range(17)),
        "prefix chunk order differs",
    )
    chosen = [c for c in chunks if c["start"] < STOP and c["stop"] > START]
    require(
        len(chosen) == 1 and chosen[0]["start"] == START and chosen[0]["stop"] == STOP,
        "exactly one consumed source chunk required",
    )
    chunk = chosen[0]
    root = fixed.source.local(Path(binding["input_stage_root"]) / chunk["attempt_path"])
    manifest_path = fixed.source.local(chunk["output_hashes"]["manifest"]["path"])
    require(manifest_path == root / "manifest.json", "consumed chunk root differs")
    manifest = fixed.source.read(manifest_path, evidence, chunk["manifest_sha256"])
    require(
        seal["inputs"].get(str(manifest_path)) == chunk["manifest_sha256"],
        "manifest input edge differs",
    )
    chunk_seal_path = root / "staging.provenance.json"
    require(
        seal["inputs"].get(str(chunk_seal_path)) == chunk["provenance_sha256"],
        "chunk seal edge differs",
    )
    chunk_seal = fixed.source.read(chunk_seal_path, evidence, chunk["provenance_sha256"])
    require(
        chunk_seal["status"] == "ok" and chunk_seal["stage"] == "prepare_independent_mask_chunk",
        "successful independent mask chunk required",
    )
    normalized_outputs = {
        key: dict(path=value["path"], sha256=value["sha256"])
        for key, value in chunk_seal["outputs"].items()
    }
    require(normalized_outputs == chunk["output_hashes"], "consumed sidecar/output set differs")
    require(
        manifest["binding"] == chunk_seal["params"]["binding"] == binding["detector_binding"],
        "detector input binding differs",
    )
    frames = np.arange(START, STOP, dtype=np.int64)
    require(
        manifest["frames"] == frames.tolist()
        and manifest["source_interval_half_open"] == [START, STOP]
        and manifest["fps"] == 166
        and manifest["image_shape_hw"] == [1200, 1200],
        "native manifest differs",
    )
    require(set(manifest["masks"]) == set(VIEWS), "mask camera membership differs")
    packed = {}
    for view in VIEWS:
        descriptor = manifest["masks"][view]
        path = fixed.source.relative(root, descriptor["path"])
        role = chunk["output_hashes"][view + "_masks"]
        require(
            path == fixed.source.local(role["path"])
            and descriptor["sha256"] == role["sha256"] == seal["inputs"].get(str(path)),
            "consumed mask payload input differs",
        )
        before = file_stat(path)
        payload = (
            path.read_bytes()
        )  # Hash and decode this captured byte string; no second file read.
        digest = hashlib.sha256(payload).hexdigest()
        require(
            digest == descriptor["sha256"] and file_stat(path) == before,
            "mask payload changed while captured",
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            require(
                len(z.files) == len(set(z.files))
                and set(z.files) == set(chunk_masks.SCHEMA_DTYPES),
                "packed mask sidecar membership differs",
            )
            arrays = {key: z[key] for key in z.files}
        require(
            all(arrays[k].dtype == dtype for k, dtype in chunk_masks.SCHEMA_DTYPES.items()),
            "packed mask sidecar dtype differs",
        )
        require(
            np.array_equal(arrays["frame_idx"], frames)
            and np.array_equal(arrays["image_shape_hw"], [1200, 1200])
            and arrays["view_available"].shape
            == arrays["frame_mask_sha256"].shape
            == (STOP - START,),
            "packed native frame extent differs",
        )
        for array in arrays.values():
            array.flags.writeable = False
        packed[view] = arrays
        mask_records[view] = dict(
            path=str(path),
            sha256=digest,
            file_stat=before,
            arrays={k: proof.signature(v) for k, v in arrays.items()},
            payload_reads=1,
            payload_decodes=1,
        )
    # The exact same cached untracked rows and at(view,native) selector used by
    # resumable_observations. Instantiating SourceSequenceReader would hash MP4s.
    seed = binding["seed"]
    upstream = seed["SL_and_calibration_sha256"]
    require(len(upstream) == 7, "exact three SLEAP and four calibration files required")
    for path, digest in upstream.items():
        require(seal["inputs"].get(path) == digest, "SL/calibration input edge differs")
        evidence.add(Path(path), digest)
    nodes = observations.CachedNodes(seed["input_root"], manifest["experiment"], frames)
    require(
        np.array_equal(nodes.frames, frames) and set(nodes.paths) == set(VIEWS),
        "cached native/view order differs",
    )
    require(
        {str(Path(p).resolve()) for p in nodes.paths.values()}
        == {p for p in upstream if Path(p).suffix == ".npz"},
        "SLEAP cache membership differs",
    )
    calibration_paths = [Path(p) for p in upstream if Path(p).name == "calibration_manifest.json"]
    require(len(calibration_paths) == 1, "one calibration manifest required")
    calibration_path = calibration_paths[0]
    config = json.loads(calibration_path.read_text(encoding="utf8"))
    required_models = {
        str(fixed.source.relative(calibration_path.parent, r["path"]))
        for r in config["model_files"]
    }
    require(
        len(required_models) == 3
        and required_models | {str(calibration_path)}
        == {p for p in upstream if Path(p).suffix == ".json"},
        "calibration model sidecars differ",
    )
    rig = calib.load_rig(calibration_path.parent)
    require(
        tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm",
        "calibration view order/units differ",
    )
    return nodes, rig, packed, mask_records, chunk


def execute(args):
    require(args.execute, "explicit reviewed CPU execution required")
    output = args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence = proof.Output(output), proof.Evidence()
    started, rows, mask_records, last = proof.utc(), [], {}, None
    try:
        evidence.add(PRODUCER / "tracking.provenance.json", PRODUCER_SHA)
        seal = json.loads((PRODUCER / "tracking.provenance.json").read_text(encoding="utf8"))
        require(
            seal["status"] == "ok" and seal["stage"] == codec.STAGE, "successful saved10s required"
        )
        for module, digest in (
            (sparse, SPARSE_SHA),
            (dense, sparse.REFERENCE_SOURCE_SHA256),
            (codec, CODEC_SHA),
            (fixed, FIXED_SHA),
        ):
            evidence.add(Path(module.__file__), digest)
        evidence.add(Path(fixed.source.__file__), fixed.SOURCE_SHA)
        fixed_path = WORK / fixed.STAGE / "qualification.provenance.json"
        evidence.add(fixed_path, FIXED_SEAL_SHA)
        fixed_seal = json.loads(fixed_path.read_text(encoding="utf8"))
        require(
            fixed_seal["status"] == "ok",
            "fixed-three qualification failed",
        )
        for module in (observations, prepare_sequence, chunk_masks, calib):
            path = Path(module.__file__).resolve()
            evidence.add(path, seal["inputs"][str(path)])
        evidence.add(Path(__file__))
        for module in (proof, codec.proof):
            path = Path(module.__file__).resolve()
            expected = seal["inputs"].get(str(path), fixed_seal["inputs"].get(str(path)))
            require(proof.valid_sha(expected), "frozen proof helper source pin missing")
            evidence.add(path, expected)
        for path in list(evidence.paths):
            if Path(path).suffix == ".py":
                out.write(
                    "code_snapshot/" + Path(path).relative_to(REPO).as_posix(),
                    Path(path).read_bytes(),
                )
        summary = member(seal, "summary.json", evidence)
        plan = member(seal, "plan.json", evidence)
        inputs = member(seal, "input_index.json", evidence)
        index = member(seal, "diagnostic_index.json", evidence)
        require(
            summary["process_interval"]
            == plan["process_interval"]
            == index["process_interval"]
            == [START, STOP]
            and summary["diagnostic_frames"] == STOP - START,
            "saved diagnostic extent differs",
        )
        require(
            plan["original_binding"] == inputs["original_binding"],
            "saved original input binding differs",
        )
        policy = plan["new_scoring_policy"]
        require(
            policy["weight"] == 3
            and policy["cap_per_view"] == 9
            and policy["bodies_per_view"] == 3
            and policy["fixed_camera_denominator"] == 3
            and policy["effective_from_native"] == START,
            "saved scoring normalization differs",
        )
        require(
            policy["observation_config"]
            == json.loads(json.dumps(dataclasses.asdict(dense.DEFAULT_CONFIG))),
            "saved observation config differs",
        )
        require(
            [r["frame_idx"] for r in index["rows"]] == list(range(START, STOP)),
            "diagnostic frame order differs",
        )
        expected = {
            f"diagnostics/f{f}.{ext}" for f in range(START, STOP) for ext in ("json", "npz")
        }
        require(
            {k for k in seal["outputs"] if k.startswith("diagnostics/")} == expected
            and {"diagnostics/" + p.name for p in (PRODUCER / "diagnostics").iterdir()} == expected,
            "diagnostic companion membership differs",
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                process_interval=[START, STOP],
                native_frames=STOP - START,
                producer_seal_sha256=PRODUCER_SHA,
                fixed_three_qualification_sha256=FIXED_SEAL_SHA,
                numerical_requirement=(
                    "every retained dense view field and normalized/adjusted "
                    "candidate cost BYTE EXACT"
                ),
                unsaved_dense_reference_fields=UNSAVED,
                dense_EDT_recomputed=False,
                mask_payload_loads=3,
                sequential_native_observations=True,
                raw_video_reads=0,
                checkpoint_payload_reads=0,
                GPU_used=False,
                sparse_only_timing=True,
                paired_speedup_established=False,
                adoption=False,
            ),
        )
        nodes, rig, packed, mask_records, chunk = load_observations(
            seal, inputs, evidence, mask_records
        )
        out.write(
            "observation_binding.json",
            dict(
                chunk=chunk,
                masks=mask_records,
                SL_and_calibration=inputs["original_binding"]["inputs"]["seed"][
                    "SL_and_calibration_sha256"
                ],
            ),
        )
        for t, native in enumerate(range(START, STOP)):
            last = dict(frame=native, view=None)
            saved = member(seal, f"diagnostics/f{native}.json", evidence)
            arrays = member(seal, f"diagnostics/f{native}.npz", evidence, archive=True)
            require(
                saved["archive"] == seal["outputs"][f"diagnostics/f{native}.npz"],
                "companion archive edge differs",
            )
            diagnostic = codec.restore_diagnostic(saved["tree"], arrays)
            require(
                diagnostic["frame_idx"] == native and diagnostic["policy"] == policy,
                "companion frame/policy differs",
            )
            bank = diagnostic["candidate_pose_xyz"]
            bank_signature = proof.signature(bank)
            require(
                len(bank) == index["rows"][t]["candidate_count"], "candidate pool extent differs"
            )
            frame_record = dict(
                frame=native,
                candidate_count=len(bank),
                candidate_pose_signature=bank_signature,
                views={},
            )
            contributions = []
            for view in VIEWS:
                last["view"] = view
                tick = time.perf_counter()
                data = packed[view]
                require(
                    int(data["frame_idx"][t]) == int(nodes.frames[t]) == native,
                    "mask/SL native order differs",
                )
                decoded = prepare_sequence.decode_frame_masks(data, t)
                mask_sha = prepare_sequence.array_sha256(decoded)
                require(
                    mask_sha == bytes(data["frame_mask_sha256"][t]).decode("ascii"),
                    "decoded mask SHA differs",
                )
                available = bool(data["view_available"][t])
                require(available or len(decoded) == 0, "unavailable mask has instances")
                points = nodes.at(view, native)
                points_signature = proof.signature(points)
                read_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                observation = sparse.prepare_sparse_instance_observation(
                    decoded if available else None, points
                )
                prep_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                result = sparse.score_sparse_instance_bodies(
                    bank, rig.camera(view).project, observation
                )
                score_seconds = time.perf_counter() - tick
                actual = compact_view(result, policy["cap_per_view"])
                try:
                    compared = fixed.exact_tree(
                        diagnostic["views"][view], actual, f"{native}/{view}"
                    )
                except Exception:
                    tree, mismatch_arrays = codec.split_diagnostic(
                        dict(expected=diagnostic["views"][view], actual=actual)
                    )
                    archive = out.write("mismatch.npz", mismatch_arrays)
                    out.write(
                        "mismatch.json", dict(frame=native, view=view, tree=tree, archive=archive)
                    )
                    raise
                require(
                    proof.signature(points) == points_signature
                    and prepare_sequence.array_sha256(decoded) == mask_sha,
                    "sparse score changed observation inputs",
                )
                contributions.append(actual["capped_view_contribution"])
                frame_record["views"][view] = dict(
                    eligible=observation.eligible,
                    reason=observation.reason,
                    available=available,
                    decoded_mask_sha256=mask_sha,
                    SL_points_signature=points_signature,
                    all_saved_dense_fields_BYTEEXACT=True,
                    compared_arrays=compared,
                    result_signatures={
                        k: proof.signature(v)
                        for k, v in actual.items()
                        if isinstance(v, np.ndarray)
                    },
                    timings=dict(
                        mask_decode_and_SL_seconds=read_seconds,
                        sparse_prepare_seconds=prep_seconds,
                        sparse_score_seconds=score_seconds,
                    ),
                )
            combined = np.stack(contributions, axis=1)
            sixth = policy["weight"] * combined.sum(axis=1) / policy["fixed_camera_denominator"]
            fixed.exact_tree(diagnostic["capped_contributions_by_view"], combined)
            fixed.exact_tree(diagnostic["sixth_instance_term"], sixth)
            fixed.exact_tree(
                diagnostic["adjusted_local_cost"], diagnostic["base_local_cost"] + sixth
            )
            fixed.exact_tree(
                diagnostic["adjusted_path_cost"],
                diagnostic["parent_path_cost"] + (diagnostic["base_local_cost"] + sixth),
            )
            require(proof.signature(bank) == bank_signature, "saved pose bank changed")
            rows.append(frame_record)
            if (t + 1) % 100 == 0 or native + 1 == STOP:
                print(
                    json.dumps(
                        dict(
                            event="sparse_saved_parity_progress",
                            through_frame=native + 1,
                            completed_frames=len(rows),
                        )
                    ),
                    flush=True,
                )
        for view, record in mask_records.items():
            require(
                file_stat(Path(record["path"])) == record["file_stat"],
                "mask file stat changed after capture",
            )
            require(
                {k: proof.signature(v) for k, v in packed[view].items()} == record["arrays"],
                "mask sidecar arrays changed",
            )
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "qualification initialized CUDA",
        )
        out.write("frame_comparisons.json", dict(rows=rows, process_interval=[START, STOP]))
        timing = {
            key: sum(row["views"][v]["timings"][key] for row in rows for v in VIEWS)
            for key in (
                "mask_decode_and_SL_seconds",
                "sparse_prepare_seconds",
                "sparse_score_seconds",
            )
        }
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_frames=len(rows),
                views_compared=3 * len(rows),
                candidate_rows=sum(r["candidate_count"] for r in rows),
                all_saved_dense_fields_BYTEEXACT=True,
                all_normalized_and_adjusted_costs_BYTEEXACT=True,
                unsaved_dense_reference_fields=UNSAVED,
                timings=timing,
                mask_payload_reads=3,
                mask_frame_decodes=3 * len(rows),
                mask_payload_final_rehash=False,
                captured_mask_payload_SHA_verified=True,
                mask_final_stat_and_all13_array_signature_guard=True,
                raw_video_reads=0,
                checkpoint_payload_reads=0,
                GPU_used=False,
                dense_EDT_recomputed=False,
                sparse_only_timing=True,
                paired_speedup_established=False,
                end_to_end_speedup_established=False,
                saved_workload_equivalence_only=True,
                continuous_rollout_equivalence=False,
                physical_ID_truth=False,
                adoption=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        status, error = "ok", None
    except BaseException as exception:
        status, error = "failed", repr(exception)
        out.write("failure.json", dict(error=error, last=last, completed_frames=len(rows)))
        if rows:
            out.write("completed_frame_comparisons.json", dict(rows=rows))
        raise
    finally:
        inputs = dict(evidence.paths)
        inputs.update({r["path"]: r["sha256"] for r in mask_records.values()})
        (output / "qualification.provenance.json").write_bytes(
            proof.json_bytes(
                dict(
                    stage=STAGE,
                    status=locals().get("status", "failed"),
                    error=locals().get("error"),
                    started_at=started,
                    finished_at=proof.utc(),
                    inputs=inputs,
                    outputs=out.records,
                    completed_frames=len(rows),
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
