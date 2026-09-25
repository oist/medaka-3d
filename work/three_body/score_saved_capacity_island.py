"""Fixed three-frame CPU instance-body diagnostic; no raw-video access or fitting."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import compare_temporal_point64_v2 as proof
import instance_body_score as core
import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import prepare_sequence  # noqa: E402

from medaka3d import calib  # noqa: E402

ROOT = WORK / "fast_tail_capacity_island_raw_diagnosis_v1"
SOURCE_SHA = "70ff3c73c2bf104e250ad47cb3e4ae864328145d0a95219253400f6162c6ad20"
CORE_SHA = "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"
STAGE = "saved_capacity_island_instance_body_score_v1"
FRAMES, ARMS = (77709, 77723, 77880), ("original", "fast_tail")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
require = proof.require


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "input escaped workspace")
    return path


def relative(root, name):
    require(
        isinstance(name, str)
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and ":" not in name,
        "unsafe relative path",
    )
    path = local(root / name)
    require(path.is_relative_to(root), "relative input escaped root")
    return path


def read(path, evidence, expected):
    path = local(path)
    evidence.add(path, expected)
    return json.loads(path.read_text(encoding="utf8"))


def member(record, name, evidence):
    row = record["outputs"][name]
    path = relative(ROOT, name)
    require(
        local(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "sealed diagnostic member differs",
    )
    return read(path, evidence, row["sha256"])


def bound_input(record, path, digest, evidence):
    path = local(path)
    require(record["inputs"].get(str(path)) == digest, "original diagnostic input edge differs")
    evidence.add(path, digest)
    return path


def load(evidence):
    seal = read(ROOT / "diagnosis.provenance.json", evidence, SOURCE_SHA)
    require(
        seal["stage"] == "fast_tail_capacity_island_raw_diagnosis_v1" and seal["status"] == "ok",
        "successful source diagnosis required",
    )
    data = member(seal, "source_and_frame_evidence.json", evidence)
    require(
        data["frames"] == list(FRAMES)
        and data["horizon_stop"] == 86320
        and data["native_BGR_decodes"] == 9
        and data["decoded_mask_frames"] == 9
        and data["candidate_fields_unchanged"] is True
        and data["independent_masks_bound_to_successful_Original_consumed_chunk"] is True,
        "fixed source diagnostic scope differs",
    )
    require([r["frame"] for r in data["records"]] == list(FRAMES), "source frame records differ")
    selected = {}
    for arm in ARMS:
        reference = data["candidate_references"][arm]
        path = bound_input(seal, reference["path"], reference["sha256"], evidence)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("frame_idx", "pose_xyz", "observed")}
        require(
            all(proof.signature(v) == reference["arrays"][k] for k, v in arrays.items()),
            "candidate field signature differs",
        )
        indices = np.searchsorted(arrays["frame_idx"], FRAMES)
        require(
            np.all(indices < len(arrays["frame_idx"]))
            and np.array_equal(arrays["frame_idx"][indices], FRAMES),
            "exact candidate frames missing",
        )
        selected[arm] = {key: value[indices].copy() for key, value in arrays.items()}
        for t, row in enumerate(data["records"]):
            require(
                selected[arm]["observed"][t].tolist() == row["methods"][arm]["observed"],
                "selected observed flags differ from diagnostic",
            )
    manifests = [Path(p) for p in seal["inputs"] if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one source calibration manifest required")
    calibration_path = manifests[0]
    config = read(calibration_path, evidence, seal["inputs"][str(calibration_path)])
    require(len(config["model_files"]) == 3, "three calibrated cameras required")
    for row in config["model_files"]:
        path = relative(calibration_path.parent, row["path"])
        bound_input(seal, path, seal["inputs"][str(path)], evidence)
    rig = calib.load_rig(calibration_path.parent)
    require(
        tuple(rig.views) == VIEWS and rig.world_units == "mm", "calibrated view order/units differ"
    )
    chunk = data["source_chunk"]
    manifest_path = local(chunk["output_hashes"]["manifest"]["path"])
    bound_input(seal, manifest_path, chunk["manifest_sha256"], evidence)
    manifest = read(manifest_path, evidence, chunk["manifest_sha256"])
    require(
        manifest["frames"] == list(range(chunk["start"], chunk["stop"]))
        and chunk["start"] <= FRAMES[0] <= FRAMES[-1] < chunk["stop"],
        "native source chunk differs",
    )
    packed = {}
    for view in VIEWS:
        descriptor = manifest["masks"][view]
        path = relative(manifest_path.parent, descriptor["path"])
        role = chunk["output_hashes"][view + "_masks"]
        require(
            local(role["path"]) == path and role["sha256"] == descriptor["sha256"],
            "consumed mask descriptor differs",
        )
        bound_input(seal, path, descriptor["sha256"], evidence)
        with np.load(path, allow_pickle=False) as archive:
            packed[view] = {key: archive[key] for key in archive.files}
        require(
            np.array_equal(packed[view]["frame_idx"], manifest["frames"]),
            "mask native index differs",
        )
    for module in (prepare_sequence, calib):
        path = Path(module.__file__).resolve()
        bound_input(seal, path, seal["inputs"][str(path)], evidence)
    return data, selected, rig, packed


def raw_distance_diagnostics(bank, camera, masks, config):
    samples = camera.project(core.sample_body_xyz(bank, config.interior_fractions))
    if masks is None or not len(masks) or not np.all(masks.any(axis=(1, 2))):
        return dict(raw_projected_samples=samples)
    fields = np.stack([core.distance_transform_edt(~mask) for mask in masks])
    return dict(
        raw_projected_samples=samples,
        raw_instance_distances=core.sample_distance_fields(fields, samples),
    )


def run(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh diagnostic output required")
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        evidence.add(Path(core.__file__), CORE_SHA)
        evidence.add(Path(__file__))
        evidence.add(Path(proof.__file__))
        config = core.DEFAULT_CONFIG
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_diagnosis_sha256=SOURCE_SHA,
                core_sha256=CORE_SHA,
                frames=list(FRAMES),
                views=list(VIEWS),
                arms=list(ARMS),
                frozen_config=dataclasses.asdict(config),
                thresholds_retuned_on_these_frames=False,
                candidate_pose_used_for_eligibility=False,
                all_nodes_including_latent=True,
                expectations_to_test={
                    "77709_77723_top": "eligible and mixed FastTail cost higher",
                    "77880_top": "neutral control; report unexpected eligibility",
                },
                raw_video_decodes=0,
                raw_video_files_rehashed=0,
                no_tracking_or_fit=True,
            ),
        )
        data, rows, rig, packed = load(evidence)
        records = []
        for t, frame in enumerate(FRAMES):
            bank = np.stack([rows[arm]["pose_xyz"][t] for arm in ARMS])
            before = proof.signature(bank)
            for view in VIEWS:
                source = data["records"][t]["observations"][view]
                index = frame - data["source_chunk"]["start"]
                decoded = prepare_sequence.decode_frame_masks(packed[view], index)
                require(
                    prepare_sequence.array_sha256(decoded)
                    == source["mask_sha256"]
                    == bytes(packed[view]["frame_mask_sha256"][index]).decode()
                    and len(decoded) == source["mask_count"]
                    and bool(packed[view]["view_available"][index]) == source["mask_available"],
                    "one of nine sealed mask frames differs",
                )
                masks = decoded if source["mask_available"] else None
                points = np.asarray(source["SLEAP_points"], dtype=np.float32)
                roundtrip = np.where(np.isfinite(points), points.astype(np.float64), None).tolist()
                require(roundtrip == source["SLEAP_points"], "SL float32 JSON roundtrip differs")
                observation = core.prepare_instance_observation(masks, points, config)
                result = core.score_instance_bodies(bank, rig.camera(view).project, observation)
                arrays = {
                    key: value for key, value in result.items() if isinstance(value, np.ndarray)
                }
                # Unavailable controls keep neutral scores; raw distances are diagnostic only.
                if not observation.eligible:
                    arrays.update(raw_distance_diagnostics(bank, rig.camera(view), masks, config))
                archive = out.write(f"arrays/f{frame}_{view}.npz", arrays)
                with np.load(archive["path"], allow_pickle=False) as saved:
                    require(
                        set(saved.files) == set(arrays)
                        and all(
                            proof.signature(saved[k]) == proof.signature(v)
                            for k, v in arrays.items()
                        ),
                        "scoring array roundtrip differs",
                    )
                record = dict(
                    frame=frame,
                    view=view,
                    eligible=observation.eligible,
                    reason=observation.reason,
                    neutral=result["neutral"],
                    observation_evidence=observation.evidence,
                    best_cost_by_arm=dict(zip(ARMS, result["best_cost"].tolist(), strict=True)),
                    all_six_costs=None
                    if result["assignment_costs"] is None
                    else result["assignment_costs"].tolist(),
                    best_assignment=None
                    if result["best_assignment"] is None
                    else result["best_assignment"].tolist(),
                    arrays=archive,
                    array_signatures={k: proof.signature(v) for k, v in arrays.items()},
                    unavailable_raw_distances_are_diagnostic_only=not observation.eligible,
                )
                records.append(record)
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in ("frame", "view", "eligible", "reason", "best_cost_by_arm")
                        }
                    ),
                    flush=True,
                )
            require(proof.signature(bank) == before, "pose bank changed")
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                frozen_config=dataclasses.asdict(config),
                mask_frames_decoded=9,
                raw_video_decodes=0,
                raw_video_files_rehashed=0,
                checkpoint_or_DAG_payloads_read=0,
                GPU_used=False,
                tracking_modified=False,
                human_truth_created=False,
                all_three_ID_truth=False,
                permanent_mask_IDs_assigned=False,
                full_recording_or_ID_improvement_established=False,
                scope=(
                    "Fixed three-frame model diagnosis. "
                    "Eligibility failures are results; no retuning."
                ),
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
            error=repr(error),
        )
        (output / "scoring.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "scoring.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
