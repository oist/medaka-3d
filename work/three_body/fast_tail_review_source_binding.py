"""Bind an existing raw-only review pack to the successful FastTail five minutes.

The actual custom tracking seal is mandatory; an unsealed input index or a
completed intermediate milestone cannot stand in for process completion. No
candidate coordinates, masks, raw image decode, tracking or correction occur.
The caller validates the human document and final candidate, decodes the nine
native anchors, performs matching, and executes its final evidence guard.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import map_long_review_to_final as legacy

review, raw, native = legacy.review, legacy.raw, legacy.native
require, VIEWS = raw.require, legacy.VIEWS
REPO = Path(__file__).resolve().parents[2]
FIRST, START, RESUME, STOP, CHUNK = 49800, 66400, 76360, 116200, 1660
STAGE = "fast_tail_five_minute_continuation_v1"
RUNNER = REPO / "work/three_body/continue_fast_tail_five_minutes.py"
RUNNER_SHA = "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"


def sha(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "invalid SHA256",
    )
    return value


def read(path, evidence, expected=None):
    path = raw.local(path)
    evidence.add(path, None if expected is None else sha(expected))
    return review.document(path.read_bytes(), str(path))


def custom_member(record, root, name, evidence):
    path = native.checked_relative(root, name)
    row = record["outputs"].get(name)
    require(
        row is not None
        and raw.local(row["path"]) == path
        and type(row.get("size_bytes")) is int
        and path.stat().st_size == row["size_bytes"],
        "custom final member path/size differs: " + name,
    )
    evidence.add(path, sha(row["sha256"]))
    return path


def legacy_input(record, path, digest):
    row = record["inputs"].get(str(path))
    require(
        row is not None and raw.local(row["path"]) == path and row["sha256"] == digest,
        "legacy input binding differs: " + str(path),
    )


def legacy_output(record, role, path, evidence):
    row = record["outputs"].get(role)
    require(row is not None and raw.local(row["path"]) == path, "legacy output path differs")
    evidence.add(path, sha(row["sha256"]))
    return row["sha256"]


def final_inputs(final_run, final_tracking_sha, evidence):
    """Read only sealed metadata; candidate payload verification belongs to caller."""
    root = raw.local(final_run)
    record = read(root / "tracking.provenance.json", evidence, sha(final_tracking_sha))
    require(
        record.get("stage") == STAGE and record.get("status") == "ok",
        "successful final FastTail five-minute seal required",
    )
    require(record["inputs"].get(str(RUNNER)) == RUNNER_SHA, "final runner source differs")
    evidence.add(RUNNER, RUNNER_SHA)
    summary = review.document(
        custom_member(record, root, "summary.json", evidence).read_bytes(), "final summary"
    )
    expected = dict(
        schema=STAGE,
        status="ok",
        scientific_interval=[START, STOP],
        process_interval=[RESUME, STOP],
        native_frames=49800,
        process_frames=39840,
        initialize_count=1,
        restore_count=5,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        physical_ID_truth=False,
    )
    require(
        all(type(summary.get(k)) is type(v) and summary[k] == v for k, v in expected.items()),
        "final five-minute lifecycle/extent differs",
    )
    checkpoints = summary.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 24
        and [r["previous_frame"] + 1 for r in checkpoints]
        == list(range(RESUME + CHUNK, STOP + CHUNK, CHUNK)),
        "final checkpoint sequence differs",
    )
    index = review.document(
        custom_member(record, root, "input_index.json", evidence).read_bytes(), "final input index"
    )
    plan = review.document(custom_member(record, root, "plan.json", evidence).read_bytes(), "plan")
    binding, consumed = index["original_binding"], index["prefix_chunks"]
    require(
        plan["schema"] == STAGE
        and plan["original_binding"] == binding
        and plan["scientific_interval"] == [START, STOP]
        and plan["process_interval"] == [RESUME, STOP]
        and plan["source_sha256"].get(str(RUNNER)) == RUNNER_SHA,
        "final plan/input/source binding differs",
    )
    require(
        isinstance(consumed, list) and len(consumed) == 40, "40 consumed prefix chunks required"
    )
    for i, row in enumerate(consumed):
        require(
            all(
                type(row.get(k)) is int and row[k] == value
                for k, value in (
                    ("index", i),
                    ("start", FIRST + i * CHUNK),
                    ("stop", FIRST + (i + 1) * CHUNK),
                )
            ),
            "final consumed prefix native order differs",
        )
    candidate = record["outputs"].get("candidate.npz")
    require(
        candidate is not None
        and raw.local(candidate["path"]) == root / "candidate.npz"
        and type(candidate.get("size_bytes")) is int
        and (root / "candidate.npz").stat().st_size == candidate["size_bytes"],
        "final candidate descriptor differs",
    )
    sha(candidate["sha256"])
    milestone_name = "milestones/through_116200/summary.json"
    milestone = review.document(
        custom_member(record, root, milestone_name, evidence).read_bytes(),
        "sealed five-minute milestone",
    )
    require(
        milestone.get("schema") == STAGE + "_milestone"
        and milestone.get("status") == "ok"
        and milestone.get("horizon_stop") == STOP
        and milestone.get("scientific_interval") == [START, STOP]
        and milestone.get("original_binding") == binding
        and milestone["source_sha256"].get(str(RUNNER)) == RUNNER_SHA,
        "sealed final milestone binding differs",
    )
    milestone_path = native.checked_relative(root, "milestones/through_116200/candidate.npz")
    milestone_row = record["outputs"].get("milestones/through_116200/candidate.npz")
    require(
        milestone_row is not None
        and raw.local(milestone_row["path"]) == milestone_path
        and type(milestone_row.get("size_bytes")) is int
        and milestone_path.stat().st_size == milestone_row["size_bytes"]
        and milestone["candidate"] == milestone_row,
        "sealed milestone candidate descriptor differs",
    )
    sha(milestone_row["sha256"])
    return dict(
        root=root,
        record=record,
        binding=binding,
        consumed=consumed,
        candidate_sha256=candidate["sha256"],
        milestone_candidate_sha256=milestone_row["sha256"],
        milestone_summary_sha256=record["outputs"][milestone_name]["sha256"],
    )


def verify_shared(fixed, record, evidence):
    """Require the actual registered three SL and four camera files, without decoding SL."""
    shared = fixed["seed"]["SL_and_calibration_sha256"]
    require(isinstance(shared, dict) and len(shared) == 7, "three SL plus four calibration inputs")
    paths = {str(raw.local(p)): sha(d) for p, d in shared.items()}
    require(paths == shared, "shared paths must be canonical")
    manifests = [Path(p) for p in paths if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one calibration manifest required")
    calibration_manifest = manifests[0]
    config = read(calibration_manifest, evidence, paths[str(calibration_manifest)])
    models = [
        native.checked_relative(calibration_manifest.parent, row["path"])
        for row in config["model_files"]
    ]
    sl = {
        str(p)
        for p in map(Path, paths)
        if any(p.name == f"{legacy.baseline.EXPERIMENT}__{view}.npz" for view in VIEWS)
    }
    require(
        len(models) == 3
        and len(set(models)) == 3
        and len(sl) == 3
        and set(paths) == sl | {str(calibration_manifest), *map(str, models)},
        "SL/calibration file membership differs",
    )
    for path, digest in paths.items():
        require(record["inputs"].get(path) == digest, "final shared input edge differs")
        evidence.add(path, digest)
    protocol = raw.local(fixed["protocol_path"])
    digest = sha(fixed["protocol_sha256"])
    protocol_input = fixed["detector_inputs"]["protocol"]
    require(
        raw.local(protocol_input["path"]) == protocol
        and protocol_input["sha256"] == digest
        and fixed["detector_binding"]["protocol_sha256"] == digest
        and fixed["detector_binding"]["input_sha256"]["protocol"] == digest,
        "detector and final source protocol binding differs",
    )
    require(record["inputs"].get(str(protocol)) == digest, "final protocol input edge differs")
    evidence.add(protocol, digest)
    return copy.deepcopy(shared)


def verify_source_binding(final_run, final_tracking_sha, parsed, pack_summary, pack, evidence):
    """Prove same original source for one genuine long-review pack and final five minutes."""
    final = final_inputs(final_run, final_tracking_sha, evidence)
    root, record, binding = final["root"], final["record"], final["binding"]
    fixed, consumed = binding["inputs"], final["consumed"]
    pack = raw.local(pack)
    pack_record = read(pack / "review.provenance.json", evidence)
    require(
        pack_record.get("stage") == "long_original_video_raw_review"
        and pack_record.get("status") == "ok"
        and pack_record["params"].get("all_input_source_output_sha_unchanged") is True,
        "successful original-video raw review pack required",
    )
    legacy_output(pack_record, "summary.json", pack / "summary.json", evidence)
    actual = review.document((pack / "summary.json").read_bytes(), "pack summary")
    require(
        actual == pack_summary
        and actual.get("schema") == "long_raw_review_pack_v1"
        and actual.get("technical_completion") is True
        and actual.get("model_outputs_shown") is False
        and actual.get("human_truth_created") is False
        and actual.get("original_raw_storage") == "original_source_video",
        "completed raw-only pack summary differs",
    )
    declared = pack_summary["consumed_record"]
    index = declared["index"]
    require(
        type(index) is int and 0 <= index < 40 and consumed[index] == declared,
        "pack and final consumed chunk proofs differ",
    )
    clip = review.range_pair(pack_summary["review_clip_range"], "review clip")
    require(
        parsed.review["review_clip_range"] == clip
        and declared["start"] <= clip[0] < clip[1] <= declared["stop"]
        and len(parsed.anchor_frames) == 3
        and all(clip[0] <= int(f) < clip[1] for f in parsed.anchor_frames),
        "human native anchors/clip differ from source chunk",
    )
    attempt = native.checked_relative(
        raw.local(fixed["input_stage_root"]), declared["attempt_path"]
    )
    manifest_path, seal_path = attempt / "manifest.json", attempt / "staging.provenance.json"
    require(
        raw.local(pack_summary["source_manifest"]) == manifest_path,
        "pack source manifest path differs",
    )
    manifest = read(manifest_path, evidence, declared["manifest_sha256"])
    source = read(seal_path, evidence, declared["provenance_sha256"])
    legacy_output(pack_record, "source_manifest.json", pack / "source_manifest.json", evidence)
    require(
        parsed.manifest_bytes
        == manifest_path.read_bytes()
        == (pack / "source_manifest.json").read_bytes(),
        "original manifest bytes differ",
    )
    require(
        source.get("stage") == "prepare_independent_mask_chunk" and source.get("status") == "ok",
        "successful consumed detector source required",
    )
    require(
        source["params"]["binding"] == fixed["detector_binding"], "final detector binding differs"
    )
    raw.validate_source(manifest, declared, fixed["detector_binding"])
    require(manifest["protocol_sha256"] == fixed["protocol_sha256"], "final raw protocol differs")
    outputs = {
        key: dict(path=str(raw.local(row["path"])), sha256=row["sha256"])
        for key, row in source["outputs"].items()
    }
    require(outputs == declared["output_hashes"], "consumed detector output membership differs")
    require(
        legacy_output(source, "manifest", manifest_path, evidence) == declared["manifest_sha256"],
        "detector manifest output differs",
    )
    for p, digest in (
        (manifest_path, declared["manifest_sha256"]),
        (seal_path, declared["provenance_sha256"]),
    ):
        legacy_input(pack_record, p, digest)
    prefix = raw.local(pack_summary["producer_run"])
    prefix_seal = prefix / "continuation.provenance.json"
    expected = pack_record["inputs"].get(str(prefix_seal))
    require(expected is not None, "pack prefix producer is unbound")
    legacy_input(pack_record, prefix_seal, expected["sha256"])
    producer = read(prefix_seal, evidence, expected["sha256"])
    require(
        producer.get("stage") == "three_body_long_continuation"
        and producer.get("status") == "ok"
        and producer["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded pack prefix producer required",
    )
    require(
        producer["params"]["bindings"] == binding,
        "pack prefix/final source and runtime config differ",
    )
    shared = verify_shared(fixed, record, evidence)
    videos, displayed = {}, []
    for view in VIEWS:
        descriptor = manifest["raw"][view]
        path = native.checked_relative(attempt, descriptor["path"])
        digest = sha(descriptor["sha256"])
        link = declared["raw_hardlinks"][view]
        canonical = raw.local(link["source_path"])
        expected = fixed["detector_inputs"]["video:" + view]
        stat = path.stat()
        require(
            raw.local(expected["path"]) == canonical
            and expected["sha256"] == digest
            and raw.local(link["path"]) == path
            and link["sha256"] == digest
            and os.path.samefile(path, canonical)
            and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            == (link["device"], link["inode"], link["bytes"], link["mtime_ns"]),
            "original MP4 identity/hardlink differs",
        )
        require(
            legacy_output(source, view + "_raw", path, evidence) == digest,
            "detector raw output differs",
        )
        legacy_input(pack_record, path, digest)
        # Detector inputs are role keyed; locate the exact canonical source without aliasing.
        require(
            any(
                Path(row["path"]).resolve() == canonical and row["sha256"] == digest
                for row in source["inputs"].values()
            ),
            "detector original raw input differs",
        )
        evidence.add(canonical, digest)
        evidence.add(path, digest)
        videos[view] = path
        for frame in parsed.anchor_frames:
            name = f"frames/f{int(frame)}_{view}.jpg"
            image = native.checked_relative(pack, name)
            image_sha = legacy_output(pack_record, name, image, evidence)
            displayed.append(
                dict(
                    frame=int(frame),
                    view=view,
                    path=str(image),
                    sha256=image_sha,
                    encoding="lossy JPEG96; native BGR checked separately by caller",
                )
            )
    return dict(
        manifest=manifest,
        videos=videos,
        shared=shared,
        displayed=displayed,
        proof=dict(
            final_run=str(root),
            final_tracking_sha256=final_tracking_sha,
            final_consumed_index=index,
            consumed_record=copy.deepcopy(declared),
            final_consumed_native_interval=[FIRST, STOP],
            scientific_interval=[START, STOP],
            pack_prefix_producer=str(prefix),
            prefix_coordinate_arrays_opened=False,
            prefix_candidate_sha256=producer["outputs"]["candidate"]["sha256"],
            final_candidate_sha256=final["candidate_sha256"],
            final_candidate_payload_verified_here=False,
            milestone_candidate_sha256=final["milestone_candidate_sha256"],
            milestone_summary_sha256=final["milestone_summary_sha256"],
            same_prefix_final_SL_calibration=True,
            same_prefix_final_original_binding=True,
            same_final_consumed_chunk=True,
            source_is_successful_final_run=True,
            original_MP4_sha256={v: manifest["raw"][v]["sha256"] for v in VIEWS},
            raw_images_decoded_here=0,
            all_three_ID_validation=False,
        ),
    )
