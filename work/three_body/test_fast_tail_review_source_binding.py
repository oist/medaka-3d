"""Synthetic source-lineage tests; no actual human export or completed tracking run."""

import copy
import json
from pathlib import Path

import fast_tail_review_source_binding as m
import numpy as np
import pytest
from test_map_long_review_to_final import context as legacy_context  # noqa: F401
from test_map_long_review_to_final import item, write


def custom(path):
    return dict(item(path), size_bytes=path.stat().st_size)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@pytest.fixture
def bound(request):
    f = request.getfixturevalue("legacy_context")
    root = f.chain["final_run"]
    binding = f.chain["binding"]
    fixed = binding["inputs"]
    protocol = write(f.root / "protocol.json", {"synthetic": "registered source"})
    fixed.update(protocol_path=str(protocol), protocol_sha256=item(protocol)["sha256"])
    fixed["detector_inputs"]["protocol"] = item(protocol)
    fixed["detector_binding"]["protocol_sha256"] = item(protocol)["sha256"]
    fixed["detector_binding"]["input_sha256"]["protocol"] = item(protocol)["sha256"]
    f.manifest["protocol_sha256"] = fixed["protocol_sha256"]
    write(f.manifestpath, f.manifest)
    f.source["outputs"]["manifest"] = item(f.manifestpath)
    write(f.sourceseal, f.source)
    row = f.summary["consumed_record"]
    row.update(
        manifest_sha256=item(f.manifestpath)["sha256"],
        provenance_sha256=item(f.sourceseal)["sha256"],
        output_hashes=copy.deepcopy(f.source["outputs"]),
    )
    manifest = Path(next(p for p in f.shared if p.endswith("calibration_manifest.json")))
    write(manifest, dict(model_files=[dict(path=f"cam{i}.json") for i in range(3)]))
    f.shared[str(manifest)] = item(manifest)["sha256"]
    binding["tracking_protocol_sha256"] = "a" * 64
    binding["opencv_threads"] = 32
    prefix = read(f.prefixseal)
    prefix["params"].update(bindings=binding, all_input_source_snapshot_output_sha_unchanged=True)
    write(f.prefixseal, prefix)
    f.summary.update(
        schema="long_raw_review_pack_v1",
        technical_completion=True,
        model_outputs_shown=False,
        human_truth_created=False,
        original_raw_storage="original_source_video",
        review_clip_range=[65859, 66223],
    )
    write(f.pack / "summary.json", f.summary)
    write(f.pack / "source_manifest.json", f.manifestpath.read_bytes())
    pack = read(f.pack / "review.provenance.json")
    pack.update(
        stage="long_original_video_raw_review",
        status="ok",
        params=dict(all_input_source_output_sha_unchanged=True),
    )
    for path in (f.prefixseal, f.manifestpath, f.sourceseal):
        pack["inputs"][str(path)] = item(path)
    for name in ("summary.json", "source_manifest.json"):
        pack["outputs"][name] = item(f.pack / name)
    write(f.pack / "review.provenance.json", pack)
    document = copy.deepcopy(f.parsed.review)
    document["source_manifest_sha256"] = item(f.manifestpath)["sha256"]
    f.parsed = m.review.parse_review(
        json.dumps(document).encode(), f.manifestpath.read_bytes(), clip_range=[65859, 66223]
    )
    chunks = read(f.consumedpath)[:40]
    chunks[row["index"]] = row
    write(root / "input_index.json", dict(original_binding=binding, prefix_chunks=chunks))
    write(
        root / "plan.json",
        dict(
            schema=m.STAGE,
            original_binding=binding,
            scientific_interval=[m.START, m.STOP],
            process_interval=[m.RESUME, m.STOP],
            source_sha256={str(m.RUNNER): m.RUNNER_SHA},
        ),
    )
    write(
        root / "summary.json",
        dict(
            schema=m.STAGE,
            status="ok",
            scientific_interval=[m.START, m.STOP],
            process_interval=[m.RESUME, m.STOP],
            native_frames=49800,
            process_frames=39840,
            initialize_count=1,
            restore_count=5,
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls=1,
            physical_ID_truth=False,
            checkpoints=[
                dict(previous_frame=p - 1)
                for p in range(m.RESUME + m.CHUNK, m.STOP + m.CHUNK, m.CHUNK)
            ],
        ),
    )
    write(root / "candidate.npz", b"synthetic final candidate; never decode")
    milestone = root / "milestones/through_116200"
    write(milestone / "candidate.npz", b"different archive bytes are permitted")
    write(
        milestone / "summary.json",
        dict(
            schema=m.STAGE + "_milestone",
            status="ok",
            horizon_stop=m.STOP,
            scientific_interval=[m.START, m.STOP],
            original_binding=binding,
            source_sha256={str(m.RUNNER): m.RUNNER_SHA},
            candidate=custom(milestone / "candidate.npz"),
        ),
    )
    seal = dict(
        stage=m.STAGE,
        status="ok",
        inputs={str(m.RUNNER): m.RUNNER_SHA, str(protocol): item(protocol)["sha256"], **f.shared},
        outputs={
            name: custom(root / name)
            for name in (
                "input_index.json",
                "plan.json",
                "summary.json",
                "candidate.npz",
                "milestones/through_116200/summary.json",
                "milestones/through_116200/candidate.npz",
            )
        },
    )
    f.finalseal = write(root / "tracking.provenance.json", seal)
    return f


def invoke(f):
    evidence = m.legacy.final.phase.Digests()
    value = m.verify_source_binding(
        f.chain["final_run"], item(f.finalseal)["sha256"], f.parsed, f.summary, f.pack, evidence
    )
    evidence.final_guard()
    return value, evidence


def reseal(f, relative):
    record = read(f.finalseal)
    record["outputs"][relative] = custom(f.chain["final_run"] / relative)
    write(f.finalseal, record)


def update_pack(f, relative):
    record = read(f.pack / "review.provenance.json")
    path = f.pack / relative
    record["outputs"][relative] = item(path)
    write(f.pack / "review.provenance.json", record)


def test_success_binds_real_hardlinks_and_keeps_candidate_payloads_closed(bound, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("geometry payload opened"))
    before = copy.deepcopy(bound.parsed.review)
    result, evidence = invoke(bound)
    proof = result["proof"]
    assert proof["same_final_consumed_chunk"] is True
    assert proof["final_candidate_sha256"] != proof["milestone_candidate_sha256"]
    assert proof["source_is_successful_final_run"] is True
    assert proof["raw_images_decoded_here"] == 0
    assert proof["final_candidate_payload_verified_here"] is False
    assert proof["all_three_ID_validation"] is False
    assert proof["final_consumed_index"] == 9
    assert len(result["displayed"]) == 9 and len(result["shared"]) == 7
    assert set(result["videos"]) == set(m.VIEWS)
    assert bound.parsed.review == before
    assert not any(p.endswith("candidate.npz") for p in evidence.paths)


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "wrong_stage",
        "runner",
        "lifecycle",
        "checkpoints",
        "unsealed_index",
        "short_prefix",
        "native_order",
        "boolean_index",
        "plan_binding",
        "milestone_binding",
        "milestone_candidate",
    ],
)
def test_final_success_and_exact_five_minute_source_required(bound, fault):
    f = bound
    root = f.chain["final_run"]
    if fault in ("failed", "wrong_stage", "runner"):
        seal = read(f.finalseal)
        if fault == "failed":
            seal["status"] = "failed"
        elif fault == "wrong_stage":
            seal["stage"] = "three_body_long_continuation"
        else:
            seal["inputs"][str(m.RUNNER)] = "0" * 64
        write(f.finalseal, seal)
    elif fault in ("lifecycle", "checkpoints"):
        data = read(root / "summary.json")
        if fault == "lifecycle":
            data["initialize_count"] = 2
        else:
            data["checkpoints"].pop()
        write(root / "summary.json", data)
        reseal(f, "summary.json")
    elif fault in ("unsealed_index", "short_prefix", "native_order", "boolean_index"):
        data = read(root / "input_index.json")
        if fault in ("unsealed_index", "short_prefix"):
            data["prefix_chunks"].pop()
        elif fault == "native_order":
            data["prefix_chunks"][3]["start"] += 1
        else:
            data["prefix_chunks"][0]["index"] = False
        write(root / "input_index.json", data)
        if fault != "unsealed_index":
            reseal(f, "input_index.json")
    elif fault == "plan_binding":
        data = read(root / "plan.json")
        data["original_binding"]["opencv_threads"] = 2
        write(root / "plan.json", data)
        reseal(f, "plan.json")
    else:
        name = "milestones/through_116200/summary.json"
        data = read(root / name)
        if fault == "milestone_binding":
            data["original_binding"]["opencv_threads"] = 2
        else:
            data["candidate"]["sha256"] = "0" * 64
        write(root / name, data)
        reseal(f, name)
    with pytest.raises((ValueError, AssertionError)):
        invoke(f)


@pytest.mark.parametrize(
    "fault",
    [
        "chunk",
        "native_clip",
        "manifest_bytes",
        "pack_failed",
        "legacy_pack",
        "prefix_failed",
        "prefix_config",
        "prefix_SL",
        "raw_source",
        "hardlink_stat",
        "display_hash",
        "display_escape",
        "attempt_escape",
        "protocol_edge",
        "SL_edge",
        "calibration_membership",
    ],
)
def test_wrong_review_native_source_or_configuration_is_rejected(bound, fault):
    f = bound
    root = f.chain["final_run"]
    if fault in ("chunk", "native_clip"):
        if fault == "chunk":
            f.summary["consumed_record"]["manifest_sha256"] = "0" * 64
        else:
            f.summary["review_clip_range"] = [65858, 66223]
        write(f.pack / "summary.json", f.summary)
        update_pack(f, "summary.json")
    elif fault == "manifest_bytes":
        write(f.pack / "source_manifest.json", b"{}")
        update_pack(f, "source_manifest.json")
    elif fault in ("pack_failed", "legacy_pack", "display_escape"):
        pack = read(f.pack / "review.provenance.json")
        if fault == "pack_failed":
            pack["status"] = "failed"
        elif fault == "legacy_pack":
            pack["stage"] = "old_FFV1_pack"
        else:
            name = next(n for n in pack["outputs"] if n.endswith(".jpg"))
            pack["outputs"][name]["path"] = str(f.root / "not_in_pack.jpg")
            write(f.root / "not_in_pack.jpg", b"synthetic escaped")
        write(f.pack / "review.provenance.json", pack)
    elif fault in ("prefix_failed", "prefix_config", "prefix_SL"):
        prefix = read(f.prefixseal)
        if fault == "prefix_failed":
            prefix["status"] = "failed"
        elif fault == "prefix_config":
            prefix["params"]["bindings"]["opencv_threads"] = 2
        else:
            prefix["params"]["bindings"]["inputs"]["seed"]["SL_and_calibration_sha256"] = {}
        write(f.prefixseal, prefix)
        pack = read(f.pack / "review.provenance.json")
        pack["inputs"][str(f.prefixseal)] = item(f.prefixseal)
        write(f.pack / "review.provenance.json", pack)
    elif fault == "display_hash":
        pack = read(f.pack / "review.provenance.json")
        name = next(n for n in pack["outputs"] if n.endswith(".jpg"))
        write(f.pack / name, b"changed synthetic display")
    elif fault in ("protocol_edge", "SL_edge"):
        seal = read(f.finalseal)
        key = (
            f.chain["binding"]["inputs"]["protocol_path"]
            if fault == "protocol_edge"
            else next(iter(f.shared))
        )
        seal["inputs"][key] = "0" * 64
        write(f.finalseal, seal)
    elif fault == "calibration_membership":
        manifest = Path(next(p for p in f.shared if p.endswith("calibration_manifest.json")))
        write(manifest, dict(model_files=[dict(path="../outside.json")] * 3))
    else:
        index = read(root / "input_index.json")
        row = index["prefix_chunks"][9]
        if fault == "attempt_escape":
            row["attempt_path"] = "../escape"
        elif fault == "hardlink_stat":
            row["raw_hardlinks"][m.VIEWS[0]]["inode"] += 1
        else:
            row["raw_hardlinks"][m.VIEWS[0]]["source_path"] = str(f.sourceseal)
        f.summary["consumed_record"] = copy.deepcopy(row)
        write(root / "input_index.json", index)
        reseal(f, "input_index.json")
        write(f.pack / "summary.json", f.summary)
        update_pack(f, "summary.json")
    with pytest.raises((ValueError, AssertionError)):
        invoke(f)


@pytest.mark.parametrize(
    "fault", ["protocol_path", "protocol_sha", "detector_protocol", "detector_input"]
)
def test_protocol_must_connect_detector_and_registered_source(bound, fault):
    fixed = copy.deepcopy(bound.chain["binding"]["inputs"])
    if fault == "protocol_path":
        fixed["detector_inputs"]["protocol"]["path"] = str(bound.manifestpath)
    elif fault == "protocol_sha":
        fixed["detector_inputs"]["protocol"]["sha256"] = "0" * 64
    elif fault == "detector_protocol":
        fixed["detector_binding"]["protocol_sha256"] = "0" * 64
    else:
        fixed["detector_binding"]["input_sha256"]["protocol"] = "0" * 64
    with pytest.raises(ValueError, match="source protocol binding differs"):
        m.verify_shared(fixed, read(bound.finalseal), m.legacy.final.phase.Digests())
