"""Independent detector chunks resume only from complete, unchanged evidence."""

import copy
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_three_body_source_masks import fixture as native_fixture  # noqa: F401

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import prepare_chunked_source_masks as stage  # noqa: E402


@pytest.fixture
def staged_context(native_fixture, tmp_path, monkeypatch):  # noqa: F811
    f = native_fixture
    monkeypatch.setattr(stage, "WORKSPACE", tmp_path)
    protocol = json.loads((LAB / "long_input_protocol.json").read_text(encoding="utf-8"))
    protocol["source_frame_range"] = [49798, 49804]
    protocol["chunk_frames"] = 2
    assets, asset_path = stage.single.original.verify_assets(f.cache)
    protocol["asset_manifest_sha256"] = stage.sha256(asset_path)
    protocol["source_cache_manifest_sha256"] = stage.sha256(f.cache / "source_cache_manifest.json")
    protocol["model_assets_sha256"] = {key: item["sha256"] for key, item in assets.items()}
    protocol["source_videos_sha256"] = {
        view: stage.sha256(
            f.cache / f"medaka_202512/{stage.EXPERIMENT}/{view}/{stage.VIDEO_NAMES[view]}"
        )
        for view in stage.VIEWS
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    def runtime_factory(actual_protocol):
        class VersionString(str):
            pass

        runtime = f.runtime_factory()
        runtime.metadata["test_version_subclass"] = VersionString("1.0")
        runtime.metadata["detector_contract"] = copy.deepcopy(actual_protocol["detector"])
        make = runtime.make_predictor

        def predictor(kind, model_assets, out):
            result, model = make(kind, model_assets, out)
            stage.single.copy_exclusive(
                model_assets[kind + "_config"]["path"], out / f"{kind}_source_config.yaml"
            )
            model.update(
                weights=model_assets[kind + "_weights"], config=model_assets[kind + "_config"]
            )
            return result, model

        runtime.make_predictor = predictor
        return runtime

    f.protocol = protocol
    f.protocol_path = path
    f.context_options = dict(capture_factory=f.Capture, runtime_factory=runtime_factory)
    return f


def test_registered_long_protocol_has_180_exact_native_chunks():
    protocol = stage.validate_protocol(json.loads((LAB / "long_input_protocol.json").read_text()))
    ranges = stage.chunk_ranges(protocol)
    assert len(ranges) == 180
    assert ranges[0] == (49800, 51460) and ranges[-1] == (346940, 348600)
    assert sum(stop - start for start, stop in ranges) == 298800
    assert all(a[1] == b[0] for a, b in pairwise(ranges))


@pytest.mark.parametrize(
    "field,value",
    [
        ("fps", True),
        ("fps", 166.0),
        ("views", ["side1_cam", "top_cam", "side2_cam"]),
        ("chunk_frames", 0),
        ("chunk_frames", 1661),
        ("background_frames", list(range(5000, 45000, 5000))),
        ("background_frames", [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 49468]),
        ("source_frame_range", [True, 49800]),
        ("created_utc", "2026-09-09T00:00:00+09:00"),
        ("asset_manifest_sha256", "A" * 64),
        ("limitations", []),
        ("unregistered_field", "unexpected"),
    ],
)
def test_protocol_rejects_unknown_or_changed_native_contract(field, value):
    protocol = json.loads((LAB / "long_input_protocol.json").read_text())
    protocol[field] = value
    with pytest.raises(ValueError):
        stage.validate_protocol(protocol)


@pytest.mark.parametrize(
    "field,value", [("score_threshold", 0.4), ("autocast", 0), ("batch_size", 2)]
)
def test_detector_settings_cannot_be_relaxed(field, value):
    protocol = json.loads((LAB / "long_input_protocol.json").read_text())
    protocol["detector"][field] = value
    with pytest.raises(ValueError, match="detector"):
        stage.validate_protocol(protocol)


def test_pause_failure_resume_all_views_and_exact_reader_binding(staged_context, monkeypatch):
    f = staged_context
    with pytest.raises(stage.PreparationPaused) as paused:
        stage.run(f.protocol_path, f.cache, f.output, max_new_chunks=1, **f.context_options)
    assert paused.value.summary["status"] == "incomplete"
    assert paused.value.summary["final_sequence_path"] is None
    first = f.output / "chunks/f049798-049800/attempt_0001"
    first_manifest_sha = stage.sha256(first / "manifest.json")
    assert json.loads((first / "staging.provenance.json").read_text())["status"] == "ok"
    assert (
        json.loads((f.output / "runs/run_0001/run.provenance.json").read_text())["status"]
        == "failed"
    )
    assert not (f.output / "assemblies").exists()
    assert len(f.inference_calls) == 6

    f.options["read_failure"] = ("side2_cam", 49800)
    with pytest.raises(RuntimeError, match="position mismatch"):
        stage.run(f.protocol_path, f.cache, f.output, **f.context_options)
    failed = f.output / "chunks/f049800-049802/attempt_0001"
    assert json.loads((failed / "staging.provenance.json").read_text())["status"] == "failed"
    assert not (failed / "manifest.json").exists()
    failed_sha = stage.sha256(failed / "staging.provenance.json")
    assert not (f.output / "assemblies").exists()

    f.options.pop("read_failure")
    result = stage.run(f.protocol_path, f.cache, f.output, **f.context_options)
    assert result["status"] == "complete" and result["completed_native_frames"] == 6
    assert [entry["reused"] for entry in result["chunks"]] == [True, False, False]
    assert result["chunks"][1]["attempt_path"].endswith("attempt_0002")
    assert len(result["chunks"][1]["rejected_attempts"]) == 1
    assert stage.sha256(first / "manifest.json") == first_manifest_sha
    assert stage.sha256(failed / "staging.provenance.json") == failed_sha
    assembled = f.output / result["final_sequence_path"]
    manifest = stage.single.verify_completion(assembled)
    assert manifest["video_payload_bytes_copied"] == 0 and manifest["tracking_performed"] is False
    for view in stage.VIEWS:
        raw = manifest["raw"][view]
        assert os.path.samefile(
            assembled / raw["path"], f.cache / raw["source_cache_relative_path"]
        )
        assert raw["source_offset_frames"] == 49798 and len(raw["frame_bgr_sha256"]) == 6
        with np.load(assembled / manifest["masks"][view]["path"], allow_pickle=False) as data:
            assert data["frame_idx"].tolist() == list(range(49798, 49804))
            assert np.diff(data["frame_offsets"]).tolist() == [4, 0, 0, 0, 0, 0]
            assert data["view_available"].tolist() == [True] * 6
            assert data["source_track_id"].tolist() == [-1] * 4
        assert manifest["backgrounds"][view]["frames"] == f.protocol["background_frames"]
    count = len(f.inference_calls)
    again = stage.run(f.protocol_path, f.cache, f.output, max_new_chunks=1, **f.context_options)
    assert all(entry["reused"] for entry in again["chunks"])
    assert again["final_manifest_sha256"] == result["final_manifest_sha256"]
    assert len(f.inference_calls) == count and all(cap.released for cap in f.captures)

    # Nodes are created only after staging has finished, solely to exercise the
    # existing reader. They cannot have influenced detector staging or resume.
    nodes = f.output.parent / "nodes"
    folder = nodes / "work/c01_s16_predictions/bottomup"
    folder.mkdir(parents=True)
    for view in stage.VIEWS:
        np.savez_compressed(
            folder / f"{stage.EXPERIMENT}__{view}.npz",
            view=view,
            experiment=stage.EXPERIMENT,
            nodes=["mouth", "pectoral", "tail"],
            frame_idx=np.array([49798]),
            points=np.zeros((1, 3, 2)),
        )
    from source_sequence import SourceSequenceReader

    monkeypatch.setattr(cv2, "VideoCapture", f.Capture)
    frames = list(SourceSequenceReader(assembled, nodes))
    assert [frame.frame_idx for frame in frames] == list(range(49798, 49804))
    assert frames[1].masks["top_cam"].shape[0] == 0


def test_tampered_payload_and_runtime_never_reused(staged_context):
    f = staged_context
    context = stage.prepare_context(f.protocol_path, f.cache, **f.context_options)
    parent = f.output / "chunks/f049798-049800"
    first = stage.new_directory(parent)
    manifest = stage.prepare_chunk(context, first, 49798, 49800)
    assert type(context.runtime.metadata["test_version_subclass"]) is str
    changed = copy.deepcopy(context.runtime.metadata)
    context.runtime.metadata["test_only"] = 1
    path, _, rejected = stage.select_attempt(context, parent, 49798, 49800)
    assert path is None and "binding" in rejected[0]["reason"]
    context.runtime.metadata.clear()
    context.runtime.metadata.update(changed)
    assert stage.select_attempt(context, parent, 49798, 49800)[0] == first
    payload = first / manifest["masks"]["top_cam"]["path"]
    with payload.open("ab") as stream:
        stream.write(b"unexpected changed bytes")
    path, _, rejected = stage.select_attempt(context, parent, 49798, 49800)
    assert path is None and "checksum" in rejected[0]["reason"]
    assert stage.new_directory(parent).name == "attempt_0002" and payload.exists()


@pytest.mark.parametrize("target", ["model", "code_snapshot", "input_binding"])
def test_updated_outer_seal_cannot_hide_inconsistent_internal_binding(staged_context, target):
    f = staged_context
    context = stage.prepare_context(f.protocol_path, f.cache, **f.context_options)
    parent = f.output / "chunks/f049798-049800"
    out = stage.new_directory(parent)
    manifest = stage.prepare_chunk(context, out, 49798, 49800)
    seal_path = out / "staging.provenance.json"
    seal = json.loads(seal_path.read_text())
    if target == "model":
        manifest["models"]["top"]["effective_config_sha256"] = "0" * 64
        (out / "manifest.json").write_text(json.dumps(manifest))
        seal["outputs"]["manifest"]["sha256"] = stage.sha256(out / "manifest.json")
    elif target == "code_snapshot":
        name = "code:" + next(iter(context.code_records))
        path = Path(seal["outputs"][name]["path"])
        path.write_bytes(b"different code snapshot")
        seal["outputs"][name]["sha256"] = stage.sha256(path)
    else:
        seal["inputs"]["asset:top_weights"]["sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal))
    path, _, rejected = stage.select_attempt(context, parent, 49798, 49800)
    assert path is None and rejected


@pytest.mark.parametrize("field,value", [("fps", 166.01), ("source_count", 348599)])
def test_invalid_source_metadata_fails_before_runtime_or_output(staged_context, field, value):
    f = staged_context
    f.options[field] = value
    with pytest.raises(ValueError):
        stage.run(f.protocol_path, f.cache, f.output, **f.context_options)
    assert not f.output.exists() and not f.runtime_calls
    assert all(cap.released for cap in f.captures)


def test_wrong_protocol_model_hash_fails_before_runtime_or_output(staged_context):
    f = staged_context
    f.protocol["model_assets_sha256"]["top_weights"] = "0" * 64
    f.protocol_path.write_text(json.dumps(f.protocol))
    with pytest.raises(ValueError, match="model/config digest"):
        stage.run(f.protocol_path, f.cache, f.output, **f.context_options)
    assert not f.output.exists() and not f.runtime_calls
