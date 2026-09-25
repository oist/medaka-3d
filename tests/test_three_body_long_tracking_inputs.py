"""Sealed chunk prefixes can grow; previously consumed evidence cannot change."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_three_body_chunked_source_masks import staged_context  # noqa: F401
from test_three_body_source_masks import fixture as native_fixture  # noqa: F401

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import long_tracking_inputs as module  # noqa: E402


@pytest.fixture
def fixture(staged_context, tmp_path, monkeypatch):  # noqa: F811
    f = staged_context
    context = module.staging.prepare_context(f.protocol_path, f.cache, **f.context_options)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "PROTOCOL_PATH", f.protocol_path)
    monkeypatch.setattr(module, "PROTOCOL_SHA256", module.sha256(f.protocol_path))
    monkeypatch.setattr(module, "FULL_RANGE", (49798, 49804))
    monkeypatch.setattr(module, "RECORDED_RUNTIME", copy.deepcopy(context.runtime.metadata))
    monkeypatch.setattr(module, "_make_context", lambda: context)
    seed, inputs = tmp_path / "seed", tmp_path / "tracking_inputs"
    seed.mkdir()
    inputs.mkdir()
    marker = seed / "registered_first_points.bin"
    marker.write_bytes(b"initial first-frame state only")
    seed_hashes = {str(marker): module.sha256(marker)}
    seed_binding = dict(registration_sha256=seed_hashes[str(marker)], input_root=str(inputs))
    monkeypatch.setattr(module, "_verify_seed", lambda a, b: (
        np.zeros((3, 3, 3)), SimpleNamespace(views=module.VIEWS),
        copy.deepcopy(seed_binding), dict(seed_hashes),
    ))

    def make_chunk(start, stop):
        parent = f.output / "chunks" / f"f{start:06d}-{stop:06d}"
        path = module.staging.new_directory(parent)
        module.staging.prepare_chunk(context, path, start, stop)
        return path

    first = make_chunk(49798, 49800)
    f.long_seed, f.long_inputs, f.first, f.context = seed, inputs, first, context
    f.make_chunk = make_chunk
    f.load = lambda through=49800: module.load_long_inputs(
        seed, f.output, inputs, through_frame=through
    )
    return f


def reseal_manifest(path, mutate):
    manifest_path, seal_path = path / "manifest.json", path / "staging.provenance.json"
    manifest, seal = module.read_json(manifest_path), module.read_json(seal_path)
    mutate(manifest, seal)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seal["outputs"]["manifest"]["sha256"] = module.sha256(manifest_path)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")


def test_successful_chunk_without_any_global_assembly_is_explicit_prefix(fixture):
    f = fixture
    inference_calls, runtime_calls = len(f.inference_calls), len(f.runtime_calls)
    result = f.load()
    assert result.frame_idx.tolist() == [49798, 49799]
    assert result.full_frame_range == (49798, 49804)
    assert result.chunk_roots == (f.first,)
    assert not result.full_interval_covered and not result.global_assembly_verified
    assert not (f.output / "assemblies").exists()
    assert len(f.inference_calls) == inference_calls and len(f.runtime_calls) == runtime_calls
    for link in result.chunk_records[0]["raw_hardlinks"].values():
        assert link["source_path"] in result.verified_hashes
        assert link["path"] not in result.verified_hashes


def test_new_chunks_leave_fixed_bindings_and_consumed_prefix_identical(fixture):
    f = fixture
    first = f.load()
    f.make_chunk(49800, 49802)
    second = f.load(49802)
    assert first.fixed_bindings == second.fixed_bindings
    module.verify_prefix_records(first.chunk_records, second.chunk_records)
    f.make_chunk(49802, 49804)
    full = f.load(49804)
    assert full.full_interval_covered and not full.global_assembly_verified
    assert full.fixed_bindings == first.fixed_bindings
    module.verify_prefix_records(second.chunk_records, full.chunk_records)


def test_missing_next_chunk_cannot_be_silently_skipped(fixture):
    f = fixture
    f.make_chunk(49802, 49804)
    with pytest.raises(ValueError, match=r"missing completed chunk.*49800,49802"):
        f.load(49804)


def test_failed_first_attempt_is_preserved_and_next_success_is_selected(fixture):
    f = fixture
    first_result = f.load()
    seal = f.first / "staging.provenance.json"
    value = module.read_json(seal)
    value["status"] = "failed"
    seal.write_text(json.dumps(value), encoding="utf-8")
    failed_sha = module.sha256(seal)
    with pytest.raises(ValueError, match="missing completed chunk"):
        f.load()
    replacement = f.make_chunk(49798, 49800)
    result = f.load()
    assert result.chunk_roots == (replacement,)
    assert module.sha256(seal) == failed_sha
    with pytest.raises(ValueError, match="consumed chunk evidence changed"):
        module.verify_prefix_records(first_result.chunk_records, result.chunk_records)


def test_later_success_does_not_replace_an_earlier_valid_attempt(fixture):
    f = fixture
    first = f.load()
    f.make_chunk(49798, 49800)
    repeated = f.load()
    assert repeated.chunk_roots == (f.first,)
    module.verify_prefix_records(first.chunk_records, repeated.chunk_records)


@pytest.mark.parametrize("kind", ["foreign_input", "wrong_protocol", "native_fps", "truth_claim"])
def test_resealed_foreign_or_inconsistent_metadata_is_rejected(fixture, kind):
    f = fixture

    def mutate(manifest, seal):
        if kind == "foreign_input":
            seal["inputs"]["human_labels"] = dict(path="Z:/never-open-this.json", sha256="0" * 64)
        elif kind == "wrong_protocol":
            manifest["protocol_sha256"] = "0" * 64
        elif kind == "native_fps":
            manifest["fps"] = 165
        else:
            manifest["human_truth_created"] = True

    reseal_manifest(f.first, mutate)
    with pytest.raises(ValueError, match="missing completed chunk"):
        f.load()


def test_unsealed_mask_payload_change_is_rejected(fixture):
    f = fixture
    manifest = module.read_json(f.first / "manifest.json")
    path = f.first / manifest["masks"][module.VIEWS[0]]["path"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="missing completed chunk"):
        f.load()


def test_equal_video_bytes_without_source_hardlink_are_not_accepted(fixture):
    f = fixture

    def mutate(manifest, seal):
        view = module.VIEWS[0]
        original = f.first / manifest["raw"][view]["path"]
        copied = original.with_name("copied_video.mp4")
        copied.write_bytes(original.read_bytes())
        manifest["raw"][view]["path"] = copied.relative_to(f.first).as_posix()
        seal["outputs"][view + "_raw"]["path"] = str(copied)

    reseal_manifest(f.first, mutate)
    with pytest.raises(ValueError, match="missing completed chunk"):
        f.load()


@pytest.mark.parametrize("value", [49800, 49801, 51459, 51461, 348601, True, 51460.0])
def test_non_chunk_endpoint_is_rejected_before_opening_inputs(value):
    with pytest.raises(ValueError, match=r"native integer|chunk endpoint"):
        module.load_long_inputs("absent", "absent", through_frame=value)


def test_wrong_frozen_protocol_is_rejected_before_inputs(tmp_path, monkeypatch):
    path = tmp_path / "wrong_protocol.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "PROTOCOL_PATH", path)
    with pytest.raises(ValueError, match="wrong frozen long protocol"):
        module.load_long_inputs("absent", "absent", through_frame=51460)


def test_wrong_registered_seed_hash_is_rejected_without_baseline_reads(tmp_path):
    (tmp_path / "registration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong frozen seed registration"):
        module._verify_seed(tmp_path, tmp_path / "not-opened")


def test_failed_seed_registration_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "registration.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "SEED_SHA256", module.sha256(path))
    (tmp_path / "registration.provenance.json").write_text(
        json.dumps(dict(status="failed", stage="three_body_long_seed_registration")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed registration did not complete"):
        module._verify_seed(tmp_path, tmp_path / "not-opened")


def test_verification_context_has_no_predictor_or_cuda_runtime(monkeypatch):
    def prepare(protocol_path, source_cache, *, runtime_factory):
        runtime = runtime_factory({})
        assert runtime.metadata == module.RECORDED_RUNTIME
        assert not hasattr(runtime, "make_predictor") and not hasattr(runtime, "synchronize")
        return runtime

    monkeypatch.setattr(module.staging, "prepare_context", prepare)
    assert module._make_context().metadata == module.RECORDED_RUNTIME


@pytest.mark.parametrize("mutation", ["reorder", "changed_sha", "shorter", "int_bool"])
def test_checkpoint_prefix_rejects_reordering_or_changed_evidence(mutation):
    previous = [dict(index=0, start=49800, stop=51460, manifest_sha256="a")]
    current = [*previous, dict(index=1, start=51460, stop=53120, manifest_sha256="b")]
    current = copy.deepcopy(current)
    if mutation == "reorder":
        current.reverse()
    elif mutation == "changed_sha":
        current[0]["manifest_sha256"] = "changed"
    elif mutation == "shorter":
        current = []
    else:
        current[0]["index"] = False
    with pytest.raises(ValueError, match=r"reordered|changed|shorter"):
        module.verify_prefix_records(previous, current)
