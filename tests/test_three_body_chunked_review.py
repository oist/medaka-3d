"""Independent resume review: re-sealing cannot relax the registered contract."""

import copy
import json

import pytest

from test_three_body_chunked_source_masks import stage, staged_context  # noqa: F401
from test_three_body_source_masks import fixture as native_fixture  # noqa: F401


def _completed(f):
    result = stage.run(f.protocol_path, f.cache, f.output, **f.context_options)
    out = f.output / result["final_sequence_path"]
    context = stage.prepare_context(f.protocol_path, f.cache, **f.context_options)
    index = json.loads((out / "chunks.json").read_text())
    return context, out, index["chunks"]


def test_resealed_assembly_index_still_obeys_protocol(staged_context):  # noqa: F811
    f = staged_context
    context, out, entries = _completed(f)
    paths = {
        name: out / name for name in ("chunks.json", "manifest.json", "staging.provenance.json")
    }
    original = {name: path.read_bytes() for name, path in paths.items()}

    def validate():
        return stage.validate_attempt(
            context, out, 49798, 49804, "assembled_independent_detector_chunks", entries
        )

    validate()
    for field, value in (
        ("schema", "unregistered_chunk_index"),
        ("source_frame_range", [49798.0, 49804.0]),
        ("chunk_frames", True),
        ("attempt_paths_relative_to", str(out)),
        ("chunks", entries[::-1]),
    ):
        index = json.loads(original["chunks.json"])
        index[field] = value
        paths["chunks.json"].write_text(json.dumps(index))
        digest = stage.sha256(paths["chunks.json"])
        manifest = json.loads(original["manifest.json"])
        manifest["chunk_index_sha256"] = digest
        paths["manifest.json"].write_text(json.dumps(manifest))
        seal = json.loads(original["staging.provenance.json"])
        seal["outputs"]["chunk_index"]["sha256"] = digest
        seal["outputs"]["manifest"]["sha256"] = stage.sha256(paths["manifest.json"])
        paths["staging.provenance.json"].write_text(json.dumps(seal))
        with pytest.raises(ValueError, match="different selected chunk attempts"):
            validate()
        for name, path in paths.items():
            path.write_bytes(original[name])
    validate()


def test_assembly_rejects_incomplete_reordered_or_changed_chunk_inputs(staged_context, monkeypatch):  # noqa: F811
    f = staged_context
    context, out, entries = _completed(f)
    for number, bad in enumerate((entries[:-1], entries[::-1], [entries[0]] * 3)):
        target = f.output / f"incomplete_assembly_{number}"
        with pytest.raises(ValueError, match="every registered chunk exactly once in order"):
            stage.assemble(context, f.output, target, bad)
        assert not target.exists()

    # An input changed after preflight must invalidate the entire assembly even
    # when the merger already decoded the original valid bytes successfully.
    merge = stage.concatenate_packed_masks
    altered = []

    def merge_then_change_input(paths):
        paths = list(paths)
        arrays = merge(paths)
        if not altered:
            with paths[0].open("ab") as stream:
                stream.write(b"changed after decoding")
            altered.append(paths[0])
        return arrays

    original_manifest_sha = stage.sha256(out / "manifest.json")
    monkeypatch.setattr(stage, "concatenate_packed_masks", merge_then_change_input)
    failed = stage.new_directory(f.output / "assemblies")
    with pytest.raises(ValueError, match="sealed chunk inputs changed during assembly"):
        stage.assemble(context, f.output, failed, entries)
    assert altered and not (failed / "manifest.json").exists()
    assert json.loads((failed / "staging.provenance.json").read_text())["status"] == "failed"
    assert stage.sha256(out / "manifest.json") == original_manifest_sha


def test_resealed_source_config_must_match_registered_asset(staged_context):  # noqa: F811
    f = staged_context
    context = stage.prepare_context(f.protocol_path, f.cache, **f.context_options)
    parent = f.output / "chunks/f049798-049800"
    out = stage.new_directory(parent)
    stage.prepare_chunk(context, out, 49798, 49800)
    stage.validate_attempt(context, out, 49798, 49800)
    seal_path = out / "staging.provenance.json"
    seal = json.loads(seal_path.read_text())
    registered_assets = copy.deepcopy(context.assets)
    config = out / "top_source_config.yaml"
    config.write_bytes(b"MODEL:\n  WEIGHTS: a-different-model\n")
    seal["outputs"]["top_source_config"]["sha256"] = stage.sha256(config)
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="model/config metadata differs"):
        stage.validate_attempt(context, out, 49798, 49800)
    assert context.assets == registered_assets
