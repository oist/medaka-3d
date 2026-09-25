"""Reproducible staging must not silently omit, overwrite or escape inputs."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PurePosixPath

import pytest

MODULE = Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking/stage_inputs.py"
SPEC = importlib.util.spec_from_file_location("_generative_staging_test", MODULE)
staging = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = staging
SPEC.loader.exec_module(staging)


@pytest.fixture
def tmp_path(tmp_path_factory):
    # The immutable calibration bundle name is long. Avoid pytest's additional
    # full test-function name so these fixtures fit ordinary Windows path limits.
    return tmp_path_factory.mktemp("gs")


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "source"
    for index, relative in enumerate(staging.RELATIVE_INPUTS):
        path = source.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"small fixture {index}: {relative}\n".encode())
    # A raw recording and unrelated data must never be staged by wildcard.
    (source / "do_not_copy.mp4").write_bytes(b"raw video placeholder")
    (source / "unrelated.txt").write_text("leave in source", encoding="utf-8")
    return source


def test_complete_stage_matches_fixed_allowlist_checksums_and_provenance(source_tree, tmp_path):
    out = tmp_path / "new" / "staged"
    path = staging.stage_inputs(source_tree, out)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 18
    assert {entry["relative"] for entry in manifest["files"]} == set(staging.RELATIVE_INPUTS)
    assert not (out / "do_not_copy.mp4").exists()
    assert not (out / "unrelated.txt").exists()
    for entry in manifest["files"]:
        assert "\\" not in entry["relative"]
        relative = PurePosixPath(entry["relative"])
        local = out.joinpath(*relative.parts)
        original = source_tree.joinpath(*relative.parts)
        assert local.read_bytes() == original.read_bytes()
        expected = hashlib.sha256(original.read_bytes()).hexdigest()
        assert entry["sha256"] == entry["source_sha256"] == expected
        assert entry["bytes"] == entry["source_bytes"] == original.stat().st_size
    provenance = json.loads((out / "staging.provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "ok"
    assert provenance["params"]["copies_raw_videos"] is False
    assert provenance["outputs"]["staging_manifest"]["sha256"] == staging.sha256(path)


def test_missing_input_fails_before_creating_output_parent(source_tree, tmp_path):
    missing = source_tree.joinpath(*PurePosixPath(staging.RELATIVE_INPUTS[-1]).parts)
    missing.rename(missing.with_suffix(".unavailable"))
    out = tmp_path / "new_parent" / "staged"
    with pytest.raises(FileNotFoundError, match=r"swap_events\.json"):
        staging.stage_inputs(source_tree, out)
    assert not out.parent.exists()


def test_existing_output_is_untouched(source_tree, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_bytes(b"keep these bytes")
    with pytest.raises(FileExistsError, match="overwrite"):
        staging.stage_inputs(source_tree, out)
    assert sentinel.read_bytes() == b"keep these bytes"
    assert list(out.iterdir()) == [sentinel]


@pytest.mark.parametrize("unsafe", ["../outside", "/absolute", "C:/drive", "a\\b", "a/../b"])
def test_unsafe_relative_names_fail_preflight(source_tree, tmp_path, monkeypatch, unsafe):
    monkeypatch.setattr(staging, "RELATIVE_INPUTS", (unsafe,))
    out = tmp_path / "stage"
    with pytest.raises(ValueError, match="relative path"):
        staging.stage_inputs(source_tree, out)
    assert not out.exists()


def test_resolved_source_escape_fails_preflight(source_tree, tmp_path, monkeypatch):
    original_resolve = Path.resolve
    outside = tmp_path / "outside_file"
    outside.write_bytes(b"outside requested source")
    escaping = source_tree.joinpath(*PurePosixPath(staging.RELATIVE_INPUTS[0]).parts)

    def resolve(path, *args, **kwargs):
        return outside if path == escaping else original_resolve(path, *args, **kwargs)

    # Resolve behavior simulates a link on Windows without requiring symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve)
    out = tmp_path / "stage"
    with pytest.raises(ValueError, match="outside requested root"):
        staging.stage_inputs(source_tree, out)
    assert not out.exists()


def test_corrupt_copy_fails_with_failure_provenance_and_no_success_manifest(
    source_tree, tmp_path, monkeypatch
):
    original_hash = staging.sha256
    out = tmp_path / "stage"

    def corrupt_hash(path):
        return "0" * 64 if path.is_relative_to(out) else original_hash(path)

    monkeypatch.setattr(staging, "sha256", corrupt_hash)
    with pytest.raises(RuntimeError, match="differs from source"):
        staging.stage_inputs(source_tree, out)
    assert not (out / "staging_manifest.json").exists()
    provenance = json.loads((out / "staging.provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "failed"
    assert "differs from source" in provenance["error"]


def test_cli_reports_manifest_path(source_tree, tmp_path, capsys):
    out = tmp_path / "cli_output"
    staging.main(["--source-root", str(source_tree), "--out", str(out)])
    result = json.loads(capsys.readouterr().out)
    assert result["files"] == 18
    assert Path(result["manifest"]).is_file()
