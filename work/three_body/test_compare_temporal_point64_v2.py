"""Only the changed source inventory and explicit technical retry contract."""

import ast
from pathlib import Path
from types import SimpleNamespace

import compare_temporal_point64_v2 as m
import pytest


def module(file=None, origin=None):
    return SimpleNamespace(__file__=file, __spec__=SimpleNamespace(origin=origin))


def test_relative_torch_pseudofile_never_becomes_workspace_file(tmp_path):
    pseudo = tmp_path / "_classes.py"
    pseudo.write_text("this existing decoy must not enter the source set")
    paths, record = m.source_inventory({"torch.classes": module("_classes.py")}, [], repo=tmp_path)
    assert paths == set()
    assert record["skipped_modules"][0]["stored_file"] == "_classes.py"
    assert "__file__:relative_or_synthetic" in record["skipped_modules"][0]["reasons"]


def test_actual_absolute_spec_origin_precedes_relative_file(tmp_path):
    source = tmp_path / "real.py"
    source.write_text("pass\n")
    paths, record = m.source_inventory(
        {"real": module("relative.py", str(source))}, [], repo=tmp_path
    )
    assert paths == {source}
    assert record["included_modules"][0]["preferred_source"] == str(source)


def test_absolute_workspace_file_included_without_spec(tmp_path):
    source = tmp_path / "real.py"
    source.write_text("pass\n")
    paths, _ = m.source_inventory({"real": module(str(source))}, [], repo=tmp_path)
    assert paths == {source}


@pytest.mark.parametrize("field", ["file", "origin"])
def test_missing_absolute_local_source_is_not_silently_skipped(tmp_path, field):
    info = {field: str(tmp_path / "missing.py")}
    with pytest.raises(ValueError, match="absolute local module source is absent"):
        m.source_inventory({"missing": module(**info)}, [], repo=tmp_path)


def test_missing_absolute_file_still_fails_with_valid_other_origin(tmp_path):
    origin = tmp_path / "real.py"
    origin.write_text("pass\n")
    with pytest.raises(ValueError, match="absolute local module source is absent"):
        m.source_inventory(
            {"conflicting": module(str(tmp_path / "missing.py"), str(origin))}, [], repo=tmp_path
        )


def test_venv_and_external_missing_files_do_not_trigger_workspace_reads(tmp_path):
    paths, record = m.source_inventory(
        {
            "venv": module(str(tmp_path / ".venv/Lib/site-packages/missing.py")),
            "external": module(str(tmp_path.parent / "external.py")),
        },
        [],
        repo=tmp_path,
    )
    assert not paths and len(record["skipped_modules"]) == 2
    assert all(
        "__file__:external_or_environment" in row["reasons"] for row in record["skipped_modules"]
    )


def test_explicit_main_preserves_relative_cli_source(tmp_path):
    main = tmp_path / "entry.py"
    main.write_text("pass\n")
    paths, record = m.source_inventory({"__main__": module("entry.py")}, [main], repo=tmp_path)
    assert paths == {main}
    assert record["explicit_sources"] == [str(main)]
    assert record["skipped_modules"][0]["module"] == "__main__"


def test_required_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="required source is absent"):
        m.source_inventory({}, [tmp_path / "entry.py"], repo=tmp_path)


def test_module_getattr_does_not_synthesize_file(tmp_path):
    class Dynamic:
        def __getattr__(self, name):
            raise AssertionError("module dynamic lookup forbidden: " + name)

    paths, record = m.source_inventory({"dynamic": Dynamic()}, [], repo=tmp_path)
    assert not paths and record["skipped_modules"][0]["stored_file"] is None


def test_every_existing_function_and_class_except_inventory_is_ast_identical():
    assert m.sha(m.V1_SOURCE) == m.V1_SOURCE_SHA
    old = ast.parse(m.V1_SOURCE.read_text(encoding="utf-8"))
    new = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    old_nodes = {
        node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    new_nodes = {
        node.name: node for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert set(new_nodes) - set(old_nodes) == {"source_inventory", "retry_binding"}
    for name in old_nodes:
        if name != "loaded_sources":
            assert ast.dump(old_nodes[name], include_attributes=False) == ast.dump(
                new_nodes[name], include_attributes=False
            ), name
    assert m.STAGE == "temporal_point64_comparison_v2"


def test_retry_binds_failed_seal_source_summary_but_never_tracks(tmp_path, monkeypatch):
    root = tmp_path / "prior"
    (root / "top_cam").mkdir(parents=True)
    top = root / "top_cam/summary.json"
    top.write_text("{}")
    source = tmp_path / "old.py"
    source.write_text("pass\n")
    sealed = dict(
        stage="temporal_point64_comparison_v1",
        status="failed",
        finished_at="time",
        error_type="FileNotFoundError",
        error="_classes.py missing",
        completed_views=["top_cam"],
        inputs={str(source): m.V1_SOURCE_SHA},
        outputs={"top_cam/summary.json": dict(path=str(top), size_bytes=2, sha256="b" * 64)},
    )
    (root / "comparison.provenance.json").write_bytes(m.json_bytes(sealed))
    monkeypatch.setattr(m, "V1_ROOT", root)
    monkeypatch.setattr(m, "V1_SOURCE", source)
    events = []
    result = m.retry_binding(
        SimpleNamespace(add=lambda path, expected: events.append((path, expected)))
    )
    assert [p for p, _ in events] == [root / "comparison.provenance.json", source, top]
    assert result["expected_new_model_forward_calls"] == 21
    assert result["expected_repeated_top_forward_calls"] == 7
    assert result["full_three_views_reexecution_planned"]
    assert "full_three_views_reexecuted" not in result
    assert not result["prior_predictions_reused"]
