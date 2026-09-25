"""Synthetic same-horizon bindings/arithmetic; no actual checkpoint or GPU."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import audit_tail_first_chunk as m
import numpy as np
import pytest
from test_audit_bounded_first_chunk import candidate, import_fixture


def fixture():
    bounded, arrays, parent_meta, old_meta = import_fixture()
    context = copy.deepcopy(old_meta["context"])
    context["implementation"] = "Tail"
    context["tail_recovery_policy"] = dict(
        schema="tail_recovery_continuous_policy_v1",
        enabled=True,
        target_node=2,
        allow_bounded=False,
        clears_identity_review=False,
    )
    imported = {**copy.deepcopy(parent_meta), "context": context, "restore_count": 4}
    migration = dict(
        schema="tail_recovery_original_state_import_v1",
        status="ok",
        bounded_validation=bounded,
        bounded_context=old_meta["context"],
        original_metadata=parent_meta,
        imported_metadata=imported,
        old_context=parent_meta["context"],
        new_context=context,
        arrays=m.array_signature(arrays),
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        original_implementation_not_claimed_for_new_tracking=True,
        review_cleared_by_migration=False,
    )
    return migration, arrays, parent_meta, {"context": context}, old_meta


def test_three_stage_import_only_one_restore_and_no_mutation():
    args = fixture()
    before = copy.deepcopy(args)
    m.verify_tail_import(*args)
    assert m.array_signature(args[1]) == m.array_signature(before[1])
    for i in (0, 2, 3, 4):
        assert args[i] == before[i]
    assert args[2]["restore_count"] == 3
    assert args[0]["imported_metadata"]["restore_count"] == 4


@pytest.mark.parametrize("overwrite", [False, True])
def test_seal_verification_cannot_replace_existing_pin(monkeypatch, overwrite):
    def verify(_root, _filename, _stage, hashes):
        hashes["new_output"] = "c" * 64
        if overwrite:
            hashes["already_bound"] = "b" * 64
        return {"status": "ok"}

    monkeypatch.setattr(m, "checks", SimpleNamespace(verify=verify, require=m.checks.require))
    hashes = {"already_bound": "a" * 64}
    if overwrite:
        with pytest.raises(ValueError, match="replaced"):
            m.verify_seal(None, None, None, hashes)
    else:
        assert m.verify_seal(None, None, None, hashes) == {"status": "ok"}
        assert hashes["new_output"] == "c" * 64


@pytest.mark.parametrize(
    "case",
    [
        "array",
        "double_restore",
        "nested",
        "old_context",
        "new_config",
        "enabled",
        "bounded",
        "node",
        "review",
        "flag",
        "stage",
    ],
)
def test_import_rejects_silent_changes(case):
    migration, arrays, parent, new, old = fixture()
    if case == "array":
        arrays["state_pose"].flat[0] = 1
    elif case == "double_restore":
        migration["imported_metadata"]["restore_count"] = 6
    elif case == "nested":
        migration["bounded_validation"]["directional_validation"]["arrays"] = {}
    elif case == "old_context":
        old = copy.deepcopy(old)
        old["context"]["tracker_config"]["extra"] = 1
    elif case == "new_config":
        new["context"]["tracker_config"] = {"device": "cuda", "beam_size": 7}
    elif case == "enabled":
        migration["new_context"]["tail_recovery_policy"]["enabled"] = False
    elif case == "bounded":
        migration["new_context"]["tail_recovery_policy"]["allow_bounded"] = True
    elif case == "node":
        migration["new_context"]["tail_recovery_policy"]["target_node"] = 0
    elif case == "review":
        migration["new_context"]["tail_recovery_policy"]["clears_identity_review"] = True
    elif case == "flag":
        migration["review_cleared_by_migration"] = True
    else:
        migration["schema"] = "bounded_mouth_original_state_import_v1"
    with pytest.raises(ValueError):
        m.verify_tail_import(migration, arrays, parent, new, old)


@pytest.mark.parametrize("case", ["empty", "digest", "future_path", "malformed"])
def test_uncommitted_or_wrong_pointer_before_payload(tmp_path, case):
    target = tmp_path / "checkpoints/through_068060"
    row = dict(path=str(target), manifest_sha256="a" * 64)
    if case == "digest":
        row["manifest_sha256"] = "b" * 64
    elif case == "future_path":
        row["path"] = str(tmp_path / "checkpoints/through_076360")
    (tmp_path / "checkpoint_progress.json").write_text(json.dumps([] if case == "empty" else [row]))
    with pytest.raises(ValueError):
        m.committed(tmp_path, "no" if case == "malformed" else "a" * 64, {})
    assert not target.exists()


def test_committed_first_record_ignores_future_rows(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints/through_068060"
    root.mkdir(parents=True)
    codepath = Path(m.tail.__file__).resolve()
    fixed = dict(
        schema="tail_one_minute_checkpoint_binding_v1",
        native_interval=[m.START, m.comparison.STOP],
        experimental_code_sha256={str(m.RUNNER): m.RUNNER_SHA, str(codepath): m.CLASS_SHA},
        state_import_sha256="d" * 64,
    )
    (root / "manifest.json").write_text(json.dumps({"bindings": fixed}))
    (root / "metadata.json").write_text("{}")
    (root / "arrays.npz").write_bytes(b"synthetic-not-deserialized")
    (tmp_path / "state_import.json").write_text('{"fixture": true}')
    meta = dict(previous_frame=68059, processed_frames=18260, initialize_count=1, restore_count=4)
    row = dict(path=str(root), manifest_sha256="a" * 64, arrays=39, **meta)
    future = dict(path="future-must-not-exist", manifest_sha256="bad")
    (tmp_path / "checkpoint_progress.json").write_text(json.dumps([row, future]))
    opened = []

    def fake_bind(path, digest, hashes):
        opened.append(Path(path))
        hashes[str(path)] = digest
        return Path(path)

    def fake_load(path, binding, hashes):
        assert path == root and binding == fixed
        assert all(
            str(root / name) in hashes for name in ("manifest.json", "metadata.json", "arrays.npz")
        )
        return {"fixture": np.zeros(1)}, meta, {}

    monkeypatch.setattr(m, "bind", fake_bind)
    monkeypatch.setattr(
        m,
        "comparison",
        SimpleNamespace(
            STOP=m.comparison.STOP,
            load_bound_checkpoint=fake_load,
            validate_lifecycle=lambda saved, stop, restores: (
                saved == meta and stop == 68060 and restores == 4
            ),
        ),
    )
    result = m.committed(tmp_path, "a" * 64, {})
    assert result[-1] == row
    assert result[3] == {"fixture": True}
    assert all("future-must-not-exist" not in str(path) for path in opened)
    assert not (tmp_path / "tracking.provenance.json").exists()


def support_fixture():
    part = candidate()
    # Synthetic unique rows; geometric values stand for a sealed prior proof.
    part["sources"][:] = np.arange(3)[None, :, None, None]
    valid = np.ones(part["sources"].shape, bool)
    support = dict(
        frame_idx=part["frame_idx"].copy(),
        valid_source=valid,
        ray_distance_mm=np.full(valid.shape, 1.5),
        native_pixel_error=np.full(valid.shape, 7.0),
    )
    totals = dict(
        native_frames=m.STOP - m.START,
        frame_range=[m.START, m.STOP],
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        source_rows_checked=int(valid.sum()),
        observed_nodes_checked=int(part["observed"].sum()),
    )
    return part, support, totals


def test_prior_support_exact_declared_scope():
    args = support_fixture()
    m.verify_prior_support(*args)
    args[1]["ray_distance_mm"][0, 0, 0, 0] = 2.0
    args[1]["native_pixel_error"][0, 0, 0, 0] = 12.0
    m.verify_prior_support(*args)


@pytest.mark.parametrize("case", ["native", "source", "observed", "ray", "pixel", "count", "flag"])
def test_prior_support_rejects_detached_or_inconsistent_evidence(case):
    part, saved, totals = support_fixture()
    if case == "native":
        saved["frame_idx"][0] += 1
    elif case == "source":
        saved["valid_source"][0, 0, 0, 0] = False
    elif case == "observed":
        part["observed"][0, 0, 0] = False
    elif case == "ray":
        saved["ray_distance_mm"][0, 0, 0, 0] = 2.0001
    elif case == "pixel":
        saved["native_pixel_error"][0, 0, 0, 0] = np.nan
    elif case == "count":
        totals["source_rows_checked"] -= 1
    else:
        totals["all_sources_and_observed_flags_pass"] = False
    with pytest.raises(ValueError):
        m.verify_prior_support(part, saved, totals)


def test_history_copy_keeps_all19_ordered_rows_and_offsets():
    arrays = {"history_offsets": np.arange(0, 19, 6)}
    arrays.update({"history_" + key: np.arange(18) for key in m.RECORD_FIELDS})
    arrays["previous_raw"] = np.zeros((3, 2, 2, 3), np.uint8)
    saved = m.all_history(arrays)
    assert len(saved) == 20 and "previous_raw" not in saved
    assert m.array_signature(saved) == m.array_signature({key: arrays[key] for key in saved})
    for key in saved:
        assert not np.shares_memory(saved[key], arrays[key])


def test_gain_loss_renames_review_and_sources_without_semantic_swap():
    old, new = candidate(), candidate()
    old["observed"][0, 0, 2] = False
    new["observed"][1, 2, 0] = False
    old["identity_review"][0, 0] = True
    new["sources"][0, 0, :, 2] = 3
    totals, rows = m.compare_support(old, new)
    assert totals["gained_by_node"] == [0, 0, 1]
    assert totals["lost_by_node"] == [1, 0, 0]
    assert rows[0]["bounded_review"] is True and rows[0]["tail_review"] is False
    assert rows[0]["bounded_sources"] == [0, 0, 0]
    assert rows[0]["tail_sources"] == [3, 3, 3]
    assert all(not key.startswith("directional_") for row in rows for key in row)
