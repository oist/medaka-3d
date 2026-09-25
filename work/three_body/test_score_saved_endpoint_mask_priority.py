"""Small synthetic priority/partial-codec tests; every I/O root is temporary."""

import hashlib
import io
import json

import numpy as np
import pytest
import score_saved_endpoint_mask_priority as m


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    work = root / "work" / "three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(m, "REPO", root)
    monkeypatch.setattr(m, "WORK", work)
    for name in ("OLD_ROOT", "NEW_ROOT", "MASK_ROOT"):
        monkeypatch.setattr(m, name, work / name.lower())
    for name in ("load_headers", "source_code", "load_rig"):
        monkeypatch.setattr(
            m, name, lambda *args, **kwargs: pytest.fail("real metadata/rig forbidden")
        )
    real_load = np.load

    def isolated_load(path, *args, **kwargs):
        assert isinstance(path, io.BytesIO), "only captured synthetic bytes may be decoded"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", isolated_load)
    return work


def fixture_bank(work, *, corrupt_sum=False, repeat_descriptor=False):
    n = 3
    fields = dict(
        candidate_index=np.arange(n, dtype=np.int64),
        candidate_pose_xyz=np.arange(n * 27, dtype=np.float64).reshape(n, 3, 3, 3),
        candidate_parent=np.array([0, 0, 1], np.int64),
        candidate_plan=np.zeros((n, 3, 3, 3), np.int32),
        candidate_permutation=np.tile(np.arange(3, dtype=np.int64), (n, 1)),
        base_local_cost=np.array([0.25, 0.5, 0.75]),
        original_five_loss_parts=np.ones((n, 5)),
        sixth_instance_term=np.array([0.5, 0.25, 0.0]),
        adjusted_local_cost=np.array([0.75, 0.75, 0.75]),
        parent_path_cost=np.array([2.0, 2.0, 1.0]),
        adjusted_path_cost=np.array([2.75, 2.75, 1.75]),
        capped_contributions_by_view=np.zeros((n, 3)),
    )
    if corrupt_sum:
        fields["adjusted_path_cost"][0] += 1
    arrays, tree = (
        {},
        dict(
            schema="instance_rescore_frame_diagnostics_v1",
            frame_idx=82720,
            post_fit_only=True,
            temporal_state=False,
            candidate_kind=["predict", "body_row", "body_row"],
        ),
    )
    for index, (field, value) in enumerate(fields.items()):
        # Nonstandard member names ensure no implementation assumes a00000 ordering.
        key = f"selected_{51 - index}"
        arrays[key] = value
        tree[field] = dict(__array__=key, signature=m.signature(value))
    arrays["unused_object_must_not_decode"] = np.array([object()], dtype=object)
    if repeat_descriptor:
        tree["sixth_instance_term"] = tree["base_local_cost"]
    archive = work / "bank.npz"
    with archive.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    def record(path):
        data = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

    archive_ref = record(archive)
    companion = work / "bank.json"
    companion.write_text(json.dumps(dict(archive=archive_ref, tree=tree)), encoding="utf-8")
    return dict(
        archive=archive_ref,
        companion=record(companion),
        arrays={
            field: value["signature"] for field, value in tree.items() if isinstance(value, dict)
        },
    ), fields


def test_priority_preserves_numeric_cost_ties_and_split_backups():
    path = np.array([1.0, 3.0, 1.0, 2.0, 3.0])
    split = np.zeros((5, 3), bool)
    split[0, 2] = split[3, 0] = True
    path_before, split_before = path.copy(), split.copy()
    old, priority, fallback = m.priority_orders(path, split)
    assert old.tolist() == [0, 2, 3, 1, 4]
    assert priority.tolist() == [2, 1, 4, 0, 3]
    assert not fallback
    np.testing.assert_array_equal(path, path_before)
    np.testing.assert_array_equal(split, split_before)


@pytest.mark.parametrize("all_split", [False, True])
def test_uniform_split_status_has_exact_original_order(all_split):
    path = np.array([2.0, 1.0, 1.0])
    old, priority, fallback = m.priority_orders(path, np.full((3, 3), all_split, bool))
    assert old.tolist() == [1, 2, 0]
    np.testing.assert_array_equal(old, priority)
    assert fallback is all_split


def test_path_order_uses_parent_cost_and_selected_codec_reads_only_twelve_fields(isolate):
    ref, expected = fixture_bank(isolate)
    inputs, guards = {}, {}
    actual, kinds = m.load_bank(ref, 82720, inputs, guards)
    assert set(actual) == set(m.FIELDS) and len(actual) == 12
    assert kinds == ["predict", "body_row", "body_row"]
    for field in expected:
        assert m.signature(actual[field]) == m.signature(expected[field])
        assert not actual[field].flags.writeable
    assert len(inputs) == len(guards) == 2
    old, priority, _ = m.priority_orders(actual["adjusted_path_cost"], np.zeros((3, 3), bool))
    assert old.tolist() == priority.tolist() == [2, 0, 1]
    assert np.argsort(actual["adjusted_local_cost"], kind="stable").tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "mode", ["wrong_frame", "sum", "duplicate_descriptor", "wrong_signature", "archive_sha"]
)
def test_bad_bank_contracts_fail_in_isolated_input(mode, isolate):
    ref, _ = fixture_bank(
        isolate, corrupt_sum=mode == "sum", repeat_descriptor=mode == "duplicate_descriptor"
    )
    if mode == "wrong_signature":
        ref["arrays"]["candidate_index"] = dict(dtype="<i8", shape=[3], sha256="0" * 64)
    if mode == "archive_sha":
        ref["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.load_bank(ref, 82721 if mode == "wrong_frame" else 82720, {}, {})


def test_output_guard_fails_before_any_real_metadata_or_payload(isolate):
    with pytest.raises(ValueError, match="fresh output"):
        m.run(isolate.parent / "outside")
    existing = isolate / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        m.run(existing)


def test_json_nonfinite_projection_is_explicit_null_without_changing_arrays():
    value = np.array([1.0, np.nan, np.inf])
    before = m.signature(value)
    assert m.json_safe(dict(projection=value)) == dict(projection=[1.0, None, None])
    assert m.signature(value) == before
