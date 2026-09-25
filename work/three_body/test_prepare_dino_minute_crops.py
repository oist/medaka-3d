"""Synthetic only: no actual source movie, mask archive, model or candidate."""

import copy
from itertools import pairwise

import numpy as np
import prepare_dino_minute_crops as m
import pytest
from prepare_sequence import PackedMasks


def counts():
    return {(n, v): 2 for n in m.FRAMES for v in m.VIEWS}


def test_fixed60_adjacent177_all_cartesian_and_no_original_enumerator_mutation():
    source = counts()
    before = copy.deepcopy(source)
    old_frames = m.partial.FRAMES
    plan = m.enumerate_plan(source)
    assert source == before and old_frames == m.partial.FRAMES
    assert len(plan["crops"]) == 360 and plan["maximum_forward_calls"] == 720
    assert len(plan["requests"]) == 177 and len(plan["pairs"]) == 708
    assert [(r["earlier_frame"], r["later_frame"], r["view"]) for r in plan["requests"]] == [
        (a, b, v) for a, b in pairwise(m.FRAMES) for v in m.VIEWS
    ]
    for i, row in enumerate(plan["requests"]):
        actual = [p for p in plan["pairs"] if p["request_index"] == i]
        assert [(p["earlier_mask_row"], p["later_mask_row"]) for p in actual] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        assert row["later_frame"] - row["earlier_frame"] == 166


def test_unavailable_and_available_zero_do_not_bridge_or_substitute():
    source = counts()
    source[m.FRAMES[1], m.VIEWS[0]] = None
    source[m.FRAMES[3], m.VIEWS[0]] = 0
    plan = m.enumerate_plan(source)
    assert len(plan["requests"]) == 177
    rows = [r for r in plan["requests"] if r["view"] == m.VIEWS[0]]
    assert [r["status"] for r in rows[:4]] == [
        "view_unavailable",
        "view_unavailable",
        "no_mask_pairs",
        "no_mask_pairs",
    ]
    assert all(r["mask_pair_count"] == 0 for r in rows[:4])
    assert len(plan["crops"]) == 356


@pytest.mark.parametrize("fault", ["missing", "extra", "negative", "float", "bool"])
def test_counts_reject_scope_or_noninteger(fault):
    source = counts()
    key = next(iter(source))
    if fault == "missing":
        del source[key]
    elif fault == "extra":
        source[1, "extra"] = 0
    else:
        source[key] = {"negative": -1, "float": 2.0, "bool": True}[fault]
    with pytest.raises(ValueError):
        m.enumerate_plan(source)


def records():
    return [dict(index=i, start=49800 + 1660 * i, stop=49800 + 1660 * (i + 1)) for i in range(16)]


def test_exact_consumed_prefix_returns_six_original_objects():
    source = records()
    selected = m.selected_records(source)
    assert selected == source[10:16] and all(
        a is b for a, b in zip(selected, source[10:16], strict=True)
    )


@pytest.mark.parametrize("fault", ["short", "long", "index", "start", "stop"])
def test_bad_consumed_prefix_refused(fault):
    source = records()
    if fault == "short":
        source.pop()
    elif fault == "long":
        source.append(source[-1])
    else:
        source[11][fault] += 1
    with pytest.raises(ValueError):
        m.selected_records(source)


def test_crop_exact_frozen_utility_output_and_native_nonmutation():
    image = np.zeros((21, 23, 3), np.uint8)
    image[:] = [30, 90, 210]
    mask = np.zeros((21, 23), bool)
    mask[2:19, 3:20] = True
    before = image.tobytes(), mask.tobytes()
    actual, meta = m.prepare_crop(image, mask)
    expected = m.partial.prepare_mask_crop(np.ascontiguousarray(image[:, :, ::-1]), mask)
    assert before == (image.tobytes(), mask.tobytes())
    for key, array in actual.items():
        assert m.signature(array) == m.signature(expected[key]) == meta["array_signatures"][key]
        assert not np.shares_memory(array, image) and not np.shares_memory(array, mask)


@pytest.mark.parametrize("nonempty", [False, True])
def test_empty_and_tokenless_nonempty_are_preserved(nonempty):
    image, mask = np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), bool)
    if nonempty:
        mask[3, 3] = True
    arrays, meta = m.prepare_crop(image, mask)
    assert meta["eligible_token_count"] == 0 and not arrays["token_eligible"].any()
    assert meta["crop_status"] == ("ready" if nonempty else "empty_mask")
    assert (
        np.isfinite(arrays["raw_normalized_chw"]).all()
        if nonempty
        else np.isnan(arrays["raw_normalized_chw"]).all()
    )


def test_membership_handles_actual_named_and_absolute_aliases_but_refuses_conflict(tmp_path):
    path = tmp_path / "candidate.npz"
    path.write_bytes(b"not opened as candidate")
    item = dict(path=str(path), sha256=m.assets.sha(path), size=path.stat().st_size)
    seal = dict(outputs={"candidate": item, str(path): dict(item)})
    evidence = {}
    assert m.member(seal, "outputs", path, evidence) == item["sha256"]
    assert evidence == {str(path): item["sha256"]}
    seal["outputs"][str(path)]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="conflicting"):
        m.member(seal, "outputs", path, {})


def test_binding_does_not_replace_initial_sha(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"one")
    evidence = {}
    m.bind(path, m.assets.sha(path), evidence)
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="conflicting"):
        m.bind(path, m.assets.sha(path), evidence)
    with pytest.raises(ValueError, match="final evidence"):
        m.prior.final_guard(evidence)


class Capture:
    def __init__(self, image):
        self.image = image
        self.position = 0.0
        self.closed = False
        self.bad_seek = False
        self.eof = False

    def set(self, field, value):
        self.position = float(value) + int(self.bad_seek)
        return True

    def get(self, field):
        return self.position

    def read(self):
        self.position += 1
        return not self.eof, self.image.copy()


def test_absolute_native_decode_then_sha_without_chunk_relative_shift():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    capture = Capture(image)
    value = m.native.read_native_frame(capture, 66483, m.signature(image)["sha256"], shape=(4, 5))
    assert capture.position == 66484 and m.signature(value) == m.signature(image)


@pytest.mark.parametrize("fault", ["seek", "eof", "sha"])
def test_raw_native_failures_refused(fault):
    image = np.zeros((4, 5, 3), np.uint8)
    capture = Capture(image)
    capture.bad_seek = fault == "seek"
    capture.eof = fault == "eof"
    with pytest.raises(ValueError):
        m.native.read_native_frame(
            capture,
            66483,
            "0" * 64 if fault == "sha" else m.signature(image)["sha256"],
            shape=(4, 5),
        )


@pytest.mark.parametrize("available", [False, True])
def test_packed_zero_available_distinct_from_missing(available):
    packed = PackedMasks((4, 5))
    packed.append(
        66483,
        np.empty((0, 4, 5), bool),
        np.empty(0),
        np.empty((0, 4)),
        np.empty(0, int),
    )
    data = packed.arrays()
    data["view_available"][0] = available
    before = {k: m.signature(v) for k, v in data.items()}
    selected, decoded = m.native.select_packed_frame(data, 0, 66483, 0)
    assert (decoded is not None) == available and bool(selected["view_available"][0]) == available
    assert before == {k: m.signature(v) for k, v in data.items()}
    selected["frame_mask_sha256"][0] = b"0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        m.native.validate_packed_masks(selected)


def test_output_exists_refused_before_any_input_or_model(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "load_inputs", lambda *a: pytest.fail("input opened"))
    with pytest.raises(ValueError, match="new workspace"):
        m.run(tmp_path)
