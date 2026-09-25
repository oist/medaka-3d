"""Focused CPU connection tests; no saved input, decoder, model or GPU execution."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import run_fast_tail_two_minutes as m


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    assert not m.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU forbidden in connection fixtures")

    for name in ("set_per_process_memory_fraction", "synchronize", "is_available"):
        monkeypatch.setattr(m.torch.cuda, name, forbidden)
    monkeypatch.setattr(m.qualified.qualification, "gpu_sample", forbidden)
    yield
    assert not m.torch.cuda.is_initialized()


class FakeTracker:
    initialize_count = 1
    restore_count = 4

    def __init__(self):
        self.previous_frame = m.START - 1
        self.processed_frames = m.START - m.FIRST
        self._states = [None] * 6
        self._history = []

    def advance(self, frame):
        assert frame.frame_idx == self.previous_frame + 1
        self.previous_frame = frame.frame_idx
        self.processed_frames += 1
        self._history.append([frame.frame_idx] * 6)

    def finish(self):
        raise AssertionError("intermediate finish forbidden")

    def restore_state(self, *args):
        raise AssertionError("intermediate restore forbidden")


def harness(monkeypatch, *, fail_control=None, fail_checkpoint=None, wrong_native=None):
    tracker, events, rows, saved = FakeTracker(), [], [], []

    def reader_for(first, stop):
        events.append(("reader", first))
        return SimpleNamespace(frames=np.arange(first, stop, dtype=np.int64))

    def bounded(reader, *, first, stop):
        try:
            for native in range(first, stop):
                events.append(("decode", native))
                yield SimpleNamespace(frame_idx=native if native != wrong_native else native + 1)
        finally:
            events.append(("close", stop))

    def compare(records, arrays, native):
        assert records == [native] * 6
        assert native < arrays["stop"] <= m.CONTROL_STOP
        return dict(frame_idx=native, exact=native != fail_control, all_beams=6)

    def checkpoint(stop, expected):
        events.append(("save", stop))
        saved.append(stop)
        assert tracker.previous_frame == stop - 1
        assert (expected is None) == (stop > m.CONTROL_STOP)
        if stop == fail_checkpoint:
            raise ValueError("checkpoint control mismatch saved")

    monkeypatch.setattr(m.original, "bounded_frames", bounded)
    monkeypatch.setattr(m.original, "compare_records", compare)
    kwargs = dict(
        reader_for=reader_for,
        reference_for=lambda stop: dict(arrays={"stop": stop}) if stop <= m.CONTROL_STOP else None,
        save_checkpoint=checkpoint,
        before_frame=lambda n: events.append(("before", n)),
        after_frame=lambda n: events.append(("after", n)),
        on_frame=lambda row: rows.append(row),
        synchronize=lambda: None,
        timers=m.new_timers(),
    )
    return tracker, events, rows, saved, kwargs


def test_all_19920_ordered_frames_twelve_saves_no_finish_or_restore(monkeypatch):
    tracker, events, rows, saved, kwargs = harness(monkeypatch)
    m.drive_chunks(tracker, **kwargs)
    assert [n for event, n in events if event == "decode"] == list(range(m.START, m.STOP))
    assert saved == [
        68060,
        69720,
        71380,
        73040,
        74700,
        76360,
        78020,
        79680,
        81340,
        83000,
        84660,
        86320,
    ]
    assert [n for event, n in events if event == "close"] == saved
    assert len(rows) == 19920
    assert sum(row["control"] is not None for row in rows) == 9960
    assert tracker.processed_frames == 36520
    assert tracker.previous_frame == 86319
    assert tracker.initialize_count == 1 and tracker.restore_count == 4
    assert all(value >= 0 for value in kwargs["timers"].values())


def test_checkpoint_mismatch_stops_before_next_chunk(monkeypatch):
    tracker, events, rows, saved, kwargs = harness(monkeypatch, fail_checkpoint=68060)
    with pytest.raises(ValueError, match="checkpoint control"):
        m.drive_chunks(tracker, **kwargs)
    assert len(rows) == 1660 and saved == [68060]
    assert ("reader", 68060) not in events
    assert events[-1] == ("save", 68060)


def test_native_mismatch_is_reported_before_stop_and_reader_closed(monkeypatch):
    tracker, events, rows, saved, kwargs = harness(monkeypatch, fail_control=66403)
    with pytest.raises(ValueError, match="history diverged"):
        m.drive_chunks(tracker, **kwargs)
    assert [r["frame_idx"] for r in rows] == [66400, 66401, 66402, 66403]
    assert rows[-1]["control"]["exact"] is False
    assert not saved and ("decode", 66404) not in events
    assert events[-1] == ("close", 68060)


def test_resource_failure_retains_returned_frame_and_stops(monkeypatch):
    tracker, events, rows, saved, kwargs = harness(monkeypatch)

    def resource_failure(native):
        raise ValueError("resource exhausted")

    kwargs["after_frame"] = resource_failure
    with pytest.raises(ValueError, match="resource exhausted"):
        m.drive_chunks(tracker, **kwargs)
    assert len(rows) == 1 and tracker.previous_frame == m.START
    assert not saved and events[-1] == ("close", 68060)


def test_wrong_decoded_native_never_advances(monkeypatch):
    tracker, _events, rows, saved, kwargs = harness(monkeypatch, wrong_native=m.START)
    with pytest.raises(ValueError, match="native read order"):
        m.drive_chunks(tracker, **kwargs)
    assert not rows and not tracker._history and not saved


@pytest.mark.parametrize("field,value", [("restore_count", 5), ("initialize_count", 2)])
def test_lifecycle_reset_is_rejected_before_decode(monkeypatch, field, value):
    tracker, events, rows, saved, kwargs = harness(monkeypatch)
    setattr(tracker, field, value)
    with pytest.raises(ValueError, match="lifecycle"):
        m.drive_chunks(tracker, **kwargs)
    assert not any(event == "decode" for event, native in events)
    assert not rows and not saved


def control_fixture():
    arrays = {"history_offsets": np.array([0, 6], np.int64)}
    arrays.update({"history_" + k: np.zeros(6, np.int64) for k in m.qualified.RECORD_FIELDS})
    arrays.update({"state_" + str(i): np.array([1.0], np.float64) for i in range(19)})
    arrays["history_proposal_kind"] = np.array(["flow"] * 6, dtype="U4")
    assert len(arrays) == 39
    tail = dict(implementation="Tail", other_science={"threshold": 12.0})
    fast = dict(
        implementation="Fast",
        other_science={"threshold": 12.0},
        fast_execution_policy={"enabled": True},
    )
    old_meta = dict(context=tail, initialize_count=1, restore_count=4, review=True)
    new_meta = dict(context=fast, initialize_count=1, restore_count=4, review=True)
    expected = dict(arrays=copy.deepcopy(arrays), metadata=old_meta)
    return arrays, new_meta, expected, dict(new_context=fast, tail_context=tail)


def test_checkpoint_all39_and_unicode_capacity_rule_without_mutation():
    arrays, metadata, expected, migration = control_fixture()
    arrays["history_proposal_kind"] = arrays["history_proposal_kind"].astype("U30")
    before = copy.deepcopy(metadata)
    row = m.compare_checkpoint(arrays, metadata, expected, migration)
    assert row["exact"] and len(row["all39"]) == 39
    assert row["all39"]["history_proposal_kind"]["unicode_value_comparison"]
    assert metadata == before


@pytest.mark.parametrize(
    "change", ["ulp", "zero_sign", "review", "science", "missing_field", "extra_field", "label"]
)
def test_checkpoint_rejects_undeclared_difference(change):
    arrays, metadata, expected, migration = control_fixture()
    if change == "ulp":
        arrays["state_0"][0] = np.nextafter(1.0, 2.0)
    elif change == "zero_sign":
        expected["arrays"]["state_0"][0] = 0.0
        arrays["state_0"][0] = -0.0
    elif change == "review":
        metadata["review"] = False
    elif change == "science":
        metadata = copy.deepcopy(metadata)
        metadata["context"]["other_science"]["threshold"] = 13.0
    elif change == "missing_field":
        arrays.pop("state_0")
    elif change == "extra_field":
        arrays["extra"] = np.array([1])
    else:
        arrays["history_proposal_kind"][0] = "lost"
    assert not m.compare_checkpoint(arrays, metadata, expected, migration)["exact"]


def test_candidate_gate_requires_same_horizon_and_all21_bytes():
    candidate = {"frame_idx": np.arange(m.FIRST, m.CONTROL_STOP, dtype=np.int64)}
    candidate.update({str(i): np.array([0.0], np.float64) for i in range(20)})
    expected = copy.deepcopy(candidate)
    assert m.compare_candidate(candidate, expected, m.CONTROL_STOP)["exact"]
    assert not m.compare_candidate(candidate, expected, m.STOP)["exact"]
    candidate["0"][0] = -0.0
    assert not m.compare_candidate(candidate, expected, m.CONTROL_STOP)["exact"]


def note_fixture():
    return dict(
        schema="root_shared_GPU_load_observation_v1",
        observer="ROOT",
        observed_at="2026-09-10T18:00:00+09:00",
        natural_phase_transitions_allowed=True,
        workloads=[
            dict(role=role, owned_session=i + 1, status="LIVE", pid=None)
            for i, role in enumerate(("long_reference", "D2_producer", "held_exit_witness"))
        ],
    )


def test_load_note_is_an_owned_copy_and_can_record_natural_exit():
    note = note_fixture()
    note["workloads"][1]["status"] = "CLOSED0"
    saved = m.shared_load_note(note)
    note["workloads"][0]["owned_session"] = 999
    assert saved["workloads"][0]["owned_session"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("observer", "inferred"), ("natural_phase_transitions_allowed", False), ("workloads", [])],
)
def test_invalid_shared_load_metadata_refused(field, value):
    note = note_fixture()
    note[field] = value
    with pytest.raises(ValueError):
        m.shared_load_note(note)


def test_runtime_preserves_qualified_binding_and_rejects_interop(monkeypatch):
    seen = []
    monkeypatch.setattr(
        m.original, "runtime", lambda binding, gpu: seen.append((binding, gpu)) or {}
    )
    monkeypatch.setattr(m.torch, "get_num_interop_threads", lambda: 2)
    with pytest.raises(ValueError, match="interop1"):
        m.runtime({"unchanged": 1}, gpu=False)
    assert seen == [({"unchanged": 1}, False)]


def test_consumed_member_hash_and_alias_conflict(tmp_path, monkeypatch):
    payload = tmp_path / "small.json"
    payload.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    item = dict(path=str(payload), size=2, sha256=digest)
    seal = dict(status="ok", stage="fixture", outputs={"named": item, str(payload): item})
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(seal))
    evidence = m.original.phase.Digests()
    bundle = m.read_seal(
        tmp_path, "seal.json", "fixture", hashlib.sha256(path.read_bytes()).hexdigest(), evidence
    )
    assert str(payload) not in evidence.paths
    assert m.consumed(bundle, "small.json", evidence) == payload
    assert evidence.paths[str(payload)] == digest
    seal["outputs"][str(payload)] = {**item, "sha256": "0" * 64}
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="conflicting"):
        m.read_seal(
            tmp_path,
            "seal.json",
            "fixture",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            m.original.phase.Digests(),
        )


def test_source_contract_rejects_new_precision_source():
    code = {str(m.WORK / name): digest for name, digest in m.SOURCE_PINS.items()}
    code[str(m.PLAN)] = m.PLAN_SHA
    code[str(m.Path(m.__file__).resolve())] = "0" * 64
    m.source_contract(code)
    code[str(m.WORK / "fast_directional_device_fitter.py")] = "0" * 64
    with pytest.raises(ValueError, match="qualified source"):
        m.source_contract(code)


def test_cuda_global_bytes_to_MiB_preserves_boundary():
    sample = m.cuda_memory_record(2048 * 1024**2, 16 * 1024**3)
    assert sample["free_MiB"] == 2048
    assert sample["total_MiB"] == 16384
    assert sample["measurement_api"] == "torch.cuda.mem_get_info(0)"
    m.qualified.check_headroom(sample)
    with pytest.raises(ValueError, match="2048"):
        m.qualified.check_headroom(m.cuda_memory_record(2048 * 1024**2 - 1, 16 * 1024**3))


@pytest.mark.parametrize("free,total", [(-1, 5), (6, 5), (False, 5), (1.5, 5), (0, 0)])
def test_invalid_cuda_global_reading_rejected(free, total):
    with pytest.raises(ValueError, match="global memory"):
        m.cuda_memory_record(free, total)
