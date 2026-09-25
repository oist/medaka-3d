"""Only finite wiring tests; tracker/fitter qualifications are not repeated."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import run_mask_transport200 as m


class MemoryOutput:
    def __init__(self):
        self.values = {}

    def write(self, key, value):
        assert key not in self.values
        self.values[key] = copy.deepcopy(value)


class NoFileEvidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, sha):
        self.paths[str(path)] = sha


class FakeTracker:
    def __init__(self, events, *, fault=None):
        self.events, self.fault = events, fault
        self.initialize_count = 1
        self.restore_count = self.processed_frames = 0
        self.previous_frame = None
        self._states = [object()]

    def advance(self, frame):
        self.events.append(("advance", frame.frame_idx))
        self.processed_frames += 1
        self.previous_frame = frame.frame_idx
        if self.fault == "restore":
            self.restore_count += 1
        if self.fault == "initialize":
            self.initialize_count += 1
        if self.fault == "cursor":
            self.previous_frame -= 1


def run_fake(monkeypatch, frames=(100, 101, 102), *, fault=None):
    monkeypatch.setattr(m, "FIRST", 100)
    monkeypatch.setattr(m, "COUNT", 3)
    events = []
    tracker = FakeTracker(events, fault=fault)
    kwargs = dict(
        prepare_frame=lambda f: events.append(("prepare", f.frame_idx)),
        before_frame=lambda n: events.append(("before", n)),
        after_frame=lambda n: events.append(("after", n)),
        on_frame=lambda f, s: events.append(("saved", f.frame_idx)),
        synchronize=lambda: events.append(("synchronize", None)),
    )
    return tracker, events, (SimpleNamespace(frame_idx=n) for n in frames), kwargs


def test_serial_native_prepare_advance_sync_save_order_is_exact(monkeypatch):
    tracker, events, frames, kw = run_fake(monkeypatch)
    m.run_frames(tracker, frames, **kw)
    assert events == [
        row
        for n in range(100, 103)
        for row in [
            ("prepare", n),
            ("before", n),
            ("advance", n),
            ("synchronize", None),
            ("saved", n),
            ("after", n),
        ]
    ]
    assert tracker.processed_frames == 3 and tracker.previous_frame == 102


@pytest.mark.parametrize("fault", ["restore", "initialize", "cursor"])
def test_lifecycle_drift_stops_before_saving_or_processing_next_frame(monkeypatch, fault):
    tracker, events, frames, kw = run_fake(monkeypatch, fault=fault)
    with pytest.raises(ValueError, match="lifecycle"):
        m.run_frames(tracker, frames, **kw)
    assert [x for x in events if x[0] == "advance"] == [("advance", 100)]
    assert not any(x[0] == "saved" for x in events)


@pytest.mark.parametrize("frames", [(100, 102, 101), (100, 101), (100, 101, 102, 103)])
def test_fixed_scope_rejects_reordering_missing_and_extra_frames(monkeypatch, frames):
    tracker, events, frames, kw = run_fake(monkeypatch, frames)
    with pytest.raises(ValueError):
        m.run_frames(tracker, frames, **kw)
    assert not any(x == ("advance", 103) for x in events)


def test_preparation_failure_precedes_any_fit(monkeypatch):
    tracker, events, frames, kw = run_fake(monkeypatch)

    def reject(_):
        raise ValueError("source mismatch")

    kw["prepare_frame"] = reject
    with pytest.raises(ValueError, match="source mismatch"):
        m.run_frames(tracker, frames, **kw)
    assert events == []


def test_postfit_save_check_failure_never_advances_the_next_frame(monkeypatch):
    tracker, events, frames, kw = run_fake(monkeypatch)

    def fail_after_saved(frame, seconds):
        events.append(("bad_fit_saved", frame.frame_idx))
        raise ValueError("actual fit4 mismatch")

    kw["on_frame"] = fail_after_saved
    with pytest.raises(ValueError, match="fit4"):
        m.run_frames(tracker, frames, **kw)
    assert events[-1] == ("bad_fit_saved", 100)
    assert tracker.processed_frames == 1


def test_every_frame_release_including_initial_delay_has_predeclared_horizon():
    expected = np.r_[np.full(16, m.FIRST + 15), np.repeat(np.arange(m.FIRST + 23, m.STOP, 8), 8)]
    actual = np.array([m.horizon(n) for n in range(m.FIRST, m.STOP)])
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 200 and actual[-1] == m.STOP - 1


def control_arrays(monkeypatch):
    monkeypatch.setattr(m, "COUNT", 2)
    candidate = {"frame_idx": np.array([100, 101], np.int64)}
    candidate.update({f"field{i}": np.array([0.0, np.nan]) for i in range(20)})
    core = {"history_offsets": np.array([0, 2, 4], np.int64)}
    core.update({"history_" + k: np.arange(4, dtype=np.int64) for k in m.RECORD_FIELDS})
    core.update({f"state{i}": np.array([0.0, -0.0]) for i in range(19)})
    assert len(candidate) == 21 and len(core) == 39
    return candidate, core


def test_reference_arrays_are_not_opened_before_completed_original(monkeypatch):
    candidate, core = control_arrays(monkeypatch)

    def forbidden(*args):
        raise AssertionError("reference array opened")

    monkeypatch.setattr(m, "read_arrays", forbidden)
    with pytest.raises(ValueError, match="completed Original"):
        m.original_gate(
            candidate,
            core,
            {"members": {}},
            NoFileEvidence(),
            finished=False,
            output=MemoryOutput(),
        )


@pytest.mark.parametrize("bad", [None, "history_parent", "candidate_bits", "state_bits"])
def test_control_checks_every_retained_row_and_nan_signed_zero_bytes(monkeypatch, bad):
    candidate, core = control_arrays(monkeypatch)
    old_candidate, old_core = copy.deepcopy(candidate), copy.deepcopy(core)
    if bad == "history_parent":
        old_core["history_parent"][3] += 1
    elif bad == "candidate_bits":
        old_candidate["field0"][0] = -0.0
    elif bad == "state_bits":
        old_core["state0"][1] = 0.0
    calls = []

    def read(root, members, path):
        calls.append(path)
        return old_candidate if path == "candidate.npz" else old_core

    monkeypatch.setattr(m, "read_arrays", read)
    output = MemoryOutput()
    if bad:
        with pytest.raises(ValueError, match="new ownership arms must not start"):
            m.original_gate(
                candidate, core, {"members": {}}, NoFileEvidence(), finished=True, output=output
            )
        assert output.values["original_control.json"]["passed"] is False
    else:
        result = m.original_gate(
            candidate, core, {"members": {}}, NoFileEvidence(), finished=True, output=output
        )
        assert result["passed"] is True
        assert result["candidate"]["array_count"] == 21 and result["history"]["array_count"] == 20
    assert calls == ["candidate.npz", "checkpoint/arrays.npz"]


def test_control_array_membership_mismatch_cannot_pass():
    with pytest.raises(ValueError, match="membership"):
        m.compare_array_maps({"a": np.ones(1)}, {"b": np.ones(1)}, "control")


def test_actual_fit_four_arrays_are_checked_bitwise():
    source = {
        k: np.array([[0.0, 1.0]], np.float32)
        for k in ("fit_pixels", "fit_target", "fit_weights", "fit_foreground")
    }
    for key in source:
        current = copy.deepcopy(source)
        current[key][0, 0] = -0.0
        with pytest.raises(ValueError, match=key):
            m.fit_sample_match(current, source)


def test_context_only_allows_implementation_and_explicit_transport_policy():
    old = dict(implementation="Original", tracker_config=dict(opt_steps=14), seed="fixed")
    current = copy.deepcopy(old)
    current.update(
        implementation="MaskTransportTracker",
        mask_transport=dict(maximum_frames=200, proposal_permutation_relabels_target=False),
    )
    fake = SimpleNamespace(_context=lambda: current)
    m.context_match(fake, old)
    current["tracker_config"]["opt_steps"] = 13
    with pytest.raises(ValueError, match="science context"):
        m.context_match(fake, old)


def test_t0_retained_second_anchor_and_original_nonunicode_values_are_not_skipped():
    records = [{k: np.asarray(0.0) for k in m.RECORD_FIELDS} for _ in range(2)]
    anchors = [
        SimpleNamespace(anchor_id=a, initial_native=100, links=(np.eye(3, dtype=bool),) * 3)
        for a in range(2)
    ]
    changed = copy.deepcopy(records)
    changed[1]["costs"] = np.asarray(-0.0)
    with pytest.raises(ValueError, match="t0 costs"):
        m.initial_match(changed, anchors, records, anchors)
    changed = copy.deepcopy(anchors)
    changed[1].links[0][0] = False
    with pytest.raises(ValueError, match="causal initial links"):
        m.initial_match(records, changed, records, anchors)


def point_reader_fixture(monkeypatch):
    native = m.FIRST
    reader = m.SavedPointEvidence.__new__(m.SavedPointEvidence)
    reader.members, reader.cache = {}, {}
    reader.group_counts, reader.query_sha256 = (2, 2, 2), ("a" * 64,) * 3
    reader.queries = {v: dict(mask_index=np.array([0, 0, 1], np.int64)) for v in m.VIEWS}
    xy = np.array([[0.0, 0.0], [-0.1, 0.0], [np.nan, np.nan]], np.float32)
    reader.coords = {
        (arm, v): np.repeat(xy[None], m.COUNT, axis=0)
        for arm in ("lk", "ct_first")
        for v in m.VIEWS
    }
    reader.flags = {
        (arm, v): np.ones((m.COUNT, 3), bool) for arm in ("lk", "ct_first") for v in m.VIEWS
    }
    mask = np.zeros((2, 1200, 1200), bool)
    mask[0, 0, 0] = True
    frame = SimpleNamespace(frame_idx=native, masks={v: mask for v in m.VIEWS})
    actual = m.point_core.mask_memberships(xy, mask, (1200, 1200))
    saved = dict(
        frame_idx=np.asarray(native, np.int64),
        release_native=np.asarray(m.horizon(native), np.int64),
    )
    for arm in ("lk", "ct_first"):
        saved.update({arm + "_" + k: np.asarray(v).copy() for k, v in actual.items()})
    monkeypatch.setattr(m, "read_arrays", lambda *a: saved)
    return reader, frame, saved


def test_saved_all_membership_fields_and_reported_denominators_connect_exactly(monkeypatch):
    reader, frame, _saved = point_reader_fixture(monkeypatch)
    reader.prepare(frame)
    votes = reader.provider("ct_first")(m.FIRST)
    assert len(votes) == 3
    for v in votes:
        assert v.reported_count.tolist() == [2, 1]
        assert v.single_votes.tolist() == [[2, 0], [0, 0]]
        assert v.unassigned_count.tolist() == [0, 1]
        assert not v.membership.flags.writeable
    with pytest.raises(ValueError, match="zero arm"):
        reader.provider("original")(m.FIRST)
    with pytest.raises(ValueError, match="cursor"):
        reader.provider("lk")(m.FIRST + 1)


@pytest.mark.parametrize("key", ["membership", "in_image", "pixel_xy", "available"])
def test_point_membership_drift_in_any_saved_diagnostic_field_fails(monkeypatch, key):
    reader, frame, saved = point_reader_fixture(monkeypatch)
    value = saved["lk_" + key]
    if value.dtype == bool:
        value.flat[0] = not value.flat[0]
    else:
        value.flat[0] += 1
    with pytest.raises(ValueError, match="membership"):
        reader.prepare(frame)


def test_draft_or_wrong_seal_execution_is_refused_before_any_output_directory(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    args = SimpleNamespace(
        execute=True, device="cuda", samples_seal_sha256=m.SAMPLES_SHA, output=tmp_path / "new"
    )
    with pytest.raises(ValueError, match="draft"):
        m.run(args)
    assert not args.output.exists()
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    args.samples_seal_sha256 = "e" * 64
    with pytest.raises(ValueError, match="samples"):
        m.run(args)
    assert not args.output.exists()


def test_old_named_aliases_allow_same_bytes_but_reject_conflict(tmp_path):
    row = dict(path=str(tmp_path / "candidate.npz"), sha256="a" * 64, size=20)
    members = m.member_index(dict(outputs=dict(named=row, absolute=copy.deepcopy(row))), tmp_path)
    assert len(members) == 1
    drift = {**row, "sha256": "b" * 64}
    with pytest.raises(ValueError, match="alias conflict"):
        m.member_index(dict(outputs=dict(named=row, absolute=drift)), tmp_path)


@pytest.mark.parametrize("key", ["numpy", "scipy", "opencv", "torch", "cuda", "cuda_device_name"])
def test_original_runtime_version_drift_fails_before_any_fit(key):
    versions = {
        k: "fixed" for k in ("numpy", "scipy", "opencv", "torch", "cuda", "cuda_device_name")
    }
    m.validate_runtime_versions(versions, versions.copy())
    changed = {**versions, key: "different"}
    with pytest.raises(ValueError, match="versions differ"):
        m.validate_runtime_versions(versions, changed)
