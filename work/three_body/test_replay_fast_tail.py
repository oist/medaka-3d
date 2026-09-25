"""Synthetic qualification and three-frame replay; no files, GPU, raw data or fit."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import replay_fast_tail as m

FRAMES = [66524, *range(66620, 66920)]
CPU_FRAMES = [66524, 66628, 66703, 66907]


def qualification():
    summary = dict(
        schema="fast_directional_fitter_paired_301_v2",
        status="ok",
        passed=True,
        frames=FRAMES.copy(),
        fitted_calls=602,
        fit_calls_attempted=602,
        fit_calls_returned=602,
        validated_fit_calls=602,
        all77_arrays_per_arm_pass=True,
        paired_all77_pass=True,
        CPU_reference_frames=CPU_FRAMES.copy(),
        initialize_calls=0,
        restore_calls=0,
        advance_calls=0,
        new_tracking=False,
    )

    def exact():
        return {
            f"field_{i:02d}": dict(byte_exact=True, maximum_absolute_difference=0.0)
            for i in range(77)
        }

    parities = {}
    for native in FRAMES:
        row = {arm: dict(CUDA_golden=exact()) for arm in ("original", "fast")}
        row["paired"] = exact()
        if native in CPU_FRAMES:
            for arm in ("original", "fast"):
                row[arm]["CPU_reference"] = exact()
                row[arm]["CPU_reference"]["field_00"].update(
                    byte_exact=False, maximum_absolute_difference=1e-10
                )
        parities[native] = row
    return summary, parities


def args(**changes):
    return SimpleNamespace(
        **(dict(owned_session=77700, owned_exit_code=0, reference_child_pid=1234) | changes)
    )


def test_execution_args_are_checked_without_filesystem_access(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("execution argument validation must precede filesystem work")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    m.validate_execution_args(args())


@pytest.mark.parametrize(
    "field,value",
    [
        ("owned_session", 77701),
        ("owned_session", "77700"),
        ("owned_session", np.int64(77700)),
        ("owned_session", True),
        ("owned_exit_code", 1),
        ("owned_exit_code", False),
        ("owned_exit_code", np.int64(0)),
        ("reference_child_pid", 0),
        ("reference_child_pid", -1),
        ("reference_child_pid", True),
        ("reference_child_pid", np.int64(1234)),
        ("reference_child_pid", "1234"),
    ],
)
def test_execution_args_reject_type_and_value_drift(field, value):
    with pytest.raises(ValueError):
        m.validate_execution_args(args(**{field: value}))


def test_qualification_accepts_all301_all77_and_tolerant_cpu4_without_mutation():
    summary, parities = qualification()
    before = copy.deepcopy((summary, parities))
    m.validate_qualification(summary, parities)
    assert (summary, parities) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "fast_directional_fitter_paired_301_v1"),
        ("status", "failed"),
        ("passed", False),
        ("passed", 1),
        ("frames", FRAMES[:-1]),
        ("frames", list(reversed(FRAMES))),
        ("fitted_calls", 601),
        ("fit_calls_attempted", 603),
        ("fit_calls_returned", 601),
        ("validated_fit_calls", 601),
        ("all77_arrays_per_arm_pass", False),
        ("paired_all77_pass", False),
        ("CPU_reference_frames", CPU_FRAMES[:-1]),
        ("initialize_calls", 1),
        ("restore_calls", 1),
        ("advance_calls", 1),
        ("new_tracking", True),
    ],
)
def test_qualification_rejects_incomplete_or_changed_summary(field, value):
    summary, parities = qualification()
    summary[field] = value
    with pytest.raises(ValueError):
        m.validate_qualification(summary, parities)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_frame",
        "extra_frame",
        "string_frame",
        "missing_arm",
        "missing_paired",
        "only76",
        "wrong_keys",
        "not_exact",
        "nonzero",
        "nan",
        "infinity",
        "missing_cpu",
        "cpu_over",
        "cpu_nan",
        "cpu_wrong_keys",
    ],
)
def test_qualification_rejects_any_single_parity_drift(fault):
    summary, parities = qualification()
    native = FRAMES[-1]
    row = parities[native]
    if fault == "missing_frame":
        del parities[native]
    elif fault == "extra_frame":
        parities[native + 1] = copy.deepcopy(row)
    elif fault == "string_frame":
        parities[str(native)] = parities.pop(native)
    elif fault == "missing_arm":
        del row["fast"]
    elif fault == "missing_paired":
        del row["paired"]
    elif fault == "only76":
        del row["fast"]["CUDA_golden"]["field_76"]
    elif fault == "wrong_keys":
        row["paired"]["other"] = row["paired"].pop("field_76")
    elif fault in ("not_exact", "nonzero", "nan", "infinity"):
        field = row["fast"]["CUDA_golden"]["field_76"]
        if fault == "not_exact":
            field["byte_exact"] = False
        else:
            field["maximum_absolute_difference"] = {
                "nonzero": 1e-16,
                "nan": float("nan"),
                "infinity": float("inf"),
            }[fault]
    else:
        row = parities[CPU_FRAMES[-1]]["original"]
        if fault == "missing_cpu":
            del row["CPU_reference"]
        elif fault == "cpu_wrong_keys":
            row["CPU_reference"]["other"] = row["CPU_reference"].pop("field_76")
        else:
            row["CPU_reference"]["field_76"]["maximum_absolute_difference"] = (
                1.00001e-10 if fault == "cpu_over" else float("nan")
            )
    with pytest.raises((ValueError, KeyError)):
        m.validate_qualification(summary, parities)


def record(native, rank):
    return dict(
        frame_idx=native,
        pose_xyz=np.full((3, 3, 3), native + rank / 10),
        observed=np.ones((3, 3), bool),
        sources=np.full((3, 3, 3), rank, int),
        identity_review=np.array([False, True, True]),
        mask_supported=np.zeros(3, bool),
        body_supported=np.ones(3, bool),
        missing=np.zeros(3, int),
        costs=float(rank),
        path_costs=float(native + rank),
        loss_parts=np.arange(5, dtype=float),
        score_margin=9.0,
        parent=rank,
        proposal_kind="predict",
        hypotheses=24,
        proposal_flow_nodes=np.zeros((3, 3), bool),
        proposal_flow_parent=0,
        node_missing=np.zeros((3, 3), int),
        direct_pectoral_age=np.full(3, 4, int),
    )


@pytest.fixture
def replay_fixture(monkeypatch):
    monkeypatch.setattr(m, "START", 1002)
    monkeypatch.setattr(m, "STOP", 1005)
    monkeypatch.setattr(m.original, "FIRST", 1000)
    history = [[record(native, rank) for rank in range(6)] for native in range(1000, 1005)]
    flat = [row for frame in history for row in frame]
    expected = {"history_offsets": np.arange(0, 31, 6)}
    expected.update(
        {"history_" + key: np.asarray([r[key] for r in flat]) for key in m.original.RECORD_FIELDS}
    )
    events, rows = [], []
    clock = [0.0]
    monkeypatch.setattr(m.time, "perf_counter", lambda: clock[0])

    class Tracker:
        initialize_count = 1
        restore_count = 4
        processed_frames = 2
        previous_frame = 1001

        def __init__(self):
            self._history = copy.deepcopy(history[:2])
            self._states = [object() for _ in range(6)]
            self.fault = None

        def advance(self, frame):
            events.append(("advance", frame.frame_idx))
            clock[0] += 3
            records = copy.deepcopy(history[frame.frame_idx - 1000])
            if self.fault is not None:
                self.fault(self, records)
            self._history.append(records)
            self.processed_frames += 1
            self.previous_frame = frame.frame_idx

    def before(native):
        events.append(("before", native))
        clock[0] += 100

    def after(native):
        events.append(("after", native))
        clock[0] += 200

    def sync():
        events.append(("sync", None))
        clock[0] += 2

    def result(row):
        events.append(("result", row["frame_idx"]))
        rows.append(copy.deepcopy(row))
        clock[0] += 300

    return SimpleNamespace(
        tracker=Tracker(),
        expected=expected,
        events=events,
        rows=rows,
        frames=[SimpleNamespace(frame_idx=n) for n in range(1002, 1005)],
        hooks=dict(before_frame=before, after_frame=after, on_result=result, synchronize=sync),
    )


def test_serial_replay_calls_each_frame_once_and_compares_all6_all19(replay_fixture):
    f = replay_fixture
    before = copy.deepcopy(f.expected)
    m.run_frames(f.tracker, iter(f.frames), f.expected, **f.hooks)
    assert [row["frame_idx"] for row in f.rows] == [1002, 1003, 1004]
    assert all(row["exact"] and row["all_beams"] == 6 for row in f.rows)
    assert all(row["compared_fields"] == list(m.original.RECORD_FIELDS) for row in f.rows)
    assert all(row["advance_seconds"] == 5.0 for row in f.rows)
    assert len(m.original.RECORD_FIELDS) == 19
    assert [x for x in f.events if x[0] == "advance"] == [("advance", n) for n in range(1002, 1005)]
    assert sum(x[0] == "sync" for x in f.events) == 3
    assert f.events == [
        item
        for native in range(1002, 1005)
        for item in (
            ("before", native),
            ("advance", native),
            ("sync", None),
            ("after", native),
            ("result", native),
        )
    ]
    assert f.tracker.initialize_count == 1 and f.tracker.restore_count == 4
    assert f.tracker.processed_frames == 5 and f.tracker.previous_frame == 1004
    for key in before:
        assert before[key].tobytes() == f.expected[key].tobytes()


@pytest.mark.parametrize(
    "fault", ["pose_xyz", "parent", "observed", "proposal_kind", "beam_order", "beam_count"]
)
def test_history_mismatch_is_saved_before_fail_and_no_next_frame(replay_fixture, fault):
    f = replay_fixture

    def mutate(tracker, rows):
        if fault == "pose_xyz":
            rows[5][fault].flat[0] = np.nextafter(rows[5][fault].flat[0], np.inf)
        elif fault == "parent":
            rows[5][fault] += 1
        elif fault == "observed":
            rows[5][fault][2, 2] = False
        elif fault == "proposal_kind":
            rows[5][fault] = "tail_recovery"
        elif fault == "beam_order":
            rows.reverse()
        else:
            rows.pop()

    f.tracker.fault = mutate
    with pytest.raises(ValueError):
        m.run_frames(f.tracker, iter(f.frames), f.expected, **f.hooks)
    assert len(f.rows) == 1 and not f.rows[0]["exact"]
    assert f.rows[0]["differing_fields"]
    assert [x for x in f.events if x[0] == "advance"] == [("advance", 1002)]


@pytest.mark.parametrize(
    "counter", ["initialize_count", "restore_count", "processed_frames", "previous_frame"]
)
def test_invalid_entry_counters_reject_before_advance(replay_fixture, counter):
    f = replay_fixture
    setattr(f.tracker, counter, getattr(f.tracker, counter) + 1)
    with pytest.raises(ValueError):
        m.run_frames(f.tracker, iter(f.frames), f.expected, **f.hooks)
    assert not any(x[0] == "advance" for x in f.events)


@pytest.mark.parametrize("counter", ["initialize_count", "restore_count", "processed_frames"])
def test_reset_or_restore_during_replay_is_rejected(replay_fixture, counter):
    f = replay_fixture

    def mutate(tracker, records):
        setattr(tracker, counter, getattr(tracker, counter) + 1)

    f.tracker.fault = mutate
    with pytest.raises(ValueError):
        m.run_frames(f.tracker, iter(f.frames), f.expected, **f.hooks)
    assert [x for x in f.events if x[0] == "advance"] == [("advance", 1002)]


@pytest.mark.parametrize(
    "native", [[1003, 1002, 1004], [1002, 1002, 1004], [1002, 1004], [1002, 1003]]
)
def test_reordered_duplicate_skipped_or_truncated_input_fails(replay_fixture, native):
    f = replay_fixture
    frames = (SimpleNamespace(frame_idx=n) for n in native)
    with pytest.raises((ValueError, StopIteration)):
        m.run_frames(f.tracker, frames, f.expected, **f.hooks)


def test_extra_frame_does_not_advance_past_registered_stop(replay_fixture):
    f = replay_fixture
    frames = iter([*f.frames, SimpleNamespace(frame_idx=1005)])
    with pytest.raises(ValueError):
        m.run_frames(f.tracker, frames, f.expected, **f.hooks)
    assert [r["frame_idx"] for r in f.rows] == [1002, 1003, 1004]
    assert [n for kind, n in f.events if kind == "advance"] == [1002, 1003, 1004]


def test_wrong_active_beam_count_fails_before_step(replay_fixture):
    f = replay_fixture
    f.tracker._states.pop()
    with pytest.raises(ValueError):
        m.run_frames(f.tracker, iter(f.frames), f.expected, **f.hooks)
    assert not f.events


def prefix_fixture(replay_fixture):
    full = copy.deepcopy(replay_fixture.expected)
    parent = {
        key: value[: 3 if key == "history_offsets" else 12].copy() for key, value in full.items()
    }
    return full, parent


def test_prefix_comparison_keeps_all19_and_allows_only_future_suffix(replay_fixture):
    full, parent = prefix_fixture(replay_fixture)
    m.verify_prefix(full, parent)
    full["history_pose_xyz"][12:] += 999
    m.verify_prefix(full, parent)
    full["history_proposal_kind"] = full["history_proposal_kind"].astype("U64")
    m.verify_prefix(full, parent)


@pytest.mark.parametrize("fault", ["offset", "node", "dtype", "label", "negative_zero"])
def test_prefix_rejects_one_original_value_or_dtype_change(replay_fixture, fault):
    full, parent = prefix_fixture(replay_fixture)
    if fault == "offset":
        full["history_offsets"][1] += 1
    elif fault == "node":
        full["history_pose_xyz"][11, 2, 2, 2] += 0.001
    elif fault == "dtype":
        full["history_sources"] = full["history_sources"].astype(np.int32)
    elif fault == "label":
        full["history_proposal_kind"][11] = "other"
    else:
        parent["history_pose_xyz"][0, 0, 0, 0] = 0.0
        full["history_pose_xyz"][0, 0, 0, 0] = -0.0
    with pytest.raises(ValueError):
        m.verify_prefix(full, parent)
