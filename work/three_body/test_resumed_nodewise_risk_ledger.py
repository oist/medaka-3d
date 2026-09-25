"""Review queue must not lose flagged native intervals or promote distance alone."""

import build_resumed_nodewise_risk_ledger as risk
import numpy as np


def test_half_open_runs_preserve_first_last_and_singletons():
    mask = np.array([True, True, False, True, False, True])
    assert risk.runs(mask) == [(0, 2), (3, 4), (5, 6)]
    assert risk.runs(np.zeros(7, bool)) == []


def test_context_merge_split_preserves_every_high_native_frame():
    plan = dict(
        high_risk_event_context_frames=166,
        merge_gap_frames=83,
        maximum_scene_frames=1992,
        long_scene_split_overlap_frames=166,
    )
    events = [
        dict(start_frame=risk.START, stop_frame_exclusive=risk.START + 1, high_risk=True),
        dict(start_frame=70000, stop_frame_exclusive=75000, high_risk=True),
        dict(start_frame=risk.STOP - 1, stop_frame_exclusive=risk.STOP, high_risk=True),
        dict(start_frame=80000, stop_frame_exclusive=90000, high_risk=False),
    ]
    windows = risk.scene_windows(events, plan)
    covered = np.zeros(risk.STOP - risk.START, bool)
    for lo, hi in windows:
        assert risk.START <= lo < hi <= risk.STOP
        assert hi - lo <= plan["maximum_scene_frames"]
        covered[lo - risk.START : hi - risk.START] = True
    for e in events[:3]:
        assert covered[e["start_frame"] - risk.START : e["stop_frame_exclusive"] - risk.START].all()
    assert not covered[80000 - risk.START : 90000 - risk.START].any()


def test_distance_alone_creates_no_review_scene():
    plan = dict(
        high_risk_event_context_frames=166,
        merge_gap_frames=83,
        maximum_scene_frames=1992,
        long_scene_split_overlap_frames=166,
    )
    assert (
        risk.scene_windows(
            [dict(start_frame=risk.START, stop_frame_exclusive=risk.STOP, high_risk=False)], plan
        )
        == []
    )
