"""Cropped current SLEAP controls; only readout (S1) and OKS scale (S2) differ.

The replay loop is adapted from the frozen 20260917 clean2d c27_replay.py.
No historical full-session reproduction is claimed. S2 temporarily supplies a
positive area to the installed scorer; every assignment/history setting stays
unchanged and the installed source files are never edited.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def fixed_area(view):
    if view not in ("top_cam", "side1_cam", "side2_cam"):
        raise ValueError(view)
    return (120.0 if view == "top_cam" else 100.0) ** 2


@contextmanager
def oks_area_override(area):
    import sleap_nn.tracking.tracker as installed

    original = installed.compute_oks
    if area is not None:
        if not np.isfinite(area) or area <= 0:
            raise ValueError("OKS area must be positive and finite")

        def fixed_oks(points_gt, points_pr, **kwargs):
            return original(points_gt, points_pr, **{**kwargs, "scale": float(area)})

        installed.compute_oks = fixed_oks
    try:
        yield
    finally:
        installed.compute_oks = original


def replay(frame_idx, points, score, start, stop, *, view, fixed_scale=False):
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker.from_config(
        window_size=5,
        candidates_method="local_queues",
        features="keypoints",
        scoring_method="oks",
        scoring_reduction="mean",
        track_matching_method="hungarian",
        max_tracks=None,
        tracking_target_instance_count=None,
        min_match_points=0,
    )
    skeleton = sio.Skeleton(["mouth", "pectoral"])
    legacy = np.full(len(frame_idx), -1, dtype=np.int64)
    labels = legacy.copy()
    names, reordered = {}, []
    with oks_area_override(fixed_area(view) if fixed_scale else None):
        for frame in range(start, stop):
            lo, hi = np.searchsorted(frame_idx, [frame, frame + 1])
            if lo == hi:
                continue
            batch = [
                sio.PredictedInstance.from_numpy(
                    points_data=points[row, :2].astype(float),
                    skeleton=skeleton,
                    point_scores=np.ones(2),
                    score=float(score[row]),
                )
                for row in range(lo, hi)
            ]
            # Identity, not return position, defines correspondence to C01 rows.
            source_row = {id(instance): row for row, instance in enumerate(batch)}
            returned = tracker.track(batch, frame_idx=frame)
            order = []
            for offset, instance in enumerate(returned):
                row = source_row[id(instance)]
                order.append(row)
                if instance.track is None:
                    continue
                name = instance.track.name
                if name not in names:
                    names[name] = len(names)
                labels[lo + row] = names[name]
                legacy[lo + offset] = names[name]
            if order != list(range(len(batch))):
                reordered.append(dict(frame=frame, returned_input_row_order=order))
    metadata = dict(
        condition="S2" if fixed_scale else "S1",
        tracker_type=type(tracker).__name__,
        candidate_type=type(tracker.candidate).__name__,
        window_size=tracker.candidate.window_size,
        min_match_points=tracker.min_match_points,
        min_new_track_points=tracker.candidate.min_new_track_points,
        max_tracks=tracker.candidate.max_tracks,
        scoring_method=tracker.scoring_method,
        scoring_reduction=tracker.scoring_reduction,
        oks_stddev=tracker.oks_stddev,
        use_flow=tracker.use_flow,
        fixed_oks_area_px2=fixed_area(view) if fixed_scale else None,
        oks_denominator="2 * (2 * sigma)^2 * (area + eps)",
        row_mapping="id(returned_instance) -> original input offset",
        node_scores="synthetic ones required by SLEAP adapter; never saved as measured scores",
        historical_reproduction_claim=False,
        same_context_past_only=True,
        reordered_returns=reordered,
        legacy_readout_different_rows=int(np.count_nonzero(labels != legacy)),
    )
    return labels, legacy, metadata


def predecessors(frame_idx, labels):
    previous = np.full(len(frame_idx), -1, dtype=np.int64)
    last = {}
    for row, (frame, label) in enumerate(zip(frame_idx, labels, strict=True)):
        if label < 0:
            continue
        before = last.get(int(label))
        if before is not None:
            if frame_idx[before] >= frame:
                raise AssertionError("An observation cannot share an ID within one frame")
            previous[row] = before
        last[int(label)] = row
    gaps = np.zeros(len(frame_idx), dtype=np.int64)
    valid = previous >= 0
    gaps[valid] = frame_idx[valid] - frame_idx[previous[valid]]
    return previous, gaps
