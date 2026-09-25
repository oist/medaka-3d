"""Frozen evaluation excerpts; no imports from another lab."""

from __future__ import annotations

from collections import Counter

import numpy as np

SOURCE_BASELINE = "labs/20260917_variable_count_tracklets/baseline.py"


BASELINE_SOURCE_SHA256 = "e197f632f4ca6c024e8d9a6e0b7935f6db3543cb3bf738312099dfb54d06fecc"


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


SOURCE_RUN = "labs/20260917_variable_count_tracklets/run.py"


RUN_SOURCE_SHA256 = "d16856fa57fd9adbefc92d733b3f3445e95bd78ccdc7b6c49daa9899803f9c35"


def frame_statistics(frames, points, start, stop, span):
    native = np.arange(start, stop)
    counts = np.diff(np.searchsorted(frames, np.arange(start, stop + 1)))
    missing = np.zeros((len(native), 3), dtype=int)
    nearest = np.full(len(frames), np.inf)
    for frame in native:
        lo, hi = np.searchsorted(frames, [frame, frame + 1])
        missing[frame - start] = (~np.isfinite(points[lo:hi]).all(axis=-1)).sum(axis=0)
        xy = points[lo:hi, 1]
        distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        distance[~np.isfinite(distance)] = np.inf
        if hi > lo:
            nearest[lo:hi] = distance.min(axis=1)
    proximity = nearest < 0.08 * span
    return native, counts, missing, nearest, proximity


def measure(frames, points, labels, core, start, stop, span, events):
    previous, gaps = predecessors(frames, labels)
    eligible = core & (frames > start)
    adjacent = gaps == 1
    # No edge into core from context is counted in interval-only metrics.
    links = adjacent & eligible
    native, counts, missing, nearest, proximity = frame_statistics(
        frames, points, start, stop, span
    )
    observed = np.isfinite(points).all(axis=-1)
    node_missing = ~observed.all(axis=1)
    rows_per_frame = dict(zip(native.tolist(), counts.tolist(), strict=True))
    row_count = np.array([rows_per_frame.get(int(f), 0) for f in frames])
    ids, id_counts = np.unique(labels[core & (labels >= 0)], return_counts=True)
    steps = np.full(len(frames), np.nan)
    good = previous >= 0
    steps[good] = np.linalg.norm(points[good, 1] - points[previous[good], 1], axis=-1)
    strata = []
    for count in ("0", "1", "2", "3", "4+"):
        count_mask = row_count >= 4 if count == "4+" else row_count == int(count)
        frame_count = (
            int((counts >= 4).sum()) if count == "4+" else int((counts == int(count)).sum())
        )
        for near in (False, True):
            for absent in (False, True):
                mask = eligible & count_mask & (proximity == near) & (node_missing == absent)
                n = int(mask.sum())
                accepted = int((links & mask).sum())
                strata.append(
                    dict(
                        detection_count=count,
                        proximity="near" if near else "separated",
                        any_node_missing=absent,
                        frame_count_count_stratum=frame_count,
                        eligible_observations=n,
                        adjacent_connections=accepted,
                        unconnected=n - accepted,
                        connection_rate=accepted / n if n else None,
                    )
                )
    denominator = int(eligible.sum())
    nlinks = int(links.sum())
    return dict(
        core_frames=stop - start,
        core_rows=int(core.sum()),
        retained_rows=int(core.sum()),
        retention_rate=1.0,
        context_rows=len(frames),
        frame_start=start,
        frame_stop=stop,
        eligible_observations=denominator,
        adjacent_connections=nlinks,
        adjacent_links=nlinks,
        unconnected_observations=denominator - nlinks,
        connection_rate=nlinks / denominator if denominator else None,
        abstention_rate=(denominator - nlinks) / denominator if denominator else None,
        contiguous_runs=int(core.sum()) - nlinks,
        unique_local_ids=len(ids),
        isolated_core_ids=int((id_counts == 1).sum()),
        gap_connections=int((eligible & (gaps > 1)).sum()),
        unassigned_rows=int((core & (labels < 0)).sum()),
        large_adjacent_steps=int((links & (steps > 0.3 * span)).sum()),
        maximum_adjacent_step_px=float(np.nanmax(steps[links])) if links.any() else None,
        missing_m=int(missing[:, 0].sum()),
        missing_p=int(missing[:, 1].sum()),
        missing_t=int(missing[:, 2].sum()),
        detection_count_frames={str(k): int((counts == k).sum()) for k in (0, 1, 2, 3)}
        | {"4+": int((counts >= 4).sum())},
        refusal_reasons=dict(
            Counter(str(e.get("reason")) for e in events if e.get("reason") is not None)
        ),
        refusal_event_scope=("both directional passes over full context; may repeat one endpoint"),
        strata=strata,
        identity_accuracy=None,
        verified_correct_connections=None,
        verified_wrong_connections=None,
        verified_identity_denominator=0,
        truth_status="unknown; no new human labels inferred from smoothness or old 3D slots",
    ), dict(
        native_frame=native,
        native_seconds=native / 166,
        detection_count=counts,
        missing_m=missing[:, 0],
        missing_p=missing[:, 1],
        missing_t=missing[:, 2],
        nearest_other_pectoral_px=nearest,
        close_rows=proximity,
        adjacent_connections_per_frame=np.bincount(frames[links] - start, minlength=stop - start),
        unconnected_per_frame=np.bincount(
            frames[eligible & ~adjacent] - start, minlength=stop - start
        ),
    )
