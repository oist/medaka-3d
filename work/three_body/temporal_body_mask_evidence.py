"""Pure conditional body/mask arithmetic; no images, model or identity decisions."""

from __future__ import annotations

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def initial_link_matrix(reference, count):
    """Keep every initial mask/body cell; consume the frozen geometry decision."""
    require(type(count) is int and count >= 0, "nonnegative mask count required")
    require(len(reference["slots"]) == 3, "all three bodies required")
    links = np.zeros((count, 3), bool)
    for b, row in enumerate(reference["slots"]):
        require(row["slot"] == b, "body order differs")
        r = row["assigned_mask_row"]
        require(type(r) is int and -1 <= r < count, "initial mask row out of range")
        require(type(row["geometry_eligible"]) is bool, "explicit geometry flag required")
        if row["geometry_eligible"]:
            require(r >= 0 and not row["reasons"], "eligible initial link inconsistent")
            links[r, b] = True
    require(np.all(links.sum(axis=1) <= 1), "one initial group linked to competing bodies")
    return links


def score_votes(coverage, complete, membership, selected, linked, collision, *, available):
    """sum(single-vote count / ALL selected points * body centerline coverage).

    Multi/unassigned/nonfinite reported points are never silently renormalized away.
    A score is descriptive even when one point supplies it. Candidate availability
    additionally needs >=2 single members, strict initial linkage, all3 finite
    scores, a positive exact unique maximum and no contributing-mask competitor.
    """
    require(type(available) is bool, "explicit mask availability required")
    require(coverage.ndim == 2 and coverage.shape[1] == 3, "coverage must be N,3")
    n = len(coverage)
    require(coverage.dtype.kind == "f", "float coverage required")
    require(complete.shape == linked.shape == (3,), "three body flags required")
    require(complete.dtype == linked.dtype == bool, "boolean body flags required")
    require(collision.shape == (n, 3) and collision.dtype == bool, "collision schema")
    require(membership.ndim == 2 and membership.shape[1] == n, "all-N membership required")
    require(membership.dtype == bool, "boolean membership required")
    require(selected.shape == (len(membership),) and selected.dtype == bool, "selection schema")
    require(available or n == 0, "unavailable masks cannot carry rows")
    require(np.isfinite(coverage[:, complete]).all(), "complete projection coverage nonfinite")
    finite = coverage[np.isfinite(coverage)]
    require(np.all((finite >= 0) & (finite <= 1)), "coverage outside [0,1]")
    degree = membership.sum(axis=1)
    single = selected & (degree == 1)
    votes = membership[selected].sum(axis=0, dtype=np.int64)
    single_votes = membership[single].sum(axis=0, dtype=np.int64)
    count, single_count = int(selected.sum()), int(single.sum())
    weights = single_votes / count if count else np.full(n, np.nan)
    scores = np.full(3, np.nan)
    if available and count:
        scores[complete] = np.sum(weights[:, None] * coverage[:, complete], axis=0)
    all_finite = bool(np.isfinite(scores).all())
    ties = (scores == scores.max()) if all_finite else np.zeros(3, bool)
    positive = ties & (scores > 0)
    conflicts = np.any(collision[single_votes > 0], axis=0)
    reasons = []
    usable = np.zeros(3, bool)
    for b in range(3):
        why = []
        if not linked[b]:
            why.append("no_strict_initial_link")
        if not available:
            why.append("masks_unavailable")
        elif not n:
            why.append("no_current_masks")
        if count < 2:
            why.append("fewer_than_two_selected_points")
        if single_count < 2:
            why.append("fewer_than_two_single_members")
        if not all_finite:
            why.append("incomplete_competitor_projection")
        elif not (positive[b] and int(ties.sum()) == 1):
            why.append("not_positive_unique_body_maximum")
        if conflicts[b]:
            why.append("contributing_mask_contains_competing_node")
        usable[b] = not why
        reasons.append(why)
    return dict(
        selected_count=count,
        single_count=single_count,
        multi_count=int(np.count_nonzero(selected & (degree > 1))) if available else None,
        unassigned_count=int(np.count_nonzero(selected & (degree == 0))) if available else None,
        weights=weights,
        votes=votes,
        single_votes=single_votes,
        scores=scores,
        maximum_ties=ties,
        positive_maximum_ties=positive,
        single_vote_mass=float(single_count / count) if count and available else None,
        collision=conflicts,
        available_body=usable,
        reasons=reasons,
    )


def availability_cases(temporal, direct, observed, *, initial_frame=False):
    """A=two temporal views, B=temporal plus DIFFERENT direct view, both need deficit."""
    require(type(initial_frame) is bool, "explicit initial-frame flag required")
    require(
        temporal.shape == direct.shape == observed.shape == (3, 3)
        and temporal.dtype == direct.dtype == observed.dtype == bool,
        "all view/body and body/node booleans required",
    )
    result = []
    for body in range(3):
        t = np.flatnonzero(temporal[:, body]).tolist()
        d = np.flatnonzero(direct[:, body]).tolist()
        pairs = [[v, u] for v in t for u in d if v != u]
        deficit = np.flatnonzero(~observed[body]).tolist()
        eligible_time = bool(deficit) and not initial_frame
        a, b = eligible_time and len(t) >= 2, eligible_time and bool(pairs)
        result.append(
            dict(
                body=body,
                initial_frame=initial_frame,
                missing_nodes=deficit,
                temporal_views=t,
                current_direct_views=d,
                complementary_view_pairs=pairs,
                case_A=bool(a),
                case_B=bool(b),
                case_A_and_B=bool(a and b),
                case_A_or_B=bool(a or b),
            )
        )
    return result
