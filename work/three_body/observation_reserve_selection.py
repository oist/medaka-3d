"""Experimental one-slot reservation; no state, cost, or identity is rewritten.

This module is not used by the running tracker. Post-fit supported sources are
required by the eligibility function; proposal plans are not a substitute.
Independent D2 evidence enters through the already evaluated silhouette loss.
This is an engineering search rule, not an identity-confidence calculation.
"""

from __future__ import annotations

import numpy as np


def observation_eligibility(sources, observed, loss_parts, baseline_best):
    """Require complete, exclusive three-view rows and no worse node/image fit.

    Each body's M/C/T must use the same raw SLEAP row within each view. Different
    bodies must use different rows within that view. Row numbers across views
    or times are never interpreted as persistent identities.
    """
    sources, observed, parts = map(np.asarray, (sources, observed, loss_parts))
    count = len(sources)
    if (
        sources.shape != (count, 3, 3, 3)
        or sources.dtype.kind not in "iu"
        or np.any(sources < -1)
        or observed.shape != (count, 3, 3)
        or observed.dtype != np.bool_
        or parts.shape != (count, 5)
        or not np.isfinite(parts).all()
        or not isinstance(baseline_best, (int, np.integer))
        or isinstance(baseline_best, (bool, np.bool_))
        or not 0 <= baseline_best < count
    ):
        raise ValueError("post-fit sources, direct support, five loss parts and baseline required")
    full_sources = (sources >= 0).all(axis=(1, 2, 3))
    full_direct = observed.all(axis=(1, 2))
    whole_rows = (sources == sources[..., :1]).all(axis=(1, 2, 3))
    rows = sources[..., 0]
    exclusive = np.ones(count, dtype=bool)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        exclusive &= (rows[:, first] != rows[:, second]).all(axis=1)
    node = parts[:, 0] <= parts[baseline_best, 0]
    image = parts[:, 4] <= parts[baseline_best, 4]
    checks = dict(
        all_three_views_all_nodes=full_sources,
        all_nine_nodes_direct=full_direct,
        whole_row_within_each_view=whole_rows,
        exclusive_rows_between_bodies=exclusive,
        node_loss_no_worse=node,
        silhouette_loss_no_worse=image,
    )
    return np.logical_and.reduce(list(checks.values())), checks


def select_with_observation_reserve(
    poses, path_cost, parents, permutations, eligible, beam_size=6
):
    """Keep ordinary first five and replace at most the last of six candidates.

    A candidate already kept with strong observations causes no change. Otherwise
    the cheapest eligible outsider must be distinct from the protected prefix
    under the existing same-parent/permutation/all-node 0.5 mm duplicate rule.
    Returned indices stay in real cost order. No penalty, cost or ID is modified.
    """
    poses, path, parent, permutation, eligible = map(
        np.asarray, (poses, path_cost, parents, permutations, eligible)
    )
    count = len(poses)
    if (
        count < 1
        or poses.shape != (count, 3, 3, 3)
        or path.shape != (count,)
        or not np.isfinite(poses).all()
        or not np.isfinite(path).all()
        or parent.shape != (count,)
        or parent.dtype.kind not in "iu"
        or np.any(parent < 0)
        or permutation.shape != (count, 3)
        or permutation.dtype.kind not in "iu"
        or not np.all(np.sort(permutation, axis=1) == np.arange(3))
        or eligible.shape != (count,)
        or eligible.dtype != np.bool_
        or type(beam_size) is not int
        or beam_size != 6
    ):
        raise ValueError("finite three-body candidates and unchanged six-candidate beam required")

    def duplicate(first, second):
        return (
            parent[first] == parent[second]
            and np.array_equal(permutation[first], permutation[second])
            and np.max(np.linalg.norm(poses[first] - poses[second], axis=-1)) < 0.5
        )

    order = np.argsort(path, kind="stable")
    baseline = []
    for candidate in order:
        if any(duplicate(candidate, kept) for kept in baseline):
            continue
        baseline.append(int(candidate))
        if len(baseline) == beam_size:
            break
    selected = baseline.copy()
    rejected = []
    replacement = None
    reason = "no_eligible_candidate"
    if eligible.any():
        if eligible[baseline].any():
            reason = "eligible_candidate_already_retained"
        elif len(baseline) != beam_size:
            reason = "baseline_not_full"
        else:
            reason = "eligible_outsiders_duplicate_protected_prefix"
            for candidate in order:
                if not eligible[candidate] or candidate in baseline:
                    continue
                conflicts = [kept for kept in baseline[:-1] if duplicate(candidate, kept)]
                if conflicts:
                    rejected.append(dict(candidate=int(candidate), duplicate_of=conflicts))
                    continue
                replacement = dict(
                    removed=baseline[-1],
                    admitted=int(candidate),
                    extra_path_cost=float(path[candidate] - path[baseline[-1]]),
                )
                selected[-1] = int(candidate)
                reason = "reserved_observation_candidate"
                break
    rank = np.empty(count, dtype=np.int64)
    rank[order] = np.arange(count)
    if np.any(np.diff(rank[selected]) <= 0) or selected[:-1] != baseline[:-1]:
        raise AssertionError("reservation changed cost order or protected prefix")
    return np.asarray(selected, dtype=np.int64), dict(
        baseline_indices=baseline,
        selected_indices=selected,
        stable_cost_order=order.tolist(),
        reason=reason,
        changed=selected != baseline,
        replacement=replacement,
        duplicate_rejections=rejected,
        current_best_preserved=selected[0] == baseline[0],
        real_costs_modified=False,
        reservation_guarantees_future_retention=False,
    )
