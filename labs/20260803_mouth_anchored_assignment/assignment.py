"""Mouth-anchored one-to-one tail assignment.

See README.md for the pre-registered specification and decision rule. This module
implements it and nothing else; all scoring lives in the shared harness.

The operation replaces *instance selection* with *assignment*. Round-7 top-down
decodes one instance per centroid crop, so two crops centred on two different
mouths can each emit the same tail. Selecting between those instances --- which is
what the B0.4 shared-node rule does --- raises precision but deletes an anchor,
and in 32% of the affected frames that anchor is the only evidence of a third
animal. Assigning instead keeps every anchor and rations the tails.

Deliberately **score-free**. Instance and point confidences were measured on the
21 shared-node pairs and do not separate them (0.831 vs 0.825 on the hardest
case); depending on them would also prevent this running on the retained round-7
predictions, where re-inference is not on offer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

#: Closest a labelled fish has ever been to a frame edge (k01, n=339 GT
#: instances, minimum 68.7 px). A mouth nearer than this is outside the region
#: animals occupy and is consistent with the glass reflections of §4.6.
EDGE_MIN_PX = 68.7

#: Two nodes of the same type nearer than this are one node. The median labelled
#: body span is 112.6 px, so 8 px cannot separate two animals.
DEDUP_PX = 8.0

#: Rejection ceiling as a two-sided tail probability of the labelled span
#: distribution. 0.01 is exactly the 0.5--99.5 percentile interval Gate B0
#: already adopted ([35.1, 152.4] px on top), so this adds no free parameter.
REJECT_Q = 0.01

#: Displacement within which a node is considered the same node one frame later.
#: 166 fps; 25 px is generous for a 112 px animal.
TEMPORAL_PX = 25.0

#: Finite stand-in for an impossible pairing. Must exceed the rejection cost.
_IMPOSSIBLE = 1.0e3

MOUTH, TAIL = 0, 1


@dataclass(frozen=True)
class SpanModel:
    """Empirical distribution of the human-labelled mouth--tail span, in pixels.

    Non-parametric on purpose. The labelled spans are not Gaussian --- they are a
    projection of a rigid body at every out-of-plane angle, so the short tail is
    foreshortening and the long tail is not --- and fitting a symmetric density
    to that would put the rejection ceiling in the wrong place at both ends.
    """

    spans: np.ndarray

    @classmethod
    def from_spans(cls, spans) -> SpanModel:
        array = np.asarray(sorted(float(s) for s in spans), dtype=float)
        if array.size < 100:
            raise ValueError(f"span model needs a real sample, got n={array.size}")
        return cls(array)

    @property
    def median(self) -> float:
        return float(np.median(self.spans))

    def bounds(self, q: float = REJECT_Q) -> tuple[float, float]:
        """The span interval a two-sided tail probability of ``q`` corresponds to."""
        return (
            float(np.percentile(self.spans, 100.0 * q / 2.0)),
            float(np.percentile(self.spans, 100.0 * (1.0 - q / 2.0))),
        )

    def q(self, distance: np.ndarray) -> np.ndarray:
        """Two-sided empirical tail probability: 1 at the median, 0 in the tails."""
        n = self.spans.size
        below = np.searchsorted(self.spans, np.asarray(distance, dtype=float)) / n
        return 2.0 * np.minimum(below, 1.0 - below)

    def cost(self, distance: np.ndarray) -> np.ndarray:
        """``-log q``. Zero at the median, rising without bound into the tails."""
        return np.minimum(-np.log(np.maximum(self.q(distance), 1e-12)), _IMPOSSIBLE)


@dataclass(frozen=True)
class Params:
    """Everything tunable, in one place, so a run can state what it ran."""

    edge_min_px: float = EDGE_MIN_PX
    dedup_px: float = DEDUP_PX
    reject_q: float = REJECT_Q
    temporal: bool = False
    temporal_px: float = TEMPORAL_PX

    @property
    def reject_cost(self) -> float:
        return float(-np.log(self.reject_q))


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Result:
    """One frame's output plus the bookkeeping needed to explain it."""

    instances: list[np.ndarray]
    n_anchors_raw: int
    n_dropped_edge: int
    n_dropped_dedup_mouth: int
    n_dropped_temporal_mouth: int
    n_tail_candidates: int
    n_dropped_temporal_tail: int
    n_assigned: int
    n_mouth_only: int
    n_tails_unclaimed: int


def _nodes(instances: list[np.ndarray], which: int) -> np.ndarray:
    """The (n, 2) array of non-NaN ``which`` nodes across predicted instances."""
    if not instances:
        return np.empty((0, 2))
    stacked = np.array([inst[which] for inst in instances], dtype=float)
    return stacked[~np.isnan(stacked).any(axis=1)]


def _dedup(points: np.ndarray, radius: float) -> np.ndarray:
    """Greedy: keep a point unless an already-kept point is within ``radius``.

    Order-dependent by construction. With a radius of 8 px against a 112 px
    animal the clusters are tight enough that the choice of representative moves
    a node by at most 8 px, which is well inside the 20 px match tolerance.
    """
    kept: list[np.ndarray] = []
    for point in points:
        if all(float(np.hypot(*(point - other))) >= radius for other in kept):
            kept.append(point)
    return np.array(kept) if kept else np.empty((0, 2))


def _edge_distance(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Distance from each point to the nearest frame edge. ``shape`` is (H, W)."""
    height, width = shape
    return np.minimum.reduce(
        [points[:, 0], points[:, 1], width - points[:, 0], height - points[:, 1]]
    )


def _supported(points: np.ndarray, neighbours: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask: does each point have a neighbour-frame node within radius?"""
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    if neighbours.size == 0:
        return np.zeros(len(points), dtype=bool)
    gaps = np.hypot(
        points[:, None, 0] - neighbours[None, :, 0],
        points[:, None, 1] - neighbours[None, :, 1],
    )
    return gaps.min(axis=1) <= radius


def assign_frame(
    instances: list[np.ndarray],
    shape: tuple[int, int],
    span: SpanModel,
    params: Params = DEFAULT_PARAMS,
    neighbour_mouths: np.ndarray | None = None,
    neighbour_tails: np.ndarray | None = None,
) -> Result:
    """Re-derive one frame's instances by assigning decoded tails to decoded mouths.

    Args:
        instances: Decoded (2, 2) [mouth; tail] arrays, NaN for a missing node.
        shape: Frame (height, width), for the edge rule.
        span: Labelled span distribution, fitted on labels disjoint from the
            evaluation frames.
        params: Frozen configuration.
        neighbour_mouths, neighbour_tails: Nodes decoded in frames f-1 / f+1,
            pooled. Required when ``params.temporal`` is set, ignored otherwise.

    Returns:
        A :class:`Result` whose ``instances`` are (2, 2) arrays with a real mouth
        and either a real tail or NaN. **No tail appears in two instances**; the
        one-to-one constraint makes that unrepresentable rather than filtered.
    """
    mouths = _nodes(instances, MOUTH)
    n_raw = len(mouths)

    if n_raw:
        keep = _edge_distance(mouths, shape) >= params.edge_min_px
        n_edge = int((~keep).sum())
        mouths = mouths[keep]
    else:
        n_edge = 0

    before = len(mouths)
    mouths = _dedup(mouths, params.dedup_px)
    n_dedup = before - len(mouths)

    tails = _dedup(_nodes(instances, TAIL), params.dedup_px)

    n_temporal_mouth = n_temporal_tail = 0
    if params.temporal:
        mask = _supported(mouths, _as_points(neighbour_mouths), params.temporal_px)
        n_temporal_mouth = int((~mask).sum())
        mouths = mouths[mask]
        mask = _supported(tails, _as_points(neighbour_tails), params.temporal_px)
        n_temporal_tail = int((~mask).sum())
        tails = tails[mask]

    n_tails = len(tails)
    if len(mouths) == 0:
        return Result([], n_raw, n_edge, n_dedup, n_temporal_mouth, n_tails,
                      n_temporal_tail, 0, 0, n_tails)

    # Assign-or-abstain, solved exactly: pad with one dummy "no tail" column per
    # anchor, priced at the rejection ceiling. Without the padding the solver is
    # forced to place every anchor and a tail-less mouth fabricates a body axis
    # from whatever phantom is nearest --- frame 274709 decodes three tails for
    # two animals, so this is not hypothetical.
    reject = params.reject_cost
    if n_tails:
        distances = np.hypot(
            mouths[:, None, 0] - tails[None, :, 0],
            mouths[:, None, 1] - tails[None, :, 1],
        )
        real = span.cost(distances)
    else:
        real = np.empty((len(mouths), 0))
    dummy = np.full((len(mouths), len(mouths)), reject)
    rows, cols = linear_sum_assignment(np.hstack([real, dummy]))

    out: list[np.ndarray] = []
    claimed: set[int] = set()
    for row, col in zip(rows, cols, strict=True):
        instance = np.full((2, 2), np.nan)
        instance[MOUTH] = mouths[row]
        # A dummy column, or a real column the solver only took because a cheaper
        # one was unavailable: both mean abstain. The strict inequality keeps a
        # pairing exactly at the ceiling out, matching B0.4's closed interval.
        if col < n_tails and real[row, col] < reject:
            instance[TAIL] = tails[col]
            claimed.add(col)
        out.append(instance)

    n_assigned = len(claimed)
    return Result(
        instances=out,
        n_anchors_raw=n_raw,
        n_dropped_edge=n_edge,
        n_dropped_dedup_mouth=n_dedup,
        n_dropped_temporal_mouth=n_temporal_mouth,
        n_tail_candidates=n_tails,
        n_dropped_temporal_tail=n_temporal_tail,
        n_assigned=n_assigned,
        n_mouth_only=len(out) - n_assigned,
        n_tails_unclaimed=n_tails - n_assigned,
    )


def _as_points(value: np.ndarray | None) -> np.ndarray:
    return np.empty((0, 2)) if value is None else np.asarray(value, dtype=float)


# ---------------------------------------------------------------------------
# Corrected specification (README amendment, 2026-08-03). A DIFFERENT algorithm,
# so it gets a different function and a different command, not a flag.
# ---------------------------------------------------------------------------

#: Closest a labelled *mouth* has ever been to a frame edge, on the 113 held-out
#: frames (n=320; p1 = 54.4 px). The analogue of EDGE_MIN_PX for the quantity
#: this rule actually gates. Using the centroid figure cost 3.5 coverage points.
EDGE_MIN_MOUTH_PX = 28.8

#: Cost of taking a tail the model did not decode with this mouth. Distance
#: contributes d/hi in [0.23, 1.0] over the feasible band, so 1.0 makes the
#: origin pairing win whenever it is feasible and the two terms cannot trade.
ORIGIN_PENALTY = 1.0

#: Abstaining costs as much as a foreign tail at the maximum plausible distance.
ABSTAIN_COST = 2.0


@dataclass(frozen=True)
class OriginParams:
    """Frozen configuration of the corrected specification."""

    edge_min_px: float = EDGE_MIN_MOUTH_PX
    dedup_px: float = DEDUP_PX
    reject_q: float = REJECT_Q
    origin_penalty: float = ORIGIN_PENALTY
    abstain_cost: float = ABSTAIN_COST


DEFAULT_ORIGIN_PARAMS = OriginParams()


def _cluster(points: np.ndarray, radius: float) -> tuple[np.ndarray, list[int]]:
    """Greedy dedup that also reports which cluster each input point fell into.

    The origin term needs the mapping: after deduplication a tail is no longer
    the tail of one instance, it is the representative of a cluster, and an
    instance's claim on it has to survive that.
    """
    kept: list[np.ndarray] = []
    membership: list[int] = []
    for point in points:
        for index, other in enumerate(kept):
            if float(np.hypot(*(point - other))) < radius:
                membership.append(index)
                break
        else:
            kept.append(point)
            membership.append(len(kept) - 1)
    return (np.array(kept) if kept else np.empty((0, 2))), membership


def assign_frame_origin(
    instances: list[np.ndarray],
    shape: tuple[int, int],
    span: SpanModel,
    params: OriginParams = DEFAULT_ORIGIN_PARAMS,
) -> Result:
    """Origin-preferred, geometry-arbitrated one-to-one assignment.

    A mouth keeps the tail the centered-instance head decoded in its own crop.
    Only a mouth **dispossessed** by the one-to-one constraint goes shopping, and
    it then takes the nearest feasible free tail, or abstains. This is the
    operation the lab set out to build: suppression deletes the loser, this keeps
    the loser's anchor and rations the tail.

    The audit that produced this design: over 47 pairings a span-only cost wanted
    to overrule, the model's own pairing was correct in 41.
    """
    hi = span.bounds(params.reject_q)[1]
    mouth_points, tail_points, origin = [], [], []
    for index, instance in enumerate(instances):
        mouth, tail = instance[MOUTH], instance[TAIL]
        if not np.isnan(mouth).any():
            mouth_points.append(mouth)
            origin.append(index)
        if not np.isnan(tail).any():
            tail_points.append((index, tail))

    mouths = np.array(mouth_points) if mouth_points else np.empty((0, 2))
    n_raw = len(mouths)
    if n_raw:
        keep = _edge_distance(mouths, shape) >= params.edge_min_px
        n_edge = int((~keep).sum())
        mouths, origin = mouths[keep], [o for o, k in zip(origin, keep, strict=True) if k]
    else:
        n_edge = 0

    before = len(mouths)
    mouth_clusters, mouth_membership = _cluster(mouths, params.dedup_px)
    n_dedup = before - len(mouth_clusters)
    #: Which source instances each surviving anchor speaks for.
    anchor_sources: list[set[int]] = [set() for _ in range(len(mouth_clusters))]
    for source, cluster in zip(origin, mouth_membership, strict=True):
        anchor_sources[cluster].add(source)

    raw_tails = np.array([t for _, t in tail_points]) if tail_points else np.empty((0, 2))
    tails, tail_membership = _cluster(raw_tails, params.dedup_px)
    tail_sources: list[set[int]] = [set() for _ in range(len(tails))]
    for (source, _), cluster in zip(tail_points, tail_membership, strict=True):
        tail_sources[cluster].add(source)

    n_tails = len(tails)
    if len(mouth_clusters) == 0:
        return Result([], n_raw, n_edge, n_dedup, 0, n_tails, 0, 0, 0, n_tails)

    if n_tails:
        distances = np.hypot(
            mouth_clusters[:, None, 0] - tails[None, :, 0],
            mouth_clusters[:, None, 1] - tails[None, :, 1],
        )
        cost = distances / hi
        for i, sources in enumerate(anchor_sources):
            for j, claimants in enumerate(tail_sources):
                if not (sources & claimants):
                    cost[i, j] += params.origin_penalty
        # Feasibility is the B0.4 span band, unchanged, expressed on the same
        # empirical distribution the rejected specification used. Infeasible
        # pairs are priced above abstention so the solver never prefers one.
        cost[span.q(distances) < params.reject_q] = _IMPOSSIBLE
    else:
        cost = np.empty((len(mouth_clusters), 0))

    dummy = np.full((len(mouth_clusters), len(mouth_clusters)), params.abstain_cost)
    rows, cols = linear_sum_assignment(np.hstack([cost, dummy]))

    out: list[np.ndarray] = []
    claimed: set[int] = set()
    for row, col in zip(rows, cols, strict=True):
        instance = np.full((2, 2), np.nan)
        instance[MOUTH] = mouth_clusters[row]
        if col < n_tails and cost[row, col] < params.abstain_cost:
            instance[TAIL] = tails[col]
            claimed.add(col)
        out.append(instance)

    return Result(
        instances=out,
        n_anchors_raw=n_raw,
        n_dropped_edge=n_edge,
        n_dropped_dedup_mouth=n_dedup,
        n_dropped_temporal_mouth=0,
        n_tail_candidates=n_tails,
        n_dropped_temporal_tail=0,
        n_assigned=len(claimed),
        n_mouth_only=len(out) - len(claimed),
        n_tails_unclaimed=n_tails - len(claimed),
    )


def span_model_from_labels(labels) -> SpanModel:
    """Fit the span model from a ``sleap_io.Labels`` of human annotation.

    Complete instances only: a missing node has no span, and imputing one would
    put invented geometry into the cost function.
    """
    spans: list[float] = []
    for frame in labels.labeled_frames:
        for instance in frame.instances:
            points = np.asarray(instance.numpy(), dtype=float)[:, :2]
            if not np.isnan(points).any():
                spans.append(float(np.hypot(*(points[MOUTH] - points[TAIL]))))
    return SpanModel.from_spans(spans)


class MouthAnchoredAssignment:
    """`medaka3d.stages.Detector` adapter, so the shared harness can run this.

    Thin by design: the stage boundary is where a component becomes comparable,
    not where it becomes clever.
    """

    def __init__(self, base, span: SpanModel, params: Params = DEFAULT_PARAMS):
        self.name = "mouth_anchored_assignment"
        self._base = base
        self._span = span
        self._params = params

    def detect(self, video, frames):
        for detection in self._base.detect(video, frames):
            result = assign_frame(
                detection.instances, detection.shape, self._span, self._params
            )
            yield type(detection)(**{**vars(detection), "instances": result.instances})
