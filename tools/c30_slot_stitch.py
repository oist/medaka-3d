"""C30 - stage D's join rule, reduced to re-permuting slots where continuity says so.

=============================================================================
WHAT PROBLEM THIS IS, AND WHY IT SHRANK
=============================================================================

The 2026-08-18 handoff (`NEXT_SESSION_stitching.md`) framed stage D as **973
joins**: at every segment boundary, decide which identity going out is which
identity coming in. It was never built. The uncapped rework (`c24`-`c28`)
changed the input: the slot labelling now carries 3 ids over 98.77% of frames,
and the visible identity failure has collapsed to a specific, countable event -
**the slot assignment permutes between two adjacent resolved frames**, which on
screen is three trajectories exchanging positions in one frame step.

Measured on K_1_2 at lambda 0.767 before writing this tool: **416 adjacent
resolved pairs** where a non-identity permutation of the slots is cheaper than
identity by more than 5 mm, clustering into 166 episodes at least 1 s apart, of
which **139 leave a net permutation** - a lasting switch roughly every 15 s.
The animals' minimum pairwise distance at those moments is **median 21 mm, p90
104 mm**: most switches happen while the fish are far apart, where following
3D continuity is unambiguous. Those are the ones this tool repairs.

=============================================================================
THE RULE, v2 (2026-09-03) - an evidence ladder, refusing what nothing decides
=============================================================================

v1 knew one kind of evidence: 8-frame median positions. The experimenter's
review showed the eye uses another - **the displacement vector against the
mean motion of the surrounding 0.1-0.3 s** - and the female annotation
confirmed four v1 refusals as real swaps that vector decides. The ladder, per
boundary, each step deciding only where the one above could not (thresholds
pre-registered in `decisions/2026-09-01-slot-stitching.md`, v2 section):

1. **Median rule** (v1, unchanged): per-slot medians over the last / next 8
   resolved frames; join iff a non-identity assignment wins by > 5 mm. A
   one-frame spike cannot move an 8-frame median.
2. **Velocity re-evaluation**, only at step-1 refusals: extrapolate each slot
   0.3 s of stitched trajectory forward; join iff the margin clears 5 mm AND
   the permutation matches the medians' preference. Fast parallel motion fills
   both of step 1's costs with travelled distance; the extrapolation removes
   it - the lane-keeping swap class the annotation confirmed.
3. **Zone bridge**: where detector confidence collapsed AND the geometry
   teleports (a turbulent zone), no boundary inside is decided at all - the
   31479 lesson: every position in there may be fictitious. One decision is
   made across the zone, clean end to clean end, taken only when position and
   velocity margins both clear 5 mm and agree.
4. Otherwise **refusal, always recorded** - including the window-incomplete
   paths that v1 skipped silently.

**What this cannot fix.** Single-frame spikes are geometry, not identity
(`c33`/`c34`'s domain). A lasting swap whose median comparison favours
identity outright - the head-on "bounce" - is this rule's known blind spot.
Nothing here creates identity where there is none: the output still has
breaks, and `n_groups = 1`, one experiment, female-only ground truth.

=============================================================================
WHAT IS WRITTEN
=============================================================================

A new npz per arm under ``c30_slot_stitch/``, same schema as `c28`'s
(``slot_track`` re-run by the **gated** tracker on the stitched geometry, gate
held at `c19`'s 4.0 mm / 73 frames as every arm since `c21`, POST-HOC;
per-view tracks re-propagated), plus ``slot_perm`` - the raw slot each fixed
slot reads from, per frame - so every join is auditable. ``state`` is **not**
carried: it indexes `c25`'s per-frame state table by slot order, which this
tool changes. **No existing output is modified**; `c03` still owns the fixed
paths and nothing is promoted.

Scoring: run `c09` per arm with ``--npz`` on the output, then once with no
``--npz`` to restore its fixed baseline (`c03`: preserved 269 / break 19 /
swap 8) - the same protocol every arm since `c21` has used. **A swap count
above the input arm's disqualifies the join rule regardless of everything
else** (clause 1 of the 2026-08-18 handoff, unchanged).

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c30_slot_stitch.py --lambdas 0 0.1 0.25 0.5 0.767 2 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c19_viterbi_correspondence import coverage, derive_gate
from c31_doubt_channel import SCORE_DEGRADED, best_scores

from medaka3d import load_paths
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, Reconstruction
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"

#: `c19`'s gate, held for every arm since `c21`. POST-HOC.
GATE_MM = 4.0
GAP_FRAMES = 73

#: Median window, in resolved frames (48 ms). A single-frame spike is 1 of 8
#: and cannot move the median; a real switch shifts all 8.
WINDOW = 8

#: How decisively a non-identity assignment must beat identity, in mm summed
#: over the three slots. Below this the boundary is refused, not guessed.
MARGIN_MM = 5.0

#: v2 - velocity lookback, in frames (0.3 s). The experimenter's criterion:
#: the eye compares the one-frame displacement VECTOR against the mean motion
#: of the surrounding 0.1-0.3 s. Measured 2026-09-03: at the four refused
#: boundaries the female annotation confirmed as swaps, this extrapolation
#: picks the correct permutation at every window in that range.
VELOCITY_WINDOW = 50

#: v2 - turbulent-zone definition: degraded frames (any view's best score
#: below `c31`'s floor, dilated +-2 per the 2026-09-03 window study) merged
#: within this many frames, kept only if the run contains a same-slot step
#: above ZONE_BIG_MM - ghost-scale geometry, not mere low confidence.
ZONE_MERGE_GAP = 16
ZONE_BIG_MM = 20.0

#: v4 - the medians must lean the same way before velocity may decide: below
#: this the median comparison is a coin toss, and both coin-toss joins in the
#: v3 run planted new female swaps (median margins 0.51 / 0.81 mm) while all
#: three correct ones sat at 3.8-4.4 mm. Calibrated on those five cases.
VELOCITY_NEEDS_MEDIAN_MM = 2.5

#: v4 - a zone correction asserts identity across corrupted geometry; a wrong
#: one costs an inter-animal distance (median 95 mm). The two correct bridges
#: measured 192 and 319 mm, the two wrong ones 5-11 mm. Calibrated on those
#: four cases; an order above the boundary bar.
BRIDGE_MARGIN_MM = 30.0

PERMS = tuple(permutations(range(N_ANIMALS)))


def _median_rows(rows: np.ndarray) -> np.ndarray:
    """(k, 3 slots, 3) -> (3 slots, 3) nanmedian; a slot with < 2 finite rows is NaN."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(rows, axis=0)
    enough = np.isfinite(rows).all(axis=-1).sum(axis=0) >= 2
    med[~enough] = np.nan
    return med


def candidate_boundaries(xyz: np.ndarray, ok_idx: np.ndarray, max_gap: int) -> np.ndarray:
    """Positions ``k`` where the boundary ok_idx[k-1] -> ok_idx[k] needs a look.

    Deliberately generous - a candidate costs one windowed evaluation, a miss
    costs a wrong trajectory. Flagged: any boundary whose cheapest single-frame
    assignment is not the identity, any with a same-slot step over 10 mm, and
    any that crosses an unresolved gap at all.
    """
    a, b = xyz[ok_idx[:-1]], xyz[ok_idx[1:]]
    d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
    filled = np.nan_to_num(d, nan=0.0)
    costs = np.stack([filled[:, (0, 1, 2), pm].sum(axis=1) for pm in PERMS], axis=1)
    nonid = costs.argmin(axis=1) != 0
    big = np.nan_to_num(d[:, (0, 1, 2), (0, 1, 2)], nan=0.0).max(axis=1) > 10.0
    gap = np.diff(ok_idx) > 1
    flags = (nonid | big | gap) & (np.diff(ok_idx) <= max_gap)
    return np.flatnonzero(flags) + 1


def turbulent_zones(xyz: np.ndarray, ok_idx: np.ndarray,
                    degraded: np.ndarray | None,
                    merge_gap: int = ZONE_MERGE_GAP,
                    big_mm: float = ZONE_BIG_MM) -> list[tuple[int, int]]:
    """Frame spans where the geometry itself is not to be trusted.

    Degraded stretches (detector confidence collapsed) that also contain a
    ghost-scale step. Inside one, boundary-by-boundary decisions walk through
    fictitious positions - the 31479 lesson - so `stitch` decides each span
    once, across it, or not at all.
    """
    if degraded is None:
        return []
    frames = np.flatnonzero(degraded)
    if not frames.size:
        return []
    breaks = np.flatnonzero(np.diff(frames) > merge_gap)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [frames.size - 1]])
    adjacent = np.diff(ok_idx) == 1
    step = np.linalg.norm(
        xyz[ok_idx[1:][adjacent]] - xyz[ok_idx[:-1][adjacent]], axis=-1)
    big_at = ok_idx[1:][adjacent][np.nan_to_num(step, nan=0.0).max(axis=1) > big_mm]
    out = []
    for a, b in zip(starts, stops, strict=True):
        first, last = int(frames[a]), int(frames[b])
        if ((big_at >= first - 1) & (big_at <= last + 1)).any():
            out.append((first, last))
    return out


def _velocity_costs(xyz, ok_idx, gaps, current, fence, pre_idx, target_frame,
                    velocity_window, max_gap):
    """Assignment costs from constant-velocity extrapolation, or None.

    The slot velocities come from the stitched-space trajectory over the last
    ``velocity_window`` resolved frames before ``pre_idx``, never crossing the
    fence or an over-long gap - contaminated history predicts nothing.
    """
    t0 = int(ok_idx[pre_idx])
    back = pre_idx
    while (back > fence and t0 - int(ok_idx[back - 1]) <= velocity_window
           and gaps[back - 1] <= max_gap):
        back -= 1
    tb = int(ok_idx[back])
    if t0 - tb < velocity_window // 2:
        return None
    fx0 = xyz[t0][current]
    fxb = xyz[tb][current]
    target = xyz[target_frame]
    if not (np.isfinite(fx0).all() and np.isfinite(fxb).all()
            and np.isfinite(target).all()):
        return None
    predicted = fx0 + (fx0 - fxb) / (t0 - tb) * (target_frame - t0)
    return np.array([
        np.linalg.norm(predicted - target[current[list(q)]], axis=-1).sum()
        for q in PERMS
    ])


def stitch(xyz: np.ndarray, ok_idx: np.ndarray, window: int = WINDOW,
           margin_mm: float = MARGIN_MM, max_gap: int = GAP_FRAMES,
           velocity_window: int = VELOCITY_WINDOW,
           degraded: np.ndarray | None = None):
    """The v2 evidence ladder. Pure: reads positions, returns the permutation.

    Median positions decide first; a refusal is re-evaluated by velocity
    extrapolation; a turbulent zone is decided once, across it, by position
    and velocity agreeing - or left alone. Rules and thresholds are fixed in
    `decisions/2026-09-01-slot-stitching.md` (v2 section).

    Returns ``(perm, joins, refusals)``: ``perm`` is (n_frames, 3) int8 with
    ``fixed[f, s] = raw[f, perm[f, s]]``; ``joins`` carry an ``evidence`` field
    (``median`` / ``velocity`` / ``bridge``).
    """
    n_frames = xyz.shape[0]
    candidates = candidate_boundaries(xyz, ok_idx, max_gap)
    zones = turbulent_zones(xyz, ok_idx, degraded)
    perm = np.tile(np.arange(N_ANIMALS, dtype=np.int8), (n_frames, 1))

    current = np.arange(N_ANIMALS)
    #: Index into ok_idx before which history must not reach: the last accepted
    #: join or the far side of the last over-long gap, whichever is later.
    fence = 0
    joins, refusals = [], []

    gaps = np.diff(ok_idx)
    candidate_set = set(candidates.tolist())

    for k in candidates:
        frame = int(ok_idx[k])
        if gaps[k - 1] > max_gap:           # unreachable via flags, kept explicit
            fence = k
            continue

        lo = max(fence, k - window)
        # History must also not straddle an over-long gap.
        for j in range(k - 1, lo, -1):
            if gaps[j - 1] > max_gap:
                lo = j
                break
        hist_frames = ok_idx[lo:k]
        if hist_frames.size < 2:
            refusals.append({"frame": frame, "reason": "window incomplete"})
            continue
        history = xyz[hist_frames][:, current, :]

        hi = min(k + window, ok_idx.size)
        # The future window stops at an over-long gap always, and at the next
        # candidate when enough remains: two nearby boundaries must not blur
        # into one median.
        for j in range(k + 1, hi):
            if gaps[j - 1] > max_gap:
                hi = j
                break
            if j in candidate_set and j - k >= 3:
                hi = j
                break
        future = xyz[ok_idx[k:hi]]
        if future.shape[0] < 2:
            refusals.append({"frame": frame, "reason": "window incomplete"})
            continue

        hist_med = _median_rows(history)
        fut_med = _median_rows(future)
        if not (np.isfinite(hist_med).all() and np.isfinite(fut_med).all()):
            refusals.append({"frame": frame, "reason": "window incomplete"})
            continue

        costs = np.array([
            np.linalg.norm(hist_med - fut_med[current[list(q)]], axis=-1).sum()
            for q in PERMS
        ])
        best = int(costs.argmin())
        margin = float(costs[0] - costs[best])
        pairwise = np.linalg.norm(
            fut_med[:, None, :] - fut_med[None, :, :], axis=-1)
        separation = float(pairwise[np.triu_indices(N_ANIMALS, 1)].min())
        if best == 0:
            continue
        if margin <= margin_mm:
            # Step 2 of the ladder: the medians saw a swap but could not vouch
            # for it. Extrapolated velocities may - and only for the SAME
            # permutation the medians preferred; disagreement stays a refusal.
            velocity = _velocity_costs(xyz, ok_idx, gaps, current, fence,
                                       k - 1, frame, velocity_window, max_gap)
            if velocity is not None and margin >= VELOCITY_NEEDS_MEDIAN_MM:
                best_v = int(velocity.argmin())
                margin_v = float(velocity[0] - velocity[best_v])
                if best_v == best and margin_v > margin_mm:
                    q = PERMS[best]
                    current = current[list(q)]
                    perm[frame:] = current
                    fence = k
                    joins.append({"frame": frame, "evidence": "velocity",
                                  "permutation": list(map(int, q)),
                                  "margin_mm": round(margin, 2),
                                  "velocity_margin_mm": round(margin_v, 2),
                                  "min_separation_mm": round(separation, 1)})
                    continue
            refusals.append({"frame": frame, "margin_mm": round(margin, 2),
                             "velocity_margin_mm": (
                                 None if velocity is None else round(
                                     float(velocity[0] - velocity.min()), 2)),
                             "min_separation_mm": round(separation, 1),
                             "reason": "not decisive"})
            continue

        q = PERMS[best]
        current = current[list(q)]
        perm[frame:] = current
        fence = k
        joins.append({"frame": frame, "evidence": "median",
                      "permutation": list(map(int, q)),
                      "margin_mm": round(margin, 2),
                      "min_separation_mm": round(separation, 1)})

    # ------------------------------------------------------------------
    # v3 audit: every turbulent zone is checked once, clean end to clean
    # end, in stitched space. Continuous interiors audit to the identity
    # and are left alone; a decisive contradiction - the 31479 signature,
    # interior decisions walked through fictitious geometry and netted to
    # the wrong side - is corrected from the first clean frame after the
    # zone. v2 suppressed the interior instead, lost ~125 correct joins at
    # lambda 0.767 alone, and was retired for it.
    # ------------------------------------------------------------------
    in_zone = np.zeros(n_frames, bool)
    for first, last in zones:
        in_zone[first:last + 1] = True

    def fixed_at(frame_index: int) -> np.ndarray:
        return xyz[frame_index][perm[frame_index]]

    for first, last in zones:
        entry = {"zone": [int(first), int(last)]}
        pre = int(np.searchsorted(ok_idx, first)) - 1
        while pre >= 0 and in_zone[int(ok_idx[pre])]:
            pre -= 1
        post = int(np.searchsorted(ok_idx, last, side="right"))
        while post < ok_idx.size and in_zone[int(ok_idx[post])]:
            post += 1
        if pre < 0 or post >= ok_idx.size:
            refusals.append({**entry, "reason": "zone has no clean endpoint"})
            continue
        t0, t1 = int(ok_idx[pre]), int(ok_idx[post])
        fx0, fx1 = fixed_at(t0), fixed_at(t1)
        entry.update({"pre_frame": t0, "post_frame": t1})
        if not (np.isfinite(fx0).all() and np.isfinite(fx1).all()):
            refusals.append({**entry, "reason": "zone endpoints incomplete"})
            continue
        position = np.array([
            np.linalg.norm(fx0 - fx1[list(q)], axis=-1).sum() for q in PERMS])
        best_p = int(position.argmin())
        margin_p = float(position[0] - position[best_p])
        history = [j for j in range(pre - 1, -1, -1)
                   if t0 - int(ok_idx[j]) <= velocity_window
                   and not in_zone[int(ok_idx[j])]]
        entry["position_margin_mm"] = round(margin_p, 1)
        if not history:
            refusals.append({**entry, "reason": "zone has no velocity history"})
            continue
        tb = int(ok_idx[history[-1]])
        fxb = fixed_at(tb)
        if t0 - tb < velocity_window // 2 or not np.isfinite(fxb).all():
            refusals.append({**entry, "reason": "zone has no velocity history"})
            continue
        predicted = fx0 + (fx0 - fxb) / (t0 - tb) * (t1 - t0)
        velocity = np.array([
            np.linalg.norm(predicted - fx1[list(q)], axis=-1).sum()
            for q in PERMS])
        best_v = int(velocity.argmin())
        margin_v = float(velocity[0] - velocity[best_v])
        entry["velocity_margin_mm"] = round(margin_v, 1)
        # Silence needs the same decisiveness a correction does: identity must
        # WIN by the margin on both kinds of evidence, not merely edge ahead.
        clean_p = float(position[1:].min() - position[0])
        clean_v = float(velocity[1:].min() - velocity[0])
        if clean_p > margin_mm and clean_v > margin_mm:
            continue                      # the interior audits clean, decisively
        if (best_p == best_v and best_p != 0
                and margin_p > BRIDGE_MARGIN_MM and margin_v > BRIDGE_MARGIN_MM):
            q = PERMS[best_p]
            perm[t1:] = perm[t1:][:, list(q)]
            joins.append({**entry, "frame": t1, "evidence": "bridge",
                          "permutation": list(map(int, q))})
            continue
        refusals.append({**entry, "reason": "zone unvouched"})

    return perm, joins, refusals


def step_census(xyz: np.ndarray, ok_idx: np.ndarray) -> dict:
    """Adjacent-resolved-frame same-slot steps: the quantity the eye sees jump."""
    adjacent = np.diff(ok_idx) == 1
    a, b = xyz[ok_idx[:-1][adjacent]], xyz[ok_idx[1:][adjacent]]
    step = np.linalg.norm(b - a, axis=-1)
    step = step[np.isfinite(step)]
    return {
        "steps": int(step.size),
        "p99_9_mm": float(np.percentile(step, 99.9)),
        "max_mm": float(step.max()),
        "over_10mm": int((step > 10.0).sum()),
        "over_30mm": int((step > 30.0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0],
                        help="which stored c28 arms to stitch. The full grid is "
                             "the default: a claim about the join rule has to "
                             "hold at every lambda or it is lambda-dependent")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--margin-mm", type=float, default=MARGIN_MM)
    parser.add_argument("--arm-dir", type=Path, default=None,
                        help="directory holding the arms to stitch; default is "
                             "c28's. Input selection only - the join rule is "
                             "identical whatever it reads")
    parser.add_argument("--arm-prefix", default="C28-gated",
                        help="arm filename prefix, e.g. C32-gated or C33-gated")
    parser.add_argument("--tag", default=None,
                        help="output leaf suffix when the prefix alone is "
                             "ambiguous - two chains can both end in C33-gated "
                             "arms, and a stitch of one must never land on the "
                             "other's")
    args = parser.parse_args()

    paths = load_paths()
    arm_dir = args.arm_dir or (paths.work_dir / "c28_lambda_grid_transfer"
                               / args.experiment)
    suffix = args.tag or args.arm_prefix
    leaf = (args.experiment if suffix == "C28-gated"
            else f"{args.experiment}__{suffix}")
    out_dir = paths.work_dir / "c30_slot_stitch" / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    starts, counts = {}, {}
    degraded = np.zeros(c01.N_FRAMES, bool)
    for view in VIEWS:
        detections = c01.load(paths.work_dir, args.experiment, view, RUN)
        starts[view], counts[view] = detections.start, int(detections.frame_idx.size)
        best = best_scores(detections, detections.n_frames)
        degraded |= (best >= 0) & (best < SCORE_DEGRADED)
    # +-2 dilation: the 2026-09-03 window study - a wave burst interleaves
    # deceptive-looking frames between collapsed ones.
    for h in (1, 2):
        widened = degraded.copy()
        widened[h:] |= degraded[:-h]
        widened[:-h] |= degraded[h:]
        degraded = widened

    with provenance("slot_stitch", out_dir, name="c30") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("gate_mm", GATE_MM)
        record.param("max_gap_frames", GAP_FRAMES)
        record.param("gate_is_post_hoc", True)
        record.param("lambda_is_chosen", False)
        record.param("rule_version", 2)
        record.param("velocity_window_frames", VELOCITY_WINDOW)
        record.param("zone_merge_gap", ZONE_MERGE_GAP)
        record.param("zone_big_mm", ZONE_BIG_MM)
        record.param("score_degraded_floor", SCORE_DEGRADED)
        record.param("arm_dir", str(arm_dir))
        record.param("arm_prefix", args.arm_prefix)
        arms = {}
        for lam in args.lambdas:
            arm_path = arm_dir / f"{args.arm_prefix}__{args.experiment}__lam{lam:g}.npz"
            if not arm_path.is_file():
                raise SystemExit(f"no arm at {arm_path}")
            arm = np.load(arm_path)
            record.input(f"arm_lam{lam:g}", arm_path)

            started = time.time()
            xyz = arm["xyz"].astype(np.float64)
            resolved = arm["n_views_used"] >= 2
            ok_idx = np.flatnonzero(resolved & np.isfinite(xyz).any(axis=(1, 2)))
            before = step_census(xyz, ok_idx)

            perm, joins, refusals = stitch(
                xyz, ok_idx, args.window, args.margin_mm, GAP_FRAMES,
                degraded=degraded)

            rows = np.arange(xyz.shape[0])[:, None]
            fixed_xyz = xyz[rows, perm].astype(np.float32)
            fixed_residual = arm["residual_mm"][rows, perm]
            fixed_rays = arm["slot_rays"][rows, perm]
            fixed_source = arm["source"][rows, perm]
            after = step_census(fixed_xyz.astype(np.float64), ok_idx)

            out = Reconstruction(
                experiment=args.experiment, views=tuple(VIEWS), xyz=fixed_xyz,
                source=fixed_source, residual_mm=fixed_residual,
                margin_mm=arm["margin_mm"], n_views_used=arm["n_views_used"])
            derived = derive_gate(out)
            config = dataclasses.replace(derived, max_step_mm=GATE_MM,
                                         max_gap_frames=GAP_FRAMES)
            slot_track = METHODS["gated"](out, config)
            per_view = propagate_to_views(out, slot_track, starts, counts)

            path = out_dir / f"C30-gated__{args.experiment}__lam{lam:g}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=fixed_xyz,
                residual_mm=fixed_residual, margin_mm=arm["margin_mm"],
                n_views_used=arm["n_views_used"], source=fixed_source,
                slot_rays=fixed_rays, slot_perm=perm,
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"C30-gated__lam{lam:g}", path)

            # The slot labelling itself, as three permanent ids - what `c29`
            # colours and what an analysis would consume. Written twice, from
            # the stitched geometry and straight from `c28`'s, because the
            # difference between those two `c09` scores is the whole claim.
            # With permanent ids a `break` is impossible (the old id is always
            # alive), so every identity error is scored as a swap - B-label's
            # semantics from 2026-08-18, stated so the counts are read right.
            label_track = np.where(
                out.resolved[:, None],
                np.arange(N_ANIMALS, dtype=np.int32)[None, :],
                UNTRACKED).astype(np.int32)
            raw = Reconstruction(
                experiment=args.experiment, views=tuple(VIEWS),
                xyz=arm["xyz"], source=arm["source"],
                residual_mm=arm["residual_mm"], margin_mm=arm["margin_mm"],
                n_views_used=arm["n_views_used"])
            label_paths = {}
            for tag, geometry in (("label", out), ("label-unstitched", raw)):
                view_tracks = propagate_to_views(geometry, label_track,
                                                 starts, counts)
                label_path = (out_dir
                              / f"C30-{tag}__{args.experiment}__lam{lam:g}.npz")
                np.savez_compressed(
                    label_path, slot_track=label_track, xyz=geometry.xyz,
                    residual_mm=geometry.residual_mm,
                    margin_mm=geometry.margin_mm,
                    n_views_used=geometry.n_views_used, source=geometry.source,
                    **{f"track__{v}": view_tracks[v] for v in VIEWS})
                record.output(f"C30-{tag}__lam{lam:g}", label_path)
                label_paths[tag] = str(label_path)

            evidence = {"median": 0, "velocity": 0, "bridge": 0}
            for join in joins:
                evidence[join.get("evidence", "median")] += 1
            separations = [j["min_separation_mm"] for j in joins
                           if "min_separation_mm" in j]
            arms[f"{lam:g}"] = {
                "lambda_mm": float(lam), "path": str(path),
                "label_paths": label_paths,
                "joins": len(joins), "joins_by_evidence": evidence,
                "refusals": len(refusals),
                "refused_not_decisive": sum(
                    1 for r in refusals if r["reason"] == "not decisive"),
                "join_min_separation_median_mm": float(np.median(
                    separations)) if separations else None,
                "steps_before": before, "steps_after": after,
                "gate_derived_mm": derived.max_step_mm,
                "coverage_gated": coverage(slot_track, xyz.shape[0]),
                "join_list": joins, "refusal_list": refusals,
                "seconds": round(time.time() - started, 1),
            }
            a = arms[f"{lam:g}"]
            print(f"lambda {lam:<6g} joins {len(joins):3d} "
                  f"(med {evidence['median']} vel {evidence['velocity']} "
                  f"brg {evidence['bridge']})  refusals "
                  f"{len(refusals):3d}   steps>10mm {before['over_10mm']:,} -> "
                  f"{after['over_10mm']:,}   >30mm {before['over_30mm']:,} -> "
                  f"{after['over_30mm']:,}   gated tracks "
                  f"{a['coverage_gated']['tracks']}  top3 run cov "
                  f"{a['coverage_gated']['top3_run_coverage']:.2%}   "
                  f"{a['seconds']:.0f}s")

        record.param("arms", {k: {kk: vv for kk, vv in v.items()
                                  if kk not in ("join_list", "refusal_list")}
                              for k, v in arms.items()})
        (out_dir / "c30_summary.json").write_text(
            json.dumps({"experiment": args.experiment, "n_groups": 1,
                        "n_experiments": 1, "input": args.arm_prefix + " arms",
                        "arm_dir": str(arm_dir),
                        "window": args.window, "margin_mm": args.margin_mm,
                        "gate_mm": GATE_MM, "max_gap_frames": GAP_FRAMES,
                        "gate_is_post_hoc": True, "lambda_is_chosen": False,
                        "arms": arms},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")
    print("score each arm:  c09_gt_swap_validation.py --npz <arm>   "
          "then once with no --npz to restore the baseline")


if __name__ == "__main__":
    main()
