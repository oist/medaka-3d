# Stage B — the cross-view correspondence as a piecewise-constant sequence

**Date:** 2026-08-18
**Status:** rule fixed **before** implementation; results appended below, after
the rule and after the code. Everything above the horizontal rule was written
before the first number was read.
**Builds on:** [2026-08-18 2D first](2026-08-18-2d-first-tracking.md),
[NEXT_SESSION_stitching](../NEXT_SESSION_stitching.md)
**Tools:** new `c19`; new functions in `src/medaka3d/correspond.py`. `c03`, `c13`,
`c16` and `c18` are not modified.

## What is being built and why it is not a smoother

`reconstruct` and `reconstruct_multinode` both solve the cross-view
correspondence **once per frame, independently**. Three attempts to add a
temporal term on top of that are retired in the previous decision's Addendum 2,
and the reason the best of them failed is arithmetic: a modal vote over a window
follows a genuine change only after it has held half a window, so it spends most
of its effect **delaying correct reassignments**. Swaps went 8 → 67.

Stage B is the lag-free version. The correspondence over the whole recording is
one object — a piecewise-constant sequence — and it is chosen by minimising

    J(s) = SUM_t cost_t(s_t)  +  lambda * #{t : s_t != s_{t-1}}

over all sequences at once. There is no window, no vote and no direction of
time: a change at frame *t* is paid for by evidence on both sides of *t*
simultaneously. The minimiser is exact by dynamic programming (Viterbi) over
36 states × T frames; nothing is greedy and nothing is approximated.

## The state space, and why it is constant across frames

A state is **not** a choice of detection rows. Detection rows are the detector's
own ordering within a frame and mean nothing between frames, so "the
correspondence stayed the same" would be undefined on them.

Stage A gives the meaning. `c16` assigns every detection a **2D track label** in
{0, 1, 2} per camera, and those three labels persist for the whole recording —
3 tracks per view, 99.3–99.7% of instances assigned. Measured for this
document: on every frame that holds exactly three detections, the three labels
are exactly a permutation of {0, 1, 2} in all three views (rate 1.000, no
exceptions). So the labels are a well-defined per-frame naming.

A **state** is then the pair of permutations

    s = (sigma1, sigma2),  sigma_v : top label -> side_v label

with `top_cam` fixed as the reference, giving 6 × 6 = **36 states**, and the
same 36 states mean the same thing at every frame. This is what makes
"#changes" a countable quantity rather than a coordinate artefact.

Implementation consequence: permute each view's detections into **label order**
once, and every existing routine in `medaka3d.reconstruct` /
`medaka3d.correspond` then operates in label space unchanged, because
`hypotheses(3)` already enumerates exactly these 36 with view 0 fixed. The
detector's row for each label is kept so `source` still records rows and `c09`
reads the output unmodified.

## The per-frame cost

`cost_t(s)` is `c18`'s multi-node cost: the point-to-ray triangulation residual
of the three animals under state *s*, summed over the animals, summed over the
nodes that every participating view supplies, and **divided by the number of
contributing nodes**. The division is not a change of objective — within a frame
the contributing-node set is the same for all 36 states, so it cannot move the
per-frame winner — but it keeps the cost in millimetres whatever a frame
supplies, which is what lets one lambda mean the same thing on a one-node frame
and a two-node frame.

Three cases:

* **Three views complete.** All 36 states have distinct costs.
* **Two views complete.** The absent view's permutation is unconstrained, so the
  36 costs collapse to 6 distinct values in blocks of 6. The DP is left to
  choose within the block, and it chooses by persistence. This is exactly the
  frames where the correspondence has no third camera to check it (13.9% of
  K_1_2, and 43.7% of the frames inside a measured swap interval), and carrying
  the state through them is the main thing stage B is for.
* **Fewer than two views.** Cost 0 for every state: no evidence, so the state is
  carried across at no charge.

At `lambda = 0` this reproduces `c18` frame for frame. That is asserted in the
tool, not hoped for — if it does not hold, the state space or the row/label map
is wrong, which is the failure mode that cost the most time in the previous
session.

## Choosing lambda without the ground truth

### The rule as handed over is provably unsatisfiable

The handoff says: sweep lambda and take the value whose change count matches the
**635 changes over 284,305 adjacent pairs (0.2234%)** measured on 2026-08-18.

That cannot be done, and the reason is a property of the objective rather than
of this dataset. Write C(lambda) for the number of changes on the minimising
sequence. C is **non-increasing** in lambda — the standard monotonicity of a
change-point penalty — and at lambda → 0+ the DP returns a cost-minimal
sequence, which is the per-frame independent winner with ties broken toward
persistence. So

    C(lambda) <= C(0+) <= N0   for every lambda >= 0,

where N0 is the change count of the unconstrained per-frame winner. Any target
equal to the unconstrained rate is reached only at lambda = 0, i.e. by not
smoothing at all. Sweeping to hit it is a null operation.

There is a second, independent problem with the number itself. **The tool that
produced 635 no longer exists** — it was deleted with the retired design 3, and
no surviving code in `tools/` or `src/` computes it. It was measured on whatever
labelling that design used, which was not `c16`'s (`c16` did not exist when the
0.2234% was measured). On the state space defined above, the quantity is not
even expected to be similar: with three 2D tracks alive for the whole recording,
the *true* correspondence changes only when a 2D tracker reassigns an identity,
which is a rare event, not one every 448 frames.

N0 is therefore recomputed here on this state space and reported. It is a
**diagnostic, not a selector**.

### The replacement rule, fixed here

Lambda is chosen by **held-out node cross-validation**. No human annotation is
read at any point in the selection.

1. Fit the DP with the **mouth alone** in `cost_t`. Score the resulting sequence
   by the mean **pectoral** residual per frame, over the frames where the
   pectoral is available. Repeat with the roles exchanged: fit on the pectoral,
   score on the mouth.
2. Report lambda in the scale-free unit **kappa = lambda / m**, where *m* is the
   median runner-up margin of the per-frame cost used in that fit. *m* is
   measured, not chosen, and it is what makes a kappa selected on a one-node fit
   transferable to the deployed two-node fit, whose margins are roughly three
   times larger (`c18`: p1 1.240 → 3.635 mm on three-view frames).
3. `kappa*` = the geometric mean of the two directions' minimisers.
   `lambda* = kappa* × m(two-node)`.

Why this has an interior optimum, by construction rather than by luck: too small
a kappa makes the sequence follow fluctuations of one node that the other node
does not corroborate, and the held-out residual rises; too large a kappa refuses
changes the other node does corroborate, and the held-out residual rises again.
Both directions cost, so the criterion is not monotone the way a change count
must be.

**If either CV curve turns out to have no interior minimum** over the grid, this
rule has failed and the fallback is the widest-plateau rule: take kappa at the
geometric centre of the widest interval of log kappa over which C(kappa) is
constant. Should that happen it will be stated in the results as a fallback that
was triggered, not presented as the rule.

**Nothing in the selection may consult `female_sex_gt.csv`, `c09`, `c14` or
`c15`.** The ground truth is read once, afterwards, to apply the decision rule
below.

### Reported alongside, deciding nothing

* `N0` — changes of the unconstrained per-frame winner on this state space.
* `C(lambda*)` — changes the DP kept.
* The count of 2D label discontinuities across the three views, which is the
  number of occasions on which a change could be legitimate. It bounds the true
  change count above; it does not estimate it.

## What decides — the four clauses, unchanged, plus clause 5

Measured on K_1_2 against the female annotation, `tools/c09_gt_swap_validation.py`:

1. **Swaps < 8** on the `c09` unit. An increase is disqualifying regardless of
   every other number.
2. **Fragmentation not worse.** *Weak as written and replaced by clause 5* — track
   counts across these runs were produced under different derived gates
   (`max_step_mm` 10.2 / 8.3 / 5.0), so the comparison is between gates as much
   as between methods. Track count is still reported, with its gate beside it,
   and it decides nothing on its own.
3. **Cross-view agreement >= 0.9677**, tail-derived.
4. **Reconstruction coverage >= 96.31%.**
5. **NEW — per-animal trajectory coverage.** Two definitions, both reported for
   every arm so that nothing is ever compared across definitions:
   * `top3_track_coverage` — frames held by the three largest track IDs, over
     3 × T. This is the number the handoff quotes as **15.6%** for `gated`.
   * `top3_run_coverage` — frames inside uninterrupted runs of at least one
     second, over 3 × T. This is the number the handoff quotes as **94.6–97.0%**
     for `c16`'s 2D tracks.
   The standard to argue against is 2D's, on the matching definition.

Two arms are produced from the same DP output and both are scored:

* **B-gated** — `gated` tracking on the stage-B reconstruction, with the gate
  derived from that reconstruction exactly as `c13`/`c18` derive theirs. Directly
  comparable to `c03`, `c13` and `c18`.
* **B-label** — track ID = the `top_cam` 2D label, held constant. The cheapest
  possible stage C, and the arm clause 5 exists to test. Its known weakness is
  named in advance: at a state change at least one camera relabelled, and naming
  identity after `top_cam` is wrong whenever the camera that relabelled was
  `top_cam`. Stage C is what fixes that; this arm measures how much is left to
  fix.

This is **not** the retired design 1. Design 1 forced one constant
correspondence on the whole recording. Here the correspondence is free at every
frame and only *changes* are charged; the constant-label part is a naming of the
output, applied after the correspondence is solved.

## The sample, which does not improve

**n_groups = 1, n_experiments = 1, female only.** K_1_2 is the only experiment
with identity annotation; there is no M1/M2 ground truth, so no male's swap rate
is measured here. Swap counts of 2–8 are small samples and **no ranking built on
them is stable** — a difference of one or two swaps between arms is not a result.
Every number this stage produces carries that qualifier.

---

# Results, same day — measured after the rule above was fixed

`tools/c19_viterbi_correspondence.py`, K_1_2, anchor pectoral, nodes mouth +
pectoral. **n_groups = 1, n_experiments = 1, female only.**

## The state space is the one it claims to be

| check | result |
|---|---|
| 2D labels form a clean permutation of {0,1,2} on 3-detection frames | **100.0000%**, all three views |
| `lambda = 0` reproduces `c18`'s correspondence, frames with evidence | **100.0000%** of 340,001 |

The second is the one that matters. Slots are indexed by `top_cam` **label**
here and by `top_cam` **row** in `c18`, so the two list the slots in a different
order; the comparison is of the set of per-slot view triples, which is what
"the correspondence" means. Element-wise it would have reported a disagreement
that is not one.

## Lambda, chosen with no ground truth

Both cross-validation curves have an **interior** minimum, so the pre-registered
rule stands and its fallback was not triggered.

| fit | held out | m (median margin) | best kappa | grid position | held-out mean | changes there |
|---|---|---|---|---|---|---|
| mouth | pectoral | 12.388 mm | 0.1 | 12 of 24 | **1.79229 mm** | 677 |
| pectoral | mouth | 11.408 mm | 0.5623 | 15 of 24 | **1.32153 mm** | 808 |

`kappa* = 0.2371`, `m(two-node) = 11.663 mm`, **`lambda* = 2.766 mm`**. The
curves are shallow — the mouth-fit held-out cost moves 1.873 → 1.792 → 2.201 mm
across six decades of kappa — so the minimum is real but not sharp, and a
neighbouring grid point would not have been a different decision.

## The diagnostics that decided nothing

| quantity | value |
|---|---|
| `N0`, changes of the unconstrained per-frame winner | **1,941** over 340,001 pairs (**0.5709%**) |
| the handoff's figure | 635 over 284,305 (0.2234%) — **not reproducible**, tool deleted, labelling predates `c16` |
| 2D label re-acquisitions across the three views | 9,045 (upper bound on legitimate changes) |
| changes the sequence kept at `lambda*` | **973**, giving **974 segments**, median **349 frames** (2.1 s) |

`N0` is 2.6x the handoff's number and on a different state space; neither is
evidence about the other. As derived above, no lambda could have matched either.

## The clauses

`tools/c09_gt_swap_validation.py` against `female_sex_gt.csv`. The baseline row
is the restored run; interval counts differ by one between arms because `c09`
drops anchors whose two side views name different slots, and that set depends on
the correspondence being scored.

| arm | intervals | preserved | break | **swap** | tracks | gate | resolved | top3 track cov | top3 runs >= 1 s | tail agree (s1 / s2) |
|---|---|---|---|---|---|---|---|---|---|---|
| `c03` mouth 1 node | 296 | 269 | 19 | **8** | 511 | 10.2 | 95.95% | 18.90% | 87.17% | 0.9668 / 0.9686 |
| `c13` pectoral 1 node | — | — | — | **4** | 394 | 8.3 | 97.53% | 19.04% | 92.63% | 0.9667 / 0.9682 |
| `c18` 2 nodes | — | — | — | **2** | 475 | 5.0 | 97.53% | 15.18% | 92.71% | 0.9672 / 0.9692 |
| **`c19` B-gated** | 297 | 274 | 21 | **2** | 532 | 4.0 | 97.53% | 14.67% | 92.62% | **0.9674 / 0.9694** |
| **`c19` B-label** | 297 | 247 | **0** | **50** | **3** | — | 97.53% | **97.53%** | 93.68% | 0.9674 / 0.9694 |

**B-gated**

1. **Swaps 2 < 8. Passes** — and is *equal* to `c18`, not better. With 2 swaps in
   one experiment, 2 against 2 is not a difference and neither is 2 against 4.
2. Superseded by clause 5. Track counts are 532 at a 4.0 mm gate against `c18`'s
   475 at 5.0 mm; the gates differ, so the comparison is between gates.
3. **Tail-derived agreement 0.9674 / 0.9694, above `c03`'s 0.9668 / 0.9686 on
   both pairs. Passes.** The two decision documents word this clause differently
   and they disagree on the verdict: read as the *pectoral*-derived 97.22% from
   `c04`, B-gated's 0.9698 / 0.9720 does **not** clear it — and neither does
   `c18`'s 0.9704 / 0.9727. The tail reading is the meaningful one for these
   arms, because `c13`, `c18` and `c19` all *solve* the correspondence on the
   pectoral, so pectoral-derived agreement is partly construction for them while
   the tail stays independent. That is a reason to prefer the tail reading, not
   a reason to ignore the pectoral one.
4. **Coverage 97.53% >= 96.31%. Passes** — identical to `c13`/`c18`, as it must
   be: the sequence changes which state a frame takes, never whether it resolves.
5. **Fails to advance.** 14.67% against `c18`'s 15.18% and `c13`'s 19.04% on
   the track definition; 92.62% against 92.71% on the run definition. Stage B
   does not move clause 5 at all.

**B-label is disqualified by clause 1: 50 swaps against a ceiling of 8.** It was
built to test clause 5 and it reaches 97.53% trajectory coverage — the whole
resolved recording, three tracks — so the coverage the science needs is
available. What is not available is the naming. Note also that B-label cannot
*produce* a break: with three permanent IDs, every identity error is a swap by
construction, so its 50 and `gated`'s 2 are not counting over the same outcome
space and only the disqualification transfers.

## What this actually established, and what it did not

**The correspondence was not what limited the trajectories.** Stage B did what
it was designed to do — the correspondence is now a piecewise-constant sequence
with 973 changes instead of 1,941 independent flips, chosen exactly and without
lag — and **every downstream clause is unchanged to within noise**. The
fragmentation comes from `gated` refusing links at its 4 mm distance gate, not
from the correspondence flipping. Three retired designs and this one all
attacked the correspondence; the measurement now says the correspondence was
not the bottleneck.

**A margin that looks worse and is not.** `c19` reports the runner-up margin
divided by the number of contributing nodes, where `c18` reported the sum. Ratio
`c18`/`c19` is exactly 2.000 on 90.2% of frames and exactly 1.000 on 9.8% — the
frames with one node — so the change is a unit, not a loss of discrimination,
which `lambda = 0` reproducing `c18` at 100% already implies.

**The real deliverable is the segmentation.** Within one of the 974 segments the
correspondence is constant, which means no camera relabelled inside it — a
single camera relabelling changes sigma and ends the segment. So identity is
internally consistent within a segment, and the remaining problem is **973
joins**, not 348,600 frames and not 475 fragments. B-label is the null join rule
— "keep the `top_cam` label across the boundary" — and it costs 50 swaps in 297
sampled intervals. That is the number a real join rule has to beat.

**Not established.** Nothing here says the 973 changes are the *right* changes;
they are the ones a cross-validated penalty kept, and 9,045 2D re-acquisitions
bound the truth above without estimating it. Nothing is promoted: `c03` still
runs the pipeline, and on the five clauses `c18` and `c19` B-gated are
indistinguishable on one experiment's female.
