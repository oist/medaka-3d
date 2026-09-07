# 2D first, 3D for the hard parts — and the evaluation that made the old order look fine

**Date:** 2026-08-18
**Status:** decision rule fixed **before** implementation. No result in this document.
**Builds on:** [2026-08-13 tracking](2026-08-13-tracking-3d.md),
[2026-08-15 top-down](2026-08-15-topdown-3node-tracking.md)
**Tools:** `c08`, `c09`, `c10`, `c11`; new `src/medaka3d/track2d.py`

## What changed, and it is not a tuning question

Two things were established on 2026-08-17/18 that invalidate the order the
pipeline runs in.

**1. Human identity ground truth exists and was not used.** `CLAUDE.md` says
ground truth labels only the female, and that was read as "there is nothing to
validate a tracker against". There is:
`archive/medaka-mating-pipeline/annotations/fish_sex_gt/20251225_K_1_2_FCMC/`
holds **300 confirmed female identifications** plus 25 crossing-branch
adjudications. It was made on round-7 output under a refractive calibration, but
pixel coordinates and frame numbers are properties of the video, not the model:
the nearest current mouth to an annotated one is a median **1.27 px** away in
side1 and **1.60 px** in side2. Every tracking decision taken before this date
was taken without consulting it.

Measured on it (`c09`, K_1_2, female only, n_groups = 1): **8 swaps** — her ID
handed to another fish while the old ID stayed alive — against 19 breaks and 269
preserved, over 296 intervals.

**2. The per-view metrics were never independent.** `reconstruct` solves the
cross-view correspondence **once per frame with no temporal term**, and
`track.propagate_to_views` pushes the 3D track IDs back onto the 2D detections.
So the three cameras agree by construction — `c04`'s own docstring says a rate of
1.000 there "measures nothing" — and every per-view number in `c04`, `c08` and
`c09` is a function of one 3D assignment. The evaluation had **one degree of
freedom reported as four**.

## The failure this produces, with the instance

`c11`'s `SWAP_i62551_x62446` (K_1_2, frames 62,446-63,502, 1,056 frames). Two
fish swim parallel at a closest 3D approach of 6.9 mm. Measured per view:

| view | 2D separation, median | p10 | frames under 20 px | links contradicting 2D continuity |
|---|---|---|---|---|
| top | 36.6 px | 27.5 | 1.5% | **1** of 1,055 |
| side1 | 64.8 px | 30.6 | 0.4% | **3** of 1,055 |
| side2 | 72.4 px | 43.8 | 0.0% | **1** of 1,055 |

The animals are 0.33-0.77 body lengths apart and never occlude. **A human
following either camera keeps them apart for the whole crossing.** The per-frame
3D correspondence flips on one to three frames, and because identity is carried
only by that correspondence, the flip is permanent and silent: all three track-ID
strips stay unbroken while the female changes track.

## The order this implies

Tracklets first, in 2D, per camera; 3D only where 2D declines.

Feasibility measured before deciding (K_1_2, confident links only — gate = half
the median body span, plus a refusal when the runner-up permutation is within
half a gate):

| view | tracklets | median length | **>= 1 s** | **frames covered by those** |
|---|---|---|---|---|
| top | 2,012 | 14 f | **439** | **82.3%** |
| side1 | 2,821 | 14 f | **533** | **75.5%** |
| side2 | 2,388 | 17 f | **528** | **79.3%** |

About 500 confident tracklets per camera cover four fifths of the recording. The
decision count falls from 348,600 per-frame correspondences to a few hundred
tracklet joins per view, and each join is made with a whole tracklet's motion on
both sides rather than one frame's geometry.

The join score combines both sources, because neither alone is sufficient: 2D
alone breaks at occlusion, 3D alone flips on a single frame. Time-series
continuity within a view, plus triangulation residual of the joined hypothesis
across views.

## Decision rule — fixed here, before the code exists

The 2D-first tracker replaces `gated` only if **all** of the following hold on
K_1_2 against the female annotation:

1. **Swaps fall.** Fewer than 8 on the `c09` unit. An increase is disqualifying
   regardless of every other number.
2. **Fragmentation does not worsen.** Tracks per animal after joining, no higher
   than `gated`'s.
3. **Cross-view disagreement falls.** This becomes a real measurement only under
   the new architecture, so `gated`'s figure is quoted as the constructional
   bound it is, and the new one must beat the *pectoral-derived* 97.22% that
   `c04` computed independently.
4. **Reconstruction coverage does not fall** below the current 96.31%.

If swaps fall and fragmentation worsens, that is a **trade-off and is not a
win** — same wording as `c04`, same consequence.

**n_groups = 1, n_experiments = 1, female only.** K_1_2 is the only experiment
with identity annotation. Male-vs-male crossings remain unmeasurable. Whatever
this rule selects, it selects on one fish in one trial, and the rollout question
is separate.

## What must be re-measured afterwards, not assumed

`c04`'s tracker choice rested on **track jumps**, which `c09` measured against
the female annotation and found does not indicate swaps at all — median 0 whether
or not a swap occurred, recall 12.5%. `c06`'s architecture comparison rested on
the same metric. Neither conclusion is overturned here, and neither may be
carried forward as validated. They are re-opened.

`c08`'s crossing strata are a function of the 3D assignment and must be recomputed
once the 2D tracklets exist.

---

## Addendum, same day — SLEAP's own tracker, and what it says about the swaps

`c02` (2026-08-13) is **retracted** on the point that matters here. It wrote:
*"This tool therefore reconstructs first and tracks second. It does not
implement a 2D route: the brief asked for one, and building both would make
neither answerable."* Its pre-registered rule was degenerate by its own account
— both temporal ambiguity rates came out at exactly 0.000% — and it was decided
while believing no human identity ground truth existed.

`sleap_nn.tracking.Tracker` was run per camera on the cached `s16` detections
(`c16`), with `features='keypoints'` — the multi-node similarity the
experimenter proposed — `scoring_method='oks'`, `hungarian` matching, and
`max_tracks=3`, which is a fact about the experiment rather than a guess. The
tail is excluded by default: it is the node `s10` measured part-mixing on, so a
tracker that trusts it is least reliable exactly at a crossing.

**top_cam, 348,600 frames.** 3 tracks, 99.29% of instances assigned, and runs of
at least one second covering **97.0%** of frames — against 75.5% for the
cardinality-strict linker in `medaka3d.track2d`, whose binding cut was frames
not holding exactly three detections.

**At the 18 confirmed swap frames in top_cam, the 2D tracker refused the swap
18 times out of 18.** The test is not whether its ID changed — the 3D assignment
asserts two different animals are one, so a correct 2D tracker *must* differ.
The test is where it put the animal instead, and in every case the ID from the
frame before reappeared on the **other** instance. An earlier version of the
check compared the two IDs directly and reported every refusal as agreement,
i.e. exactly backwards; the tool now records the three outcomes separately.

This does not show the 2D tracker is *right*, only that it disagreed with the 3D
correspondence at frames the experimenter confirmed by eye to be swaps. It does
settle the architectural question: **the 2D evidence was present and was
discarded**, because per-view IDs are the 3D correspondence pushed back onto the
detections and nothing existed to disagree with it.

**Still outstanding:** side1 and side2 (44 of the 62 candidate frames);
the tail-included variant; and the join rule itself — using 2D continuity as a
constraint on the cross-view correspondence rather than as a diagnostic. The
four-clause decision rule above is unchanged and none of its clauses is settled
by this addendum.

---

## Addendum 2 — the temporal constraint failed its own clause, and why

Three implementations of "use 2D continuity to constrain the correspondence"
were built and measured against the female annotation. All three are retired;
the numbers are kept here because each failure names a real property of the data.

**1. A global map from each camera's 2D labels to a common identity.** Refuted
by measurement: encoding the correspondence the cameras' labels induce, **all 36
possible correspondences occur** and the most common holds only **6.03%** of
frames. Forcing one map rewrote 93% of frames and drove the three-view residual
from 0.50 mm to 25 mm. Between *adjacent* frames the correspondence changes on
only **0.2234%** of pairs — about once every 448 frames (2.7 s). Locally stable,
globally not.

**2. A modal reference over a window, fed by the accepted choice.** Self-locking:
once the constraint is active every entry in the window came from a choice the
window dictated, so the mode cannot move and an early error is protected for the
whole recording. 96% of frames rewritten, residual 27 mm.

**3. The same window, fed by the *unconstrained* choice.** Geometrically sound —
three-view residual 0.505 mm against the baseline's 0.496, coverage unchanged at
97.53%, fragmentation improved (237 tracks against 394). **And it failed clause 1
outright: swaps 8 -> 67.** Clause 1 is "fewer than 8; an increase is
disqualifying regardless of every other number", fixed before the code existed.

Why it failed is arithmetic. A modal vote over 50 frames follows a genuine change
only after it has held ~25 frames, so every real change is resisted for 0.15 s.
At 0.22% changes per transition that predicts ~5.5% of frames fighting a correct
change, against 7.3% measured. The rule was not blocking swaps; it was mostly
delaying correct reassignments, and `max_step_mm` rose from 8.3 mm to 98.0 mm as
forced correspondences threw the 3D positions.

**What this does not retire** is the observation that motivated it: at 62 frames
the experimenter confirmed as swaps, the 2D trackers put the animal on the other
instance 60 times. The evidence is real; smoothing the correspondence in time is
the wrong way to use it. A test of whether a *change* is warranted is not the
same as a vote on what the correspondence usually is.

## Addendum 3 — the correspondence was using one keypoint the whole time

`reconstruct`'s input contract is ``uv_by_view: view -> (n_frames, 3, 2)``:
**one point per animal**. The cross-view correspondence has rested on a single
keypoint since `c02`, while the per-camera 2D tracker matches a whole pose. The
two halves of the pipeline were using different amounts of the same detections,
and three attempts were made to add a temporal constraint on top before this was
noticed.

Summing the triangulation residual over the mouth **and** the pectoral, K_1_2:

| cameras | nodes | runner-up margin p1 | p10 | margin < 1 mm |
|---|---|---|---|---|
| 3 | pectoral | 0.989 mm | 3.350 | 1.01% |
| 3 | mouth | 1.240 mm | 3.799 | 0.67% |
| 3 | **both** | **3.635 mm** | **8.553** | **0.03%** |
| 2 | pectoral | 0.006 mm | 0.588 | 13.04% |
| 2 | mouth | 0.010 mm | 1.519 | 7.66% |
| 2 | **both** | **0.252 mm** | **4.503** | **3.59%** |

The gain is largest on two-view frames, which is where there is no third camera
to contradict a wrong pairing and where one node leaves the correct
correspondence and its runner-up 0.006 mm apart. The winner changes on **6-11%**
of frames, so this is not a refinement of an answer that was already right.

`medaka3d.correspond.reconstruct_multinode` implements it, with a view judged
complete on the **anchor** alone so a missing secondary keypoint costs that
node's evidence rather than a whole camera — the failure `c13` measured. The
four-clause rule is unchanged and decides.
