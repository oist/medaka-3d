# Stage C — the correspondence without "exactly three detections per view"

**Date:** 2026-08-19
**Status:** rule fixed **before** the arms were run; results appended below the
horizontal rule. Everything above it was written before any arm was scored.
**Builds on:** [2026-08-18 Viterbi correspondence](2026-08-18-viterbi-correspondence.md),
[2026-08-18 2D first](2026-08-18-2d-first-tracking.md)
**Tools:** new `c20`; new functions in `src/medaka3d/correspond.py`; `valid=` on
`medaka3d.calib.triangulate`. `c03`, `c13`, `c16`, `c18` and `c19` are **not**
modified.

## The limiter, and where it actually lives

The predecessor repository ran SLEAP inference with `max_instances = 6`, and the
detector here is bottom-up and caps nothing at all — it reports up to **eight**
instances in a frame. So the cardinality limit was never in inference. It is in
*this* repository's correspondence stage, in one line repeated across `c03`,
`c13`, `c18` and `c19`:

```python
three = counts == N_ANIMALS          # a view may take part only on these frames
```

Its origin is indexing. Detections were addressed by **row**, so with three rows
and three slots a correspondence is a permutation, and a frame holding four rows
or two has no permutation to be. Measured on K_1_2's bottom-up run, that costs:

| instances in the frame | 0 | 1 | 2 | **3** | 4 | 5+ |
|---|---|---|---|---|---|---|
| `top_cam` | 115 | 559 | 12,841 | **327,977** | 6,925 | 183 |
| `side1_cam` | 0 | 3,770 | 18,215 | **321,538** | 4,972 | 105 |
| `side2_cam` | 71 | 4,146 | 14,146 | **327,423** | 2,715 | 99 |

A four-detection frame is discarded although all three animals are in it. A
two-detection frame is discarded although two animals are.

## What replaces it — already in the pipeline

`c16`'s per-camera 2D tracker holds exactly **three track labels per view**, for
the whole recording. The label, not the row, says which detection is which, and
a label is either present in a frame or absent — no counting anywhere. A fourth
detection carries no label and is ignored; a two-detection frame supplies two
labels and the third is absent.

`c19` already indexes its state space by those labels. So **the algorithm does
not change**: the same 36 states, the same objective

    J(s) = SUM_t cost_t(s_t)  +  lambda * #{t : s_t != s_{t-1}}

minimised exactly by the same Viterbi, with lambda chosen by the same held-out
node cross-validation and no ground truth. The only thing that changes is which
detections are allowed to reach it.

Under the label rule a view is usable on **93.7–96.1%** of frames against
92.2–94.1%, and all three views on **87.5%** against 83.7%.

## Participation becomes per animal, not per camera

`c13` established that a missing keypoint must cost a node, never a camera. The
same argument runs along the other axis. Under the label rule the slot for top
label *a* is triangulated from whichever views hold *its* label under the state
being scored, so one camera losing one animal weakens **that animal's** slot by
one ray and leaves the other two at three rays each.

`triangulate` therefore takes a `valid=` mask: a ray marked absent is left out
of the normal equations entirely and the residual averages over the rays that
took part. Without the mask its behaviour is unchanged — any non-finite ray
still makes the whole row NaN, which is the right answer when the caller
intended all the rays to be there.

The per-frame cost becomes the **mean over the (node, slot) terms the frame
supports**, in mm. On a frame where every view holds every label of every node
that is exactly `c19`'s cost divided by three, so the per-frame winner, the
runner-up *ratio* and therefore the cross-validated kappa are unchanged and
lambda simply moves by the same factor. `c20` asserts the equality against a
live `state_costs` run rather than trusting it, and asserts through `c18`'s
stored output that the correspondence itself is reproduced at lambda = 0.

## The one bias, stated in advance

When two views are each short of a label, different states leave different
numbers of slots with fewer than two rays, so the divisor is not constant across
states within such a frame and a state that explains less can be charged less.

The alternative — forbidding states that explain fewer detections — is worse,
and for a reason specific to this experiment: when both side cameras lose the
**same** animal, the correct state is exactly the one that leaves a slot short.
Forcing maximum explanation would systematically mispair precisely the frames
where an animal is genuinely hidden from two cameras.

So the bias is accepted, and `c20` reports the share of frames it can act on.
Every one of those frames is a frame the old rule discarded outright, and they
enter a sequence model that already weighs persistence against weak evidence.

## What cannot be recovered, so that the ceiling is honest

An animal seen by fewer than two cameras has no 3D position, whatever the
correspondence says. `n_views_used` is therefore the **minimum** over the three
slots and `resolved` keeps its meaning — every animal has a position — so a
frame where one animal is visible to one camera is still unresolved. This puts a
hard ceiling on coverage that is a property of the detections, not of the
solver, and `c20` prints it beside the achieved figure.

## What decides — the five clauses, unchanged from `c19`

Measured on K_1_2 against the female annotation, `tools/c09_gt_swap_validation.py`:

1. **Swaps < 8** on the `c09` unit. An increase is disqualifying regardless of
   every other number. The incumbents are `c18` = 2 and `c19` B-gated = 2, so at
   this sample size "not worse" is the only claim available.
2. **Fragmentation** reported with its derived gate beside it; decides nothing
   on its own.
3. **Cross-view agreement ≥ 0.9677**, tail-derived. Expressed in slots rather
   than in detector rows, because rows are no longer a permutation of {0,1,2};
   the question asked is `c04`'s, unchanged.
4. **Reconstruction coverage ≥ 96.31%.**
5. **Per-animal trajectory coverage**, both definitions, never compared across
   definitions. The standard to argue against is `c16`'s 2D: 94.6–97.0% on the
   run definition.

Two arms, the same two as `c19`:

* **C-gated** — `gated` on the stage-C reconstruction, gate re-derived from it.
* **C-label** — track ID = the `top_cam` 2D label held constant. Known in
  advance to be disqualified by clause 1 (`c19`'s B-label took 50 swaps); it is
  carried so clause 5 has its ceiling.

**n_groups = 1, n_experiments = 1, female only.** Swap counts of 2–8 are small
samples and no ranking built on them is stable. This document can show a change
is not harmful; it cannot show it helps.

---

## The sample, which does not improve

One trio, one day, the female only. 298 anchors, 298 intervals. Every clause
verdict below rests on counts between 1 and 52 and none of them is a stable
ranking.

## The reduction to `c19` holds exactly

`c20` re-runs `state_costs` live and compares:

* On the **75.37%** of frames where every view holds every label of every node —
  identical evidence — `3 × new cost = old cost` to `3.05e-05` mm, the float32
  round-off, and the per-frame winner agrees on **100.0000%**.
* At lambda = 0 the correspondence reproduces `c18`'s stored `source` on
  **100.0000%** of those frames.
* On the **8.31%** of frames where a view is short of one mouth, the winner
  moves on **1.50%**. That difference is intended: `c19` drops the mouth from the
  whole frame, stage C keeps it for the two animals whose mouths are there.

kappa\* is **0.2371** — the *same two grid points* `c19` selected, from an
independent fit. The scale-free penalty transfers exactly as the previous
document predicted. lambda\* is 0.7667 mm, which is 2.30 mm on `c19`'s ×3 scale
against its 2.766 mm; the difference is entirely in *m*, the median runner-up
margin, which falls from 3.888 to 3.233 mm because the population it is measured
over now includes the weak frames.

## What the reconstruction gained

| | `c19` | `c20` | ceiling |
|---|---|---|---|
| view usable — top / side1 / side2 | 94.08 / 92.24 / 93.93% | **96.12 / 93.69 / 94.73%** | — |
| frames resolved | 97.53% | **98.55%** | 98.93% |
| frames with three cameras on every animal | 83.68% | **87.50%** | — |
| animals on two cameras rather than three | — | 4.04% of resolved slots | — |

98.93% is the hard ceiling: on the other 1.07% some animal is visible to fewer
than two cameras and has no 3D position under *any* correspondence. Stage C
reaches 98.55% of the 98.93% available.

## What the correspondence did — the finding

**The admitted frames triple the number of state changes the sequence keeps:
973 → 3,090.** N0, the unconstrained per-frame count, goes 1,941 → 6,004.

This is **not** lambda's rescaling. Holding lambda at `c19`'s own effective
penalty (0.9219 mm on this scale) still keeps **2,939**; even at 4.0 mm — four
times the cross-validated value — it keeps 1,753. The extra changes come from
the frames themselves.

The mechanism is in the objective. Every frame contributes its cost with the
same weight, because the cost is a *mean* over the terms the frame supports —
the normalisation chosen so that one lambda means one thing everywhere. A frame
resting on four rays therefore pushes exactly as hard as one resting on nine. A
long run of weakly-informative frames each preferring the wrong state by 0.01 mm
outvotes a lambda of 0.77 mm after 77 frames, and there are now a great many
such frames.

## `derive_gate` measures renaming, not movement

The derived gate goes **4.0 mm → 72.2 mm**, and that single number produces most
of the table below. `derive_gate` takes the 99.9th percentile of the
frame-to-frame 3D step, but a step across a **correspondence change** is a slot
being renamed, not an animal moving:

| p99.9 of the per-slot step | `c19` | `c20` |
|---|---|---|
| over pairs where the state held | 3.77 mm | 23.03 mm |
| over pairs where the state changed | 82.67 mm | 177.69 mm |
| over all pairs — **what the gate uses** | **3.97 mm** | **72.16 mm** |

With `c19`'s 870 changes in 340,000 pairs (0.26%) the contaminated pairs sat
below the 99.9th percentile and the gate was unaffected. With 2,616 in 343,000
(0.76%) they set it. The gate is not a property of this reconstruction being
worse; it is a statistic estimated on a sample that includes non-movements, and
`c20` is the first run with enough changes to expose it.

## The clauses

`tools/c09_gt_swap_validation.py` against `female_sex_gt.csv`. The last row is
**post-hoc**: the same stage-C reconstruction with the gate held at `c19`'s
value, which is the controlled version of a comparison that otherwise compares
gates as much as methods.

| arm | gate mm | preserved | break | **swap** | tracks | resolved | 3 cams | top3 track | top3 runs ≥ 1 s | tail agree (s1 / s2) |
|---|---|---|---|---|---|---|---|---|---|---|
| `c03` mouth 1 node | 10.2 | 269 | 19 | **8** | 511 | 95.95% | 75.37% | 18.90% | 87.17% | 0.9668 / 0.9686 |
| `c13` pectoral 1 node | 8.3 | — | — | **4** | 394 | 97.53% | 83.68% | 19.04% | 92.63% | 0.9667 / 0.9682 |
| `c18` 2 nodes | 5.0 | — | — | **2** | 475 | 97.53% | 83.68% | 15.18% | 92.71% | 0.9672 / 0.9692 |
| `c19` B-gated | 4.0 | 274 | 21 | **2** | 532 | 97.53% | 83.68% | 14.67% | 92.62% | 0.9674 / 0.9694 |
| `c19` B-label | — | 247 | 0 | **50** | 3 | 97.53% | 83.68% | 97.53% | 93.68% | 0.9674 / 0.9694 |
| **`c20` C-gated** | **72.2** derived | 288 | 2 | **8** | 185 | **98.55%** | **87.50%** | 59.54% | **96.97%** | 0.9650 / 0.9678 |
| **`c20` C-label** | — | 246 | 0 | **52** | 3 | 98.55% | 87.50% | **98.55%** | **97.78%** | 0.9650 / 0.9678 |
| *`c20` C-gated, gate held* | *4.0 post-hoc* | 260 | 37 | **1** | 1,651 | 98.55% | 87.50% | 9.06% | 93.13% | 0.9650 / 0.9678 |

**C-gated, as pre-registered.**

1. **Swaps 8. Fails** — the clause reads "< 8" and 8 is `c03`'s figure, the worst
   incumbent. Breaks fell 21 → 2 over the same intervals, which is the signature
   of a gate that has stopped refusing anything: at 72.2 mm in a 192 mm tank,
   `gated` is `nearest`.
2. 185 tracks at 72.2 mm against `c19`'s 532 at 4.0 mm. The gates differ by a
   factor of 18; this compares gates.
3. **Tail-derived 0.9650 / 0.9678. Fails** on the reading `c19` used — 0.9650 is
   below `c03`'s 0.9668 on side1. The pairing is computed in slot order because
   detector rows are no longer a permutation of {0,1,2}; that implementation was
   checked against `c04`'s row-order one on `c19`'s output and agrees to
   1e-9 on all four pairs, so this is a real decrease and not a change of ruler.
4. **Coverage 98.55% ≥ 96.31%. Passes**, and is the highest in the project.
5. **96.97% on the run definition clears the 2D standard (94.6–97.0%) for the
   first time from a `gated` arm**, and 59.54% on the track definition against
   `c19`'s 14.67%. At a 72.2 mm gate, so this is as much clause 2's confound as
   a result.

**C-label** takes 52 swaps against `c19`'s 50 — disqualified by clause 1, as
stated in advance. Its trajectory coverage rises to 98.55% / 97.78%.

**C-gated with the gate held at 4.0 mm** takes **1 swap**, the lowest figure any
arm in this project has produced, against 37 breaks and 1,651 tracks. So the
extra state changes do not hand identities to the wrong animal — under a gate
that can refuse, they make the tracker abstain instead. One swap against `c19`'s
two is not a difference at this sample size; it is, however, not worse, and the
8 in the pre-registered row is attributable to the gate rather than to the
correspondence.

## Verdict: nothing is promoted, and the reasons split cleanly

**The reconstruction half succeeded.** Coverage, three-camera share and the
per-view usable rate all improve, the reduction to `c19` is exact where the
evidence is the same, and the ceiling is now visible and nearly reached. Removing
the cardinality rule was correct.

**The sequence half did not.** Three times as many correspondence changes are
kept, and neither gate setting turns that into a better tracker: at a loose gate
identity is lost (8 swaps), at a tight one continuity is (37 breaks, 1,651
tracks). Clause 3 also moves the wrong way.

Two specific repairs follow, and both are separate decisions:

1. **Weight each frame by the evidence it carries.** Replace the mean over terms
   with the sum — equivalently, weight by the number of contributing (node, slot)
   terms — so a frame resting on four rays counts less than one resting on nine.
   On a frame where every view holds every label this is again a constant factor
   away from `c19`'s cost, so the reduction property survives. This is the direct
   answer to the mechanism measured above and it changes the objective, so it is
   a new command, not a flag.
2. **Derive the gate on pairs where the state held.** The gate models how far a
   fish moves in 6 ms and must not be estimated from pairs where a slot was
   renamed. This is a two-line change to `derive_gate` and it would have made no
   difference to any earlier run — `c19`'s gate moves 3.97 → 3.77 mm.

Neither is applied here. `c03`, `c13`, `c16`, `c18` and `c19` are untouched, and
`c09`'s fixed output was restored to the `c03` baseline (preserved 269, break 19,
swap 8) after scoring, so `c11`/`c14`/`c15` still read what they read before.

---

# Addendum — the one swap stage C did not fix, and the repair that did not pay

**Date:** 2026-08-19, same session. **Tool:** new `c21`; `strict` and
`max_ceiling_mm` in `medaka3d.track`. `c20` and everything before it unmodified.

The experimenter asked whether removing the cardinality rule had removed the swap
they had reviewed by eye in `SWAP_i82121_x80554_p12.mp4`. It had not: **that
interval, 82,121 → 83,265, is the one swap left** in the gate-held arm. The other
one, 213,866 → 214,710, became a break.

## What the frames say

Read against `c19`'s own `diag_82713-82720_2d_labels.png`, K_1_2 at 82,715–82,717:

* `top_cam` label 0 is the fish that turns and darts. At 82,716 it leaves
  **side1 (as that camera's label 2) and side2 (as that camera's label 0) on the
  same frame**. Under the chosen state those are the same animal, so one animal
  drops to a single camera while the other two keep three. This is exactly the
  case the main document named in advance as the reason not to force maximum
  explanation, and the solver handled it correctly: it kept the state and
  reported the animal as unresolvable.
* **The female is `top_cam` label 2, and she is visible in all three cameras on
  every frame of the window.** Stage C computes her position at 82,716 from three
  cameras at 0.22 mm residual. `c19` had nothing there at all.
* `n_views_used` is the **minimum** over the three animals, so the frame is still
  marked unresolved and the linker never sees her. Removing the per-view
  cardinality rule left a per-frame one standing.

So the experimenter's reading was right in every particular: the animal that
vanished is not the animal that lost its identity, and the 2D evidence for her
was intact. Confirming it from the other side, `c20`'s **C-label arm — identity
named by the `top_cam` 2D label — preserves this interval.**

## Why she lost it anyway

`gated` links 82,715 straight to 82,717, and the assignment problem there has
**four** live tracks, not three:

| track | age | ceiling | → slot 0 | → slot 1 | → slot 2 |
|---|---|---|---|---|---|
| 612 the animal that left | 2 | 8.0 mm | 12.48 | 39.00 | 11.00 |
| 627 **the female** | 2 | 8.0 mm | **2.52** | 41.65 | **2.07** |
| 632 the still fish | 2 | 8.0 mm | 40.83 | 0.04 | 40.63 |
| 636 a 56-frame-old fragment | 56 | **224.0 mm** | 16.15 | 27.57 | 15.07 |

The ceiling is `max_step_mm × frames absent`, unbounded, so a fragment 56 frames
old carries **224 mm in a 192 mm tank** and can claim any animal in the volume.
Whichever order the ceiling and the assignment are applied in, the cheapest total
puts the female on slot 0.

Three rules were therefore candidates and `c21` crosses them: `whole`/`partial`
(does a frame offer the animals it has?), `gated`/`strict` (is the ceiling
applied after the assignment or before?), and `unbounded`/`bounded` (may a stale
track reach further than `VELOCITY_FRAMES × max_step_mm` = 12 mm?). One gate,
4.0 mm / 73 frames, for all eight arms.

## The result

`c09` against `female_sex_gt.csv`. `gated-whole-unbounded` is `c20`'s own arm and
is bit-identical to it, which `c21` asserts.

| arm | preserved | break | **swap** | tracks | resolved | top3 runs ≥ 1 s | 82,121 → 83,265 |
|---|---|---|---|---|---|---|---|
| **gated-whole-unbounded** (= `c20`) | 260 | 37 | **1** | 1,651 | 98.55% | 93.13% | swap |
| gated-whole-bounded | 256 | 41 | **1** | 1,914 | 98.55% | 93.13% | swap |
| strict-whole-unbounded | 261 | 32 | **5** | 1,180 | 98.55% | 93.14% | swap |
| strict-whole-bounded | 259 | 36 | **3** | 1,785 | 98.55% | 93.14% | **break** |
| **gated-partial-unbounded** | 260 | 37 | **1** | 1,676 | **99.94%** | **94.31%** | swap |
| gated-partial-bounded | 256 | 41 | **1** | 1,947 | 99.94% | 94.31% | swap |
| strict-partial-unbounded | 261 | 32 | **5** | 1,183 | 99.94% | 94.31% | swap |
| strict-partial-bounded | 259 | 36 | **3** | 1,811 | 99.94% | 94.31% | **break** |

**The diagnosis was right about the frame and wrong about the fix.** Only
`strict` **and** `bounded` together remove the reviewed swap — neither alone does,
which is what the four-track table predicts — and they remove it by turning it
into a **break**, not by carrying her identity through. And the price is other
intervals: swaps go 1 → 3. Clause 1 reads "an increase is disqualifying
regardless of every other number", so `strict` is disqualified in both variants,
and the local repair costs more than the local failure.

`gated` holds at 1 swap in every variant. Reasoning from one frame produced a
mechanism that is real, verifiable in a unit test with the measured numbers, and
still a net loss across 298 intervals. That is the whole reason clause 1 is
scored on every interval rather than on the one that was watched.

## The one thing worth keeping

`partial` — a frame offering the animals it has — is free on identity and pays on
coverage: **resolved 98.55% → 99.94%**, `top3_run_coverage` 93.13% → 94.31%,
swaps unchanged at 1, breaks unchanged at 37, 25 more tracks. It is the missing
half of stage C: `c20` removed the per-view cardinality rule and left the
per-frame one, and this removes that. Nothing else in this addendum survives its
own measurement.

Still not promoted, because `c03`, `c13`, `c18` and `c19` were all scored on
frames where every animal has a position and clause 3 has not been re-derived for
a partial frame. What `partial` needs before promotion is stated rather than
assumed: a definition of `views_2` / `views_3` for a frame whose animals rest on
different numbers of cameras, and clause 3 recomputed under it.

`c09`'s fixed output was restored to the `c03` baseline after scoring.
