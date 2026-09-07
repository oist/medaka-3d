# Handoff — per-view and per-instance confidence, from the gate review

**Date:** 2026-08-24
**Reads:** [2026-08-19 uncapped cardinality](decisions/2026-08-19-uncapped-cardinality.md)
and its addendum. **Tools:** `c20`, `c21`, `c22`.
**Status:** a human review of 14 rendered scenes, plus the measurements that
confirm or correct each observation. **No design is implemented yet.** Nothing
here has been promoted and no threshold has been chosen from it.

## Where this came from

`c22` renders the links `gated` refused — 127 of them at a 4.0 mm gate, filed by
how large the refused step was. The experimenter watched all 14 clips at or above
8 mm and reported per scene. That review is the substance of this document; the
numbers below exist to check it, not to replace it.

**This is the first identity evidence in the project that is not female-only.**
`female_sex_gt.csv` labels one fish; these scenes were judged on all three, so
male-vs-male identity has been looked at by a human here for the first time. It
is 14 scenes in one experiment, judged by eye, and it is recorded as an
observation, not as a ground-truth set.

## What the experimenter reported

Mechanism, in their words: the mapping between 2D tracks and 3D geometry, SLEAP
losing a detection at a dash or a sharp turn, and nodes exchanging between
instances around a crossing — **the three coincide, and that is when identity
swaps**.

Per scene, condensed:

| scene | observation |
|---|---|
| `ge12mm_031.99mm_f257714` | 1328 → 1329 swap. `top_cam` separates all three and could hold identity; side1 has 1327 and 1321 merged (1321's mouth+pectoral with 1327's tail); side2 crowded and occluded |
| `ge12mm_024.14mm_f113382` | **the crop drifted and hid the moment** — re-rendered, see below |
| `ge12mm_016.08mm_f116602` | mouth and pectoral stable in every view; 2D alone would hold it |
| `ge12mm_016.03mm_f139575` | side2 drops the blue fish at the transition but `top_cam` and side1 still hold it. Why a new ID? `top_cam` keeps 3 nodes on 3 instances — a per-view weight (matching node count, edge lengths) might have saved it |
| `ge12mm_014.61mm_f67637` | blue lost in `top_cam`; side1's 519 picks up blue's head; blue absent one frame only, three instances recover the next, and an orange ID is assigned anyway. side2: blue merges with 540's tail |
| `ge12mm_012.15mm_f188435` | **cross-view matching is wrong** at the transition — 1044 is the true blue, 1038 is another fish, and side1's 1046 is 1038's reflection in the wall. An instance-matching problem, not a tracking one |
| `ge8mm_011.11mm_f333537` | side2 loses blue, recovers mouth+pectoral four frames later, but blue's tail is merged with the attacked fish's mouth+pectoral |
| `ge8mm_009.63mm_f133911` | `top_cam` fuses 883's tail with 884's mouth+pectoral; side1 fuses 844's mouth with 833's pectoral+tail; side2 holds. A genuinely hard frame for SLEAP |
| `ge8mm_009.62mm_f65942` | side2 fuses two fish's nodes for several frames; `top_cam` loses blue at the dash frame. **The two coincide and that is what breaks continuity** |
| `ge8mm_009.40mm_f245893` | side2 mis-matches a wall reflection as 1280; the ID swaps when the reflection disappears. 2D alone would not have made that match |
| `ge8mm_009.27mm_f66130` | side1 swaps two fishes' tails; blue lost at the dash (motion blur, mouth occluded by the turn). `top_cam` and side2 could hold it in 2D |
| `ge8mm_008.30mm_f79318` | side1 fuses blue's tail with 623's mouth+pectoral; other two views fine |
| `ge8mm_008.28mm_f66204` | every view is followable in 2D. **No visible reason for the swap** |
| `ge8mm_008.26mm_f98924` | sides are correct on mouth+pectoral; `top_cam` is the hard view here |

Their overall reading:

* a per-instance confidence is wanted — number of detected nodes, a 2D body
  length that is stable across neighbouring frames rather than one frame's
  value, and the fact that a fish pointing along a camera's ray separates its
  mouth, pectoral and tail poorly;
* **mouth and pectoral are more trustworthy than the tail** for identity, because
  the tail is what merges between individuals;
* the dash is not refused because the fish is faster than the gate so much as
  because at that instant one view loses the instance and tails merge;
* **the 12 mm band is behaving as a swap detector — every one of them is a real
  identity error**, so the gate itself should not be loosened;
* and the suggestion: spend more care, and more computation, on scenes that look
  suspicious than on scenes that are easy.

## What the measurements say

All on K_1_2, `gated-partial-unbounded`, 127 refusals.

**1. The detection loss is real and it is not a coincidence.** A view that held
all three labels three frames earlier is short of one *at* the refusal on
**29.1%** of events for the pectoral and **22.8%** for the mouth, against
**2.7%** and **4.0%** at 20,000 random frames — 11x and 5.7x. Confirmed.

**2. The node ordering the experimenter proposed is measured, and it is
pectoral > mouth > tail**, not mouth+pectoral as one group. Share of frames
missing at least one label of that node, whole recording:

| view | pectoral | mouth | tail |
|---|---|---|---|
| `top_cam` | **3.9%** | 6.7% | 10.3% |
| `side1_cam` | 6.3% | 10.3% | 11.7% |
| `side2_cam` | 5.3% | 9.1% | 11.6% |

At the refusal frames every figure roughly triples, and the ordering holds.

**3. The view ordering is measured too, and `top_cam` is the reliable one** — as
the experimenter read off the clips. At refusals it is short of a pectoral label
on 9.4% of events against side1's 18.9% and side2's 19.7%.

**4. But the refused step is usually a genuine dash.** The median refused step is
4.49 mm when the incoming animal has three cameras and 4.66 mm when it has two,
and the camera count fell across the refusal in only 24 of 127 events. What
losing a camera does is fatten the tail: p90 6.13 mm on three cameras against
**9.63 mm** on two. So detection loss makes the step *noisier*, and the dash
makes it *large*; both are needed and neither alone explains the refusal.

**5. `f139575`, answered.** Slot 1 falls from three cameras to two when side2
drops the fish, and the fish is accelerating into a dash — its per-frame step
runs 2.9 → **4.0** → 5.5 → 14.0 mm. The break happens at the 4.0 mm step, exactly
at the ceiling. Two cameras still held the animal and its position still existed;
what ended the track was the size of the step, with the lost camera making that
step less trustworthy rather than impossible.

**6. `f113382` is not a fast fish — it is a wrong pairing correcting itself, and
the pipeline already knew.** The triangulation residual for that animal reads
**33.99, 34.19, 34.35, 20.93 mm** on the four frames before the transition and
**0.49 mm** on the frame after. A correct pairing on this rig is sub-millimetre.
The correspondence state does not change across it (11 → 11), so what moved is
which detections the 2D labels point at — a `c16` labelling error that the state
space **cannot express away, because the state is defined on those labels**. The
24.14 mm "step" is the 3D point jumping from a wrong triangulation to a right
one. The experimenter's reading of `f188435` — "an instance-matching problem,
not a tracking one" — is the same failure, and it is the more important of the
two families in this review.

**7. The residual is an unused wrong-pairing detector.** Across 1,037,114
animal-frames: median **0.46 mm**, p99 2.15 mm — and **0.69% sit above 5 mm**,
which no correct correspondence can produce. Those frames come in 1,285 runs, median
2 frames, longest 127 frames (0.8 s). **Gate refusals fall within two frames of
such a run 19.7% of the time against a 1.61% base rate — 12x.** Nothing in the
pipeline reads this. It is computed, stored in every npz as `residual_mm`, and
never checked.

## The design this points to

**A. Weights, not masks.** `medaka3d.calib.triangulate` already takes
`valid=` — a per-ray boolean added for stage C. Generalising it to a per-ray
**weight** is a small, contained change: the normal equations become
`sum(w_i P_i)`, the residual a weighted RMS, and `valid` is the special case
w ∈ {0, 1}. Then `correspond.label_costs` weights each (node, slot) term rather
than averaging them equally.

Weight sources, all computable from what is already cached:

* per (view, instance): how many of the three nodes were placed; the instance's
  2D span against **its own local median** rather than one frame's value, which
  is the experimenter's point and is what makes it robust at a dash;
* per node: pectoral > mouth > tail, in the ratio measured above rather than
  assumed;
* per view: `top_cam` above the sides, likewise measured.

Every weight must be justified by a measured loss rate and none by intuition,
or this becomes four tuned constants and the project has a rule against that.

**B. An absolute residual check, which costs nothing.** Minimising over 36 states
returns a winner even when every state is wrong. A frame whose winning state
still leaves an animal above R mm is not a correspondence — it is a refusal, and
should reach the tracker as "no evidence here" rather than as a confident wrong
point. R is derivable: the residual distribution is bimodal, sub-millimetre for
correct pairings and above 10 mm for wrong ones. This is the single cheapest
thing in this document and it should be measured first.

**C. Effort proportional to doubt, which is the experimenter's suggestion.** The
two quantities that separate easy scenes from suspicious ones are already
computed per frame: the runner-up **margin** and the **residual**. A frame with a
large margin and a small residual is settled and needs nothing. A frame with a
small margin, a large residual, a lost label or a merged node is where more nodes,
a wider search, or appearance should be spent. This makes the expensive machinery
affordable by not running it on the 98% of frames that do not need it.

## What not to do

**Do not loosen the gate.** The experimenter's review says every scene in the
12 mm band is a real identity error, so the gate is separating correctly at the
top of its range. The earlier proposal to sweep the gate upward is withdrawn for
that reason: it would trade a measured refusal for an unmeasured swap.

## Left open from the previous turn

* The clause revision — purity as a rate rather than a count, coverage per
  animal-frame, workload counted in **track identities** rather than contiguous
  runs. Not yet written into a document.
* The female ground truth is at its resolution floor (1 swap in 261 commitments);
  it cannot rank anything finer. This review is the beginning of a way around
  that, since it judges males too.

## Repaired here

`c22`'s crop followed one identity and then jumped to the other, so a
transition where the two were far apart could leave the tile — which is exactly
what hid `ge12mm_024.14mm_f113382`. The crop is now pinned to the midpoint of
the two identities within 40 frames of the refusal and widened to span them.
**All six 12 mm scenes have been re-rendered; `f113382` should be looked at
again.**

---

# B measured — the temporal criterion works, the absolute one does not

**Date:** 2026-08-24. **Tool:** new `c23`, rule fixed in its docstring before the
arms ran. `c20`, `c21`, `c22` unmodified. `c09`'s fixed output restored to the
`c03` baseline (269 / 19 / 8) afterwards.

## The reflection, confirmed as a false positive holding a label

The experimenter's reading of `f113382` — "no instance loss or fusion; a
reflection in the wall is being detected, and that is what makes the
triangulation match wrong" — is exactly right, and the detections say so:

`side1_cam` reports **four** instances on 113,378-113,383. `c16` runs with
``max_tracks=3`` and **has no reject option**, so three of the four get a label.
For five frames label 0 sits on the detection at (1115, 309) — static, near the
wall — while the real third fish at (780, 390) carries **no label at all**. At
113,383 the tracker moves label 0 onto the fish and the residual falls from
34.35 mm to 0.49 mm in one frame. The correspondence state never changes
(11 → 11): the state is defined on labels, so no amount of sequence modelling
can reach this.

**A false positive does not merely add an instance; with a fixed track count it
takes a real animal's label.** That is a `c16` design consequence and it is new.

## The two criteria are complementary, and the measurements say so

Absolute catches the reflection on all six frames. Relative catches only
113,382, because a residual that is *stably* 34 mm has a local median of 34 mm
and a ratio of 1.06. Conversely the relative flag fires on 991 animal-frames
that never reach 4 mm at all.

Enrichment against the 127 gate refusals, before either was applied:

| flag | fires on | refusals within ±2 frames | base rate | enrichment |
|---|---|---|---|---|
| absolute > 4 mm | 0.76% of animal-frames | 19.7% | 1.66% | 12x |
| **relative > 5.19x** | **0.094%** | **33.9%** | **0.32%** | **106x** |
| either | 0.83% | 41.7% | 1.91% | 22x |

## What each did to the tracking

`gated` + `partial`, gate held at 4.0 mm / 73 frames, `c09` against the female
annotation. The `none` arm is bit-identical to `c21`'s, which `c23` asserts.

| arm | withheld | preserved | break | **swap** | error rate | identities ≥ 1 s | coverage | median lifetime |
|---|---|---|---|---|---|---|---|---|
| `none` (= `c21`) | 0 | 260 | 37 | **1** | 0.38% | 241 | 97.99% | 14.3 s |
| `absolute` | 7,894 | 263 | 31 | **3** | 1.13% | 212 | 98.00% | 15.4 s |
| **`relative`** | **971** | **267** | **30** | **1** | **0.37%** | **196** | **98.03%** | **17.6 s** |
| `both` | 8,648 | 267 | 27 | **3** | 1.11% | 186 | 98.03% | 16.9 s |

**The temporal criterion — the experimenter's addition — is the one that works.**
At 971 animal-frames withheld, 0.094% of those that have a position:

* swaps unchanged at 1, error rate 0.37% against 0.38%;
* **breaks fall 37 → 30 and preserved rises 260 → 267**, so it buys continuity
  rather than spending it;
* identities holding at least a second fall **241 → 196**, a 19% smaller
  appearance workload, and their median lifetime rises 14.3 s → **17.6 s**;
* coverage rises 97.99% → 98.03%.

Better on every measured axis, for a tenth of a percent of the data.

**The absolute criterion, which this document proposed first, fails its own
pre-registered rule: swaps 1 → 3.** Withholding 0.76% of animal-frames in
sustained runs opens gaps the linker then bridges wrongly. `both` inherits the
failure. By clause 1 as written, an increase is disqualifying, so `absolute` and
`both` are out and only `relative` survives.

That is a real correction: the absolute threshold looked better on the
enrichment table and is worse in use, because *what a flag catches* and *what
removing it does* are different questions.

## Where the absolute population should go instead — measured, not assumed

Withholding is the wrong action for the sustained wrong pairings, because they
are **recoverable**. On the 7,513 three-camera animal-frames above 4 mm,
dropping one camera's ray brings the residual from a median of **29.18 mm** to
below 1 mm on **92.3%** of them, and below 0.5 mm on 75.4%.

Two controls, because a two-ray fit is small by construction and c13 says so:

* **Sharp.** On those frames the best drop is 0.270 mm, the second best 9.092 mm,
  the worst 31.896 mm; the second best exceeds 5x the best on **77.2%**. One
  camera is identified, not any camera.
* **Unbiased.** The same test on 40,000 already-correct animal-frames gives a
  near-uniform culprit (top 29.5% / side1 36.1% / side2 34.4%) and a sharpness of
  only 12.9%. The geometry is not producing the answer.

And the culprit distribution on bad frames is **top_cam 48.3% / side1 31.0% /
side2 20.7%** against that 29.5 / 36.1 / 34.4 null.

**This is the experimenter's first point, quantified: `top_cam` is globally the
most reliable view — it loses a pectoral label on 3.9% of frames against the
sides' 6.3% and 5.3% — and it is also the view most often carrying the wrong ray
when a triangulation is wrong.** Global reliability and per-scene reliability are
different quantities and here they point opposite ways, so a single per-view
weight cannot express what is needed.

## What to build next

**Drop the ray, not the animal-frame.** `triangulate` already takes a per-ray
`valid=`; the repair is to set it False for the identified camera rather than
discarding the whole animal-frame. That recovers 92% of the absolute population
instead of losing it, and it is the same mechanism as a per-scene view weight
with w ∈ {0, 1}.

**The one caution, stated before it is built:** choosing which ray to drop by
which choice minimises the residual is choosing on the statistic being measured,
and a two-ray fit always fits. It needs an independent check, and one is
available at no cost — does the recovered position agree with the animal's
position on the neighbouring frames? A repair that fixes the residual and moves
the animal 20 mm is not a repair. That test should be written before the repair.
