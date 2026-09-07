# The doubt channel, momentary view demotion, and ray rejection at doubted frames

**Date:** 2026-09-01. **Written before the runs.** Results are appended per
step and nothing else is edited. **Status: measurement only; nothing is
promoted; `c03` still owns the fixed output paths.** New tools `c31`, `c32`,
`c33`; `c30` gains input-directory arguments (same algorithm, different input —
rule 5 is not touched). No existing output is modified.

**Follows:** [2026-09-01 slot stitching](2026-09-01-slot-stitching.md).
**Executes:** step D of `NEXT_SESSION_uncapped_followup.md` (the doubt score,
promoted there) and a *bounded* revival of its step E (ray rejection, deferred
there). **Does not contradict** the 2026-08-24 rejection of the per-view weight
table: that was a **static** weight measured on calm frames; what is built here
is a **momentary** demotion triggered per frame, which that record never
tested.

## What the experimenter's review established (2026-09-01, frames ~5,400-40,000)

Four diagnosed scenes, all verified against stored arrays:

| scene | diagnosis (by eye) | signature (measured) |
|---|---|---|
| 30299/30307, S2 | FP near S1's tail, then wall reflection while occluded | side1 occluded (2 det.); a 3rd side1 det. joins for 1 frame; residual 0.67 -> 5.30 / 4.70 mm; step out-and-back |
| 30735, S0 | side2 instance misassigned | 2-ray frame - residual blind (1.45 mm); side2 held 1 det. for the whole window; S0 dropped to 1 ray at 30733 |
| 30841, S0 | occlusion + wall reflection | rays 3, residual **19.03 mm**, margin 0.0, top_cam 4 det. |
| 31479, S0<->S2 | top_cam degraded by surface waves during a male-male fight | top best score 1.13-1.29 vs calm median 1.94 (p10 1.79); count churn 2-4; **both raw and stitched leave the scene physically exchanged**; `c30` fired two decisive-looking joins (margin ~336 mm) that net to identity |

The 31479 case is the important negative: **continuity cannot repair a stretch
where the geometry itself is corrupted.** The repair has to happen where the
corruption enters - the reconstruction's use of the degraded view.

Concentration of candidate signals, K_1_2 stitched lambda 0.767, against
126 out-and-back spikes (set A) and 210 residual decisive-exchange boundaries
(set B), halo +-2 frames:

| signal | fires on | captures A | captures B |
|---|---|---|---|
| residual > 3 mm on a 3-ray slot | 0.35% | 93.7% | 64.3% |
| slot ray-count changed | 3.2% | 88.1% | 94.8% |
| top_cam best score < 1.5 | 0.61% | 42.1% | 61.0% |
| margin < 1.5 mm | 7.6% | 68.3% | 87.1% |
| OR of the four | **12.0%** | **100%** | **100%** |

## Step 1 - `c31`, the doubt channel (measurement only)

Per-frame signals computed from quantities already in every reconstruction
output plus the `c01` cache; **no pipeline behaviour changes.** Thresholds are
fixed now, calibrated on K_1_2 and hypotheses anywhere else (`n_groups = 1`):

* `residual_spike`: any slot with 3 rays and residual > **3.0 mm**
* `margin_low`: margin < **1.5 mm**
* `ray_change`: any slot's ray count differs from the previous frame
* `view_degraded` per view: best instance score < **1.5** (calm median 1.94,
  p10 1.79; the 31479 wave scene runs 1.13-1.29). A view with no detections is
  absent, not degraded.
* `spike`: out-and-back - both adjacent steps > 10 mm, return within 3 mm
* `exchange_boundary`: non-identity slot assignment cheaper by > 5 mm
* `doubt` = residual_spike | margin_low | ray_change | any view_degraded

Output: a doubt npz per input arm plus ranked **doubt zones** (flagged frames
merged within 1 s) - the review-scene queue the experimenter asked for. The
handling period at the start should appear as the largest zone; that is a
check, not a finding.

## Step 2 - `c32`, momentary view demotion (changes the correspondence)

Re-run `c25`'s solve on `c27`'s serial labels exactly as `c28` did, except:
on frames where **exactly one** view is degraded (best score < 1.5), that
view's rays are marked absent for that frame before the solve. The machinery
is `c25`'s existing absent-node handling - **no cost function is edited and no
new constant is introduced**; a slot supported by the two healthy views
triangulates from them and pays no mu. Frames with two or more degraded views
are left untouched and counted (`undemotable`): demotion requires two healthy
views. Labels in the demoted view are still assigned (by lambda continuity);
they just carry no geometric weight that frame.

Then `c30` stitches the `c32` arms and `c09` scores, exactly as before.

**Reading rules, fixed now.**

* All seven lambdas. A claim is carried only if it holds across the grid.
* **Disqualifier:** `c09` on the gated arm may not exceed its `c28`
  counterpart's swap count at any lambda (baseline 0/0/0/0/0/1/3 across the
  grid).
* Success indicators, stated in advance: the 31479 scene comes out
  **unexchanged**; exchange boundaries (set B) shrink; stitched-label swaps
  (6-35 across the grid) do not increase. `resolved` and
  `three_cameras_every_animal` will drop slightly on demoted frames; recorded,
  not scored.
* Cost: ~18 min per arm, ~2 h for the grid.

## Step 3 - `c33`, ray rejection at doubted frames (changes geometry, not labels)

The 2026-08-24 plan's "drop the offending ray", revived at the scale the doubt
channel leaves: only (frame, slot) with **3 rays and residual > 3.0 mm**. For
each, triangulate leave-one-out for each view (the same view dropped across
the cost nodes); **adopt the best drop only if** its residual <= **1.5 mm**
and the runner-up drop is at least **1.0 mm worse** (the culprit must be
sharp - prior measurement: dropping one ray fixes 93.1% of the >4 mm
three-camera frames and identifies the culprit sharply on 72.1%). Adoption
replaces `xyz`/`residual`/`slot_rays` for that (frame, slot); the label
assignment is untouched. Everything else is left as it is and counted.

Applied to the `c32` arms (the production candidate chain
`c27 -> c32 -> c33 -> c30`), and separately to the `c28` lambda 0.767 arm so
the effect is attributable. Scored: spike set A before/after; `c09` on the
re-stitched, re-gated result (geometry changes can move gated breaks - that is
why `c09` runs again).

## Result - step 1

**Appended 2026-09-01.** `c31` on the stitched lambda 0.767 arm; fire rates as
pre-measured (doubt 11.97%). `tests/test_doubt_channel.py` pins the spike and
exchange detectors. **512 doubt zones.** The pre-registered check passed: the
largest zone is frames **0-5,121** - the handling period, found by the
instrument rather than told to it (the experimenter's own estimate from the
video was "~5,400"). The second-worst zone, frames **30,069-32,348**, contains
**all four scenes the experimenter diagnosed by eye** - the review queue and
the human agree on where the trouble is.

## Result - step 2

**Appended 2026-09-01. The all-views demotion is DISQUALIFIED by its own
pre-registered gate.** Both halves of that sentence matter.

Demoted: top 1,699 / side1 3,834 / side2 6,023 frames; 1,428 refused (two or
more views degraded at once). ~16 min per arm.

**What it repaired.** The 31479 wave scene comes out **unexchanged** on the
full solve - slots hold the same physical fish straight through, the 174 mm
step becomes 3.0 mm, and `demoted__top_cam` fires exactly at the diagnosed
frame. Raw-arm exchange boundaries 459 -> 381, spikes 109 -> 93, and the
residual tail improves at every lambda (0.767: p99.9 3.56 -> 3.01 mm,
> 4 mm 869 -> 628). `resolved` pays 98.771% -> 98.145%, as pre-registered.

**What disqualifies it.** `c09` on the stitched chain:

| lambda | gated swaps c28-chain -> c32-chain | stitched-label swaps |
|---|---|---|
| 0 | 0 -> 0 | 35 -> 45 |
| 0.1 | 0 -> 0 | 8 -> 14 |
| 0.25 | 0 -> **1** | 11 -> 17 |
| 0.5 | 0 -> **1** | 10 -> 15 |
| 0.767 | 0 -> **1** | 6 -> **11** |
| 2 | 1 -> **2** | 7 -> 12 |
| 8 | 3 -> 3 | 8 -> 13 |

The gated swap count exceeds its `c28` counterpart at four lambdas and the
stitched-label swaps rise at all seven. The gate says stop, so it stops:
**the all-views variant is not carried.**

**The likely mechanism, and the variant it licenses.** The experimenter's
proposal named `top_cam` alone (surface waves). The generalisation to all
views is what was tested - and the side views trip the 1.5 floor about three
times as often (1.4% / 1.7% vs 0.61%), while `c09`'s anchors live entirely in
the side views. A `--demote-views top_cam` variant - the proposal as actually
made - is pre-registered here with the same reading rules and the same
disqualifier, and run next. Nothing about the all-views result is carried
into it.

### The top_cam-only variant, scored

Demoted 1,699 frames (top alone); `resolved` pays only 98.771% -> 98.668%.
The full chain `c32-top -> c33 -> c30`, against chain B (`c28 -> c33 -> c30`):

| | chain B | c32-top chain |
|---|---|---|
| gated swaps across the grid | 0/0/0/0/0/1/3 | 0/0/0/0/0/1/3 - **gate passed** |
| stitched-label swaps across the grid | **35/6/9/8/5/6/6** | 40/11/12/11/8/9/9 |
| steps > 10 mm, lambda 0.767 | 457 | **312** |
| steps > 30 mm, lambda 0.767 | 334 | **195** |
| 31479 wave exchange | still exchanged | **repaired** (holds [0,1,2], step 3.0 mm) |

**Read against the rules:** the disqualifier does not fire, but the
pre-registered success indicator - stitched-label swaps do not increase -
fails at every lambda. So the variant is **not carried into production**,
with two facts recorded beside that verdict rather than under it: the 3-5
interval increase sits inside the range the standing rule says ranks nothing
("swap counts of 0-8 rank nothing"), and the class the demotion repairs -
a lasting male-male exchange, eyeball-confirmed at 31479 - is exactly the
class the female-only annotation cannot see. The instrument that would settle
this does not exist yet; reviewed scenes over the wave zones (`c22`) are how
it comes to exist.

## Addendum, 2026-09-03: the wobble signal, and demotion triggered by it

The experimenter's second review pass (now in
`assets/groundtruth/swap_event_gt/`, with per-view clean/dirty judgements)
proposed a new per-view signal: **the temporal wobble of the 2D mouth-tail
length** against its own track's rolling median. Measured before adoption, on
the production arm, against the same bad sets:

| signal (any view) | fires | captures exchange boundaries |
|---|---|---|
| wobble dev > 0.3 | 5.0% | **83.4%** |
| score < 1.5 (existing) | 3.4% | 74.1% |

Per event it reproduced the annotator's own view judgements where the score
was silent: 302762 (top node-switching: wobble 0.53 top / 0.03-0.04 sides),
260434 (side1 fusion: **2.03**), 31479 (top ripples: 0.95), 333619 (top:
0.45). Its blind spot is a *stable* detection of the wrong animal (30735) -
shared with the score, left to the occlusion counts, and stated. Adopted into
`c31` as `view_wobble__<view>` at 0.3 (doubt_version 2; doubt now fires on
15.7%, zones 512 -> 339).

**Pre-registered, run next: `c32 --trigger wobble`** - the same demotion
machinery with the degradation floor `wobble > 0.5` (beyond every view's calm
p99 of 0.34-0.48; fires on 1.4% of frames over all views together), all views
eligible, exactly-one rule unchanged. Same reading rules and the same
disqualifier as every demotion variant; the chain scored is
`c32-wobble -> c33 -> c30`, against chain B. Nothing from the score-triggered
runs is carried into it.

## Addendum, 2026-09-03: `c34` - the out-and-back flip repair

The experimenter asked whether what a single-frame 3D jump can identify is
repairable in a final pass. It is, for exactly one class: a slot flip that
**decisively reverts within <= 3 resolved frames** - the class `c30`'s
8-frame medians are blind to by design, holding three of the annotated events
(21232, 67142, 333619). Rule, prototyped in memory before the tool was
written: entry assignment beats identity by > 5 mm, the exact inverse beats
identity by > 5 mm at the exit, no other flip between; the permutation is
applied to the window only, nothing downstream. A lasting exchange has no
decisive inverse and cannot fire it; a spike keeps identity as its best
assignment and cannot either (`tests/test_microflip.py`).

Applied to the production chain, per lambda: 28-661 windows (94% one-frame).
At lambda 0.767: 65 windows, all three annotated transients repaired
(32.6 -> 3.1, 61.2 -> 3.7, 45.7 -> 1.5 mm), steps > 30 mm 334 -> 193; at
lambda 0, 661 windows and 2,094 -> 612. `c09` across the grid: gated
**identical at every lambda** (0/0/0/0/0/1/3), stitched-label swaps **equal or
lower at every lambda** (35/6/9/8/5/6/6 -> 33/4/7/6/5/6/6). A strict
improvement; **carried**. The production chain is now
`c27 -> c28 -> c33 -> c30 -> c34`.

## Addendum, 2026-09-03: node-exchange evidence (annotator's second proposal)

Measured: an instance's node landing nearer another instance's same node at
the previous frame than its own (self-step > 15 px). Exchange rates order
**tail >> pectoral > mouth** in every view (0.12-0.22% / 0.08-0.11% /
0.015-0.03%) - the dynamic version of `s10`'s static finding. As a signal it
is the sharpest yet: fires on **1.30%** of frames, captures **88.1%** of
exchange boundaries and 71.2% of spikes. Per event it names view AND node:
302762 top:PT, 31479 top:MPT, 323819 side1:T, 333619 top:MPT - matching the
annotator's per-view judgements; its misses (21232, 30735, 76930) are the
occlusion classes where the counterpart node is undetected, the count
signals' domain. Not yet a `c31` signal or a demotion trigger; **node-granular
demotion** (mask only the exchanged node's ray, keep the stable ones - the
annotator's "carry identity on the stable part") is the natural next
mechanism, pre-registration pending the wobble-demotion verdict.

### The wobble-triggered demotion, scored (chain `c32-wobble -> c33 -> c30 -> c34`)

Demoted top 788 / side1 2,168 / side2 1,926 frames; `resolved` pays only
98.771% -> 98.590%. Against the production chain (with `c34`), label swaps
across the grid: **40/6/3/3/3/6/5** vs 33/4/7/6/5/6/6 - better at four
lambdas (0.25/0.5/0.767/8, with **295 / 0 / 3** at 0.767, the best figure
yet), equal at one, **worse at 0 and 0.1**. Gated swaps 0/0/0/0/0/1/**1** -
no lambda exceeds baseline and lambda 8 improves 3 -> 1: **the disqualifier
does not fire.** Steps at 0.767 after `c34`: 265 / 142 vs 322 / 193.

**Verdict: not carried, with the reasons beside it.** The pre-registered
success indicator - label swaps do not increase at any lambda - fails at
lambda 0 and 0.1, so the claim is lambda-dependent; and within the usable
range the differences (3 vs 5) sit inside the band the standing rule says
ranks nothing. One finding matters more than the scoreline: at 31479 the
wobble demotion **converted the loud failure into a quiet one** - the 174 mm
teleport is gone, but the animals still exchange slots through 3-4 mm steps
while fighting at close range, invisible to every jump detector and to the
eye. A demotion that silences the symptom without fixing the identity is
worse for review than the teleport it removed. (The score-triggered variant
had repaired this scene outright; the wobble floor of 0.5 also missed 333619
at 0.45.) Demotion by whole-view drop is hereby parked: the mechanism the
evidence now points at is **node-granular demotion** - mask only the
exchanged node's ray, keep the stable ones - driven by the node-exchange
signal (fires 1.3%, captures 88.1%, names view and node), possibly with both
triggers. Pre-registration required before it runs.

## The chain that stands, and what is still open

*(Rewritten 2026-09-08; the 2026-09-03 text here predated the stitch's v4
and named the wave-exchange class as open. It is not.)*

**Production chain: `c27` serial -> `c28` -> `c33` ray rejection ->
`c30` v4 stitch -> `c34` micro-flip repair.** At lambda 0.767 (drawn, not
chosen): stitched-label **296 / 0 / 2** on 298 female intervals, gated
**272 / 26 / 0**, adjacent steps > 10 mm **1,170 -> 325**, > 30 mm
**674 -> 197**. Across the grid, label swaps 30/5/4/3/2/2/4 and gated swaps
0/0/0/0/0/1/3. The wave-exchange class that this record's step 2 tried to
reach by demotion was reached instead by the stitch's zone audit (v4,
[slot stitching](2026-09-01-slot-stitching.md)): 31479 is repaired with
margins 319 / 326 mm. Nothing is promoted; `c03` still owns the fixed paths.

Open, in order of value - the full list with mechanisms is in
[`TRACKING_PIPELINE.md`](../TRACKING_PIPELINE.md): (1) node-granular demotion
driven by the node-exchange signal, pre-registration pending; (2) the 2-ray
misassignment class (30735), which no residual can see; (3) the two declared
refusals at lambda 0.767 (76930 at 3.26 / 3.85 mm; the 2,494-frame unvouched
zone at 133,417-135,911), which are review-queue items; (4) the quiet
close-range exchange the wobble demotion produced at 31479, which no jump
detector can see. `n_groups = 1` throughout: every number above is one trio
of fish on one day.

## Addendum, written 2026-09-08: measurements of 2026-09-03 not recorded earlier

### The score trigger's window - detection at one frame, dilate, never smooth

Asked what window the score trigger should use. Sub-1.5 episodes are short:
top 364 episodes post-handling, **154 of them one frame long, median 2,
p90 11**; side1 766 / median 2 / p90 13; side2 863 / median 2 / p90 14.
Rolling-median smoothing therefore erases the events it is meant to find:

| window | 6 ms → | fires | captures exchange boundaries | covers 31476-81 |
|---|---|---|---|---|
| 1 | 6 ms | 3.36 % | **59.1 %** | yes |
| 3 | 18 ms | 3.29 % | 40.9 % | yes |
| 5 | 30 ms | 3.23 % | 45.5 % | yes |
| 9 | 54 ms | 3.09 % | 31.8 % | **no** |
| 17 | 102 ms | 2.83 % | 27.3 % | no |
| 33 | 199 ms | 2.47 % | 22.7 % | no |

(22 exchange boundaries in the post-repair arm - small n.) Dilating each hit
instead: +-1 keeps 59.1 % at 4.27 %, **+-2 keeps 59.1 % at 4.90 % and covers
the wave burst fully** (its interior holds a deceptive 1.81-score frame at
31478), +-4 costs 5.95 %. The costs are asymmetric - a false one-frame
demotion loses one frame of one view's redundancy, a missed real one corrupts
the correspondence - so the answer is **1 frame, dilated +-2**, which is what
the stitch's zone definition uses.

### Second-pass dissections of two annotated scenes

**333619 - "2D is clean in all three views, yet 3D swaps over a large
distance".** True on both counts. The three (side1, side2) label pairs are
identical before and after; what changed is which `top_cam` label attaches to
each pair, because `top_cam` intermittently loses one animal around this
window (top label None on one slot before 333615 and on another after
333621). Each loss re-poses the assignment; both pairings score margins of
10-17 mm - two confident answers - and the two transition frames hold mixed
pairings whose triangulations are ghosts, hence the 45 mm there-and-back. The
side1 -> side2 map is invariant across the boundary, which identifies
`top_cam` as the camera that relabelled - the 2026-08-18 handoff's never-used
evidence, demonstrated. Repaired by `c34`.

**31479, second look.** The annotator's premise that both side views were
clean does not hold at the tracklet level: over 31470-31490 `top_cam` changed
its label set 14 times (worst per-label step 54 px), **`side1_cam` 7 times
with a 276 px label jump**, `side2_cam` twice (23 px). The exchange partner
was not the nearby fighter (30.5 mm away) but the **female, 173.9 mm away** -
the wave corrupted `top_cam` over the whole tank. And the clean endpoints
(31474 vs 31484) disagree with the interior's net decision by **334.7 mm in
position and 338-342 mm by velocity at every window in 0.1-0.3 s** - which
is why the v4 zone audit, comparing exactly those endpoints, repairs it.

### The vector criterion, measured globally

Of 1,099 same-slot steps > 5 mm (post-handling), **627 point against the
animal's own 0.3 s heading** (cos < 0) and only 252 align with it
(cos > 0.5, the real bursts); the magnitude ratio to local mean speed has
p99 = 7.3. Direction separates swap teleports from burst swimming; magnitude
alone cannot.

## Result - step 3

**Appended 2026-09-01 - the attribution half, `c33` on the `c28` arms.** (The
production-chain half, on the `c32` arms, is below the step-2 result.)
`tests/test_doubt_channel.py` pins the repair on the pinhole fixture rig: a
planted contaminated ray in a named view is dropped and the true point
restored to < 0.05 mm; two contaminated views are refused.

At lambda 0.767: **1,344 flagged, 928 adopted** (69%); refused 372 not sharp,
44 not clean, 0 without geometry. The dropped view is `top_cam` 425 / side1
316 / side2 187 - the same ordering the 2026-08-24 plan measured. Across the
grid the adopted count runs 872-1,323, rising with lambda as the smoother
correspondence carries more contaminated frames rather than relabelling them.

Effect at lambda 0.767, `resolved` unchanged at 98.771% (geometry only):

| quantity | before | after |
|---|---|---|
| out-and-back spikes | 109 | **65** |
| adjacent steps > 10 mm | 1,170 | 913 |
| adjacent steps > 30 mm | 674 | 536 |
| 3-ray slots with residual > 3 mm | 1,344 | **416** |

The experimenter's three diagnosed spike scenes, before -> after (step in mm):
30299 S2 **6.9 -> 0.3**, 30307 S2 **5.8 -> 0.2**, 30841 S0 **27.6 -> 5.8**
(residual 19.03 -> 1.48). The steps that remain are exchanges (identity,
`c30`'s domain) and 2-ray contamination, which residual cannot see - both
stated in advance as out of scope.

### The production half: `c28 -> c33 -> c30`, scored

With the all-views `c32` disqualified, the chain that stands is `c27` serial
stage A -> `c28` correspondence -> `c33` ray rejection -> `c30` stitch.
Against the previous best (`c28 -> c30`), at every lambda:

| lambda | label swaps prev -> now | gated P/B/S prev -> now | steps>10mm prev -> now |
|---|---|---|---|
| 0 | 35 -> 35 | 272/26/0 -> **274/24/0** | 3,501 -> 3,288 |
| 0.1 | 8 -> **6** | 270/28/0 -> 272/26/0 | 1,013 -> 734 |
| 0.25 | 11 -> **9** | 270/28/0 -> 272/26/0 | 857 -> 573 |
| 0.5 | 10 -> **8** | 268/30/0 -> 271/27/0 | 801 -> 520 |
| 0.767 | 6 -> **5** | 269/29/0 -> **272/26/0** | 750 -> **457** |
| 2 | 7 -> **6** | 261/36/1 -> 264/33/1 | 754 -> 414 |
| 8 | 8 -> **6** | 257/37/3 -> 256/38/3 | 842 -> 352 |

**Carried grid-wide:** the ray rejection never raises a swap count on either
scoring, converts a few gated breaks into preserved at most lambdas, and
shrinks the visible steps at all of them. At lambda 0.767 the stitched slot
labelling reaches **293 preserved / 0 break / 5 swaps (1.7%)** on the 298
female intervals - the best figure this project has produced. Steps > 30 mm
fall 509 -> 334.

The wave-exchange class (31479) is **not** addressed by this chain - that
repair lived in the demotion, whose all-views form failed the gate. The
top_cam-only variant is being scored separately; failing that, the honest
treatment is a stitch refusal zone over `view_degraded` stretches, left as
the next piece of work.
