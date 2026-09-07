# Which detector mixes parts least: the 2×2 at 20,000 frames

**Date:** 2026-08-09
**Tool:** [`tools/s10_shared_node_20k.py`](../../tools/s10_shared_node_20k.py),
pinned by [`tests/test_shared_node_20k.py`](../../tests/test_shared_node_20k.py)
**Results:** `work_dir/s10_shared_node_20k/s10_results_{top,side1,side2}.json`, with
provenance sidecars
**Supersedes for this question:** the 40-frame 2×2 in
[`s06_locked_benchmark_3node.py`](../../tools/s06_locked_benchmark_3node.py)

## Verdict

**Run production inference with top-down 2-node, round 8 — the incumbent. The
third node does not earn its place in either architecture. Do not read the
bottom-up result as a win: the primary metric cannot measure bottom-up at all.**

| | third node earns its place? | on what evidence |
|---|---|---|
| top-down | **No** | wins on top, loses badly on both side views, on every metric |
| bottom-up | **No** | primary metric is floored here; the metrics that see anything say worse on both sides |

## What was measured

20,000 frames per model per view — 4 experiments × 5 evenly spaced blocks of
1,000 contiguous frames, deterministic starts `[34360, 104080, 173800, 243520,
313240]` of 348,600. 4 models × 3 views × 20,000 = **240,000 inferences**, all
completed; no arm was sampled, truncated or estimated.

Thresholds were derived at run time from `labels.round8.<view>.train`, not
hardcoded, and reproduce the brief's table exactly: top median span 112.78 px →
tail-jump 60 px; side 94.05 px → 50 px; shared-node 5 px in both.

## Pooled results

Rates are per frame. `swap%` is defined below and is **not** part of the
pre-registered rule.

### top — 112.78 px span, jump > 60 px

| model | shared% | mouth/tail pairs | jump% | swap% | inst/frame | frames <3 |
|---|---|---|---|---|---|---|
| topdown 2-node (round8) | 1.095 | 2 / 219 | 0.522 | 0.0578 | 2.944 | 1422 |
| topdown 3-node (n409) | 0.820 | 2 / 167 | 0.377 | 0.0900 | 2.944 | 1422 |
| bottomup 2-node (round7) | **0.000** | 0 / 0 | 0.296 | 0.1159 | 2.990 | 891 |
| bottomup 3-node (n409) | **0.000** | 0 / 0 | 0.241 | 0.0818 | 2.979 | 857 |

### side1 — 94.05 px span, jump > 50 px

| model | shared% | mouth/tail pairs | jump% | swap% | inst/frame | frames <3 |
|---|---|---|---|---|---|---|
| topdown 2-node (round8) | 3.355 | 1 / 723 | 0.549 | 0.0679 | 3.007 | 1156 |
| topdown 3-node (n250) | 5.605 | 1 / 1255 | 0.670 | 0.1511 | 3.007 | 1156 |
| bottomup 2-node (round7) | 0.005 | 0 / 1 | 0.358 | 0.0596 | 2.867 | **2612** |
| bottomup 3-node (n250) | 0.010 | 1 / 1 | 0.542 | 0.0501 | 2.981 | 1419 |

### side2 — 94.05 px span, jump > 50 px

| model | shared% | mouth/tail pairs | jump% | swap% | inst/frame | frames <3 |
|---|---|---|---|---|---|---|
| topdown 2-node (round8) | 3.780 | 2 / 775 | 0.561 | 0.0885 | 3.012 | 1007 |
| topdown 3-node (n250) | 7.365 | 2 / 1583 | 0.972 | 0.2733 | 3.012 | 1007 |
| bottomup 2-node (round7) | 0.000 | 0 / 0 | 0.406 | 0.0719 | 2.870 | **2647** |
| bottomup 3-node (n250) | 0.005 | 0 / 1 | 0.565 | 0.1041 | 2.985 | 1242 |

Mouth sharing is negligible everywhere — at most 2 pairs per view against
hundreds of tail pairs. The brief predicted that on structural grounds and it
holds: the mouth is the top-down anchor, and duplicating it is a different
failure.

## The primary metric barely applies to bottom-up, and this changes the reading

Bottom-up scores 0.000% on top and ≤0.010% on the sides. That is **not** evidence
that it keeps animals separate. The nearest tail-to-tail distance *within a
frame*, pooled over all three views — 238,414 frames carrying at least two tails:

| model | min | p0.1 | p1 | p5 | median |
|---|---|---|---|---|---|
| topdown 2-node | **0.01** | 0.13 | 1.28 | 24.07 | 141.1 |
| topdown 3-node | **0.01** | 0.11 | 0.45 | 7.17 | 143.9 |
| bottomup 2-node | **4.33** | 14.19 | 22.12 | 40.64 | 142.4 |
| bottomup 3-node | **3.97** | 14.52 | 23.98 | 42.22 | 143.1 |

Top-down puts two tails 0.01 px apart; its 1st percentile is 1.28 px (2-node)
and 0.45 px (3-node), both *below* the 5 px threshold. Bottom-up's 1st
percentile is 22–24 px, and its 0.1th percentile is still 14 px. Its minimum
over 238,414 frames does dip to 4.33 px, so a shared node is not literally
impossible — it does produce a handful — but the distribution is floored so far above the
threshold
that the metric is measuring the floor rather than the model: **4 shared pairs
from the two bottom-up arms across all three views, against 4,732 from the two
top-down arms.**

The mechanism is peak NMS plus PAF grouping: bottom-up will not emit two peaks a
few pixels apart, and each peak is assigned to at most one instance. Top-down has
no such constraint — two centroids can independently regress a tail onto the same
animal. Above the threshold the two architectures' distributions are ordinary
and overlapping (medians 141–144 px in all four).

So the six `challenger wins` verdicts the pre-registered rule prints for
bottom-up are artifacts of a degenerate denominator, not measurements. **The
rule was applied as written and its output is recorded below unaltered — but it
is wrong about the architecture comparison, and the reason is in the arithmetic
rather than in the data.** The metric remains valid *within* top-down, where
both arms share the round-8 centroid head and emit byte-identical instance
counts (2.944 / 3.007 / 3.012 per frame in the two arms of every view), so a
shared-node difference there is a difference in node placement and nothing else.

This also retires hypothesis (a) from the 40-frame run — "bottom-up mixes parts
less than top-down, regardless of skeleton". Bottom-up was always going to score
0. At n=40 that read as a clean sweep; at n=20,000 it reads as a property of
PAF grouping.

### What the rule printed, unaltered

| comparison | top | side1 | side2 |
|---|---|---|---|
| topdown 3-node vs topdown 2-node | challenger wins | no win | no win |
| bottomup 3-node vs bottomup 2-node | indistinguishable (underpowered) | indistinguishable (underpowered) | indistinguishable (underpowered) |
| bottomup 2-node vs topdown 2-node | challenger wins | challenger wins | challenger wins |
| bottomup 3-node vs topdown 3-node | challenger wins | challenger wins | challenger wins |

The six bottom-up "wins" are the artifact. The three "indistinguishable" cells
are correct but for a reason the rule cannot state: both arms sit on the floor,
not near zero by merit. The top-left "challenger wins" is the permutation trade
described below. **The only cell that means what it says is
`topdown 3-node vs topdown 2-node` on the two side views.**

## Does the third node earn its place?

### top-down: no

| view | shared% 3-node vs 2-node | relative | p | jump% | p |
|---|---|---|---|---|---|
| top | 0.820 vs 1.095 | **−25.1%** | 0.0047 | 0.377 vs 0.522 | 0.00023 |
| side1 | 5.605 vs 3.355 | **+67.1%** | 1.5e-27 | 0.670 vs 0.549 | 0.0082 |
| side2 | 7.365 vs 3.780 | **+94.8%** | 4.7e-55 | 0.972 vs 0.561 | 1.0e-15 |

On top it passes the rule outright. On both side views it fails on every metric,
by margins far outside noise. The direction is consistent per group: on side2 the
3-node arm is worse in **4 of 4** groups, on side1 in 3 of 4.

Production reconstruction consumes all three cameras, and one model pair serves
both side views. A detector that is 25% better on one camera and 67–95% worse on
the other two is a net loss. **The third node does not earn its place in
top-down.**

### bottom-up: not established, and what can be seen says no

Shared-node cannot discriminate: 0–2 shared pairs per view in each arm, which is
the floor described above rather than a measurement. On the metrics that can:

| view | jump% 3-node vs 2-node | p | swap% | p |
|---|---|---|---|---|
| top | 0.241 vs 0.296 | 0.065 | 0.0818 vs 0.1159 | 0.064 |
| side1 | 0.542 vs 0.358 | **3.2e-06** | 0.0501 vs 0.0596 | 0.49 |
| side2 | 0.565 vs 0.406 | **1.1e-04** | 0.1041 vs 0.0719 | 0.068 |

No significant improvement anywhere; significant *degradation* on both side
views' tail-jump. **The third node does not earn its place in bottom-up
either**, on this evidence.

## The permutation trade — found, and only by the metric that was missing

**The pre-registered `tail_jump` metric cannot detect the permutation chimera it
was introduced to detect.** This is a property of the arithmetic, not of the
data. `tail_jump` asks, for each predicted tail, how far the *nearest* tail in
the adjacent frame is. A permutation — A takes B's tail, B takes C's — moves the
instance-to-tail *assignment* and leaves the *set* of tail positions untouched.
Every tail still finds a tail 0 px away. The metric reads a clean 0. This is
demonstrated, not argued, in `tests/test_shared_node_20k.py`.

`tail_swap` was added to close that hole: link instances across adjacent frames
by **mouth** (Hungarian, rejecting any link whose mouth moved more than the
threshold — at 166 fps a mouth cannot cross half a body length in 6 ms), then
ask how far *that instance's* tail moved. A stationary mouth with a teleporting
tail is the permutation. It is deliberately **not** in the decision rule: the
rule was fixed before the gap was found, and widening a pre-registered rule
after seeing the numbers is how a rule stops being one.

It changed a verdict:

> **top view, top-down 3-node vs 2-node.** Passes the rule — shared-node
> −25.1% (p=0.005), tail-jump −27.8% (p=0.0002), printed as `challenger wins`.
> Its mouth-linked swap rate is **0.0900% vs 0.0578%, a 56% increase,
> p=0.045**. The one arm that won its comparison outright is the arm that
> converted duplications into permutations, and the pre-registered metric
> reported that as an improvement.

On the side views the two metrics agree (3-node worse on both), so the trade
does not change those verdicts — but it is the single reason the top-view result
should not be read as "the third node helps on top".

## Architecture, judged on the metrics that can compare them

| view | jump: bottomup 2-node vs topdown 2-node | p | swap | p |
|---|---|---|---|---|
| top | 0.296 vs 0.522 (bottom-up better) | 1.3e-09 | 0.1159 vs 0.0578 (**top-down better**) | 0.00079 |
| side1 | 0.358 vs 0.549 (bottom-up better) | 1.3e-06 | 0.0596 vs 0.0679 | 0.58 (tied) |
| side2 | 0.406 vs 0.561 (bottom-up better) | 1.5e-04 | 0.0719 vs 0.0885 | 0.32 (tied) |

Bottom-up is genuinely better on tail-jump in all three views. On the
permutation-sensitive metric it is significantly *worse* on top and
indistinguishable on both sides.

And it buys some of that with missed detections. Frames with fewer than three
instances, of 20,000:

| view | topdown 2-node | bottomup 2-node |
|---|---|---|
| top | 1422 (7.1%) | 891 (4.5%) |
| side1 | 1156 (5.8%) | **2612 (13.1%)** |
| side2 | 1007 (5.0%) | **2647 (13.2%)** |

Round 7's bottom-up misses an animal in 13% of side frames — 2.3× the incumbent.
For 3D reconstruction from three cameras that is the expensive failure, and it is
also a reason its part-mixing rates flatter it: an animal that was never detected
cannot be mixed with anything. The `/1k pairs at risk` column in the tool exists
for this and is in the JSON.

**Keep top-down 2-node round 8.** It is best or statistically tied on the
permutation-sensitive metric in all three views, and its side-view detection
completeness is 2.3× better than the bottom-up alternative.

## The confound, and why it now looks load-bearing

The 2-node top-down arm is round 8 itself — 1,382 (top) / 2,764 (side) training
frames. The 3-node arms trained on 409 / 250. A difference between them is
skeleton **or** data volume, and this measurement cannot separate the two.

What makes this more than boilerplate is that **the degradation tracks the data
deficit, not the skeleton**:

| view | training ratio, 2-node : 3-node | 3-node outcome |
|---|---|---|
| top | 1382 : 409 = **3.4×** | shared-node −25% (better) |
| side | 2764 : 250 = **11.1×** | shared-node +67% / +95% (much worse) |

The arm with the mild deficit improved; the arm with the severe deficit
collapsed. That is exactly what "250 frames is not enough" predicts, and it is
an alternative explanation for the entire side-view result that this design
cannot exclude.

### What the control costs

A 2-node centered-instance head trained on the *same* 409 top / 250 side frames
with round-8 parameters, then run through this tool.

- Training: the s07 3-node runs took **2.36 h** (top, 107 epochs) and **1.77 h**
  (side, 102 epochs) of measured wall clock, from their `training_log.csv`. A
  2-node head on the same frames is the same cost or slightly less.
- Inference: 20,000 frames × 3 views at the measured ~23 fps = **0.7 h**.
- **Total ≈ 5 h local, no HPC.**

**It is now worth running.** Before this measurement the confound was a caveat;
now it is the leading rival explanation for the headline result, and 5 h buys a
clean answer to "skeleton or data volume". Note this is a scope decision for the
experimenter — the control was offered and declined once, before the pattern
above existed.

## Cost and where this ran

Local, RTX 5070 Ti. **The whole 240,000-inference run took ~3.5 h, not the
~7.4 h the brief projected, and saion was not needed.** Two throughput findings
worth keeping, both measured:

1. **Batch size 4, not 16.** The 3-node bottom-up head emits stride-2 confmaps
   and PAFs at 1200×1200; batch 16 peaks at **14.29 GiB** on a 16 GiB card and
   thrashes to 2.8 fps, where batch 4 peaks at 3.80 GiB and runs 18.9 fps —
   **6.8× on identical arithmetic**. Batch 4 was faster for every other model
   too (top-down went 5.9 → 10.5 fps). Verified as a throughput knob rather than
   an algorithm change: against batch 16 over 200 frames, instance counts were
   identical in every frame and the median per-frame coordinate delta was
   0.0002 px (99th percentile 0.0005 px) against a 5 px threshold.
2. **Arrival order mattered more than the model.** The 3-node bottom-up ran at
   0.69 fps as the fourth model in a process and 1.61 fps alone — same model,
   same frames, 2.3× apart on nothing but the earlier models' cached
   allocations. The tool now releases the card between models.

`paf_workers` was tested and is byte-identical to `paf_workers=0` but only ~10%
faster; it is not used.

## n_groups = 4, and what that bounds

Groups **G, H, I, K** — 4 of the corpus's **14**. Four experiments is four
experiments. The independent unit is the group, so every table above is also
reported per group in the JSON, and the pooled numbers are sums over groups, not
averages over folders.

What this does and does not support:

- **Supported.** The direction of the top-down 3-node side-view failure is
  consistent in 4/4 groups on side2 and 3/4 on side1, at p ≈ 1e-27 and below.
  That is a property of the model, not of one recording.
- **Supported.** The bottom-up shared-node floor is architectural and needs no
  sampling argument at all — it follows from the 9.72 px minimum over every
  frame observed.
- **Not supported.** Any absolute rate. Group H is 3–5× worse than group G for
  every model in every view (top-down 2-node on top: 2.500% vs 0.740%). With 4
  of 14 groups, the pooled rates are not a corpus estimate, and the 10 unmeasured
  groups could move them.
- **Not supported.** Anything about tracking, identity over time, or 3D
  reconstruction. This measures single-frame part assignment and one-frame
  temporal coherence, nothing further.

None of these numbers required ground truth, so the locked benchmark's 19–28%
contamination cannot reach them. The only human labels read were the two medians
used to derive thresholds.

## Next

1. **Run the 2-node/409+250-frame control** (~5 h, local). It is the one
   measurement that separates skeleton from data volume, and the side-view
   result now hinges on it.
2. **Do not adopt bottom-up on the shared-node result.** If bottom-up is
   revisited, judge it on tail-swap and detection completeness, and fix the 13%
   side-view miss rate first.
3. **Retire shared-node as a cross-architecture metric.** It is sound within
   top-down and degenerate for bottom-up; using it to rank architectures again
   would repeat the error this document records. `tail_swap` should be the
   primary metric for any future part-mixing comparison, with `tail_jump`
   retained as the coherence check it actually is.
4. Register the s10 outputs, and the still-unregistered locked HITL-v2
   benchmark, in `MANIFEST.json`.
