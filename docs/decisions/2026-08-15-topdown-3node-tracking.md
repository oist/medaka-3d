# Top-down 3-node, scored as a production candidate: it loses, and the tracking chain says why

**Date:** 2026-08-15
**Tools:** [`c05`](../../tools/c05_saion_inference_td.py) (full-length top-down inference),
[`c01`](../../tools/c01_s16_predictions.py) / [`c03`](../../tools/c03_track_3d.py) /
[`c04`](../../tools/c04_score_tracking.py) (unchanged, run with `--run topdown`),
[`c06`](../../tools/c06_compare_architectures.py) (the comparison and its rule)
**Builds on:** [2026-08-09 detector part-mixing](2026-08-09-detector-part-mixing-20k.md),
[2026-08-13 reviewed labels](2026-08-13-reviewed-labels-bottomup-3node.md),
[2026-08-13 tracking](2026-08-13-tracking-3d.md)
**Results:** `work_dir/c06_compare_architectures/`, `work_dir/c0{3,4}_*/topdown/`,
predictions at `sleap_output_3node/c05_td_K_1_all/`

## Verdict

**Keep bottom-up.** Top-down 3-node cleared exactly one of the five
pre-registered clauses — reconstruction coverage, by 0.23 percentage points —
and failed the other four, three of them by margins that are not close.

The rule's own word for that outcome is *trade-off*, and it is printed unaltered
below. **The word undersells it.** Top-down bought 0.23 points of coverage and
paid **900–2,100×** more shared tails, **1.9–5.8×** the mouth-linked swap rate,
and **11×** the track jumps. This is not a close call.

**The run answered the question it was commissioned for.** Part-mixing does
*not* get absorbed by three cameras and a 3D correspondence — it propagates.
Top-down's tail duplication survives reconstruction, shows up as 11× the
physically-impossible track motion and a 2.6-point drop in the independent
cross-view check, and the tracking evidence therefore **confirms** the 2026-08-09
and 2026-08-13 decisions rather than overturning them.

**One genuine top-down advantage was found and is worth keeping in view:** it
almost never fails to place a mouth — **0.021–0.029%** of complete frames against
bottom-up's **1.065–1.507%**, a 50× difference. That is the node the whole
correspondence pipeline is anchored on. It did not rescue top-down here, but it
is the most interesting thing this run produced.

## The comparison was made possible by controlling a confound, not by ignoring one

**`sleap-nn/0.3.1`, the runtime `s16` ran under, was withdrawn from saion between
2026-08-11 and 2026-08-14** — modulefile gone, install directory emptied from
~15 GB to 2.5 MB with no `bin/`. The first submission died instantly at
`module load` under `set -euo pipefail`, which is how this was found rather than
inferred.

Running top-down under 0.3.3 while the bottom-up baseline came from 0.3.1 would
have varied architecture and runtime together. Controlled by measurement:
three bottom-up chunks — **75,000 frames, 225,000 instances** — re-run under
0.3.3 on the same A100 class with the same model and batch size, against the
0.3.1 originals:

| view | frames | same instance count | coordinate delta |
|---|---|---|---|
| top_cam | 25,000 | 100.00% | **100.00% exactly zero** |
| side1_cam | 25,000 | 100.00% | **100.00% exactly zero** |
| side2_cam | 25,000 | 100.00% | **100.00% exactly zero** |

Bit-identical. The output also still loads under the local `sleap-io==0.9.2` pin,
which a newer writer is not obliged to guarantee. Two related cluster facts, both
verified by probe jobs and recorded in [`OIST_HPC.md` §2a](../OIST_HPC.md):
`module avail` is only meaningful from a **compute node**, and `module load`
never put `python` on `PATH` — `s16` worked because of an absolute interpreter
path, so the `module load` line was doing nothing and is now removed.

## What ran

3 experiments × 3 views × 348,600 = **3,137,400 inferences**, the identical
frames `s16` covered, so the two architectures are compared frame for frame.
Top-down is two checkpoints — round 8's centroid head plus the `s14` 3-node
centered-instance head trained on the same 543 reviewed moments — against
bottom-up's one. That asymmetry is a property of the architectures.

saion `largegpu`, 8-task array, **27–29 fps** on A100 (about 60% of bottom-up's
rate, because every frame runs two networks). All 8 tasks `COMPLETED`, 126/126
chunks, no `.partial`, no errors; 3:16–5:55 per task against the 12 h wall. Task
0 was requeued mid-run and reported *"1 written, 15 already present"* — the
idempotent skip absorbed it, which is the whole point of that design.

Throughput was **the same in all three views** (median 28.8 / 28.8 / 29.2 fps),
unlike bottom-up where `s16` measured top ~25% slower than side. The 17–33 fps
spread within the run is node and I/O variation, not a view effect: the slowest
chunks occurred on an otherwise idle node.

## 1. Detection, at full length

| run | view | inst/frame | <3 | =3% | >3 | miss mouth% | miss tail% | shared mouth/tail | swap% |
|---|---|---|---|---|---|---|---|---|---|
| bottomup | top | 2.984 | 35,397 | **94.64** | 20,693 | 1.065 | 1.737 | 30 / **19** | **0.0484** |
| bottomup | side1 | 2.952 | 58,121 | **93.07** | 14,351 | 1.492 | 1.774 | 11 / **20** | **0.0400** |
| bottomup | side2 | 2.949 | 56,551 | **93.67** | 9,657 | 1.507 | 1.747 | 7 / **145** | **0.0378** |
| topdown | top | 2.949 | 61,481 | 92.62 | 15,703 | **0.021** | 1.815 | 33 / **16,992** | 0.0910 |
| topdown | side1 | 2.988 | 59,133 | 90.24 | 42,893 | **0.027** | 2.342 | 15 / **42,402** | 0.2102 |
| topdown | side2 | 2.988 | 56,967 | 90.35 | 43,938 | **0.029** | 2.480 | 16 / **36,391** | 0.2180 |

**The shared-tail column is the 2026-08-09 finding reproduced at 116× the scale
and it is worse than that decision could see.** Per 1,000 frames: bottom-up
0.02 / 0.02 / 0.14, top-down **16.25 / 40.55 / 34.80**. Mouth sharing stays
negligible in both (7–33 pairs), exactly as that decision predicted on structural
grounds — the mouth is the top-down anchor, and duplicating it is a different
failure from duplicating a tail.

**`tail_swap` — the metric 2026-08-09 named as shared-node's successor —
confirms it on the metric that is valid across architectures:** 0.0910 vs 0.0484%
on top (p = 2e-82), 0.2102 vs 0.0400% on side1 and 0.2180 vs 0.0378% on side2
(both p < 1e-300). Top-down is 1.9× worse on top and **5.3–5.8× worse on the
sides**.

Two things run the other way and are stated because they are real:

- **Mouth completeness.** Top-down leaves a mouth unplaced in 0.021–0.029% of
  complete frames; bottom-up in 1.065–1.507%. The centroid head anchors on the
  mouth, so it is essentially always placed. Since a missing mouth costs the 3D
  route a whole view for that frame, this is the mechanism behind top-down's
  slightly better reconstruction coverage.
- **Cardinality is a wash, differently.** Bottom-up under-detects (more `<3` on
  the sides); top-down over-detects (4.1–4.2% of side frames carry `>3` against
  bottom-up's 0.9–1.4%). Neither is exact-3 more often — bottom-up wins that in
  every view.

## 2. Tracking, same pipeline, same tracker, same thresholds

Both architectures went through the identical reconstruction and the `gated`
tracker chosen on 2026-08-13. Comparing two detectors through two different
trackers would have confounded them.

| run | resolved% | residual | jumps | / pairs | jump% | frag@90% | enc kept% | x-view% |
|---|---|---|---|---|---|---|---|---|
| bottomup | 96.31 | **0.399 mm** | **111** | 8,286,286 | **0.0013** | **99** | **82.56** | **97.22** |
| topdown | **96.55** | 0.416 mm | 1,228 | 8,331,369 | 0.0147 | 107 | 82.51 | 96.95 |

Per view, track jumps: bottom-up **8 / 58 / 45**, top-down **178 / 591 / 459**.

The independent cross-view check separates cleanly on the node that can see part
mixing. Re-derived from the **pectoral** — never missing in either run, and not
the node the trackers link on — agreement is 97.22% vs 96.95%. Re-derived from
the **tail**, it is **96.75% vs 94.16%**: a 2.6-point gap that appears only in
the node top-down duplicates. That is the part-mixing arriving downstream, and it
is the most direct evidence in this document that the failure is not absorbed.

Encounter preservation is indistinguishable (82.56% vs 82.51%) — and that metric
was already flagged on 2026-08-13 as one that rewards confidence, since `nearest`
scores 100% on it while being the worst tracker in the set. It separates nothing
here and is reported for completeness.

## The pre-registered rule, printed unaltered

Bottom-up is the incumbent; ties go to the incumbent.

| | clause | challenger vs incumbent | |
|---|---|---|---|
| **PASS** | 1 coverage not lower | 0.9655 vs 0.9631 | |
| **FAIL** | 2 track jumps ≥20% lower, p<0.05 | 0.0147% vs 0.0013% | rel **−1000.3%**, p = 2.4e-203 |
| **FAIL** | 3 fragmentation not higher | 107 vs 99 | |
| **FAIL** | 4 cross-view not lower | 0.9695 vs 0.9722 | |
| **FAIL** | 5 `tail_swap` not significantly worse | top 0.0910 vs 0.0484 (p=2e-82); side1 0.2102 vs 0.0400 (p<1e-300); side2 0.2180 vs 0.0378 (p<1e-300) | |

> **-> TRADE-OFF, not a replacement. Keep bottomup.**

Clause 5 existed to stop a tracking metric retiring a measured detection defect
without ground truth. **It was not needed.** Top-down failed the tracking clauses
too, so the conflict the rule was braced for never arose, and the two lines of
evidence agree.

## Seeing it — [`c07`](../../tools/c07_render_tracking.py)

Both architectures drawn on the **identical video frame**, so anything that
differs on screen is the model and nothing else. Under
`work_dir/c07_render_tracking/<experiment>/`, for all three days:

* `montage/` — a 2 × 3 still, architectures down and views across, cropped to a
  fixed **6 body lengths** so two tiles are directly comparable;
* `clips/` — the same grid as MP4 over a 2 s window, written at 30 fps from
  166 fps source, so it plays at 1/5.5 speed. Tracking is temporal and a still
  cannot show it;
* `continuity/` — 3D pairwise separation over the window with each tracker's own
  identity assignment as a strip beneath, and the identity-change count stated.

Frames are chosen **deterministically** in four categories — `shared_tail`,
`track_jump`, `encounter`, and `routine`. The last exists so the set is not only
failures: a reader shown only broken frames cannot judge how often they occur.

Two examples worth opening first, both from K_1_1:

* `montage/shared_tail_f121326_side1_cam.png` — top-down collapses two instances
  onto one animal, 0.11 px apart (ringed red), where bottom-up places one.
* `montage/track_jump_f150064_side1_cam.png` — top-down piles **three** instances
  onto a single fish while a second fish beside it gets none; bottom-up keeps
  them apart. This is the 4.1% `>3` over-detection and the 122 px track jump in
  one picture.

And the identity strips make the fragmentation difference countable rather than
adjectival: over the same 332-frame window at K_1_1 frame 151,281, bottom-up
makes **104** identity changes and top-down **165**.

## What this settles and what it does not

**Settles.** Part-mixing is not absorbed by multi-view geometry. The concrete
chain: top-down duplicates a tail in ~3.5–4.1% of side-view frames → the
tail-derived cross-view correspondence disagrees 2.6 points more often → the
tracker asserts 11× more physically impossible motion. A reader could reasonably
have expected three cameras to average the problem away. Measured, they do not.

**Does not settle.** Whether either tracker is *right*. There is still no human
M1/M2 track anywhere in this project, so nothing here measures identity, only
geometric possibility. Both architectures could be swapping animals at every
crossing and every number above would look the same. That is Gate A's job, priced
in the 2026-08-13 decision at 390 encounters across ≥3 groups.

**Does not settle** whether top-down's mouth completeness is worth harvesting. A
detector that places a mouth 50× more reliably and duplicates tails 900× more
often is not obviously useless — it is differently wrong. Nothing here tests a
hybrid, and nothing here should be read as having tested one.

## n_groups = 1

Every frame on both sides of this comparison is **group K** — one trio of fish on
three consecutive days, repeated measures rather than replicates. 13 of the
corpus's 14 groups are unmeasured.

What survives that limit: the *mechanism*. Shared tails are a property of
top-down's architecture (two centroids can independently regress a tail onto the
same animal; bottom-up's peak NMS and PAF grouping make it near-impossible), and
the 2026-08-09 measurement found the same direction in 4 of 4 groups on side2 and
3 of 4 on side1. The propagation chain measured here is arithmetic, not sampling.

What does not survive it: every absolute rate. 0.2180% is not a corpus estimate
of top-down's side2 swap rate, and 96.55% is not its coverage.

## Next

1. **Nothing changes in production.** Bottom-up 3-node stays; the `s16`
   predictions and the `gated` tracks remain the working set.
2. **Do not retrain or re-score top-down 3-node again.** Three independent
   measurements now say the same thing — 2026-08-09 on 20,000 frames × 4 groups,
   2026-08-13 on the reviewed labels, and this one on 3.1 M frames through a
   full tracking chain. A fourth would be re-asking a settled question.
3. **The bottom-up mouth gap is now the cheapest known improvement.** Bottom-up
   loses 1.065–1.507% of instances' mouths where top-down loses 0.021–0.029%, and
   every one of those costs the reconstruction a whole view for that frame.
   Either switch the tracking anchor to the **pectoral** (missing in 0.000% of
   complete frames, already recommended on 2026-08-13) or investigate whether
   bottom-up's mouth confmap threshold is simply too strict.
4. **Encounter ground truth remains the binding constraint**, unchanged.
5. **Register in `MANIFEST.json`**: the `c05` top-down predictions alongside the
   `s16` bottom-up ones, the promoted calibration, and the still-unregistered
   locked HITL-v2 benchmark.
