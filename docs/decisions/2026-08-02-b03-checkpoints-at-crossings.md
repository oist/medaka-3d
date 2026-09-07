# 2026-08-02 — B0.3: architecture comparison at crossings (complete)

> **⚠️ DO NOT USE THESE NUMBERS TO DECIDE AN ARCHITECTURE SWITCH.**
>
> **The evaluation splits leak.** Every figure below comes from each checkpoint's
> *own* validation split, drawn from the same labelling pool as its training data —
> §4.4's leaky-split problem, noted in the caveats at the time as a footnote when
> it was in fact decisive.
>
> Measured effect: on frames held out from the round-7 packages, top-down's top
> crossing recall is **73.33%**, against **84.44%** on the in-sample split below
> under an identical criterion (both-node @20 px). Bottom-up is **80.00%** on
> both. **The leak flatters top-down by ~11 points and is large enough to invert
> the top-view ranking.**
>
> A second sensitivity compounds it: the margin ranges **4.4–12.2 points**
> depending only on the match criterion (mean-node vs both-node, 5 px vs 20 px),
> with top-down winning on this frame set under all four — so criterion choice
> alone does not explain the inversion, but it is not fixed either.
>
> **What still stands:** the resolution correction (both bottom-up arms scored at
> `scale=1.0`; the half-resolution arm was a confound worth ~half the original
> apparent gap), the empty-prediction-file finding, and the frame-to-video
> permutation bug. **What does not:** the "do not switch architecture" conclusion,
> which requires a held-out comparison with the criterion fixed in advance.

**Both views scored at equal input resolution. Do not switch architecture —
top-down wins everywhere, and on the production view the two architectures are
nearly equivalent on separated frames (85.2% vs 83.2%) while top-down leads by
14.5 points at crossings. That is exactly the effect an all-frames average
hides.**

> **CORRECTED 2026-08-02.** The first version of this record compared
> full-resolution top-down against `round7_260704.bottomup_*`, which trains at
> `preprocessing.scale = 0.5`. That **confounded architecture with input
> resolution** — at half scale, two fish 25 px apart are 12.5 px apart in the
> model's input, which is the regime under test. The primary bottom-up arm is now
> `round7s1.bottomup_*` (`scale = 1.0`, matching top-down); the half-resolution
> run is retained as a labelled sensitivity arm.
>
> **The verdict is unchanged but the magnitude was inflated.** Resolution
> accounted for a large share of the apparent gap: on top, bottom-up at crossings
> is 73.3% at full resolution versus 60.0% at half. Original figures are kept
> below as the sensitivity row.

**Reproduce:**
```
.venv\Scripts\python.exe tools/b03_run_top_inference.py          # regenerates the missing top predictions
.venv\Scripts\python.exe tools/b03_score_checkpoints_at_crossings.py
```

Supersedes the partial version of this record: `top` was previously unscorable.

---

## Method

Greedy one-to-one matching of predicted to GT instances by mean node distance. A
GT instance counts as detected when its match is within tolerance. A frame is a
**crossing** when two GT instances have nodes closer than 25 px (~0.25 body
length, per B0.1). Recall is over GT instances.

Side predictions were already saved next to the checkpoints. **Top predictions
had to be regenerated** — all four round-7 top runs shipped an empty
`labels_pr.val.0.slp` (0 frames against 129 GT). Regenerated with
`sleap-nn==0.3.1` into `work_dir`; the inherited `/bucket` files were not
modified.

**GPU.** Regenerated on the RTX 5070 Ti with `torch==2.13.0+cu130` (4.3 fps, vs
0.8 fps on the CPU wheel). The two runs produced **identical instance counts**
(376 top-down, 357 bottom-up) and identical scores to 0.01 px, so nothing below
depends on the device. `sm_120` is present in the wheel's arch list — the check
OIST_HPC pitfall 26 asks for.

---

## TOP — the production view (round 7 is production for top)

| architecture | scale | regime | frames | GT / pred | @5px | @10px | @20px | median err |
|---|---|---|---|---|---|---|---|---|
| **top-down** (production) | 1.0 | separated | 99 | 297 / 289 | **85.2%** | 94.3% | 95.3% | 1.83 px |
| | | crossing | 30 | 90 / 87 | **87.8%** | 91.1% | 93.3% | 2.13 px |
| **bottom-up** (primary) | 1.0 | separated | 99 | 297 / 284 | **83.2%** | 93.3% | 93.6% | 2.03 px |
| | | crossing | 30 | 90 / 75 | **73.3%** | 80.0% | 81.1% | 2.39 px |
| bottom-up (sensitivity) | 0.5 | separated | 99 | 297 / 286 | 78.8% | 91.6% | 92.6% | 2.69 px |
| | | crossing | 30 | 90 / 71 | 60.0% | 72.2% | 76.7% | 2.98 px |

**This is the clearest result in B0.** At equal resolution the two architectures
are nearly indistinguishable on separated frames — 85.2% vs 83.2%, a 2.0-point
gap — and then diverge to **14.5 points** at crossings. A metric averaged over all
frames, which are overwhelmingly separated, would have called these two
equivalent. **The regime, not the average, is what separates them.** That is
precisely the failure mode B0.3 was written to detect.

**Top-down does not degrade at crossings on this view at all.** Recall@5px is
*higher* at crossings (87.8% vs 85.2%); at @10px it falls only 3.2 points. It
finds 87 of 90 instances.

**Bottom-up degrades and loses animals**: 73.3% vs 83.2% at @5px, emitting **75
predictions for 90 GT instances** — ~17% of animals undetected in crossing
frames.

**Resolution mattered, and explains much of the original overstatement.** The
half-resolution arm reads 60.0% at crossings against the full-resolution 73.3%.
Roughly half of what the first version of this record attributed to architecture
was input scale.

## SIDE — round 7 is the fallback here (round 8 is production for side)

| architecture | scale | regime | frames | GT / pred | @5px | @10px | @20px | median err |
|---|---|---|---|---|---|---|---|---|
| **top-down** | 1.0 | separated | 200 | 595 / 595 | **92.1%** | 94.3% | 94.6% | 1.10 px |
| | | crossing | 59 | 174 / 174 | **84.5%** | 86.2% | 89.1% | 1.27 px |
| **bottom-up** (primary) | 1.0 | separated | 204 | 606 / 562 | **77.6%** | 83.5% | 84.8% | 1.89 px |
| | | crossing | 55 | 164 / 124 | **58.5%** | 68.3% | 70.1% | 2.42 px |
| bottom-up (sensitivity) | 0.5 | separated | 210 | 626 / 573 | 77.2% | 86.1% | 87.9% | 2.41 px |
| | | crossing | 49 | 147 / 104 | 50.3% | 61.9% | 66.7% | 3.19 px |

Architecture gap at @5px: **14.5 points separated → 26.0 points crossing, 1.8×
wider.** Bottom-up emits 124 predictions for 164 GT instances — ~24% of animals
undetected.

Unlike top, side shows a large architecture gap even on separated frames, so here
the all-frames average would already have picked top-down. Resolution matters less
on this view (77.6% vs 77.2% separated) but still costs 8 points at crossings.

---

## Gate B0 — met

The gate asks for a detection-quality number at crossings versus overall, for both
architectures, and says it decides retrain-vs-switch-architecture.

**Neither branch is supported.**

- **Switch architecture: no.** At equal input resolution, top-down beats bottom-up
  in every regime on both views. The concern that motivated B0.3 — a model winning
  on average while losing where it matters — turns out to point the *same* way as
  the production choice. On **top** it is stark: the two are within 2.0 points on
  separated frames and 14.5 apart at crossings, so an all-frames average would
  have called them equivalent and the crossing-stratified number is the only thing
  that distinguishes them.
- **Retrain on crossing frames (B0.5): not supported either.** B0.1 found the
  training set is already 62.8% close-proximity frames, so it is not short of
  them, and top-down does not collapse on them.

**What the evidence does point at** is rare, large localisation failures rather
than systematic degradation. Top-down's median error barely moves at crossings
(side: 1.10 → 1.27 px) while its 90th percentile jumps (side: 3.27 → 15.60 px).
That is the signature of "one tail joined to two mouths" — and B0.2 found a
concrete case in the round-7 production output (`K_1_1`, two predicted tails
3.4 px apart, one with a NaN mouth). **B0.4's geometric filter is the right next
move**, keyed to the ~19 mm labelled span B0.2 measured, not the 25–35 mm body
length.

---

## The round-8 gap — and why the reported failure remains unexplained

The experimenter's report is about **round 8**. B0.3 measures **round 7**, because
round 8 has no bottom-up arm — only `centroid` and `centered_instance` runs exist,
so no architecture comparison is possible there. **Confirmed by the experimenter
2026-08-02: round 8 was trained top-down only.** So round 7 is the only round that
can answer the gate's question, and B0.3's scope is correct rather than a
shortcut.

There is also a **round9b** side label set (3,118 frames, superseding round 8's
2,764 — see B0.1), but **no round-9 model runs exist anywhere on `Z:`**: it is
labels only, not yet trained. It cannot be scored, and it does not change the
architecture question, which is settled on round 7.

Round 8's own saved validation predictions do exist and are non-empty. They are
also **far too small to say anything about crossings**:

| round-8 run | val frames | crossing | separated |
|---|---|---|---|
| top `centered_instance_n1382` | 20 | **5** | 15 |
| side `centered_instance_n2764` | 40 | **6** | 34 |

**The round-8 promotion decision rests on 5 crossing frames for top and 6 for
side.** That is the entire validation evidence at crossings for the model reported
to fail at crossings. No stratified number computed from it would be worth
reporting, so none is computed here.

This is the real gap: **round 7 is measured and healthy at crossings; round 8 is
the one under suspicion and is effectively unmeasured there.** Closing it needs a
crossing-stratified evaluation set — which is exactly what B0.2 built (1,888
encounters) and what Gate A will freeze.

---

## Two data defects found while doing this

**1. All four round-7 top runs saved empty prediction files.** 0 frames against
129 GT, ~28 KB of header. The first version of the scoring tool reported this as
0.0% recall for every top architecture — a badly wrong finding that would have
read as catastrophic model failure. The tool now refuses to score a missing
artifact, and the predictions have been regenerated.

**2. The frame-to-video assignment in the saved validation labels is a
permutation.** Only **17 of 129** top frames have a `frame_idx` present in their
*own* video's `source_inds`; all 129 are present somewhere in the union. The video
list maps to datasets in the order video6, video1, video7, video3, video5, video4,
video0, video2. Trusting `frame.video` runs inference on the wrong image, or
raises `IndexError: Frame index 309598 out of range`.

Two traps in fixing it, both recorded in `tools/b03_run_top_inference.py`:

- The frame index must **not** be rewritten to a dataset row. `HDF5Video` keys its
  `frame_map` on the *source* index (69857 → row 0) and overrides `has_frame` to
  consult it, so `frame_idx` was already correct. Converting it — the obvious fix —
  breaks reading.
- After inference, GT and predictions **cannot be joined on video filename**:
  `predict()` restores source video paths, so one frame is `labels.pkg.slp` in the
  predictions and `edits.pkg_full.slp` in the ground truth. The scorer now selects
  a join strategy per file, verifies it covers every GT frame, and prints which it
  used — a silent mis-join produced the 10.4% recall seen mid-investigation.

**The self-check that validates the remapping:** if frames had been paired with
the wrong images, predictions would not land on the ground truth. Top-down scores
85–96% with a 1.83 px median error, so the images matched.

---

## Dependency change this required

`sleap-nn==0.3.1` requires `sleap-io>=0.9.2,<0.10.0`, so the sleap-io pin moved
**0.7.0 → 0.9.2**. 0.9.2 is exactly the version `PRIOR_FINDINGS.md` §4.5 flags as
never having had its save→reopen round trip re-validated after the 0.8.0
view-scramble bug.

That validation is now a test rather than a comment:
`tests/test_sleap_io_roundtrip.py` writes a three-view labels file whose
coordinates encode which view each instance belongs to, reopens it, and asserts
the mapping survived. **It passes on 0.9.2 on this machine.** It also pins-checks
both versions, so moving either pin fails loudly until the round trip is re-run.

---

## Caveats

- These are each run's **own validation split**, drawn from the same pool as its
  training data — 129 top / 259 side frames, not a held-out benchmark. They
  compare architectures against each other, nothing more. Gate A's frozen `test`
  split is still required before any promotion claim.
- **Input resolution is now controlled but other training hyperparameters are
  not.** `round7s1.bottomup_*` matches top-down on `preprocessing.scale` only;
  no audit was made of the remaining config differences, so residual confounds
  may persist. The half-resolution arm demonstrates how much a single
  uncontrolled setting was worth — roughly half the apparent gap on top.
- Top predictions were regenerated with `sleap-nn` 0.3.1 defaults
  (`peak_threshold=0.2`, `max_instances=None`). The side predictions were
  produced by the original pipeline with settings that are not recorded. **Top and
  side numbers are therefore not comparable to each other** — only top-down vs
  bottom-up *within* a view is, since those share settings.
- The two architectures' crossing/separated partitions differ slightly on side
  (200/59 vs 210/49) because each is computed from that run's own GT file.
- 25 px as the crossing threshold is a choice; B0.1 gives the distribution behind
  it. The direction of the ranking is stable, the exact percentages are not
  threshold-free.
- `MANIFEST.json` declares only **23 of 158** files under `sleap_models/`. The
  bottom-up runs and every `labels_{gt,pr}.val` file used here are undeclared and
  unchecksummed — design rule 3 is still unsatisfied. Regenerating overwrites an
  inherited index, so it is left for explicit approval.
