# 2026-08-02 — B0.1: has the production model ever seen a crossing?

> **DEFINITION CORRECTED 2026-08-02.** This record originally headlined
> **2,603 / 4,146 (62.78%)**, which is the *all-visible-centroid* count — it treats
> a mouth-only instance's single node as its centroid, and ~9% of instances are
> missing a node.
>
> **The primary figure is the complete-instance count: 2,107 of 3,974 eligible
> frames = 53.02%** below 100 px, with 172 frames ineligible for holding fewer
> than two complete instances. Independently recomputed here. The 62.78% and the
> minimum-visible-node 80.20% are **sensitivities that answer different
> questions** and are not interchangeable with it.
>
> The conclusion is unaffected — "the model has never seen a crossing" is false
> under every definition — but rejecting that claim does **not** show the label mix
> is adequate for the failure regime. The tables below are the original
> all-visible figures; read them as the sensitivity, not the headline.

**Yes, constantly. The roadmap's expected finding is wrong, and so is the
inference that produced it.**

**Reproduce:** `.venv\Scripts\python.exe tools/b01_crossing_frames_in_training.py`

> **Updated 2026-08-02 (later): re-run on the latest and largest labels.** The
> original pass used `labels.round8.combined` (4,146 frames). Repeated on the
> newest set per view — `labels.round8.top.train` (1,382) plus **round9b side**
> (3,118), 4,500 frames total — **every conclusion is unchanged**: 62.51% of
> frames within 100 px (vs 62.78%), 24.40% with nodes inside a quarter body
> length (vs 24.58%), **0** frames sharing a node, median span 101 px (vs 102).
> The finding is not an artifact of dataset choice. Both figures are given below.

---

## The prediction, and why it did not apply

ROADMAP B0.1 says: *"Count labelled frames in round8 `combined_labels.pkg.slp`
containing two instances closer than 100 px. §2.10 predicts approximately zero.
Expected finding: the model has essentially never seen a crossing."*

**§2.10 does not say that about this dataset.** It measures two different things:

- *"every clean-mask frame has 3D distance >20 mm"* — the **3D mask** pipeline;
- *"inter-male distance min 117 px, median 386 px, 0% closer than 100 px"* — the
  **Detectron2 appearance-classifier crops**, after *"the concrete-tracklet gate
  removes close-contact frames"*.

Both describe datasets produced by a **filter that gates out close contact**. The
SLEAP training labels never passed through that gate. §2.10 is correct about what
it measured; the roadmap generalised it to a dataset it never covered. The pixel
figures are not even commensurable — §2.10's are measured in *crop* space, this
is a full 1200×1200 frame.

This is the failure mode the project's own rules name: *enumerate, don't
extrapolate*.

---

## What the training set actually contains

`labels.round8.combined` = `Z:\StephensU\medaka\sleap_models\retrain_20260722_round8\combined_labels.pkg.slp`

- **4,146 labelled frames**, 27 embedded videos, skeleton `[mouth, tail]`.
- **top 1,382 / side 2,764** — which exactly equals `labels.round8.top.train`
  (1,382) + `labels.round8.side.train` (2,764). This file *is* the training set;
  the val splits (20 top + 40 side) are separate.
- Instances per frame: 3 in 4,143 frames, 2 in 1, 4 in 2. Every frame is
  multi-animal.
- Provenance survives in `video.source_video`: mostly the round-7 label packages
  (`retrain_20260704/{top,side}/labels.pkg.slp`), plus three videos taken
  straight from `20251225_K_1_2_FCMC`'s three camera folders.

### Close-proximity frames, by three definitions

Percentages are of the 4,146 frames with ≥2 instances.

| metric | <50 px | <100 px | <150 px | <200 px |
|---|---|---|---|---|
| centroid | 1,154 (27.8%) | **2,603 (62.8%)** | 3,319 (80.1%) | 3,643 (87.9%) |
| min-node | 2,219 (53.5%) | 3,325 (80.2%) | 3,681 (88.8%) | 3,835 (92.5%) |
| mouth–mouth | 1,143 (27.6%) | 2,299 (55.5%) | 2,979 (71.9%) | 3,298 (79.6%) |

Closest-pair distance across all frames: centroid min **2 px**, 1st pct 10 px,
median 78 px. Min-node min **3 px**, 1st pct 8 px, median 46 px.

**At the roadmap's own threshold the answer is 2,603 frames, not ~0.**

### In body lengths — the scale-free version

Median labelled mouth–tail span is **102 px**, so:

| metric | <0.25 body | <0.5 body | <1 body | <2 body |
|---|---|---|---|---|
| centroid | 324 (7.8%) | 1,184 (28.6%) | 2,649 (63.9%) | 3,654 (88.1%) |
| min-node | 1,019 (24.6%) | 2,261 (54.5%) | 3,342 (80.6%) | 3,850 (92.9%) |

**A quarter of all training frames have two fish with nodes closer than a quarter
of a body length.** The model has seen contact, repeatedly, with human labels
resolving it.

### Confirmed on the latest and largest labels

Every label set in the manifest, enumerated rather than assumed:

| set | frames | instances | note |
|---|---|---|---|
| `labels.round7.top` | 1,295 | 3,885 | superseded |
| `labels.round7.side` | 2,590 | 7,771 | superseded |
| `labels.round8.top.train` | **1,382** | 4,146 | **latest top** — no round-9 top exists |
| `labels.round8.side.train` | 2,764 | 8,293 | superseded by round9b |
| `labels.round8.combined` | 4,146 | 12,439 | = round8 top.train + side.train |
| round9_side `train_labels` | 3,086 | 9,259 | superseded by round9b |
| **round9b `train_labels`** | **3,118** | **9,354** | **latest side** |

round9b is not merely bigger: **every one of its 3,118 frames carries exactly 3
instances**, where round9_side still had one 2-instance and two 4-instance
frames. It is the completed set.

Latest-and-largest is therefore **round8 top + round9b side = 4,500 frames**,
354 more than `round8.combined`. Re-running on it:

| metric | <100 px | <0.25 body | <1 body | shared node | median span |
|---|---|---|---|---|---|
| round8.combined (4,146 fr) | 62.78% | 24.58% | 63.89% | **0** | 102 px |
| round8 top + round9b side (4,500 fr) | **62.51%** | **24.40%** | **63.20%** | **0** | 101 px |

**No conclusion moves.** The result is a property of the annotation, not of which
round is chosen.

Note round9b is a 1.2 MB labels-only file with no embedded images — fine for
coordinate analysis, but inference on it would need the source videos.

### Spot-check of the raw coordinates

Because this claim overturns a documented expectation, four of the close frames
were read back node by node. They are genuine:

- frame 81617, gap 29.6 px — two fish swimming parallel, both fully labelled
  (`mouth=[642.5, 672.3] tail=[525.7, 685.1]` vs `mouth=[641.4, 702.5]
  tail=[532.1, 713.9]`);
- frame 163702, gap 24.6 px — two fish crossing, mouths 39 px apart;
- frame 161265, gap 23.4 px — two instances **labelled at the mouth only**, tails
  `NaN`.

### Partially-labelled instances

Of 12,439 instances: **90.5% have both nodes**, 3.9% mouth only, 5.3% tail only,
0.4% empty. So ~9% of training instances are missing a node. Some of the close
pairs above are therefore proximity between partially-labelled animals — still a
real measurement of proximity, but worth knowing before treating the labels as
complete. It also means the model was trained with a non-trivial rate of absent
tails, which is a candidate explanation for the NaN-mouth predictions seen in
B0.2's round-7 output.

### No shared nodes in the training data

**0 frames** have two instances within 1 px of each other on any node. So the
"one tail joined to two mouths" failure is **not** present in the labels — it is
generated at inference. The human annotator always separated them.

---

## Pixel scale — an open question from B0.1, now closed by B0.2

B0.1 flagged that 1200 px = 200 mm implies a median fish of 17.0 mm, against the
25–35 mm in `CLAUDE.md`. B0.2's empirical estimate — the extent the animals
actually cover, against the calibrated 192 mm inner span (§2.11) — gives
**0.1887 mm/px**, i.e. the tank occupies ~1,017 of the 1,200 px.

At that scale the median mouth–tail span is **19.3 mm**, still short of 25–35 mm.
The most likely reading is that the **`tail` node sits at the tail base or fork,
not the fin tip**, so the labelled span is not the whole animal.

**This matters for B0.4.** A geometric plausibility filter keyed to "body length
25–35 mm" would reject most *correct* instances. The filter must be calibrated
against the labelled mouth–tail distribution (median 19.3 mm, 5–95%
6.8–26.4 mm), not against the animal's true length. Do not implement B0.4 until
this is confirmed against the calibration model.

---

## What this changes

1. **B0.5 ("retrain with crossing frames") loses its stated rationale.** The
   training set is not short of crossing frames. Adding more of the same is
   unlikely to fix a failure that is not caused by their absence. Do not schedule
   the annotation pass B0.5 implies on this argument.
2. **The failure is at inference, not in the data.** Labels never share a node;
   predictions do (see B0.2 — a round-7 `K_1_1` frame has two instances whose
   tails are 3.4 px apart, one with a NaN mouth). That points at decoding,
   grouping, or NMS — which is exactly what B0.3 measures, and it makes B0.3 the
   load-bearing step of this phase.
3. **B0.4's threshold must be derived, not assumed** — see above.
4. **ROADMAP B0.1's "expected finding" should be struck**, and §2.10 annotated to
   say which datasets it covers.

## Caveats

- "Two instances close together" is not identical to "a crossing with occlusion".
  These counts establish that close-proximity, human-resolved frames are abundant;
  they do not prove every one is a hard case.
- Training frames are annotator-selected, not a random sample of video. A high
  crossing rate may reflect deliberate selection of hard frames — which would
  *strengthen* the conclusion that coverage is not the problem.
- The count is over labelled frames only. It says nothing about whether the
  *augmentation* or *sampling* during training preserved that proportion.
