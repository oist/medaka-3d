---
hypothesis: >
  Round 8's centered-instance head spends its capacity at the wrong spatial
  resolution: 45% of its parameters sit in a 12x12 middle block where the fish
  occupies 3.5 cells, while keypoint placement happens at stride 1-8. Moving that
  capacity to high resolution — crop 384, max_stride 32, filters 24 -> 64, rate
  held at 2.0 — will improve both-node recall on the locked benchmark without
  touching detection, because the centroid head is unchanged.
stage: detector
expires: 2026-12-01
decision_rule: >
  Baseline is ROUND 8 (production), measured on the locked HITL-v2 benchmark with
  round-8's centroid head unchanged in both arms. ADOPT if ALL THREE hold:
  (1) both-node recall @20 px >= round 8 on the ALL arm (>= 93.33%) AND on the
  CORRECTED arm (>= 91.67%); (2) false positives <= round 8 (0 on ALL);
  (3) shared-tail pairs <= round 8 (1 on ALL). REJECT if both-node recall falls
  more than 2.0 points below round 8 on the ALL arm. Otherwise PARK.
  Animal coverage is NOT a criterion — the centroid head is unchanged, so
  coverage is expected to be identical and any movement in it is a bug to
  investigate, not a result. `val/loss` is explicitly NOT a criterion.
---

# Centered-instance capacity at high resolution

Pre-registered 2026-08-04, before training. **Round 8 is the baseline; round 7 is
not referenced.**

## Why this might work

Two measurements, taken before this was designed.

**1. Detection is not the problem — placement is.** The archive's oracle
evaluation on the locked benchmark: given ground-truth centroids, round 8 scores
**117/117 matched, 0 extra, 0 missed**. Its full top-down run scores the same
117/117 with 0 extra. So the centered-instance head never loses an animal; what it
can still get wrong is where it puts the two nodes. That makes **both-node recall
@20 px** the metric and rules coverage out as one.

**2. Round 8's capacity sits where the fish is not.** At crop 512,
`max_stride=64`, `filters=24`, `filters_rate=2.0` the channel ladder is
24, 48, 96, 192, 384, 768, 1536. The 1536-channel middle block runs at 8x8 for a
512 px crop, where a 113 px fish spans **1.8 cells**. Measured on the built model,
**the two middle blocks hold 55% of all 70.8M parameters.**

The proposed ladder at crop 384, `max_stride=32`, `filters=64`, `rate=2.0`:

| stride | spatial | fish spans | round 8 | **this run** |
|---|---|---|---|---|
| 1 | 384² | 113 cells | 24 | **64** |
| 2 | 192² | 56 | 48 | **128** |
| 4 | 96² | 28 | 96 | **256** |
| 8 | 48² | 14 | 192 | **512** |
| 16 | 24² | 7 | 384 | 1024 |
| 32 | 12² | 3.5 | 768 | 2048 |

2x–2.7x the channels at every level where a keypoint is actually localised, at
**125.6M parameters** — 1.8x round 8 — and a *smaller* receptive field.

**Why not `filters_rate` instead.** A 32/2.5 ladder reaches 221.8M parameters —
77% more than this — but puts **123M of them (55%) in the 12x12 block**, and has
*fewer* channels than this run at strides 1, 2, 4 and 8. It buys parameters where
this task has almost nothing to represent. Rate is the cheaper knob per parameter
precisely because low-resolution parameters cost little memory and do little work;
optimising parameters-per-GB is the wrong objective and was an error in an earlier
recommendation of mine.

## What would make it fail

**Overfitting.** 125.6M parameters against 1,382 frames is ~91,000 per frame, and
the one capacity experiment run here went the other way — halving parameters
improved the in-sample/held-out gap. Rotation augmentation at `p=1.0` multiplies
the effective data, which is the counter-argument, but it is an argument and not
a measurement. Checked directly by scoring in-sample and on the locked benchmark,
not inferred from the loss.

**The crop is the known cost, and it is quantified.** `anchor_part='mouth'`, so
the crop must reach from mouth to tail, and rotation puts that vector at every
angle. Labelled spans reach **163.2 px**. At crop 384 (192 px half-width):

| | tails outside the crop |
|---|---|
| scale 1.0 | **0.00%** |
| scale 1.3 (the augmentation setting, unchanged) | **2.84%** |

So ~2.8% of augmented training instances will have the tail clipped — against 0%
at round 8's crop 512. That is the price of the resolution, it is accepted
deliberately, and it is the first thing to suspect if tail recall drops. Crop 320
was rejected for this reason: it loses **37%** at scale 1.3.

**The benchmark is small.** The locked ALL arm is 120 instances (1 instance =
0.83 points); the CORRECTED arm is 48 (1 instance = 2.08 points). The CORRECTED
arm can only detect large effects, and the rule requires the ALL arm to move too.

## What is held fixed, and why

**The centroid head is round 8's, untouched.** It is a 7.85M-parameter blob
detector for three fish at one scale and it does that job well; every detection
error it makes is a miss, not a placement error, and this experiment does not
address misses. Holding it fixed also means any change is attributable to one
head.

**Trained on round 8's own data split** — `train_labels.pkg.slp` (1,382 frames)
and `val_labels.pkg.slp` (20 frames), `validation_fraction=0.0` — identical to
round 8's config, so the label set is not a variable.

**Intensity augmentation stays at `p=0.0`**, as round 8 ships it. Three things
already differ; adding a fourth would make attribution worse. It is a reasonable
follow-up once this is measured, and it is the change most likely to help
cross-group generalisation.

Everything else is round 8's: `sigma=2.5`, `output_stride=2`, `convs_per_block=2`,
rotation ±180° `p=1.0`, scale 0.7–1.3 `p=1.0`, 200 epochs, batch 16, Adam at
1e-4, online hard-keypoint mining, `reduce_lr_on_plateau`. Seed fixed at 0.

## Measured cost

125,609,538 parameters; peak **7.86 GB** for a real forward+backward at
384×384 batch 16, ≈**8.80 GB** with Adam state. Fits on the 15.9 GB card with
headroom. (Configs reported as fitting above ~14 GB are spilling into Windows
shared memory and are unusably slow — that is not a real fit.)

## Notes

**2026-08-04** — Pre-registration written. Nothing trained.

**2026-08-04, amendment written before the side run — the top-only scope was a
mistake.**

The top run (k11) was designed, trained and scored without a side counterpart, and
its RESULT recommended stopping centered-instance work on the strength of it.
**That recommendation was drawn from one of three cameras and should not have been
stated that broadly.** The study's output is 3D, every downstream identity claim
consumes all three views, and side is plausibly the *harder* view:

| | median span | partial instances | training frames |
|---|---|---|---|
| top | 112.8 px | 5.9% | 1,382 |
| **side** | **94.0 px** | **11.4%** | **2,764** (1,382 moments × 2 cameras) |

The animal projects 17% smaller and nearly twice as many labelled instances are
missing a node. Nothing about "capacity is not the binding constraint on top"
transfers to that.

**Side arm, same hypothesis, same three changes.** Round-8's side
centered-instance head is *architecturally identical* to its top one — crop 512,
`filters=24`, `filters_rate=2.0`, `max_stride=64`, 70,755,866 params — so the
change is the same one:

```text
crop_size   512 -> 384        max_stride  64 -> 32        filters  24 -> 64
filters_rate held at 2.0  ->  125,609,538 params
```

**The crop reduction is better justified on side than on top.** Required
half-width is the mouth-to-tail span under ±180° rotation; side spans reach
153.3 px against a 192 px half-width at crop 384:

| crop | side tails lost @1.0 | **@1.3** | (top, for comparison) |
|---|---|---|---|
| 320 | 0.00% | 13.35% | 37.02% |
| **384** | **0.00%** | **0.04%** | 2.84% |

0.04% against top's 2.84%. The clipping risk that the top run had to accept
essentially does not exist here.

Trained on `labels.round8.side.train` (2,764) / `labels.round8.side.val` (40),
round-8's own split. Side centroid **not** retrained. Intensity augmentation held
at `p=0.0`, as round 8 ships it. Both side cameras are in one package and are
trained together, exactly as round 8 did it — but **side1 and side2 must be
scored separately**, since they are different viewpoints with different occlusion
geometry and pooling would hide a one-camera failure.

**Decision rule for the side arm** — same shape as the front matter, with round-8
*side* as the baseline, measured on the locked HITL-v2 benchmark's 40 side1 + 40
side2 frames (which are frame-disjoint from all training splits, and must be
contamination-checked against round-8 side output before any score is read, as the
top arm was):

1. both-node recall @20 px >= round-8 side, on the ALL arm and the CORRECTED arm,
   **and reported per camera**;
2. false positives <= round-8 side;
3. shared-tail pairs <= round-8 side.

ADOPT only if all three hold on **both** cameras. If the two cameras disagree,
retain round 8 and report per camera — do not average them.

Expected cost: 2,764 frames is double the top run, so ~224 s/epoch and
**~12 h for 200 epochs**.
