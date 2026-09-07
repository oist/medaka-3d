---
hypothesis: >
  The centered-instance head fails to represent ~10 of 19 crossing tails at any
  threshold because its capacity is spent on receptive field rather than width;
  halving max_stride and widening the filters will make those tails decodable,
  which will show up as a fall in the mouth-only count under K03c assignment.
stage: detector
expires: 2026-12-01
decision_rule: >
  ADOPT the retrained centered-instance head if, on the 113 Group-K top frames
  held out from round 7, with the round-7 centroid head unchanged, ALL THREE
  hold: (1) mouth-only instances under K03c primary assignment fall below the
  round-7 value of 59; (2) animal coverage @20 px >= 88.79%; (3) both-node
  recall @20 px >= 74.93%. REJECT if both-node recall falls below 72.93%
  (baseline - 2.0) or coverage below 86.79%. Otherwise PARK. `val/loss` is
  explicitly NOT a criterion and no decision may cite it.
---

# Centered-instance capacity retrain

Pre-registered 2026-08-03, before any training run.

## Why this might work

The argument is not "bigger is better" — it is a specific measurement that the
loss function cannot see.

Of the 19 held-out frames where two mouths claim one tail, **10 have a true tail
that was never decoded at any threshold**. Dropping the decoder peak threshold
20× (0.2 → 0.01) recovers **1 of those 10**
([k-misdetection-taxonomy](../../docs/decisions/2026-08-03-k-misdetection-taxonomy.md)).
So the tail is not suppressed, it is *absent*: the model does not put a peak
there. That is a representational gap, and it is the one thing K03c assignment
provably cannot fix — assignment can only ration candidates that exist, which is
exactly why its mouth-only count rose from 40 to 59.

Meanwhile the confmap loss sat flat at 0.00023 train / 0.00029 val from epoch 95
to 200 while the model was failing this way. MSE over a 256×256 grid is dominated
by the empty background; a tail peak in the wrong place moves it by almost
nothing. **This lab is judged on the task metric and the loss curve is evidence
about training, not about quality.**

Three changes, each with a reason:

| change | from | to | why |
|---|---|---|---|
| `max_stride` | 64 | 32 | 6 down-blocks gives a receptive field far past the ~320 px of context a 512 px crop of a 113 px animal contains. The depth is not buying anything. |
| `filters` | 24 | 32 | Spend the freed capacity on width. 32 rather than 48 because the training set is 1,295 frames from **one group** and the overfitting risk is real. |
| intensity augmentation | all `p=0.0` | `p=0.5` | Noise, contrast and brightness are configured and disabled. Nearly free, and it is the one change that targets the cross-group generalisation a rollout needs. |

Everything else is held at the round-7 values — `crop_size=512`,
`anchor_part='mouth'`, `filters_rate=2.0`, `convs_per_block=2`,
`output_stride=2`, `sigma=2.5`, 200 epochs, Adam at 1e-4, online hard-keypoint
mining. Crop size in particular is **not** touched: shrinking it was measured and
refuted (crop 320 fixes 0 of 18 shared-node failures).

## Which labels, and why not the obvious ones

Trained on **`retrain_20260704/top/labels.pkg.slp`** — the round-7 package, 1,295
frames — not on round 8.

The 113 evaluation frames are *by definition* round 8 minus round 7. Training on
round 8 would put every one of them in the training set and destroy the only
honest evaluation this project has. Training on round 7 also isolates the
variable: same data, same recipe, **only the architecture changes**, so a
difference is attributable to the architecture. The cost is that ~700 extra
labelled frames go unused, and that is the right trade.

The **centroid head is not retrained.** The failure is a tail decoded inside a
crop, and the mouth anchors are mostly right (66 of 113 frames decode exactly 3).
Changing one head at a time is the only way a result stays attributable.

## What would make it fail

**Overfitting is the likeliest outcome.** 1,295 frames of one trio against a
wider network, with a loss that will not reveal it — the val split is drawn from
the same frames and inherits the same three fish, the same tank and the same
lighting. A model can memorise this set and still not put a peak on the tail of a
fish it half-occludes.

Second: the premise may simply be wrong. If the missing tails are absent because
the animal is genuinely occluded in the top view rather than because the network
lacks capacity, no architecture recovers them and mouth-only will not move. That
outcome is worth knowing and is why the criterion is the mouth-only count and not
a loss.

Third: **`n_groups = 1` still.** A better number here is a better number on group
K. It says nothing about the other 11 groups, and the intensity augmentation is a
bet on generalisation that this evaluation set structurally cannot test.

## Notes

**2026-08-03** — Pre-registration written. Nothing trained.

**2026-08-03, amendment written before scoring — the baseline in the rule is the
wrong model, and the rule is being applied anyway.**

The decision rule above keys on round-7 numbers (mouth-only 59, coverage 88.79%,
recall 74.93%) because `MANIFEST.json` said round 7 was production for top. The
experimenter has corrected that and the correction is verified: **round 8 is
production for both views**, and the manifest figure was a hardcoded constant
echoed back, not a measurement. See
[`docs/decisions/2026-08-03-k-detection-quality.md`](../../docs/decisions/2026-08-03-k-detection-quality.md).

Two consequences for this lab, both stated before the numbers are read:

1. **The frozen rule is applied unchanged.** It compares like with like — k04 and
   round 7 were trained on the *same* 1,295-frame round-7 package, so the
   difference between them is the architecture and nothing else. That is the
   question this lab asked and the rule still answers it. Rewriting the baseline
   after seeing the correction would turn a pre-registration into a preference.
2. **A production comparison is reported alongside it, and is explicitly not the
   rule.** k04 versus round 8 confounds architecture with labels — round 8 has
   1,382 "strictmouth" frames against k04's 1,295 — so it cannot attribute a
   difference to either. It is reported because "does the retrain beat what is
   actually deployed" is the question that matters operationally, even though it
   is not the question that was pre-registered. `tools/k07_production_baseline.py`.

Also recorded: the honest evaluation set for this lab is smaller than assumed.
**93 of the 113 held-out frames are in round-8's training data**; only **20** are
held out from both checkpoints. Round-7 and k04 are honest on all 113 (neither
trained on any of them); round 8 is honest only on the 20.
