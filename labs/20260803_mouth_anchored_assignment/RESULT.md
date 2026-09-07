# RESULT — mouth-anchored one-to-one tail assignment

**Verdict: parked.**
**New `expires:` — 2026-12-01** (see the blocker below; it is Gate A, not a date).

Run 1, as pre-registered: **rejected**, and the rule was not relaxed.
Run 2, the amended origin-preferred cost: **parked** — 2 of 4 criteria met, and
the 2 it misses are missed by 0.59 and 2.95 points.

Full write-up: [`docs/decisions/2026-08-03-k-detection-quality.md`](../../docs/decisions/2026-08-03-k-detection-quality.md).

## What was established

On the 113 Group-K top frames held out from round 7, against unfiltered round-7
top-down:

| | baseline | b04 (adopted at Gate B0) | **assignment** |
|---|---|---|---|
| shared **tail** pairs @5 px | 28 | 19 | **0** |
| false positives | 52 | 43 | **41** |
| animal coverage @20 px | 88.79% | 88.20% | 88.20% |
| both-node recall @20 px | 74.93% | 74.34% | 71.98% |
| mouth-only | 40 | 40 | 59 |
| tail-only | 12 | 12 | 0 |

- **The shared tail is gone structurally**, not filtered — a one-to-one
  constraint makes it unrepresentable. `tests/test_assignment_lab.py` holds that
  property under randomised frames.
- **Fewer false positives than the filter Gate B0 adopted** (41 vs 43), while
  *keeping* every anchor instead of deleting the loser.
- **It reaches its own ceiling.** k01 predicted 9 of 19 shared-tail frames could
  recover the animal and 10 could not, because in those 10 no true tail was ever
  decoded. Reassignment recovers **exactly 9**, at 9 correct against 4 wrong.
- Corpus-wide, on all 65 experiments in 12 groups by the cardinality anchor:
  `>3` counts fall everywhere, `<3` rises by ≤ 0.2 points, exact-3 is flat to
  marginally better. Group K, three arms: b04 and assignment reach the same `>3`
  rate (3.58% vs 3.60%), but b04 costs **+1.89 points of new misses and −1.27
  points of exact-3** where assignment costs +0.22 and +0.38.

## Why it is parked and not promoted

Both-node recall fell 2.95 points against a pre-registered tolerance of 2.0. That
number is real and is not being argued away.

But the loss is **not error, it is abstention**: mouth-only instances rise 40 →
59, and both-node recall scores an honest mouth-only instance and a silently
swapped tail as equally wrong. The entire argument for this method is that those
two outcomes are *not* equally wrong downstream — a mouth-only instance keeps
position and identity continuity and withholds orientation, while a swapped tail
corrupts an identity chain with no symptom.

**Blocker: there is no metric in this repository that can price that trade.**
Pricing it needs a score over *tracks*, not over frames — which is exactly
[Gate A](../../docs/ROADMAP.md). Promoting on a 2.95-point loss against the only
criterion currently measurable would be choosing the metric to fit the method.

**Unpark when Gate A exists.** The decision rule for that run should be written
before it and should key on identity-swap rate, not on both-node recall.

## What the failure of run 1 established, which is the more useful result

The specification's cost — *deviation of the mouth–tail distance from the
labelled span distribution* — **does not work, and the reason generalises**: a
population statistic is a *plausibility* signal, not a *discriminative* one.
**45.3%** of foreign mouth–tail pairings fall inside the acceptance band, and
because the true span varies with foreshortening, a foreign tail nearer the
median routinely outranks the animal's own. Measured: **29 tails moved
correct→wrong against 2 the other way**, and **19 of 22 rejected tails had been
correct**.

The fix was in the audit, not in tuning: **the pairing the model itself decoded
is the most informative feature available** — correct in 41 of the 47 cases a
geometric cost wanted to overrule. The centered-instance head decodes a tail
inside a crop anchored on one mouth; that is a joint prediction and run 1 threw
it away.

Also recorded because it is the project's recurring failure: I applied **68.7 px**
— k01's minimum ground-truth *centroid* distance to the frame edge — as a gate on
*mouths*, whose true minimum is **28.8 px**. Worth 3.5 of the 10.3 lost coverage
points. Third instance of a number carried to a quantity it was not measured on.

## What it does not establish

- **`n_groups = 1`.** Every labelled frame is Group K.
- **~48% of the ground truth is verbatim round-7 top-down output**, so every
  number here is a bound biased *toward* the baseline being compared against.
- The 113 frames are the subset a human chose to correct — worst-case
  proportions, not corpus-wide ones.
- Cardinality outside K bounds *counting* only. Localisation outside K is
  unmeasured and unmeasurable without labels from another group.

## Code

`assignment.py` stays in the lab until Gate A settles it. Two entry points,
deliberately separate because they are different algorithms and this repository
does not let a flag change which one runs:

- `assign_frame` — the rejected run-1 specification, kept so the rejection stays
  reproducible;
- `assign_frame_origin` — the amended specification, the one that is parked.
