# RESULT — centered-instance capacity retrain

**Verdict: parked.**
**New `expires:` — 2026-12-01.**

The pre-registered rule **passed on all three criteria**. It is not being
promoted, because between writing the rule and reading it a frame-disjoint test
set was found on which the improvement does not reproduce. The blocker is named
below and it is not a date.

Full write-up:
[`docs/decisions/2026-08-03-k-detection-quality.md`](../../docs/decisions/2026-08-03-k-detection-quality.md).

## The rule, applied as frozen

Baseline is round 7 on the 113 frames held out from it — the comparison the lab
was designed around, and a fair one: k04 and round 7 trained on the *same*
1,295-frame package, so only the architecture differs.

| criterion | result | |
|---|---|---|
| 1. mouth-only under K03c assignment < 59 | 59 → **45** | **PASS** |
| 2. animal coverage ≥ 88.79% | 88.79% → 88.79% | **PASS** |
| 3. both-node recall ≥ 74.93% | 74.93% → **78.17%** | **PASS** |

The hypothesis was specific and it held: the head now represents tails it
previously did not at any threshold, and the quantity chosen in advance to detect
that — mouth-only instances — fell by 14. Raw shared tails fell **28 → 11**.

**Two predictions in the lab were wrong, and in the same direction.**

- *"Overfitting is the likeliest outcome."* It **narrowed**: in-sample-to-held-out
  FP ratio 14.1× → **11.4×**, recall gap +13.78 → **+9.76** points.
- *"filters 24 → 32 ... spend the freed capacity on width."* The network came out
  **smaller** — 31,405,858 parameters against 70,755,866. Removing the deepest
  encoder block costs more than widening the rest adds. This is a *reallocation*,
  and the lab should have said so. That it generalises better on **fewer**
  parameters is the more interesting result.

`val/loss` ended at 0.000268 against round-7's 0.00029, and was flat for the last
~80 epochs while the task metrics were still moving. As pre-registered, it decided
nothing. That was the right call.

## Why parked and not promoted

**It does not reproduce on a fair sample.** After the rule was written, two things
came to light:

1. **Round 8, not round 7, is production** — the manifest's claim was a hardcoded
   constant, and the archive's own 2026-07-27 report says so explicitly.
2. **A locked, human-annotated, frame-disjoint benchmark already exists** — 40
   frames across all three K days, zero overlap with round-7 training, round-8
   train or round-8 val.

On that benchmark k04 is **not** better than round 7 (coverage 95.00% vs 95.83%,
both-node recall 90.83% vs 91.67%, FP 6 vs 5), and on the arm whose annotation is
independent of round 8 all three models sit at identical coverage with recall
differing by a single instance.

The 113 frames are the subset a human chose to correct — enriched for difficulty
by construction. A change that helps on hard frames scores large there and
negligibly on a fair sample. **The rule passed; its evaluation set turns out to be
unrepresentative, which the rule could not have known.** Reporting that is not
revoking it.

**Blocker: this needs re-running against the production model on the locked
benchmark, with the rule rewritten to key on that benchmark rather than the 113.**
Unpark by doing that. The likely honest outcome is that the architecture change is
worth keeping for crossing-heavy frames and is not a general improvement.

## What it established that is worth keeping

- **The representational gap is real and fixable.** Mouth-only 59 → 45 and shared
  tails 28 → 11 are the mechanism the lab predicted, measured on the metric chosen
  before the run.
- **Confmap MSE is useless for this decision, demonstrated rather than asserted.**
  Better final loss, flat for 80 epochs, while the metrics that matter moved.
- **Capacity was not the binding constraint.** A model with 44% of round-7's
  parameters generalises better. Depth at `max_stride=64` was buying nothing over
  a ~320 px context.
- **No model separation is established, which is itself the finding.** On the
  only leakage-free arm available — 16 locked-benchmark frames containing no
  verbatim round-8 instance — round 7, round 8 and k04 all score **97.92%
  coverage** and differ by one instance on both-node recall. Round 8's apparent
  lead elsewhere comes from having trained on 93 of the 113 frames and from being
  30% of the locked benchmark's ground truth. **An earlier version of this file
  said "labels beat architecture"; that is withdrawn as unsupported.** 48
  instances cannot rank three models.

## What it does not establish

- **`n_groups = 1`.** Group K throughout.
- **The intensity augmentation is untested.** It was included to help cross-group
  generalisation and every evaluation set here is Group K, so nothing measured it.
- **k04 versus round 8 is not attributable** — different labels *and* different
  architecture.
- The locked benchmark is frame-disjoint but **30% of its instances are verbatim
  round-8 output**, so it cannot rank round 8 either.

## Artefacts

- Checkpoint: `work_dir/k04_retrain/k04.centered_instance.ms32_f32_intensity/`
  (copied to `/bucket`; local scratch retained at `C:/sleap_cache/k04_runs/`).
- Requested config: `…/k04.centered_instance.ms32_f32_intensity.requested_config.json`.
- Reproduce: `tools/k04_retrain_centered_instance.py`, score with
  `tools/k04b_score_retrain.py`, compare with `tools/k07_production_baseline.py`
  and `tools/k08_locked_benchmark.py`.
