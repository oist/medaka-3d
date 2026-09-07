# RESULT — centered-instance capacity at high resolution (crop 384 / ms 32 / f64)

**Verdict: parked.** **New `expires:` — 2026-12-01.**

The pre-registered rule fails on one criterion by **one instance**. Every other
difference between this model and round 8 is also one instance. The honest
finding is not about the model — **the benchmark cannot resolve models this
close**, and that is now the binding constraint on "beat round 8".

Reproduce: `tools/k11_retrain_ci_highres.py`, scored by
`tools/k08_locked_benchmark.py`.

## The rule, applied as frozen

Baseline is round 8 on the locked HITL-v2 benchmark, round-8 centroid in both arms.

| criterion | round 8 | k11 | |
|---|---|---|---|
| 1a. both-node recall, ALL arm ≥ 93.33% | 93.33% | **94.17%** | **PASS** |
| 1b. both-node recall, CORRECTED arm ≥ 91.67% | 91.67% | 91.67% | **PASS** (tie) |
| 2. false positives ≤ 0 | 0 | **1** | **FAIL** |
| 3. shared-tail pairs ≤ 1 | 1 | 1 | **PASS** |

Not ADOPT. Not REJECT (recall is above, not 2 points below). **PARK.**

Coverage moved 97.50% → 96.67%, one instance. The pre-registration said coverage
should be identical because the centroid is unchanged and that any movement is "a
bug to investigate". **That reasoning was wrong and is corrected here**: coverage
is measured by matching at ≤ 20 px mean error, so a worse *localisation* can drop
an instance below the match threshold without the centroid changing at all.
Coverage is not purely centroid-driven.

## The measurement that matters more than the verdict

The locked benchmark is 120 instances of which **36 (30%) are verbatim round-8
output**, so round 8's median error on it is 0.02 px for tail and 0.00 px for
mouth — not an error rate, an identity. Medians are unusable.

Excluding those 36 at the **instance** level (keeping all 40 frames, n = 82 —
higher power than the frame-level CORRECTED arm's 48):

| model | n | both-node @20 px | mouth @20 px | tail @20 px |
|---|---|---|---|---|
| round 7 | 82 | 91.46% | 93.90% | 91.46% |
| round 8 (production) | 82 | 92.68% | 95.12% | 92.68% |
| **k11** | 82 | **93.90%** | **96.34%** | **93.90%** |

k11 > round 8 > round 7, **each by exactly one instance**. One instance is 1.22
points here. This is not a demonstrated improvement; it is a rank order that a
single annotation could reverse.

**This arm was not pre-specified** and is reported as a post-hoc refinement of the
frame-level CORRECTED arm, not as the criterion.

Localisation, upper quantiles (robust to the verbatim contamination that destroys
the medians): **k11 tail p95 = 8.63 px against round 8's 9.33 px.**

## Two predictions in the pre-registration, both wrong

**The crop-384 clipping worry is refuted.** The lab flagged that 2.84% of tails
fall outside a 384 crop at `scale_max=1.3` and named it "the first thing to
suspect if tail recall drops". Tail error by ground-truth span:

| span bin | n | round 8 tail p95 | k11 tail p95 |
|---|---|---|---|
| 0–100 px | 37 | 10.74 | **9.46** |
| 100–130 px | 52 | **6.20** | 6.77 |
| 130–163 px | 25 | 5.67 | **4.54** |

k11 is *best* on the longest-span bin — exactly where clipping would bite. The
crop reduction did not cost tail accuracy.

**Overfitting did not appear either.** 125.6M parameters on 1,382 frames was the
named risk; k11 has the best uncontaminated both-node recall of the three models.

## Where k11 clearly loses, and why it is not decisive

On **round-8's own validation split** (20 frames, 60 instances) k11 is worse on
every metric: mAP 0.4287 vs 0.5146, mOKS 0.7741 vs 0.8061, PCK@5 0.8983 vs
0.9492, distance p95 5.99 vs 4.88 px.

Two reasons not to read that as the answer, both structural: that split is where
**round 8 selected its own checkpoint**, and the archive's Phase-1 audit records
that its annotations **were initialised from round-7 predictions**, with round 8
**warm-started from round 7**. It is the arm most favourable to round 8 that
exists. It is reported because it is real and it points the other way from the
locked benchmark.

## The actual blocker

Detection quality on the top view is now at **93.9% both-node recall on
uncontaminated instances with 0–1 false positives per 40 frames**, and the three
candidate models sit within one instance of each other. To demonstrate that any
model beats round 8 would need an evaluation set able to resolve ~1% differences —
several hundred uncontaminated instances, annotated without seeding from any
model. The locked benchmark has 82.

**Recommendation: stop tuning the centered-instance head.** It is not where the
remaining error is. Round 8's residual failure is **misses** — 6.14% of frames
with fewer than 3 detections on K_1_2 — and those are centroid-stage and cannot be
addressed here. If detection work continues it should target that; otherwise the
sensible move is on to tracking, and back to detection only if tracking
diagnostics point at it.

## Artefacts

- `work_dir/k11_retrain/k11.centered_instance.crop384_ms32_f64_r2/`
  (local scratch retained at `C:/sleap_cache/k11_runs/`), 125,609,538 params,
  200 epochs at 112 s/epoch, `metrics.val.0.npz` present.
- Capacity probe: `work_dir/k10_capacity_probe.centered_instance.json`.
