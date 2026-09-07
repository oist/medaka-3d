# Why the two lambda selectors disagree — the diagnostic, before it is run

**Date:** 2026-08-26. **Written before the run.** Only the "Result" section is
appended afterwards.
**Status:** diagnostic only. **No tool is modified.** `c25` is read and imported,
never edited; if a fix turns out to be warranted it gets its own document and its
own number.
**Follows:** step B of [`NEXT_SESSION_uncapped_followup.md`](../NEXT_SESSION_uncapped_followup.md),
and [2026-08-26 stage A vs correspondence](2026-08-26-stage-a-vs-correspondence.md),
which made B a prerequisite rather than a cleanup item.
**Tool:** new `c26`.

## The disagreement

`c19.select_kappa` returned `kappa = 0.2371` on `c19`'s cost scale, and `c20`
imports that same function and returned `0.2371` again on a cost scale 3.6x
apart. Two independent confirmations of the quantity `kappa` exists to make
transferable. `c25.sweep_kappa`, a separate reimplementation of the same rule on
`c25`'s state space, returns 0.75–1.33 and prefers values in the degrading half
of its own sweep. Two bugs in it have been found and fixed and it is still not
trusted, so **every lambda in the 2026-08-25 record is post-hoc.**

This is a debugging task with an oracle. A selector that disagrees with two
independent runs of the rule it claims to implement is wrong against a known
answer, and the question is only *which term* produces the disagreement.

## The bench, and why there is only one

**`c16`'s labels.** There, and only there, do the two selectors see the same
geometry:

* `c25 --check-reduction` proved its per-state cost equals `c20`'s `label_costs`
  to **1.846e-06 mm** against a derived tolerance of 6.412e-05 mm, with **0**
  winner disagreements, on the 17,830 frames holding three labels in all three
  views and supporting all six terms.
* The Step A run on 2026-08-26 measured `c25`'s state space on `c16`'s labels at
  **mean 214, max 216** — its floor, which is `c19`'s 36 correspondences times
  the 6 slot renamings `c19` does not carry because it pins slot naming to
  `top_cam`.

On `c24`'s labels the two do not see the same geometry and no comparison between
them is possible. That is why the disagreement was never localised.

## The window, disclosed

Frames **100,000–120,000** — the window `c24` used for its own measurements.
Both selectors are run on that same window, and **`c19`'s selector is re-run on
the window rather than read out of the whole-recording store**, so the comparison
is like-for-like. The stored whole-recording oracle

    kappa* 0.2371   (fit mouth 0.3162, fit pectoral 0.1778)
    margins  mouth 3.228   pectoral 3.197   total 3.233 mm

is quoted only as the target the window has to reproduce. **If the window does
not reproduce it, the window is not a bench and nothing else in this document is
readable.** That check comes first and can fail.

## The prediction, with its direction, before the numbers exist

`c19` and `c20` charge `lambda` once when **the state** changes. `c25` charges
`lambda` per **slot** that moves — 1, 2 or 3 per transition. If nothing else
differed, `c25`'s `kappa*` would have to be **smaller** than `c19`'s, by roughly
the mean number of slots that move on a genuine correspondence change, which
`c25`'s own run measured at 2 to 3. That predicts `kappa` of order **0.08–0.12**.

**The observed 0.75–1.33 is off in the opposite direction**, by about an order of
magnitude. So the penalty unit cannot be the explanation on its own, and this
run is not expected to end by confirming it.

## What will be read, and how — decided now

1. **The three margins** (mouth, pectoral, total), against 3.228 / 3.197 /
   3.233 mm. `kappa = lambda / m`, so a margin that is *k* times too small
   inflates `kappa` by *k*.
2. **The two held-out curves** on the same 25-point grid, against `c19`'s: where
   the minimum sits, and whether it is interior.
3. **Slot changes along each curve**, which measures the penalty-unit conversion
   rather than assuming it.

**Attribution, fixed in advance:**

* margins agree to float32 and the curve minima disagree → the DP or the penalty
  unit, and (3) says which;
* margins disagree → the fault is in `kappa`'s denominator, and the curves must
  be re-compared in **lambda (mm)** rather than in `kappa` before anything is
  concluded from them;
* both agree on this window → the disagreement is a property of `c24`'s labels,
  not of the selector, and B changes shape entirely.

**No threshold is chosen.** The oracle is a value returned twice independently;
agreement means landing on the same grid point, because the grid is the
selector's own resolution — `KAPPA = geomspace(1e-4, 1e2, 25)`, a factor of
10^0.25 = 1.78 per step.

**No fix is applied in this step**, and no lambda is selected from it. This
document establishes where the fault is, not what to do about it.

## Result

**The pre-registered bench check fired and stopped the run, and then the oracle
itself failed an audit. There is no known answer for `c25` to be wrong against.**

### The window is not a bench — as the document said it might not be

`c19`'s selector, re-run on frames 100,000–120,000 of `c16`'s labels:

| direction | on the window | stored, whole recording |
|---|---|---|
| fit mouth | kappa* **0.1778** (index 13) | 0.3162 (index 14) |
| fit pectoral | kappa* 0.1778 (index 13) | 0.1778 (index 13) |
| combined | **kappa 0.1778**, lambda 0.581 mm | kappa 0.2371, lambda 0.767 mm |

One grid point out on one direction, so `c26` refused to print the comparison.
That is the tool working. It is also the first sign of what follows.

### The oracle audit — kappa = 0.2371 was never returned twice

Reading the two stored `lambda_choice` records:

| run | fit mouth | fit pectoral | index sum | reported kappa |
|---|---|---|---|---|
| `c19` | index **12**, kappa 0.1000 | index **15**, kappa 0.5623 | **27** | 0.23714 |
| `c20` | index **14**, kappa 0.3162 | index **13**, kappa 0.1778 | **27** | 0.23714 |

**Per direction the two runs did not agree at all.** Mouth moved 12 → 14 and
pectoral moved 15 → 13 — a factor of 3.16 each way, in opposite directions.

`KAPPA` is `geomspace(1e-4, 1e2, 25)`, so `KAPPA[i] = 10^(-4 + i/4)`, and the
rule reports the geometric mean of the two directions:

    sqrt(KAPPA[i] * KAPPA[j]) = 10^(-4 + (i + j)/8)

**which depends only on i + j.** Both pairs sum to 27, so both report
10^(-0.625) = 0.23714 **by construction, to every digit**. Any pair summing to
27 — (12,15), (13,14), (11,16) — returns the same number.

The identity of the two reported values is an arithmetic consequence of two
coincidentally equal index sums. It is **not** an estimator converging on a
transferable quantity, which is what the record has been reading it as since
2026-08-18.

### The curves cannot separate the choices they are said to have made

Each run scored at the *other* run's chosen grid point, in its own held-out mean:

| run | direction | at its own kappa* | at the other's | cost of moving |
|---|---|---|---|---|
| `c19` | mouth | 1.792293 mm | 1.798594 mm | **+0.35%** |
| `c20` | mouth | 0.958600 mm | 0.962501 mm | **+0.41%** |
| `c19` | pectoral | 1.321528 mm | 1.322768 mm | **+0.09%** |
| `c20` | pectoral | 0.720630 mm | 0.721550 mm | **+0.13%** |

A factor of 3.16 in `lambda` costs between 0.09% and 0.41% of the held-out
score. And `c20`'s pectoral direction is a **numerical tie**:

    index 12  kappa 0.1000   0.72064461 mm
    index 13  kappa 0.1778   0.72062967 mm     <- chosen
    difference 1.49e-05 mm, 0.002%

Had that tie fallen the other way the index sum would be 26 and the reported
kappa would be **0.1778** — which is exactly what `c26` just returned on the
window. The window did not disagree with the whole recording so much as land on
the other side of the same tie.

`c20`'s mouth curve is worse than flat: 0.95860 at index 14, then 0.97971,
1.18468, 1.37853, 1.38511 at 15–18, then back to 0.96675 at 19. That is not a
bowl with a minimum in it.

### What this establishes

1. **`kappa = 0.2371` is not an oracle** and step B cannot be run as scoped.
   `c25.sweep_kappa` returning 0.75–1.33 is no longer evidence of a bug in
   `c25`, and no longer evidence of correctness either. The comparison the
   handoff proposed has no reference to be measured against.
2. **The pre-registered rule does not identify lambda on this data.** Held-out
   node cross-validation resolves a factor of 3.16 in `lambda` at 0.09–0.41% of
   its own score, one direction of one run being a tie at 0.002%. The argmin is
   reading noise.
3. **The 2026-08-25 observation that swaps are 0 across a sevenfold range of
   lambda is the same fact seen from the other side**, not independent
   reassurance about it. The criterion is nearly blind to `lambda` over the
   range where the identity outcome does move — `c25`'s own table has
   `preserved` going 272 → 258 and `swap` 0 → 3 as `lambda` goes 0 → 8 mm.

### What it does not establish

`n_groups = 1`, `n_experiments = 1`. This is an audit of two stored runs on
K_1_2 with `c16`'s labels, plus one 20,000-frame re-run. It says the rule does
not identify `lambda` **here**. It does not say the rule is wrong in principle,
and it says nothing about whether `c25`'s port is faithful — that question is
now unanswered rather than answered badly.

### Corrections to the record

* [2026-08-25 uncapped tracklets](2026-08-25-uncapped-tracklets.md), "What is
  not done": *"kappa = 0.2371 was returned by both `c19` and `c20` on cost
  scales 3.6x apart, which is what kappa was introduced to do."* **Retracted.**
  The two runs chose grid points a factor of 3.16 apart in each direction; only
  the geometric mean coincided, and it did so because the index sums did.
* [`NEXT_SESSION_uncapped_followup.md`](../NEXT_SESSION_uncapped_followup.md),
  step B: *"This is a debugging task with an oracle."* **Retracted.** There is
  no oracle.
* [TRACKING_PIPELINE.md](../TRACKING_PIPELINE.md), stage 2: *"`c19` and `c20`
  chose kappa* = 0.2371 independently, from the same two grid points."* The
  second clause is false — they chose **different** grid points, and `c20`
  imports `select_kappa` from `c19`, so the two were never independent
  implementations in the first place.

### Where this leaves C

The falsifiability problem step B existed to solve is real and unchanged: on
K_1_1 and K_1_3 there is no `c09`, so a post-hoc `lambda` cannot be checked.
But it cannot be solved by fixing a port.

The remaining honest option is to **stop reporting a point `lambda` at all** —
run K_1_1 and K_1_3 across the `lambda` range and require every transfer claim
(state count, residual tail, `resolved`, per-camera label counts) to hold across
it. A conclusion that survives the whole range needs no selector; one that does
not survive it was never safe to draw from a selected point either. That is
falsifiable, it costs a sweep rather than a debugging session, and it does not
rest on a number this document has just shown is unidentified.

That is a change of plan, not a step in one, and it is recorded here for a
decision rather than acted on.
