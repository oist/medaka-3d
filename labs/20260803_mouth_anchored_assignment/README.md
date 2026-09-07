---
hypothesis: >
  Framing detection as a one-to-one assignment of decoded tails to decoded mouths
  eliminates shared-tail instances structurally, without discarding the evidence
  that instance-level suppression discards, because a mouth that loses the
  contest keeps its anchor and abstains on orientation instead of disappearing.
stage: detector
expires: 2026-12-01
decision_rule: >
  PROMOTE into src/medaka3d/ if, on the 113 Group-K top frames held out from
  round 7, against unfiltered round-7 top-down output, ALL FOUR hold:
  (1) shared-node output pairs at 5 px = 0;
  (2) animal coverage @20 px is >= baseline;
  (3) both-node recall @20 px is >= baseline - 2.0 points;
  (4) false-positive instances (unmatched output instances) <= baseline.
  REJECT if (1) fails, or if (2) falls by more than 2.0 points. Otherwise PARK
  with the blocker named. Metric definitions are frozen in the section
  "Frozen evaluation criterion" below; they are the k01/b03 definitions unchanged.
---

# Mouth-anchored one-to-one tail assignment

Pre-registered 2026-08-03, **before any number in this lab was produced**.

## Why this might work

[`docs/decisions/2026-08-03-k-misdetection-taxonomy.md`](../../docs/decisions/2026-08-03-k-misdetection-taxonomy.md)
establishes four things that jointly determine the design:

1. The largest false-positive category is **shared_node, 21/52 = 40.4%** — the
   reported "one tail joined to two mouths".
2. **Suppression destroys evidence.** Of the 47 held-out frames containing a
   false positive, **15 (32%)** have only 2 of the 3 animals covered. Deleting
   the shared-node loser leaves two animals, not three. So B0.4's rule A, which
   Gate B0 adopted, is the wrong operation *at crossings* even though it raised
   precision.
3. **Confidence cannot arbitrate.** The false positive has the lower instance
   score in only 16/21 pairs, and it fails exactly on the hard ones (0.831 vs
   0.825 on frame 274709). The algorithm below therefore uses **no scores at
   all** — only geometry — which also makes it apply unchanged to the retained
   round-7 predictions, where re-running inference is not an option.
4. Both heads use `anchor_part='mouth'`, so the mouth is the structurally
   privileged node. The mouths are mostly right: **66 of 113** frames decode
   exactly 3 mouths, rising to **70** after the edge rule. It is the tails that
   are mis-assigned.

A one-to-one constraint makes a shared tail *structurally impossible* rather
than *detected and removed*. That is the entire claim.

## What would make it fail

The most likely failure is the **phantom tail**. Frame 274709 decodes 3 distinct
tails for 2 real ones. With three mouths and three tails all at plausible
separations, a global one-to-one match will consume all three and hand the
tail-less mouth a fabricated body axis — trading a shared tail for a wrong tail,
which is no better and is harder to notice. This is why the rejection ceiling is
mandatory and why "no tail" must be a first-class outcome, not a fallback.

The second failure is that it **cannot invent evidence**. In 10 of 19 shared-tail
frames a true tail was never decoded at any threshold (a 20× decoder sweep
recovers 1 of 10). Assignment converts those to mouth-only. The measured ceiling
for this method on that set is therefore **9 recovered / 10 abstained**, and
anything claiming more is a bug. Task 4's retrain, not this lab, is the lever on
the other 10.

Third: **`n_groups = 1`.** Every labelled frame is Group K. Nothing here is
evidence about the other 11 groups; that is what the cardinality check on the
retained corpus predictions is for, and it can only detect count errors, never
localisation errors.

## The method

1. **Anchors** = every non-NaN decoded mouth. Drop any mouth whose distance to
   the nearest frame edge is `< 68.7 px` — the minimum ever observed for a
   labelled fish (k01, n=339 GT instances). Deduplicate at 8 px: two mouths
   within 8 px cannot be two animals when the median body span is 112.6 px, and
   if they were, nothing downstream could tell them apart.
2. **Tail candidates** = every non-NaN decoded tail, deduplicated at 8 px. The
   decoded pool only — a confmap-peak pool was measured and refuted (recovers
   1 of 10 missing tails at a 20× lower threshold).
3. **Cost** `= -log q(d)`, where `d` is the mouth–tail distance and `q(d)` is the
   two-sided tail probability of `d` under the **empirical distribution of
   human-labelled spans**. `q = 1` at the median, `q -> 0` in the tails. No
   parametric assumption, no mm conversion (B0.4's reasoning: the mm/px figure is
   an operational proxy and the `tail` node sits at the tail base).
4. **Rejection ceiling** `C = -log(0.01)`, implemented as `n_anchors` dummy
   columns in the assignment matrix so the optimum is over *assign-or-abstain*,
   not over assignment alone. `q < 0.01` is exactly the span interval Gate B0
   already adopted — **[35.1, 152.4] px** — so the ceiling introduces **no new
   free parameter**. A ceiling of `q < 0.10` is carried as a pre-specified
   sensitivity, not as a tuning knob.
5. **Assignment** = `scipy.optimize.linear_sum_assignment` over the padded
   matrix. Global, one-to-one, optimal.
6. **Output** = one instance per surviving anchor: mouth plus assigned tail, or
   an explicit **mouth-only** instance. Tails that no anchor claims are dropped.
   A tail-only prediction therefore ceases to exist as an instance and becomes a
   candidate instead — a deliberate change that removes the tail-only half of
   category E.

### Where the span model comes from, and why not the obvious place

The span distribution is taken from **`labels.round7.top`**, not from
`labels.round8.top.train` which B0.4 used. The 113 evaluation frames are by
construction *round 8 minus round 7*, so they are inside the round-8 package.
Fitting the cost function on them and then evaluating on them would leak. Round 7
is disjoint from the evaluation set by definition.

### Temporal gate — a secondary arm, not the primary

Specified as a **gate**, not a soft cost term, because a soft term needs a weight
and a weight is a knob: a candidate (anchor or tail) is *supported* if a decoded
node of the same type lies within **25 px** in frame `f-1` or `f+1`, and
unsupported candidates are removed from the pool before assignment. 25 px is
generous for one frame at 166 fps. The **primary** configuration is
**temporal OFF**, because it is the configuration that applies to every frame
including the first and last, and because the labelled set is 113 scattered
single frames where neighbour predictions had to be generated specially.

## Frozen evaluation criterion

Unchanged from `tools/k01_misdetection_taxonomy.py` and
`tools/b03_heldout_comparison.py`, so the numbers stay comparable to Gate B0:

- **Matching** — per frame, Hungarian on mean error over nodes visible in *both*
  the GT and the output instance; a pair is *matched* at mean error `<= 20 px`.
- **animal coverage @20 px** — matched GT instances / GT instances. This is the
  metric suppression damages, and it is the reason this lab exists.
- **both-node recall @20 px** — the Gate B0 primary: the assigned output instance
  has both nodes present and *each* within 20 px.
- **false positives** — output instances not matched to any GT instance
  (the k01 denominator).
- **shared-node pairs** — two output instances with any node pair `< 5 px`
  (k01's `SHARE_PX`; also reported at b03's 1 px).
- **mouth-only count** — output instances with no assigned tail. Not scored as
  good or bad; it is the quantity Task 4's retrain must reduce.
- **Regimes** — k01's four bands from the closest pair of complete GT instances,
  in body lengths: contact `<0.25`, crossing `0.25-1`, approach `1-2`,
  separated `>2`.
- **Reported on the 113 held-out frames only.** Never the pooled 1,402: 1,288 of
  those are round-7's own training data and score 0.033 FP/frame against 0.460.

## Notes

**2026-08-03** — Pre-registration written. Nothing run.

**2026-08-03, run 1 — the rule returns REJECT.** `tools/k03_assignment_heldout.py`.

| | baseline | b04 filter | assignment (q<0.01) |
|---|---|---|---|
| animal coverage @20 px | 88.79% | 88.20% | **78.47%** |
| both-node recall @20 px | 74.93% | 74.34% | **60.77%** |
| false positives | 52 | 43 | **56** |
| shared-node pairs @5 px | 28 | 19 | **2** |
| shared **tail** pairs | 28 | 19 | **0** |
| mouth-only | 40 | 40 | 53 |

Criterion 1 was met in substance — **shared tails are 0**, structurally — but the
2 remaining shared-node pairs are mouth-on-tail coincidences, which the criterion
as written also counts, so it fails as written. Criteria 2, 3 and 4 fail outright.
**Verdict: REJECT.** The rule is not being relaxed to rescue the method.

**2026-08-03, diagnosis.** `tools/k03b_assignment_diagnosis.py`. Two causes,
separable and unequally interesting:

1. **A specification error of mine.** `EDGE_MIN_PX = 68.7` is k01's minimum
   distance from an instance *centroid* to the frame edge, and I applied it to
   *mouths*. Measured directly: GT mouths reach **28.8 px** from the edge
   (p1 = 54.4), and 11 of 320 sit inside 68.7 px. Worth **3.5** of the 10.3
   coverage points. This is the project's recurring failure — a number carried
   across to a quantity it was not measured on.
2. **An error in the specified cost, and it is the larger half.** With the edge
   rule off entirely, coverage still falls 6.8 points. The pairing audit says why:

   | of baseline instances whose mouth survived (300) | n |
   |---|---|
   | tail unchanged | 207 |
   | tail **moved** to another | 45 — of which **29 correct→wrong**, **2 wrong→correct** |
   | tail **rejected**, instance becomes mouth-only | 22 — of which **19 were correct** |

   Reassignment is 29:2 *harmful* and rejection is 19:3 *harmful*. The reason is
   that span-deviation is a **plausibility** signal, not a **discriminative** one:
   the true pairing has median cost 0.85 and a foreign pairing 17.6, but
   **45.3%** of foreign pairings still fall inside the acceptance band, and the
   true span varies with foreshortening, so a foreign tail nearer the *median*
   routinely outranks the true tail. A cost that ranks candidates against a
   population statistic cannot rank them against each other.

**Container check, for the record:** the temporal arm needed `f±1`, which the
round-8 package does not embed, so those frames came from the raw mp4.
Re-predicting the same 113 frames from the mp4 gives **113/113 frames agreeing on
count and 353/353 instances matching within 1 px** — package and raw video are
the same pixels, so the temporal arm measures the model, not the container.

---

## Amendment, 2026-08-03 — corrected specification

**This amendment is written before the corrected run and after the first result
is known.** That is weaker than the pre-registration above and is labelled as
such; the original verdict stands as REJECT and is not revised.

The audit above contains the design: **the pairing the model itself decoded is
the most informative feature available**, correct in 41 of the 47 cases where a
geometric cost wanted to overrule it. The centered-instance head decodes a tail
*inside a crop anchored on one mouth* — that is a joint prediction, not two
independent points, and the original cost threw that information away.

So the corrected cost is **origin-preferred, geometry-arbitrated**:

```text
cost(mouth i, tail j) = origin_penalty(i, j) + d_ij / hi
    origin_penalty = 0.0   if tail j was decoded in the crop anchored at mouth i
                   = 1.0   otherwise
    feasible       iff q(d_ij) >= 0.01, i.e. d in [35.3, 152.6] px  (unchanged)
    abstain cost   = 2.0   (primary)
```

`d_ij / hi` lies in `[0.23, 1.0]` over the feasible band, so a penalty of 1.0 makes
the origin pairing win whenever it is feasible — no tuning, the two terms cannot
trade. The consequence is exactly the operation this lab set out to build: a mouth
keeps the tail the model gave it; **only a mouth dispossessed by the one-to-one
constraint goes shopping**, and it then takes the nearest feasible free tail or
abstains. Abstain at 2.0 means "as costly as a foreign tail at the maximum
plausible distance", so abstention fires when no feasible tail is free.
Sensitivity arm at 1.5 (abstain unless a free feasible tail is within `0.5*hi`).

Edge rule corrected to the **mouth**-derived bound, 28.8 px — the exact analogue,
on the right quantity, of what 68.7 px is for centroids.

**Amended decision rule.** Identical to the front matter, except criterion 1 is
stated on the quantity the method actually constrains:

1. **shared-tail pairs @5 px = 0** (structural), and shared-node pairs @5 px
   reported alongside, not gated on;
2. animal coverage @20 px `>=` baseline;
3. both-node recall @20 px `>=` baseline − 2.0 points;
4. false positives `<=` baseline.

Criterion 1 is narrowed deliberately and the reason is stated in advance: a mouth
landing within 5 px of another instance's tail is a *localisation* coincidence
that a one-to-one tail constraint cannot address and was never claimed to. Gating
on it would make the criterion untestable rather than strict. Criteria 2–4 are
untouched. Run: `tools/k03c_assignment_origin.py`.
