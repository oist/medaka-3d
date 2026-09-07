# 2026-08-03 — Group K detection quality: assignment, corpus counts, retrain, annotation

All five tasks complete. Nothing here is a population claim: every labelled frame
in this project is **Group K**, so `n_groups = 1` throughout.

**Read the correction section first.** Two inherited claims turned out to be
false, and both change how the rest of this document should be read.

Reproduce:

```text
.venv\Scripts\python.exe tools/k03_assignment_heldout.py      # task 1 run 1, REJECTED
.venv\Scripts\python.exe tools/k03b_assignment_diagnosis.py   # why it failed
.venv\Scripts\python.exe tools/k03c_assignment_origin.py      # corrected, PARKED
.venv\Scripts\python.exe tools/k03d_render_assignment.py      # task 2 visual check
.venv\Scripts\python.exe tools/k04_retrain_centered_instance.py   # task 4 train
.venv\Scripts\python.exe tools/k04b_score_retrain.py          # task 4 frozen rule
.venv\Scripts\python.exe tools/k05_cardinality_corpus.py --stride 10   # task 3
.venv\Scripts\python.exe tools/k06_partial_instance_worklist.py       # task 5
.venv\Scripts\python.exe tools/k07_production_baseline.py     # round 8 re-baseline
.venv\Scripts\python.exe tools/k08_locked_benchmark.py        # the locked benchmark
```

## Summary of verdicts

| task | outcome |
|---|---|
| 1 — mouth-anchored assignment | run 1 **REJECTED** on its own rule; corrected version **PARKED** (blocker: Gate A) |
| 2 — evaluate on held-out frames | done; shared tails → 0 everywhere, FP 52 → 41 vs round 7 |
| 3 — corpus cardinality, 65 experiments / 12 groups | done; **K is not special** — the detector counts no better on the group it trained on |
| 4 — capacity retrain | rule **PASSED**, verdict **PARKED** — does not reproduce on the locked benchmark |
| 5 — annotation worth doing | **464 partial instances, ≈3.9 h**, work list written |
| — | **two inherited claims corrected**: round 8 is production; a locked benchmark already existed |

Labs: [`20260803_mouth_anchored_assignment`](../../labs/20260803_mouth_anchored_assignment/),
[`20260803_centered_instance_capacity`](../../labs/20260803_centered_instance_capacity/).

---

## CORRECTION, and it governs everything below: round 8 is production, not round 7

Raised by the experimenter, verified, **upheld**. `MANIFEST.json` records
`production_models.sleap_top = "round7"`. That is not a measurement — it is a
hardcoded constant, `PRODUCTION = {"top": "round7"}`, in
`tools/build_asset_manifest.py`, which the manifest echoes back so that an
assertion reads as a finding. The chronology refutes it:

| | date |
|---|---|
| round-7 top models trained | 2026-07-04 |
| **every retained corpus prediction produced** | **2026-07-07 → 07-13** |
| round-8 models trained, **top and side** | 2026-07-23 |

The corpus predictions name `round7_260704.centroid_n1295` and
`.centered_instance_n1295` in their own `.slp` provenance. They are round 7
because round 8 **did not exist yet**, not because anyone selected round 7 for
top. Round 8 postdates round 7 for both views, and round 8 is the model the
experimenter reported as failing at crossings — the premise Phase B0 was opened
on. CLAUDE.md and the manifest generator are corrected.

Round-8 top is **architecturally identical** to round-7 top: `filters=24`,
`filters_rate=2.0`, `max_stride=64`, `crop_size=512`, `anchor_part='mouth'`,
`sigma=2.5`, 200 epochs, batch 16, **70,755,866 parameters**. The only difference
is the label set — **1,382 "strictmouth" frames against 1,295**.

### The evaluation set was concealing this

The 113 frames used by B0.3, B0.4, k01 and every arm above are *round-8
(train + val) minus round-7*. Against the packages:

- **93 of the 113 are in round-8's training data.** Scoring production there
  measures memorisation — k01 already showed that costs a factor of 13.8 for
  round 7 (0.033 FP/frame in-sample against 0.460 held out).
- **20 of the 113 are round-8's validation split**, and **none of those 20 are in
  round 7**.

Those 20 are the only frames *in the label packages* held out from **both**
checkpoints — and, as the next two sections show, they are neither the only such
frames in the project nor genuinely independent. A better set (40 frames, three
days, frame-disjoint) was sitting in the archive, and the annotation of these 20
was initialised from round-7 predictions. Both were derivable at any point and
neither was looked for.

### What production actually does, on the 20 honest frames

| model | coverage @20 px | both-node recall | FP | FP/frame | shared tails |
|---|---|---|---|---|---|
| round 7 top-down | 98.33% | 93.33% | 7 | 0.350 | 3 |
| **round 8 top-down (production)** | 96.67% | 93.33% | **0** | **0.000** | **0** |
| k04 retrain (round-7 labels) | 98.33% | 93.33% | 7 | 0.350 | 1 |

And across all 113 frames — where round 8 is flattered by having trained on 93 of
them, so read it as its best case:

| model | coverage | recall | FP | FP/frame | shared tails | mouth-only |
|---|---|---|---|---|---|---|
| round 7 raw | 88.79% | 74.93% | 52 | 0.460 | 28 | 40 |
| round 7 + assignment | 88.20% | 71.98% | 41 | 0.363 | **0** | 59 |
| **round 8 raw (production)** | 93.22% | 85.25% | 6 | **0.053** | 5 | 15 |
| **round 8 + assignment** | 93.51% | 84.07% | 3 | 0.027 | **0** | 22 |
| k04 retrain raw | 90.86% | 79.65% | 43 | 0.381 | 11 | 37 |
| k04 retrain + assignment | 88.79% | 78.17% | 31 | 0.274 | **0** | 45 |

**Three things follow, and they reframe the whole task.**

1. **k01's headline false-positive rate of 0.46/frame is round 7, not
   production.** Round 8 on the same frames is **0.053/frame**, and **0.000** on
   the 20 it was not trained on.
2. **The reported "one tail joined to two mouths" is largely a round-7
   phenomenon.** Round 7 emits 28 shared tails across the 113 and 3 on the honest
   20; round 8 emits 5 and **0**. Task 1's method was built to remove a failure
   that production, on this evidence, mostly does not have. It still removes it
   where it occurs — round 8's 5 → 0 — but the prize is far smaller than the
   round-7 numbers implied.
3. **No model separation is established, and the apparent separation is
   leakage.** Round 8 leads on this table by trained on 93 of these 113 frames.
   Strip that — and strip the 30% of the locked benchmark that is verbatim
   round-8 output — and **round 7, round 8 and k04 land on identical coverage
   (97.92%) with both-node recall differing by one instance**. See the locked
   benchmark section. The honest statement is that on leakage-free data these
   three models are indistinguishable, and 16 frames / 48 instances is far too
   small to rank them.

**What I got wrong.** I took `production = round7` from the manifest and built
four tasks on it without testing it, which is exactly the failure CLAUDE.md
exists to prevent — an inherited claim repeated as settled.

### The archive had already run this, and says so explicitly

Searching the retired repo's `report/` tree — which nothing in this rebuild had
done — turns up `report/20260724/sleap_topdown_optimization/`, a full round-7
versus round-8 evaluator, and `report/20260727/.../EVALUATION_REPORT.md`, whose
first line under **Outcome** is:

> **"Round 8 remains the production model for both top and side views."**

Dated 2026-07-27. The experimenter's correction is the archive's own recorded
conclusion.

**My 20-frame numbers reproduce the archive's exactly.** Its
`round7_vs_round8_top_pilot_full.json` evaluates the same 20 round-8 validation
frames and reports round 7: `n_gt 60, n_detected 66, n_extra 7,
shared_tail 3`; round 8: `n_gt 60, n_detected 58, n_extra 0, shared_tail 0`.
Instance for instance, that is the table above, from an independent
reimplementation. Round 8 also wins precision 1.000 vs 0.894 and f1 0.983 vs
0.937, losing recall 0.967 vs 0.983.

**And it corrects my claim, which was too strong.** I called those 20 frames
"held out from both". The archive's Phase-1 audit is stricter and right:

> "The audit found **no evaluation cohort genuinely unseen by both models**. The
> Round 8 pilot event is a valid Round 8 fine-tune holdout, but six synchronized
> Round 7 source frames occur in that event/guard interval, **Round 8 was
> warm-started from Round 7**, and **the pilot annotations were initialized from
> Round 7 predictions**. It is therefore classified
> `held_out_round8_but_seen_round7`, not as a common locked test."

Round-7's median localisation error on those 20 frames is **0.0004 px** — its
predictions *are* the labels. So the 20 are disjoint by frame key and not by
provenance, and round 7 is flattered on localisation there. Round 8 still wins
precision and shared-tail rate despite that, which is the useful part.

Everything below the next section was computed against **round 7** and remains
correct as a statement about round 7. Read it that way.

---

## The locked benchmark exists, and this rebuild did not know it

Gate B0 concluded that **no unbiased evaluation set exists** and made building
one the strongest argument for Phase A. That conclusion was reached without
reading the archive, and it is wrong.

```text
Z:\StephensU\medaka\benchmarks\sleap_hitl_v2
   \phase1-hitl-v2-preserve-20260802T174200JST-60e13a7746d1\final
   \shared_benchmark_40_hitl_v2_locked.pkg.slp
```

40 synchronised source frames × 3 views = 120 view-frames, **360 human
instances, exactly 3 per view-frame** (345 complete, 15 partial), spanning
**all three K days** — K_1_1 14, K_1_2 13, K_1_3 13. Preserved during the
migration on 2026-08-02.

Its 40 top frames are **disjoint from round-7 training, round-8 train, and
round-8 val — all three, zero overlap.** That is a frame-locked test set, and it
is a better one than the 113: three days rather than one, and not selected as
"frames a human chose to correct".

**Defect, reported as one:** it is **not in `MANIFEST.json`**. A human-made
artefact of precisely the kind design rule 3 exists to checksum and rule 7 exists
to keep catalogued is sitting outside the catalogue, unverified, where the next
session will fail to find it exactly as this one nearly did. It should be added.

### Contamination first, scores second

Frame-disjoint is not annotation-independent. Measured with the same ≤ 0.01 px
test Gate B0 used:

| model | GT instances byte-identical to its own output |
|---|---|
| round 7 | **0.0%** (0/120) |
| **round 8 (production)** | **30.0%** (36/120) |
| k04 retrain | **0.0%** (0/120) |

The HITL runbook offered an "optional Round-8 prediction review" path alongside
the manual one, and 30% of the locked benchmark's instances are round-8 output
accepted unchanged. **This is Gate B0's governing finding repeated one round
later, in the artefact built to escape it.** So the benchmark is bracketed the
same way — an ALL arm and a CORRECTED arm keeping only the 16 frames containing
no round-8-verbatim instance.

### Results on the locked benchmark

| arm | coverage | both-node recall | FP | FP/frame | shared tails | mouth-only |
|---|---|---|---|---|---|---|
| **ALL 40 frames** | | | | | | |
| round 7 raw | 95.83% | 91.67% | 5 | 0.125 | 2 | 6 |
| round 7 + assignment | 95.83% | 90.00% | 4 | 0.100 | **0** | 8 |
| round 8 raw (production) | 97.50% | 93.33% | 0 | 0.000 | 1 | 3 |
| round 8 + assignment | 97.50% | 92.50% | 0 | 0.000 | **0** | 4 |
| k04 retrain raw | 95.00% | 90.83% | 6 | 0.150 | 1 | 4 |
| k04 + assignment | 94.17% | 90.00% | 5 | 0.125 | **0** | 5 |
| **CORRECTED — 16 frames independent of round 8** | | | | | | |
| round 7 raw | **97.92%** | 91.67% | 1 | 0.062 | 1 | 2 |
| round 8 raw (production) | **97.92%** | 91.67% | 0 | 0.000 | 0 | 2 |
| k04 retrain raw | **97.92%** | **93.75%** | 1 | 0.062 | 0 | 1 |

**Round 8's lead is an artefact of its own output being 30% of the truth.** On
the arm where the annotation is independent of it, **all three models have
identical coverage (97.92%)** and their both-node recall differs by one instance.

*Do not over-read the CORRECTED arm either:* 16 frames, 48 instances, so one
instance moves recall 2.1 points. "k04 is best" is a one-instance claim and is
not one. The defensible statement is that **on independent annotation the three
models are indistinguishable.**

### What the locked benchmark says about the whole session

**Every model is far better than the 113-frame set implied.** Round 7 scores
95.83% coverage and 0.125 FP/frame here against 88.79% and 0.460 there. The 113
are the frames a human chose to correct — enriched for difficulty by
construction, as k01 warned. **k01's 0.46 FP/frame is a worst-case number, not a
detection rate**, and this document's round-7 sections should be read with that
in mind.

**Shared tails are rare on a fair sample**: 2, 1 and 1 across 40 frames for
round 7, round 8 and k04. Assignment still drives every one of them to zero, at a
cost of 1–2 extra mouth-only instances. That is the method working as designed on
a problem an order of magnitude smaller than the round-7 numbers advertised.

---

## The headline (round 7)

**Mouth-anchored one-to-one assignment eliminates the reported failure
structurally — shared tails go 28 → 0 — and cuts false positives 52 → 41,
beating the geometric filter Gate B0 adopted (43). It costs 2.95 points of
both-node recall, which is more than the pre-registered tolerance, so the
verdict is PARK, not promote.**

The recall is not lost to error. It is converted into **abstention**: mouth-only
instances rise 40 → 59, each one an animal whose position is kept and whose
orientation is withheld. Whether that trade is good cannot be settled by
both-node recall, which scores an honest abstention and a silently swapped tail
identically. It needs the downstream metric Phase A exists to build.

**The first specification was rejected on its own pre-registered rule, and the
rule was not relaxed.** That rejection is the most useful result here.

**On production the same method still drives shared tails to 0 (5 → 0) and
halves false positives (6 → 3), at a much smaller absolute scale.**

---

## Task 1 — the method, and the specification that failed first

### Run 1: REJECT

The specification as given — anchors = decoded mouths gated at 68.7 px from the
frame edge, tail candidates deduplicated at 8 px, Hungarian assignment with
cost = *deviation of the mouth–tail distance from the labelled span
distribution*, rejection ceiling mandatory — was implemented verbatim and scored
against a rule fixed in writing beforehand.

| | baseline | b04 filter | run 1 |
|---|---|---|---|
| animal coverage @20 px | 88.79% | 88.20% | **78.47%** |
| both-node recall @20 px | 74.93% | 74.34% | **60.77%** |
| false positives | 52 | 43 | **56** |
| shared **tail** pairs @5 px | 28 | 19 | **0** |
| mouth-only | 40 | 40 | 53 |

Three of four criteria failed. **Verdict: REJECT.**

### Why it failed — two causes, unequally interesting

**One is mine.** `EDGE_MIN_PX = 68.7` is k01's minimum distance from an instance
*centroid* to the frame edge, and the specification applies it to *mouths*. A
mouth sits up to half a body span nearer the edge than its centroid. Measured
directly on the same 113 frames:

| ground-truth node | n | min | p1 | median |
|---|---|---|---|---|
| mouth | 320 | **28.8 px** | 54.4 | 189.3 |
| tail | 335 | 44.6 px | 80.0 | 214.0 |
| centroid | 339 | 68.7 px | 84.9 | 204.4 |

11 of 320 ground-truth mouths sit inside 68.7 px. Worth **3.5** of the 10.3
coverage points. This is the project's recurring failure mode — a number carried
across to a quantity it was never measured on — and it is the third instance
recorded in these documents.

**The other is in the specified cost, and it is the larger half.** With the edge
rule disabled entirely, coverage still falls 6.8 points. The pairing audit
(`k03b`) shows what the solver actually did:

| of the 300 baseline instances whose mouth survived | n |
|---|---|
| tail unchanged | 207 |
| tail **moved** to a different candidate | 45 — **29 correct→wrong**, 2 wrong→correct |
| tail **rejected**, instance becomes mouth-only | 22 — **19 had been correct** |

Reassignment was 29:2 harmful and rejection 19:3 harmful. The reason is that
span-deviation is a **plausibility** signal, not a **discriminative** one. The
true pairing has median cost 0.85 against a foreign pairing's 17.6 — but
**45.3% of foreign pairings still fall inside the acceptance band**, and because
the true span varies with foreshortening, a foreign tail nearer the *median*
routinely outranks the animal's own. A cost that ranks candidates against a
population statistic cannot rank them against each other.

### The corrected specification

Written into the lab README as a dated amendment **before** it ran, and labelled
there and in the tool as weaker evidence than the original pre-registration
because run 1's result was already known.

The audit contains the design: over the 47 pairings a geometric cost wanted to
overrule, the model's own pairing was correct in **41**. The centered-instance
head decodes a tail *inside a crop anchored on one mouth* — a joint prediction,
not two independent points — and run 1 threw that away.

```text
cost(mouth i, tail j) = origin_penalty(i, j) + d_ij / hi
    origin_penalty = 0.0  if tail j was decoded in the crop anchored at mouth i
                   = 1.0  otherwise
    feasible       iff d in [35.3, 152.6] px   (the B0.4 band, unchanged)
    abstain cost   = 2.0
```

`d/hi` spans `[0.23, 1.0]` over the feasible band, so a penalty of 1.0 makes the
origin pairing win whenever it is feasible; the two terms cannot trade and there
is nothing to tune. The result is the operation the task asked for: **a mouth
keeps the tail the model gave it, and only a mouth dispossessed by the one-to-one
constraint goes shopping** — taking the nearest feasible free tail, or abstaining.
The edge rule uses the mouth-derived 28.8 px.

---

## Task 2 — the 113 held-out frames

Frame set, matcher, both-node recall, regimes and taxonomy are imported unchanged
from `b03_heldout_comparison` and `k01_misdetection_taxonomy`, so these numbers
sit on the same scale as Gate B0.

| | baseline | b04 (adopted at Gate B0) | **assignment** |
|---|---|---|---|
| animal coverage @20 px | 88.79% | 88.20% | **88.20%** |
| both-node recall @20 px | 74.93% | 74.34% | **71.98%** |
| precision | 71.95% | 73.68% | 71.76% |
| false positives | 52 | 43 | **41** |
| shared-node pairs @5 px | 28 | 19 | **1** |
| shared **tail** pairs @5 px | 28 | 19 | **0** |
| mouth-only | 40 | 40 | 59 |
| tail-only | 12 | 12 | **0** |

By regime — the FP concentration at contact and crossing is where the method does
its work:

| regime | frames | FP baseline | FP assignment | shared@5 baseline | shared@5 assignment |
|---|---|---|---|---|---|
| contact | 12 | 4 | **1** | 5 | 0 |
| crossing | 73 | 32 | **26** | 17 | 1 |
| approach | 19 | 10 | 8 | 3 | 0 |
| separated | 7 | 5 | 5 | 2 | 0 |

Taxonomy mix, held-out only: `A shared_node` **21 → 1**, `B duplicate` 3 → 8,
`D collapsed` 2 → 0, `E partial` 19 → 21, `G unclassified` 7 → 11.

Two of those movements need stating plainly rather than buried:

- **`B` and `G` rise.** They do not represent new errors — they are the *same*
  instances, reclassified. k01 applies its categories in priority order, so an
  instance that used to be caught by `A shared_node` falls through to `B` or `G`
  once the shared tail is gone. The total is what moved: 52 → 41.
- **`E partial` barely moves (19 → 21)** even though tail-only instances went
  12 → 0. Removing a tail-only prediction removes it as an instance; the
  mouth-only instances that replace it are still category `E`. This is precisely
  why `E` is Task 5's job and not Task 1's.

### The one remaining shared-node pair

Criterion 1 in the original front matter gated on shared nodes of *any* kind and
so failed at 2; the amendment narrowed it, in advance and with the reason stated,
to **shared tails**, which is the only thing a one-to-one tail constraint can
control. The residual pair is a predicted *mouth* landing within 5 px of another
instance's *tail* — a localisation coincidence, not a grouping error. It is
reported, not gated.

### Does the method reach its own ceiling?

The working document predicted a hard limit: in 10 of 19 shared-tail frames no
true tail was ever decoded (a 20× decoder sweep recovers 1 of 10), so at best
**9 animals recover and 10 abstain**.

| | run 1 cost (edge corrected) | **origin cost** |
|---|---|---|
| tail unchanged | 207 | 246 |
| moved to a **correct** tail | 11 | **9** |
| moved to a **wrong** tail | 34 | **4** |
| abstained | 22 | 15 |

**The corrected method recovers exactly 9** — the predicted ceiling — and now
moves 9 tails right for every 4 wrong, against 11:34 before. It is not
underperforming its limit; the limit is the model's.

### Visual check

`k03d` renders all 23 frames containing a shared tail, before and after, at the
same crop. Two are worth naming:

- **Frame 274709**, the phantom case the task called out: two mouths converge on
  one tail; after assignment the dispossessed mouth takes the correct free tail
  and the frame goes to 2/3 complete, 0 mouth-only. The recovery works.
- **Frame 30076**: three animals, and **only one tail decoded in the whole
  frame**. Three mouths abstain. This is the representational gap Task 4 targets,
  and the picture shows there is nothing for assignment to ration.

### Verdict on the lab: PARK

| criterion (amended, pre-registered) | result | |
|---|---|---|
| 1. shared-tail pairs @5 px = 0 | 28 → 0 | **PASS** |
| 2. animal coverage ≥ baseline | 88.79% → 88.20% (−0.59) | FAIL |
| 3. both-node recall ≥ baseline − 2.0 | 74.93% → 71.98% (−2.95) | FAIL |
| 4. false positives ≤ baseline | 52 → 41 | **PASS** |

**Blocker, named as the lab rules require:** both-node recall cannot arbitrate
this. It scores a mouth-only instance and a silently swapped tail as equally
wrong, and the entire argument for the method is that those two outcomes are not
equally wrong downstream — a mouth-only instance keeps position and identity
continuity and loses orientation, while a swapped tail corrupts an identity chain
without a symptom. Settling it needs a metric over **tracks**, which is Gate A.
Parked until Phase A's harness exists; not promoted to `src/medaka3d/` on a
2.95-point loss against a criterion it was measured on.

**This does not change Gate B0's adoption of the B0.4 filter.** It does, however,
mean the shared-node rule should not be applied *as suppression at crossings*,
for the reason k01 established: in 15 of 47 FP frames only 2 of 3 animals are
covered, so deleting the loser discards the only evidence of a third animal.

---

## Task 4 — the capacity retrain: rule PASSES, but it does not reproduce

`max_stride` 64 → 32, `filters` 24 → 32, intensity augmentation `p=0.0` → `0.5`.
Everything else held at round-7 values. Trained on the **round-7** package so the
113 stay held out, and only the centered-instance head was changed — the centroid
head is round-7's, so a difference is attributable to one thing.

Note the resulting network is **smaller**: 31,405,858 parameters against round-7's
70,755,866. Dropping the deepest encoder block costs more than widening the rest
adds, so this is a *reallocation* of capacity, not an increase. 200 epochs,
~88 s/epoch, seed fixed at 0 (round 7 left it unset and is not reproducible).

### The pre-registered rule, applied as frozen

| criterion (baseline = round 7, on the 113) | result | |
|---|---|---|
| 1. mouth-only under K03c assignment < 59 | 59 → **45** | **PASS** |
| 2. animal coverage ≥ 88.79% | 88.79% → 88.79% | **PASS** |
| 3. both-node recall ≥ 74.93% | 74.93% → **78.17%** | **PASS** |

**→ ADOPT**, under the rule as written. The premise held: the head does now
represent tails it previously did not, and raw shared tails fall **28 → 11**.

**Overfitting narrowed rather than widened**, which is the opposite of what the
lab predicted as its likeliest failure:

| | in-sample FP/frame | held-out FP/frame | ratio | recall gap |
|---|---|---|---|---|
| round 7 | 0.033 | 0.460 | 14.1× | +13.78 pts |
| **k04** | 0.033 | 0.381 | **11.4×** | **+9.76 pts** |

`val/loss` finished at 0.000268 against round-7's 0.00029 — better, and, as
pre-registered, **not used to decide anything**. It was flat for the last ~80
epochs while these task metrics moved.

### Why the verdict is still not "promote"

**The result does not reproduce on the locked benchmark.** On 40 frame-disjoint
frames across three K days, k04 is *not* better than round 7 (coverage 95.00% vs
95.83%, recall 90.83% vs 91.67%, FP 6 vs 5), and on the CORRECTED arm the three
models are indistinguishable.

The most economical explanation is that the 113 frames are enriched for
difficulty — they are the subset a human chose to correct — so a change that
helps on hard frames shows a large gain there and a negligible one on a fair
sample. That is a real property of the model, but it is not the +3.24 recall
points the frozen rule scored.

**The rule passed and I am not revoking it; I am reporting that its evaluation set
turns out to be unrepresentative, which the rule could not have known.** The
honest summary: *the architecture change is real and helps on crossing-heavy
frames; on a fair sample it is not distinguishable from round 7.*

### Retraction: "labels beat architecture" was an overclaim

An earlier version of this document, of the ROADMAP entry and of both lab
`RESULT.md` files stated that round 8 beats round 7 and k04 because of its
labels, and offered that as the strongest argument for Task 5. **That is not
supported and is withdrawn.**

Round 8's margin appears on exactly two kinds of data, both of which contain its
own output:

| comparison | round 7 | round 8 | k04 | leak |
|---|---|---|---|---|
| the 113 frames, coverage | 88.79% | **93.22%** | 90.86% | round 8 trained on **93 of 113** |
| locked benchmark, ALL 40, coverage | 95.83% | **97.50%** | 95.00% | **30%** of instances are verbatim round 8 |
| **locked, 16 frames independent of round 8** | **97.92%** | **97.92%** | **97.92%** | none |

On the only leakage-free arm the three models are **identical on coverage** and
differ by one instance on both-node recall. **No model separation is established
by any evaluation in this project.**

Task 5's case does not need that claim and never did. It rests on its own
measurement: 5.9% of top training instances are missing a node, the model
reproduces that as 36.5% of held-out false positives, and no geometric rule can
remove a one-node instance because it is not implausible, merely incomplete. 464
instances, ≈3.9 hours. That argument stands on its own; the round-8 comparison
was never needed to make it and should not have been used.

## Task 5 — the annotation worth doing

`k06` enumerates it. The recommendation from the working document survives
measurement, and the size is now exact rather than approximate.

| package | instances | complete | mouth missing | tail missing | empty | partial |
|---|---|---|---|---|---|---|
| `labels.round7.top` | 3,885 | 3,664 | 150 | 66 | 5 | **5.69%** |
| `labels.round8.top.train` | 4,146 | 3,903 | 168 | 70 | 5 | **5.86%** |
| `labels.round8.top.val` | 60 | 59 | 1 | 0 | 0 | 1.67% |
| `labels.round8.side.train` | 8,293 | 7,350 | 486 | 414 | 43 | 11.37% |
| `round9b` side | 9,354 | 8,380 | 498 | 432 | 44 | 10.41% |

*Terminology, because the two documents differ:* k01 names a partial instance by
the node that is **present** ("mouth-only"); the table above names the node that
is **missing**. k01's "1.7% mouth-only, 3.9% tail-only" is this table's
70 tail-missing (1.69%) and 168 mouth-missing (4.05%). Same instances.

**The job: 464 partial instances** — 221 in round 7, 243 in round 8 train (the
working document's "~244", confirmed) — across 426 frames. At 0.5 min each that
is **3.9 hours**, against the **367.1 h** that annotating every B0.2 encounter
candidate would take. **334 of the 464 sit within 100 px of a complete instance**,
i.e. at a crossing, so they are both the hardest for an annotator and the ones
whose absence the model most visibly reproduces.

Work list written to `work_dir/k06_partial_instances/`: `worklist.csv` (one row
per instance with video, frame, which node is missing, and the position of the
node that is present), `frames.txt` for driving a labelling GUI, `worklist.json`.

**The tool does not do the annotation and should not.** Completing a node is a
human judgement about which pixels are the animal.

Note what this cannot fix: `E partial` was 36.5% of held-out false positives, and
assignment already removed the tail-only half by construction. Completing the
labels targets the remaining half plus the model's general tendency to emit
one-node instances. It is a bet, sized at 3.9 h, not a certainty.

---

## Task 3 — the corpus, and the rollout question

No labels exist outside Group K, so the only free label is **cardinality**: one
female and two males per trial, every labelled frame holds exactly 3 instances,
so `n > 3` implies false positives and `n < 3` implies misses. `k05` applies it
to the retained round-7 `top_cam` predictions for **all 65 experiments across 12
groups**, every 10th frame (16.6 Hz — successive frames at 166 fps are
near-duplicates, and the decimation is deterministic so the run reproduces).

**Read this as a bound on one failure mode.** Counting is blind to a swapped
tail, blind to a mouth 40 px off the animal, and blind to three detections on two
fish plus a reflection. **A count of 3 is not a correct frame.**

### The rollout answer: K is not special

| | experiments | frames | `=3` raw → assigned | `<3` raw → assigned | `>3` raw → assigned |
|---|---|---|---|---|---|
| **K** (trained on) | 6 | 209,160 | 89.05% → **89.45%** | 6.70% → 6.87% | 4.25% → **3.68%** |
| **non-K** (transfer) | 59 | 2,056,740 | 90.15% → 90.18% | 6.44% → 6.63% | 3.41% → 3.19% |

**The detector counts no better on the group it was trained on than on the 11 it
has never seen — it counts very slightly worse (−1.10 points of exact-3).** The
main risk named for the rollout is, on this metric, not present.

The between-group spread swamps the difference. Exact-3 before assignment runs
from **I 84.96%** to **G 94.14%** — a 9.2-point range against a 1.1-point
K-versus-rest gap. Whatever drives detection quality across this corpus, it is
not training-set membership. Per group:

| group | exp | `=3` raw | `=3` assigned | `<3` raw → asg | `>3` raw → asg |
|---|---|---|---|---|---|
| G | 6 | 94.14% | 94.22% | 4.02 → 4.14 | 1.84 → 1.63 |
| M | 6 | 93.70% | 93.78% | 4.54 → 4.65 | 1.75 → 1.57 |
| F | 3 | 91.80% | 91.73% | 4.00 → 4.49 | 4.20 → 3.79 |
| P | 6 | 91.45% | 91.57% | 6.39 → 6.47 | 2.16 → 1.96 |
| O | 6 | 90.89% | 90.97% | 6.35 → 6.43 | 2.76 → 2.59 |
| L | 6 | 90.81% | 90.77% | 5.11 → 5.39 | 4.08 → 3.84 |
| H | 6 | 89.56% | 89.66% | 6.27 → 6.33 | 4.16 → 4.01 |
| **K** \* | 6 | **89.05%** | **89.45%** | 6.70 → 6.87 | 4.25 → 3.68 |
| E | 6 | 88.16% | 88.15% | 7.75 → 8.13 | 4.09 → 3.72 |
| D | 3 | 87.94% | 87.47% | 7.98 → 8.60 | 4.08 → 3.93 |
| N | 5 | 87.47% | 87.63% | 5.67 → 5.79 | 6.86 → 6.59 |
| I | 6 | 84.96% | 85.00% | 12.19 → 12.25 | 2.85 → 2.74 |

\* the only group with labels, and the only group the detector saw.

Assignment moves counts in the direction its design predicts and by very little:
`>3` down everywhere (−0.57 on K, −0.22 elsewhere), `<3` up slightly (+0.17,
+0.19). Exact-3 is flat to marginally better in 9 of 12 groups. Surplus
detections on K fall 9,335 → 8,021 (−14%).

**This does not say assignment is corpus-safe.** It says it does not break
counting. Localisation outside K remains entirely unmeasured and cannot be
measured without labels from another group.

### The B0.4 filter does not survive contact with the other 11 groups

This is the most operationally important result in the document, and it is about
the rule **Gate B0 adopted for production**.

Full three-arm run, all 65 experiments, 12 groups, every 20th frame (1.13 M
frames), each arm through its own production implementation:

| | `=3` raw | `=3` **b04** | `=3` assignment | `<3` raw | `<3` **b04** | `<3` assignment |
|---|---|---|---|---|---|---|
| **K** (trained on) | 89.02% | 87.85% | **89.44%** | 6.73% | 8.51% | 6.91% |
| **non-K** (59 exp) | 90.15% | **82.52%** | **90.19%** | 6.44% | **14.58%** | 6.62% |

**On the 11 groups it was never measured on, the B0.4 filter destroys 7.6 points
of exact-3 and more than doubles the miss rate.** Per group the damage runs from
−1.2 (K, M, L, F) to **−18.6 (H)**, −15.0 (I), −14.9 (N), −12.5 (E).

**The mechanism is the span rule, and it is exact.** Instances dropped per frame
by the span bound against the exact-3 damage:

| group | span drops/frame | shared-node drops | Δ exact-3 |
|---|---|---|---|
| M | 0.015 | 622 | −1.22 |
| L | 0.017 | 598 | −1.28 |
| F | 0.017 | 150 | −1.38 |
| **K** \* | **0.018** | 953 | **−1.17** |
| D | 0.032 | 247 | −2.89 |
| O | 0.036 | 594 | −3.12 |
| P | 0.046 | 579 | −3.87 |
| G | 0.051 | 502 | −4.82 |
| E | 0.140 | 799 | −12.51 |
| I | 0.178 | 724 | −15.03 |
| N | 0.198 | 723 | −14.85 |
| H | **0.243** | 795 | **−18.62** |

Monotone, and pooled across the corpus the span rule accounts for **97,523 of
b04's 104,809 deletions (93%)** while the shared-node rule — the part that
addresses the reported failure — accounts for 7,286 and is flat across groups.

**K has the lowest drop rate of any group, because the bounds were derived from
K.** The band `[35.1, 152.4] px` is the 0.5–99.5 percentile of *Group-K labels*.
Group medians are similar (105–119 px), so this is not a scale shift; it is the
**upper tail**. K's 99th percentile predicted span is 150.0 px, inside the bound;
H's is 161.3, E's 161.5, N's 159.6. The bound slices into the legitimate
distribution of every group with a longer tail than K's.

**This is the project's stated failure mode, committed against a production
rule.** B0.4 was measured on one group and adopted globally, and the ROADMAP's
own instruction is *enumerate, don't extrapolate*. Gate B0 flagged `n_groups = 1`
as a limitation; what it could not know is how large the cost would be.

**Recommendation: do not apply the B0.4 span rule outside Group K.** Either
derive bounds per group from that group's own predictions, or — better, since it
needs no labels — drop the span rule as an instance-deletion filter and keep it
only where assignment uses it: as a **feasibility test for pairing a tail**, which
converts an out-of-band instance into a mouth-only anchor instead of deleting the
animal. That difference is exactly why assignment costs +0.18 points of `<3` on
non-K where b04 costs **+8.14**.

### Suppression versus assignment

Both arms reduce `>3` by a similar amount (non-K: b04 3.41 → 2.89, assignment
3.41 → 3.19). Only b04 pays for it in missed animals. This is k01's "suppression
destroys evidence" reproduced at corpus scale, across 12 groups, with no
annotation required — and it is the clearest argument the assignment reframing
has, far stronger than anything the 113 frames could show.

## What none of this establishes

- **`n_groups = 1`.** Every labelled frame is Group K. The detector was trained
  only on K. Task 3 is the only handle on the other 11 groups and it can measure
  counting alone.
- **The evaluation set is not unbiased.** ~48% of held-out GT instances are
  byte-identical to round-7 top-down output, because round 8 was labelled by
  correcting round-7 predictions. Every number above is a bound, not a
  measurement, and it is biased *toward* the baseline — the thing assignment is
  being compared against.
- **113 frames.** Single-category percentages move several points on a handful of
  instances.
- **The frame set is enriched for difficulty by construction** — it is the subset
  a human chose to correct. These are worst-case proportions.
