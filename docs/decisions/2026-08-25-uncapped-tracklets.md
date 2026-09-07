# Stage A without a track cap, and stage C without a fixed cardinality

**Date:** 2026-08-25
**Status:** measured. Nothing is promoted. `c16`, `c19`, `c20`, `c21` and `c23`
are **not** modified and their outputs are not overwritten.
**Builds on:** [2026-08-19 uncapped cardinality](2026-08-19-uncapped-cardinality.md),
[2026-08-18 Viterbi correspondence](2026-08-18-viterbi-correspondence.md),
[2026-08-18 2D first](2026-08-18-2d-first-tracking.md)
**Tools:** new `c24`, `c24b`, `c25`; new `tests/test_variable_cardinality.py`.
**Origin:** the experimenter's proposal — *the final identity count is three,
but the per-camera candidates must not be culled, because detection score and
"reflection or fish" are not empirically correlated.*

## What the cap was actually doing

`c16` runs `sleap_nn`'s tracker with `max_tracks=3` on the reasoning that three
animals are present in every frame. That is true about the animals and false
about the detections, and the difference is not academic:

| detections in the frame (`top_cam`, K_1_2) | frames | labelled by `c16` | unlabelled |
|---|---|---|---|
| 3 | 327,977 | 3 | 0 |
| **4** | **6,925** | **3** | **6,925** |
| 5 | 126 | 3 | 252 |
| 6–8 | 57 | 3 | 183 |

**7,360 instances on `top_cam` alone carry no label**, 5,200 on `side1_cam`,
2,927 on `side2_cam`. A cap does not refuse a false positive. It hands out three
labels and leaves whatever is left over unlabelled, and *which* three it keeps is
decided by a Hungarian match with no rejection threshold.

`c23` found one scene by eye — K_1_2 113,374–113,383, where a reflection in the
tank wall holds label 0 for ten frames while the real third fish carries no
label. **That scene is not rare.** Over the 14,501 frames where some view reports
four or more detections, comparing the cheapest correspondence buildable from
`c16`'s three labels against the cheapest over *all* the detector's instances:

| the capped three are worse by | frames | runs |
|---|---|---|
| > 1 mm, and the better answer uses an unlabelled instance | 5,398 | 1,284 |
| **> 4 mm** | **4,515** | **1,003** |
| > 10 mm | 3,425 | 712 |

The method reproduces the scene read by eye — 113,374–113,382 all flagged at
10.9–11.3 mm, and 113,383, the frame where the label lands back on the fish, at
0.000 mm and not flagged.

## Why score cannot be used to choose

Measured on `side1_cam` 100,000–120,000, uncapped:

| | mean score | |
|---|---|---|
| instances on the three longest-lived labels (n = 56,503) | 1.702 | 5th pct 1.079 |
| instances on every other label (n = 2,739) | 1.683 | 95th pct 1.958 |

**92.7% of the non-primary instances score above the 5th percentile of the
primary ones.** Culling by score removes real animals at close to the rate it
removes artefacts. `max_instances`, `tracking_pre_cull_to_target` and any
score-ranked "keep the best three" are unusable on this data. The experimenter's
premise is confirmed.

## Two facts about `sleap_nn` 0.3.1, read off the code

1. **`max_tracks` silently changes the candidate maker.** `Tracker.from_config`
   switches `candidates_method='fixed_window'` to `'local_queues'` whenever
   `max_tracks` is set, logging at INFO. `c16` passes `fixed_window` and its
   provenance records `fixed_window`; **local queues is what ran.** Verified by
   instantiating the tracker with `c16`'s arguments and reading
   `type(tracker.candidate).__name__`. Dropping the cap alone would hand the run
   back to `fixed_window` and change two things at once — that arm fragments
   into **14** labels on the same 20,000 frames where local queues creates 5.
2. **There is no similarity threshold.** `scores_to_cost_matrix` is `-scores`
   with NaN sent to `inf`, and `assign_tracks` takes the Hungarian answer
   whatever it costs. The only gate is in the Kalman path, which requires the
   very instance count this pipeline is removing. And
   `LocalQueueCandidates.current_tracks` is appended to and never emptied, so a
   track last seen 300,000 frames ago is still a candidate.

The second is why uncapping alone is not enough: it removes *"there was no id
left for the real fish"* and leaves *"the id was put on the wrong instance"*.

## `c24` — stage A, uncapped

`c16`'s tracker, same nodes, same scoring, same window, with two changes:
`max_tracks=None`, and `candidates_method='local_queues'` stated explicitly so
the auto-switch cannot make the record wrong again. The tracker's *effective*
configuration is read back off the object and recorded.

Chunked 30 ways per camera and run in parallel — the tracker is sequential
within a camera and single-threaded, 56 minutes per camera measured. Each chunk
starts 500 frames early and discards the warm-up. `--check-chunking` runs one
chunk at the stated warm-up and at four times it and compares the **partition**
(ids are not comparable between runs): **0 of 35,580 instances differ**, so the
parallel run is equivalent to the serial one away from the boundaries. Label ids
do not cross a boundary, which costs 29 renames per camera and nothing else —
`c25` charges one lambda per slot for a rename, so a boundary cannot create a
swap, only a break the sequence model pays for once.

| | wall clock | core time | labels created | of which boundary renames | unlabelled |
|---|---|---|---|---|---|
| three cameras | **460 s** | 13,002 s (28.3x) | 139 / 136 / 135 | 87 each | **0** |

Every instance `c16` left unlabelled now carries a label: 7,360 + 5,200 + 2,927.
The reflection scene has **0 unlabelled instances**.

**The extra labels are not a fourth animal.** On `side1_cam` 100,000–120,000 the
label sets {0,1,3}, {0,2,4}, {1,2,3}, {0,2,3}, {1,2,4} occupy **12.6%** of
frames against **0.91%** where four labels are genuinely present at once. They
are handovers: one label ending and another continuing the same animal.

## `c24b` — cutting links an animal cannot make

One rule, in 2D only: a link between consecutive appearances of a label `dt`
frames apart is cut when a tracked node moves more than `gate_px * dt`. Cutting
relabels; nothing is deleted, no instance loses its label, and the count of
labelled instances is unchanged — all three asserted.

`gate_px` is `s10`'s threshold, computed by `s10`'s own functions from the same
human labels: half the median mouth–tail span of `labels.round8.<view>.train`,
rounded up to 10 px. **60 px** for `top_cam`, **50 px** for the sides.

| | links | cut | labels | one-frame move p99.9 | max |
|---|---|---|---|---|---|
| `top_cam` | 1,038,717 | **807** (0.078%) | 139 → 946 | 34.1 px | 1,069 px |
| `side1_cam` | 1,025,109 | **721** (0.070%) | 136 → 857 | 30.0 px | 1,194 px |
| `side2_cam` | 1,025,941 | **678** (0.066%) | 135 → 813 | 27.0 px | 1,200 px |

**2,206 physically impossible links in one experiment.** The largest crosses
1,200 px in one frame at 166 fps. The gate sits four times above the p99.9 of
what a fish actually does, and still finds these.

This stays in 2D deliberately. The 3D residual would find the same links and
more, but `c16` exists to be *an independent series that can refuse a 3D
reassignment*; cutting 2D tracklets on 3D evidence spends that independence to
buy a cleaner input, and the refusal is worth more.

## `c25` — the correspondence over a variable number of labels

**Alphabet: the labels the frame holds, and nothing else.** Keeping recently
seen labels was measured and rejected before the tool was written — the mean
state count per frame goes 66 (present only) → 11,523 (present within 25 frames)
→ 27,839 (within one second), because two states differing only in which
*absent* label a slot points at cost exactly the same and are still counted
separately. Identity across a gap is carried by the **slot**, which is what the
sequence model is for.

**State: an injective, maximal assignment of the present labels to three slots.**
Per view `perm(k, 3)` for `k >= 3` and `perm(3, k)` below: 1, 3, 6, 6, 24, 60 for
k = 0…5. At `k = 3` that is 6, exactly `c19`'s six permutations of one view. The
frame's state is the product over the three views — 216 on an ordinary frame,
against `c19`'s 36 fixed states, the factor 6 being the slot naming `c19` pins to
`top_cam` and this cannot, because a slot's identity has to survive `top_cam`
changing which labels it holds. Measured over K_1_2: **median 216, mean 250,
p99.9 3,456, max 172,800**.

**Cost: a fixed denominator.** `c20` found that a mean over the terms a state
happened to support lets a state that explains less win, and its state changes
tripled. Here the denominator is `n_slots x n_nodes` and a (slot, node) no two
cameras support costs `mu` millimetres — the price of leaving an animal
unexplained, in the same unit as the residual. `mu = 4.0 mm`, the trough of the
residual histogram `c23` measured.

**Penalty: per slot**, not per state. A handover changes one slot; charging it as
a whole-state change is what makes handovers look like correspondence errors.

    J(s) = SUM_t SUM_a cost_t(s_t,a)  +  lambda * SUM_t #{a : s_t,a != s_t-1,a}

Solved exactly by Viterbi over per-frame state lists. Frames past 2,000 states
are **pruned losslessly**: the penalty is at most `3 lambda` per transition, so
replacing a state by the frame's cheapest costs at most `6 lambda` entering and
leaving, and a state more than `6 lambda` above the cheapest can never lie on the
optimal path. On the frames it fires, 3,456 states fall to **96**; over K_1_2,
172,800 fall to at most 30. The objective is bit-identical with and without it.

### The reduction, asserted

`--check-reduction` runs on `c16`'s capped labels. On the 17,830 frames holding
three labels in all three views and supporting all six terms, both objectives are
a mean over the same six terms, so this enumeration must reproduce `c20`'s
`label_costs`:

| | |
|---|---|
| max cost gap | **1.846e-06 mm**, against a *derived* tolerance of 6.412e-05 mm |
| winner disagreements | **0** (ties counted as ties: 0) |

The tolerance is computed from `label_costs` storing float32 at the observed
magnitudes rather than chosen, because a chosen tolerance hides exactly this
class of bug. The winner — the *set* of slot label triples — is compared
outright, since float32 cannot move it except at a tie.

## What it does to identity

`c09`, 298 intervals of the female's human annotation, K_1_2. Gate held at
`c19`'s 4.0 mm / 73 frames for every arm, which is post-hoc and recorded as such:
a gate derived on a reconstruction that resolves 98.7% of frames collapses to
`max_gap_frames = 1`, and comparing two reconstructions through their own gates
compares gates as much as methods.

| labels | lambda (mm) | preserved | break | **swap** |
|---|---|---|---|---|
| `c24` | **0** | **272** | **26** | **0** |
| `c24` | 0.1 | 270 | 28 | **0** |
| `c24` | 0.25 | 270 | 28 | **0** |
| `c24` | 0.5 | 269 | 29 | **0** |
| `c24` | 0.767 | 269 | 29 | **0** |
| `c24` | 2 | 263 | 34 | 1 |
| `c24` | 8 | 258 | 37 | 3 |
| `c24b` | 2 | 265 | 32 | 1 |
| **`c21` gated-partial-unbounded** (incumbent) | 0.767 | 260 | 37 | **1** |
| `c03` baseline | — | 269 | 19 | 8 |

**Swaps are 0 everywhere in `lambda` ∈ [0, 0.767] mm**, and the preserved count
varies by three intervals across that range — indistinguishable at this sample
size. Past 0.767 mm it degrades. `c20`'s cross-validated 0.767 mm sits exactly at
the top of the safe range, and `kappa = 0.2371` was returned by both `c19` and
`c20` on cost scales 3.6x apart, which is what `kappa` was introduced to do.

**Reconstruction quality**, same runs: resolved **98.6–98.8%** against `c20`'s
98.55% and `c19`'s 97.53%; three cameras on every animal 88.6%.

**Slot changes behave as the design intends.** Over 348,600 frames, as lambda
goes 0.5 → 8 mm, one-slot changes barely move (12,380 → 12,642) while two- and
three-slot changes — genuine correspondence changes — fall from 1,940 to 1,027.
The penalty suppresses correspondence changes and does not charge for handovers.

### The `C25-label` arm fails

Assigning identity = slot with no 3D linker gives **88–93 swaps and 0 breaks** at
every lambda. Identity does not survive without the linker; what the linker buys
is converting 26–37 of those into breaks, which are visible to every automatic
metric, instead of swaps, which are not.

## What this establishes, and what it does not

**Establishes.** Removing the cap and generalising the cardinality did not break
the incumbent, and on all three counts it is at least as good: preserved 272
against 260, breaks 26 against 37, swaps 0 against 1. 15,487 instances that had
no label now have one. 2,206 impossible links are cut. The enumeration reduces
exactly to `c20` where the two are comparable.

**Does not establish.** Swap 0 against 1 is a one-interval difference on 298
intervals, **n_groups = 1, n_experiments = 1, female only**. `c09` and every
decision since `c21` say the same thing: swap counts of 1–8 do not rank anything
stably. The 12-interval difference in `preserved` is larger but is still one
experiment. Nothing here justifies promoting `c25` over `c19`/`c20`; it justifies
running all of it on more than one experiment.

## Corrections to the record

* **`c16`'s provenance names a candidate maker that did not run.** It records
  `candidates: fixed_window`; `sleap_nn` switched it to `local_queues` because
  `max_tracks` was set. Any statement about `c16` that rests on the candidate
  method is unsupported.
* **`c23`'s reflection scene is 113,374–113,383, ten frames, not five.** The
  four-detection run starts at 113,374; `c23` reports 113,378–113,383 because
  that is where its residual flag fires.
* **A triple containing a reflection does not map outside the tank.** The
  expectation is reasonable and the data refuses it: at 113,378 the residual is
  33.90 mm and the point is at (167.0, 140.3, 147.8) mm, inside the 192 x 192 x
  186 mm volume. Least squares over three rays with two of them right pulls the
  solution inside; the disagreement shows up in the residual, not the position.
  Over the whole recording only **182 animal-frames** land more than 10 mm
  outside, of which 101 have a residual under 4 mm — so the tank test finds a
  small population the residual misses, and misses the 7,813 the residual finds.
  It is worth adding as a *secondary* check, not as the primary one.

## What is not done

* The kappa selection ported into `c25` (`sweep_kappa`) returns 0.75–1.33 where
  `c19` and `c20` both returned 0.2371, and the values it prefers are in the
  degrading half of the sweep above. Two bugs in it were found and fixed — the
  margin median was taken over all frames rather than the frames a node
  discriminates, and it was computed from the first block rather than the whole
  window — and it is still not trusted. **The lambda used above is post-hoc.**
  Given that swaps are 0 across a 7-fold range of lambda, this does not affect
  the conclusion, and fixing it is not urgent.
* `c25` has not been run on `c16`'s labels end to end, so the "same
  correspondence, different stage A" comparison is missing.
* `C25-label`'s 88–93 swaps have not been compared against `c19`'s and `c20`'s
  own label arms.
* One experiment. K_1_1 and K_1_3 are cached and were not touched.
