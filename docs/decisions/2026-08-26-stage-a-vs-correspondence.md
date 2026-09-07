# Which change carried the result — stage A, or the correspondence?

**Date:** 2026-08-26. **Written before the run, and not edited afterwards
except to append the measured numbers under "Result".**
**Status:** measurement only. Nothing is promoted. `c03` still owns the fixed
output paths. No existing tool or output is modified.
**Follows:** [2026-08-25 uncapped tracklets](2026-08-25-uncapped-tracklets.md),
step A of [`NEXT_SESSION_uncapped_followup.md`](../NEXT_SESSION_uncapped_followup.md).

## The question

`c24` (stage A without the `max_tracks = 3` cap) and `c25` (a correspondence
over a variable number of labels) were introduced in the same session and
measured only together. The K_1_2 result — 272/26/0 at `lambda = 0`, 269/29/0 at
`lambda = 0.767` — is therefore attributed to nothing in particular.

## The design, and why it is an L rather than a square

| stage A | correspondence | status |
|---|---|---|
| `c16` (capped) | `c20` (36 fixed states) | measured: `c21` gated-partial-unbounded, **260 / 37 / 1** at `lambda` 0.767 |
| `c16` (capped) | `c25` (variable cardinality) | **this run** |
| `c24` (uncapped) | `c25` (variable cardinality) | measured: **269 / 29 / 0** at `lambda` 0.767 |
| `c24` (uncapped) | `c20` | **not constructible.** `c20`'s state space is the 36 permutations of three labels per view; uncapped labels break that premise. This is why the fourth cell is absent by design and the `c16`/`c25` cell is absent by omission. |

Everything except stage A's labels is held: `mu = 4.0` mm, nodes
`pectoral, mouth` with the pectoral as anchor, `lambda = 0.767` mm, linker
`gated` with `partial`, gate **4.0 mm / 73 frames**.

**`lambda` and the gate are both POST-HOC and are recorded as such by the tool.**
`lambda = 0.767` is `c20`'s cross-validated value, carried over rather than
re-derived, because `c25`'s ported `sweep_kappa` is not trusted (it returns
0.75–1.33 where `c19` and `c20` independently returned `kappa = 0.2371`). The
gate is held at `c19`'s because `derive_gate` takes the 99.9th percentile of the
3D step, and a step across a correspondence change is a **renaming**, not a
movement — a derived gate is a property of its own reconstruction and cannot be
compared across two.

## How the answer will be read — fixed before the numbers exist

Scored by `c09` on the 298 confirmed female intervals of `female_sex_gt.csv`,
arm `C25-gated`. The comparison is against the two cells above.

* **The correspondence carries it** if `c16`+`c25` lands at swap 0 with
  `preserved` near `c24`+`c25`'s 269. Future effort belongs in stage C.
* **Stage A carries it** if `c16`+`c25` lands near `c21`'s 260 / 37 / 1.
  Future effort belongs upstream, in the detections and the 2D tracker.
* **Both contribute** if it lands between the two. Report it as that and do not
  rank the two changes on this data.

No threshold is being chosen, because none can be. **Swap counts of 0–8 rank
nothing** — `n_groups = 1`, `n_experiments = 1`, female only, 298 intervals in
the only experiment that has identity annotation. Differences in `preserved` are
reported as interval counts carrying the same caveat. **Zero swaps is the floor
of the instrument and no arm will be claimed to beat it.**

Recorded alongside, as description rather than as a criterion: `resolved`,
mean states per frame, slot label changes, and the count of unlabelled
instances each stage A leaves (`c16` 15,487; `c24` 0).

## Protocol

`c09` writes to a fixed path that `c11`/`c14`/`c15` read. It is run with
`--npz` for the arm, then once with no `--npz` to restore the `c03` baseline
(preserved 269 / break 19 / swap 8), and the restoration is verified before
anything is reported.

## Result

`c25` on `c16`'s labels, `lambda = 0.767` mm, gate held at 4.0 mm / 73 frames,
arm `C25-gated`, scored by `c09` on the 298 confirmed female intervals:

| stage A | correspondence | preserved | break | **swap** |
|---|---|---|---|---|
| `c16` capped | `c20`, 36 fixed states (`c21` incumbent) | 260 | 37 | **1** |
| **`c16` capped** | **`c25`, variable cardinality** | **256** | **42** | **0** |
| `c24` uncapped | `c25`, variable cardinality | 269 | 29 | **0** |
| `c03` baseline | — | 269 | 19 | 8 |

**The two columns answer differently, and only one of them can answer at all.**

* **swap.** `c16`+`c25` is already at zero, so the swap column reads as "the
  correspondence carries it". It cannot be believed: the difference it rests on
  is **1 interval out of 298**, and every document since `c21` records that swap
  counts of 0–8 rank nothing. This column separates nothing.
* **preserved / break.** Here the answer is legible and it points the other way.
  Changing the correspondence alone **costs** 4 preserved intervals
  (260 → 256) and adds 5 breaks. Changing stage A as well recovers 13 and
  removes 13 (256 → 269, 42 → 29). **The gain in the 2026-08-25 result is
  attributable to the uncapping, not to the variable cardinality.**

The pre-registered third reading — "both contribute" — is what the data says,
with the contributions in opposite directions on the only column that can carry
a difference this size.

### The residual tail settles it independently, and more sharply

Recomputed on the stored arrays over three-camera animal-frames
(`slot_rays == 3`), the same way for both rows, ~990,400 animal-frames each:

| stage A | correspondence | median | p99 | p99.9 | max | > 4 mm |
|---|---|---|---|---|---|---|
| `c16` capped | `c25` | 0.466 | 2.349 | **72.56** | 107.1 | **0.797%** (7,894) |
| `c24` uncapped | `c25` | 0.464 | 1.236 | **3.57** | 88.4 | **0.088%** (871) |

This procedure reproduces the stored **871** of the 2026-08-25 record exactly,
so the two rows are comparable. `c16`+`c25`'s p99.9 of 72.56 mm sits with
`c21`'s 72.58 — **the contaminated tail is untouched by the correspondence and
collapses 20x under the uncapping alone.** No 36-state-versus-variable
difference is visible in it.

This is the expected result and it was named in advance: the correspondence's
state space is built on stage A's labels, so **a labelling error is unreachable
from stage C**. When a reflection holds a label, no enumeration over
assignments of that label can put the animal back.

### Recorded alongside, as description

| | `c16`+`c25` | `c24`+`c25` |
|---|---|---|
| resolved | 98.75% | 98.77% |
| three cameras on every animal | 88.60% | 88.6% |
| states per frame, mean | 214 (max 216) | 250 (max 172,800) |
| slot label changes | 16,674 (1 slot: 10,894) | 16,114 (1 slot: 12,447) |
| gate `derive_gate` would have set | **69.3 mm** / 65 frames | 5.7 mm / 65 frames |
| instances stage A leaves unlabelled | 15,487 | 0 |

The derived gate is the same diagnostic read a third way. On `c16`'s labels
`derive_gate` still asks for 69.3 mm in a 192 mm tank — the p99.9 of the 3D step
is a **renaming**, not a movement — and under the uncapping it falls to 5.7 mm,
near the 4.0 mm held here. Holding the gate remains post-hoc and remains the
controlled comparison, but the derived value is now evidence about the input.

`c25` on `c16` also runs the state space at its floor: mean 214 of a maximum
216, because a capped view never holds four labels. The generality `c25` was
built for is only exercised once the cap is gone, which is the same statement
as the rest of this section.

### What this does not establish

`n_groups = 1`, `n_experiments = 1`, **female only**, 298 intervals, one trio of
fish. The 13-interval difference in `preserved` is the largest signal here and
it is still one experiment. Swap 0 is the floor of the instrument and is not
claimed as an improvement over swap 0. Nothing is promoted; `c03` still owns the
fixed output paths, and `c09`'s fixed output was restored to the `c03` baseline
(269 / 19 / 8) and verified after the arm was scored.

### Consequence for what comes next

Effort belongs **upstream, in stage A and the detections**. `c25` is not thereby
useless — it is what makes an uncapped stage A usable at all, since `c20`'s
36-state space cannot accept more than three labels per view — but it is an
enabler, not the source of the gain. Any further refinement of the
correspondence has to be argued on something other than these 298 intervals.

---

## Addendum, later the same day — a confound was found in this comparison, and then retired

**Appended 2026-08-26.** The conclusion above is unchanged. The route to it was
not sound when written, and both halves of that are worth recording.

**The confound.** This document compared `c16`'s labels with `c24`'s and
attributed the difference to the cap. `c16` ran the tracker **serially**;
`c24` ran it in **30 parallel chunks**. `c24`'s claim that the chunked run is
equivalent to a serial one rested on one measured cell, and that cell turned out
to be one of the few that pass — see
[the transfer document](2026-08-26-transfer-to-k11-k13.md), part one. So the
comparison mixed two changes, exactly the fault this document existed to fix one
level up.

**The measurement that retired it.** Stage A was re-run serially on all three K
days (`c27`) and the whole `lambda` grid re-solved on those labels (`c28`). The
two stage-A runs disagree on **about half** the instances when compared within a
chunk — 47.15% / 46.63% / 53.82% by view — and produce **the same
reconstruction**: `resolved` equal to four decimal places at every `lambda`,
p99.9 within 0.09 mm, and `c09` identical at five of the seven `lambda` values
and within two intervals at the other two. **272 / 26 / 0 reproduces exactly on
a serial run.**

So the comparison in the table above, with both arms now serial:

| stage A (both serial) | correspondence | p99.9 mm | > 4 mm | `c09` at lambda 0.767 |
|---|---|---|---|---|
| `c16` capped | `c25` | **72.56** | **7,894** | 256 / 42 / 0 |
| `c27` uncapped | `c28` | **3.56** | **869** | **269 / 29 / 0** |

**Why the labelling can change by half and change nothing.** `c25`'s alphabet is
*the labels a frame holds*, and identity across frames is carried by the
**slot**. Two stage-A runs that disagree about which label a fish keeps over
time, but agree about which detections belong to which animal *within* a frame,
present the correspondence with the same problem. That invariance is why the
slot exists, and it had never been tested until the chunking failure forced it.
