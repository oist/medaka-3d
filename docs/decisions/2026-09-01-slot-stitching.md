# Stage D, reduced: re-permuting slots where 3D continuity is decisive

**Date:** 2026-09-01. **Written before the run.** Only "Result" is appended
after. **Status: measurement only.** Nothing is promoted; `c03` still owns the
fixed output paths; no existing tool or output is modified. New tool `c30`
reads `c28`'s stored arms and writes under `c30_slot_stitch/`.

**Follows:** [2026-08-26 transfer to K_1_1/K_1_3](2026-08-26-transfer-to-k11-k13.md).
**Revives:** the stage-D join rule of
[`NEXT_SESSION_stitching.md`](../NEXT_SESSION_stitching.md) (2026-08-18), which
framed 973 joins and was never built.

## The observation that triggered this

The experimenter watched `c29`'s full-length render of K_1_2 (lambda 0.767) and
reported that the 3D reconstruction is close to right, but the three plotted
identities show **large switches** — trajectories exchanging positions in a
step. That is the slot assignment permuting, and it was measured before any
code was written:

* **416 adjacent resolved pairs** where a non-identity permutation of the three
  slots is cheaper than identity by > 5 mm; they cluster into 166 episodes
  ≥ 1 s apart, of which **139 leave a net permutation** — a lasting switch
  about every 15 s of recording.
* At those moments the minimum inter-animal distance is **median 21 mm, p90
  104 mm**. Most switches happen with the animals far apart, where 3D
  continuity identifies the join with no ambiguity.
* Separately, 448 slot-steps > 10 mm where identity *is* the best permutation —
  triangulation spikes, the "jagged" parts of the same report. **Not identity
  errors, not addressed here**, and a join rule must not fire on them.

A greedy single-frame probe (in memory, nothing written) halved the > 10 mm
steps; the residue was oscillation from single-frame comparisons. Hence the
windowed rule below.

## The rule — fixed before the run

At every candidate boundary between adjacent resolved frames (gap ≤ the gate's
73 frames): compare per-slot **median positions over the last 8 resolved
frames** (stitched space) and the **next 8** (raw); score all six assignments;
adopt a non-identity assignment only if it beats identity by **> 5 mm**
summed over slots; otherwise **refuse and record**. Across gaps > 73 frames
nothing is decided (the gate's own abstention). Single-frame spikes cannot move
an 8-frame median. `tests/test_slot_stitch.py` pins all three behaviours on
synthetic trajectories: the injected permutations are found exactly, the spike
does not fire, a close encounter is refused.

The output npz has `c28`'s schema (slots permuted consistently across `xyz`,
`residual_mm`, `slot_rays`, `source`; the **gated tracker re-run** on the
stitched geometry, gate held at 4.0 mm / 73 frames, post-hoc as always;
per-view tracks re-propagated) plus `slot_perm` so every join is auditable.
`state` is dropped: it indexes slot order that this tool changes.

## Reading rules, fixed now

* **All seven lambda arms are stitched.** A claim about the join rule is
  carried only if it holds at every stored lambda; else it is lambda-dependent.
* **`c09` per arm is the disqualifier.** Baseline at lambda 0.767 serial:
  **269 preserved / 29 break / 0 swap**. If any arm's swap count exceeds its
  input arm's, the rule fails regardless of everything else (clause 1 of the
  2026-08-18 handoff, unchanged). Expected direction, stated in advance: some
  breaks become preserved, because the gated tracker previously broke at
  switch frames; swaps stay at their input count.
* The visible symptom is scored as **adjacent-frame same-slot steps > 10 mm
  and > 30 mm**, before and after.
* Refusals are counted and kept: a refused boundary is a break on screen, not
  a hidden guess.
* **Swap counts of 0–8 rank nothing** (every decision since `c21`). One
  experiment, one trio, female-only annotation. Male–male switches are
  unmeasurable; a switch this rule repairs is repaired **geometrically**, and
  no claim is made that the repaired identity is the right animal beyond what
  the female intervals can check.

## `c09` protocol

Run with `--npz` per arm; then **once with no `--npz`** to restore the fixed
baseline, and verify it: `c03` — preserved 269 / break 19 / swap 8
(`c11`/`c14`/`c15` read that file).

## v2, pre-registered 2026-09-03: the evidence ladder

The v1 rule refused four boundaries that the female annotation later confirmed
as real swaps (72143, 76930, 105541, 302762), and walked frame-by-frame
through the ghost stretch at 31479, netting to identity while the clean
endpoints disagree by 334.7 mm. Both faults have the same shape: v1 knows only
one kind of evidence - windowed median positions - and the experimenter's
review identified the missing kind: **the displacement vector against the
0.1-0.3 s mean motion vector**. Measured before this registration: velocity
extrapolation picks the correct swap at all four refused boundaries at every
window in 0.1-0.3 s (margins 2.5-19.8 mm), and the clean-to-clean bridge at
31479 is decisive at 334.7 mm (position) / 338-342 mm (velocity), both
agreeing. Background: of 1,099 steps > 5 mm, 627 point against the recent
heading (cos < 0, suspect) and only 252 align with it (real bursts).

**The ladder, fixed now.** At each boundary, in order; a step decides only
when the one before it could not:

1. **Median rule** (v1, unchanged): 8-frame median positions, join iff a
   non-identity assignment wins by > 5 mm.
2. **Velocity re-evaluation** - only at v1's refusals (non-identity best,
   margin <= 5 mm): extrapolate each slot from its last 50 resolved frames
   (0.3 s; decisive at every window in 0.1-0.3 s on the measured cases) and
   re-score. Join iff the velocity margin > 5 mm **and** it names the same
   permutation the medians preferred; any disagreement stays a refusal.
3. **Zone bridge** - a turbulent zone is a run of degraded frames (any view's
   best score < 1.5, dilated +-2 - the window study of 2026-09-03: detection
   at 1 frame, smoothing destroys capture, +-2 dilation covers interleaved
   deceptive frames; runs merged within 16 frames) that contains at least one
   same-slot step > 20 mm. Inside such a zone no boundary is decided at all;
   one decision is made across it, last clean frame to first clean frame,
   and is applied only when the **position and velocity margins both** exceed
   5 mm **and agree**. Anything else is a recorded refusal and the zone is
   left as it is - a visible break, not a guess.
4. **Refusal** (v1, unchanged), now always recorded, including the previously
   silent skip paths (260434's boundary made no record at all in v1; that was
   a fault).

Reading rules: all seven lambdas; disqualifier unchanged (`c09` gated swaps
may not exceed the production chain's 0/0/0/0/0/1/3 at any lambda); scored as
`c30v2 -> c34` against the production chain's stitched-label 33/4/7/6/5/6/6.
Stated expectation: the four refused female swaps join at step 2 (76930's
measured margin of 4.8 mm at 0.3 s sits at the bar and may stay refused) and
31479 bridges at step 3, taking lambda 0.767 label swaps from 5 toward 1-2.
The head-on "bounce" class - a lasting swap whose median comparison favours
identity outright - is NOT covered by this ladder and is recorded as its
known blind spot.

### v2, scored - NOT carried, and the diagnosis that licenses v3

Chain `c30v2 -> c34`, against production (33/4/7/6/5/6/6): label swaps
**46/16/14/11/11/13/16 - worse at every lambda.** Gated unchanged
(0/0/0/0/0/1/3), so the disqualifier is silent; the label criterion is not.
The mechanism is visible in the join counts: at lambda 0.767 the median joins
fell **369 -> 219**, because the zone rule suppressed every boundary inside
any degraded run containing one big step - and replaced ~125 correct joins
with 14 bridges. The suppression was the error, not the bridge: v2 threw away
working decisions wholesale to protect against the rare ghost stretch.

**v3, pre-registered now: the bridge becomes a post-pass audit, not a gate.**
Steps 1-2 run exactly as v2 (no suppression - every v1 join survives; the
velocity re-evaluation stays as registered). Afterwards, each turbulent zone
(definition unchanged) is audited once: clean endpoint to clean endpoint, in
stitched space. If the zone's interior decisions left the trajectories
continuous, the identity wins the audit **by the same 5 mm margin on both
kinds of evidence** and nothing happens - silence requires the decisiveness a
correction would.
If position and velocity margins both exceed 5 mm and agree on a non-identity
assignment - the interior netted to the wrong side, the 31479 signature - the
correction is applied from the first clean frame after the zone. Anything
else is a recorded, unvouched zone. Same reading rules, same disqualifier,
same comparison target.

### v3, scored - the ladder works case by case, and its two leaks are measured

Chain `c30v3 -> c34` label swaps **32/5/4/3/4/4/6** vs production 33/4/7/6/5/6/6
(better at five lambdas, worse by one interval at 0.1); gated identical. Case
level, against the annotation: the velocity step joined **72143, 105541,
302762 exactly as registered** (median margins 3.78-4.44 under the bar,
velocity 7.3-19.0 over it), 76930 stayed refused at 3.26/3.85 as predicted,
and the zone audit **repaired 31479** (zone 31473-31496, margins 319/326,
identity restored from 31497). Four of the five known female swaps fixed.

**But three new female swaps appeared** (82,121-83,265 / 90,200-90,687 /
212,276-213,061), and their mechanism is in the records: two velocity joins
fired where the median margin was a coin toss (**0.51 and 0.81 mm** - the
successful three sat at 3.8-4.4), and two bridges fired with margins an order
below the successful ones (**9-11 mm** over a 63-frame zone and **5-7 mm**
over a 2,494-frame zone, where constant-velocity extrapolation means nothing -
against **192 and 319 mm** for the two correct bridges).

**v4, pre-registered now - two tightenings, and their provenance stated:**
step 2 additionally requires the median margin itself to reach **2.5 mm**
(half the bar - the medians must lean, not flip a coin); a zone correction
requires both margins to exceed **30 mm** (an order above the boundary bar -
a bridge asserts identity across corrupted geometry, and a wrong one costs an
inter-animal distance, median 95 mm). **Both numbers are calibrated on this
experiment's annotated cases - five velocity joins and four bridges - and are
hypotheses anywhere else**; the experimenter's review of the next render and
the K_1_1/K_1_3 transfers are their test. Same reading rules, same
disqualifier. Expected: the four fixes hold, the three regressions vanish,
lambda 0.767 ends at 76930 alone.

### v4, scored - CARRIED

Label swaps **30/5/4/3/2/2/4** against production 33/4/7/6/5/6/6 - better at
six lambdas, worse by one interval at 0.1 (inside the band the standing rule
says ranks nothing; recorded, not hidden); gated **identical at every
lambda**, so the disqualifier is silent. At lambda 0.767: **296 / 0 / 2**
(99.3% of the 298 female intervals), and the two survivors are both designed
refusals rather than misses - 76,698-77,285 is the 76930 boundary refused at
3.26/3.85 mm exactly as registered, and 130,760-138,892 contains the
2,494-frame chaos zone whose bridge v4 now rejects (margins 7.6/5.4 against
the 30 mm bar) and records as **unvouched** - in v3 that same weak bridge
"fixed" this interval by luck while planting the 212k swap. Steps after `c34`
essentially unchanged (325/197 vs 322/193).

The case-level ledger against the annotation: 72143, 105541, 302762 joined by
velocity; 31479 repaired by the zone audit (margins 319/326); the three v3
regressions (82k, 90k, 212k) gone with the tightened bars. **The production
chain is now `c27 -> c28 -> c33 -> c30(v4) -> c34`.** The v4 thresholds are
calibrated on this experiment's annotated cases and are hypotheses anywhere
else; the standing blind spots are the head-on bounce class, the 76930-type
sub-bar refusal, and long unvouched zones - all three end in the review
queue, not in silence.

## Result

**Appended 2026-09-01, after the run.** Tool `c30`; `tests/test_slot_stitch.py`
green; all seven arms stitched in ~35 s each.

### The join pass itself

| lambda | joins | refusals | steps > 10 mm | steps > 30 mm |
|---|---|---|---|---|
| 0 | 1,556 | 144 | 6,143 → 3,501 | 3,853 → 2,208 |
| 0.1 | 485 | 59 | 1,548 → 1,013 | 840 → 631 |
| 0.25 | 435 | 65 | 1,323 → 857 | 762 → 569 |
| 0.5 | 389 | 54 | 1,227 → 801 | 719 → 551 |
| 0.767 | 376 | 59 | 1,170 → 750 | 674 → 509 |
| 2 | 334 | 48 | 1,118 → 754 | 610 → 459 |
| 8 | 259 | 45 | 1,128 → 842 | 602 → 508 |

Joins exceed the 139 pre-measured net episodes because flips arrive in bursts
the episode clustering collapsed — the densest burst sits in the **handling
period** around frame 3,400, where the render shows the fish being poured from
a cup and no correspondence is meaningful. The steps that remain are dominated
by single-frame triangulation excursions, which the rule correctly does not
touch: they are noise, not identity.

### `c09`, the number that decides

**The slot labelling** — three permanent ids, break impossible by construction,
every identity error a swap (B-label semantics). 298 intervals:

| lambda | unstitched  P/B/S | stitched  P/B/S |
|---|---|---|
| 0 | 218 / 0 / **80** | 263 / 0 / **35** |
| 0.1 | 219 / 0 / **79** | 290 / 0 / **8** |
| 0.25 | 219 / 0 / **79** | 287 / 0 / **11** |
| 0.5 | 221 / 0 / **77** | 288 / 0 / **10** |
| 0.767 | 226 / 0 / **72** | **292 / 0 / 6** |
| 2 | 220 / 0 / **78** | 291 / 0 / **7** |
| 8 | 223 / 0 / **74** | 289 / 0 / **8** |

* **Carried grid-wide:** stitching reduces slot-labelling swaps at every
  lambda. The unstitched labelling swaps the female in 24–27% of intervals —
  the ground-truth measurement of exactly what the experimenter saw in `c29`'s
  render — and the stitched labelling in 2.0–3.7% everywhere except lambda 0.
* **Lambda-dependent:** the residue. Lambda 0 keeps 35 swaps; every other
  arm keeps 6–11. The 2026-08-18 B-label reference was **50 swaps at 97.53%
  coverage**; the stitched labelling at lambda 0.767 is **6 swaps at 98.77%**.

**The gated arm — clause 1.** After stitching, `c09` on the re-run gated
tracker is **identical to `c28`'s stored serial numbers at every lambda**
(272/26/0, 270/28/0, 270/28/0, 268/30/0, 269/29/0, 261/36/1, 257/37/3). No arm
gained a swap; the gated tracker re-solves its per-frame assignment and is
permutation-invariant on this data to the last interval. The stitch therefore
adds a usable 3-identity labelling **without touching the zero-swap instrument**.

**Baseline restored and verified:** `c09` with no ``--npz`` reads `c03` —
preserved 269 / break 19 / swap 8.

### What remains open

* **6 swaps at lambda 0.767** (2.0% of intervals) survive: switches the rule
  refused or mistimed, presumably during close encounters. 59 refusals are
  recorded with frames and margins in `c30_summary.json` for review — new
  reviewed scenes (`c22`) at those frames are the way this number shrinks
  honestly.
* The repaired identity is repaired **geometrically**. The female intervals
  confirm it 292 times out of 298; male–male switches remain unmeasurable.
  `n_groups = 1`, one experiment, one trio.
* Nothing is promoted. `c03` still owns the fixed output paths.
