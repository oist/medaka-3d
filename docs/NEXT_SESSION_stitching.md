# Next session — join 974 segments into three individuals

Handoff from 2026-08-18, second pass. Read
[2026-08-18 viterbi correspondence](decisions/2026-08-18-viterbi-correspondence.md)
first — it holds stage B's rule, its results, and the reason the task below is
smaller than it was this morning. Then
[2026-08-18 2D first](decisions/2026-08-18-2d-first-tracking.md) for the four
clauses and the three retired designs.

## Where the problem now sits

`gated` mints a new track ID whenever it cannot link, and nothing merges the
fragments. That has not changed: **475 tracks for 3 fish, largest three covering
15.18%** of the recording, and the pipeline still does not output three
trajectories.

(The previous handoff quoted 15.6% for the same run. It is the same measurement
over a different denominator — 15.18% of all frames is 15.56% of the 97.53% that
resolve. Clause 5 now names its denominator, which is why it names two
definitions.)

What changed is *what is left to do*. Stage B (`c19`) solves the cross-view
correspondence as one piecewise-constant sequence over the whole recording,
exactly, by Viterbi over 36 states × 348,600 frames. The result is **974
segments, median 349 frames**, and inside a segment the correspondence is
constant — which means **no camera relabelled inside it**, because a single
camera relabelling changes the state and ends the segment. Identity is therefore
internally consistent within a segment.

**The remaining problem is 973 joins.** Not 348,600 frames, not 475 fragments.

## The two things stage B settled that change the next step

**1. The correspondence was not the bottleneck.** Stage B did exactly what it
was designed to do — 973 changes instead of 1,941 independent per-frame flips,
no lag, no window — and **every downstream clause came out unchanged**:
swaps 2 (equal to `c18`), coverage 97.53% (identical), trajectory coverage 14.67%
against `c18`'s 15.18%. Four designs in a row have now attacked the
correspondence. The fragmentation comes from `gated` refusing links at its 4 mm
distance gate, not from the correspondence flipping. **Do not build a fifth
correspondence method.**

**2. The coverage the science needs is already available; the naming is what is
wrong.** The `B-label` arm names identity by the `top_cam` 2D label and keeps it
across every segment boundary — the null join rule. It reaches **97.53%
trajectory coverage with three tracks**, and costs **50 swaps** against the
female annotation. So:

| arm | trajectory coverage | swaps |
| --- | --- | --- |
| `c18` / `c19` B-gated, via `gated` | 14.67–15.18% | **2** |
| `c19` B-label, null join rule | **97.53%** | 50 |
| what is needed | ~95% | < 8 |

Both ends exist. Nothing in between does. **That gap is the next task.**

## What to build — stage D, the join rule

At each of the 973 segment boundaries, decide which identity in the outgoing
segment is which identity in the incoming one. Three animals, so six
possibilities per boundary; 973 boundaries. Evidence available at a boundary and
not used by either arm above:

* **3D motion continuity across the boundary** — position and velocity of each
  of the three animals on both sides. `gated`'s job, but applied at 973 known
  places instead of at every frame, and with a whole segment's motion on each
  side rather than one frame's.
* **Which camera actually relabelled.** A boundary is a change in
  `sigma = (sigma1, sigma2)`, and it decomposes exactly. Write
  `tau = sigma2 . sigma1^-1` for the side1 -> side2 map. A relabel by `pi` in
  `top_cam` right-composes both: `sigma_v -> sigma_v . pi^-1`, which leaves
  `tau` fixed. A relabel in `side1_cam` moves `sigma1` and `tau` and leaves
  `sigma2` fixed; in `side2_cam`, `sigma2` and `tau` move and `sigma1` is fixed.
  So the quantity that stays fixed across a boundary names the camera that
  relabelled, and the other two cameras' labels then carry identity across it
  unaided. **This is the cheapest evidence in the problem and nothing has used
  it yet.**
* The 2D tracker's own behaviour at the boundary frames.

The state array is saved: `state` in both `c19` npz files, `(348600,) int16`,
values 0–35 indexing `medaka3d.reconstruct.hypotheses(3)`.

Start with the second bullet alone — it may be most of the answer, it costs
almost nothing, and it produces a falsifiable prediction (a boundary where all
three cameras appear to have relabelled at once is a boundary the rule cannot
explain, and how many of those there are is a measurement).

## What decides

The four clauses in [the 2D-first decision](decisions/2026-08-18-2d-first-tracking.md),
plus clause 5, both restated with their measured baselines in
[the stage B decision](decisions/2026-08-18-viterbi-correspondence.md). Score
with `tools/c09_gt_swap_validation.py --npz <output>`.

1. **Swaps < 8** on the `c09` unit. An increase disqualifies regardless of
   everything else. Current best: **2** (`c18` and `c19` B-gated, indistinguishable).
2. Superseded by clause 5 — track counts across runs use different derived gates
   (10.2 / 8.3 / 5.0 / 4.0 mm) and compare gates as much as methods.
3. **Tail-derived cross-view agreement**, both pairs, against `c03`'s
   0.9668 / 0.9686. Use the tail, not the pectoral: `c13`, `c18` and `c19` all
   *solve* on the pectoral, so pectoral-derived agreement is partly construction
   for them. Current best: 0.9674 / 0.9694.
4. **Reconstruction coverage >= 96.31%.** Current: 97.53%.
5. **Per-animal trajectory coverage**, two definitions, never mixed:
   `top3_track_coverage` (frames held by the three largest IDs / 3T) and
   `top3_run_coverage` (frames in runs >= 1 s / 3T). Current 14.67% / 92.62%;
   `B-label` shows 97.53% / 93.68% is reachable; `c16`'s 2D standard is 94.6–97.0%
   on the run definition. `tools/c19_viterbi_correspondence.py:coverage` computes
   both — use it rather than re-deriving, so arms stay comparable.

## Traps — the old six still apply, plus four from today

Everything in the previous handoff's trap list is still true. Re-read it in git
history if needed; these four are new.

* **`c09` writes to a fixed path.** Still true and still the easiest way to break
  `c11`/`c14`/`c15`. Run it with `--npz` for each arm, then **once with no
  `--npz`** to restore. Verified restored today: 296 intervals, 269 / 19 / 8.
* **Slots are indexed by label in `c19` and by detection row in `c18`**, so the
  same correspondence lists its slots in a different order. Compare with
  `c19.slot_signature`, which sorts the per-slot view triples. Element-wise
  comparison reports a disagreement that is not one — it cost the first run of
  the λ = 0 check.
* **Two orders of the same detections exist and are not interchangeable.**
  `by_label` for the solver, `by_row` for the npz, because `c04.node_pairing` and
  `c09.female_anchors` both read detector order. `c19.label_ordered` returns both
  and the npz is written from `by_row`.
* **A margin can change by a unit and look like a regression.** `c19` reports the
  runner-up margin per contributing node where `c18` reported the sum; the ratio
  is exactly 2.000 on 90.2% of frames and 1.000 on the rest. Check the ratio
  before reading a margin change as a result.

## Rules that held today and should hold next time

* **Write the decision rule before the code, and say so when it fails.** The λ
  rule handed over — sweep until the change count matches 635 — is *provably*
  unsatisfiable: the change count is non-increasing in λ and already equals the
  unconstrained count at λ = 0. That was derived and written down before
  implementing, and the replacement (held-out node cross-validation) was
  pre-registered with its own fallback. The fallback was not needed; both curves
  had interior minima.
* **A number whose tool has been deleted is not a number.** Nothing reproduces
  635 / 284,305. The same quantity on the current state space is **1,941 /
  340,001 (0.5709%)**, 2.6× larger, and neither is evidence about the other.
* **Assert the invariant that would fail silently.** `c19` refuses to run unless
  λ = 0 reproduces `c18`; it does, on 100.0000% of 340,001 frames with evidence.
  That single check covers the label ordering, the row map and the state
  enumeration at once.

## Constraints that do not go away

**n_groups = 1, n_experiments = 1, female only.** K_1_2 is the only experiment
with identity annotation. Male-vs-male crossings are unmeasurable. **Swap counts
of 2–8 are small samples and no ranking built on them is stable** — `c18`'s 2
and `c19`'s 2 are the same number, and 2 against `c13`'s 4 is not a result.
B-label's 50 is far enough outside that range to be a real disqualification, and
it is also not comparable outcome-for-outcome: with three permanent IDs, B-label
cannot produce a `break`, so every identity error there is scored as a swap.

## Tools and modules

Added today: `c19` viterbi correspondence.
New in `medaka3d.correspond`: `StateCosts`, `state_costs`, `viterbi_sweep`,
`viterbi_path`, `reconstruct_from_states`.
Unchanged and still production: `c03`.
`ruff check src/ tools/ tests/` and `pytest -q` are green; `tests/test_correspond.py`
covers the sequence solver, including that λ = 0 is the per-frame winner and that
a large λ collapses to one state.
