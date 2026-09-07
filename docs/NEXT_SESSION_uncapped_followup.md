# Prompt for the next session — after the cap came off

**Written:** 2026-08-26, at the end of the session that produced `c22` and `c23`
and reviewed `c24`/`c24b`/`c25`.
**Read first:** [`decisions/2026-08-25-uncapped-tracklets.md`](decisions/2026-08-25-uncapped-tracklets.md),
then [`TRACKING_PIPELINE.md`](TRACKING_PIPELINE.md) for the pipeline as a whole.
**Status: nothing is promoted.** `c03` still owns the fixed output paths.

## The one fact that reorders everything

On K_1_2 the new arrangement scores **272 preserved / 26 breaks / 0 swaps** for
every λ in [0, 0.767] mm. The old incumbent (`c21`) was 260 / 37 / 1.

**Zero is the floor of the instrument.** `female_sex_gt.csv` gives 298 anchor
intervals in the only experiment with identity annotation, and no refinement can
be shown to beat zero swaps on them. The question has stopped being *which
improvement next* and become *how would we know it was an improvement*.

Everything below follows from that, including the decision to defer work that is
correct.

## What the uncapping did to the ray-level plan — measure, don't re-derive

A plan written on 2026-08-24 proposed three changes: drop the offending ray
rather than the whole animal-frame; weight terms by view × instance × node; and
route computation by doubt. Measured again after `c24`/`c25`:

| reconstruction | median | p99 | p99.9 | > 4 mm | 3-cam > 4 mm |
|---|---|---|---|---|---|
| `c21` gated-partial | 0.460 | 2.154 | **72.58** | 0.761% | **7,513** |
| `c25` on `c24`, λ 0.767 | 0.459 | 1.375 | **4.00** | **0.100%** | **871** |

The contaminated tail fell 18× at p99.9 and the target population fell 8.6×.
`residual_mm` is **not** clipped in the new output — the survivors still reach
88 mm — so this is a real collapse, not an artefact of the new cost. The
reflection scene reads 0.77–0.89 mm across all ten frames where it used to run
33–34 mm.

The mechanism survives at the smaller scale: of the 871, **93.1%** are still
fixed by dropping exactly one ray, the culprit is sharply identified on 72.1%,
and it is `top_cam` 47.4% of the time. **The idea was right and there is now a
tenth as much of it to be right about**, in a reconstruction that commits no
swaps. It is deferred, not refuted.

## The work, in order

### A. Run `c25` on `c16`'s labels, end to end — do this first

Two things changed at once — stage A lost its cap (`c24`) and the correspondence
gained variable cardinality (`c25`) — and nothing separates their contributions.
`work/c25_variable_cardinality/c16/` exists with a summary and `chosen.npy` but
**no arms**, so the cell is missing rather than absent by design.

```
tools/c25_variable_cardinality.py --labels c16 --lam 0.767 --gate-mm 4.0 --max-gap-frames 73
tools/c09_gt_swap_validation.py --npz <the C25-gated npz>
tools/c09_gt_swap_validation.py          # no --npz: restores the baseline
```

Cheapest question, largest answer. If the uncapping carries the result, future
effort belongs upstream in stage A; if the correspondence carries it, the
opposite.

### B. Make λ derivable again — and it is not optional

`c25`'s ported kappa selection (`sweep_kappa`) returns 0.75–1.33 where `c19` and
`c20` both returned **0.2371**, and the values it prefers sit in the degrading
half of its own sweep. Two bugs were found and fixed and it is still not trusted,
so **every λ in the decision document is post-hoc.**

The decision document rates this "not urgent" because swaps are zero across a
sevenfold range of λ. **That argument holds only on the one experiment where the
answer can be checked.** On K_1_1 and K_1_3 there is no `c09` to run, so λ cannot
be confirmed after the fact — it has to be derived, or the run is unfalsifiable.
**B is the prerequisite for C, not a cleanup item.**

This is a debugging task with an oracle: κ = 0.2371 was returned independently by
`c19` and `c20` on cost scales 3.6× apart, which is exactly what κ exists to do.
A selector that disagrees with both is wrong against a known answer.

### C. Run the whole pipeline on K_1_1 and K_1_3

Both are cached by `c01` and untouched. State plainly what this can and cannot do:

* **Cannot** score swaps. `female_sex_gt.csv` covers K_1_2 alone.
* **Can** test transfer: does κ land in the same place, does the mean state count
  stay near 216, does the residual tail stay collapsed, does `resolved` hold near
  98.7%, do the per-camera label counts stay near 135–139 (and 813–946 after
  `c24b`).
* **Can** generate new `c22` review packs — the only way the evidence covering
  **males** grows, since the fourteen scenes read on 2026-08-24 are all of it.

**The caveat travels with any result:** K_1_1, K_1_2 and K_1_3 are the same three
fish on three consecutive days. Agreement across them is repeated measures, not
replication. Generalisation means a different group, and 23 of the 24 trials have
never been analysed.

### D. The doubt score — promoted, because C changes what it is for

Both inputs are already in every output file and cost nothing: `margin_mm` and
`residual_mm`. The 2026-08-24 plan ranked this last because there was nothing
expensive to route *to*.

Step C changes that. **Where there is no ground truth, a calibrated per-frame
confidence is not a router — it is the instrument.** It is how a run on K_1_1 can
be called good or bad at all, and the behaviour analysis will need it regardless:
every trajectory it consumes should arrive with a number saying how much to
believe it.

Calibrate on the 298 female intervals and the reviewed scenes: does a frame the
score calls settled in fact carry no identity error?

### E. Deferred, with one piece worth carrying

**Ray rejection** would act on 871 animal-frames, 0.084% of the recording, in a
reconstruction that commits no swaps. It cannot be shown to help. Revisit if C
finds a fatter tail elsewhere; the leave-one-out diagnostic is already written, so
checking costs minutes.

**The robust cost is nearly free and should ride along.** `c25` already charges
`μ = 4.0 mm` for a (slot, node) that no two cameras support — a truncation, in
millimetres, for the *absent* case. Applying the same constant to the *present but
wrong* case is `min(rms, μ)`: **no new parameter and no new threshold to justify**,
and it stops one contaminated term at 88 mm from dragging a whole state's score.
Do it the next time that cost function is touched; do not make it a project.

**Dropped entirely:** the per-view weight table. On frames where the
correspondence cannot be ambiguous, the three cameras' ray errors are
0.427 / 0.474 / 0.467 mm at the median — an inverse-variance spread of 1.25× — and
that number measures contamination rate, not precision. It was never worth new
constants, and the contamination it proxied for has mostly gone.

## Traps

* **`c09` writes to a fixed path.** Run it with `--npz` per arm, then **once with
  no `--npz`** to restore. The baseline is `c03`: preserved 269, break 19, swap 8.
  Verify it after every batch; `c11`/`c14`/`c15` read that file.
* **A derived gate is not comparable between reconstructions.** `c20` showed
  `derive_gate` takes the 99.9th percentile of the 3D step, and a step across a
  correspondence change is a *renaming*, not a movement. Hold the gate at `c19`'s
  4.0 mm / 73 frames and record that as post-hoc, as every arm since `c21` has.
* **`c16`'s provenance names a candidate maker that did not run.** It records
  `fixed_window`; `sleap_nn` switched it to `local_queues` because `max_tracks`
  was set. Any claim about `c16` resting on the candidate method is unsupported.
* **Swap counts of 0–8 rank nothing.** Every decision document since `c21` says
  it. A 12-interval difference in `preserved` is larger but is still one
  experiment, one trio.

## Where the numbers in this document come from

All measured in the session of 2026-08-24/26 on K_1_2, and reproducible from the
stored outputs: `work/c21_link_refusal/`, `work/c23_residual_refusal/`,
`work/c25_variable_cardinality/`, and `work/c22_gate_review/c22_events.json`.
