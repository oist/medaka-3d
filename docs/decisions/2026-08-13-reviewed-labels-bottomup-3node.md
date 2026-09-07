# Reviewed labels fix bottom-up, not top-down — and the full K corpus is inferred

**Date:** 2026-08-13
**Tools:** [`s02 --pectoral-mode reviewed-only`](../../tools/s02_merge_pectoral.py),
[`s14`](../../tools/s14_saion_train_3node.py),
[`s15`](../../tools/s15_compare_reviewed_models.py),
[`s16`](../../tools/s16_saion_inference_bu.py)
**Builds on:** [2026-08-09 detector part-mixing](2026-08-09-detector-part-mixing-20k.md)

## Verdict

**More labels fixed bottom-up and did not fix top-down.** Trained on the same
543 human-reviewed moments, the bottom-up 3-node arm improved on every metric
in every view; the top-down arm got *worse* on the permutation-sensitive one.
The 2026-08-09 conclusion — that the third node does not earn its place in
top-down — survives a 2–4× increase in training data.

**Production 3-node inference for group K is complete**: 3,137,400 frames,
bottom-up, all three views, verified.

## What changed since 2026-08-09

The earlier decision named the training-data confound as the leading rival
explanation and priced the control at ~5 h. The experimenter instead ran a
**review pass over the full 1,382-moment package**, returning 543 reviewed
moments. That is a stronger experiment than the control: it tests the data
hypothesis directly, on both architectures at once.

### The labels

543 moments, every pectoral human-placed — no model fill, no interpolation.

| view | train frames | train instances | val |
|---|---|---|---|
| top | 513 | 1,539 | 30 / 90 |
| side | 1,056 | 3,169 | 30 / 90 |

Against the previous 3-node splits (409 top / 250 side) that is **1.3× on top
and 4.2× on side** — and side was the arm that collapsed.

Selection is **moment-level, and that is an assumption rather than a
measurement**: the annotator shows all three views of a moment at once, so a
moment with any moved point is taken as inspected in all three. Instance-level
selection would keep only the 2,194 points that moved and discard the ones the
reviewer saw and judged correct, which are labels too.

Two data-provenance findings surfaced while assembling this:

- **600 view-frames of existing human side annotation were never merged.**
  Round 2's batches returned side edits (19.6–36.5% of points moved) but were
  registered as authoritative for `top_cam` only. The exclusion was deliberate
  and its reasoning was sound — round 2's *side* seed was interpolation, so
  importing wholesale would launder interpolation as annotation. Classifying
  each point against the seed ledger keeps the protection and spends the data.
- **The 2026-08-10 session was nearly lost.** 543 reviewed moments survived
  only in the full save; the quick-save file held one moment. Root cause and
  fix in the annotator are recorded below.

## The comparison, on identical frames

`s15` scores old and new arms over **the same 20,000 frames per view** that
`s10` used, reusing its cached predictions for the old arms, so this is a
paired measurement rather than two independent ones.

**`val/loss` is not the comparison and cannot be.** The reviewed-only
validation split shares only 9 of 30 moments with the split `s07`/`s03` used,
so old-vs-new val numbers describe different data; and bottom-up's loss carries
PAF terms centered-instance has no analogue for, so architecture-vs-architecture
val numbers are not the same quantity. Training curves below are convergence
evidence, nothing more.

`swap%` is the mouth-linked `tail_swap` — the only metric here that can see a
permutation. Shared-node remains valid *within* an architecture and invalid
*across* one, for the reason established on 2026-08-09.

### bottom-up: improved everywhere

| view | shared% | jump% | **swap%** | frames <3 | frames >3 |
|---|---|---|---|---|---|
| top old | 0.000 | 0.241 | 0.0818 | 857 | 451 |
| top **new** | 0.000 | **0.178** | **0.0224** | **641** | **216** |
| side1 old | 0.010 | 0.542 | 0.0501 | 1419 | 1009 |
| side1 **new** | 0.000 | **0.345** | 0.0531 | **1135** | **330** |
| side2 old | 0.005 | 0.565 | 0.1041 | 1242 | 950 |
| side2 **new** | 0.005 | **0.408** | **0.0798** | **1059** | **366** |

- **top swap 0.0818 → 0.0224%, a 73% reduction, p = 9.3e-06.** On the sides the
  swap change is not significant (p = 0.83, 0.18).
- **tail_jump falls in all three views** (−26%, −36%, −28%).
- **Cardinality improves at both tails.** Exact-3 frames rise in every view
  (18,692→19,143; 17,572→18,535; 17,808→18,575). Over-detection more than
  halves (451→216, 1009→330, 950→366) and misses fall (857→641, 1419→1135,
  1242→1059).

That last row settles the obvious objection. Mean instances per frame dips
slightly on the sides (2.981→2.960, 2.985→2.964), which would be the signature
of a model buying a clean swap rate by detecting less. It is not: **both the
`<3` and `>3` tails shrank at once**, so the dip is over-detection being
removed, not animals being missed.

### top-down: no improvement, and a real regression

| view | shared% | p | **swap%** | p | jump% |
|---|---|---|---|---|---|
| top | 0.820 → 0.730 | 0.305 | 0.0900 → **0.1248** | 0.071 | 0.377 → 0.461 |
| side1 | 5.605 → **6.200** | **0.012** | 0.1511 → **0.2009** | **0.044** | 0.670 → 0.711 |
| side2 | 7.365 → 7.765 | 0.13 | 0.2733 → **0.3359** | 0.057 | 0.972 → 0.894 |

Swap rose in all three views, significantly on side1 and marginally on the
other two. Shared-node rose significantly on side1. Cardinality is *identical*
(1422 / 1156 / 1007 unchanged), as it must be — both arms share round 8's
centroid head, so only node placement can differ.

**The data-volume hypothesis is supported for bottom-up and refuted for
top-down.** Same labels, same frames, opposite directions.

## Training

saion `largegpu`, four single-GPU jobs in parallel. All four early-stopped.

| model | epochs | best val/loss | wall |
|---|---|---|---|
| topdown CI × top | 127 / 200 | 0.000409 @85 | 56 min |
| topdown CI × side | 94 / 200 | 0.000543 @52 | 43 min |
| bottomup × top | 138 / 200 | 0.000092 @96 | 78 min |
| bottomup × side | 160 / 200 | 0.000106 @118 | 118 min |

Per-node confmap loss orders the same way in both top-down views —
mouth < pectoral < tail — so the pectoral is learnable but harder than the
anchor, as expected.

## Full-length inference (s16)

Bottom-up only, since it is the only arm the labels improved.

**3 experiments × 3 views × 348,600 = 3,137,400 frames.** 8-task array,
126 chunks of 25,000 frames, round-robin across tasks because top runs ~25%
slower than side and a contiguous split would strand the slow work on one task.

- All 8 tasks `COMPLETED`, 2 h 13 m – 3 h 03 m. 126/126 chunks, no `.partial`.
- Measured throughput: 32.4 fps (top), 41.5 / 44.3 fps (side).
- Copied to `/bucket/StephensU/medaka/sleap_output_3node/s16_bu_K_1_all/`,
  every file byte-size matched against `/work`.

| experiment / view | frames | missing | inst/frame | =3% | <3% |
|---|---|---|---|---|---|
| K_1_1 side1 | 348,600 | 0 | 2.972 | 94.0 | 4.3 |
| K_1_1 side2 | 348,551 | 49 | 2.963 | 93.9 | 4.8 |
| K_1_1 top | 348,435 | 165 | 2.988 | 95.5 | 2.7 |
| K_1_2 side1 | 348,600 | 0 | 2.941 | 92.2 | 6.3 |
| K_1_2 side2 | 348,529 | 71 | 2.944 | 93.9 | 5.2 |
| K_1_2 top | 348,485 | 115 | 2.981 | 94.1 | 3.8 |
| K_1_3 side1 | 348,589 | 11 | 2.944 | 93.0 | 6.1 |
| K_1_3 side2 | 348,598 | 2 | 2.941 | 93.2 | 6.2 |
| K_1_3 top | 348,549 | 51 | 2.984 | 94.4 | 3.5 |

464 frames missing in total (max 0.047% of any unit) — frames where nothing was
detected, not frames that went unprocessed; every chunk is present.

**The low detection rate seen in the 2,000-frame smoke was an artefact of where
it sampled.** Across the recording in twelfths, only the first twelfth is low
(2.68–2.92); the rest sit at 2.9–3.0. That is the handling period `s10`
deliberately excluded from its block sampling. One real anomaly remains: on
K_1_3 the fifth twelfth dips on **both side cameras** (2.83 / 2.79) while top
holds at 3.02 — a transient occlusion or lighting condition, not a model
property, and unexamined.

TensorRT was considered and not used. The environment has it (`tensorrt
11.0.0.114`, `onnx 1.22.0`) and `sleap_nn.predict` exposes `export_dir` /
`runtime`; the project's TRT dead-end (`peaks_castOut`) is specific to the
local Blackwell card and would not have applied on A100. It was declined
because the run already finished in 1/4 of the wall limit, so the only thing an
unvalidated second runtime could add is doubt about whether a later difference
came from the model or the runtime.

## Cluster facts, verified on this account

`docs/OIST_HPC.md` and `CLAUDE.md` both say saion renamed `largegpu` to
`gpu-a100` and to prefer the new name. **That is wrong for this account** and
should be corrected there:

```
sinfo                            both exist, same nodes saion-gpu[23-26]
sacctmgr show assoc              largegpu present, gpu-a100 absent
sbatch --test-only -p largegpu   scheduled on saion-gpu23
sbatch --test-only -p gpu-a100   Invalid account or account/partition combination
```

Three more, each of which cost a failed submission:

- **`largegpu` MaxWall is 12 h**, `GrpTRES cpu=128,gres/gpu=8,mem=1T`. Asking
  for 24 h does not fail at submit — it pends forever on
  `AssocMaxWallDurationPerJobLimit`, which looks like a queue and is a dead job.
  8 GPUs is also why the array is 8 tasks and not more.
- **`#SBATCH` directives are read by Slurm, not the shell.** `${PARTITION:-x}`
  is a literal and fails with `invalid partition specified`.
- **saion's scratch is `/work`, not `/flash`** (that is deigo's). `/bucket` is
  read-only from compute nodes.

## The annotator quick-save defect, fixed

The 2026-08-10 review of 543 moments existed only in the full save; the
quick-save file held one moment, and had it been relied on the session was gone.

`Ctrl+S` was not broken — verified by driving the real window through both save
orderings on the real seed, with every edit persisting. The defect: when the
resume file fails to load, the handler printed a warning to a console nobody
watches and **continued with the seed**, leaving the 2-minute autosave timer
armed against the file that had just failed to load. Reproduced against the
pre-fix build, which overwrote the unreadable file inside the autosave window.

Fixed in `multiview_annotator_3node`: a failed resume is now fatal with
instructions, and every quick save keeps one `.prev` generation.
`tests/test_quicksave_safety.py` fails 4 checks on the pre-fix build and passes
on the current one; the existing 48 checks still pass. A gap was also closed in
`run_tests.bat`, where `test_zoom` had no `errorlevel` check and its failures
were being swallowed.

## n_groups

**n_groups = 1.** Every frame trained on and every frame inferred is group K —
one trio of fish on three consecutive days. `K_1_1`, `K_1_2`, `K_1_3` are
repeated measures, not replicates.

The `s15` comparison is broader: n_groups = 4 (G, H, I, K) over 20,000 frames
per view, so the *ranking* of the arms rests on four groups even though the
labels came from one.

What this does not support: any absolute rate as a corpus estimate (10 of 14
groups are unmeasured), and anything about tracking, identity over time, or 3D
reconstruction. This measures single-frame part assignment and one-frame
temporal coherence.

## Next

1. **Tracking (Phase C) on the s16 predictions** — not started, deliberately.
2. **Do not retrain top-down on more pectoral labels.** Two independent
   measurements now say the third node does not pay there.
3. **The K_1_3 fifth-twelfth side dip** is unexplained and worth one look
   before the predictions are used for 3D.
4. Correct the `largegpu`/`gpu-a100` claim in `CLAUDE.md` and
   `docs/OIST_HPC.md`, and register the s14 checkpoints, the s16 predictions
   and the still-unregistered locked HITL-v2 benchmark in `MANIFEST.json`.
