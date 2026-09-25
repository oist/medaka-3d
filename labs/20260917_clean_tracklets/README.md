---
hypothesis: Independent conservative 2D fragments can supply reliable temporal anchors, while two clean views can reconnect a broken third view without inheriting SLEAP tracking IDs.
stage: linker
expires: 2026-10-17
decision_rule: Keep all detections; qualify on synthetic ambiguity tests, the fixed 20-second sequence and eight reviewed anchors, report coverage alongside errors, and do not promote without identity review.
---

# Clean 2D fragments and cross-view completion

This is a new experiment, separate from the previous A/B/C comparison. SLEAP
supplies detections and their mouth/pectoral/tail grouping, never temporal IDs.
No C27 labels, old reconstructed trajectories, or human identity annotations
enter the proposed tracker. Old outputs are loaded only after estimation for
comparison. Human anchors evaluate results, not parameter selection.

## Fixed design before real-data execution

1. Independently associate each camera's detections using pectoral-only motion
   (`P`), coherent node motion (`N`), and node motion with measured image flow
   (`F`). Keep every input row. Ambiguous crossings, missing observations,
   conflicting nodes, or insufficient assignment margin end only the affected
   fragment. Forward/backward agreement is required. Do not bridge gaps in 2D.
2. Reconstruct from refractive camera geometry and physical 3D continuity.
   Agreement between independent clean fragments is temporal evidence. Do not
   reuse the old C28 penalty on changes in packed SLEAP label triples.
3. Record a separate cross-view join only when the other two clean views
   support the same continuation across the interval, the endpoint geometry
   agrees, and competing assignments are sufficiently worse. Keep the initial
   clean fragments immutable. Missing pixels remain missing; a projection is
   a prediction and is never written as a detection.

Numerical settings are frozen in dataclasses and copied into each run's
protocol and source snapshot. No event-specific thresholds are allowed.

## Evaluation and honest limits

Use experiment `20251225_K_1_2_FCMC`, native 166 fps. The primary interval is
`[75270,78590)` (20 seconds). Additional known cases are the prior exact
`known_30735`, `known_31479`, and `known_323819` intervals. Together these
cover all eight latest human review anchors, including no-swap controls
76791, 323819 and 77415. Include prior `control_50000` and `holdout20s`
(`240000:243320`) as additional regression sequences. The latter was already
examined in the previous experiment and is not a fresh independent holdout.
Use 332 native frames of context on each side, as in the previous comparison.

Report per-view retention, clean edge coverage, fragmentation, reasons for
cutting, accepted/rejected joins and their support, 3D observation coverage,
native-frame displacements and available female endpoint scores. A decrease
in jumps obtained by abstaining is not automatically an improvement. Short
clips contain all three fish in all three views. Preserve raw motion without
interpolation/smoothing. Compare with the saved old C34 and previous C v3
outputs, with a single prefix-based permutation used only for display names.

The eight review notes are anchors rather than complete framewise ground
truth. Changed assignments and smoother trajectories are not verified swap
corrections. Male-versus-male identities remain unvalidated. This is one trio
of fish (`n_groups=1`). SLEAP's within-frame grouping can itself mix nodes;
this experiment rejects inconsistent bundles but does not retrain detection
or silently redistribute individual nodes between fish.

Every artifact is accompanied by provenance and exact source arrays. The
previous report and all original tracking outputs remain intact.

## Reproduction

From the repository root:

```powershell
.venv/Scripts/python.exe labs/20260917_clean_tracklets/run.py
.venv/Scripts/python.exe labs/20260917_clean_tracklets/run_consensus.py
.venv/Scripts/python.exe labs/20260917_clean_tracklets/audit.py --output work/clean_tracklets_v2
```

`run.py` saves the initial experiment to `work/clean_tracklets_v1`.
`run_consensus.py` reuses byte-identical P/N/F 2D results and writes a separate
`work/clean_tracklets_v2`. The new command requires a fresh destination. The
default numerical configuration is the same in both 3D estimators.

The latter command addresses an observation-level ambiguity exposed by the
initial run: multiple competitive assignments can agree on two measured views
while disagreeing only on the third. In that case, keep their common two-view
observation after independent geometry and motion checks. Leave the uncertain
third view unassigned. If fewer than two measured sources agree across all
competitive joint assignments, abstain. This changes the unit of abstention,
not the numerical margin or the independent 2D tracker. Synthetic tests check
both recovery and refusal of ambiguous fish identities. The initial results
and objective are preserved for comparison.

`build_tracklets(frame_idx, points, view=..., feature_mode=..., flow=...)` is
the 2D entry point. Nodes are M/P/T and frames are native integer indices.
Returned labels and `clean_edge` cover every input detection. P uses pectoral
only; N/F use mouth and pectoral association, with tail consistency as a veto.
Optical flow supplies additional image evidence and cannot override hard
motion, shape or close-interaction exclusions. Returned IDs identify local
fragments, not permanent biological identities. `reconstruct(...)` consumes
these independent fragments and returns measured sources, 3D coordinates,
identity-review flags, and a separately audited join layer.
