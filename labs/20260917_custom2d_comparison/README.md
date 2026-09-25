---
hypothesis: Bidirectional motion checks and independent variable-cardinality 2D linking can reduce wrong temporal connections without sacrificing detection coverage, but downstream 3D correspondence may remain limiting.
stage: linker
expires: 2026-11-17
decision_rule: >
  Compare frozen serial SLEAP labels (A), bidirectional cuts of those labels (B),
  and an independent bidirectional motion tracker (C) on exactly the same
  detections. Keep all detections, calibrations, 3D objective and postprocessing
  fixed. Report fragmentation, missingness, motion discontinuities, endpoint
  female evidence and all-three-fish visual comparisons separately. Neither
  smoother coordinates nor fewer linked jumps establishes fewer biological
  identity swaps. No production promotion from this one-trio development panel.
---

# Three fixed-detection tracking conditions

Authorized by the experimenter on 2026-09-17, implementing the three conditions
in `docs/decisions/2026-09-16-custom-2d-tracking-plan.md`.

The experiment keeps every original C01 detection row. A uses the existing full
serial C27 labels; SLEAP is not restarted on the 20-second crop. B and C use the
same per-camera pixels and frame times. B cuts original edges without deleting
detections. C never consumes SLEAP labels or 3D coordinates. Tail positions are
retained for reconstruction/display but do not decide 2D identity.

## Fixed protocol, before real-data comparison

- Primary: native frames 75270:78590, 3320 frames at 166 fps (20 seconds).
- Reuse all 19 cases from the preceding bidirectional protocol: the primary,
  12 historical neighbourhoods, 3 previous false-repair windows and 3 unlabelled
  controls. Overlapping cores are not summed as independent evidence.
- Add a temporally separate 20-second exploratory window 240000:243320 after
  freezing tracker settings. It is another interval of the same fish group,
  not an independent biological replicate or an identity-labelled test set.
- Give every condition 332 native frames of context on each side. Save native
  coordinates and source rows; no temporal interpolation or smoothing.
- Use common C25/C28 reconstruction at lambda 0.767 and common C33, C30 v4,
  C34 postprocessing. Save both stages. Diagnose the windowed A result against
  the stored full-sequence C28 using one fixed slot permutation. Do not silently
  equate windowed and full-sequence inference.
- Align display names once from a common pre-core prefix of 17 fully resolved
  reference frames. Do not use the reviewed swap interval or female annotations
  to select the alignment or tune the tracker.
- Freeze B/C settings before examining their real-data output. The tracker
  module records the parameters, candidates and cut reasons. Main pixel scales
  derive from the previous C24b gates (top60 / sides50 px per native frame).
- Report 2D detection coverage, connected edges, gaps, number/length of
  fragments, and same-ID displacement distributions together. Report 3D
  availability, native-step counts above4/10mm, geometric residuals, and source
  agreement separately from identity.
- Use the existing female annotations only through the common evaluation code.
  The eight recent human reviews are contextual labels with unspecified panel
  and exact event intervals; they are not a complete framewise ID ground truth.
  The 76930/76988 clips overlap and must not be counted as independent swaps.
- Review all three fish in all three cameras. Clearly mark the no-swap controls
  76791,323819,77415. Changed assignments are candidates, not verified fixes.
- Lambda sensitivity is a separate check; a broad superiority/promotion claim
  requires all seven historical lambda settings and independent identity review.

## Reproducibility and delivery

Inputs are verified against the prior audit and staged locally from configured
shared roots. Each run records hashes, settings, native ranges, correspondence
outputs and runtime. Existing C27/C28/C34 and prior review files stay unchanged.
The final local HTML report includes measured tables, figures, comparison
videos and links to the per-case numerical evidence.

## Implementation qualification

The first four-case run is retained under `analysis/`, with its code snapshot.
It exposed an implementation defect in C: after an ambiguous observation, an
unlinked old history competed indefinitely with newly created histories for the
same fish. A synthetic slowly moving fish with one duplicated detection
reproduces the failure without using real-data identity outcomes. The correction
ends old histories which had an admissible current observation but were not
linked; true missing-observation histories remain eligible for the fixed gap.
No numeric threshold is changed. This conservative comparison is preserved in
`analysis_v2/`, with its exact source snapshot.

The completed condition C adds the short two-hypothesis deferred check specified
in [deferred_protocol.md](deferred_protocol.md). It uses later observations and
a symmetric past check to choose between two continuations, retaining ambiguous
detections and refusing unresolved/three-way cases. Existing consensus tracklets
are never split. Its numerical limits are fixed before real-data execution.
This is a bounded first implementation: synchronous two-tracklet boundaries,
three observations on each flank, and at most five native intervals of lookahead.

`analysis_v3/` is the final comparison. A/B are byte-verified copies from v2;
C uses the added deferred check and the same downstream reconstruction. The v2
C summary remains alongside each final case as an explicit ablation. Unchanged
2D partitions retain their original numeric names and exact 3D bytes.

## Reproduction commands

From the repository root, using its Python environment:

```powershell
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/run.py --output work/custom2d_comparison_v1/analysis_v2
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/run_deferred.py
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/review_windows.py --analysis work/custom2d_comparison_v1/analysis_v3
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/event_figures.py
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/audit_delivery.py
.venv/Scripts/python.exe labs/20260917_custom2d_comparison/render_reviews.py --analysis work/custom2d_comparison_v1/analysis_v3 --output work/custom2d_comparison_v1/review_v3 --include-c28-31479
```

Preserved outputs are not overwritten by the experiment runners. Use fresh
output directories for a changed protocol. The report adapter and its provenance
are in `work/custom2d_comparison_v1/report_inputs/`; the report app source is in
`work/custom2d_comparison_v1/report_app/`. The final offline export is
`work/custom2d_comparison_v1/comparison_report.html`.
