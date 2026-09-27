# Stages A-C results: variable-detection-count 2D tracklets

**Verdict: parked (not promoted).** Stages A-C are complete: implementation, runs of all
conditions, HTML and comparison material, and verification.
P1 fragmented the main 20 s into 11,292 more intervals than P0 and did not meet the
acceptance criteria.
Settings were not changed to fit the results. No improvement in physical-fish ID accuracy is
claimed.

## Main 20 s [75270,78590), 166 fps, the same 3 fish

All conditions retain 29,386 rows. Original points, dropouts, detection scores and global_row
match C01.
The denominator of the link rate is 29,377 observations (excluding the first core frame).
This is the fraction of adjacent links, not the rate of correct individual continuation.

| Condition | Contiguous intervals | Adjacent link rate | Unlinked rate | Same label across a gap |
|---|---:|---:|---:|---:|
| C27 | 224 | 99.27% | 0.73% | 211 |
| S1 | 217 | 99.29% | 0.71% | 206 |
| S2 | 208 | 99.32% | 0.68% | 197 |
| P0 | 323 | 98.93% | 1.07% | 0 |
| P1 | 11,615 | 60.49% | 39.51% | 0 |
| P1aux | 6,491 | 77.94% | 22.06% | 0 |
| P1_margin_low | 6,348 | 78.42% | 21.58% | 0 |
| P1_margin_high | 14,704 | 49.98% | 50.02% | 0 |

Adjacent links have both ends inside the core. Same labels across a gap also include those whose
end is inside the core and whose start is in the leading context.
The protocol's general metric_edge_scope wording is correct only for adjacent links.
For the number of gaps with both ends strictly inside the core, see
`scientific_scope_audit.json`.
P0/P1 commit adjacent links only; P1's relations across a gap stay in the candidate table and
are not merged.

S1 isolates the returned-row ordering problem through object correspondence. S2 changes only the
OKS scale from S1, to top=14400 / side=10000 px². Both are re-runs with local history in the
current environment, not identical reproductions of the earlier whole-recording processing of
the saved C27.
P0 was confirmed byte-identical to the original source and identical in result on all 33 known
case-views.

P1's competition evaluation with open endpoints came with many holds and forward/reverse
disagreements.
The auxiliary M/T support reduced fragmentation, but it remains higher than P0. Strata by
proximity test, node dropout and detection count, and the event evidence, are saved in the
HTML/CSV. The cause is not asserted to be true fish ID errors or reflections.
For the actual mechanism of candidate competition, see the scientific audit record.

## Scope and undetermined items

The plan's 11 cases × 3 views were extended with the 20 s [332,3652), selected in advance from
metadata only, and run after the settings were fixed. The selection rationale, which excluded
known annotation positions, review intervals, long-interval experiments and background-image
positions together with their context, and its SHA are in `selection_audit.json`.
This is the same recording and the same group, not a test demonstrating generalization to
independent groups.
Detections and saved C27 for the whole recording already existed, so this is also not proof
that "nobody has looked at it before".

Local 17-frame tallies and review questions for 13 anchors × 3 views are saved in
`case_findings.json`.
The no-swap reviews of f76791/f323819/f77415 are limited in camera, panel and per-2D-row
correspondence, and were not repurposed as all-individual 2D ground truth. No new human labels
were created.
The denominator of confirmed individual ground truth is 0, so the correct-continuation rate and
wrong-link rate are undetermined.
Reflections are undetermined for all rows. 3D reconnection and assignment correction remain as
stage D.

## Artifacts and verification

- [HTML report](../../work/variable_count_tracklets_v1/report.html)
- 42 MP4 (3 full 20 s, 39 slow-motion event clips), 126 PNG, 36 native-frame SVG.
- Decoded all 8,274 output frames of every video. The full videos are sampled at 30 fps;
  the event videos play all 166 native frames at 30 fps. The HTML can switch between 6
  conditions.
- Independent audit of the 288 condition/case/view arrays, 21 real-detection-count regressions,
  6 full-row shuffles of real input, and 416 existing files unchanged.
- Synthetic tracking, SLEAP control, rendering and lab-structure tests and Ruff pass. Details in
  `test_receipt.json`.
- Browser switching, playback and reference links checked. Details in `browser_receipt.json`.
- Saved: C01, committed links, hold evidence, predictions in separate arrays, reconnection
  candidates, node quality, all settings, all stratified tallies, provenance and SHA.

Adoption is on hold because of over-fragmentation on real data and the absence of complete
individual ground-truth annotation.
The next design change will examine how open-endpoint competition is represented, and compare
it as a separate version from the current fixed conditions.
This lab expires 2026-10-17. Production and earlier results were not changed.
