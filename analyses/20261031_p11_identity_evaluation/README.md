# P-1-1 identity evaluation

Question: How well does the existing geometry-based repository pipeline preserve
F/M1/M2 identity against the human-reviewed P-1-1 reference?

Owner: P-1-1 research team; execution by Codex at the user's request.
Recording: `20260112_P_1_1_FCMC` (group P, trial 1, day 1).
Starting revision: `c601b7bc3fc298bd5d2c3734b3b45bf926551e6b`.
The pre-existing working-tree changes are unrelated and remain intact.
The preserved October 28 run records revision
`50d56d039653617687091c5bfb4689538fab2479` and Python 3.13.17; this rerun uses
the project environment's Python 3.13.11. Its original stage provenance is
archived in `previous_run_provenance/`. Runtime differences are not a controlled
speed benchmark across these environments.

## Inputs and execution

Machine roots come from `configs/paths.local.toml`; relative paths and all
scoring settings are in `config.toml`. The user requested generated results
under `jingyi/P-1-1`, overriding the default analysis-export destination.
The active output run is `id_evaluation_20261031_r2`.

- Human reference: `reviewed_3d.csv`, pinned in the run's `inputs.json`.
- A checksum-verified copy is archived under configured
  `asset_root/groundtruth/identity_reviews/<recording>/<checksum prefix>/`.
- S14 bottom-up detections are reused. Every cached source frame, node coordinate,
  and instance score was checked against the supplied per-camera SLEAP files.
  New pose-model inference is not part of this tracking rerun.
- Rerun: unchanged C27 serial tracking per view, C28 correspondence, C33 ray
  rejection, C30 v4 stitching, C34 micro-flip repair. All seven inherited lambda
  values are included; none is selected on these labels.
- The run contains code snapshots, hashes, calibration references, input hashes,
  native timestamp arrays, stage logs, and `medaka3d.provenance` sidecars.
- `run_pipeline.py` redirects only roots and invokes repository implementations.
  Cameras use the existing independent processes; time is never chunked.
  C28, C33, C30, and C34 use seven independent lambda processes, each spanning the complete
  recording. This orchestration change was made while C27 was running and does
  not modify the algorithm. Per-stage wrapper snapshots/configurations record
  the exact orchestration version; the initial source snapshot is preserved.
  C30's inherited provenance field still says `rule_version=2`; the pinned
  implementation contains the documented v4 settings (2.5 mm median support
  for velocity decisions and 30 mm bridge margins). This stale metadata field
  was not rewritten, and the executable source is retained.

## Reference semantics and evaluation protocol

The user defined `watched` as whether a frame was displayed in the review GUI,
regardless of playback speed; skipped frames are false. `analysed` marks frames
at or after a manually chosen start. `uncertain` was not used in this review.
These flags do not establish an independent frame-by-frame identity audit.

The reference has 348,600 frames and three rows per frame, without duplicate
frame/individual keys. ID_1 is M2, ID_2 is M1, and ID_3 is F. `analysed` starts
at frame 4,152; five frames, 251,065 through 251,069, were not displayed.

Use the pectoral point in both sources. The review shares most geometry with
saved tracker output, so this assesses reviewed identity agreement, not
independent 3D localization accuracy. Preserve every original input.

Native acquisition counters are one-based, while SLEAP and review indices are
zero-based. All three cameras have contiguous counters and strictly increasing
elapsed timestamps. Same-index elapsed camera times differ by at most 3.51 ms
in the inspected recording. Absolute hardware synchronization has not been
independently re-established. Retain the existing native-index camera alignment.
A deterministic frame-offset diagnostic finds substantially smaller position
differences at zero offset than at offsets -2, -1, +1, and +2.

The main window begins at 300 seconds on the top-camera acquisition clock:
first included frame 49,794 (300.002278797 s), not the nominal-FPS frame 49,800.
The final timestamp is 2100.265585614 s. Durations use left-held acquisition
intervals, no extrapolation after the last timestamp, and retain the initial
partial interval in the unprocessed coverage denominator. The full expected
post-preparation denominator is 5400.796756843 fish-seconds.

The fixed identity anchor is [298, 300) seconds, before the scored window.
Require at least 100 unambiguous matches per fish and 95% agreement in one
unique, one-to-one permutation. Never remap after a crossing. This maps anonymous
slots to supplied review labels; it does not independently establish the animals'
physical names.

Primary reference positions must be finite, watched, analysed, non-interpolated,
and free of uncertain/bad3d/nodes_bad flags. `nodes_bad` is a conservative
whole-row exclusion because it does not identify which node is unreliable.
Match mutually closest positions within 5 mm with at least a 2 mm margin against
all competing positions. Unclear matches are unscored, not forced or marked
correct. Report 2/5/10 mm sensitivity with the same fixed anchor mapping.
Matching thresholds are evaluation choices, not calibrated error probabilities.

Report identity agreement over scored time together with its scored denominator,
correct-ID time over all expected time, joint three-fish time, and all unscored
reasons. Break correct/wrong intervals at missing or ambiguous observations.
Assignment changes across a gap are recorded separately from adjacent changes;
neither count is automatically an independently reviewed crossing count.

Gated local-track purity is retrospective and many-to-one. It must never be
interpreted as trial-wide identity coverage; fragmentation is reported alongside it.
The proximity diagnostic uses 20 mm geometric separation, not courtship labels.
A failure can begin at a crossing and persist during separated swimming.
Frames and repeated observations are not independent biological replicates.

## Findings and validation

<!-- RUN_RESULTS_START -->
Completed the full native-frame C27 -> C28 -> C33 -> C30 v4 -> C34 chain
for all seven inherited settings, followed by reviewed-ID scoring.
No new pose inference or appearance model was run. No lambda is promoted.

| Lambda (mm) | ID agreement on scored time | Scored / expected time | All three correct / recording time |
| --- | --- | --- | --- |
| 0 | 38.73% | 98.87% | 17.62% |
| 0.1 | 27.27% | 98.86% | 15.05% |
| 0.25 | 49.83% | 98.86% | 29.32% |
| 0.5 | 45.02% | 98.86% | 24.59% |
| 0.767 | 84.11% | 98.85% | 74.26% |
| 2 | 19.72% | 98.84% | 11.68% |
| 8 | 22.52% | 98.79% | 11.67% |

The historically displayed lambda 0.767 is shown for continuity,
not selected using this reference. Its individual results are:

| Reviewed identity | Agreement on scored time | Scored / expected | Correct / expected | Longest observed disagreement |
| --- | --- | --- | --- | --- |
| F | 98.78% | 98.18% | 96.98% | 7.98 s |
| M1 | 76.35% | 98.63% | 75.30% | 48.14 s |
| M2 | 77.34% | 99.75% | 77.15% | 77.43 s |

Stage wall time: 84.6 minutes. Independent scalar checks:
19,087, with zero discrepancies. Raw-slot fresh-versus-prior comparison:
16,265,637 differing array elements across XYZ, source rows, slot labels,
residuals, and view counts, across all seven conditions.

This measures agreement with supplied reviewed IDs on shared geometry.
It does not establish independent 3D accuracy or generalize beyond this trio.

Numeric slot columns can be renamed between runs. Using the original
fixed pre-window anchors to align each run once into F/M1/M2 order,
the post-preparation comparison gives:

| Lambda | Changed matched identity (fish-s) | Changed scoring status (rows) | Max aligned XYZ difference (mm) |
| --- | --- | --- | --- |
| 0 | 3447.062077 | 381,165 | 187.09985 |
| 0.1 | 0.000000 | 0 | 0 |
| 0.25 | 0.000000 | 0 | 0 |
| 0.5 | 0.000000 | 0 | 0 |
| 0.767 | 0.000000 | 0 | 0 |
| 2 | 0.000000 | 0 | 0 |
| 8 | 0.000000 | 0 | 0 |

Differences remain after fixed-anchor alignment for lambda 0. Their cause has not been isolated across
the execution environments. They are not attributed to a tested method improvement.
<!-- RUN_RESULTS_END -->

An observation-only contract audit of the first 2,000 frames per camera found
one returned-order mismatch in the top view at frame 20: `[0, 1, 2, 4, 3, 5]`.
C24 assigns output labels by returned-list offset, so two labels are written to
the wrong detection rows. The side-view prefixes had no observed mismatch.
This prefix is outside the scored window; its contribution to later identity
errors is not measured. An extended top-camera audit with continuous history
through frame 51,999 found only the same frame-20 defect; none of its 2,206
post-preparation frames had a returned-order mismatch. This is not a full-trial
audit and does not establish that the defect caused the measured ID errors.
The unchanged baseline retains this defect. A separately
versioned adapter repair should be compared before attributing the remaining
disagreement to the limits of geometry or selecting an appearance method.

Executed checks: twelve synthetic scorer invariants passed; 29 combined focused
checks passed for the scorer, slot stitching, micro-flip repair, ray rejection,
and variable-cardinality correspondence. Two existing all-NaN median warnings
occurred in synthetic stitch tests. The full suite was not run.
The report notebook executed all nine code cells in a fresh project-virtualenv
kernel without error outputs. All four exported figures were visually checked
for readable labels, legends, denominators, and unclipped content.
The completed C27 rerun reproduced all 3,098,155 per-detection labels exactly
across the three cameras; details are in `c27_previous_comparison.json`.

Attempt r1 stopped during input inspection because NumPy promoted a mixed
unsigned/signed index addition to a non-integer dtype. No tracking ran in r1.
The preparation code now explicitly converts point offsets to int64. Its failure
provenance is retained; r2 uses a distinct directory. No source input was changed.

## Commands

Use the project virtual environment:

```powershell
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/run_pipeline.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/evaluate.py --source rerun --version v1
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/audit_results.py --source rerun --version v1
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_notebook.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/package_results.py
```

The run directory must be new; completed pipeline and scoring outputs are not
overwritten. Set a new run ID for another run. The notebook builder starts a
fresh project-virtualenv kernel. Packaging checks completed execution and
reconciled scoring, verifies that the original review is unchanged, and records
an artifact checksum manifest. Inspect the generated figures before handoff.

Next step: isolate the adapter row-correspondence defect in a separately
versioned repair, then review the remaining identity changes in source video.
Compare segmentation-derived appearance links on held-out encounters with
geometry held fixed; do not split adjacent frames randomly or claim transfer
to different fish from this one trio. No appearance experiment ran in this task.

Proposed comparison, not executed:

1. Review the onsets of long M1/M2 disagreements and successful encounters in
   the source videos. Separate unreliable positions, camera correspondence
   failures, and incorrect links between otherwise usable track segments.
   Include difficult or unscored encounters when defining the review queue.
2. Fix and version the adapter row correspondence separately, retaining this
   unchanged baseline. Quantify its effect rather than assuming it explains
   the main-window disagreements.
3. On separately held-out encounters, compare the same geometry with and
   without appearance-assisted pre/post-encounter links. Use isolated views
   before and after overlap to test whether mask-cropped appearance provides
   additional identity evidence. Keep uncertain links unresolved. Report
   correct event links, newly introduced errors on previously correct links,
   assessed-event coverage, and subsequent wrong-ID time. This experiment
   would evaluate the proposed appearance evidence; segmentation alone is
   not evidence that the identity problem has been solved.
