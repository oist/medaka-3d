# P-1-1 identity evaluation

Question: How well does the existing geometry-based repository pipeline preserve
F/M1/M2 identity against the human-reviewed P-1-1 reference?

**2026-11-02 follow-up:** The original results below remain the frozen baseline.
The [row-mapping repair and join review](ROW_MAPPING_REPAIR.md) use a separate
run, `id_row_mapping_20261102_r1`; neither the baseline nor the human reference
is overwritten. The repair comparison and its review ledger are not an
appearance experiment or a method promotion.
The full seven-setting rerun found exactly two changed top-camera labels at
native frame 20 and no difference in the compared final 3D/label fields or ID
scoring. The follow-up records the connection evidence and review limitations.
The executed comparison is in [row_mapping_review.ipynb](row_mapping_review.ipynb).

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

Historical next step (superseded by the completed follow-ups below):
isolate the adapter row-correspondence defect in a separately
versioned repair, then review the remaining identity changes in source video.
Compare segmentation-derived appearance links on held-out encounters with
geometry held fixed; do not split adjacent frames randomly or claim transfer
to different fish from this one trio. No appearance experiment ran in this task.

Historical proposed appearance comparison, not executed:

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

## Geometry/motion and localized review follow-up, 2026-11-04

The user requested geometry/motion experiments using the existing P-1-1 manual
annotation, followed by short flip warning intervals; appearance is deferred.
See [the result](GEOMETRY_MOTION_RESULT.md),
[executed notebook](geometry_motion_review.ipynb), and
[registered experiment](../../labs/20261104_geometry_flip_risk/README.md).

The core warning rule concentrates 96.97--100% of sustained reference-change
groups into 5.01--5.27% of recording time across seven arms. At 0.767, it exports
129 warning intervals (90.795 s) and 130 lower-risk local intervals. Those local
intervals are not automatically verified physical names. Permissive C30 improves
some local connections but does not consistently improve fixed-anchor identity
time; retain the existing automatic output. The final run is
`geometry_motion_risk_20261104_r3`; the failed r2 comparison is preserved and
explained. The next informative checks are the ten warning intervals containing
the sustained reference changes and their clean context. Existing annotation
is usable immediately; more human review is not a prerequisite to experiments.

## Selective-connection study proposal

The [six-recording registration draft](SELECTIVE_CONNECTION_PLAN.md) incorporates
the proposed K_1/P_1 development study, reference quality checks, and
dependence-aware evaluation. On 2026-11-04, the supervisor selected a 1% error
rate per automatically accepted connection as the development goal; this is not
an achieved result or an external guarantee.

The 2026-11-05 revision limits manual effort. Development uses only the six
existing ID-pass reviews, with conflicts treated as unknown, and nested
leave-one-recording-out cross-validation. The first holdout is K_2 only; P_2 is
a conditional second stage decided from a label-blind count. A blinded audit of
about 20 decisions checks for gross reference failure. New-trio holdouts are
undecided and outside the current workload, so no unseen-trio claim is made.
The companion `selective_connection_protocol.draft.toml` is explicitly not
executable or frozen: the risk unit, threshold rule, go/no-go rule, holdout
manifests and reviewers, and merged extraction revision remain to be settled.
No new extraction, reference conversion, model fit, or holdout content
inspection ran for this draft.

## Label-free contrastive appearance link, 2026-11-06

The supervisor redirected effort from human ID correction to strengthening the
tracking itself and proposed idtracker.ai-style contrastive learning for
appearance. Step 1 tested whether an appearance signal links geometry fragments
across warnings on P-1-1. See the
[registered experiment](../../labs/20261106_contrastive_appearance_link/README.md)
and [its result](../../labs/20261106_contrastive_appearance_link/RESULT.md).
The run is `appearance_link_20261106_r1`, configured by
`config_appearance_link_v1.toml` and produced by `run_appearance_link.py`
(`extract`, `train`, `evaluate`).

At lambda 0.767 the pre-registered rule reads carry-forward. Per-camera CNNs
were trained only on same-fragment positives and coexisting-fragment
negatives. On the reference-known boundaries they decided 71 correctly and
none wrongly; 56 were undecidable. They corrected five of the nine boundaries
the chain gets wrong; the other four are undecidable. Constrained
whole-recording naming reduced lower-risk wrong-ID time from 815.558 to 21.804
fish-s, with no newly wrong time. All of the remainder lies in 34 intervals too
short or too crowded for non-overlap crops, mostly the 527--559 s
close-contact cluster. The lab is parked, not promoted. This is one recording,
trained and scored transductively, and the evaluation reference is the Jingyi
review. Warning interiors were not renamed.

Registration 2, same day: a naming stage that labels every post-preparation
frame from the appearance clusters or leaves it unknown, and crops with the
other fish masked by their poses (`run_appearance_naming.py`, run
`appearance_naming_20261106_r1`). Both registered rules passed. With masked
crops, 97.41% of reference-assessed fish-seconds are named and 0.181 fish-s
of those are wrong (chain: 848.417 wrong). Most of the remaining 138.03
unknown fish-s lie in two long close encounters (613.7 s and 510.1 s), whose
flanking interval is too short to name. Per-frame names and `xyz` by identity
are in `evaluation/identity_tracks_N_B.npz`. These are the lab's outputs, not
a promoted extraction stage.

Registration 3, same day (`run_appearance_gap.py`, run
`appearance_gap_20261106_r1`): bridging across unnamed slivers (S) and
per-frame appearance inside gaps with flank-constrained switching (G). Both
registered rules rejected: on gap time, S is 2.04% wrong and G 11.0% wrong
against a 1% limit. The error is in the two long close encounters. There,
single-crop appearance agrees with the reference only 70--76% of the time,
against 93--98% elsewhere. A crop identity audit projected the reference
fish into each camera. It confirms that training crops show the slot's fish
(100.00% of the 98--99% that are clearly attributable). The encounter errors
come from crops with partly covered bodies, or with two fish close in one
view; such crops are rare in training data. Registration 2's output (97.41%
named, 0.181 fish-s wrong) remains the lab's current result.

Registration 4, 2026-11-07 (`run_appearance_clean.py`,
`config_appearance_clean_v1.toml`; runs `jingyi/P-1-1/appearance_clean_20261107_r1`
and `work/appearance/20251225_K_1_2_FCMC/appearance_clean_20261107_k12`):
training and all appearance evidence restricted to clean crops (no own-body
occlusion, no other keypoint within 80 px, at least three detections in the
camera-frame), side crops flipped dorsal-up, one encoder for both side
cameras with cross-camera positives, silhouette-based stopping, and PCA
figures of the validation embeddings during training. On P-1-1 (development),
96.18% of assessed fish-seconds are named and 0.042 fish-s wrong; appearance
inside gaps with clean evidence falls from 11.0% to 0.35% wrong on gap time
but names less than bridging, so the registered G-versus-S rule reads
reject. On K_1_2, judged once against 255 confirmed female identifications
after 300 s: 250 named, 0 wrong, 5 unknown; the registered rule (at most 1%
wrong, at least 90% named) is met, while the chain holds the female in the
other slot at 59 of those anchors. Male identity on K_1_2 is unverified (14
male-only name changes against slot continuation). The K_1_2 fragments were
computed label-free from the production chain output of 2026-09-03 with the
identity-core warning rules. Registration 4's pipeline is the lab's
carry-forward state; nothing is promoted.

Registration 5, 2026-11-07 (`run_appearance_clean.py --config`, GPU): two
single-change variants of the Registration 4 pipeline, judged on single-crop
accuracy over P-1-1's clean reference-clear warning crops against
Registration 4 on the common crop set (adopt at +2 points pooled with no
camera more than 1 point below; reject when any camera is more than 2 points
below or the K_1_2 anchor rule fails). Variant A, torchvision ResNet18 with
a bias-free 8-d head on Registration 4's own crops
(`config_appearance_resnet_v1.toml`; runs
`jingyi/P-1-1/appearance_resnet_20261107_r1` and
`work/appearance/20251225_K_1_2_FCMC/appearance_resnet_20261107_k12`), is
rejected: top-view accuracy 90.96% against 99.02%, side views within 0.5
points, pooled 96.25% against 98.96%. Its K_1_2 anchor counts equal
Registration 4's (250 named, 0 wrong, 5 unknown of 255). The silhouette
stopping rule let the ResNet stop after 300 steps; architecture and early
stop are not separable inside this registration. Variant R, 128 px crops
with 112 px input decoded again from video
(`config_appearance_res128_v1.toml`; runs
`jingyi/P-1-1/appearance_res128_20261107_r1` and
`work/appearance/20251225_K_1_2_FCMC/appearance_res128_20261107_k12`), is
inconclusive: pooled accuracy 98.96% against 98.96% on the 3,758 common
crops, no camera moving by more than half a point, K_1_2 anchor counts
again equal to Registration 4's. Neither variant is adopted; 56 px input
and the four-block CNN of Registration 4 remain the carry-forward state.
Two failed training attempts of variant R (a whole-array float conversion
and GPU memory oversubscription) were fixed in memory handling only, before
any evaluation existed; the lab RESULT.md records them.

The [six-recording study](../20261108_six_recording_identity/README.md) extends
the comparison to K_1/P_1. The supervisor's
[2026-11-09 operating policy](../../docs/decisions/2026-11-09-separate-s-g-outputs.md)
uses S as the experimental default and retains G through a separate export
command and output. Never fill S's withheld identities from G. Historical
comparison runs and registered verdicts remain unchanged.
