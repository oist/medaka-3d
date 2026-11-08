# Row-mapping repair and join review

Date: 2026-11-02. Owner: P-1-1 research team; implementation and execution by
Codex following the user's agreed sequence. Status: repair comparison, join
ledger, and review package completed; event adjudication remains open.
No commit, push, PR, appearance training, or physical-identity promotion.
Starting repository revision: `fd8d86f81715239b39ef8496299768b43d0679c4`.
The repair is an uncommitted working-tree change; frozen source hashes identify
the executed implementation rather than implying that HEAD contains the fix.

## Question and fixed inputs

Does repairing the adapter's returned-instance/source-row correspondence change
the full post-preparation P-1-1 identity agreement? What evidence is available
for the remaining local-segment connections?

- Frozen baseline: `id_evaluation_20261031_r2`, all seven lambda settings.
- Candidate: `id_row_mapping_20261102_r1`, configured in
  [config_row_mapping_v1.toml](config_row_mapping_v1.toml).
- The baseline input inventory, configuration, and artifact manifest are pinned
  by SHA-256. Consumed baseline scoring/C27/C34 artifacts are verified against
  that manifest. The candidate must reproduce the baseline's review, detection,
  SLEAP, native-timing, and calibration input hashes before tracking starts.
- Research inputs and generated outputs remain in the configured shared
  `jingyi/P-1-1` location. The original annotation archive is reused and checked.
- The acquisition clock, first-300-second exclusion, matching thresholds,
  [298, 300) s fixed anchor, and scoring denominators are unchanged.

The first sandboxed storage check could not see the mapped drive and denied
UNC access. An authorized read outside the sandbox succeeded for both paths;
this was an execution-boundary limitation, not missing research data.

## Adapter correction

C24 is imported by serial C27. Previously it wrote the first returned label to
the first input row, and so on. The library can return matched instances before
newly assigned instances, reordering the rows. The reproduced top-camera frame
20 order was `[0, 1, 2, 4, 3, 5]`.

The corrected adapter captures the input instance object's source row before
calling the tracker, then writes each returned label to that row. Coordinates
are not used as keys. Dense numeric names are still allocated in return order;
the library configuration and internal tracking history are unchanged. A missing,
duplicate, or foreign returned object now raises an explicit contract error.
`effective_tracker_config.row_assignment` records `input_instance_identity_v1`.

The comparison reruns the complete recording from frame zero, independently
for each camera; it never chunks time or assumes a preparation-frame defect
cannot affect later correspondence. C28/C33/C30 v4/C34 run for every inherited
lambda. Runs are aligned once by their original pre-window identity anchors;
raw slot-number differences alone do not count as identity differences.

Completed C27 comparison: all **3,098,155 detection rows** were compared.
Exactly two labels differ, both in top-camera native frame 20 (0.120494 s):
global rows 54/55, frame-local rows 3/4, change from 2/4 to 4/2 respectively.
Both side-camera arrays are identical. No post-preparation C27 label differs.
The completed downstream comparison finds no propagated difference in the
compared final output fields or the identity scoring.
See `c27_row_comparison/changed_rows.csv` and `summary.json`.

## Repair result

All seven inherited settings completed C27 -> C28 -> C33 -> C30 v4 -> C34.
The summed stage wall time was **86.3 minutes**, without pose inference; this is
an execution record, not a controlled performance benchmark. Python/package
versions and all seven installed tracker-source files match the frozen baseline.

For every lambda, the full-recording final label-output arrays are identical
for the compared fields: XYZ, source rows, slot labels, residuals, and camera
counts. Fixed-anchor alignment also finds zero post-preparation differences,
zero changed score-status observations, and zero changed matched-identity
fish-seconds. The independent scoring audit checks **19,087** scalar exact-coordinate
correspondences with **zero discrepancies**, and reconciles every scoring category.

| Lambda | Baseline agreement | Repair agreement | Scored fish-seconds, both runs |
| --- | --- | --- | --- |
| 0 | 38.73% | 38.73% | 5339.969 |
| 0.1 | 27.27% | 27.27% | 5339.366 |
| 0.25 | 49.83% | 49.83% | 5339.318 |
| 0.5 | 45.02% | 45.02% | 5338.969 |
| 0.767 | 84.11% | 84.11% | 5338.764 |
| 2 | 19.72% | 19.72% | 5337.933 |
| 8 | 22.52% | 22.52% | 5335.402 |

Agreement is pooled across all three fish on scored time. The full declared
denominator is **5400.796757 fish-seconds** in every arm. At lambda 0.767,
F/M1/M2 agreement remains **98.78% / 76.35% / 77.34%**. The source-row bug is
real and repaired, but it does not explain this recording's remaining scored
identity errors. This does not establish a universal limit of geometry or an
accuracy claim for another recording.

## Join-ledger definitions

The final policy is [config_join_ledger_v2.toml](config_join_ledger_v2.toml).
It is an exploratory diagnostic policy, fixed across arms and both runs, not
a calibrated identity-acceptance rule.

- Local-track evidence aggregates all assessable post-preparation observations
  for an ID, even when it moves output slots. Tracks without reference matches
  remain in the inventory; the earlier 458 count included only matched tracks.
- A segment is a contiguous native-frame run in one output slot with a valid
  position and unchanged local-track ID. Missing frames always break segments.
- A candidate edge connects successive such segments in the same output slot.
  It represents the identity continuation asserted by that output slot.
- Both the whole local track and the contiguous segment must have at least
  100 matched frames, 0.5 matched seconds, 95% matched observed duration, and
  no positive-duration minority reference label. Endpoints must also match
  their supported reference labels. Otherwise the edge is undecidable, with
  explicit reasons. These settings are not optimized on the resulting rates.
- An edge is correct/wrong when its supported endpoint reference labels are
  equal/different. Across a gap this evaluates endpoint correspondence only;
  the gap interior remains unassessed. It does not count exchanges inside the
  gap or add accepted trajectory coverage there.
- Temporal groups have at most a 0.25 s span of edge end times. The grouping
  does not use transitive single linkage, merge raw intervals, adjudicate an
  encounter, or establish independence for cross-validation. Members, review
  frame ranges, and the unreviewed status remain explicit.
- Observed assignment changes and raw disagreement intervals are retained
  separately, including changes inside mixed tracks. Thus a small number of
  assessed wrong edges cannot be read as the full number of failures.

Report correct/wrong/undecidable edges, assessed/all-edge coverage, and their
denominators alongside the existing fixed-anchor correct/wrong fish-time and
longest continuous correct intervals. Edge counts and fish-time answer different
questions; greater statistical efficiency has not been demonstrated.
Edge boundaries and the assessable subset can differ between lambda settings.
These tables are therefore not a matched-event method ranking; that comparison
requires the subsequently adjudicated, common event set.

In the baseline lambda 0.767 arm, segment-level reference durations reconcile
exactly to the existing fixed-anchor totals. Of **848.417 wrong-ID fish-seconds
across all three fish**, **788.132** occur in whole local tracks passing the
support/purity policy, **58.327** in mixed tracks, and **1.958** in tracks with
insufficient support. This supports distinguishing local tracking from global
naming, but does not identify all causal joins or establish a swap-event count.
The 13 mixed tracks cover 612.591 observed fish-seconds, of which 575.189 have
reference matches; they cannot be treated as 13 uniformly wrong trajectories.

The completed V2 ledgers have **56/56 identical CSV tables** between baseline
and repair. All seven C30 replays in each run reproduce the saved permutations
and decision lists exactly. Counts are unchanged:

| Lambda | Correct | Wrong | Undecidable | All candidate edges |
| --- | --- | --- | --- | --- |
| 0 | 54 | 23 | 1491 | 1568 |
| 0.1 | 67 | 6 | 1110 | 1183 |
| 0.25 | 70 | 4 | 1056 | 1130 |
| 0.5 | 77 | 3 | 1108 | 1188 |
| 0.767 | 72 | 2 | 1147 | 1221 |
| 2 | 66 | 2 | 1207 | 1275 |
| 8 | 63 | 1 | 1361 | 1425 |

At lambda 0.767, **74/1221 (6.06%)** of candidate edges are assessable under
the fixed policy. The two wrong endpoints share one temporal review candidate.
The 302 total temporal review groups are not 302 independently verified events.

## C30 and geometric context

The C30 audit calls the saved, unchanged stitch implementation with the same
inputs and observes its execution. It must reproduce the complete saved slot
permutation and the exact join/refusal lists before its diagnostics are exported.
The audit records candidate boundaries, zone decisions, velocity invocation and
adoption, refusal reasons, and actual history endpoints/durations on the native
clock. The configured velocity history is 50 native frames; it is not silently
treated as exactly 0.3 acquisition seconds.

C30 decisions are associated with final-output edges by temporal inclusion or
proximity, explicitly distinguished. This is not causal attribution: C34 may
change the final correspondence, and a final wrong edge does not by itself prove
that an earlier velocity proposal was wrong. Physical correctness remains
unadjudicated until source/reference review supports it.

Context columns contain nearest-other-fish distance, signed wall clearance,
signed depth below the water surface, and contributing-camera counts at each
endpoint. Tank dimensions come from the pinned calibration manifest. Negative
clearance is retained, not clipped. Reflection classification is unavailable;
BORIS is not yet aligned. Neither is silently recorded as false. No appearance
or body-length feature has been measured or used for a decision.

## Execution and validation

Use the locked project virtual environment and a new output run directory:

```powershell
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/run_row_mapping_repair.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/compare_tracker_rows.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_join_ledger.py --variant baseline
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_join_ledger.py --variant repair
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_video_review.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_full_contact_sheets.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/build_repair_notebook.py
.venv/Scripts/python.exe analyses/20261031_p11_identity_evaluation/package_row_mapping_repair.py
```

The runner selects its explicit configuration through `MEDAKA3D_P11_CONFIG`,
which child stage processes inherit. The default October 31 configuration is
not edited. Existing result directories are refused, not overwritten. The
baseline ledger is written under the new candidate run, never into the baseline.

Focused checks cover reordered and in-place reordered returns, identical point
coordinates, nonzero starting frames, untracked objects, malformed returned
sets, the pinned real SLEAP tracker, constructed identity exchanges, mixed
tracks, missing reference, low support, native-duration weighting, bounded
temporal grouping, and unchanged C30 replay with measured history spans.
The full repository suite is outside this change's validation scope.

Executed focused validation: **66 passed**, with the same two all-NaN median
warnings in the synthetic ghost-zone stitch tests. Ruff passed for the changed
adapter, evaluation helper, new tests, and this analysis directory. These checks
do not certify the research identities.

[row_mapping_review.ipynb](row_mapping_review.ipynb) was executed from a fresh
project-virtualenv kernel, in order, with four completed code cells, no error
outputs, and one embedded figure. Its HTML copy is `row_mapping_review.html`
in the shared run directory. The comparison figure was visually inspected.
The notebook runtime emitted Windows event-loop fallback and local kernel
transport warnings; execution completed. The two synthetic-test warnings and
these runtime notices do not stand in for an unrun full-repository test suite.
The first executed notebook is retained in `notebook_preview_v1`; its unused
display-loop variable triggered Ruff B007. The final notebook renames that
variable and was executed again from a fresh kernel, without changing calculations.

The final package includes `handoff_source`, source checksums, an artifact
manifest, and `handoff_checks.json`. Packaging rechecks the consumed frozen
baseline artifacts and both the original human review and its curated backup.

## Concrete baseline review candidate

At native frame **154,838** (932.878944810 s), lambda 0.767 has two supported
wrong endpoint joins: M1 to M2 and M2 to M1. They belong to the same temporal
review candidate, not two independently established encounters. The unchanged
C30 log records a refusal at this boundary: median improvement **1.98 mm**,
velocity improvement **8.95 mm**, and minimum median separation **7.3 mm**.
The trace finds a 50-frame history from 154,787 to 154,837, spanning
**0.3012452102 s**. Velocity and median calculations propose the same permutation,
but the median improvement is below v4's **2.5 mm** prerequisite.

This is a specific threshold/geometry review target, not authorization to tune
that prerequisite on this one failure. Single native-frame source previews show
the interacting fish near the tank floor, with overlap in the side views; a
still frame cannot establish continuity through the crossing. No new physical
identity labels or adjudicated event were created from that inspection.

The video review pack is configured in `config_video_review_v1.toml`. It includes
this paired wrong endpoint candidate, the temporally nearest supported correct
join, and another nearby correct join with fish separation below 20 mm. These
are purposive diagnostic examples, not a random sample or held-out partition.
Each clip retains every native frame in a two-second context on each side and
plays at 25 frames per second. Burned-in native indices and acquisition times,
plus per-frame CSV mappings, distinguish source time from slowed playback time.
The existing equal-index camera alignment is retained; hardware synchronization
is not newly certified. Source videos are checksum-verified. Exports do not
create new human annotations or adjudicate event boundaries.

All three exports contain **663 consecutive native frames**. Their encoded
frame counts and per-frame source mappings were checked, and the encoded
three-camera boundary image was visually inspected. The video banners sit
outside the source images. Initial contact-sheet labels covered the top image
margin; that presentation defect is preserved in the first export and corrected
in `full_contact_sheets_v2`. Use `video_review_v1/index_full_frame.html` for the
review pack. The corrected contact sheets place all labels outside the images.

## Attempts and limitations

`join_ledger_v1/baseline` is preserved as an initial attempt. It classified every
gap edge as undecidable even when its endpoint reference labels were supported.
V2 corrects that definition: assess the endpoint join, keep the interior unknown.
V1 also redundantly described an endpoint as conflicting when no supported label
was available; V2 reports the actual missing-support reason. Support thresholds
and candidate-group tolerance did not change.

The review labels share upstream geometry and `watched` means displayed, not
an independent frame-by-frame identity certification. This remains one trio.
The ledger is a review aid, not accepted trial-wide identity-bearing data.

The initial follow-up proposal required source-video review before another
experiment. That gate was too restrictive: the existing manual annotation can
already support geometry/motion comparisons and reference-guided diagnostics.
The user subsequently deferred appearance and requested localized flip warnings;
see [the executed geometry/motion follow-up](GEOMETRY_MOTION_RESULT.md). Additional
source review should resolve conflicts found by experiments. Any future appearance
comparison still needs an explicit event/time-block partition without image leakage.
Human review labels must be backed up to curated shared storage immediately.
No review-time estimate, swap-event count, or achievable accuracy is inferred
from temporal candidate counts.

The curated `catalog/experiment_list/group_list.csv`, parsed through
`medaka3d.cohort`, defines P's female as `yellow`, M1 as `red`, and M2 as `white`;
it supplies no relative-size descriptor for these males. The source and its
checksum are frozen in `cohort_context_v1`. This does not newly verify that the
review's names or any particular source-frame fish match those descriptors.
P's note is `only change female`; C is also flagged by the cohort helper as
possibly sharing individuals. A future across-group holdout must resolve those
reuse notes rather than assuming group letters guarantee disjoint fish.

## Reserved appearance and body-length comparison

This protocol is unexecuted and appearance is currently deferred by the user.
Existing supported manual labels may be used without a blanket new-review gate;
temporal candidate IDs alone are not an independent evaluation split.

1. Freeze an event manifest with native start/stop frames, participating local
   segments, reference-support status, source-video hashes, adjudication, and
   any unresolved endpoints. Unsupported examples stay out of supervised
   training and remain in the declared review/coverage denominator.
2. Assign folds before feature/model selection. Keep each reviewed crossing
   together; exclude overlapping crops and shared tracklet images across folds,
   and declare temporal buffers around held-out events. If this leaves too few
   independent review units, report that limitation rather than falling back
   to a random frame split. This recording still contains only one trio.
3. Freeze the repaired geometry and the same candidate-edge set. Compare
   geometry alone, geometry plus appearance, geometry plus 3D body length,
   and their combination separately. Extract appearance features from the pinned
   original camera frames. Missing nodes or invalid length estimates
   remain missing. Body length shares reconstruction errors with geometry;
   it is not an independent visual observation.
4. Use training/validation events to set any confidence or abstention threshold.
   On held-out events, report the paired outcome table: wrong-to-correct joins,
   correct-to-wrong joins, unchanged joins, and abstentions, with the unassessed
   reference subset and all-edge denominator visible. Do not infer join accuracy
   from the number of adjacent frames voting for one class.
5. Replay supported connection changes with the same fixed pre-window anchor
   and unchanged positions. Report subsequent wrong-ID fish-time, correct-time
   coverage, missing/unresolved coverage, and longest uninterrupted correct
   intervals alongside the connection outcomes. Do not choose a lambda or
   publish a review-time/accuracy target from these diagnostic examples.

BORIS remains a possible additional source only after physical naming, camera
and recording time, and spatial correspondence are checked. Its usable recording
count is not asserted here; the repository's historical inventories disagree.
