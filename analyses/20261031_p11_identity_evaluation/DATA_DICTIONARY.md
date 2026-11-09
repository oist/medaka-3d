# P-1-1 evaluation outputs

The run evaluates one recording, `20260112_P_1_1_FCMC`. It is not a population
benchmark. Read `analysis.html` first; the executed notebook contains the same
figures and inspectable calculations. All CSV files use a header row and UTF-8.

## Tracking exports

`scoring_rerun_v1/automatic_pectoral_3d_lam*.csv.gz` has one row per native
frame and automatic slot after 300 seconds. Missing observations remain as rows.
The three original numeric slots are not guaranteed persistent fish identities.
CSV floating values use eight significant digits and lossless gzip compression
level 3. Use the NPZ arrays for the full stored coordinate and timestamp precision.

| Field | Meaning |
| --- | --- |
| `frame` | Zero-based native index; acquisition counter equals `frame + 1` |
| `recording_time_s` | Native top-camera elapsed acquisition time, seconds |
| `raw_slot` | Original automatic output slot, 0, 1, or 2 |
| `predicted_anchor_label` | F/M1/M2 name mapped once from the predeclared [298, 300) s anchor; unvalidated outside it |
| `position_valid` | Finite XYZ with at least two contributing camera rows; not an independent spatial accuracy certification |
| `label_available` | Position valid and automatic output identity label nonnegative |
| `output_identity_label` | C34-label numeric output; -1 means unavailable |
| `local_track_id` | C34-gated local segment ID; -1 means unresolved; IDs do not establish continuity between segments |
| `identity_support` | Always `automatic_unvalidated`; do not treat these exports as accepted identity-bearing trajectories |
| `X`, `Y`, `Z` | Pectoral 3D position in millimetres in the pinned calibration coordinate system |
| `residual_mm` | Pipeline geometric residual; small values do not prove identity correctness |

Use `position_valid` for position availability and retain gaps. No spatial
interpolation was added by this evaluation. Pipeline NPZ outputs additionally
retain source-camera detection row indices and full-recording arrays.

## Scoring tables

`identity_summary.csv` contains one row per lambda, matching threshold, and
reviewed individual. The main threshold is 5 mm; 2 and 10 mm are sensitivity
checks. The competitor-distance margin is 2 mm throughout.

- `expected_seconds`: complete post-preparation recording duration for one fish.
- `correct_id_seconds`, `wrong_id_seconds`: unambiguous spatial correspondences
  agreeing or disagreeing with the one fixed identity mapping.
- `id_agreement_on_scored_time`: correct / (correct + wrong).
- `scored_time_fraction`: (correct + wrong) / expected.
- `correct_id_time_fraction`: correct / expected; unavailable time remains in
  the denominator.
- `*_frames`: corresponding native observation counts, including the final
  observation, whose elapsed duration is zero.
- `*_run_*`: uninterrupted observed runs; gaps and unresolved matches break runs.
- `unprocessed_leading_seconds`: interval from 300 s to the first eligible
  acquisition, retained in expected coverage and not extrapolated backward.

`reference_unavailable` includes unavailable XYZ, unwatched frames, pre-analysis
frames, and uncertain/bad3d/nodes_bad/interpolated reference rows. Reference
flags are reported separately in `reference_inventory.json`. `prediction_missing`
means there is no usable predicted candidate. `distance_mismatch` and
`spatial_ambiguity` describe failed geometric matching gates. `anchor_unresolved`
means an accepted spatial match lacks a supported fixed name mapping.

`scoring_arrays_lam*.npz` retains the full native frame grid. `eligible` declares
the post-preparation window; `status_code` indexes the stored `status_names`.
The `matched_slot` is indexed by the stored `reference_labels`, with -1 for an
unaccepted correspondence. Distances and margins are diagnostic even when a
match is unaccepted. `duration_s` uses left-held acquisition intervals, with
zero duration before the main window and at the final acquisition.

`wrong_identity_intervals.csv` gives half-open frame and time intervals of
observed disagreement. It does not establish identity inside intervening gaps.
`assignment_change_events.csv` separates adjacent changes from changes observed
only at gap endpoints. These are not adjudicated crossing events.

`per_second_agreement.csv` assigns each native interval to the integer second
containing its starting acquisition. Bin durations can differ slightly from
one second; the plot divides by the actual `observed_bin_seconds`. The timeline
is an overview, not an exactly resampled one-Hz trajectory.

`gated_local_track_purity.csv` uses a retrospective majority label for each local
track. Many fragments may receive the same fish name. High local purity is not
high trial-wide identity accuracy. `proximity_context.csv` describes geometric
separation below 20 mm; it is not a courtship annotation or a causal comparison.

## Evidence and limitations

`inputs.json`, `timing_inventory.json`, input/code hashes, stage logs, execution
wrappers, and provenance sidecars record what ran. `validation.json` reconciles
the scoring totals, independently checks scalar correspondences, and compares
fresh output arrays with the preserved earlier run.

The root `config.toml` is the initial preflight configuration, preserved intact.
Per-stage configurations are in `source/execution_wrappers/*.json`; the final
scoring configuration is in `scoring_rerun_v1/scoring_configuration.json` and
the maintained analysis configuration is in `analysis_source/config.toml`.

The human review and automatic output share derived geometry. Scores measure
agreement with reviewed identities, not independent 3D localization accuracy.
`watched` only means a frame was displayed in the GUI, at any playback speed.
The unused `uncertain` flag does not prove every displayed frame was decidable.
No appearance model was trained or evaluated. No lambda was selected or promoted.

## Separately versioned row-mapping follow-up

`id_row_mapping_20261102_r1` preserves the baseline and uses a corrected source-row
adapter. See [ROW_MAPPING_REPAIR.md](ROW_MAPPING_REPAIR.md) for the comparison
protocol, input pins, and validation status. All ledger tables below are written
under the new run, including the tables calculated from the old baseline.

`join_ledger_v2/{baseline,repair}/lam*/` contains:

| Table | Unit and interpretation |
| --- | --- |
| `tracks.csv` | One automatic local ID; time-weighted reference-label support, including IDs without reference matches |
| `segments.csv` | One uninterrupted native-frame run with a valid position, a fixed output slot, and an unchanged local ID |
| `edges.csv` | One asserted same-slot continuation between successive segments; correct/wrong/undecidable endpoint correspondence |
| `event_candidates.csv` | One bounded-span temporal group of edges for review, with member edge IDs and half-open review frame ranges; not an adjudicated event |
| `c30_decisions.csv` | One C30 candidate boundary or turbulent-zone decision, obtained by replay that matches the stored output exactly |
| `c30_associations.csv` | One temporal relationship between a final C34 edge and a C30 decision; it does not establish causation or slot correspondence between stages |
| `observed_assignment_changes.csv` | Original reference-to-slot changes, including changes within mixed tracks; gap-endpoint changes do not establish the number of intervening exchanges |
| `wrong_identity_intervals.csv` | Original per-fish disagreement intervals, unmerged; overlapping M1/M2 intervals are fish-seconds, not distinct recording time |

Track and segment `label_status` distinguish `supported_reference_label`,
`mixed`, `insufficient_support`, `reference_incomplete`, and `unassessed`.
`majority_label` is retained as a diagnostic even when it is not supported.
`supported_reference_label` is null in that case; it is not a delivered
trial-wide ID. `purity` and `matched_fraction` use acquisition-duration weights;
they are not calibrated correctness probabilities.

`edges.outcome` evaluates the supported reference labels at the two ends of the
asserted continuation. `gap_interior_identity_status=unassessed` means no claim
is made inside a gap, even when its endpoint correspondence is correct.
`undecidable_reasons` preserves all applicable support limitations. Unassessed
edges remain in `edge_count`; correct/(correct+wrong) must be accompanied by
assessed/all-edge coverage. Neither denominator is a biological sample size.

Geometric endpoint context is in millimetres. `wall_clearance_signed_mm` is the
minimum signed distance to the four vertical tank walls.
`depth_below_surface_signed_mm` is calibrated water height minus Z. Negative
wall clearance or depth, or depth greater than calibrated water height, is not
clipped. `contributing_cameras` counts nonnegative source rows for that slot.
Reflection and BORIS status are explicitly unknown/not aligned.

C30 history endpoints are native frame indices; `velocity_history_span_seconds`
uses their native acquisition timestamps. `velocity_called` includes attempted
evaluations; consult `velocity_costs_available` and the unavailable reason.
`velocity_adopted` denotes a velocity or bridge correction, not proof that the
individual was correctly identified. `physical_identity_correctness` remains
`not_adjudicated`. An association marked `nearby_only` is weaker than temporal
inclusion and must not be silently treated as the cause of an edge outcome.

`summary.json` keeps the full expected fish-time separately from the summed
eligible-grid intervals. Their small difference is the initial partial interval
after 300 s, already documented above. V1 remains an explicitly superseded
diagnostic attempt; it is not mixed into V2 comparisons.
# Geometry/motion follow-up exports

The follow-up is `geometry_motion_risk_20261104_r3`; its source geometry remains
`id_row_mapping_20261102_r1`. These tables are exploratory review evidence, not
accepted identity-bearing trajectories. All times are native recording seconds.

| File / fields | Meaning |
| --- | --- |
| `lam*/warning_intervals_identity_core.csv` | Geometry-only warnings: <10 mm pair separation, >10 mm adjacent step, non-identity nearest assignment, or missing position; padded by 0.05 s. |
| `start_frame`, `stop_frame_exclusive` | Half-open native-frame interval; no forced maximum duration. |
| `start_s`, `end_s`, `duration_s` | Acquisition-time bounds and sum of eligible native hold durations. |
| `review_start_*`, `review_end_s`, `review_stop_frame_exclusive` | An additional 0.5 s of viewing context on both sides, clipped to the analysis recording range. |
| `reasons` | Comma-separated detector signals; no manual labels enter detection. |
| `missing_geometry_s` | Time in the warning with at least one invalid animal position. |
| `audit_v1/review_queue_lam*.csv` | Warning table plus retrospective reference evidence and a display priority. |
| `reference_group_ids`, `reference_group_count` | Reference-derived sustained change brackets contained in the warning; not detector inputs or physical swap counts. |
| `review_priority` | Assignment disagreement, large step, then proximity/missing; not a calibrated probability. |
| `lam*/lower_risk_intervals.csv` | Contiguous valid-position intervals outside the core warnings. Earlier global naming mistakes can persist here. |
| `slot*_reference_coverage` | Matched reference duration divided by segment duration for that raw slot. |
| `slot*_reference_labels_observed`, `slot*_reference_purity` | Number of matched physical labels and time-weighted majority fraction inside this interval. Missing reference does not vote. |
| `slot*_majority_label` | Diagnostic majority reference name, or unknown; not an automatic identity prediction. |
| `group_capture_*.csv`, `raw_capture_*.csv` | Full containment versus any overlap of reference transition uncertainty brackets, with explicit warning IDs. |
| `audit_v1/c30_connection_changes_lam*.csv` | Fish-level edges wrong under at least one method, distinguishing adjacent observations and gap endpoints. |
| `baseline_correct`, `candidate_correct` | Whether consecutive observations of the same reference fish keep their output slot; not trial-wide naming correctness. |
| `correction_summary.csv` | Fixed-anchor fish-seconds correct, wrong, repaired, newly wrong, and unassessed; all-time and declared time blocks. |
| `reference_endpoint_oracle` | Explicit reference-only link composition diagnostic; not deployable accuracy or an upper bound. |

The frozen source scorer's complete denominator remains in force. Missing,
unassessed, and the leading partial acquisition interval are not converted to
correct time. A clean-looking local segment does not resolve its relationship
to the trial anchor across an unchecked warning.
