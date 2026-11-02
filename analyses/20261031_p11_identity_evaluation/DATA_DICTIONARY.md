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
