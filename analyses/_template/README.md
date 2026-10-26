# Trajectory QC: analysis record template

**Status: synthetic tutorial; no research result.** Copy this directory for a
new question and replace the fields below. Keep all authored text in English.
Follow [the workspace guide](../README.md) and [AGENTS.md](../../AGENTS.md).
After copying, the workspace guide remains `../README.md`.

## Question and ownership

- Question: How much of the declared window has observed positions attached to
  supported trial-wide identities?
- Owner: UNASSIGNED
- Supervisor/reviewer: UNASSIGNED
- Issue/PR: Not opened
- Analysis status: Tutorial only
- Completion criterion: State the usable scope, missingness, evidence, and
  limitations, and let another person rerun the calculation.

## Inputs and method

- Recording/trial/group: SYNTHETIC / not applicable
- Extraction run and commit: Synthetic fixture; no extraction run
- Input SHA-256: See `config.toml`; checked before loading
- Upstream provenance and schema: This README describes the fixture only
- Identity evidence: Known by construction in the fixture; no animal reviewed
- Body reference: Invented point, in mm; not a measured anatomical node
- Analysis window: `[300, 306)` s of recording time; synthetic samples every 1 s
- Expected denominator: 6 time samples x 3 individuals = 18 fish-samples
- Acceptance rule: Finite, position-valid, observed, identity-supported, with a
  non-null trial-wide ID. This example consumes supplied flags; it does not
  establish that a real identity is correct.
- Research adaptation: Specify source columns, units/axes, body reference,
  frame/time and native-camera mappings, sampling, acceptance policy, and
  independent identity-evidence references before using real data.

## Run and inspect

Run `analysis.ipynb` using the project virtualenv. Its working directory must be
this analysis folder, as is usual when opening it in JupyterLab. Restart and run
all cells. From the repository root, a copied analysis can also be executed with
the `nbconvert` command in [the workspace guide](../README.md#setup-and-first-run).
For the unmodified template, use `analyses/_template/analysis.ipynb` as that
command's notebook path.

The last cell exports the summary, figure, frozen configuration, and provenance
under a unique run directory. Synthetic exports use ignored `_outputs/`; research
exports use configured `work_dir/analyses/`. The provenance covers the export
step and records the calculation's inputs/settings/source hash. Earlier failed
cells remain notebook errors; do not report a completed export if they fail.

## Findings

Template verification only: within the 18 expected synthetic fish-samples,
16 have finite coordinates, 15 are observed and position-valid, and 13 also have
a declared supported trial-wide identity. The primary example fraction is
13/18 (72.2%). These are teaching-fixture counts, not medaka coverage estimates.

For research, replace this section after inspecting executed results. Include
the output run ID, numerical results, plots, review scope, and the decision they
support. An unexecuted proposal is not a finding.

## Attempts and changes

| Date | Change or attempt | Evidence/output run | Outcome and next decision |
| --- | --- | --- | --- |
| Not started | Synthetic template supplied | No research run | Agree real inputs and scope with supervisor |

Record unsuccessful approaches and parameter changes here. Git records code
revisions; this table explains why the approach changed.

## Limitations and next step

This notebook assumes one regularly sampled recording, one body-reference point
per local segment/frame, at most three reconstructed fish per sample, and at
most one accepted observation per trial-wide ID per sample. It does not consume
raw SLEAP detection candidates, infer IDs, calibrate confidence, perform
statistical tests, or handle irregular sampling. Additional raw/reflection
detections must not be silently discarded to satisfy this example contract.

Next: assign an owner, pin a real export and its evidence, validate any adapter,
and update the configuration and interpretation. Preserve unsupported intervals
in the denominator. Do not infer F/M1/M2 names from anonymous A/B/C labels.

## Input table

`synthetic_tracks.csv` is a tiny, explicitly invented regular-grid fixture.
The `frame` field is a recording-frame identifier; do not infer real timestamps
or native-camera frame mappings from it. Every row is one local segment's body
reference at one time. A missing row is still part of the expected denominator.

| Column | Meaning |
| --- | --- |
| `frame` | Non-negative integer recording-frame index |
| `recording_time_s` | Documented common recording time in seconds |
| `segment_id` | Local continuous segment identifier, never an identity claim by itself |
| `trial_fish_id` | Supported anonymous trial-wide individual, or empty when unresolved |
| `x_mm`, `y_mm`, `z_mm` | Position in documented mm axes; missing values stay missing |
| `position_valid` | 0/1 flag for passing the declared position criteria |
| `identity_valid` | 0/1 flag; 1 requires a non-null, independently supported trial ID in research |
| `is_observed` | 0/1 flag distinguishing observed reconstruction from interpolation/prediction |

The fixture includes three preparation rows at 299 s, an absent position for A
at 301 s, unresolved correspondence at 302 s, an interpolated B position at
303 s, unresolved correspondence at 304 s, and a missing C row at 305 s. Thus the
denominator is not the number of rows loaded or successfully reconstructed.

Do not rename arbitrary tracker IDs to `trial_fish_id`, create support flags from
non-null coordinates, or fill missing rows to make a real file pass. A real
input needs the source export's data dictionary and identity evidence. Reject
or adapt incompatible formats explicitly.
