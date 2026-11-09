# Identity candidates S and G on fifteen unreviewed recordings (K_2, P_2, E_1, E_2, G_1)

Status: **registered 2026-11-09 before any run of this configuration existed.**
This executes a frozen experimental method, not a scoring study: none of the
fifteen recordings has an ID-pass except E_1_1, and no reference is read by
any step. Findings appear only under "Findings" after execution.

## Question

For the fifteen recordings with 3-node detections and chain outputs but no
human identity review, what do the frozen appearance candidates S (default)
and G (separate, optional) look like: how much of the chain-valid time after
300 s does each name, how much does each withhold, and how many gaps have
disagreeing anchors? Nothing here measures accuracy. The outputs are the
inputs of a later review, not a delivered identity.

Owner: Machiroi (supervisor). PR: [#4](https://github.com/oist/medaka-3d/pull/4). Policy:
[separate S and G outputs](../../docs/decisions/2026-11-09-separate-s-g-outputs.md).

## Inputs, fixed and read-only

| Input | Location (under the data parent) | Notes |
| --- | --- | --- |
| Chain output | `work/c34_microflip/<rec>__chain-c28-c33-c30v4-c34/C34-label__<rec>__lam0.767.npz` | one file per recording, lambda 0.767 as in the six-recording study |
| Detections | `work/c01_s16_predictions/bottomup/<rec>__<view>.npz` | bottom-up 3-node model `retrain_20260810_3node_reviewed`, inferred on a Windows machine with `--max_instances 6` (per-recording provenance beside the files) |
| Video | `medaka_202512/<rec>/{top_cam/cam0_top.mp4, side1_cam/cam1_side.mp4, side2_cam/cam2_side.mp4}` | crops decoded from video, frame step 8 |
| Frame times | `medaka_202512/<rec>/top_cam/frametimes.npy` | |

Recordings, in run order: K_2_1, K_2_2, K_2_3, P_2_1, P_2_2, P_2_3, E_1_1,
E_1_2, E_1_3, E_2_1, E_2_2, E_2_3, G_1_1, G_1_2, G_1_3 (full names in the
configuration). K_2 and P_2 are later recordings of the same animals as K_1
and P_1. E_1 and E_2 are repeated trials of the same E trio; G_1 is a second
new trio. The fifteen recordings cover four groups (K, P, E, G), not five
independent trios (`medaka3d.cohort`).

## Method, fixed

`analyses/20261031_p11_identity_evaluation/run_appearance_clean.py` with
`config_appearance_fifteen_v1.toml` (same folder): the Registration 4
pipeline of the appearance lab, unchanged from `config_appearance_six_v1.toml`
except for the recording blocks and run ids (`appearance_unrev_20261109_<key>`).
Steps per recording: bundle, extract, prepare, train (one encoder per
recording, GPU), evaluate (writes `evaluation/identity_tracks_{N_carry,S,G}.npz`
and label-free diagnostics). Then the two policy commands export S and G
separately:

```powershell
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_s.py --run <run> --sha256 <recorded S sha> --output <data parent>\work\identity_candidates\<rec>\<run id>\S
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_g.py --run <run> --sha256 <recorded G sha> --output <data parent>\work\identity_candidates\<rec>\<run id>\G
```

G is absent by construction when no gap has disagreeing anchors; that case
is recorded, not filled. Runs are staged on local disk
(`appearance_staging_dir` of `configs/paths.local.toml`) because the share
runs at about 10 MB/s, and copied afterwards to `work/appearance/<rec>/<run id>/`
without the intermediate crop arrays, with a SHA-256 manifest.

Execution order: a driver runs bundle, extract and prepare for the
recordings in sequence (video decode, CPU) while a second thread runs
train, evaluate and the exports on each prepared recording (GPU). Failures
are recorded per step and the next recording proceeds; nothing is retried
with changed settings.

## Quantities reported (label-free)

Per recording, from `evaluation/summary.json` and the identity tracks, on
recording time >= 300 s where the chain has three valid slots:

- intervals by naming kind (global, partial_two, partial_one, unknown),
  anchors, gaps, gaps with disagreeing anchors, gap seconds, switch rate;
- named and withheld fish-seconds of N_carry, S and G, and the fraction of
  chain-valid fish-time named;
- training facts (steps, stopping silhouette, cross-camera agreement);
- whether G exists.

No threshold is applied and no candidate is ranked here; the named fraction
is not a coverage figure (chain-invalid time is reported separately).

## Limits declared now

- No accuracy is measured. E_1_1 has an ID-pass; scoring it against its review
  is a separate step to be registered in the six-recording study's framework
  (its reference adapter and quantities), not part of this run.
- The per-recording encoder is trained on the recording itself with the
  stopping rule of Registration 4; a trio whose fish look alike may stop at
  the step cap rather than the silhouette target, and the diagnostics will
  say so.
- K_2 and P_2 do not test generalisation to new animals; E and G do, but
  only descriptively until reviewed.
- Detections of these fifteen recordings come from the capped Windows
  inference, not the uncapped cluster inference used for K_1.

## Findings

Executed 2026-11-09 01:05--05:48 by `run_fifteen.py` (this folder; its state
file `run_fifteen_state_20261109.json` records every step, return code and
duration). Per recording: bundle 6--7 s, extract 893--960 s, prepare
172--196 s, train 193--365 s, evaluate 22--48 s. **14 of 15 recordings ran
through evaluate and the S export; P_2_3 failed in evaluate.** G exists for
12 recordings and is absent by construction for G_1_1 and G_1_2 (no gap
with disagreeing anchors). The label-free quantities below come from
`diagnostics.py` (output `diagnostics.csv`); none of them is an accuracy.

Exports: `work/identity_candidates/<rec>/appearance_unrev_20261109_<key>/S/`
for the 14 evaluated recordings and `.../G/` for the 12 with G, each holding
`identity_tracks_{S,G}.npz`, `identity_policy.json` and the export
provenance. Runs copied to `work/appearance/<rec>/<run id>/` without the
crop arrays by `archive_runs.py` (manifest `staged_copy_manifest.json` in
each destination).

### P_2_3: evaluate failed, not retried

Traceback in `run_fifteen_p23_evaluate.log`: `label_centroids` returned no
centroid for two of the three identities and `np.stack` failed. Cause,
read from the run's `clean/<view>.npz` metadata: interval 0 (the seed
interval, 300.00--300.99 s, 164 frames) has clean crops per slot of
top 0 / 5 / 0, side1 0 / 0 / 4, side2 0 / 7 / 0; with `MIN_CROPS = 5` two
slots have no descriptor in any camera, so `constrained_labels` initialises
one centroid, every other interval's cost is NaN, and all 356 remaining
intervals stay `unknown` (`evaluation/interval_names.csv` of the run).
Bundle, extract, prepare and train of P_2_3 are normal (72,103 clean top
crops; both encoders reached the silhouette target). P_2_1's seed interval
is shorter (0.51 s) but had descriptors for all three slots, so seed
duration alone does not predict the failure. The frozen method seeds on
interval 0 regardless of its evidence; choosing another seed interval is a
different algorithm and needs its own registration and command (design
rule 5), so P_2_3 has no S or G from this run.

### Structure of the chain after 300 s (lower-risk intervals and gaps)

Intervals are named `global` (all three slots), `partial_two`, `partial_one`
or `unknown`; anchors are the `global` and `partial_two` intervals; a gap is
the stretch between consecutive anchors; the switch rate is disagreeing
gaps per gap second.

| Recording | seed interval s | lower-risk intervals (median s) | global / partial_two / partial_one / unknown | anchors | gaps | disagreeing gaps | gap s | switch rate /s | G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K_2_1 | 4.42 | 640 (0.65) | 246 / 35 / 71 / 288 | 281 | 280 | 47 | 377.4 | 0.125 | yes |
| K_2_2 | 6.69 | 351 (1.36) | 187 / 27 / 47 / 90 | 214 | 213 | 9 | 165.4 | 0.054 | yes |
| K_2_3 | 5.24 | 181 (6.28) | 148 / 3 / 9 / 21 | 151 | 150 | 9 | 52.4 | 0.172 | yes |
| P_2_1 | 0.51 | 211 (2.58) | 137 / 8 / 28 / 38 | 145 | 144 | 16 | 112.8 | 0.142 | yes |
| P_2_2 | 4.06 | 174 (4.72) | 126 / 6 / 21 / 21 | 132 | 131 | 17 | 108.4 | 0.157 | yes |
| P_2_3 | 0.99 | 357 (1.19) | evaluate failed | | | | | | |
| E_1_1 | 2.60 | 372 (1.66) | 199 / 22 / 70 / 81 | 221 | 220 | 12 | 239.2 | 0.050 | yes |
| E_1_2 | 6.05 | 265 (2.05) | 163 / 11 / 38 / 53 | 174 | 173 | 8 | 100.4 | 0.080 | yes |
| E_1_3 | 11.29 | 223 (3.00) | 139 / 9 / 27 / 48 | 148 | 147 | 10 | 139.7 | 0.072 | yes |
| E_2_1 | 9.68 | 507 (1.13) | 249 / 36 / 78 / 144 | 285 | 284 | 24 | 271.2 | 0.089 | yes |
| E_2_2 | 31.94 | 289 (2.35) | 170 / 27 / 28 / 64 | 197 | 196 | 6 | 154.1 | 0.039 | yes |
| E_2_3 | 20.47 | 263 (2.33) | 168 / 15 / 32 / 48 | 183 | 182 | 13 | 104.0 | 0.125 | yes |
| G_1_1 | 1.67 | 76 (11.49) | 68 / 2 / 0 / 6 | 70 | 69 | 0 | 16.8 | 0.000 | no |
| G_1_2 | 3.48 | 58 (12.02) | 49 / 3 / 3 / 3 | 52 | 51 | 0 | 18.7 | 0.000 | no |
| G_1_3 | 97.38 | 33 (36.50) | 33 / 0 / 0 / 0 | 33 | 32 | 1 | 6.3 | 0.159 | yes |

### Named and withheld fish-time (chain-valid frames at >= 300 s, three valid slots)

Planned post-300 s time is 1800.3 s per recording (5400.8 fish-s); the
chain-valid column is the denominator used here, and the difference is the
chain-invalid time (0.1--21.6 s; E_1_1 has 21.6 s). Named % is named
fish-time over three times the chain-valid time. Withheld is in fish-s.
"G - S" is G's named fish-s minus S's; "conflict" is fish-time where both
name a slot and disagree.

| Recording | chain-valid s | N_carry named % | S named % | S withheld | G named % | G withheld | G - S | S/G conflict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K_2_1 | 1798.1 | 82.5 | 94.7 | 284.8 | 95.0 | 268.6 | +16.3 | 1.37 |
| K_2_2 | 1799.6 | 94.2 | 99.4 | 34.6 | 99.7 | 17.7 | +16.9 | 0.00 |
| K_2_3 | 1800.0 | 98.8 | 99.8 | 12.3 | 99.8 | 11.6 | +0.7 | 0.00 |
| P_2_1 | 1799.7 | 95.7 | 98.5 | 83.6 | 98.5 | 79.2 | +4.4 | 0.54 |
| P_2_2 | 1799.7 | 96.5 | 98.1 | 104.2 | 99.1 | 50.3 | +54.0 | 0.40 |
| P_2_3 | 1796.7 | evaluate failed | | | | | | |
| E_1_1 | 1778.7 | 91.2 | 98.2 | 94.1 | 98.7 | 67.3 | +26.8 | 0.00 |
| E_1_2 | 1799.9 | 96.3 | 99.0 | 54.7 | 99.0 | 53.3 | +1.5 | 0.00 |
| E_1_3 | 1798.8 | 95.0 | 98.2 | 99.6 | 98.3 | 90.8 | +8.8 | 0.00 |
| E_2_1 | 1798.4 | 89.2 | 97.3 | 145.0 | 98.0 | 110.2 | +34.8 | 0.00 |
| E_2_2 | 1798.9 | 94.5 | 98.1 | 101.4 | 97.9 | 115.5 | -14.2 | 0.00 |
| E_2_3 | 1799.1 | 96.6 | 99.7 | 13.7 | 99.6 | 19.7 | -6.1 | 0.15 |
| G_1_1 | 1800.1 | 99.8 | 100.0 | 0.0 | absent |  |  |  |
| G_1_2 | 1800.2 | 99.7 | 100.0 | 0.0 | absent |  |  |  |
| G_1_3 | 1800.2 | 100.0 | 100.0 | 0.3 | 100.0 | 0.2 | +0.0 | 0.00 |

Across the 14 evaluated recordings S names 94.7--100.0% of chain-valid
fish-time and withholds 0.0--284.8 fish-s; N_carry names 82.5--100.0%.
Withholding follows fragmentation: K_2_1 (640 intervals, median 0.65 s, 288
unknown) and E_2_1 (507 intervals, 144 unknown) withhold the most; the G_1
trio has 33--76 long intervals and is almost fully named by N_carry alone.
G names between 14.2 fish-s less (E_2_2) and 54.0 fish-s more (P_2_2) than
S; where both name, they disagree on at most 1.4 fish-s per recording
(K_2_1 1.37, P_2_1 0.54, P_2_2 0.40, E_2_3 0.15, zero elsewhere). These are
descriptions of the candidates, not evidence that either is right.

Independent readout of `diagnostics.csv`, 2026-11-09: S names 98.638% of
pooled chain-valid fish-time across the 14 completed recordings. Across the
12 paired recordings, G names more time on 10 and less on two, with a net
increase of 143.916 fish-s; G_1_3 contributes only 0.012 fish-s of that gain.
The total simultaneous S/G disagreement is 2.458 fish-s. This does not
measure the correctness of G's additional names and does not change the
separate-output policy.

### Training and a note on `name` versus position validity

All 30 encoders, including both models of P_2_3, stopped at the silhouette
target (0.9189--0.9734) between 300 and 1700 steps. On the 14 evaluated
recordings, nearest-centroid agreement of the calibration crops is
0.986--0.995 in every camera. The identity-track `name` array is assigned per
interval or gap and is not masked by chain validity: S carries a name on
0.2--4.6 fish-s of chain-invalid frames per recording and G on up to 21.0
(E_1_1, whose invalid time is 21.6 s), where `xyz_by_identity` is NaN.
Consumers must take coverage from finite positions, not from `name`.

| Recording | top steps | top silhouette | side steps | side silhouette | S named on chain-invalid fish-s | G named on chain-invalid fish-s |
| --- | --- | --- | --- | --- | --- | --- |
| K_2_1 | 400 | 0.929 | 1700 | 0.935 | 4.4 | 4.2 |
| K_2_2 | 300 | 0.933 | 1000 | 0.919 | 1.9 | 1.9 |
| K_2_3 | 1000 | 0.960 | 1400 | 0.919 | 0.9 | 0.9 |
| P_2_1 | 500 | 0.945 | 900 | 0.924 | 1.4 | 1.5 |
| P_2_2 | 800 | 0.942 | 900 | 0.935 | 1.5 | 1.5 |
| P_2_3 | 500 | 0.929 | 1000 | 0.928 |  |  |
| E_1_1 | 800 | 0.973 | 1100 | 0.940 | 4.6 | 21.0 |
| E_1_2 | 300 | 0.934 | 1200 | 0.947 | 1.0 | 1.0 |
| E_1_3 | 700 | 0.962 | 1000 | 0.932 | 3.4 | 3.4 |
| E_2_1 | 700 | 0.939 | 900 | 0.939 | 4.3 | 4.7 |
| E_2_2 | 900 | 0.954 | 1100 | 0.940 | 2.9 | 2.4 |
| E_2_3 | 600 | 0.941 | 900 | 0.945 | 3.0 | 2.9 |
| G_1_1 | 600 | 0.971 | 800 | 0.924 | 0.4 |  |
| G_1_2 | 500 | 0.964 | 500 | 0.922 | 0.2 |  |
| G_1_3 | 700 | 0.971 | 1000 | 0.935 | 0.2 | 0.2 |

### What this run does not say

No accuracy is measured. E_1_1's ID-pass remains unscored (separate
registration). K_2 and P_2 are later recordings of reviewed animals; the
additional groups are E (both E_1 and E_2) and G. Whether the K_1 and P_1
per-trio encoders carry over to K_2 and
P_2 without retraining is tested, label-free, in
[analyses/20261109_cross_trial_appearance](../20261109_cross_trial_appearance/README.md)
(2026-11-09). The named fractions are of chain-valid time after 300 s
and are not coverage figures.

The supervisor reports that E_1_1 scoring and S/G video checks are being
conducted separately; their outcomes are not incorporated into this run.

## Execution evidence

Generated diagnostics, state and logs remain outside Git. The exact local
files cited above, together with pre-commit source snapshots, were copied
without alteration to
`work/analyses/identity_commit_review/identity_commit_review_20261109/`
under the configured data parent, preserving their repository-relative
paths. `manifest.json` records sizes and SHA-256 hashes, verified after
copy. Local copies remain available and are ignored by Git.

## Attempts and changes

| Date | Change or attempt | Evidence/output run | Outcome and next decision |
| --- | --- | --- | --- |
| 2026-11-09 | Registered; driver started 01:05 | `appearance_unrev_20261109_*` | 14 of 15 evaluated and exported (S for 14, G for 12) by 05:48; P_2_3 evaluate failed on a seed interval without descriptors for two slots, not retried; G absent for G_1_1 and G_1_2 (no disagreeing gap). Next decision: whether to register a seed-interval rule as a separate command |
| 2026-11-09 | Runs copied to `work/appearance/<rec>/<run id>/` without crop arrays | `archive_runs.py`, `staged_copy_manifest.json` per destination | completed 2026-11-09: 15 runs, 1,070 files, 62.1 GB in 74 min (about 14 MB/s), every file SHA-256 verified after copy; P_2_3 copied with its partial evaluation outputs (interval_names.csv, boundaries.csv) |
