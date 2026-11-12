# Identity candidates S and G on six unreviewed recordings (G_2, O_1)

Status: **registered 2026-11-12 before any run of this configuration existed.**
This executes a frozen experimental method, not a scoring study: none of the
six recordings has an ID-pass and no reference is read by any step. Findings
appear only under "Findings" after execution.

## Question

For the six recordings that have 3-node detections and chain outputs but no
appearance run (G_2_1, G_2_2, G_2_3, O_1_1, O_1_2, O_1_3), what do the frozen
appearance candidates S (default) and G (separate, optional) look like: how
much of the chain-valid time after 300 s does each name, how much does each
withhold, and how many gaps have disagreeing anchors? Nothing here measures
accuracy. The outputs are the inputs of a later review, not a delivered
identity.

Owner: Machiroi (supervisor). PR: not opened. Policy:
[separate S and G outputs](../../docs/decisions/2026-11-09-separate-s-g-outputs.md).
This is the same question, method and reporting as
[the fifteen-recording run](../20261109_unreviewed_identity_candidates/README.md),
applied to the recordings whose chain outputs were written after that run
was configured.

## Inputs, fixed and read-only

| Input | Location (under the data parent) | Notes |
| --- | --- | --- |
| Chain output | `work/c34_microflip/<rec>__chain-c28-c33-c30v4-c34/C34-label__<rec>__lam0.767.npz` | one file per recording, lambda 0.767; SHA-256 and writing revision below |
| Detections | `work/c01_s16_predictions/bottomup/<rec>__<view>.npz` | bottom-up 3-node model `retrain_20260810_3node_reviewed`, inferred on a Windows machine with `--max_instances 6` |
| Video | `medaka_202512/<rec>/{top_cam/cam0_top.mp4, side1_cam/cam1_side.mp4, side2_cam/cam2_side.mp4}` | crops decoded from video, frame step 8 |
| Frame times | `medaka_202512/<rec>/top_cam/frametimes.npy` | |

Chain label files as read on 2026-11-12 before the run, with the
`environment` block of each `c34.provenance.json` (host `DESKTOP-9MO13KR`):

| Recording | SHA-256 of `C34-label__<rec>__lam0.767.npz` | c34 revision | c34 finished (UTC) |
| --- | --- | --- | --- |
| 20260109_G_2_1_FCMC | `9e636b913137e7893922f3f47bedb543bd3405e417aa803fb50fbf378031a9f2` | `50d56d0`, dirty | 2026-11-08 09:53 |
| 20260109_G_2_2_FCMC | `13b5037af54bf7516e5e7ce38558f42ca6eed057fb93af1a8bf6a25ba5e030d6` | `50d56d0`, dirty | 2026-11-08 19:56 |
| 20260109_G_2_3_FCMC | `8ac96b91666fe983a9e22ce46fc4f240ca4e0d98b145bf22b6e5cab9a59c88d7` | `50d56d0`, dirty | 2026-11-09 04:30 |
| 20260109_O_1_1_FCMC | `d5db35961ab890f922235522f35e5362677798d0b54231579b281239ed59848f` | `03cd1b6`, dirty | 2026-11-09 14:59 |
| 20260109_O_1_2_FCMC | `0a42332ff26ffec13a02ced7e077f49f15a1de20b9270e313e26d7083b67a963` | `03cd1b6`, dirty | 2026-11-09 23:41 |
| 20260109_O_1_3_FCMC | `25dd122a7496c0c8d86c75959b6c4ace5961c3d62a7b3c99c11d09f9a2523c08` | `3e81bf1`, clean | 2026-11-10 09:17 |

The bundle step hashes every input it reads (chain, gated chain, chain
provenance, three detection caches, frame times; videos during extraction)
into the run's provenance.

Recordings, in run order: G_2_1, G_2_2, G_2_3, O_1_1, O_1_2, O_1_3. By
`medaka3d.cohort`, G_2 is trial 2 of group G (the trial whose first trial,
G_1, is in the fifteen-recording run) and O_1 is trial 1 of group O; each is
three consecutive days of one trial. The six recordings are two trials of
two groups, not six independent units. `group_list.csv` carries no note of
shared or replaced individuals for G or O.

## Method, fixed

`analyses/20261031_p11_identity_evaluation/run_appearance_clean.py` with
`config_appearance_g2o1_v1.toml` (same folder): the Registration 4 pipeline
of the appearance lab. The configuration was checked before the run to equal
`config_appearance_fifteen_v1.toml` in `lab`, `lambda_mm`, `[crops]`,
`[train]`, `[gap]`, `[evaluate]` and every per-recording setting other than
the recording name, its output folder and its chain folder; run ids are
`appearance_unrev_20261112_<key>` with keys `g21 g22 g23 o11 o12 o13`. Steps
per recording: bundle, extract, prepare, train (one encoder per recording,
GPU), evaluate (writes `evaluation/identity_tracks_{N_carry,S,G}.npz` and
label-free diagnostics). Then the two policy commands export S and G
separately:

```powershell
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_s.py --run <run> --sha256 <recorded S sha> --output <data parent>\work\identity_candidates\<rec>\<run id>\S
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_g.py --run <run> --sha256 <recorded G sha> --output <data parent>\work\identity_candidates\<rec>\<run id>\G
```

G is absent by construction when no gap has disagreeing anchors; that case
is recorded, not filled. Runs are staged on local disk
(`appearance_staging_dir` of `configs/paths.local.toml`) and copied
afterwards to `work/appearance/<rec>/<run id>/` without the intermediate
crop arrays, with a SHA-256 manifest (`archive_runs.py`).

Execution order (`run_candidates.py`, the fifteen-recording driver with this
configuration): one thread runs bundle, extract and prepare for the
recordings in sequence (video decode, CPU) while a second runs train,
evaluate and the exports on each prepared recording (GPU). Failures are
recorded per step and the next recording proceeds; nothing is retried with
changed settings.

## Quantities reported (label-free)

Per recording, from `evaluation/summary.json` and the identity tracks
(`diagnostics.py`), on recording time >= 300 s where the chain has three
valid slots:

- intervals by naming kind (global, partial_two, partial_one, unknown),
  anchors, gaps, gaps with disagreeing anchors, gap seconds, switch rate;
- named and withheld fish-seconds of N_carry, S and G, and the fraction of
  chain-valid fish-time named;
- training facts (steps, stopping silhouette, cross-camera agreement);
- whether G exists.

No threshold is applied and no candidate is ranked here; the named fraction
is not a coverage figure (chain-invalid time is reported separately).

## Limits declared now

- No accuracy is measured and none can be: no ID-pass exists for any of the
  six recordings on 2026-11-12. O_1_3 also has no review package yet (its
  package build was interrupted on 2026-11-10); the package is not an input
  of this run.
- **The chain outputs were written from three recorded code states.** The
  committed chain tools (`c27`, `c28`, `c30`, `c33`, `c34`) are identical in
  `50d56d0`, `03cd1b6` and `3e81bf1`, but `tools/c24_uncapped_tracklets.py`,
  which serial C27 imports, differs: the source-row mapping repair (commit
  `3b5b64e`) is in `03cd1b6` and `3e81bf1` and not in `50d56d0`. So the G_2
  chains match the code state of the fifteen-recording inputs, and the O_1
  chains are the first appearance inputs written after the repair. The only
  recorded old/new comparison is P_1_1's (two labels differ, both in
  top-camera frame 20, no post-300 s difference;
  [ROW_MAPPING_REPAIR.md](../20261031_p11_identity_evaluation/ROW_MAPPING_REPAIR.md)).
  Five of the six provenance records say `git_dirty: true`; the uncommitted
  changes on the writing machine are not recorded and cannot be inspected
  from here.
- The runner writes a fixed note into `bundle/summary.json` ("Chain predates
  the 2026-11-02 adapter repair"). That note is not true of the O_1 runs;
  the runner is frozen and was not edited, so read the revision table above
  instead.
- The per-recording encoder is trained on the recording itself with the
  stopping rule of Registration 4; a trio whose fish look alike may stop at
  the step cap rather than the silhouette target, and the diagnostics will
  say so.
- The frozen method seeds identities on lower-risk interval 0 regardless of
  its evidence. P_2_3 failed in evaluate for that reason in the
  fifteen-recording run; a recording that fails the same way here is
  recorded as having no S or G and is not retried.
- G_2 does not test generalisation to new animals (same group as G_1); O_1
  is a group not seen by any earlier appearance run, described only.
- Detections come from the capped Windows inference, not the uncapped
  cluster inference used for K_1.

## Findings

Executed 2026-11-12 03:54--05:42 by `run_candidates.py` (this folder; its
state file `run_state.json` records every step, return code and duration).
Per recording: bundle 5--7 s, extract 819--882 s, prepare 175--194 s, train
189--362 s, evaluate 20--47 s. **All six recordings ran through evaluate and
the S export; no step failed.** G exists for five recordings and is absent by
construction for G_2_2 (no gap with disagreeing anchors). The label-free
quantities below come from `diagnostics.py` (output `diagnostics.csv`); none
of them is an accuracy.

Exports: `work/identity_candidates/<rec>/appearance_unrev_20261112_<key>/S/`
for all six recordings and `.../G/` for the five with G, each holding
`identity_tracks_{S,G}.npz`, `identity_policy.json` and the export
provenance. Each exported track file was re-hashed after the run and equals
the staged `evaluation/identity_tracks_{S,G}.npz` it was exported from; no
`G` folder exists for G_2_2.

### Structure of the chain after 300 s (lower-risk intervals and gaps)

Intervals are named `global` (all three slots), `partial_two`, `partial_one`
or `unknown`; anchors are the `global` and `partial_two` intervals; a gap is
the stretch between consecutive anchors; the switch rate is disagreeing
gaps per gap second.

| Recording | seed interval s | lower-risk intervals (median s) | warning s | global / partial_two / partial_one / unknown | anchors | gaps | disagreeing gaps | gap s | switch rate /s | G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G_2_1 | 9.88 | 28 (53.45) | 4.1 | 26 / 0 / 0 / 2 | 26 | 25 | 2 | 4.2 | 0.476 | yes |
| G_2_2 | 48.19 | 28 (44.16) | 4.1 | 27 / 0 / 0 / 1 | 27 | 26 | 0 | 4.2 | 0.000 | no |
| G_2_3 | 127.99 | 32 (26.17) | 6.6 | 30 / 0 / 0 / 2 | 30 | 29 | 1 | 7.0 | 0.144 | yes |
| O_1_1 | 7.59 | 231 (2.60) | 124.5 | 146 / 14 / 23 / 48 | 160 | 159 | 9 | 151.6 | 0.059 | yes |
| O_1_2 | 50.04 | 92 (10.44) | 102.1 | 76 / 1 / 6 / 9 | 77 | 76 | 3 | 105.1 | 0.029 | yes |
| O_1_3 | 26.78 | 136 (6.58) | 60.3 | 111 / 4 / 6 / 15 | 115 | 114 | 5 | 66.8 | 0.075 | yes |

The G_2 chains are split into 28--32 long lower-risk intervals with 4--7 s
of warning time, like G_1 in the fifteen-recording run (33--76 intervals).
The O_1 chains have 92--231 intervals and 60--125 s of warning time. G_2_1's
switch rate is two disagreeing gaps over 4.2 gap seconds, a ratio on a very
small denominator. Every seed interval is at least 7.6 s and had descriptors
for all three slots; the P_2_3 failure did not recur.

### Named and withheld fish-time (chain-valid frames at >= 300 s, three valid slots)

Planned post-300 s time is 1800.3 s per recording (5400.8 fish-s); the
chain-valid column is the denominator used here, and the difference is the
chain-invalid time (0.05--0.93 s). Named % is named fish-time over three
times the chain-valid time. Withheld is in fish-s. "G - S" is G's named
fish-s minus S's; "conflict" is fish-time where both name a slot and
disagree.

| Recording | chain-valid s | N_carry named % | S named % | S withheld | G named % | G withheld | G - S | S/G conflict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G_2_1 | 1800.2 | 99.96 | 99.99 | 0.5 | 99.99 | 0.4 | +0.02 | 0.00 |
| G_2_2 | 1800.2 | 99.99 | 100.00 | 0.0 | absent | | | |
| G_2_3 | 1800.2 | 99.91 | 100.00 | 0.2 | 100.00 | 0.2 | 0.00 | 0.00 |
| O_1_1 | 1799.7 | 94.01 | 98.72 | 69.3 | 98.61 | 75.3 | -5.95 | 0.00 |
| O_1_2 | 1799.6 | 95.07 | 95.88 | 222.4 | 97.47 | 136.5 | +85.85 | 0.00 |
| O_1_3 | 1799.3 | 98.12 | 98.88 | 60.4 | 98.88 | 60.7 | -0.34 | 0.00 |

S names 95.88--100.00% of chain-valid fish-time and withholds 0.0--222.4
fish-s; pooled over the six recordings S names 98.911% and N_carry 97.840%.
The G_2 trio is almost fully named by N_carry alone, as G_1 was. Withholding
is concentrated in O_1: O_1_2 withholds the most under S (222.4 fish-s),
where S adds 0.8 percentage points to N_carry and G names 85.8 fish-s more
than S; on O_1_1 and O_1_3 G names slightly less than S. Net over the five
paired recordings G names 79.6 fish-s more than S, almost all of it on
O_1_2.

Entry-by-entry comparison of the `name` arrays (frame x slot, whole
recording, unweighted): S and G never both name an entry differently on any
of the five paired recordings. G_2_3's G file is byte-identical to its S
file (its one disagreeing gap changes nothing); on G_2_1 G names four
entries (two frames) that S withholds. On O_1 the two differ in which
entries they withhold: S-only / G-only named entries are 6,773 / 5,804
(O_1_1), 1,919 / 16,181 (O_1_2) and 3,815 / 3,686 (O_1_3). These are
descriptions of the candidates, not evidence that either is right.

### Training and a note on `name` versus position validity

All 12 encoders stopped at the silhouette target (0.9175--0.9629) between
300 and 1400 steps; none reached the step cap. Nearest-centroid agreement of
the calibration crops is 0.990--0.999 in every camera. As in the earlier
runs, the identity-track `name` array is assigned per interval or gap and is
not masked by chain validity: S carries a name on 0.1--1.8 fish-s of
chain-invalid frames per recording and G on 0.1--1.5, where
`xyz_by_identity` is NaN. Consumers must take coverage from finite
positions, not from `name`.

| Recording | top steps | top silhouette | side steps | side silhouette | S named on chain-invalid fish-s | G named on chain-invalid fish-s |
| --- | --- | --- | --- | --- | --- | --- |
| G_2_1 | 500 | 0.963 | 600 | 0.937 | 0.2 | 0.2 |
| G_2_2 | 400 | 0.947 | 600 | 0.918 | 0.3 | |
| G_2_3 | 300 | 0.943 | 700 | 0.937 | 0.1 | 0.1 |
| O_1_1 | 600 | 0.942 | 1400 | 0.931 | 1.3 | 1.4 |
| O_1_2 | 700 | 0.939 | 1200 | 0.944 | 1.4 | 1.5 |
| O_1_3 | 600 | 0.948 | 1000 | 0.932 | 1.8 | 1.4 |

### What this run does not say

No accuracy is measured: none of the six recordings has an ID-pass, so no
candidate here has been compared with a human review, and a high named
fraction is not a low error rate. The named fractions are of chain-valid
time after 300 s and are not coverage figures.

The difference between G_2 and O_1 in fragmentation and withholding is not
attributed. The O_1 chains are a different group **and** were written after
the source-row mapping repair, and the two cannot be separated from these
six recordings; no O_1 chain exists from the pre-repair code state. G_2 is
the same group as G_1, so its agreement with G_1's pattern is one group seen
twice, not a replication.

Next step: these candidates are inputs for review. An ID-pass of a G_2 or
O_1 recording would allow scoring under the six-recording study's
registered framework; that scoring needs its own registration.

## Execution evidence

Generated diagnostics, state and logs remain outside Git (`.gitignore` of
this folder). The local files cited above (`run_state.json`, the per-step
logs holding each exact command line, `diagnostics.csv`,
`archive_runs.log`), together with pre-commit snapshots of this folder's
scripts and of `config_appearance_g2o1_v1.toml` (none modified after 03:53,
before the driver started) and of this README, were copied without
alteration to
`work/analyses/g2_o1_identity_candidates/g2_o1_identity_candidates_20261112/`
under the configured data parent, preserving their repository-relative
paths. `manifest.json` records sizes and SHA-256 hashes, verified after
copy. Local copies remain available and are ignored by Git.

## Attempts and changes

| Date | Change or attempt | Evidence/output run | Outcome and next decision |
| --- | --- | --- | --- |
| 2026-11-12 | Registered; configuration checked against the fifteen-recording configuration; chain label files hashed; driver started 03:54 | `appearance_unrev_20261112_*` | six of six evaluated and exported by 05:42 (S for six, G for five); G absent for G_2_2 (no disagreeing gap); no step failed and nothing was retried. Next decision: which G_2 or O_1 recordings get an ID-pass, so that these candidates can be scored under a new registration |
| 2026-11-12 | Runs copied to `work/appearance/<rec>/<run id>/` without crop arrays | `archive_runs.py`, `staged_copy_manifest.json` per destination | completed 2026-11-12: 6 runs, 431 files, 25.0 GB in 25 min (about 17 MB/s), every file SHA-256 verified after copy; 15 crop files left out per run; the staged copies with crop arrays remain on the local disk |
