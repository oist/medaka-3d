# Does a trio's appearance carry over to a later experiment period? K_1 -> K_2, P_1 -> P_2

Status: **registered 2026-11-09 before any run of this configuration existed;
executed the same day (run `cross_trial_20261109_r2`, findings below).** This
is a label-free consistency test of frozen encoders, not a scoring study: no
recording of K_2 or P_2 has a reviewed identity, so nothing here is an
accuracy.

## Question

The per-trio encoders of the six-recording study were trained on all three
days of K_1 (and of P_1) and recover the same three appearance clusters on
each of those days. The fifteen-recording run trained a new encoder per
recording for K_2 and P_2, so it cannot say whether the appearance learned in
K_1/P_1 is still usable on the later recordings of the same animals. Here the
K_1 encoders, their channel normalisation and the three K_1 reference
centroids are frozen and applied to the three K_2 recordings (P_1 to P_2
likewise), with no retraining and no adjustment of the reference. Three
things are measured:

1. **Naming against the frozen reference.** Each target's lower-risk
   intervals are named against the K_1 (P_1) reference centroids with the
   lab's own decision rules, per camera and with all cameras, and compared
   with the target's own per-recording encoder names under the one
   permutation of the target's three codes that agrees most. Agreement means
   the frozen K_1 identities induce the same within-recording partition as a
   fresh encoder; it does not verify either.
2. **Within-period against cross-period variation.** The target's label-free
   clusters (its own codes, and k-means on the frozen embeddings) are matched
   to the reference centroids by best one-to-one assignment with the margin
   to the second-best assignment, exactly the six-recording study's cross-day
   diagnostic. The displacement of each matched cluster from its reference
   centroid is set against the smallest separation between reference
   identities and against the day-to-day displacement of the same identity
   inside K_1 (P_1).
3. **Stability per day, camera and time window.** The matching and the
   agreement are repeated per camera and per 300 s bin, since a recording
   whose averages match may still lose the correspondence in part of it.

Owner: Machiroi (supervisor). PR: [#4](https://github.com/oist/medaka-3d/pull/4).

## Inputs, fixed and read-only

All paths in `config.toml`, relative to the data parent (`data:`) or to the
local staging directory of `configs/paths.local.toml` (`staging:`).

| Input | Location | Notes |
| --- | --- | --- |
| Frozen encoders and normalisation | `work/appearance/20251225_K_1_1_FCMC/appearance_group_20261108_k11/models/{top,side}.pt`; `work/appearance/20260113_P_1_2_FCMC/appearance_group_20261108_p12/models/{top,side}.pt` | Phase 3 of the six-recording study; the three copies written per trio are byte-identical (checked 2026-11-09) |
| Reference embeddings and names | the six `appearance_group_20261108_*` runs (`embeddings/`, `evaluation/identity_tracks_N_carry.npz`) and the runs holding their prepared crops (`clean/<view>.npz`, metadata only) | all on the share |
| Review alignment of the reference codes | `work/six_recording_identity/six_identity_20261108_r2/scores/phase3_per_group/scores.json` | gives F/M1/M2 to each day's codes; checked label-free before use |
| Target crops | `appearance_unrev_20261109_{k21,k22,k23,p21,p22,p23}` (`clean/<view>.npz`), read from local staging, SHA-256 checked against each run's prepare provenance and the shared copy's `staged_copy_manifest.json` | same crop pipeline as the sources (video, frame step 8, 64 px) |
| Target own-encoder names | `evaluation/interval_names.csv` and `identity_tracks_N_carry.npz` of the same runs | consistency comparison only; P_2_3 has partial interval output but no usable comparison or identity tracks (its evaluate failed on the seed interval) |

## Method, fixed

`cross_trial_embed.py` with `config.toml`, one invocation, output under
`work/analyses/cross_trial_appearance/<run_id>/` with a provenance record:

- The reference centroid of an identity in a view is the crop-weighted mean,
  over the three source days, of the clean lower-risk embeddings that each
  day's N_carry naming gives that identity. Before pooling, the best
  one-to-one assignment of per-day centroids between every pair of days must
  be the identity (it reproduces the study's `cross_day_centroids.json`);
  the run stops otherwise. Re-embedding a sample of the model run's own crops
  measures agreement with its stored embeddings (sanity check, recorded;
  the implementation reports the difference without an acceptance threshold).
- Target crops are embedded with the frozen weights and normalisation
  (`lab.embed`, encoder in eval mode). Interval descriptors are the lab's
  (`describe`, `MIN_CROPS = 5`, whole-interval rows, overlapping crops
  dropped). Naming uses `decide` (three described slots) and
  `assign_partial` (one or two) against the frozen centroids; no centroid is
  updated.
- The own-code permutation is the one of six maximising agreeing
  duration over intervals fully named by both procedures (global, or
  partial_two by elimination), weighted by three fish. All three names must
  agree for an interval to count: reported agreeing/disagreeing fish-seconds
  classify complete intervals, not individual fish. The second-best
  permutation's score is recorded too.
- Cluster matching: cost = sum of the three matched centroid distances,
  margin = second-best minus best. Per-bin matching requires at least
  `MIN_CROPS` crops per cluster in the bin; where a target has no own
  encoder, the k-means clusters of the view stand in.
- Silhouettes on a 3,000-crop sample per view with the lab's k-means and
  silhouette, under k-means labels, own codes and frozen names.

Nothing is retrained, no threshold is applied, and no target output is
exported as an identity candidate.

## Limits declared now

- No accuracy. Agreement between the frozen naming and the target's own
  encoder is agreement between two label-free procedures that share the
  crops and the chain; both can be wrong the same way.
- The reference identities carry the K_1/P_1 reviewers' names. In K_2/P_2
  they are hypotheses carried by appearance alone; the physical descriptors
  of `group_list.csv` are not consulted.
- Detections of K_2/P_2 come from the capped Windows inference; K_1's from
  the uncapped cluster inference (P_1 from the reviewed 3-node model).
- The encoders were stopped by a silhouette rule on their own trio; a
  cluster structure that is clean on the targets may still be a different
  partition from the animals (a video check of a few crops per name is the
  cheap confirmation).

## Findings

Run `cross_trial_20261109_r2`, 2026-11-09 10:16--10:22, GPU; log
`run_20261109_r2.log`. All 18 target crop files passed both SHA-256 checks;
re-embedding 2,000 crops of each model run's own recording reproduced the
stored embeddings to 8e-7 or better. The label-free cross-day check of the
reference reproduced the six-recording study's figures (K_1 cost
0.028--0.119, margin 1.94--2.08; P_1 cost 0.032--0.076, margin 1.78--2.24;
every best assignment the identity). All numbers below are from
`report/summary.md` and the per-recording CSVs of the run.

### 1. Naming against the frozen reference agrees with a fresh encoder

The frozen reference names exactly the intervals the per-recording encoders
name (the described-slot structure is shared), and gives them the same
names up to the permutation of codes, except on four two-fragment intervals
(`partial_two`, third slot by elimination) of 0.5--1.0 s each:

| Target | Lower-risk intervals | Frozen global / partial_two / partial_one / unknown | Named by both, fish-s | Agree | Disagreeing intervals (fish-s), all `partial_two` | All-camera margin median (min) |
| --- | ---: | --- | ---: | ---: | --- | --- |
| K_2_1 | 640 | 246 / 35 / 71 / 288 | 4,265.9 | 99.87% | 2 (5.5) | 1.86 (0.51) |
| K_2_2 | 351 | 187 / 27 / 47 / 90 | 4,904.6 | 99.94% | 1 (3.1) | 1.82 (0.53) |
| K_2_3 | 181 | 148 / 3 / 9 / 21 | 5,243.7 | 99.97% | 1 (1.4) | 1.53 (0.94) |
| P_2_1 | 211 | 137 / 8 / 28 / 38 | 5,061.3 | 100.00% | 0 | 1.95 (1.36) |
| P_2_2 | 174 | 126 / 6 / 21 / 21 | 5,075.5 | 100.00% | 0 | 1.96 (1.45) |
| P_2_3 | 357 | 190 / 12 / 54 / 101 | no usable own naming | -- | -- | 1.88 (1.04) |

No interval named globally by both procedures disagrees; the second-best
permutation of codes agrees on 2.9, 3.1 and 1.4 fish-s in K_2_1--K_2_3 and on
0.0 in P_2_1 and P_2_2. Naming from each
camera alone agrees with the all-camera naming on every global interval of
every recording (top 116--211, side 114--197 intervals compared per
recording, 0 disagreements). The own-code permutations are K_2_1 and K_2_2
code0/1/2 = M1/F/M2, K_2_3 = M2/F/M1, P_2_1 = M2/M1/F, P_2_2 = M1/F/M2, and
the cluster matching below returns the same assignment in every view.

P_2_3, whose own evaluate failed because its seed interval had descriptors
for one slot only, is fully named by the frozen P_1 reference on 190 global
and 12 partial_two intervals (4,771 fish-s, margin median 1.88). Seeding from
another recording's reference is a different algorithm from Registration 4
(design rule 5); this is a diagnostic, not an identity candidate, and no
export was made.

### 2. P_1 -> P_2: small shifts; K_1 -> K_2: distinction kept, one male shifted

Cluster matching of the targets' own-code clusters (k-means for P_2_3) to
the pooled reference centroids, and the displacement of each matched cluster
from its reference centroid, compared with the within-period figures of the
source trio:

| Target | View | Best cost | Margin | Within-target separation | Displacement F / M1 / M2 | Per-crop nearest-centroid agreement F / M1 / M2 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| K_1 within period (3 days) | top / side | 0.03--0.06 / 0.05--0.12 | 2.06--2.08 / 1.94--2.04 | 1.04 / 1.02 | max 0.010 / 0.038 / 0.012 (top), 0.023 / 0.093 / 0.017 (side) | 0.997--0.999 / 0.984--0.996 |
| K_2_1 | top | 0.111 | 1.95 | 0.99 | 0.011 / 0.035 / 0.065 | 1.000 / 0.996 / 0.963 |
| K_2_1 | side1, side2 | 0.170, 0.166 | 1.76, 1.77 | 0.89, 0.90 | 0.007 / 0.149 / 0.014; 0.004 / 0.144 / 0.017 | 0.988 / 0.877 / 0.986; 0.993 / 0.888 / 0.979 |
| K_2_2 | top | 0.095 | 1.99 | 1.01 | 0.014 / 0.032 / 0.049 | 1.000 / 0.998 / 0.979 |
| K_2_2 | side1, side2 | 0.180, 0.189 | 1.74, 1.73 | 0.87, 0.87 | 0.013 / 0.159 / 0.008; 0.014 / 0.169 / 0.007 | 0.988 / 0.875 / 0.992; 0.983 / 0.867 / 0.991 |
| K_2_3 | top | 0.294 | 1.76 | 0.95 | 0.017 / 0.237 / 0.039 | 1.000 / 0.900 / 0.984 |
| K_2_3 | side1, side2 | 0.349, 0.345 | 1.44, 1.42 | 0.73, 0.72 | 0.024 / 0.309 / 0.016; 0.006 / 0.327 / 0.012 | 0.983 / 0.732 / 0.987; 0.992 / 0.713 / 0.989 |
| P_1 within period (3 days) | top / side | 0.03--0.08 / 0.04--0.07 | 2.21--2.24 / 1.78--1.82 | 1.14 / 0.92 | max 0.025 / 0.015 / 0.036 (top), 0.033 / 0.030 / 0.023 (side) | 0.999--1.000 / 0.986--0.994 |
| P_2_1 | top; side1, side2 | 0.044; 0.059, 0.050 | 2.23; 1.75, 1.76 | 1.12; 0.89, 0.89 | 0.025 / 0.005 / 0.014; 0.006 / 0.037 / 0.016; 0.005 / 0.030 / 0.015 | 0.996 / 0.999 / 1.000; 0.995 / 0.978 / 0.992; 0.993 / 0.981 / 0.991 |
| P_2_2 | top; side1, side2 | 0.064; 0.020, 0.030 | 2.18; 1.82, 1.82 | 1.09; 0.91, 0.92 | 0.052 / 0.005 / 0.006; 0.009 / 0.005 / 0.006; 0.020 / 0.003 / 0.006 | 0.987 / 0.998 / 1.000; 0.986 / 0.994 / 0.985; 0.971 / 0.995 / 0.991 |
| P_2_3 (k-means) | top; side1, side2 | 0.075; 0.063, 0.059 | 2.15; 1.75, 1.76 | 1.08; 0.88, 0.89 | 0.066 / 0.016 / 0.010; 0.029 / 0.061 / 0.009; 0.039 / 0.055 / 0.009 | 0.968 / 0.996 / 0.999; 0.979 / 0.958 / 0.992; 0.970 / 0.960 / 0.992 |

Displacements and per-crop agreement of the targets are over crops in
intervals the frozen rule names globally; separation is the smallest distance
between two centroids. Per-identity figures by camera and 300 s bin are in
`report/<recording>_crop_bins.csv`.

**P_2.** Every displacement (0.003--0.066) lies inside or just above the
within-period day-to-day scale of P_1 (0.015--0.036), the matching costs
(0.02--0.08) and margins (1.75--2.23) equal the within-period ones, and
per-crop agreement with the frozen centroids is 0.958--1.000 against
0.986--1.000 inside P_1. Thus matching remains well separated on P_2, but
some displacements exceed the source day-to-day maxima and crop agreement
decreases slightly. These measurements do not establish zero drift or
statistical equivalence between periods.

**K_2.** F and M2 behave like P_2 (displacement at most 0.065, agreement
0.963--1.000). The reviewer's M1 does not: its side-view centroid sits
0.14--0.17 from the K_1 reference on K_2_1 and K_2_2 and 0.31--0.33 on K_2_3
(top 0.24), against a within-period maximum of 0.09; M1's per-crop agreement
falls to 0.87--0.89 (K_2_1, K_2_2 sides), 0.90 (K_2_3 top) and 0.71--0.73
(K_2_3 sides). The shift grows inside the first two days (M1 side
displacement per 300 s bin 0.10--0.13 in the first bins of K_2_1 and K_2_2,
0.26 and 0.31 in their last bins) and is present throughout K_2_3
(0.26--0.41). The whole-recording displacement stays below a third of the
pooled source centroid separation. All globally named intervals agree with
the fresh encoder (margins at least 0.51); four partial_two intervals differ.
The k-means clusters of every
view still match the reference with margins 1.69--2.01, and the silhouette
of the frozen embeddings under k-means remains 0.84--0.93 (sides 0.83--0.88)
against 0.957/0.925 on the source trio's validation sample. The silhouette
under the frozen names (0.69--0.80 in K_2 side views) says the same thing
from the other side: crops carrying M1's name have moved towards the
boundary with another cluster. Which animal this is, and whether the change
is in the fish or in the capped detections, cannot be told from embeddings;
a side-by-side look at M1 crops from K_1_3 and K_2_3 is the cheap check.

### 3. The correspondence holds in every camera and every time window

For each recording, view and 300 s bin (six time bins, 18 view-bin
comparisons per clustering method), the bin's
own-code clusters (k-means for P_2_3) match the reference with the same
assignment as the whole recording; no bin reverses it. Bin margins are
1.57--2.08 (top) and 1.26--1.92 (side) in K_2, the lowest in K_2_3's first
three bins, and 1.99--2.26 / 1.69--1.82 in P_2. The only interval-level
disagreements are the four partial_two intervals above (one bin each).

### What this says and does not say

- A per-trio encoder trained on one three-day period can be applied as is to
  the same animals' next three-day period: on all five testable recordings
  globally named intervals agree with a freshly trained encoder, and
  single-camera naming agrees with all-camera naming where both are global.
  Cluster assignments are stable across the measured time windows. This is consistency between two
  label-free procedures, not accuracy; the four partial_two intervals show
  where elimination-based naming is fragile under either reference.
- In group K the appearance of one male (the reviewer's M1) shifted between
  the periods and within K_2, far more than any identity moved inside K_1.
  The distinction survives, but crop-level consistency declines. A
  corresponding loss of identity accuracy has not been measured, and
  nothing here says the shift would stay bounded in a third period.
- In group P the shifts remain small relative to between-identity
  separation, although some exceed the within-period range.
- The frozen P_1 reference fully names 202 of P_2_3's 357 lower-risk
  intervals; 54 are partially named and 101 remain unknown. Using this
  reference for candidate export would require a separately registered
  command, not a flag on the existing one.
- K_2 and P_2 identities remain unverified; the F/M1/M2 names in the
  targets are appearance hypotheses.

## Artifacts

`work/analyses/cross_trial_appearance/cross_trial_20261109_r2/` under the
data parent (114 MB): `cross_trial_appearance.provenance.json` (status ok; 90
inputs with SHA-256, 42 outputs), `embeddings/<recording>/<view>.npy` (frozen-model
embeddings of every prepared clean crop, row-aligned with the run's
`clean/<view>.npz`), `reference/{K_1,P_1}.{npz,json}` (pooled and per-day
centroids, counts, the label-free cross-day check, within-period figures),
`report/<recording>_{intervals,bins,crop_bins}.csv`, `report/summary.json`,
`report/summary.md`. The first complete build, `cross_trial_20261109_r1`, is
kept beside it: same method and bit-identical embeddings (max absolute
difference 0.0), fewer reported columns; r2 is the reported run.

Command, from the repository root with the project virtualenv:

```powershell
.venv\Scripts\python.exe analyses\20261109_cross_trial_appearance\cross_trial_embed.py
```

No extraction output, appearance run, review file or earlier run was
modified.

### Pre-commit verification and code corrections, 2026-11-09

The recorded run remains immutable. Generated execution logs and the exact
pre-commit source/configuration were archived under
`work/analyses/identity_commit_review/identity_commit_review_20261109/`
with repository-relative paths and a verified SHA-256 `manifest.json`.
Logs remain locally available but are excluded from Git.

The r2 provenance uses the same `model_top` / `model_side` input keys for
both groups, so its flat input map retains P_1's entries. K_1's model paths
and hashes remain in `reference/K_1.json` and `report/summary.json`. The
committed runner uses group-specific input keys to retain all four models
on future runs, and refuses a shared-copy manifest with a missing crop
checksum. These changes do not alter embedding or naming calculations;
the existing r1/r2 outputs have not been regenerated.

The agreement metric is conditional on fully named lower-risk intervals
and uses the best code permutation on those same intervals. Crop-level
figures select intervals globally named by the frozen model. Neither
quantity estimates full-window identity accuracy. Source day-to-day
displacement is a maximum pairwise distance; target displacement is measured
from a pooled source centroid, so they are descriptive comparisons with
different reference points.

Pre-commit verification reproduced the five available complete-interval
agreement fractions from the saved interval tables, checked all 42 output
hashes and all four model hashes, and confirmed stable assignments in all
108 selected view-bin comparisons. The archived original runner matches
r2's recorded source hash. The focused cross-trial, S/G export, appearance
and cohort tests passed (83 passed, two skipped). No model or embedding run
was repeated for this commit.

## Attempts and changes

| Date | Change or attempt | Evidence/output run | Outcome and next decision |
| --- | --- | --- | --- |
| 2026-11-09 | Registered; pre-flight re-embedding of 2,000 K_1_1 crops per model reproduced the stored embeddings (max distance 8e-7) | shell check before the run | run started |
| 2026-11-09 10:06 | First invocation stopped at the first target: the shared-copy manifest stores `{size, sha256}` per file and the check compared the dict with the digest | traceback in the first log (removed); partial output (reference only) deleted | check fixed to read `sha256`; the digests themselves matched |
| 2026-11-09 10:07--10:12 | `cross_trial_20261109_r1`, complete | `run_20261109_r1.log` | agreement 99.87--100%, 0 disagreeing global intervals; the residual disagreement was traced to partial_two intervals, and per-identity displacement found M1 of K drifting |
| 2026-11-09 10:16--10:22 | `cross_trial_20261109_r2`: per-identity displacement and agreement per view and bin, k-means per-bin matching for targets without an own encoder, disagreement listed over all fully named intervals, columns renamed | `run_20261109_r2.log`; embeddings bit-identical to r1 | reported run. Next decision: whether to video-check M1 crops of K_1_3 against K_2_3, and whether a cross-recording-reference seeding command is worth registering for recordings like P_2_3 |

## Limitations and next step

See "Limits declared now" and "What this says and does not say". Next:
a video check of M1 crops across K_1_3 and K_2_3 (a few per camera) to see
whether the shift is in the animal's appearance; if the frozen-reference
naming is to be used for anything downstream, register it as its own command
with its own evaluation rather than as an option of Registration 4.
