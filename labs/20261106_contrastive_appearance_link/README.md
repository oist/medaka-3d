---
hypothesis: On P-1-1, a per-recording appearance embedding trained without identity labels, by contrasting images within geometry fragments against images of temporally coexisting fragments, decides correspondences across geometry warnings better than the current chain's slot continuation.
stage: identifier
expires: 2026-11-27
decision_rule: >
  At lambda 0.767, with no abstention, carry the contrastive embedding (A3)
  forward to linker integration only if both hold: (a) on reference-known
  boundaries where the chain's slot continuation (A0) is wrong, A3 selects the
  reference mapping in at least half; and (b) constrained whole-recording
  labelling with A3 reduces fixed-anchor wrong-ID fish-seconds inside the
  lower-risk intervals by at least 50 percent relative to A0, while A3 makes no
  more wrong boundary mappings than A0 over all reference-known boundaries.
  Reject A3 as built if (a) is below 25 percent or (b) gives no reduction.
  Otherwise report the result as inconclusive. The colour baseline (A1) is
  scored identically and reported; it does not change the verdict on A3.
---

# Label-free contrastive appearance for linking geometry fragments (P-1-1)

Owner: P-1-1 research team; execution by Claude Code, requested by the
supervisor on 2026-11-06. Registered before any crop, feature or model exists.
This is step 1 of the tracking-strengthening plan: does an appearance signal
exist that can link fragments across encounters? It does not modify the
extraction chain, the reference, or any production output.

## Why this might work

On P-1-1 at lambda 0.767, 788.132 of the 848.417 wrong-ID fish-seconds lie in
local tracks that pass the reference support/purity policy
([ROW_MAPPING_REPAIR.md](../../analyses/20261031_p11_identity_evaluation/ROW_MAPPING_REPAIR.md)).
The fragments are mostly right; their names across joins are wrong. Female,
M1 and M2 agreement is 98.78 / 76.35 / 77.34 percent, so male-male joins carry
the error. Geometry warnings contain all 21 sustained reference-change groups,
but the tested geometry/motion bridges did not reduce wrong-ID time
([GEOMETRY_MOTION_RESULT.md](../../analyses/20261031_p11_identity_evaluation/GEOMETRY_MOTION_RESULT.md)).
Evidence that survives an encounter is needed; appearance is the candidate.

The experimenter can distinguish the fish by colour and slight body-length
differences (`group_list.csv`: P female yellow, M1 red, M2 white). idtracker.ai
v6 learns identification images from same-fragment positives and coexisting-
fragment negatives without labels and matches disconnected blocks
([eLife 107602](https://elifesciences.org/articles/107602v1)). This lab tests
that principle on our fragments; it is a reimplementation of the principle,
not idtracker.ai itself.

## What would make it fail

- The red/white difference is weak in video. A 2026-11-06 contact sheet of eight
  separated frames showed no obvious male colour difference by eye.
- Illumination varies with tank position (the supervisor's observation), so
  appearance may encode location rather than individual.
- Side-view reflections and overlapping fish contaminate crops
  ([PRIOR_FINDINGS 4.4](../../docs/PRIOR_FINDINGS.md)).
- Crop-builder mismatch between training and inference collapses a classifier
  ([PRIOR_FINDINGS 2.3a](../../docs/PRIOR_FINDINGS.md)); here one function builds
  every crop and a test pins its geometry.
- Fragment ends next to an encounter are short; few images may not suffice.

## Fixed inputs

All from the configured data parent, verified against recorded SHA-256 values:

- Tracking: `jingyi/P-1-1/id_row_mapping_20261102_r1`, C34-label lambda 0.767
  (`xyz`, `source`), checked against that run's `artifact_manifest.json`.
- Detections: the same run's `c01_s16_predictions/bottomup/*.npz`
  (`frame_idx`, `points`), checked against `inputs.json`.
- Fragments: `geometry_motion_risk_20261104_r3/lam0.767/` `warning_mask.npz`
  and the start/stop frames of `lower_risk_intervals.csv` (label-free).
- Video: the three raw source videos, checked against the hashes recorded by
  `video_review_v1`. Native frame index is the shared index across cameras, as
  in all prior P-1-1 work; it is not newly certified.
- Evaluation only: `scoring_rerun_v1/scoring_arrays_lam0.767.npz` and the
  reference columns of `lower_risk_intervals.csv` (Jingyi review, SHA-256
  `4f188db9...`). Reference labels never enter crop selection, training,
  descriptors, decisions or clustering, except that the existing fixed
  298--300 s anchor names the first interval's slots, as in every prior score.

## Fixed method

**Fragments.** A fragment is (lower-risk interval, output slot). Inside a
lower-risk interval positions are present and no identity-core warning fires.
The three fragments of one interval coexist and are treated as different fish.

**Crops.** Every 8th native frame inside lower-risk intervals. For each slot and
camera whose detection row contributed (`source >= 0`), take that row's 2D
mouth/pectoral/tail. Centre at the mouth--tail midpoint (pectoral if either is
missing); rotate so the tail-to-mouth axis points right (fallbacks: pectoral to
mouth, tail to pectoral; no crop with fewer than two nodes). Native 160 px
square, resampled to 64 px, colour kept, fixed scale (no body-length
normalisation, so size is not erased). A crop is flagged `overlap` when any
other detection row in that camera and frame has a keypoint within 80 native px
of the centre; overlap crops are excluded from training and descriptors and
counted.

**A0, chain.** The C34 slot continues through the warning (identity mapping).

**A1, colour.** The six statistics of `labs/20260909_three_body` (median r and
g chromaticity, mean saturation-weighted hue cosine and sine, median saturation
and value) inside a body-axis ellipse (semi-axes 0.35 and 0.07 of the crop's
mouth--tail length, at least 2 px), z-scored per camera over all crops.

**A3, contrastive.** One small CNN per camera (four blocks of one 3x3 conv,
batch norm, ReLU and 2x2 max-pool, 32/64/128/128 channels; global average
pool; 128-unit hidden layer; 16-d output), trained from random initialisation on 56 px random crops of the
64 px images. Each step: 128 positive pairs (two crops of one fragment) and 128
negative pairs (crops of two fragments of the same interval), sampled with
intervals weighted by crop count. Loss: squared distance for positives,
squared hinge at distance 1 for negatives. Adam, learning rate 1e-3, weight
decay 1e-4, 3000 steps, seed 20261106, CPU. No early stopping, no label use,
no tuning after any evaluation. Inference uses the central 56 px.

**Descriptors and decisions.** For a boundary between consecutive lower-risk
intervals, describe each pre fragment by its last 3.0 native seconds and each
post fragment by its first 3.0 s: per camera, the mean feature over non-overlap
crops, used when both sides have at least 5 crops. Pair distance is the mean
over shared cameras of Euclidean distance. If any of the nine pair distances is
undefined, the boundary is undecidable. Otherwise choose the permutation of
minimum summed distance; margin is second-best minus best. Whole-fragment
descriptors are a declared secondary variant.

**Whole-recording labelling.** (i) Sequential: name interval 0 by the anchor
and compose boundary decisions; an undecidable boundary carries A0's mapping.
(ii) Constrained global: whole-fragment descriptors; three per-camera centroids
initialised from interval 0; each interval's three fragments are assigned to
the three identities by minimum-cost permutation; centroids are crop-weighted
means; iterate to convergence (at most 50 rounds). An interval it cannot
assign (a fragment without any descriptor) carries A0's naming, like the
sequential fallback, so both labellings are scored on the same time.

## Evaluation (reference used here only)

- Boundary reference mapping: from the per-slot majority labels of both
  intervals, known only if all six fragments have a label with purity 1 and
  coverage at least 0.5 and each side's labels are distinct.
- Boundary outcomes for A0/A1/A3: correct / wrong / undecidable, overall and on
  the A0-wrong subset; male-only edges reported separately. Margin versus
  correctness is reported as a curve; no threshold is selected here.
- Wrong-ID fish-seconds inside lower-risk intervals by
  `medaka3d.eval.flip_risk.paired_identity_time`, with corrected and newly
  wrong time. Warning interiors are not renamed and are reported unchanged.
- Secondary, reported only: whole-fragment descriptors; A3 trained only on
  intervals ending before 1100 s and evaluated on boundaries after 1200 s;
  same-fish versus different-fish descriptor distance by 3D separation of the
  fragments (illumination check); 3D-position correlation of A1 value.

One recording, one trio, one lambda. A positive result shows a usable signal
on P-1-1, not a transferable accuracy; transfer to K_1 and P_1_2--3 requires
their references and is step 3 of the plan.

## Notes

- 2026-11-06: registered before extraction. Visual contact sheet (scratch only)
  of eight separated frames: female distinguishable by body shape in side views;
  males not obviously different in colour by eye; side-view reflections enter
  crops.
- 2026-11-06, before extraction: the encoder uses one convolution per block.
  Two per block measured 1.18 s per CPU training step (about 2.5 h per model);
  one per block measured 0.52 s. This is a compute-budget choice made before
  any crop existed. The environment's PyTorch is a CPU build; the GPU is unused.

## Registration 2 (2026-11-06): naming stage and pose-masked crops

Registered after [RESULT.md](RESULT.md) (registration 1, read out carry-forward)
and before any masked crop, masked model or naming-stage score exists. The
supervisor asked for both follow-ups at once. Inputs are registration 1's
fixed inputs plus its run `appearance_link_20261106_r1` (crops and A3
embeddings), checked by SHA-256. The run is `appearance_naming_20261106_r1`.

**Part 1, naming stage (N).** Turns fragment descriptors into per-frame names
`name[t, slot]` in {identity 0--2, unknown} over all post-preparation time.

1. Constrained global labelling exactly as registration 1, on intervals whose
   three fragments are all described.
2. Final centroids are the crop-weighted means of the converged labels. An
   interval with exactly two described fragments assigns them by minimum-cost
   one-to-one mapping to the centroids. Its third fragment takes the remaining
   identity by elimination. With one described fragment, it takes the
   nearest centroid and the other two stay unknown. With none, all three stay
   unknown. A0 naming is never used as a fallback.
3. Every lower-risk frame takes its interval's slot names.
4. Each warning interior: slot `s` is named `k` only if both flanking
   lower-risk intervals name slot `s` as `k`; otherwise it is unknown. The
   conservative variant, with every warning interior unknown, is reported as
   secondary.
5. Output per slot and frame: name, and `xyz` regrouped by identity (NaN where
   unknown).

Scoring is over all reference-assessed post-preparation time (lower-risk plus
warning). Named time is correct or wrong against the fixed 298--300 s anchor;
unknown time is counted separately, never as correct. Interval 0 is named by
the anchor, as before.

*Part 1 rule.* Carry N forward to transfer only if, with A3 embeddings, both
hold. First, wrong-ID fish-seconds are at most 1% of named fish-seconds.
Second, named fish-seconds are at least 90% of reference-assessed
fish-seconds. Reject N if wrong exceeds 1% of named. Otherwise report it as
inconclusive. Registration 1 already implies most of the lower-risk outcome:
assigned intervals had no wrong time. The new evidence is the partial
assignment, the warning carry and the coverage.

**Part 2, pose-masked crops (B).** The r1 crops are masked in crop
coordinates; video is not decoded again. Every other detection row of the same
camera and frame is drawn as a band along its mouth, pectoral and tail points.
The band has half-width 0.14 of that row's mouth--tail length, extends 0.15
length beyond the tail point and 0.05 beyond the mouth. Its ends are round,
reaching a further half-width, as OpenCV draws thick lines. A row without both
ends uses a length of 0.7 of the native crop (112 px). Band pixels are filled
with the camera's per-channel median crop value. The same band around the
crop's own row defines its body. A crop is usable when at most 20% of its own
body band lies under other rows' bands. This replaces the 80 px overlap
exclusion. Every crop, usable or not, gets the same masking, so one builder
serves training and inference. B is trained per camera exactly as A3 (same
architecture, loss, sampler, 3000 steps, seed), on usable masked crops.
Descriptors, decisions and N are unchanged apart from the usable set.

*Part 2 rule.* Adopt B over A3 only if all three hold, both run through N.
First, N with B names more reference-assessed fish-seconds than N with A3.
Second, its wrong-ID time is at most 1% of its named time. Third, B makes no
more wrong boundary decisions (end window, reference-known boundaries) than
A3. Reject B if it names less time, exceeds 1% wrong, or adds wrong boundary
decisions. Otherwise report it as inconclusive. Also reported: undecidable
boundaries, unassigned intervals, and the four chain-wrong boundaries of the
527--559 s cluster.

**Secondary, reported only.** Label-free physical naming. Is the male
identity with the higher median red chromaticity (A1 `median_r_fraction`,
non-overlap crops) the one the anchor calls M1? `group_list.csv` records M1
as red. Median top-camera mouth--tail length per identity is also reported.
One image sheet per camera shows each identity's non-overlap crops from 24
random intervals, for inspection by eye. (Added during training, after a
scratch dry run of the A3 naming path showed the redder male carries the
anchor's M2 label. It is a visual output only and changes no rule.)

Note, 2026-11-06: the supervisor stated that it is unknown whether the colour
descriptors behind the review's labels match the fishes' appearance in video.
The objective is identity consistency within each recording. Matching
identities across trials of a group is desirable but not required. The
physical-naming readout is therefore an observation, not a check of the
reference.

## Registration 3 (2026-11-06): bridging and naming inside gaps

Registered after Registration 2's result and before any warning-frame crop,
calibration or score exists. Inputs are Registration 2's run
`appearance_naming_20261106_r1`: B models, B embeddings and the N_B interval
names, recomputed and checked. The run is `appearance_gap_20261106_r1`.

**Anchors and gaps.** An anchor is a lower-risk interval with all three slots
named by N_B, the global or elimination names. A gap is all frames strictly
between consecutive anchors: warning interiors plus unnamed intervals. Gap
time is fixed by N_B and is the same for every variant below.

**S, bridging.** For each slot, every gap frame takes name `k` if the
preceding and following anchors both name that slot `k`; otherwise it is
unknown. Registration 2's warning carry is the special case without unnamed
intervals.

**G, appearance inside gaps.** Crops are taken at every frame with
`t % 8 == 0` inside warning interiors. They use the same crop builder and
the same pose masks as B, and are embedded with B's models. Unnamed lower-risk
intervals reuse their B crops. Only usable crops (own-body occlusion <= 20%)
give evidence.

- *Per-crop likelihood.* For camera `v`, identity `k` and crop embedding `e`:
  `p_v(k | e) = softmax_k(-|e - c_vk|^2 / tau_v)`. The centroids `c_vk` are
  crop-weighted means of N_B-named whole-fragment descriptors. `tau_v` is the
  maximum-likelihood value on a grid of 501 points in log10 [-4, 1]. It is fit
  to N_B's names on lower-risk usable crops that Registration 1 flagged as
  overlap, the nearest available analogue of contact crops. No reference
  label is used.
- *Gap model.* States are the six slot-to-identity permutations. The first
  and last nodes are fixed to the flanking anchors' names; the sampled frames
  are the nodes between. The log-emission of a permutation at a sampled frame
  is `alpha` times the sum, over slots and cameras with a usable crop, of
  `log p_v`, with `alpha = 1/4`. That factor allows for correlation between
  frames 48 ms apart. Between nodes `dt` apart, a node stays in its
  permutation with probability `exp(-r dt)` and otherwise moves to each of the
  other five equally. `r` is label-free: the number of gaps whose two anchors
  differ in any slot, divided by total gap duration.
- *Decision.* Forward-backward posteriors per sampled node. A slot is named
  `k` at a node when its marginal posterior is at least 0.99, otherwise it is
  unknown.
- *Frames between nodes.* Anchor end and start frames are nodes with certain
  names. A frame takes slot name `k` only if the nodes on both sides name it
  `k`.

*Rules.* All are scored on gap time only, against the reference, with the
fixed anchor.

- *S.* Adopt S over Registration 2's N_B warning carry if S names more gap
  fish-seconds, with wrong-ID time at most 1% of its named gap time. Reject S
  if its wrong time exceeds 1%. Otherwise report it as inconclusive.
- *G.* Adopt G over S if G names more gap fish-seconds than S, with wrong-ID
  time at most 1% of its named gap time. Reject G if its wrong time exceeds 1%
  or it names less. Otherwise report it as inconclusive.

Also reported:

- whole-recording named and wrong time;
- the two long encounters (613.681 s, 32.432 s; 510.108 s, 17.785 s);
- the nine chain-wrong boundaries: whether G names their gap and where it
  places the switch;
- `tau_v` and `r`;
- a sensitivity check, reported only: `alpha` in {1, 1/16}, and posterior
  thresholds 0.9 and 0.999.

The mask geometry was chosen from overlays of candidate bands on 60 sampled
overlap crops (scratch only), judging coverage of the other fish's body and
fins. No identity outcome was computed for any candidate.

## Registration 4 (2026-11-07): clean crops, one side model, silhouette stopping; judgement on K_1_2

Registered after Registration 3's result and its crop-identity audit, and
before any flipped crop, pooled model, silhouette value, K_1_2 crop or score
exists. The audit found the training labels correct and the Registration 3
failure in crops unlike the training data: partly occluded crops and crops
where two fish are close in one view. idtracker.ai v6 excludes such crossing
images from training and identification and interpolates identity across
them; Registration 3 did the opposite. The supervisor also asked whether the
two side cameras should share one model and for a view of how identity
separation develops during training.

**Runs.** `appearance_clean_20261107_r1` on P-1-1 (development: its scores
may be read while working) and `appearance_clean_20261107_k12` on K_1_2
(judgement: its score is read once, after the rule below is fixed). Both run
the same code with one configuration file naming both recordings.

**Fixed inputs, P-1-1.** Registration 1's crops, Registration 2's masks and
Registration 3's warning crops and masks, each checked against the hash its
run recorded; the detection caches, chain, warning mask, warning and
lower-risk intervals and scoring arrays as before. No video is decoded again.

**Fixed inputs, K_1_2.** The production chain output
`work/c34_microflip/20251225_K_1_2_FCMC__chain-c28-c33-c30v4-c34/`
(`C34-label` and `C34-gated` at lambda 0.767, written 2026-09-03), the
detection caches `work/c01_s16_predictions/20251225_K_1_2_FCMC__*.npz` that
the chain consumed, the three raw videos and the top-camera `frametimes.npy`.
Their SHA-256 values are recorded by this run; there is no earlier manifest.
The chain predates the 2026-11-02 adapter repair, whose effect on P-1-1 was
two labels at frame 20 and no downstream difference
([ROW_MAPPING_REPAIR.md](../../analyses/20261031_p11_identity_evaluation/ROW_MAPPING_REPAIR.md));
K_1_2 is not rerun. Evaluation only: `assets/groundtruth/fish_sex_gt/
20251225_K_1_2_FCMC/female_sex_gt.csv` (300 confirmed female
identifications, 256 after 300 s). No male reference exists for K_1_2; the
contact-episode notes lie before 300 s and the swap notes name another arm's
slots, so neither is used.

**K_1_2 fragments, label-free.** The identity-core warning rules of
`geometry_motion_risk_20261104_r3` (missing position, step over 10 mm,
non-identity nearest assignment, proximity under 10 mm, padding 0.05 s) are
applied to the chain output with `labs/20261104_geometry_flip_risk/linker.py`.
Lower-risk intervals are the contiguous runs of post-300 s frames with all
three positions valid and no warning, as in P-1-1. Time and weights come
from `frametimes.npy` through `interval_weights(times, 300)`. Crops are every
8th native frame inside lower-risk intervals and inside warnings, built by
the Registration 1 extractor and the Registration 2 masks, unchanged.

**X, clean crops.** A crop is clean when all three hold: its own-body band is
not occluded at all (Registration 2's fraction is 0); no other detection row
of that camera and frame has a keypoint within 80 native px of the crop
centre (Registration 1's overlap flag is false); and that camera-frame holds
at least three detection rows, so each fish has its own detection. Only
clean crops enter training, validation, descriptors, temperature fitting and
gap evidence. Unclean crops are embedded and counted but never used. On
P-1-1's lower-risk crops this keeps 73.5 / 69.7 / 73.7 percent (top, side1,
side2; counted label-free before this registration).

**D, dorsal-up side crops.** The builder points the heading right by
rotation, so a side-view fish swimming left is shown belly-up (34.8 and 58.7
percent of side1 and side2 crops). After masking, a side crop whose heading
has |h| > 90 degrees is flipped vertically; the top view is unchanged. The
flip depends on the heading only, so the one builder still serves training
and inference, and the P-1-1 crops are flipped in place without decoding.

**W, one side model.** Side1 and side2 clean crops are pooled. A fragment is
(interval, slot) regardless of camera, so a positive pair may come from the
two cameras at once and a negative pair from any cameras; intervals are
weighted by pooled clean count; channel statistics are pooled. The top
camera keeps its own model. Architecture, loss, optimiser, batch (128 + 128
pairs), learning rate, weight decay and seed are Registration 1's.

**T, silhouette stopping.** A validation set of 3000 clean training crops
per model is drawn once (seed 20261106). Every 100 steps, from step 0, the
validation crops are embedded (central 56 px), clustered by k-means into
three clusters (ten random restarts, at most 100 Lloyd iterations, seed), and
the silhouette score of the best-inertia partition is recorded. Training
stops after 30 consecutive evaluations without a new best, or after two
consecutive evaluations at or above 0.91, or at step 6000; the parameters at
the best evaluation are the model. These are idtracker.ai v6's thresholds.
Each evaluation also stores the validation embeddings, the median positive
and negative pair distances of the batch, and, for the side model, the
median distance between side1 and side2 crops of the same fragment at the
same frame.

**V, separation during training (reported).** For each model, a 2-D principal
component projection of the stored validation embeddings at evaluations
0, 1, 3, 10, the best and the last, coloured by k-means cluster; for the
side model also by camera. On P-1-1 the same panels are coloured by the
reference identity of the crop's slot, drawn at evaluation time only. The
silhouette, pair-distance and cross-camera curves are plotted against step.

**N, S, G as before, on clean crops.** Whole-fragment descriptors use clean
crops per camera (side1 and side2 stay separate keys, embedded by the shared
model; at least 5 clean crops). Naming, partial assignment, warning carry
(`N|carry`) and bridging (`S`) are Registrations 2 and 3. `G` is Registration
3's gap model with two changes: evidence comes from clean crops only, and
each camera's temperature is fit by maximum likelihood on the clean
lower-risk crops of named intervals, because the former overlap set is
excluded by X. Alpha 1/4, posterior 0.99 and the label-free switch rate are
unchanged.

**Reference on K_1_2.** A confirmed female identification is matched, in each
side view where it is linked, to the detection row whose mouth lies within
5 px of the annotated mouth (c09's rule), then to the slot whose source row
that is; frames where the views disagree or no slot is unique are dropped.
Anchors after 300 s are scored. The female identity index is the name the
female's slot carries at the first scored anchor that is named; at every
later anchor the female's slot is correct if it carries that name, wrong if
it carries another, unknown if none. The chain (slot continuation) is scored
the same way. Anchors are points, not fish-seconds.

*P-1-1 readouts* (development; reported, not the judgement): `N|carry`
scored as Registration 2 (wrong at most 1% of named, named at least 90% of
assessed) and `G` against `S` on gap time as Registration 3; the two
encounters; boundary outcomes; and the comparison with Registrations 2 and 3
on identical regions.

*K_1_2 rule* (judgement). Carry the clean pipeline (X, D, W, T with `N|carry`)
forward to the remaining recordings only if, on K_1_2, wrong anchors are at
most 1% of named anchors and named anchors are at least 90% of scored
anchors. Reject if wrong anchors exceed 1% of named. Otherwise report it as
inconclusive. Male identity on K_1_2 is unverifiable: the number of
boundaries where the appearance names differ from slot continuation on male
slots, decided and undecidable boundaries, and interval assignment counts
are reported as label-free diagnostics, not accuracy. `G` on K_1_2 is
reported on anchors inside gaps, secondary.

**Limits declared now.** One trio per recording. K_1_2's reference tests the
female only; a male-male swap on K_1_2 cannot be detected. The k-means
stopping assumes three animals. P-1-1's reference has now informed four
registrations; its readouts are development evidence.

## Registration 5 (2026-11-07): input resolution and encoder architecture, on the GPU

Registered after Registration 4's result and before any 128 px crop, ResNet
model, GPU-trained encoder or score exists. The supervisor asked whether
56 px input suffices and whether the encoder should be idtracker.ai's
ResNet18 with an 8-d output, and allowed the machine's RTX 5070 Ti. Two
single-change variants of the Registration 4 pipeline (X, D, W, T, N, S and G
unchanged) are run on both recordings; they are not combined here.

**R, resolution.** The native 160 px crop is resampled to 128 px instead of
64 (0.8 of native instead of 0.4); training uses 112 px random crops and
inference the central 112 px, the same 7/8 fraction as 56 of 64. The fish's
mouth--tail length then spans about 75--90 px instead of 38--46. Pose masks,
the clean criterion and the dorsal-up flip are computed at the new scale with
the same geometry in body-length units. The encoder is Registration 4's
four-block CNN, whose global average pool absorbs the input size. Crops are
decoded again from video for both recordings, with the Registration 1
extractor at the new size.

**A, architecture.** torchvision's ResNet18 at random initialisation, with the
final layer replaced by a bias-free linear map to 8 dimensions (idtracker.ai
v6's embedding), trained on Registration 4's own prepared 64 px crops (input
56 px). Loss, optimiser, batch, seed and stopping are Registration 4's, so
the architecture is the only change; idtracker.ai's D_neg = 10 is not copied.

**Device.** Both variants train on the GPU with the CUDA 13.0 build of
torch / torchvision (`pyproject.toml`, `uv.lock`, both recorded in the run
snapshots). Correction added 2026-11-07 after the runs: this paragraph first
said 2.14.0 / 0.29.0; the cu130 index carried 2.14.1 / 0.29.1 and the
snapshots record those. Seeds are fixed, but GPU arithmetic differs
from the CPU runs, so Registration 4's numbers are the comparison, not a
reproduction target. The device is recorded in provenance.

**Runs.** `appearance_res128_20261107_r1` / `_k12` from
`config_appearance_res128_v1.toml` and `appearance_resnet_20261107_r1` /
`_k12` from `config_appearance_resnet_v1.toml`, both through
`run_appearance_clean.py --config`. The P-1-1 runs are development, the
K_1_2 runs the judgement, as in Registration 4.

**Primary readout, single-crop accuracy on P-1-1 warning crops.** For each
camera: clean warning crops whose slot has a reference identity and whose
detection is clearly attributable (the Registration 3 audit's criterion: the
nearest reference fish projects within 25 px of the row's pectoral and the
second nearest at least 25 px farther). The prediction is the nearest
N-centroid of the crop's embedding; accuracy is against the slot's reference
identity. The same number is computed for Registration 4's encoders on the
same crops, so each camera has a baseline and a variant value on a common
crop set, (frame, slot, camera) triples clean in both runs. Pooled accuracy
is the crop-weighted mean over cameras on the common set.

**Safety condition.** The K_1_2 anchor rule of Registration 4, unchanged: at
most 1% wrong of named anchors and at least 90% named.

*Rule, per variant.* Adopt the variant over Registration 4 only if the K_1_2
rule is met and the pooled single-crop accuracy on the common P-1-1 warning
set is at least 2 percentage points above Registration 4's, with no camera
more than 1 point below. Reject if the K_1_2 rule fails or any camera is
more than 2 points below. Otherwise report it as inconclusive. If both
variants are adopted, their combination is a further registration.

Also reported: silhouette at the stop and the stopping step; decided and
undecidable boundaries; N, S and G on P-1-1 as in Registration 4; the
cross-camera distance; training time on the GPU; the accuracy on each run's
full clean reference-clear warning set, not only the common set.

**Limits declared now.** The primary readout uses P-1-1's reference, which has
informed every registration; the variants are judged for safety on K_1_2's
female only. The warning crops are a few thousand per camera, so differences
under about 1 point are within sampling noise; the 2-point margin is the
reason for that threshold.
