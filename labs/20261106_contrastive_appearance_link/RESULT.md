# Result: registrations 1--2 met, 3 rejected on P-1-1; registration 4 met on P-1-1 and judged met on K_1_2; registration 5 variant A rejected, variant R inconclusive

Date: 2026-11-06, updated 2026-11-07. Verdict: **carry forward** (Registration 4
below passed its K_1_2 judgement); not promoted into `src/medaka3d/`. Expiry
remains 2026-11-27.

The pre-registered decision rule reads **carry_forward**: the label-free
contrastive embedding (A3) meets both conditions on P-1-1 at lambda 0.767. It
is parked rather than promoted because this is one recording of one trio,
trained and evaluated transductively on the same video. The external blocker
to promotion is a second recording with an independent identity reference
that played no part in designing this method (K_1 or P_1_2--3 through the
ID-pass reference adapter). Registration 2 (below) adds a naming stage over
all post-preparation time and pose-masked crops. With both, 97.41% of assessed
fish-seconds are named, 0.181 fish-s of them wrong; the rest is explicitly
unknown.

## Decision-rule readout

| Condition | Threshold | A3 | Met |
| --- | --- | --- | --- |
| (a) chain-wrong boundaries where A3 selects the reference mapping | >= 0.5 | 5 / 9 = 0.556 | yes |
| (b) lower-risk wrong-ID reduction, constrained global labelling | >= 50% | 815.558 -> 21.804 fish-s, -97.3% | yes |
| (b) wrong boundary mappings, A3 vs A0 | A3 <= A0 | 0 vs 9 | yes |

The four chain-wrong boundaries A3 did not correct (527.911, 536.364, 544.498,
559.156 s) are all undecidable, not wrong.

## Boundary decisions (127 reference-known boundaries)

| Method | Window | Correct | Wrong | Undecidable | Chain-wrong corrected | Newly wrong | Male edges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 chain | -- | 118 | 9 | 0 | 0 / 9 | 0 | 238 / 254 |
| A1 colour | end 3 s | 62 | 6 | 59 | 5 / 9 | 6 | 125 / 136 |
| A1 colour | whole | 69 | 2 | 56 | 5 / 9 | 2 | 138 / 142 |
| A3 contrastive | end 3 s | 68 | 0 | 59 | 5 / 9 | 0 | 136 / 136 |
| A3 contrastive | whole | 71 | 0 | 56 | 5 / 9 | 0 | 142 / 142 |
| A3 early (secondary) | end 3 s | 68 | 0 | 59 | 5 / 9 | 0 | 136 / 136 |
| A3 early (secondary) | whole | 71 | 0 | 56 | 5 / 9 | 0 | 142 / 142 |

Margin (second-best minus best summed distance): A3 correct decisions have
minimum 0.057 and median 1.941 (end window). A1's wrong decisions reach a
margin of 1.223; a threshold above that would also discard 24 of A1's 62
correct decisions. No threshold was selected.

## Identity time inside lower-risk intervals (fixed 298--300 s anchor)

| Naming | Wrong fish-s | Corrected | Newly wrong |
| --- | --- | --- | --- |
| A0 chain | 815.558 | -- | -- |
| A1 sequential | 4,039.330 | 331.326 | 3,555.099 |
| A1 constrained global | 26.287 | 790.886 | 1.615 |
| A3 sequential | 62.508 | 753.050 | 0.000 |
| A3 constrained global | **21.804** | 793.754 | **0.000** |

A3 global labelling converged in 2 rounds. A1 sequential fails because one wrong
boundary renames everything after it; A3 sequential has no wrong boundary, and
its residual comes from undecidable boundaries carrying A0's mapping.

**Where the residual is.** All 21.804 wrong fish-s of A3 global lie in the 34
intervals it could not assign, which carry A0 naming by the registered
fallback. Every interval A3 assigned is fully correct in lower-risk time. The
34 intervals total 23.02 s of the 1,709.47 s lower-risk time; 23 are shorter
than 0.5 s. They are unassignable because some fragment has fewer than five
non-overlap crops in every camera. Short duration accounts for 11 of them.
The rest are overlap exclusions: intervals 37 and 38 (539.9--544.5 s) have
41--47 crops per slot per camera but 0--29 outside the 80 px overlap radius.
Twelve of the 34 carry any wrong time. The 527--559 s close-contact cluster
(intervals 29--43) holds 19.298 of the 21.804 residual fish-s and four of the
nine chain-wrong boundaries.

## Evidence that the embedding encodes individuals, not position

1. **It overrules geometric continuity.** At the five decidable chain-wrong
   boundaries, A3 picks the mapping the chain's continuity rejected. An
   embedding of tank position or lighting would agree with the chain there.
2. **Same-fish fragments from different intervals collapse together.**
   Training never pairs fragments across intervals. Yet over male fragments
   from different intervals, same-fish descriptor distance has median
   0.008--0.058 against 0.997--1.035 for the other male. The AUC is 0.998--1.000
   in every 3D-separation bin (0--30 to >100 mm). A1 colour shows the
   illumination effect the supervisor described: same-fish distance rises with
   separation (top 0.63 -> 1.30), and the AUC falls from 0.93 to 0.66--0.81.
3. **Temporal stability.** A3 early, trained only on intervals ending before
   1100 s (28,878--32,200 crops per camera against 76,685--81,275), made the same
   permutation on every boundary as the full model. That includes all 54 boundaries
   after 1203 s (3 / 3 chain-wrong corrected, 0 wrong). Its embeddings differ
   from the full model's (maximum absolute difference 0.49--0.67); the
   identical scores are agreement, not reuse.

A1 colour alone already supports constrained global labelling (26.3 fish-s).
A3's advantage is boundary-level: no wrong decision, and position-independent
separation.

## Run facts

- Crops (every 8th frame inside lower-risk intervals): top 105,868 (24,593
  overlap), side1 104,365 (27,680), side2 105,069 (24,585). No pose-less crop.
  Video SHA-256 values match `video_review_v1`; five seek checks per camera
  were identical to sequential decoding.
- Training: 3000 steps per model, CPU, about 26 min each. Final medians:
  positive-pair distance 0.008--0.022, negative-pair distance 1.025--1.182.
- Reference used only in evaluation; the A0 baseline reproduces the frozen
  lower-risk score (815.558 fish-s) by assertion.

## Limits

- One recording, one trio, one lambda; n = 1 for any transfer claim.
- Transductive by design (as in idtracker.ai): agreement here says the signal
  exists in this video, not that a model carries to other fish or days.
- The reference is the Jingyi review. Agreement with it is consistency between
  two sources, not independent ground truth.
- Only lower-risk time is scored. The 129 warning intervals (90.795 s) are not
  renamed; their baseline wrong-ID time is 848.417 - 815.558 = 32.859 fish-s by
  subtraction, not recomputed here.
- 56--59 of 127 boundaries are undecidable as built. The A0 fallback is the
  only source of residual error; abstaining there would turn it into unknown
  time instead.

## Registration 2: naming stage and pose-masked crops

Both rules were registered in the lab README before any masked crop or score
existed. Part 1 reads **carry_forward**; Part 2 reads **adopt**. The lab
verdict stays **parked** on the same blocker, a second recording.

Scored over all reference-assessed post-preparation time (5,338.76 fish-s:
lower-risk and warning interiors), fixed 298--300 s anchor:

| Naming | Correct | Wrong | Unknown | Named | Wrong / named |
| --- | --- | --- | --- | --- | --- |
| A0 chain | 4,490.347 | 848.417 | 0 | 100% | 15.9% |
| N with A3, warning carry | 5,168.194 | 0.096 | 170.474 | 96.81% | 0.0019% |
| N with A3, warnings unknown | 5,096.847 | 0.000 | 241.917 | 95.47% | 0 |
| **N with B, warning carry** | **5,200.553** | **0.181** | **138.030** | **97.41%** | **0.0035%** |
| N with B, warnings unknown | 5,112.174 | 0.000 | 226.590 | 95.76% | 0 |

| Rule | Threshold | Value | Readout |
| --- | --- | --- | --- |
| Part 1, N with A3: wrong / named | <= 1% | 0.0019% | met |
| Part 1, N with A3: named / assessed | >= 90% | 96.81% | met -> carry_forward |
| Part 2, named fish-s, B vs A3 | B > A3 | 5,200.73 vs 5,168.29 | met |
| Part 2, B wrong / named | <= 1% | 0.0035% | met |
| Part 2, wrong boundary decisions, B vs A3 | B <= A3 | 0 vs 0 | met -> adopt |

Checks that passed by assertion: the chain naming reproduces 815.558 fish-s
(lower-risk) and 848.417 fish-s (whole recording, `paired_identity_time`); A3
boundary counts and global names reproduce registration 1; 2,048 stored A3
embeddings per camera match recomputation from the r1 models exactly.

**Masking.** Usable crops rose from 76,685--81,275 per camera to
101,680--102,685; 85% of registration 1's overlap crops became usable (20,821 /
24,996 / 22,201 per camera) and one non-overlap crop became unusable. Boundary
decisions with B: 105 correct, 0 wrong, 22 undecidable (A3: 68 / 0 / 59).
Intervals named globally: 118 of 130 (A3: 96); one more by elimination; 11
unknown (A3: 15, plus 19 partial). Lower-risk unknown time fell from 18.695 to
3.368 fish-s. Undrawn objects remain unmasked: side-view surface reflections
(the same fish mirrored) and parts of strongly bent or out-of-focus bodies.

**Where the remaining unknown is (N with B).** 134.66 of the 138.03 unknown
fish-s are in 27 warning interiors, almost all with all three slots unknown.
Two long close encounters hold 107.98 fish-s: 613.681 s for 32.432 s and
510.108 s for 17.785 s. In each, a flanking lower-risk interval is a sliver
(6 ms and 18 ms) with no crop, so no flank names exist to carry. The four
chain-wrong boundaries of the 527--559 s cluster stay undecidable for the same
reason (slivers of 0.018--0.24 s with 0--5 sampled crops). The naming stage
still names the longer intervals around them.

**Where the remaining wrong time is.** 0.181 fish-s, all on males, in carried
slots of three warnings (367.119 s, 0.084; 539.660 s, 0.090; 1972.851 s, 0.006).
In each, the slot's flanks agree but the reference changes inside. A carried
name cannot see a swap that reverts within one warning.

**Physical naming, observation only.** In all three cameras the anchor's M2 is
the male with the higher median red chromaticity and saturation (top camera
0.401 / 0.392 against M1's 0.374 / 0.288), whereas `group_list.csv` records M1
as red. The supervisor noted that it is unknown whether those colour
descriptors match the video. The objective is consistency within a recording,
so this is recorded, not resolved. The female's median top-camera mouth--tail
length is 106.8 px against 118.8 and 118.0 px for the males.

## Registration 3: bridging (S) and appearance inside gaps (G)

Both rules were registered before any warning-frame crop or score existed.
**Both read reject.** The lab's current output therefore remains Registration
2's N_B with warning carry. N_B gives 119 anchors and 118 gaps (91.94 s; 226.59
assessed fish-s). The label-free switch rate is 9 disagreeing gaps / 91.94 s =
0.098 per second. Calibrated `tau` is 0.186 / 0.195 / 0.155 (top / side1 /
side2), from 20,786--24,947 crops per camera.

Scored on gap time (fixed anchor):

| Naming | Named | Wrong | Wrong / named | Unknown | Readout |
| --- | --- | --- | --- | --- | --- |
| N_B warning carry (Registration 2) | 88.560 | 0.181 | 0.20% | 138.030 | baseline |
| S, bridging | 200.628 | 4.085 | **2.04%** | 25.961 | reject (> 1%) |
| G, appearance inside gaps | 217.155 | 23.883 | **11.00%** | 9.435 | reject (> 1%) |

Sensitivity, reported only: `alpha` = 1 gives 12.1% wrong and 1/16 gives 7.5%;
posterior 0.9 gives 11.7% and 0.999 gives 9.8%. No setting comes near 1%.

**Where the errors are.** Nearly all of the error is in the two long close
encounters:

| Encounter | S named / wrong | G named / wrong |
| --- | --- | --- |
| 613.681 s, 32.432 s | 69.093 / 3.904 (5.7%) | 63.677 / 12.869 (20.2%) |
| 510.108 s, 17.785 s | 17.707 / 0.000 | 36.571 / 10.766 (29.4%) |

S carries across encounter 49 because its flanking anchors agree, but a swap
and return inside it leaves 3.9 fish-s wrong. Outside these two gaps, G's
wrong time is 0.175 fish-s. It falls in two of the three warnings already
wrong under N_B; the third (1972.851 s) is corrected. The eight short gaps
holding the other chain-wrong boundaries have no G error, with 0.08--0.27 fish-s
each left unknown. Inside the encounters G also names the female wrongly
(10.9 fish-s), which no earlier stage did.

**Why G fails there: the crop evidence is wrong, not the gap model.** This is
a post hoc scratch diagnosis, used to explain and not to tune. Single-crop
nearest-identity accuracy against the reference:

| Crops | top | side1 | side2 |
| --- | --- | --- | --- |
| lower-risk, Registration 1 overlap, usable (calibration set) | 97.8% | 98.0% | 98.4% |
| other warning interiors | 95.5% | 92.8% | 93.7% |
| the two long encounters | 76.2% | 70.3% | 71.5% |
| encounters: female-slot crops called male | 0% | 52% | 41% |

Wrong encounter crops are confident: median maximum posterior 0.975--0.990.
Only 65% of encounter slot-frames with two or more cameras agree across
cameras, and 88.5% of encounter frames lack a source row for some slot and
camera. Tempering cannot repair evidence that is systematically wrong.

**Crop identity audit (2026-11-06, supervisor's question).** Is the training
data what it claims to be? For every crop, the three reference fish's 3D
pectoral points (`reviewed_3d.csv`, excluding interpolated, bad-3D and
uncertain rows) were projected with the tracking run's refractive
calibration. The crop's detection row was assigned to the nearest projection,
counting only when that projection is under 25 px away and the second-nearest
is at least 25 px further. As a sanity check, the chain's own `xyz` reprojects
a median 2.6 px from its source rows.

- *Training crops* (lower-risk, usable): 98.3--99.3% are clearly attributable.
  Of those, **100.00%** show the slot's reference fish (99.96% under a looser
  40 / 10 px rule). No two slots share a detection row.
- *Warning crops*: 80--83% are clearly attributable. Of those, 99.93--100%
  show the slot's fish.

The chain's detection rows are therefore correct; an earlier draft of this
section that blamed slot-to-detection mismatches is withdrawn. The encounter
errors come from crop content. In the encounters, model accuracy on clearly
attributable, unoccluded crops is 97.2--97.6% in the side views and 86.3% in
the top view. It falls to 19--70% when the own body is 1--20% covered by
another row's band, a range the usable rule admits. It falls to 14--43% when
two fish are close in that view. Such crops are rare in lower-risk training
data (own-body occlusion q90 0.01--0.08), so the encoder never learned them
and is confidently wrong on them.

**Deviation recorded.** The warning crops' mask fill was computed from the
warning crops themselves (top [33, 41, 35], side1 [141, 151, 150], side2 [134,
144, 143] BGR), not from B's lower-risk crops ([31, 39, 33], [142, 150, 149],
[135, 144, 142]). That is a difference of at most 2 grey levels, noticed after
extraction and left as is.

## Registration 4: clean crops, one side model, silhouette stopping; judgement on K_1_2

Registered 2026-11-07 before any flipped crop, pooled model, silhouette value
or K_1_2 crop existed (README, Registration 4). Two runs share
`run_appearance_clean.py` and `config_appearance_clean_v1.toml`:
`appearance_clean_20261107_r1` on P-1-1 (development, read while working) and
`appearance_clean_20261107_k12` on K_1_2 (judgement, read once).

### What changed

- **X, clean crops.** Training, descriptors, temperature and gap evidence use
  only crops with no own-body occlusion, no other keypoint within 80 native px
  of the centre, and at least three detection rows in that camera-frame. This
  keeps 73.5 / 69.7 / 73.7% of P-1-1's lower-risk crops (top, side1, side2)
  and 68.1 / 69.3 / 72.9% of K_1_2's; inside warnings 24--28% and 16--17%.
- **D, dorsal-up.** Side crops whose heading points left are flipped
  vertically after masking: 34.8% and 58.7% of P-1-1's side1 and side2 crops,
  38.4% and 53.9% of K_1_2's. Previously these fish were shown belly-up.
- **W, one side model.** Side1 and side2 pooled: 150,165 (P-1-1) and 148,381
  (K_1_2) clean training crops, positives across cameras allowed.
- **T, silhouette stopping.** k-means (3 clusters) silhouette on 3000
  validation crops every 100 steps; stop at two evaluations >= 0.91, 30
  without improvement, or 6000 steps; best parameters kept.

### Training, and separation during training

| Model | Recording | Training crops | Fragments | Stopped | Best step | Silhouette step 0 -> best | Cluster purity vs reference, step 0 -> best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top | P-1-1 | 77,805 | 327 | target | 700 | 0.34 -> 0.96 | 0.38 -> 0.999 |
| side | P-1-1 | 150,165 | 320 | target | 1000 | 0.31 -> 0.92 | 0.44 -> 0.987 |
| top | K_1_2 | 71,270 | 487 | target | 300 | 0.34 -> 0.96 | no reference |
| side | K_1_2 | 148,381 | 511 | target | 700 | 0.34 -> 0.92 | no reference |

All four models stopped on the 0.91 target after 300--1000 steps, against the
fixed 3000 of Registrations 1--3. Cluster purity (reference colouring, drawn
at evaluation only) rises from chance at random initialisation to 99.9%
(top) and 98.7% (side) at the best step. In the side model the two cameras
overlap inside every cluster, and the median embedding distance between the
same fish seen by the two side cameras at the same frame ends at 0.04,
against about 1.0 between different fish: the model is camera-invariant.
Figures: `evaluation/figures/{top,side}_separation.png` (PCA panels at
evaluations 0, 1, 3, 10, best, last; rows: k-means cluster, reference
identity on P-1-1, camera) and `{top,side}_curves.png` (silhouette, pair
distances, cross-camera distance). The same figures exist for K_1_2 without
the reference row.

### P-1-1, development readouts

| Method | Named / assessed | Wrong fish-s | Wrong / named |
| --- | --- | --- | --- |
| Chain (A0) | 100% | 848.417 | 15.9% |
| Registration 2, N_B carry | 97.41% | 0.181 | 0.003% |
| Registration 4, N carry | 96.18% | 0.042 | 0.001% |
| Registration 4, S | 99.36% | 4.085 | 0.077% |
| Registration 4, G, clean evidence | 98.79% | 0.729 | 0.014% |

Lower-risk time: N names 99.26% with 0 wrong (Registration 2: 99.93%).
Intervals: 91 global, 9 partial-two, 13 partial-one, 17 unknown; clean
descriptors need five clean crops per camera, so more slivers stay unknown
than with B. Boundaries (end window, 127 reference-known): 66 correct, 0
wrong, 61 undecidable; 5 of the 9 chain-wrong boundaries corrected as before,
the 527--559 s cluster still undecidable.

Gap time (108.39 s in 99 gaps):

| Method | Named gap time | Wrong fish-s | Wrong / named |
| --- | --- | --- | --- |
| N carry | 25.8% | 0.042 | 0.06% |
| S | 87.6% | 4.085 | 1.70% |
| G, Registration 3 (all usable crops) | 70.9% | 23.9 | 11.0% |
| G, Registration 4 (clean crops) | 76.6% | 0.729 | 0.35% |

The two long encounters: at 613.7 s, G names 35.8 s correctly, 0 wrong, 33.3
s unknown (S: 65.2 correct, 3.9 wrong); at 510.1 s, G names 21.0 s correctly,
0.48 s wrong, 17.4 s unknown (S: 17.7 correct, 0 wrong, 21.2 unknown).

Rules. The N carry rule is met (0.001% wrong, 96.18% named). G versus S on
gap time reads **reject** by the letter: the registered rule requires G to
name more gap time than S, and G names 210.8 against S's 240.9 fish-s. S
itself exceeds the 1% bound (1.70%), so neither variant is adopted on gap
time by the rules as written. Against Registration 2's carry (25.8% named) G
would meet both adoption conditions; that comparison was not the registered
rule and is reported only. Excluding crossing-like crops removed the
Registration 3 failure mode: wrong gap time fell from 23.9 to 0.73 fish-s.

### K_1_2, judgement

Label-free fragments from the production chain output (2026-09-03): 230
lower-risk intervals (1694.5 s) and 229 warnings (105.8 s) of 1800.3 s
post-preparation. Reference, read once: 299 of 300 confirmed female
identifications map to one slot (1 view disagreement; 9 and 13 side-view
links lie farther than 5 px from any detection); 255 fall after 300 s and
are scored, 4 of them inside warnings and 7 inside gaps.

| Method | Correct anchors | Wrong | Unknown | Named |
| --- | --- | --- | --- | --- |
| Chain (A0) | 196 | 59 | 0 | 100% |
| N carry | 250 | 0 | 5 | 98.0% |
| N strict | 248 | 0 | 7 | 97.3% |
| S | 254 | 0 | 1 | 99.6% |
| G, clean evidence | 255 | 0 | 0 | 100% |

**Rule: met.** 0 wrong of 250 named anchors (limit 1%) and 98.0% named
(floor 90%). The chain's 59 wrong anchors lie between its two female swaps,
the same two events `c09` counts as interval outcomes; the appearance names
undo both, as the anchors confirm. On the 7 gap anchors N names 2, S 6 and G
7, all correctly.

Label-free diagnostics, not accuracy: 138 global, 12 partial-two, 22
partial-one and 58 unknown intervals; 95 of 229 end-window boundaries
decided (median margin 2.0), 134 undecidable; across the 149 consecutive
named intervals the names change at 16 against slot continuation, 2
involving the female (the chain's two swaps) and 14 male-only. The male-only
changes cannot be checked on K_1_2.

### Run facts, Registration 4

CPU only, 12 threads per model, two models in parallel. P-1-1: bundle 3 min,
prepare 22 min, train 26 min, evaluate 2 min. K_1_2: extract 53 min (three
videos, 37,350 sampled frames), prepare 12 min, train 16 min, evaluate 2
min. Both evaluations were re-run once after the label-free male diagnostic
was corrected (it had skipped pairs separated by an unnamed interval); no
rule input changed. The K_1_2 chain predates the 2026-11-02 adapter repair;
see the registration.

### Limits, Registration 4

- K_1_2's reference is the female only; a male-male error there is
  invisible. The 14 male-only name changes are unverified.
- Stopping fired at 300--1000 steps. Whether longer training changes anything
  was not measured; the rule fixed it.
- The clean criterion leaves 58 of K_1_2's 230 intervals unnamed, mostly
  short ones; they cost 2% of anchors.
- 56 px input on a CPU PyTorch build; no GPU was used.

## Registration 5: input resolution (R) and encoder architecture (A), on the GPU

Registered 2026-11-07 after Registration 4's result and before any 128 px
crop, ResNet model or GPU-trained encoder existed (README, Registration 5).
Each variant changes one thing in the Registration 4 pipeline and runs on
both recordings through `run_appearance_clean.py --config`. Primary readout:
single-crop nearest-centroid accuracy on P-1-1's clean, reference-clear
warning crops, against Registration 4's encoders on the common crop set;
safety: the K_1_2 anchor rule; adopt at a pooled gain of at least 2 points
with no camera more than 1 point below, reject when any camera is more than
2 points below or K_1_2 fails. Neither variant is adopted: A is rejected and R
inconclusive with a measured gain of 0.0 points, so Registration 4's pipeline
(56 px input, four-block CNN) remains the carry-forward state.

### Variant A: ResNet18 with a bias-free 8-d head

Runs `appearance_resnet_20261107_r1` (P-1-1) and `_k12`,
`config_appearance_resnet_v1.toml`. The encoder is torchvision's ResNet18 at
random initialisation with an 8-d output; crops, masks, clean flags, loss,
batch, seed and stopping are Registration 4's. Because the prepared 64 px
crops are Registration 4's own, the common crop set equals each run's full
set: 1,117 / 1,239 / 1,403 reference-clear warning crops (top, side1, side2).

**Rule: reject.** Single-crop accuracy on the common set:

| Camera | Registration 4 (CNN, 16-d) | Variant A (ResNet18, 8-d) | Difference, points |
| --- | --- | --- | --- |
| top | 99.02% | 90.96% | -8.06 |
| side1 | 99.03% | 98.63% | -0.40 |
| side2 | 98.86% | 98.36% | -0.50 |
| pooled (3,759 crops) | 98.96% | 96.25% | -2.71 |

The top camera is more than 2 points below, which is the reject condition;
the side cameras are within sampling noise of the baseline.

Training and separation:

| Model | Recording | Stopped | Best step | Silhouette step 0 -> best (last) | Cluster purity vs reference, step 0 -> best | Cross-camera median at best |
| --- | --- | --- | --- | --- | --- | --- |
| top | P-1-1 | target | 300 | 0.20 -> 0.92 (0.92 at 400) | 0.39 -> 0.991 | n/a |
| side | P-1-1 | target | 1000 | 0.27 -> 0.94 (0.93 at 1100) | 0.43 -> 0.988 | 0.036 |
| top | K_1_2 | target | 400 | 0.22 -> 0.96 | no reference | n/a |
| side | K_1_2 | target | 800 | 0.30 -> 0.94 | no reference | 0.034 |

All four models stopped on the 0.91 target; the P-1-1 top model after 300
steps (38,400 pairs, under one pass over its 77,805 crops), where
Registration 4's CNN stopped at 700. At that best step the separation figure
shows one male's validation cluster trailing towards the other two while the
female's is already tight. The side model is camera-invariant as before
(same fish across the two side cameras at median distance 0.036, different
fish at about 1.0).

P-1-1 naming and gap readouts, same definitions as Registration 4:

| Method | Named / assessed | Wrong fish-s | Wrong / named |
| --- | --- | --- | --- |
| Registration 4, N carry | 96.18% | 0.042 | 0.001% |
| Variant A, N carry | 96.17% | 0.337 | 0.007% |
| Registration 4, G, clean evidence | 98.79% | 0.729 | 0.014% |
| Variant A, G, clean evidence | 98.74% | 0.175 | 0.003% |

S is unchanged (99.36%, 4.085 fish-s wrong). Lower-risk time: N names
99.26% with 0.295 fish-s wrong. Intervals 91 / 9 / 13 / 17 and boundaries 66
correct, 0 wrong, 61 undecidable (5 of 9 chain-wrong corrected), all as in
Registration 4. Gap time: N carry names 25.7% with 0.337 fish-s wrong
(0.48%), G names 75.5% with 0.175 wrong (0.08%); at the 613.7 s encounter G
names 37.5 s correctly, 0 wrong, 31.6 s unknown, and at 510.1 s 17.7 s
correctly, 0 wrong, 21.2 s unknown. The N carry rule is met; G versus S
reads reject as in Registration 4.

K_1_2, judged once on the same 255 female anchors: N carry 250 correct, 0
wrong, 5 unknown (98.0% named); N strict 248 / 0 / 7; S 254 / 0 / 1; G 255 /
0 / 0; the 7 gap anchors named 2 / 6 / 7 by N, S and G, all correctly. The
anchor rule is met, and every count equals Registration 4's, including the
label-free diagnostics (138 / 12 / 22 / 58 intervals, 95 boundaries
decided with median margin 1.9, 16 name changes across 149 consecutive
named intervals, 2 involving the female and 14 male-only).

Reading. On the readout registered as primary, ResNet18 with silhouette
stopping is worse than the small CNN on top-view warning crops and equal on
the side views; the identity-track readouts barely move because they rest on
clean lower-risk crops, where both encoders are near-saturated. The stopping
rule fixed by Registration 4 let the ResNet stop after 300 steps; whether the
deficit belongs to the architecture or to that early stop is not separable
inside this registration, and a ResNet run under a different stopping rule
would be a new registration. idtracker.ai's D_neg = 10 was not copied.
Registration 4's pipeline remains the carry-forward state.

### Run facts, Registration 5, variant A

Device `cuda:0` (RTX 5070 Ti, torch 2.14.1+cu130), 12 CPU threads per model,
two models in parallel. Wall times: P-1-1 bundle 6 min, prepare under 1 min
(prepared crops are read from the Registration 4 run), train 56 min,
evaluate 7 min; K_1_2 bundle 3 min, train 19 min, evaluate 3 min. The
training wall time is not a GPU time: the models ran 400--1100 steps while
variant R's six video decodes wrote 128 px crops to the same network share,
and the share's throughput during that window was about 10 MB/s. Which of the
two (share throughput or CPU contention from the decoders) dominated was not
isolated.

### Limits, Registration 5, variant A

- The single-crop readout uses P-1-1's reference, which has informed every
  registration; K_1_2 checks the female only.
- One seed per encoder; the stopping rule is Registration 4's and fired early
  for the ResNet (see Reading).
- Wall times were measured under concurrent I/O load and do not describe the
  GPU.

### Variant R: 128 px crops, 112 px input

Runs `appearance_res128_20261107_r1` (P-1-1) and `_k12`,
`config_appearance_res128_v1.toml`. Crops were decoded again from video at
128 px (0.8 of the native 160 px crop; fish span about 75--90 px), masks,
clean flags and the dorsal-up flip computed at that scale, training on
random 112 px windows and inference on the central 112 px. Encoder, loss,
batch, seed, stopping and all decisions are Registration 4's. Re-decoding
and re-masking changed the clean sets slightly: 77,952 / 72,803 / 77,456
clean lower-risk crops on P-1-1 (64 px: 77,805 / 72,756 / 77,409) and
71,308 / 72,322 / 76,127 on K_1_2 (64 px: 71,270 / 72,283 / 76,098), so the
common reference-clear warning set holds 3,758 of the 3,759 crops.

**Rule: inconclusive.** Single-crop accuracy on the common set:

| Camera | Registration 4 (64 px, 56 px input) | Variant R (128 px, 112 px input) | Difference, points |
| --- | --- | --- | --- |
| top | 99.02% | 98.57% | -0.45 |
| side1 | 99.03% | 99.52% | +0.48 |
| side2 | 98.86% | 98.79% | -0.07 |
| pooled (3,758 crops) | 98.96% | 98.96% | 0.00 |

The pooled gain is 0.0 points against the 2-point adoption threshold, and
no camera moves by more than half a point. The resolution question the
supervisor asked is answered on this readout: 56 px input suffices, and
resolution is not what limits identity on warning crops.

Training and separation:

| Model | Recording | Training crops | Fragments | Stopped | Best step | Silhouette step 0 -> best (last) | Cluster purity vs reference, step 0 -> best | Cross-camera median at best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top | P-1-1 | 77,952 | 327 | target | 500 | 0.41 -> 0.95 | 0.39 -> 0.9997 | n/a |
| side | P-1-1 | 150,259 | 320 | target | 1500 | 0.38 -> 0.94 (0.92 at 1900) | 0.42 -> 0.995 | 0.033 |
| top | K_1_2 | 71,308 | 488 | target | 500 | 0.41 -> 0.97 (0.94 at 600) | no reference | n/a |
| side | K_1_2 | 148,449 | 511 | target | 900 | 0.35 -> 0.93 | no reference | 0.037 |

The figures (`evaluation/figures/` of both runs) show the same picture as
Registration 4: three tight clusters, one per reference fish on P-1-1, the
two side cameras mixed inside every cluster. Nearest-centroid agreement of
the clean lower-risk crops with their interval's name is 99.94 / 99.36 /
99.59% (top, side1, side2).

P-1-1 naming and gap readouts, same definitions as Registration 4:

| Method | Named / assessed | Wrong fish-s | Wrong / named |
| --- | --- | --- | --- |
| Registration 4, N carry | 96.18% | 0.042 | 0.001% |
| Variant R, N carry | 96.10% | 0.042 | 0.001% |
| Registration 4, S | 99.36% | 4.085 | 0.077% |
| Variant R, S | 99.18% | 4.031 | 0.076% |
| Registration 4, G, clean evidence | 98.79% | 0.729 | 0.014% |
| Variant R, G, clean evidence | 98.81% | 0.729 | 0.014% |

Lower-risk time: N names 99.18% with 0 wrong. Intervals 91 / 8 / 14 / 17;
boundaries 66 correct, 0 wrong, 61 undecidable, 5 of 9 chain-wrong
corrected, as in Registration 4. Gap time (98 gaps): N carry names 26.0%
with 0.042 fish-s wrong, S 84.4% with 4.031 wrong (1.70%), G 77.4% with
0.729 wrong (0.34%); at the 613.7 s encounter G names 37.9 s correctly, 0
wrong, 31.2 s unknown, and at 510.1 s 21.1 s correctly, 0.48 s wrong, 17.4 s
unknown. The N carry rule is met; G versus S reads reject as before.

K_1_2, judged once on the same 255 female anchors: N carry 250 correct, 0
wrong, 5 unknown (98.0% named); N strict 248 / 0 / 7; S 254 / 0 / 1; G 255 /
0 / 0; the 7 gap anchors named 2 / 6 / 7 by N, S and G, all correctly. The
anchor rule is met, and every count equals Registration 4's and variant
A's, including the label-free diagnostics (138 / 12 / 22 / 58 intervals, 95
boundaries decided with median margin 2.0, 16 name changes across 149
consecutive named intervals, 2 involving the female and 14 male-only).

### Run facts, Registration 5, variant R

Device `cuda:0` (RTX 5070 Ti, torch 2.14.1+cu130), 12 CPU threads per
model, two models in parallel, the two recordings one after the other in
the final attempt. Extract took 3.5 h and prepare 1.5 h per recording on
2026-11-07, both recordings in parallel, the CPU near 6%: six decoders wrote
5.4 GB per view to the network share at about 10 MB/s in total, and
prepare read it back. Training on 2026-11-08: P-1-1 train 35 min and
evaluate 8 min, K_1_2 train 29 min and evaluate 8 min, each train including
the load of 4 + 7 GB of crops from the share.

Two earlier train attempts failed before any evaluation existed, so no
score was seen before the rule was applied:

- 2026-11-07, 17:23--17:57: `channel_stats` converted the whole crop array
  to float64 at once, 55 GiB for the side model's 150k crops at 128 px; the
  allocation failed on both recordings after the top models had trained.
  Fix: a chunked two-pass computation of the same per-channel mean and
  population standard deviation (test added). Partial outputs deleted; the
  failed P-1-1 train record kept as
  `appearance_clean_train.failed-1.provenance.json`.
- 2026-11-07 18:14 to 2026-11-08 10:20: both recordings in parallel, three
  training processes on the GPU. Inference at 1024 crops per batch on
  112 px input reserved 5--6 GB per process through the caching allocator;
  the 16 GB card was oversubscribed and two processes spilled into system
  memory (1.0 and 2.1 GB). The GPU read 100% busy at 64 W; K_1_2's top model
  took 4.7 h for 600 steps and the others never finished. Killed after 16 h.
  Fixes: the inference batch now scales with crop area (1024 at 64 px, 256
  at 128 px; the embedding does not depend on the batch, test added) and
  the recordings run one at a time. Partial outputs deleted again.

Neither fix changes a statistic or a decision; both change memory use
only. The run snapshots (`source/`, `source_evaluate/`) hold the code of
the attempt that produced each output.

### Limits, Registration 5, variant R

- One seed per encoder; the stopping rule fired at 500--1500 steps as in
  Registration 4.
- The 128 px crops are a fresh decode with masks at a new scale, so the
  crop sets differ from Registration 4's by a few tens of crops per camera;
  the common-set comparison removes that from the primary readout.
- The single-crop readout uses P-1-1's reference, which has informed every
  registration; K_1_2 checks the female only. The male-only name changes on
  K_1_2 are still unverified.
- Wall times include network-share I/O and do not describe the GPU.

## Next steps (each needs its own registration)

1. Registration 5 (2026-11-07/08) closed both encoder questions on the
   registered readout: ResNet18 with an 8-d head is rejected, and 128 px
   input is inconclusive at a measured gain of 0.0 points, so 56 px input
   and the four-block CNN stay. Any further encoder registration needs its
   own stopping rule declared, since the 0.91 silhouette target stops a
   ResNet before one pass over the crops, and should stage its crops on
   local disk before training (see the variant R run facts).
2. A gap policy: G with clean evidence is five times less wrong than S but
   names less; a combined rule must be declared before it is scored.
3. Transfer to P_1_2--3 and K_1_1/K_1_3: done on 2026-11-08 in
   `analyses/20261108_six_recording_identity/` (Registration 4's pipeline
   unchanged, per-recording and per-trio encoders, scored against the
   ID-pass reviews of all six recordings through a position-matched
   reference adapter). Five of six recordings meet the delivery thresholds
   with every candidate; the sixth disagrees with the review on one 95 s
   male--male episode where the chain's geometry sides with the appearance.
   The per-trio encoders also match the three individuals across the three
   days of each trio with a wide margin. See that README for numbers and
   limits; nothing here is promoted by it.
4. Promotion of the stage into `src/medaka3d/` once the gap policy is settled
   (design rule 1).

## Artifacts

Run directory, under the configured data parent:
`jingyi/P-1-1/appearance_link_20261106_r1` (`crops/`, `models/`,
`embeddings/`, `evaluation/`, three provenance records, source snapshots).
Commands, in order:

```powershell
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_link.py extract
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_link.py train
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_link.py evaluate
```

Configuration: `analyses/20261031_p11_identity_evaluation/config_appearance_link_v1.toml`.

Registration 2 run: `jingyi/P-1-1/appearance_naming_20261106_r1` (`masked/`,
`models/`, `embeddings/`, `evaluation/` with `identity_tracks_N_A3.npz` and
`identity_tracks_N_B.npz` holding per-frame slot names and `xyz` regrouped by
identity, `identity_sheet_*.png`, four provenance records, source snapshots).
Configuration `config_appearance_naming_v1.toml`; commands, in order:

```powershell
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_naming.py mask
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_naming.py train
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_naming.py evaluate
```

Registration 3 run: `jingyi/P-1-1/appearance_gap_20261106_r1` (`crops/` and
`masked/` for 1,885 sampled warning frames, `embeddings/`, and `evaluation/`
with `gap_table.csv`, `identity_tracks_S.npz` and `identity_tracks_G.npz`).
Configuration `config_appearance_gap_v1.toml`; commands, in order:

```powershell
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_gap.py extract
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_gap.py embed
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_gap.py evaluate
```

Registration 4 runs: `jingyi/P-1-1/appearance_clean_20261107_r1` and
`work/appearance/20251225_K_1_2_FCMC/appearance_clean_20261107_k12`
(`bundle/` with the label-free fragments and hashed inputs, `clean/` and
`clean_warning/` flipped crops with the clean flag, `models/` with
`*_history.npz` holding the validation embeddings at every evaluation,
`embeddings*/`, `evaluation/` with `interval_names.csv`, `boundaries.csv`,
`identity_tracks_{N_carry,S,G}.npz`, `figures/`, and on K_1_2
`female_anchors.csv`). Configuration `config_appearance_clean_v1.toml`;
commands, in order (`extract` is K_1_2 only):

```powershell
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py bundle --recording p11
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py prepare --recording p11
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py train --recording p11
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py evaluate --recording p11
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py bundle --recording k12
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py extract --recording k12
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py prepare --recording k12
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py train --recording k12
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py evaluate --recording k12
```

Registration 5 runs: `jingyi/P-1-1/appearance_resnet_20261107_r1`,
`work/appearance/20251225_K_1_2_FCMC/appearance_resnet_20261107_k12`
(variant A, from `config_appearance_resnet_v1.toml`; `clean/` and
`clean_warning/` are read from the Registration 4 runs and not copied) and
`appearance_res128_20261107_r1` / `_k12` under the same parents (variant R,
from `config_appearance_res128_v1.toml`, with its own 128 px `crops_all/`,
`crops/`, `warning_crops/`, `masked/`, `warning_masked/`, `clean/` and
`clean_warning/`). Layout otherwise as for Registration 4, plus
`evaluation/summary.json["warning_crop_accuracy"]` on P-1-1. Commands are
Registration 4's with `--config <file>` appended to every call; `extract` is
needed for variant R on both recordings and for neither recording of
variant A.

Tests: `tests/test_contrastive_appearance_lab.py`. No extraction output, human
label, reference file or earlier registration's output was modified.
