# Development log from self-iteration

## 2026-09-09 18:07 — 100 s inheriting run completed, review volume measured

The reference method's `[49800,66400)`, 16,600 frames, completed with exit code 0 and success
provenance. Initializations 1, restores 3, external resets 0; all 6 candidates and 99,593
history rows saved. Chest direct support 49,521/49,800; the same support excluding ID hold
49,012; hold 739 fish-frames. Tracking time for the added 40 s was 2383.95 s; input validation
312.37 s, reader preparation 61.12 s and checkpoint saving 25.79 s are separate. It ran in
parallel with independent D2 processing. The current winning path is provisional and does not
represent the full 30 minutes or ground truth for biological IDs.

The new post-run review aggregation passed 20 CPU tests, Ruff and independent review, and was
applied to 60 s/100 s. The existing risk gap 4 and finite margin 1.0 were not changed; all
risks, the ledger and candidate probes were saved. 60 s had 0 risks; 100 s merged 5
body-support losses and 2 candidate conflicts into 1 scene. Scene range 65859:66223 (364
frames, 2.19277 s of raw frames), probes 65931/66151/66217. The raw-frame check has not been
done. At 100 s the +inf margin covers 13,653 frames; a state with no competing candidate is not
treated as high confidence. The aggregation is a diagnostic from the SHAs, config, metadata and
array values of successful outputs, not an independent full-history/raw-frame/geometry audit.

The 30-minute auto-continuation wrapper passed 37 tests + 30 audit-backend tests and runs in a
root-owned process. The bootstrap audit after the parent 100 s completed is running; the next
1-minute extension starts after that audit succeeds and 6 input chunks complete. Each phase
inherits all candidates and all histories; on failure it stops and leaves a record. Neither the
rejected reflection method nor human corrections are mixed into the frozen reference inference
engine. A plan was saved to independently inspect all support, and the raw frames at 543
pre-fixed time points, after the full 30 minutes finish. Small I/O processing for the C34
comparison and raw-frame review is in preparation.


## v0 — Fitting the bodies of all three fish to the images simultaneously (2026-09-09)

The new inference engine places 8 anisotropic Gaussians on the 3 nodes of each fish and
computes projected silhouettes against the underwater rays given by the existing refractive
calibration. It compares the union of the three silhouettes with independent masks and
optimizes all 27 coordinates simultaneously. Pose candidates include simple motion prediction,
alternative SLEAP correspondences, exchange of whole fish, and recovery from the current
frame's observations. Up to 18 candidates are evaluated, 6 histories are retained, and at the
end a path is chosen by tracing back through the past. This is an approximation with a finite
set of candidates.

No positions or IDs from the old method are input apart from the fixed initial pose. Mid-run
reinitializations: 0. The 7 conditions of the old C34 and the human clicks on the female are
read only after inference ends, and are passed to a common evaluation function. The initial
pose derives from the old method, computed using the whole recording, so the initialization
condition is optimistic.

### Known development interval 333519:333719, native 166 fps, 200 frames

| Observation condition | Pectoral fin with direct support in ≥2 views | Same support excluding ID hold | Tracking time |
|---|---:|---:|---:|
| SLEAP only | 583 / 600 (97.17%) | 416 / 600 (69.33%) | 31.69 s |
| SLEAP + independent Detectron2 masks | 562 / 600 (93.67%) | 414 / 600 (69.00%) | 79.50 s |
| Top by background subtraction, side by Detectron2 | 561 / 600 (93.50%) | 414 / 600 (69.00%) | 101.49 s |

Used an RTX5070Ti, PyTorch CUDA float64 and 14 Adam steps. Times include restoring the local
video and masks and tracking, and exclude the preceding D2 inference, input-hash verification
and evaluation. CPU review-image generation and tests partly ran in parallel, so this is not a
strict speed comparison.

In this first implementation, adding masks lowered the direct-support rate and did not improve
the hold rate. Raw data, config, code snapshot and input hashes for each run are kept in
`work/three_body/v0_transient_{nodes,detectron2,top_contrast}/`. The female was clicked at
only one time point in this interval, so the female's temporal ID retention cannot be
evaluated either. Support rate is not ID accuracy. An independent correspondence check of the
three fish has not been obtained.

### Cause diagnosis and the mechanism to change next

- Around 333544 a sharp turn and a large ripple on the water surface are visible. The
  estimated S2 has no node supported in all three views, and its predicted position is left
  behind. Whether calibration error from the ripple or detection failure is the main cause is
  not confirmed from this alone.
- The silhouette-error gradient is weak at moving a candidate onto a fish at a distant
  position. Adding independent masks alone does not automatically resolve large movements or
  the correspondence of bent bodies.
- In the first version, the ID hold is inherited forward once body support has been
  interrupted for 4 frames. This is a conservative first condition; there is not yet a
  resolution mechanism for committing a long interval from a check of a few scenes.
- The maximum single-frame displacement from a recovery candidate for S2 was 29.47 mm (SLEAP
  only) and 33.18 mm (D2). The cap on predicted velocity does not limit the change of position
  after recovery. This is a latent pose under hold and cannot be used to claim that the body
  actually moved at that speed.

The next mechanism is 3D candidate proposal from forward-backward-consistent optical flow
between adjacent raw images. It follows image features near the previous body and generates
candidates independently of the untracked SLEAP rows in the current frame. It is not used when
the image has no texture, when forward and backward disagree, or when views do not agree. This
is neither a relaxation of the success conditions nor a change that turns self-generated IDs
into ground truth.

### Limits learned from synthetic data

In a test fitting independent ellipsoid silhouettes, silhouette error on same-image pixels not
used for training improved from 0.197 to 0.071, while pectoral-fin coordinate error worsened
from 1.871 to 1.979 mm. When the approximate body shape differs from the true silhouette, a
good silhouette match does not guarantee correct anatomical points. In the mask-only
condition, direct node support and accepted coordinates both stayed at 0. Reproduction
conditions and the original failure checks were saved to
`work/three_body/synthetic_findings.json`.

In a synthetic crossing test, 3D trajectories matching the initial IDs were kept with a
maximum pectoral-fin error of 0.573 mm, but in the top view, where projections came within
1.39–2.67 px of each other, the row correspondence of 4 nodes was swapped. Agreement of close
2D detection rows and 3D individual continuity are not treated as the same evaluation.

## Current judgement

### Known development interval 76620:77286, native 166 fps, 666 frames

| Observation condition | Pectoral fin with direct support in ≥2 views | Same support excluding ID hold | Tracking time |
|---|---:|---:|---:|
| SLEAP only | 1971 / 1998 (98.65%) | 1510 / 1998 (75.58%) | 98.74 s |
| SLEAP + independent Detectron2 masks | 1882 / 1998 (94.19%) | 1773 / 1998 (88.74%) | 248.10 s |
| Top by background subtraction, side by Detectron2 | 1906 / 1998 (95.40%) | 1906 / 1998 (95.40%) | 335.85 s |

The top background-subtraction condition passed the pre-set adoption-rate criterion of 90%
overall and 80% per fish on this interval. Per-fish direct support/adoption rates were 96.70%
/ 95.65% / 93.84%. Hold flags were 0. In every condition the female's raw ID matched between
the 2 time points, and the 2 mask conditions also matched on accepted ID. With SLEAP only, the
female's ID at the later point was on hold, so that span could not be evaluated with accepted
IDs. This is the result for a single female endpoint span and guarantees nothing about the IDs
of the two males or about out-and-back exchanges within the interval.

v0 is not adopted as the finished version. Because the effect of masks differs by interval,
the motion-based candidates above will be implemented and compared. ID holds are not released
in order to produce a low hold rate in the automatic results without grounds. Independent
correspondence evaluation of the three fish, and verification of the new method over 1 minute
and about 30 minutes without reinitialization, are not complete.

## v1 — Adding candidates from raw-image motion every frame

Image features from before the loss were checked with forward-backward Lucas-Kanade, and 3D
candidates consistent in ≥2 views were added. Each history's existing prediction was kept, and
the objective and ID-hold conditions were not relaxed. A union-only path whose values and
gradients exactly match the existing renderer was also used. In a standalone GPU measurement,
render + backprop sped up by about 1.23×. Because image-motion computation is added, tracking
as a whole takes longer than v0.

| Interval · observation condition | Direct support on pectoral fin | Same support excluding ID hold | Tracking time |
|---|---:|---:|---:|
| 200 fr · Detectron2 | 586 / 600 (97.67%) | 417 / 600 (69.50%) | 97.34 s |
| 200 fr · top background subtraction | 583 / 600 (97.17%) | 417 / 600 (69.50%) | 126.89 s |
| 666 fr · Detectron2 | 1942 / 1998 (97.20%) | 1942 / 1998 (97.20%) | 317.52 s |
| 666 fr · top background subtraction | 1904 / 1998 (95.30%) | 1904 / 1998 (95.30%) | 399.97 s |

The Detectron2 condition increased direct support over v0 on both intervals. At 666 fr it
passed 90% overall / 80% per fish, and the female's accepted IDs at the 2 time points matched.
At 200 fr, S2 still had the 333536:333550 missing span and the ID hold after it. The large
latent position jump fell from a maximum of 33.18 to 28.92 mm, but this is judged to be neither
a physically smooth recovery nor a correct ID recovery. The next short-interval improvement
optimizes both sides of this gap jointly.

Independent post-hoc diagnostics found, among directly supported correspondences at 200 fr, no
new positional-correspondence differences from any of the 7 baselines. At 666 fr it chose the
same side as old λ0/2/8 and a different side from λ0.1/0.25/0.5/0.767. The female's
independent clicks at 2 time points support the former's endpoint correspondence. This is not
extended to ground-truth IDs for the two males or for all frames. See
`work/three_body/diagnostics_*` for details.

`review_events.py` counts new missing spans even when the hold flag is already set. For D2 at
200 fr, v0's 2 events fell to 1 event in v1. Counting only the entries into continuous holds
must not lead to the misreading that a long recording needs only 3 checks. The estimated
number of check intervals is not a measurement of the number of cases a human can correct.

## v2 — Jointly optimizing both sides of a missing interval

Pre-conditions are in `V2_PROTOCOL.md`. For the 31 frames 333528:333559, which contain the v1
D2 missing span, 6 right-end correspondences × 2 kinds of initial trajectory, 12 candidates in
total, were jointly optimized for 100 steps. Nodes with observation support in the 2 frames at
each end were fixed, and no additional external initialization was done. The same 256 pixels
per camera and frame were used for all candidates, and the actual pixels and weights were also
saved as NPZ with SHA. GPU run: 18.42 s. The forward tracker's initial pose, original result
and ID holds were not changed.

| End correspondence (left ID → original right slot) | Best cost | Direct support on pectoral fin (per fish / 31) | Max latent displacement over all fish / frame |
|---|---:|---:|---:|
| 0,1,2 | 3.73347 | 31 / 31 / 19 | 2.740 mm |
| 2,1,0 | 3.86200 | 26 / 31 / 24 | 4.874 mm |
| 0,2,1 | 4.28522 | 31 / 21 / 26 | 11.142 mm |
| 2,0,1 | 4.77608 | 27 / 29 / 22 | 29.201 mm |
| 1,2,0 | 4.99260 | 30 / 21 / 27 | 41.121 mm |
| 1,0,2 | 5.26973 | 30 / 29 / 19 | 38.193 mm |

Re-evaluating the original v1 with **the same new objective and the same 256 pixels** gave a
cost of 4.50135. The best, H1, reduced S2's maximum latent displacement from 28.924 to 2.470 mm
and increased pectoral-fin direct support from 17 to 19 frames. The image-error term fell
slightly, from 0.40861 to 0.39861. However, the runner-up H11 has an even smaller image term,
0.39468. The total-cost difference of 0.12852 between best and runner-up is not converted into
an ID probability, and both correspondences are saved. S2's 12 frames of direct missing remain.

In the raw-frame comparison too, the S2 estimated line at 333544 stays near the large
water-surface pattern, and neither position nor ID can be confirmed. Obtaining a smooth latent
interpolation is kept distinct from correct reconstruction of the fish body and ID recovery.
Outputs are in `work/three_body/temporal_v2_transient_333528-333559/`; the comparison, saved
for 31/31 frames at 16× slow motion and fully decode-verified, is in
`work/three_body/temporal_v2_transient_review_v2/`.

The next diagnostics are to compare silhouette error on same-image pixels not used in the
optimization, and to project independent masks onto a 3D grid to look for position candidates
that the current estimated fish bodies do not explain. Additional diagnostics derived from the
same images are not treated as independent ground truth for individual ID.

### Additional diagnostics by iteration count and unused pixels

Because the cost was still decreasing at 100 steps, a single convergence diagnostic extending
to 400 steps without changing conditions was recorded in advance in
`work/three_body/v2_convergence_registration.json`. Checkpoint values of all 12 candidates
over the first 100 steps and the SHA of the actual-sample NPZ matched exactly. The best, H1,
had cost 3.64609, and improvement stopped over the last 4 checkpoints. S2 pectoral-fin support
was 21/31, maximum latent displacement 2.601 mm. The gap to the runner-up H11 was 0.12615, and
the correspondence conflict remained. The 77.53 s GPU run partly overlapped D2 staging, so it
is not used for a controlled speed comparison.

All candidates were re-evaluated on 23,808 samples with zero overlap with the optimization
pixels. This is spatial resampling within the same 31 raw images, not validation by
independent frames or by ground-truth IDs.

| Silhouette error on the same unused pixels | Original v1 | H1 | H11 |
|---|---:|---:|---:|
| 100 steps | 0.411813 | 0.410218 | 0.406716 |
| 400 steps | 0.411813 | 0.404083 | 0.403065 |

H1's image-error improvement was about 0.39% at 100 steps and about 1.88% at 400 steps. In
both, H11's image error is smaller. Simply increasing the number of iterations did not settle,
from silhouettes alone, which correspondence is true. Saved to
`work/three_body/temporal_v2_disjoint_pixels_v1/` and
`work/three_body/temporal_v2_400_steps_disjoint_pixels/`.

### Diagnostic of the engineering tank lower bound

At 666 fr there were mouth nodes with observation support pinned at z=-6 mm: 201 points in D2
and 99 in the top condition. Intersecting the same adopted rays without the constraint placed
200/201 points (D2) and 99/99 points (top) below that lower bound. D2 median ray RMS: free
solution 0.387 mm, solution with x/y optimized on the z=-6 plane 0.907 mm, saved pose 0.918
mm. Ray intersection angles were sufficient; this was not an ill-conditioned case of nearly
parallel rays. However, the world origin derives from the training cube pose, and the
calibration manifest does not guarantee absolute surveyed coordinates either. This is not
taken as the true floor position or ID, and the frozen 1-minute configuration is not changed.
All 300 points, hashes and the residual audit were saved to
`work/three_body/floor_bound_audit_v1/`.

## v3 — Additional starting trajectories built as paths from 3D mask candidates

Using the existing refractive calibration, 884,736 points at 2 mm spacing were projected into
the 3 views and checked for independent-mask support. Empty detections and missing views were
neutral, and 2-view support refuted by a known third view was distinguished from strict3. At
333544, strict3 candidates were found at positions ≥17.55 mm from the optimized S2. On the
other hand, narrowing to the top 48 points had also dropped the 2-view support remaining near
S2. 6 independent-audit tests and a recomputation over the real 31 frames confirmed the limits
of the finite candidates, the grid and healthy suppression. Geometric false candidates can
arise even under strict3, so projection support is not taken as ground truth for individuals
or landmarks.

`V3_PROTOCOL.md` fixed the procedure: keep candidates spatially dispersed and build 6 starting
trajectories with a DP that includes a velocity term. In this first test, which translates the
missing body, bodies going outside the bounds are rejected rather than clipped. All
candidates, rejections and DP information were saved, and the original v2's 12 candidates
were also kept. The 6 candidates were additionally optimized with the same 400 steps, the same
real pixels and the same objective. The maximum difference on recomputing the objective for
the old 12 candidates was 0.

| Same end correspondence as original, 0,1,2 | v2 scene | v3 mask-grid |
|---|---:|---:|
| Cost | 3.64609 | 3.82222 |
| S2 pectoral fin with direct support / 31 | 21 | 16 |
| S2 max latent displacement / frame | 2.601 mm | 5.659 mm |
| Silhouette error on the same optimization pixels | 0.39150 | 0.40975 |

The additional starting trajectories lowered the local optimum for 1 of the 6
correspondences but did not change the overall best or runner-up. This **translation-only
mask-grid initialization is not adopted**. Support for points in the image must be
distinguished from the consistency of a whole body that has orientation and bending. This
result does not reject all mask-derived methods. GPU 65.47 s; 16 independent additional
input/boundary tests passed. Outputs are in `work/three_body/temporal_v3_mask_starts/`. The
original accepted coordinates and ID holds were not changed.

It was also re-evaluated on the saved 23,808 unused pixels. All 6 v3 proposals were worse
than the original v1's silhouette error of 0.411813: best 0.423259, same-correspondence H0
0.423665. None beats v2 400-step H1's 0.404083 either. Pixels, targets and weights match the
previous evaluation exactly and were not re-extracted. All candidates were saved to
`work/three_body/temporal_v3_disjoint_pixels_v1/results.json`.

In a comparison at 3 raw-image time points, the problem remained that grid-initialized
proposals also use another fish's region and move to poses that are not consistent as a whole
body. At 333550 the top view has a bent fish-like image, but the D2 union does not pick up
that region. Treating masks as strong ground truth also has limits. Comparisons on the same
raw images and projected values were saved to `work/three_body/v3_visual_audit/VISUAL_QA.md`.
These observations are not human ground truth for individual ID.

### Diagnostic of bridging a missing span with backward image motion

From the internal pose at 333558, on the future side, the pose was propagated 30 steps through
the actual raw images in descending order down to 333528. No intermediate external poses or
SLEAP detections were supplied; LK forward-backward error and multi-view 3D consistency were
checked. Pectoral-fin flow became valid 7 times even within S2's 14-frame missing span, but at
the left boundary the difference from the original pose grew to 62.83 mm. Locally valid flow
alone could not prevent a transfer onto a different image feature. A non-operational
diagnostic that resets the starting side of each pair to the original forward pose was also
saved separately and kept distinct from self-propagation. This reverse-flow-only proposal is
not adopted either. SHAs of the 93 raw images, per-feature forward-backward checks and all 3D
proposals are in `work/three_body/reverse_motion_v1/`. Committed IDs, accepted coordinates and
the frozen 1-minute method were not changed.

### Computation prototypes for future long-interval runs

Batched CPU comparison of ID arrangements across candidates kept the original 6-permutation
costs and decisions bit-for-bit, and cut a 6×18 comparison from 7.48 to 0.29 ms. The CUDA
Graph union image term, including public-input validation and copies, was confirmed to have
zero difference in values, loss and all pose gradients on 2 real frames at B1/6/18. 23 tests
passed. Reference medians for render + gradient: 10.39→2.34 / 11.82→3.66 / 11.34→7.36 ms.
These were measured in parallel with D2 staging and are not a speed-up rate for tracking as a
whole. They are not built into the frozen 1-minute tracking; adoption will be decided after
verifying that all tracking outputs match.

Afterwards a separate class that runs the whole objective in a CUDA Graph was implemented. The
original Adam, best-iteration selection and final visibility check were kept; on 2 real image
frames, values, 5 components, pose gradients and outputs after 14 steps at B1/6/18 matched
exactly. Further, 333519:333719 was tracked with a single initialization, and all 19 arrays of
the original v1 matched exactly. The SHA of the saved NPZ itself also matched. Tracking took
72.576 s, but its concurrent-load conditions differ from the original record of 97.335 s, so
it is not treated as a controlled speed comparison. LRU rebuilds: 194 hits, 6 captures, 2
evictions, 0 re-captures. Peak torch allocated 396.1 MiB. Passed 29 tests and an independent
static review; results and seal saved to
`work/three_body/graph_accelerated_transient_200_v1/`. This is a computational speed-up, not
an improvement in ID quality.

### Long-duration input storage

A separate reader that reads the source MP4 via a local hardlink was added; all 2598 images,
masks and SLEAP inputs for the 200+666 frames matched the old FFV1 stage exactly. Video
duplication is 0 bytes; masks and backgrounds alone take about 11 MB. Passed 16 tests and an
independent static review. Future consumers are required not to treat a successful reader
load alone as completion, but also to confirm completion of the creation provenance, the
manifest and the audit SHAs. A mask-only CLI that runs new D2 inference was also created as a
separate file and passed 12 CPU tests.

For the candidate future 30-minute window 49800:348600, only the first frame was audited. All
9 initial nodes of the 3 fish had direct support in ≥2 views and lay within the current
engineering coordinate range. However, the original fixed 9-frame background rule leaves only
2 frames in this long window, so as is the CLI fails explicitly. The proposal to use 9 frames
at 5000-frame steps before the start must be registered and verified as a separate background
rule, and has not been adopted yet.
`work/three_body/long_window_first_frame_audit/result.json` is an audit of the start frame and
does not mean a 30-minute tracking run or its completion. The limitation of an optimistic
initial pose derived from C34, which processed the whole recording, also remains.

## Continuous 1-minute technical verification started

`MINUTE_PROTOCOL.md` fixed D2 + flow, 9960 native frames and 0 external reinitializations.
Independent detections with batch 1 are being produced in the existing D2 environment. A local
continuation process, `work/three_body/continue_minute.py`, which launches the registered
command exactly once after input completion and SHA confirmation, was started. Progress is in
`work/three_body/minute_control_v1/status.json`, and details after tracking starts go to
`tracking.log` in the same directory. There is no automatic retry on failure and no mid-run
splicing. This is a record of a run in progress, not a statement that the new method's
1-minute verification is complete.

### Parallel verification as of 04:12 JST

Independent D2 detection for the 1 minute completed on all 29,880 images (4711.946 s).
Tracking with fixed v1 was started once and is continuing past half of the 9960 native frames.
The mid-run online log shows all three fish in ID hold. Coordinates, ID acceptance rate and
scene count from the final backtrace are not yet determined; a separate watcher that runs the
common comparison exactly once after completion is waiting. The common evaluation of the
existing 9 arms can be recomputed independently and passed 30 tests. The new 1-minute result
has not been compared yet.

For long-duration inputs, the 30 minutes 49800:348600 and a background from 9 pre-start frames
were fixed as a separate rule in `long_input_protocol.json`. 27 raw images and 3 median
backgrounds were checked on CPU, and no conspicuous fish-shaped afterimages were found. Pixels
differ from the old background; this implies neither identical input nor improved ID
performance. A stager that detects independently in 10-second pieces and can resume is being
implemented. `chunk_masks.py` checks and joins split arrays; 39 new + 7 existing tests and a
split→join of real 200×3 images confirmed an exact match of all arrays. The 30-minute
inference and tracking have not started yet.

`review_scene.py`, which builds partial reviews from raw images only, was verified on real
inputs of 5 frames × 3 views. Decoded SHAs of all 15 images, the native start position and the
video frame count matched. The range check that prevents mistakenly loading another scene's
review JSON passed 23 pure-JavaScript tests. Browser click and save behaviour remains
unverified because of tool constraints. This verifies the build function on an arbitrary short
input; it is not a problem scene chosen from the 1-minute result, nor human ID ground truth.

Measuring the CPU flow projection on 20 real pairs, projection was 18.04% of the
proxy-measured time. 160 calls comparing OpenCV thread counts 1/2/4/32, with the order
reversed, matched exactly on all 15 output arrays; the median was 132.71 ms at 4 threads
against 139.15 ms at 32. The difference is a few ms; the frozen 1-minute run and the next
method's settings were not changed. The inconsistent overall cProfile ratios are not used.

The discriminability of colour features extracted from inside the bodies of independent D2
instances is also being diagnosed on CPU. Templates per view/slot are built from the initial
166 frames, and 50 time points, every 10 frames within the following 500 frames, are
examined. This is an appearance diagnostic that uses saved slots as provisional labels; it is
not full-frame tracking, independent ID accuracy, or adoption of validated appearance.

This appearance diagnostic completed. Of 150 subsequent individual-time points in the top
view, features were obtained for 69, and for 58 of them the nearest neighbour was the same
provisional slot. Of the 11 provisional-slot mismatches, in 10 the colour components excluding
lightness also supported the other slot. side1 had no training material for one slot, so a
3-class comparison was not possible; in side2 only 8 of 15 were the same provisional slot. All
requested counts and unscored cases are kept in the denominator. 648 raw-image SHAs were
verified; it passed 7 tests and an independent denominator check. Colour differences in the
top view could be measured, but they are confounded with pose, core area, reflection and wrong
masks, and are not interpreted as ID accuracy. These fixed templates are not adopted for
releasing holds. Details are saved in `work/three_body/appearance_core_v1_interval/RESULT.md`
and `SAVED_ARRAY_ANALYSIS.md`.

`review_plan.py` selects candidate check time points before and after each risk scene at which
the mouths of all three fish are directly supported. It does not extend candidates outside the
fixed context or across missing native frames, and keeps both insufficient probes and
incomplete scenes. Passed 12 tests and 3 independent tests. On the existing transient it
proposed 333534/333550/333632, while keeping S2's hold and the human-unconfirmed state. The
interval has 0 risk scenes, which does not rule out the 8 scenes of direct pectoral-fin
missing or undetected swaps. This proposes material for checking; it is not human labels or
correction results.

## 05:22 JST: Continuous 1 minute completed, and the next iteration to improve mouth-observation recovery

Fixed v1 D2 + flow ran all 9960 native frames of 76620:86580 with a single initialization and
0 external resets (tracking 4848.422 s, plus D2 preparation 4711.946 s separately). The common
comparison and independent audit also completed. Pectoral-fin direct support is
29312/29880=98.099%; acceptance with ID is 8954/29880=29.967%. It does not reach the
acceptance criterion of 90% overall / 80% per fish. It rose from the old generative-model
method's acceptance of 3.561%, but the female was S1 at annotation 78799 and S0 at 79547. Of
the 13 raw adjacent intervals, 12 kept / 1 changed; accepted: 3 kept / 0 changed / 10
unscored. The 1 change is an ID mismatch between endpoints and does not establish the actual
number or timing of swaps within the interval. In the old raw result all 13 female intervals
keep S1.

The 6 scenes that merge 8 body-missing cases and 4 finite-margin risks (1103 context frames in
total) cannot by themselves cover the ID risk. Positional correspondence with the old
generative model changed between 79108 and 79109, and at 79109 all three fish had review
false. With C34 lambda2 there is disagreement from even earlier, 78332, so the comparison
references do not agree with each other either. This is not scored as male ground truth or as
a true swap.

A raw-frame check page for scene0,
`work/three_body/minute_reviews/scene0_79236-79488/index.html`, was created, and human input
was requested to check the correspondence of the three fish at 79316/79405/79480. Images,
video, native frames, SHAs and links have been checked; the human JSON has not been received.
A separate model-diagnostic image confirmed a 16.3704 mm latent relocation of S0 at 79372,
proximity of hypotheses, and D2 missing fish bodies and detecting reflection regions. The
relocation is not interpreted as swimming speed or as recovery of the same individual.

In scene1's original context 80261:80432 there is no time at which the mouths of all three
fish are supported simultaneously. S1 kept pectoral-fin support over 79965:80370 (405 frames)
while having lost the mouth. This overlaps S2's mouth missing span 80340:80463 by 30 frames.
Because the original tracker creates large observation-recovery candidates only when the body
is missing, mouth failures while the pectoral fin is still tracked remain. A new pure helper,
`node_recovery.py`, builds candidates for a directly supported pectoral fin from unowned mouth
rays in 2/3 views. It keeps the original box, RMS, body length and the existing strict support
criteria. 80261 S1 and 80431 S2 have geometric candidates; 80344/80370 S2, without
pectoral-fin support, are rejected. However, 80344 S1 may pick up another fish's unowned
mouth, so candidates alone cannot be taken as ground truth. Next come a competing fit that
keeps all the original candidates, and a comparison by continuous tracking under identical
conditions.

The independent top-view appearance diagnostic also dropped its dependence on SLEAP. 6
features could be computed from all 30 D2 regions at the same fixed 12 time points, and the
features matched the earlier 7 conditional cases exactly. At 79108 features close to the
initial T0 remain, but there is also a Y-shaped region containing images of multiple fish, so
it is not assigned to a single ID. All regions and limitations are recorded in
`appearance_minute_all_top_instances/RESULT.md`.

The stager that saves long-duration independent detections partway and reuses them was
verified on 6 real frames in 3 chunks: 18 images, 55 detections and 13 packed arrays matched
exactly. The main 30-minute run has not started. The main stored structures are estimated at
about 6–7 GiB, but tracking alone extrapolates to about 30–40 hours. This is not a measured
30-minute processing time. Fixes will be compared before a main run with the failed 1-minute
method as is.

### Version adding mouth candidates immediately: not adopted after continuous 200 frames

`node_recovery_tracker.py` and a separate CLI were implemented, and passed 42 helper/tracker
tests, 14 runner tests, Ruff and an independent source review. For 24 candidates, the original
fit and the Graph fit matched byte-for-byte on all 4 returned arrays on real inputs. In the
local fit at 4 times, mouth support and loss improved at 80261/80431, but at 80344 a candidate
suspected of linking parts of a different fish also won. Local-loss improvement alone does not
justify adoption.

The new arm tracked all 200 frames of 333519:333719 with a single initialization. All input
SHAs, the initial pose and the original config were checked against the earlier v1, and all
19 arrays and the completion seal were saved (65.310 s; the speed ratio against the old 97.335
s, under different concurrent load, is not a controlled comparison). Direct pectoral-fin
support 586→565/600, accepted 417→417. S2's direct pectoral-fin missing grew from 1 interval
of 14 frames to 2 intervals of 15+20 frames. Mouth candidates were selected in 7 frames, and
just before the first, 333535, all mouths were supported. A version that adds candidates even
on a momentary prediction miss is not adopted as an improvement. This degradation is in
positional availability; without independent male ID ground truth it does not establish an
increase in ID errors.

Next, a separate state extension restricted to the case "on the same beam parent, direct mouth
missing continues for ≥4 frames and direct pectoral-fin support for ≥4 frames" is being
prepared. The original source and the immediate version's results are kept, and node history
propagates as explicit state of each beam parent. In addition, a diagnostic is under way that
compares the 6 ID correspondences from pixel colour and the refractive renderer's visible
contribution, independent of SLEAP and of instance separation. Unadopted and unverified
results are not counted as completion of the 1 minute or the 30 minutes.

### 06:12 JST: Persistent-mouth-missing version passes short intervals; 1-minute re-verification under way

A separate tracker was implemented in which PersistentNodeState holds the run history of node
missing and of direct pectoral-fin support, updated along the chosen parent. The 4-frame
persistence condition was fixed in advance. At 200 frames all 19 original arrays matched
(direct 586, accepted 417/600, 64.469 s); at 666 frames direct and accepted were 1948/1998
(original 1942, 208.140 s), and the female kept S1 at the 2 annotations. This is not proof of
ID improvement for all 3 individuals by human evaluation. After passing the short-interval
gate, the continuous 1-minute persistent_mouth_continuous_one_minute_v1 was started: session
66787, 4125/9960 as of 06:06. The one-shot comparison watcher after the completion seal is
session 45190. The new evaluate_variant_minute.py correctly verifies the new runner's input
path-roles and uses the original metrics. 60 new+old tests and an audit that the real 200
inputs are identical are done. The original evaluator and the running source were not changed.

Pixel-colour rankings at the fixed 12 time points agree 12/12 on the lowest correspondence
even with a robust mixture model. However, this is no ID evidence for fish that are not
visible. The per-mask diagnostic showed double explanation and an unexplained separate region
coexisting at 79372. Whole-body Gaussian proximity does not reject the suspicious mouth
candidate at 80344, so that penalty proposal is not adopted as a solution. A ranking-update
version adding appearance was implemented separately; it is registered and fixed from the
initial 166 online states only. A final backtrace changing past branches is distinguished from
future information leaking into the registration. The pose/source/support/review at
registration time are saved separately and audited. Real video/GPU comes after the current
1-minute comparison.

### 06:41 JST: Persistent-mouth-missing version completes 1 minute; appearance version on to 200 frames

9960 frames were completed with 1 initialization and 0 external resets, and inputs, source,
candidates and summary were verified. Tracking 3109.437 s. The shared 1-minute comparison also
completed. Direct pectoral fin 29388/29880 (original 29312), accepted 8025 (original 8954),
acceptance rate 26.86%. Per individual 1670/2704/3651; first review at 78331/79371/80347. The
female's 14 raw annotations were all S1, giving endpoint agreement on 13 intervals, but
accepted: 3 kept / 10 unscored. The first S0 hold came 991 frames earlier. Reduced
availability is not confused with ID accuracy, and without independent ground truth for all 3
individuals this is not counted as a pass. The original C34 and the earlier generative model
were also compared on the same metrics.

The version with appearance passed 101 related CPU tests, Ruff and independent review. It
records the online state, features, templates and view support at registration, and a
separate audit that recomputes the 166 frames from raw images and SL was prepared. The new
evaluator and audit do not change the old 1-minute evaluator. Following the fixed
200→666→1-minute screening, appearance_transient_200_v1 (session 23734) was started. It is
currently running and its effect is undetermined.

### 07:11 JST: Version choosing pose candidates by appearance completes short intervals; on to ID optimization over fixed geometry

Appearance 200 passed in 69.784 s with direct 586 / accepted 417. For the first 166 frames the
21 original arrays matched; after that poses change but support and review did not. 666:
223.555 s, direct and accepted 1946 (persistent-mouth version 1948), review 0, female S1 at 2
annotations. The initial 166 online registrations of both intervals were recomputed from raw
images and SL, and all 8 arrays matched byte-for-byte. It fell 2 below the fixed gate, so the
appearance version's 1 minute is not started.

At 77068 S1, a 1.191 mm difference in pectoral-fin position moved the side1 ray distance from
0.939 to 2.127 mm, over the 2 mm threshold. At 77108 S2 the previous 2 rays also fit the new
pose and are unused, but are not in the chosen correspondence table. Pre-filter candidates
were not saved, so the internal stage at which the correspondence vanished cannot be
identified. At both time points the silhouette and colour losses fell and the node and shape
terms worsened. A drop of 2 in support alone does not establish ID degradation, but the next
experiment, separating the effect of changing positions from the effect on ID
correspondence, is being prepared.

IDENTITY_DP_PROTOCOL.md was recorded before running on real data. It uses the completed
persistent-mouth version's geometry and the independently audited initial-166 top-view
templates. Only the initial state is fixed; the minimum path over the whole time span is found
from the 6 permutations per native frame and adjacent direct pectoral-fin motion. Colour is
scored only after registration. The 0 cost for links across missing data has an avoidance
bias, and the method cannot fix geometry duplication or shortfall either. Original review and
NaNs are retained, and proposed IDs and the difference from competing paths are separate
outputs. Passed 33 DP / 16 colour CPU tests; not yet run on real data. A new runner, strict
input verification and independent evaluation are in preparation. Human ground truth for all
3 individuals has not been received.

### 07:41 JST: Adjacent 6-state DP on 200 frames confirms missing-cost avoidance

identity_dp_transient_200_v1: 5.93999 s, all 200 native, 1 initial ID condition, 0 new
geometry fits. 17408 top-view pixels were scored over 34 frames after registration. SHAs of all
outputs, inputs and code were checked; re-scored on CPU from 600 raw frames, the max colour
difference over the 6 candidates was 6.67e-16 and visible contribution 1.71e-13, within the
specified precision. The formal evaluation and the no-colour comparison also completed.
Original support 586 / conservative accepted 417 were preserved, but with only a single female
endpoint, interval retention is unscored. The CPU image audit is not proof of correct ID.

Only at 6 single frames, 333538/540/542/544/546/548, does it become P210, returning to 012 the
next frame. All are during registration, with colour unary 0. While S2's pectoral fin is
missing, it assigns the healthy S0 to the link with the missing one, eliminating S0's motion
term and reducing the scored fish from 2 to 1. Selected 89.623752 against ID-fixed 96.076985.
A positive best-competitor gap does not prevent this flaw of the objective. This arm is not
extended to 666 / 1 minute.

The next protocol, IDENTITY_MEMORY_PROTOCOL.md, was registered before running on real data.
The last direct-observation event is held in each ID candidate's state, and on re-observation
a Brownian-type displacement/time and log time are evaluated. When all 3 fish are supported,
history and cumulative cost are kept while merging equivalently into 6 states. Exceeding the
limits of 100k per layer / 1M states total / 6M edges is flagged explicitly as not completed,
with no approximate pruning. State counts are measured from real support only, and the
mathematical implementation is verified by brute force. Positions and correspondence of IDs
not re-observed by the end are not committed, and original review and NaNs are preserved.

### 08:26 JST: Continuous 1 minute completed for ID optimization that retains observation history

The new variable-state DP was verified on small-scale exhaustive path enumeration, a huge
common offset, different histories with the same correspondence, and re-observation after
missing; there is no pruning within the limits. It kept P012 at 200/666; all 1998 images at
666 were recomputed on CPU and matched GPU scoring to the specified precision. Continuous 1
minute: 449.836 s, 9960 frames, 403381 states, 2420250 edges, max layer 20700 states. No
manual ID fixing or 3D reinitialization midway. The existing 3D shape computation (3109.437 s)
and D2 preparation (4711.946 s) are not included in this ID-stage time.

The formal comparison, a joint comparison with the existing 10 methods and a SHA audit of all
outputs completed. Direct support 29388, accepted 8025/29880 and original review 21691 are
retained. The female's raw endpoints agree on 13 intervals, but after acceptance 3 agree / 10
unscored. The MAP using colour is P021 only for the 2 frames 82719:82721, where all 3 nodes of
both fish concerned are unobserved. Direct pectoral-fin coordinates and accepted coordinates
match the original shape byte-for-byte. Without colour, it becomes P102 for 21 frames around
79372; the two results differ in 23 frames. Correspondence changes are not counted as
biological swaps.

Next: re-measure per-D2-instance explanation rates for the current shape, and diagnose whether
the difference from no-colour changes direct observations. The duplicate-body finding from old
v1 is not carried over to the current version. 5 priority check scenes and alternative paths
were saved, but this is not a guarantee of covering all risks or of completion with a few
corrections. Human ground truth for all 3 individuals has not been received.

### 08:55 JST: Current body duplication re-confirmed; controls for candidate position/orientation and curvature registered

The memory motion-only path moves 80 nodes with real support to other IDs over 21 frames
(mouth 21 / chest 29 / tail 30). Everything affected is in the original review state, and
accepted values do not increase. In the 2 frames of the colour path, all affected nodes are
missing. Relative to fixed P012, motion-only is motion -.2021 / colour +5.6014, and with colour
it is motion 0 / colour -.0244. Image information contributes within the model, but ground
truth for biological ID is not confirmed.

13 time points / 109 instances of the current persistent were re-measured from the raw inputs,
reproducing the original sample loss/bridge/direct. At 79372 S0/S1 doubly explain topm2,
while 93.6% of topm0 and 99.4% of side1m2 were unexplained. The raw frame also contains a
strongly bent body. This was confirmed on the current version, not carried over from old v1.

The new BODY_PROPOSAL_PROTOCOL.md was frozen before real fits. A generates rigid
position/orientation candidates for each raw body from 2 mm volume components derived from top
D2, compared with the same nominal number of random starts under the original objective. B
fixes 3 nodes/width/sigma and varies only tail curvature. Both use the same fit pixels and a
fixed spatial evaluation that excludes all fit pixels. Limits recorded: top non-detection,
grid dropout for thin bodies, ghost volume, a single tail curve. Control points are not
counted as observed nodes, review is not released, and ID improvement is verified separately.

Passed: rigid placement 17 CPU, volume extraction 29 CPU, curve renderer 24 new + 14 old CPU,
local fit 9 CPU + 1 CPU/GPU. Independent review found a uint64 frame wrap, which was fixed.
Scores match with the real tracker's nested objective as oracle, and curve 0 matches the old
renderer exactly. The real 13-frame comparison has not started yet.

### 09:23 JST: A/B real-image controls completed; isolating the cause in the correspondence model

A: 1384 fits in total at the same 13 time points, pure fit 19.1168 s, sealed run 91.06 s.
Nominal candidate counts are equal, but admitted were mask 920 / random 477 (each arm
including the 13 shared baselines), so compute was not equal. Mask candidates improved the
score at 2/13, random at 0/13. At 79322 it increased explanation of the same region as another
body; at 82719 it gained mouth support at the cost of a worse spatial evaluation. At 79372 the
refit original scene was selected. All SHAs, candidate correspondences, support and evaluation
arithmetic were independently audited, and A/B fit and evaluation pixels matched exactly at
all 13.

B is a curvature-only control with nodes and Gaussian scales fixed. Mean spatial evaluation
error fell 8.371%, improving on average at 13/13 time points but worsening at 8/39 view-time
points. The unexplained U shape and double explanation at 79372 remain. At 82720 the offset
hit its upper limit, with arc/chord 1.479. Arc-length increase is allowed, so this is not
taken as an anatomical improvement. SHAs of 9 raw images and CPU/GPU alpha re-render agreement
were confirmed, and root also visually inspected the raw-frame comparison at 79372.

In the scene diagnostic over all 385 original/post-fit at 79372, candidates covering the
unexplained region did exist. trial6 reduces top/side1 unexplained to 71.32/70.05% but loses
on node term +1.0931 and weighted chest motion +.6356. The current independent view×node
assignment allows splices such as S0 side1 chest row2 / tail row1. Next: diagnose
correspondence consistency within an SL instance, and how independent top contrast responds
to D2 misses/reflections. No change that picks weights to make a candidate win, and no release
into acceptance without human ground truth, has been made.

### 09:40 JST: Detection-row coherence and top background subtraction also fall short of standalone adoption

instance_assignment.py was added. In each view it exclusively assigns 3D body → SLEAP instance
row, and bad nodes within that row can choose missing. The comparison with per-node
assignment uses the same forward half-ray, float32 cache, robust error, weight [1,1,.35],
missing 9 and normalization. Passed 12 CPU tests, an independent Torch check on 60 examples
(max 1.78e-15) and Ruff. All 1384 original A shapes at all 13 time points were fixed and
re-scored; on all candidates this confirmed a max node-term difference of 2.67e-15 for the
original fixed plan, coherent >= nodewise, and preservation of the original non-node terms.
Pure CPU scoring 6.5023 s, no additional fits. All 13 read roles/paths and all 23 output SHAs
were also confirmed.

Mask-arm mean spatial error: original .43992958, fresh nodewise .44081149, coherent
.44584030. Total directly supported nodes 76/77/72 (117 possible each). Coherent changed the
shape candidate at 4 time points relative to fresh, but the new candidate at 79372 moved S0
near S2, and the main unexplained regions worsened to 98.78% (top) and 99.96% (side).
Forbidding splices alone does not yield a true image explanation of the three fish. Old
review/coordinates unchanged.

An independent 13-time-point diagnostic with the existing top foreground_contrast fixed also
picks up bent images that D2 misses, but connects them to reflections and other fish. At
79381, 98.98% of the reflection-like m3 is foreground, and core is 100%. The current
calibration's ray-box intersection and the finite water-surface entry test agree on all 1.44
million pixels, and the reflection remains under either. Because the camera's horizontal
position lies within the face and refraction preserves the horizontal sign, re-entry from an
external entry does not occur in this case. raw/D2/contrast figures, all components, raw-frame
SHAs and the 8-frame background median were recorded; there is no threshold change and no
introduction into tracking.

No new method has been adopted on automatic agreement metrics alone. Input of the anonymous
correspondence of the three fish on raw frames is needed, and the existing first check page is
waiting. The point at which this is used for correction is kept separate from independent
evaluation. Zero swaps for all 3 individuals and 30-minute completion are not claimed; the
current 1-minute result and all failure cases are retained.

### 10:16 JST: Implementation that builds a separate output from human endpoint correspondences, and a zero-review control

review_correction.py and run_review_correction.py were added. The actual human JSON for all 3
individuals has not been received. Using the existing extractor and risk ledger, only
confirmed endpoints connected to the initial anonymous G0/G1/G2 are applied to conditional
intervals. Unannotated parts between the two ends are left uncommitted even when in the same
raw slot, and beyond the right end the application runs only up to the next risk/source
boundary. Remaining IDs are not determined by elimination from partial checks; contradictions
and dependency cuts are handled independently of the order of all links. Nodes without
observations keep NaN in the measurement arrays.

Passed 30 core and 17 CLI new CPU tests, Ruff and independent code review. Froze core SHA
aecf5af0c86bdaad5faf4f55eaea2956f758389ca59b10bd923fc5d041bd229f and CLI SHA
f5da6b5e3a23dadbffe7efeab71c0369ec9f2b9c4e8df8d30ae96e18abd3fddb.
DENSE_REVIEW_CORRECTION_PROTOCOL.md was made an implementation addendum that does not change
the old protocol.

It was applied to the actual persistent 1 minute with 0 reviews: 49.197 s, 9960 frames,
exit 0. The original NPZ was copied byte-exact; new observations / refits / new permutations /
new acceptances are 0. Initial claims stop at 78328 for G0 and at 79371 for G1/G2. Because the
risk ledger holds all 3 slots when the pair involved in a candidate conflict is unknown,
acceptance falls by 971, from the original 8025 to 7054. The entire reduction is S2's direct
chest coordinates in 79371..80341; the original geometry and the original automatic output
were retained. This is a difference in hold policy, not an improvement or an actual swap
count. A separately assigned audit of the real output is in progress.

Direct support in >=2 views of all 9 mouths was confirmed at scene0's 3 anchors. However, G0
is already on hold before the first anchor, 79316, so this scene alone cannot necessarily
connect to all initial IDs. The range of the existing raw check and the minimum additional
entry evidence are being investigated. 30-minute tracking and real human correction have not
started yet.

The independent real-output audit v2 completed. Without re-running the core itself, it
reconstructed the 12 expected arrays from the initial tracklets: all dtype/shape/bytes
matched, and SHAs of 84 inputs, 37 outputs and 32 saved sources were confirmed. The record of
the first audit's failure, in which the assumed source dtype int32 differed from the frozen
implementation's int64, was kept. The implementation and the real results are unchanged. It
also confirmed that all 971 additional holds are G2 in 79371..80341, and the ledger rule that
applies the boundary to all 3 IDs because the pair involved in the candidate conflict is
unknown. root checked the independent report and the success seal.

G0 has not only 78328 but also a risk at 79178:79186. scene0's entry 79316 has mapping
[-1,1,2], so this page alone cannot connect to all initial IDs. A raw-frame review of an
additional 1023 frames including 78325/79177/79316 is being built from the same minute
manifest. It is a bridge for checking continuity over about 6.16 s; support in >=2 views of
all 9 mouths was confirmed. Old v1's entry state is not reused. Even if real correspondences
are obtained from the additional page, the open intervals stay blank, which may increase
missing compared with short local checks. Its usage rate and human burden are recorded
separately. No guarantee is made that the full 1 minute or 30 minutes can be completed with a
few checks.

10:27 JST: the additional raw-frame pack and QA completed. producer 29146 / QA 23062 both
exit 0; the records of 3069 raw images and 3089 output SHAs were verified, and the 3 PNGs for
the shared 79316 match the old scene0 exactly in pixels and file. root visually inspected the
top raw frames at the 3 time points and also checked the final QA report and seal SHA. Only the
first QA exited before output, because of the Windows code-key separator notation; it
completed after the QA side's notation handling was aligned, and the raw-frame pack was not
changed. The procedure for the 2 pages and the saved JSON was recorded in
minute_reviews/REVIEW_INSTRUCTIONS.md. Actual UI click/save operations are not verified. All
runs finished; human JSON not received; the goal for all 3 IDs not met.

### 10:57 JST: State save/restore of all candidates and a separate-process comparison on the real 200 frames

With the human JSON not received, ID correction cannot actually be applied, so work proceeded
on the execution gaps toward the 30 minutes. In continuous_tracker.py, 101 lines of the
original persistent's numerical processing were separated into an advance in the same order.
It saves all beam/history/parent/velocity/body-review/node ages/previous raw FrameObservation,
and restores them strictly onto the original config/rig/initial in a new process. The
checkpoint codec is NPZ/JSON only, with no pickle; it verifies array dtype/shape/bytes and
caller inputs/source binding, and writes the success seal last. The resume reader keeps the
original manifest and distinguishes the FFV1 relative offset from the source MP4 absolute
frame.

Passed 135 independent CPU cases (core 41 / codec 57 / reader 37), Ruff and independent code
review. With the real reader, 27 view pairs (54 decodes) matched exactly on raw/mask/SL between
FFV1 and MP4. All 13 mask arrays for the 200 and the backgrounds also matched. The original
source is unchanged. The TorchVersion→str conversion for the type-strict codec was fixed in
review before the real GPU run.

The real 200 on the RTX5070Ti: single process 62.332 s, and the 3-process 85→121→200 was also
byte-exact with the old persistent on all 21 arrays and the recovery counters. logical
initialization 1, restore 2, external pose reset 0. The final checkpoint holds all 1194
history rows and 6 active states, payload 10928998 bytes; save + reload on the single side
took about 1.946 s. The original candidate coordinates were opened only after the new full
output was saved. These 200 had mouth-recovery 0, so this is not taken as having confirmed a
real recovery boundary for the new mechanism.

It was confirmed that the same existing 666 has 2 mouth_recovery events (76891/77000), and an
additional plan was pre-registered as a separate artifact. Without changing the old
source/runner, a single 666 run and a 3-process 271→380→666 run save and restore just before
each recovery. Plan SHA 684fcb2d3b9c5f94ebfa761d4ef770172f89dd7891491a74f846d56dbc48a5f7. The
single 666 is in progress in root session 11060. An independent audit of the 200 results and a
registration audit of the 49800 initial values run in parallel. These show execution
equivalence, not evidence of ID accuracy for the three fish, a full 30-minute run, or zero
swaps.

### 11:30 JST: Equivalence across the 666 recovery boundaries, and the first 10 s of long-duration input

The single run of 666 and the separate-process run 271→380→666 completed. The original 21
arrays, all 39 saved arrays and 3984 history rows were byte-exact, and the mouth recoveries at
76891/77000 were preserved across the restore just before each. Single 204.342 s; split
tracking total 212.411 s. All source/inputs/saves/outputs were independently audited, and the
raw frames, masks, SL and background at the preceding frames also matched. logical init 1 /
restores 2 / external reset 0. root's full GPU comparison finished.

The initial pose at 49800 was newly registered, and direct support of all 9 points and the
original inputs/outputs were independently audited. The initial correspondence is an
optimistic initial value from C34; mouth z=-.413303 mm is within the engineering bound, and
consistency with the measured floor is not confirmed. The new loader reads only this
registered first-frame output and SL/calib, and does not open C34 trajectories or human GT.

Independent D2 generated the 1660 frames [49800,51460) according to long_input_protocol. root
28108 exited 1 intentionally at requested_max_new_chunks, recording chunk success and overall
incomplete separately. Generated interval 448.122 s, predictor total 275.877 s, masks 2846576
bytes, source video 15.115 GB via hardlink. Independent recomputation of all 4980 packed-mask
images, 9 raw-frame probes, 27 background samples and 3 median PNGs matched. This is only
1/180 of the full 30 minutes; detection counts are not taken as fish counts or ID accuracy.

long_tracking_inputs.py passed 26 CPU cases / Ruff / independent review / real 1660 load QA.
source SHA bea59f4e65235df8bce49bed0f1db2e92815f998b3bd7e018f456748f9e173bb. Fixed bindings
are separated from the growing consumed-chunk evidence, and on restore the past prefix is
verified strictly. The loader hashes each original raw 3 times (frozen context twice + tail
once) but avoids duplication per chunk alias.

run_long_continuation.py passed 9 new CPU integration + the existing 135 cases in the same
process; independent review: Go. source SHA
76b1ba9b06b90338bda27318665f6606ca00565b9809780e635558096303b61d. The new
long_tracking_protocol.json SHA
fda72ac8304be4e46032853a13f895ddb9bf25f2436096b65e27e2418a619c85 was sealed before the first
long fit. The numerical config is identical to the existing minute; the input's fixed 9-frame
background is registered separately. All beams are saved to the checkpoint before the
provisional backtrace of the prefix is output, allowing changes from later observations. first
1660 has been running in root 31801 since 11:24:59; at 11:29:58, 675 frames / 199.672 s of
tracking. Processing cost, ID accuracy and required human effort for the full 30 minutes are
undetermined. No raw review JSON saved by a human has been received.

11:48 JST final: first 1660 real tracking and independent audit completed. root 31801 / Hooke
41952 / Feynman 40874 all exit 0; no GPU jobs. Tracking 487.399 s, input preparation 80.542
s, reader 13.635 s, save 3.862 s. 9953 candidate histories / 6 active were held in a
13305884-byte checkpoint. logical init 1 / reset 0. Chest 4980/4980 directly observed and
conditionally accepted, review 0. The mouth-recovery age condition triggered 0 times.

Support audit v2: 43024 sources within ray 2 mm / reprojection 12 px, 14711/14940 nodes
directly supported. Max 1.995829363 mm / 9.593341316 px. The 229 missing points (S0 tail 2, S2
mouth 5 and tail 222) are not filled in as observed coordinates. Only the wrapper of the first
support audit, v1, interpreted output role names as absolute paths and failed without opening
the candidates, so the failed artifact/source were kept and v2, with the reference handling
fixed, passed.

The main audit sealed 193 inputs/sources and 31 outputs; all 39 checkpoint arrays, all 9953
parent histories and the independent backtrace of 21 arrays matched in bytes. A re-decode of
the last 3 native frames × 3 views and the last full set of 12 fields also matched.
Generalization to the full 30 minutes and biological ID accuracy are not established. The
100% chest acceptance, with a different background and on an initial interval, is not counted
as an improvement rate for ID retention over the existing 1 minute. An actual human review
JSON is needed for the next ID correction and evaluation.

### 16:25 JST: Parallel GPU execution and a pose comparison including reflection

The first 20 s of the new long interval completed with save/restore of all 6 candidates;
initialization 1 / restore 1 / reinitialization 0. The latter 1660 frames were added with
tracking 479.642 s. The independent audit confirmed support over all 3320 frames (85036
sources, 29555 nodes), 19913 parent histories and inheritance across the boundary, and an
exact match of the additional 4980 raw images and masks. Pectoral-fin direct support and
conditional acceptance at 9957/9960 do not mean a count of correct biological IDs.

Frozen tracking of 200 frames was run in parallel with D2 and matched the original standalone
run exactly on all 21 outputs / 39 saved states. D2 for all 1660 side1 images and the first
150 side2 images, including the overlapping span, was re-inferred standalone. 1810 images /
6079 detections matched exactly on all 13 packed fields, order, scores, masks, raw images and
background. On this limited comparison, a plan was registered to process 4 input chunks in
the same tracking process while running detection for the next 4 chunks on the GPU
concurrently. root 73621 tracking is currently at about 32 s; root 89160 detection is on the
next interval. A CPU logger records usage and per-stage times. Kernel overlap and equivalence
over the full 30 minutes are not verified.

The new light-path model for reflection uses nominal wall y192, outer face y196, 4 mm glass
and fixed refractive indices, and computes a finite underwater reflection / glass round-trip
path from the first intersection with a tank face. Agreement with an independent NumPy
computation, the optical conditions, Torch gradients and 14 Adam updates of the old direct
model was confirmed. The 2 modes at 3744 real pixels have a max response difference of
2.22e-16. The product of the Gaussian response and the Fresnel weight is not a photometric
image likelihood.

A first diagnostic over 13 fixed-pose scenes found direct response 0 on the 1855 pixels of
79381 top D2 row3; adding reflection gave a mean of 0.443537 (unit) / 0.226664 (Fresnel).
However, truncating the direct light path inside the tank collides with the old pose's z<0,
and overall image error worsened. The measured floor position has been asked of the user,
with no answer received. In a newly registered additional control that keeps the old infinite
direct light path, error went 0.445953→0.424787 (unit) and 0.432878 (Fresnel), but the
response on the background and side2 worsened. This is not a pose improvement or ID
verification. That D2 row3 contains 0 SLEAP points; of the 301-point mask-correspondence
diagnostic, the original selection of 218 points was also independently checked.

The next local pose optimization, 13 scenes × 3 methods, was registered. All methods share
the initial pose, an exclusive SLEAP correspondence computed only once, motion and length
priors, 14 updates, engineering bounds and fixed evaluation pixels. A 3×3 evaluation of no
reflection / same direct path + unit reflection / + Fresnel reflection, together with all 39
results and all 15 iterations, is saved. A reflection method is considered for short temporal
tracking only if its mean image error is at least 1e-10 lower than both the no-reflection
refit and that pose with reflection alone added, it loses no pectoral-fin support across the
39 scene-slots, and it does not lose mouth support at 82719/82720. The criteria are not
changed after the results. Passed 37 base CPU + 14 CLI tests / Ruff and independent code
review, and pre-registration 51077 succeeded. The real comparison 62015 was started on CPU.
There are no local-fit results yet, and real human correspondences for all 3 individuals have
not been received.

### 16:47 JST: Verdict on the local reflection fits and preparation of a temporal comparison

The 39 registered fits exited 0 in 77.058 s, seal
32d293451420857bfc7e415fc43a88ad6fa9591cf2b7fc8a84c69059a119d49c. The independent audit
checked 65 outputs, the fixed-pixel 3×3 / all-view / all-layer image errors, 39 final supports
and 585 iteration records, and re-confirmed that both methods pass the criteria. Mean error:
direct refit 0.445106, unit diagonal 0.418523, Fresnel diagonal 0.430849. Comparing only the
pose effect with the same renderer: unit 0.424360→0.418523, Fresnel 0.432096→0.430849. Under
the common direct scorer, unit worsened by 0.001104 and Fresnel improved by 0.0000768. side2
worsened for both. Observation support is exactly the same for all methods and time points:
mouth 27, chest 25, tail 23 of 39. There is no improvement in reacquisition.

Fresnel was chosen as the next candidate, and 200 frames [79236,79436) were fixed for a
same-condition comparison against direct. Only the old minute's 79236 is registered
separately as the initial pose, and velocity, missing age and review for both methods start
from the initial state. This local development test does not inherit the old 1-minute global
ID, and the start pose's dependence on future information is stated explicitly. There is no
reinitialization within the short interval; after all 21 arrays are saved, 2×2 and per-slot
support, acceptance and longest missing are compared on the same evaluation pixels.
Thresholds of the higher-level policy are not relaxed. Reflection is added to the fit cost
only, and the mask bridge is computed from the old direct image's visible_alpha/ownership.
Candidates, beam, flow, ages, sticky review and backtrace are inherited.

The new reflection_tracker.py passed 6 CPU tests / Ruff / independent review, SHA
a6512910ea9fee3395fc33e30e0d1c1d7dd24565f40bfd37444c70751709d8df. The 2 batched 14-update
runs are within 1e-12 of the existing single local fit, and direct matches the original fit
exactly on all 21 temporal outputs. In the first support-reference test, opt_steps=0 was
rejected by the existing config, so without changing the implementation only the test's
reference was changed to suppressing updates in the existing fitter, and the original
visibility decision was compared on the same final pose.

A separate helper for the GPU numerical comparison also passed 22 CPU tests / Ruff, and
registered conditions on 3 fixed poses at 79381 with 1024 pixels × 3 views: response/gradient
1e-9, pose after 14 updates 1e-8 mm, score/parts 1e-9, support bools identical. plan
f8617fd0181c07694fc2c889302b47f55b9f3ea97897a5604449fe943b83a2a5. The real GPU comparison and
temporal tracking have not been run yet. Long-interval reference 73621 is processing the first
1 minute. Independent D2's next 4 chunks all succeeded, bringing the total to 10/180, i.e. 100
s, complete, and 89160 exited 1 at the planned limit. Full 30-minute tracking and independent
ID evaluation for all 3 individuals are not complete.

### 17:23 JST: 200-frame reflection method rejected; long-interval reference continues

The CPU/CUDA qualification at fixed 79381 passed: max response difference 8.22e-15, gradient
1.17e-17, pose difference after 14 updates 5.68e-14 mm, direct-support bools identical. root
independently recomputed the differences for the 23 saved arrays. The new
run_reflection_temporal.py passed 30 CPU tests and independent review, actually registered the
inputs, seeds and qualification, and processed 200 native frames each for direct/Fresnel on
the real GPU. Initialization 1, reset 0; after both outputs were saved, the fixed fit-excluded
pixels were evaluated. Tracking 84.069 s and 119.498 s; the GPU slot was used alone, running
sequentially.

The 2×2 evaluation is [[0.3764513251,0.3740366476],[0.3979229864,0.3942335778]], and the
Fresnel method fails both image conditions. Chest direct support 586→478/600, per slot
[186,200,200]→[136,142,200], max consecutive missing [14,0,0]→[54,28,0]. Total accepted
375→390, but slot0 went 83→82 and slot1's review onset was delayed 79328→79371. Without
relaxing the support, image or per-slot conditions, the method was rejected and extension to
666 was stopped. The independent audit checked all 21 arrays × 2, all 200 supports, the
fixed-pixel arithmetic and 15 raw frames; re-aggregation difference 0. Audit seal
7111f4ea63a2c6be6f2c7386e4c6dfdb4ae57dcb20f1c113729a8dce116e4de7.

Following the decision draft written before the results were opened, support gains/losses
over all 200 and the saved pose/source/proposal/loss at 79318/79372 ±16 frames each were
diagnosed. Chest support gains 0; losses slot0 50, slot1 58. Non-finite score_margin is
written as null in JSON with its original state stated, and the original arrays were kept.
Figures were checked by actually displaying them. This is a comparison of saved backtraces,
not a diagnostic that establishes the causal point where the online run first diverged, or
whether biological individual IDs are correct. No new fits or reflection-parameter search
were done.

The separate long-interval reference completed the first 1 minute [49800,59760);
initialization 1, restores 2, reset 0, all 6 beams and 59753 histories were independently
audited. Support over all 255901 rays / 88345 nodes and 4 reader boundaries was confirmed.
Concurrent execution of the next 40 s and the remaining 170 chunks of independent detection
was pre-registered, and root tracker 77711 and D2 stager 51059 were launched. The latter
reuses the existing 10 successful chunks and proceeds without a limit through all of 348600.
A new operational supervisor and a generic phase audit are being prepared in separate files;
the frozen numerical method is not changed. Human correspondences for all 3 individuals and
the answer on floor position have not been received. The full 30 minutes and zero swaps have
not been reached.

## 2026-09-09 18:57 JST — Redundant mask checks measured, and a separate implementation

For the 100 s long interval, the audit of all 39 checkpoint / 21 candidate, parent history
and additional recurrence, and the 2 mm/12 px geometric support audit over denominators of
all 423637 sources / 149400 nodes, completed. The 2.19 s raw-frame review pack was also
completed, but no actual answers have been received and it is not used for ID accuracy. Old
supervisor 2103 and D2 stager 51059 continue concurrently.

The cost of each phase's registration, runner and audit repeating the semantic mask check
over the whole prefix was investigated. For fixed `[64740,66400)`, 3 views, mean of 2 runs:
NPZ SHA 0.002662 s, decompressing 13 arrays 0.036919 s, original validator 20.439379 s. A
local receipt bound to exact bytes/manifest/seal/source/runtime/schema was newly implemented,
and a loader that keeps all fields of the original LongInputs and a phase audit were put in
separate files. The old source and tracker binding are unchanged, and receipts are saved to
external independent provenance.

Real `[49800,51460)` comparison: original 74.690642 s, cold 74.943090 s, fresh-context warm
46.296146 s. All 12 fields, Rig/camera, seed projections of 9 points × 3 views and 91 input
SHAs exact; cold 3 miss / warm 3 hit. No raw-frame decode, GPU, tracking, GT or old-candidate
reads were done. This is a sequential single-chunk comparison, so it is not extrapolated to a
speed-up rate for the whole video. Success seal
05829e508cf117e0d463472341191cc37af37f796b762b65f3ecc2b30213afdb. ROOT re-verified all 41
output SHAs, and 38 receipt/loader CPU tests and Ruff succeeded. The new phase audit also
succeeded on 17 CPU tests but has not yet been applied to a real phase.

Adoption is limited to the new supervisor/phase auditor; the frozen v1 runner is launched as
is in a separate process. Handoff conditions, which do not swap out the running monitor and
retain evidence of successful children and of the owning process's exit, were recorded in
work/three_body/LONG_SUPERVISOR_HANDOFF_PLAN.md. The old owner has not been stopped and the new
monitor has not been started. The full 30 minutes and zero swaps for all 3 individuals have
not been reached.

## 2026-09-09 19:31 JST — 160 s completed and the audit path actually switched over

first supervised child 66400→76360 exited 0 at 19:20:56, and the old phase audit succeeded
at 19:28:21. Cumulative 26560 frames / 160 s, init 1 / restores 4 / reset 0, all 6 beams /
159353 history retained. Using an independent OS handle witness and stop control, the old
owning supervisor 2920 was stopped at 19:28:24 at a boundary where no next child was
starting. The old overall success seal was not back-filled, and the old lock was also kept.
Independent verification of the real 77 evidence SHAs succeeded. New supervisor 32348 has been
in bootstrap audit since 19:29 and inherits the same frozen tracker. Preceding 93 + handoff
50 CPU tests succeeded. Details: work/three_body/LONG_HANDOFF_RESULT.md.

The 13 risks in the provisional 160 s form 4 scenes (1 scene with insufficient probes), 7.04
s of raw frames in total. slot0/2 remain in ID hold after the first ambiguous scene. Direct
chest support 78855/79680 and accepted 58959 are reported separately. The range/probes of
raw-frame scene0 are the same as at 100 s. Human JSON not received; 30 minutes and zero swaps
for all 3 individuals not reached.

## 2026-09-09 19:53 JST — 160 s comparison against all existing configurations, and female endpoint evaluation

CPU comparison long_support_comparison_160s_v1 succeeded (Hooke 57621 exit 0). 128 support
blocks for all 7 C34 + new were saved, and all 173 outputs re-verified. New chest support
78855/79680=98.9646%; C34 79206–79375 (99.4051–99.6172%); the new method did not exceed them
in support rate. New original accepted 58959=73.9947%; the difference from C34 in what
acceptance means is recorded. Original SL/calib shared, original chest coordinates unchanged;
the C34 reconstruction of mouth and tail is a derived diagnostic. Passed 15 CPU cases / Ruff
and independent code review.

ROOT's separate CPU evaluation long_female_endpoint_160s_v1 also exited 0. All 24 confirmed
annotations were matched with a common 5 px, and the same 23 adjacent intervals scored. New
P23/S0/U0; S for all 7 C34 in fixed order 7/1/0/0/0/1/1. All 24 anchors usable in every
condition; identical for raw support and after applying the original acceptance. Range
51312–75987, adjacency 453–1799 frames (2.73–10.84 s). This is not continuous GT for all 3
individuals nor an actual swap count. Evaluation 10 CPU / independent review and ROOT's full
66 comparison/import/female combined tests succeeded. ROOT re-verified all 10 output SHAs.

New bootstrap succeeded at 19:37:41, with exact agreement on all 21 items of the old audit.
New registration succeeded at 19:41:13 with 48 receipt hits / 18 misses; child 9344 started
at 19:41:46, and real progress to native 76449 was confirmed. The Feynman mixed-chain adapter
succeeded on 59 new + 65 old CPU tests, and the real bootstrap 48-receipt verification v2
succeeded. The failed v1, caused by a SHA function-name mistake in the first diagnostic
driver, is retained. The final 30-minute loader has not been run. Raw-review ingestion
implements a read-only exception for 1 externally specified JSON; 41 tests succeeded; no
actual answers received. The full 30 minutes and retention of all 3 IDs are still not
reached; D2 and long-interval tracking continue on the RTX5070Ti.
