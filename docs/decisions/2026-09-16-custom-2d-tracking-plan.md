# Plan: compare 2D tracking with SLEAP detections held fixed

Date: 2026-09-16. Status: investigation of code and stored data, and recording of the human
review, complete. No comparison experiment with a new 2D tracking method, and no production
use, has been done yet.

## Decision

Custom 2D tracking offers room for improvement. In particular, against the current setting,
which commits links even at low similarity, it has the advantage of being able to handle past
and future motion, link ambiguity, and time spent lost. That it outperforms SLEAP, however, has
not been verified. The appropriate comparison keeps the detector fixed as the current SLEAP and
makes only the temporal linking interchangeable.

An additional check against the original C27/C28 found **a mixture of jumps within a single SLEAP
ID and jumps caused by 3D-side correspondence and post-processing**.
f76235 has a large movement inside an original 2D ID, whereas f30735 reassigns the same side2 2D
ID to a different slot.
Around f31479, C28 has a jump of about 173 mm at 31481, which in C34 becomes jumps of about
175 mm at 31479 and about 171 mm at 31497.
At each of these boundaries the movement within the same original 2D ID is small, so the 3D
correspondence and the subsequent ID processing have to be examined together.
A plan that replaces only the 2D tracking therefore cannot resolve all of the present failures.

The swaps found here are not all asserted to originate in SLEAP.
Body-part mixing within a detection row, cross-view correspondence, temporal 2D labels and the
final 3D slot ID are separate problems.
`source` in the stored C34 is the detection row number within the frame, not the original C27 2D
tracklet ID.

## The human review recorded here

[The annotation, with the original wording and video hashes](../reviews/2026-09-16_multiview_identity_review.json)
was recorded.
In the source videos, left is C34 and right is the unaccepted candidate. The user did not specify
which side was judged, so the judgement is not to be read as applying to both left and right,
nor as permission to adopt the candidate.
Frames are anchors that identify scenes, not annotations of exact swap start and end times.

| frame | human judgement | use in the comparison experiment |
|---|---|---|
| 76791 | IDスワップなし (English: "no ID swap") | A control against mis-repair. Selection of a geometric candidate is not treated as an ID error. |
| 76988 | S1とS0がスワップ (English: "S1 and S0 swap") | The previous candidate, which focused on the S1/S2 tails, did not evaluate this ID problem. |
| 76235 | S0とS2がスワップ (English: "S0 and S2 swap") | Evaluate body-part residuals and individual continuity separately. |
| 76930 | S0とS1がスワップ (English: "S0 and S1 swap") | All three fish, including S1, must be displayed and evaluated. |
| 30735 | side2でS0がS1へ飛び、戻る (English: "in side2, S0 jumps to S1 and comes back") | A temporary within-view link failure. Distinguish it from a permanent ID swap across all views. |
| 31479 | S0とS1が大きく飛んでスワップ (English: "S0 and S1 jump a long way and swap") | Validate ID repair inside the anomalous interval. |
| 323819 | スワップなし、パーフェクト (English: "no swap, perfect") | A large geometric residual is not grounds for changing the ID. |
| 77415 | S1/S2のノード交換はあるがIDスワップなし (English: "there is S1/S2 node exchange, but no ID swap") | Body-part repair must not wrongly change the ID of the whole fish. |

Tally: 4 scenes with an ID swap, 3 without, and 1 temporary within-view jump.
The 8 scenes are not 8 independent individuals or 8 experiments; they are selected scenes within
the same experiment.
In particular, the 76930 and 76988 videos overlap by 108 frames. The 4 "reviews with a swap" are
not counted as 4 independent, de-duplicated swaps. They are merged once the exact intervals of
occurrence are confirmed.
f75398 was not included in this response and is undetermined.

The annotation and provenance are stored under the configured
`asset_root/groundtruth/swap_event_gt/20251225_K_1_2_FCMC/`
`20260916_multiview_node_review_v1/`, SHA256-verified.
2 entries were added to MANIFEST, keeping the existing 3301 entries.
See the [archive record](../reviews/2026-09-16_multiview_identity_review.archive-status.json).

## What was confirmed in the current code

1. `c27` uses SLEAP's temporal tracking via `c24.run_tracker`.
   Mouth + pectoral fin, OKS, local queues of each track's last 5 observations, Hungarian
   assignment, no cap on the number of tracks.
   Per-node scores are passed as 1, so neither the detection score nor the matching score can be
   interpreted as a probability that the ID is correct.
   The current settings use no image, optical flow or Kalman.
   SLEAP itself in the current environment has these options, so this is not called a
   shortcoming of the library as a whole.
2. `c25`/`c28` charge λ per slot whose three-view label triple changes.
   The SLEAP ID is a finite penalty, not a fixed constraint. Within a frame, evaluation is by
   geometry and missing observations.
   But if fish exchange inside the same 2D label, false continuity can be carried in while the
   label stays unchanged.
   The current transition contains no continuity of position or heading.
3. The mouth and pectoral fin of one detection row are handled as a unit, so changing the
   temporal ID alone does not fix body-part mixing.
   The exclusion of unassigned rows must also be distinguished from the soft penalty on label
   triples.

Code evidence: [`c24`](../../tools/c24_uncapped_tracklets.py),
[`c27`](../../tools/c27_uncapped_tracklets_any.py),
[`c25`](../../tools/c25_variable_cardinality.py),
[`c28`](../../tools/c28_lambda_grid_transfer.py).
In standard SLEAP too, post-detection tracking is a separate step:
[official tracking guide](https://github.com/talmolab/sleap-nn/blob/main/docs/guides/tracking.md).

## How this differs from methods tried before

A custom [`track2d.py`](../../src/medaka3d/track2d.py) already exists.
It compares pectoral-fin point assignments across 2 adjacent frames and cuts when ambiguous, but
it handles only frames with exactly 3 detections.
It has no velocity prediction, backward check, variable detection count or recovery after missing
observations, and the current C27 does not use it.
The earlier comparison gave many short tracklets, and there is no basis for simply reverting to
it.

C24b already cut 2206 links with large 2D movement.
In the female gated evaluation at λ=2 at the time, this went from 263 kept / 34 split / 1 swap to
265/32/1, and the swap count did not change. That is a past metric, different from the current
all-slot ID evaluation.
The element to add and test now is **link decisions that use past and future prediction and
ambiguity**, not distance alone.

The earlier 50-frame majority vote is recorded as increasing female swaps even while geometric
metrics improved.
Long smoothing or fixing IDs is not a substitute for improvement.
Sources: [the 2D-first record](2026-08-18-2d-first-tracking.md),
[the C24/C24b record](2026-08-25-uncapped-tracklets.md). The numbers are quoted as reported at
the time, not newly re-measured.

## Why f31479 remains

Comparison with the original C28 also confirmed that the size of the problem changes in later
stages.
Each row is in that stage's slot order; the same slot number in different stages does not
necessarily mean the same individual.

| boundary | C28 per-slot 3D movement, mm | C34 per-slot 3D movement, mm |
|---|---|---|
| 31478→31479 | 1.02 / 12.08 / 0.20 | 174.92 / 172.92 / 0.20 |
| 31480→31481 | 173.63 / 173.15 / 0.39 | 3.33 / 3.53 / 0.39 |
| 31496→31497 | 0.94 / 0.79 / 1.07 | 170.57 / 171.04 / 1.07 |

At the entry, the maximum mouth/pectoral-fin movement within the same original C27 ID is top
18.07 px, side1 2.78 px, side2 4.52 px.
At the exit, the maximum over all views is 8.03 px. C28 does not change the 2D label triple at
the exit.
Treating the ~170 mm jumps at these boundaries as jumps inside an original SLEAP ID is
inappropriate.
For the large movement at C28's 31481 as well, the maximum within the same original C27 ID is top
5.61 / side1 2.34 / side2 3.98 px.
Because C28 also has large movements, treating C34 post-processing as the sole cause is also
inappropriate.
On the other hand, the change of top-label selection in C28 at the entry, the 12.08 mm movement,
and nearby 2D inconsistencies remain, so this does not mean that everything upstream is correct
either.
A wrong link between fish that overlap in 2D can occur with a small movement, so a small 2D
movement alone is not a guarantee of ID correctness.

Re-reading the stored C34, the S0/S1 exchange spans the 18 frames 31479–31496 and reverts at
31497.
The entry movement is 174.92/172.92 mm and the exit 170.57/171.04 mm.
The earlier record "repaired 31479" is qualified in the same sentence by "identity restored from
31497".

- C33 is a coordinate repair that drops a doubtful view; it does not change IDs.
- C30's zone audit changes `perm[t1:]` after the anomalous interval is exited. It does not
  re-track the interior of the interval.
- C34's out-and-back exchange repair covers at most 3 resolved frames. An 18-frame exchange is
  out of scope.
- The bidirectional processing here rejected the candidate 31476–31483 as
  `invalid_independent_flank`.
  The left reference 31459–31475 contains a 12.81 mm jump, and the right reference 31484–31500
  contains the ~171 mm return at 31497.
  Because the anomaly had been split into small pieces, the anomaly itself had ended up inside the
  before/after references.

Computing the hypothesis of exchanging S0/S1 over 31479–31496 only, without modifying the stored
arrays, brings the entry down to 2.61/12.08 mm and the exit to 0.79/0.94 mm.
**The problem is not that 3D cannot repair it; what is missing is processing that repairs the
interior of the anomalous interval.**
The 12.08 mm at the entry remains, however, so the label change alone cannot be said to have
normalised the coordinates too.

Recomputing C33's drop-view from the stored 2D points gives, for 31479 S0, best 0.18168 mm,
runner-up 0.36895 mm, margin 0.18727 mm, which does not meet the required 1 mm margin.
This is a reproduction diagnostic from the currently stored points, not the original C33 run
log.

Evidence: [the C30 decision record](2026-09-01-slot-stitching.md),
[`c30`](../../tools/c30_slot_stitch.py), [`c33`](../../tools/c33_ray_rejection.py),
[`c34`](../../tools/c34_microflip.py),
[the rejection record from this work](../../work/bidirectional_reconnect_v1/analysis_v2/known_31479/analysis.json).

The [numerical diagnostic JSON](../../work/multiview_node_review_v1/human_review_followup/f31479_diagnosis.json)
and the [reproduction script](../../labs/20260916_multiview_node_review/audit_reviewed_31479.py)
were also saved.
Unchanged SHA256 of the original inputs, provenance, JSON read-back and Ruff were checked.

## Next comparison experiments

### 1. Identify the stage where each failure first arises

For the present 20 s 75270:78590 and for the 8 scenes and known cases here, line up, per view:
detection row → C27 2D ID → C28 three-view correspondence → final slot.
After checking the original C27/C28 outputs, classify each as a wrong link within 2D, a
cross-view correspondence error, or body-part mixing.
The C01/C34 currently in local staging alone cannot settle this attribution.
The original C27 serial labels for the 3 views and C28 λ=.767 were confirmed to exist in the
configured shared `work_dir`.
There is no need to restart SLEAP on a short interval and change the history.

An additional check was run over ±40 frames around each of the 8 scenes. Representative
observations follow.

| scene | observation of original 2D labels and selected points | stage at which to try improvement |
|---|---|---|
| 76235 | In top, movement of 76.38 px (mouth) and 66.91 px (pectoral fin) within the same C27 ID=1. In side2, the slot assignments of ID=2/1 are also exchanged. | Both 2D linking and three-view correspondence. |
| 30735 | side2 C27 ID=0 is reassigned from S1 to S0. The coordinates of the same original ID do not jump far. | Three-view correspondence; assignment during missing observations. |
| 76930 | In side2, the C27 IDs 0/2 chosen for S0/S1 are exchanged. | How original 2D IDs are linked to 3D slots. |
| around 31479 | As in the table above, C28 has a large movement at 31481, which in C34 becomes large movements at entry 31479 and exit 31497. | 3D correspondence, and ID handling inside and at the boundaries of the interval in C30 and similar. |

These are observations for isolating the stage of the tracking process; 2D label agreement alone
is not taken as biological-ID ground truth.
An unchanged label at the centre of 76988 also does not mean the whole overlapping review video
is swap-free.

The final evidence is the [per-stage diagnostic v2](../../work/multiview_node_review_v1/human_review_followup/stage_a_review_audit_v2.json)
(main boundaries under `key_edges`) and the [reproduction script](../../labs/20260916_multiview_node_review/audit_reviewed_stage_a.py).
For all 3 views, the SHA256 of the current shared C01 and of the local C01 match, and row counts
and frame order against the C27 labels were checked.
The original C27 provenance records no input SHA, so a hash proof back to the inputs used at the
time is not possible.
Diagnostic read-back, code and output hashes, and Ruff were checked.

For f31479, merge the related jump and return into one anomalous interval and take the stable
intervals outside it as the before/after references.
Loosening the rejection threshold while keeping a fixed small window is not the approach taken.
This within-interval repair is validated as a condition separate from the 2D tracking-method
comparison.

### 2. Three conditions with detections held fixed

| condition | 2D temporal linking | purpose of comparison |
|---|---|---|
| A | stored current SLEAP serial tracking | baseline |
| B | audit A's links with past/future motion and ambiguity, and cut suspect links | can it improve without full replacement? |
| C | custom variable-detection-count, bidirectional tracking on the same full set of detections | can it improve without using SLEAP's links? |

Detection points, calibration, downstream geometry, handling of missing observations, and initial
and boundary conditions are matched.
So that scenes where post-processing added errors are not lost, evaluate both immediately after
C28-equivalent reconstruction and after C30/C34-equivalent ID post-processing.
First examine the mechanism at the current λ=.767; before any general claim of improvement or
adoption decision, also check all 7 existing λ.
The main 20 s interval and the failure cases are the development set; after the thresholds are
fixed, a separate 10–30 s becomes the unused evaluation set.

Stored A carries a serial history over the whole recording. A result from a fresh SLEAP run over
the 20 s alone is not called the same baseline.
In local comparisons, sufficient context before and after is given under the same conditions,
and the reproduction of A and the boundary effects are checked first.
For a window where a stable boundary cannot be obtained, the context is extended; if it is still
unclear, the limitation as a local test is stated explicitly.
The arbitrary numeric IDs of each method are not compared directly; what is compared is whether
detections of the same fish are linked.

### 3. Minimum specification for custom linking

- Use robust motion prediction of mouth + pectoral fin and the prediction error in both
  directions.
  A movement 80° or more off the heading is a cue for suspicion; at low speed or in natural
  turns, do not cut on angle alone.
- Make explicit the margin over the next-best assignment, the time elapsed since observation, and
  missing observations, and allow no-match.
  When a crossing produces a near-tie, keep a short candidate sequence and re-select from
  subsequent frames.
- Handle detection counts of 2 or 4 as well. Do not cut detections down to the top 3 on the
  grounds that 3 fish exist.
  Even if one fish is ambiguous, do not give the other two new IDs along with it.
- Separate tail oscillation and partial node exchanges from ID exchange of the whole fish.
  Body-part quality is handled only with measurable information; do not invent a node confidence
  that does not exist.
- Save each tracklet's cut reason, candidates, cost margin and detection-row origin.
  An extension that re-selects 2D links using 3D is reported separately from the 2D-only
  condition.

### 4. Criteria and visual review material

The primary evaluation targets are wrong 2D links and ID swaps in the final 3D.
Number of splits, time lost, fraction of detections retained, and body-part mixing are measured at
the same time.
A method that lowers only the apparent swap count by cutting all links or smoothing coordinates
does not count as an improvement.

Confirm that the scenes with no swap in this review, 76791/323819/77415, are not mis-repaired, and
check actual fish continuity in the scenes with a swap, 76235/76930/76988/31479, and in the
view-limited 30735.
Until the reviewed side and the event intervals are established, these are not made a complete
ground-truth sequence for automatic scoring.
The existing female-only metric and the gated tracker's 0 swaps are not a substitute for
final-slot ID accuracy over all three fish either.

Always display labels for all 3 views and all 3 fish, and present the baseline and candidate
videos, entry / middle / exit images, and time series of 2D links and 3D IDs at the same frame
numbers.
Keep the human-confirmed range and the mechanical diagnostics separate.

## Visual review material added here

Previously, the focus setting left S0 in 76988 and S1 in 76930 undisplayed.
Using the same 166 native frames as the originals, videos and stills showing all three fish were
added.
No new candidate generation or ID repair was done.

- [f76988 video, all three fish](../../work/multiview_node_review_v1/human_review_followup/cg20s_f76988_allslots_native_slow.mp4)
- [f76930 video, all three fish](../../work/multiview_node_review_v1/human_review_followup/cg20s_f76930_allslots_native_slow.mp4)
- [generation and verification manifest](../../work/multiview_node_review_v1/human_review_followup/manifest.json)

Both videos were checked by reading back every frame and by SHA256. Left / right are the original
C34 / the unaccepted body-part correspondence candidate.
These two are not results of a new 2D tracking method.

## 2026-09-17: implementation and comparison of stages 1–3

At the user's request, A/B/C were implemented and compared on 20 intervals in total: the main
20 s, known cases, controls, and a different 20 s from the same group of fish.
C includes a limited implementation that re-evaluates two ambiguous candidates using subsequent
observations.
The detailed specification, measurements and limitations are in the
[experiment result](../../labs/20260917_custom2d_comparison/RESULT.md); the visual material is in
the [HTML report](../../work/custom2d_comparison_v1/comparison_report.html).

In the main 20 s, final movements over 10 mm were A/B/C = 3/4/3, and the ID change at the female
endpoints remained at f76698→f77285 under every method. The maximum movement at f31479 fell
174.92→12.81 mm, while in the no-swap control f323819 C's large movements rose 4→6. A reduction
in swaps has not been demonstrated, and full replacement is on hold. The final version is
`work/custom2d_comparison_v1/analysis_v3/`.
The original C27/C28/C34 were not modified.
