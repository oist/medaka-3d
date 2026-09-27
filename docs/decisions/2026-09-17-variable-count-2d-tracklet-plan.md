# Plan: independent 2D tracklets that handle varying detection counts, reflection and occlusion

Date: 2026-09-17. Status: **a plan for the next session. Neither the implementation of this plan
nor any tracking experiment was done in this session.**

## 1. Purpose and target for the next session

Without using SLEAP's temporal IDs as input, build 2D tracklets in each camera made only of
confident links.
Fish that are apart and whose correspondence is clear are tracked for long stretches; where
crossings, occlusion or competition with reflections occur, links are left uncommitted.
Afterwards, using the reliable views as cues, fragments in the other views are filled in during
3D correspondence.

**The required scope for the next session is stages A–C below: implementation of the 2D method,
comparison under the same conditions, and an HTML report including videos of each view.**
Reconnection in 3D is separated out as stage D and proceeds only after the 2D problems are
confirmed to have decreased.
The present 20 s and the known scenes take priority; reprocessing the whole recording,
retraining the detector, and producing CG are not part of this comparison.

## 2. Fixed premise: separate the three physical fish from 2D detection counts and ID counts

Even though the tank holds three physical fish, the detection count in a view can be **0, 1, 2,
3, or 4 or more**.
Reflections off the walls and water surface, misses, occlusion during crossings, and detections
in which several individuals merge into one are allowed for.
The cause is not decided automatically, as in "it is the 4th, so it is a reflection" or "there
are 2, so it is occlusion".

| layer | meaning of the ID | count / handling |
|---|---|---|
| detection | `(view, native_frame, C01 global_row)`. Retains the original points, dropouts and score | All rows stored. Not capped at 3 |
| 2D tracklet | an observation fragment whose continuity within one view is certain | Unlimited count. Unlinked rows are also retained, as single fragments / uncommitted observations |
| correspondence to physical fish | a correspondence hypothesis between fragments and the actual three fish | Unassigned allowed. 0–3 physical fish may be observed; surplus fragments are not forcibly merged |

Therefore none of the following is introduced: truncation to the top 3 detections, processing
that always folds all tracklets into S0/S1/S2, fictitious detections for missing fish, or
resetting all IDs on a change in detection count.
The existing P/N/F also retain every detection row; this does not mean the earlier methods capped
at 3 rows.
This time the contract is written down and is upheld also at dropouts, competition and
reappearance.

The output keeps at least the following separately.

- Source detections, per-node observation masks, detection score. Per-node scores are not
  invented where the source data lack them.
- Committed adjacent-frame links and their grounds, reasons for hold / no-link, within-camera
  tracklet ID.
- Assessment of mouth / pectoral fin / tail coherence. Not to be confused with the certainty of
  temporal links.
- Physical-fish / reflection interpretation. The initial value is undetermined; a suspected
  reflection carries its grounds and source.
- Broken endpoints and reconnection candidates. Predicted positions go in a separate array from
  observed points; predictions do not fill in the original dropouts.

**Even a long, smooth tracklet can be a reflection. Confidence in a link is not the probability
of being a physical fish.**

## 3. Design of the 2D tracking

### 3.1 Continue through clear intervals

The pectoral fin (P) is the basis for linking, with one-to-one assignment within a frame.
The options of not linking, a new fragment, and an uncommitted observation are handled
explicitly.
A missing mouth (M) or tail (T) alone does not cut a link where P is observed and the
correspondence is unique.
Valid M/T are added step by step as auxiliary information, but a dropout or a suspected mix-in of
a node from another individual is not tied directly to an ID change for the whole fish. Where P
itself is missing, the P-based committed link is held.
Substitute tracking on M/T alone is considered later as a separate condition and is not mixed
into the present P condition.

"Apart" is not decided from the inter-individual distance in the current frame alone.
Confirm that other candidates are far enough away relative to the prediction error to the
candidate, the inter-frame movement, and the position uncertainty.
Evaluate competition both among the candidates seen from each detection and among the candidates
seen from each tracklet.
Short-term predictions of lost endpoints are also included as competing candidates, so **a single
current detection is not automatically safe.**
Detections whose reflection status is undetermined are not removed from the candidate set without
grounds either.

Record the upper bounds on distance and prediction error and the margin over the next-best
candidate; do not simply distribute detections to three fish at minimum cost.
Even when the fish are far from one another, a large jump from the previous position is not
permitted on distance uniqueness alone.
Proximity is a trigger for closer examination; the current rule that always cuts both links just
because fish are close is to be revisited.
Tracking of a third fish that is not in proximity continues independently.

### 3.2 Judge suspect links from both temporal directions

The existing P method already has forward/backward link agreement. It is not double-counted as a
new feature.
With the same before/after context, save the prediction error, candidate ranking, competition and
node support in both directions.
Validate adjacent-frame agreement first; do not add a complex long-interval search at the same
time.

A movement about 80° or more off the heading is a cue for re-evaluation but does not by itself
cut a link.
Taking into account directional instability during small movements, genuine sharp turns and
projection effects, judge it together with movement distance and candidate competition.
When forward and backward disagree, or the candidate margin is small, do not force a single ID.
Even if a tie can be broken by row order, the link is not treated as having become certain.

### 3.3 When a fish disappears and when it reappears

| situation | default behaviour |
|---|---|
| 3 detections become 2 | Continue linking the visible fish; save the endpoint of the vanished observation fragment as open |
| 1 detection / 0 detections | Do not draw in fish that are not observed. Keep short-term predictions separately, with uncertainty growing with elapsed time |
| one detection contested by two fish | Do not duplicate that row into two committed tracks. Do not decide it is a merger of individuals either; hold the relation |
| rises to 4 or more | Store all detections; treat those without correspondence grounds as new / uncommitted fragments |
| separate again after a crossing | Create a fragment for the new clear interval. Identity with the pre-crossing fragments is evaluated separately |

The first-pass committed tracklets are based on adjacent-frame links.
Relations across a gap are kept apart as a **table of reconnection candidates** that does not
modify the original fragments.
Each candidate records the endpoints before and after, elapsed time, forward/backward prediction
errors, other competing endpoints, and the reason for adoption / hold.
The validity period of an endpoint is stated both in seconds and in native frames, avoiding links
to stale positions with no expiry.
Dropping an endpoint from the candidates does not delete the original observations or fragments.
"The detections returned to 3" or "they separated at the exit" alone does not commit the
entry-to-exit ID correspondence.

### 3.4 Handling reflections

The first 2D implementation does not presuppose complete classification of reflections.
If a physical fish and a reflection can both be tracked smoothly at the same time, keeping them
as separate fragments may be correct.
Excluding by proximity to a wall alone would also lose physical fish near the wall, so near-wall
judgements stay diagnostic information.
Not being found in another view is not, on its own, evidence of a reflection either.

Later stages narrow down the physical-fish correspondence using calibration-based 3D position,
reprojection and temporal consistency.
If a mirror-image geometric model is used, first examine the existing reflection-related
implementations and their calibration and validation status.
An unvalidated mirror-image model is not brought into the present 2D judgement.

## 4. Keep comparisons separate: implementation fixes, score, tracking method, parameters

In [the immediately preceding result](../../labs/20260917_clean2d_video_comparison/RESULT.md),
all 29,386 detections in the main 20 s were retained, while the number of contiguous intervals
was stored SLEAP 224, P 323, N 1,257, F 1,324.
P is visible in every row but M is missing in 743 rows, and there were scenes where the
required-node condition of N/F created unnecessary splits.
Many fragments and missing detections are different things, and cutting large jumps is not by
itself a swap improvement.

The [parameter audit](../../work/clean2d_video_comparison_v1/parameter_audit.md) also confirmed
the degeneration of OKS in the current SLEAP when M is missing, and the write-back problem with
the returned row order in C24/C27.
The stored full-length results and the short re-runs differ in history and runtime environment,
so they are not called an identical reproduction.

| comparison condition | change / role |
|---|---|
| stored C27 | Earlier whole-recording SLEAP result. Shown unchanged for reference |
| S1 | SLEAP in the current environment using the same local input and history. Only the correspondence between input rows and return values is stored correctly |
| S2 | Only the OKS scale changed from S1. The area, which becomes zero with a missing M or with certain postures, is replaced by a positive, orientation-independent scale |
| P0 | Frozen pectoral-fin-only independent tracking. Includes the existing forward/backward agreement |
| P1 | The P-based method of this plan. Handles relative candidate competition, partial node dropouts, variable detection count and open endpoints |
| P1 + auxiliary nodes | A separate condition adding valid M/T support to P1. A dropout is not a blanket reason to cut |

For S2, first declare a fixed per-view scale with reference to the scale of the existing P
condition, and fix it before looking at results.
An adaptive scale from past reliable body lengths, if needed, is a separate condition; multiple
changes are not hidden.
The change is not merely replacing zero area with a tiny value, nor merely moving the
orientation-dependent M/P box area to the past.
Changing the no-link penalty and the like together with S2 would make the mechanisms
inseparable, so such changes, if made, are separate conditions.

S1/S2/P0/P1 use the same C01 detection rows, evaluation intervals, and 332 frames of context
before and after.
The difference between past-only SLEAP and the methods that use past and future is stated
explicitly. All are given the same input interval and are evaluated on the central interval.
P1's feature additions and numeric-threshold changes are recorded separately. Run a small
pre-specified sensitivity comparison and report every setting.
Do not pick a convenient setting per scene; per-view settings are applied identically to all
target intervals.
(Optical) flow is not made mandatory; it goes to a separate comparison if P1's results show that
its added value is needed.

## 5. Target data and known scenes

Experiment `20251225_K_1_2_FCMC`, 166 fps, `top_cam / side1_cam / side2_cam`.
It is one group of the same three fish and is not treated as multiple independent experiments.
Intervals include the start and exclude the end.
The canonical copy of all 11 cases is the
[existing protocol.json](../../work/clean2d_video_comparison_v1/protocol.json).

- Main 20 s: `[75270, 78590)`. Review positions 76235, 76791, 76930, 76988, 77415.
- Known scenes: 21232, 30735, 31479, 105541, 260434, 302762, 323819, 333619.
- `control_50000` and `holdout20s [240000,243320)` have already been investigated, so they serve
  as regression checks.
  The latter is not to be called a new unseen test.

Concrete examples of changing detection counts were also confirmed by reading the stored NPZ
files.

| case / view | native frame and detection-row count | evaluation note |
|---|---|---|
| known_30735 / side2 | 30734–30736: 1 each; 30746–30747: 2 each | Separate a change in the 3D assignment of the same 2D fragment from a switch in 2D itself |
| cg20s / 3 views | 76930: top=2, side1=2, side2=3 | Example of the visible count differing per view |
| known_31479 / side1 | 31479: 4; 31486, 31488, 31499: 5 each | Whether the surplus rows are reflections is undetermined. Do not shrink to 3 |
| known_260434 | 260434: side1=2, side2=4 | An old note records a 3-frame fusion in side1. Cause of the side2 surplus undetermined |
| known_323819 / side1 | 323819: 3; 323820: 2; 323821–323824: 3 each; 323825: 1 | Later human judgement: no ID swap. A control against mis-repair |

The source of the table above is the detection-row counts in
`work/clean2d_video_comparison_v1/{case}/tracking_F_{view}.npz`.
F's ID decisions are not used as ground truth. The core of 11 cases × 3 views, 28,962 view-frames
in total, contains no 0-detection frames, so 0 detections and recovery from them are validated
with synthetic tests.
Even where old notes mention occlusion or reflection, this is not read as a reflection label on
individual detection rows here.

Input locations (relative to the repository):

```text
C01 detections:
work/generative_tracking_inputs/work/c01_s16_predictions/bottomup/
  20251225_K_1_2_FCMC__{view}.npz

Stored SLEAP:
work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any/
  20251225_K_1_2_FCMC/mouth-pectoral_oks_serial/track2d__{view}.npy

Videos:
work/three_body/source_cache/medaka_202512/20251225_K_1_2_FCMC/
  top_cam/cam0_top.mp4
  side1_cam/cam1_side.mp4
  side2_cam/cam2_side.mp4
```

C01 retains the grouping of nodes within an instance, but the independent method does not read
SLEAP temporal IDs.
Loading the SLEAP labels for comparison is kept separate from estimation by the new method.
Validation with the detection / input order permuted is also run, mapping back to the original
global_row to compare links.

## 6. Evaluation and visual checks

### Required structural tests

1. With 0/1/2/3/4/5 detections, and with the sequence 3→2→1→0→2→4→3, row retention, one-to-one
   assignment and continuity of the other individuals are upheld.
2. Distinguishes a link with a clear P despite missing M/T, a missing P, and a suspected node
   mix-in.
3. When two trajectories compete for one detection, IDs are not merged and the same observation
   is not duplicated.
4. Handles a stable surplus image, a physical fish near the wall, and cases where several past
   endpoints compete for one detection.
5. Does not permit a large jump to a distant current candidate on uniqueness alone, and does not
   decide the entry ID from separation after a crossing alone.
6. Committed links do not change when row order changes, and ties, forward/backward disagreement
   and missing observations are not committed without grounds.
7. Predicted points do not leak into observed points or detection rates, and the original points,
   dropout mask and global_row match.

### Quantities compared

Stratify by detection count (0/1/2/3/4 or more), separated / close, and presence of node dropout.
Do not let improvement in easy intervals offset deterioration at crossings in overall values
alone.

- Tally separately: retention rate of source detections, unlinked / uncommitted observations,
  number of contiguous intervals, adjacent links and links across a gap.
- Where ground truth exists, tally separately: correct continuation, wrong links, holds, and
  unnecessary cuts in clear intervals.
- ID continuity where the entry-exit correspondence can be confirmed. Separate node exchanges
  from individual-ID exchanges.
- Wrong links between physical fish and reflections, within the range where reflection labels
  were confirmed. Anything else is presented as undetermined cases.
- State the denominator of each metric, the evaluated frames, and the annotated range.

Adoption conditions: satisfy the contract (all rows retained, etc.), create no new wrong links in
confirmed clear intervals, and reduce unnecessary splits.
A decrease in fragment count or jump count alone is not a success condition.
Holding every link would reduce wrong links, so the correct-continuation rate and the hold rate
are always reported alongside.
Smoothness in unannotated intervals is not substituted for ID accuracy.

The existing female GT and [the recent human review](../reviews/2026-09-16_multiview_identity_review.json)
are not frame-level 2D ground truth for all three fish. The old C30 slot names are not reused as
2D ground-truth IDs either.
Unknown correspondences are left undetermined and, with concrete entry/exit images and videos
prepared, reported as items a human can answer.
Items awaiting additional annotation are kept apart from items confirmed by automated tests, and
implementation and rendering are completed first.

### HTML and videos

- For each view, a SLEAP / P0 / P1 comparison side by side on the same frames and the same full
  set of detections. S1/S2 are also selectable.
- Include a 20 s whole-tank video, zoomed slow-motion videos of known events, and stills from
  before and after.
- Display detection count, local fragment ID, observed M/P/T, and the reason each link was
  adopted / held.
  The same colour in different methods is not evidence of the same individual. If a display-level
  matching is applied, it is labelled explicitly and kept separate from the IDs used for
  estimation.
- Fragments whose correspondence to physical fish is uncommitted are not labelled S0/S1/S2.
  Suspected reflections may also be displayed as undetermined.
- Observed points are drawn with solid lines and filled dots; displayed predictions are
  distinguished with dashed lines or similar. Invisible fish are not drawn as if observed.
- Provide time-series plots of native frame and time, detection count, candidate distance and
  prediction error, forward/backward disagreement, and node dropout.
- Publish all settings and comparison tables, and describe per scene the 2D improvements, the
  undetermined items, and the problems remaining on the 3D side.
- Decode the generated videos and check the HTML's toggles, playback, links and displayed numbers
  before handing over.

## 7. Execution order

### Stage A: fix the input contract and comparison conditions

Record the SHAs of the existing protocol, stored results and code, and create a new experiment
directory.
Verify S1's row correspondence, S2's scale change, and the reproduction of P0.
Register synthetic variable-detection-count examples and the real data in the table above as
regression tests.
Report improvements from the conditions that modify SLEAP separately from improvements from the
independent method.

### Stage B: implement P-based variable-detection-count tracklets

Implement P1 and try adjacent linking that responds to candidate competition, and separation of
node quality.
Use the short-term state of lost endpoints in the competition evaluation, and first save a
reappearance as a separate fragment.
Also build the comparison condition with M/T support added, and check what changes relative to P
only.
Recording reconnection candidates is made possible, but committed merges across a gap and 3D
estimation are not mixed into the basic 2D result.

### Stage C: compare and report on the 20 s and known scenes

Run on the 11 cases above and all 3 views, with particular attention to the intervals a human
described as clear and to the no-swap controls.
Save the HTML, per-view videos, stills, plots, settings, and machine-readable detections / links /
summaries.
Everything up to here is the required deliverable for the next session. If there is no
improvement, report the result and the cause, and do not conclude superiority.
For validation after the settings are fixed, choose 20–30 s that do not overlap
already-investigated intervals, and record the selection rationale.
If the settings are changed using results from the unseen interval, a new validation interval is
required.
State explicitly that it is a different interval of the same three fish, and do not treat it as
generalisation to independent groups.

### Stage D: after the 2D check, evaluate correspondence across gaps and across views

Evaluate reconnection candidates over short before/after intervals, using fragments from reliable
views as ID cues.
Even where a track is broken in another view, verify the correspondence using the observations
available at the same instant and the calibration.
Continuity in all 3 views is not required; make use of the geometry of the 2 available views.
One continuous view can be a cue, but it alone does not commit an ambiguous correspondence in
another view or a new 3D depth.
3D predictions for times when only one view has observations are distinguished from
observation-based 3D.

The existing [clean_tracklets implementation and result](../../labs/20260917_clean_tracklets/RESULT.md)
already has links by two-view agreement and by support from another view. Check its input and
output conditions before reuse.
Links made by 2D itself and links supported by 3D go to separate outputs; IDs assigned in 3D are
not circulated back as grounds for the independent 2D.
Re-examine the 3D assignment failures in scenes where 2D was already continuous, such as f30735,
f31479 and f302762.
The condition of three physical fish is used in this correspondence layer, retaining reflection,
unassigned and invisible states.

## 8. Implementation, storage locations, and how to resume

- New experiment: `labs/<run_date>_variable_count_tracklets/`. The README states the hypothesis,
  stage, expiry and criteria, and a RESULT is left.
- New output: `work/variable_count_tracklets_v1/` as the candidate; if it already exists, do not
  overwrite it, choose the next unused name.
- Reference comparison code: `tracking.py`, `c27_replay.py`, `parameter_audit.py` and the
  rendering and aggregation code in `labs/20260917_clean2d_video_comparison/`. Keep the existing
  code and results frozen and keep the experiment separate.
- Do not add direct imports between labs, or make production code import lab code. Record the
  source and SHA of any frozen implementation that is reused.
- Use `.venv/Scripts/python.exe`. Run the relevant contract, tracking and rendering tests and
  Ruff, verifying the scope needed for the change.
- Attach `medaka3d.provenance.provenance` and SHA records to inputs, code, settings and outputs.
  This plan and any inferences are not registered as new human ground-truth labels.
- Keep existing uncommitted changes, the production pipeline, and earlier HTML and NPZ files.

Instruction to hand to the next session:

> Read `docs/decisions/2026-09-17-variable-count-2d-tracklet-plan.md` and implement and run stages A–C.
> There are three physical fish, but allow 0 to 4-or-more detections and tracklets per view, and handle reflection, occlusion and non-assignment.
> Compare P-based independent 2D tracking that does not use SLEAP temporal IDs against SLEAP under the same conditions.
> For the present 20 s, the known clear intervals, the failure cases and the no-swap controls, complete an HTML report containing per-view comparison videos, images and plots.
> Keep integration into 3D separate as stage D, and do not conflate it with the present 2D improvement.
