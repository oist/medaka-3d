# Plan: jump detection and local reconnection from past-and-future motion

2026-09-16. Proposal stage. No existing tracking has been modified and no new tracking has been
run.

## Hypothesis

In scenes where the positions of the three fish in each frame are broadly correct and only the
temporal correspondence is wrong, re-evaluating the correspondence using the stable motion
before and after a suspect interval may detect and repair inconsistencies that a past-only
motion evaluation misses, and exchanges in which fish facing each other appear to bounce off one
another. Trajectory smoothness and individual-ID correctness are evaluated separately.

Merely computing the acceleration or angle of the same three points in reverse order as well
adds no new information. Motion is estimated from separate stable intervals on the past side and
the future side, and these constrain the entry and exit of the anomaly. The work starts as
offline processing. An online version would need a delay to wait for future frames.

## How this differs from existing attempts

- C30 v4 compares position medians over 8 points before and after, then, for a restricted set of
  held candidates, extrapolates from up to 50 past frames. The implemented velocity is the
  difference between the two ends of the history. It does not use future-side velocity, and
  when the position comparison finds the current correspondence best, it does not re-evaluate
  direction.
  See: `tools/c30_slot_stitch.py:208`, `:284`, `:314`.
- C30 v3 fixed known cases but also created new female exchanges. The cause was ambiguous joins
  and weak extrapolation into long broken intervals; v4 tightened the repair conditions.
  See: `docs/decisions/2026-09-01-slot-stitching.md:154` onward.
- C34 checks the entry permutation against the inverse exit permutation for short permutation
  exchanges that revert within 3 resolved frames. It does not repair isolated coordinate spikes
  or sustained exchanges.
  See: `docs/decisions/2026-09-01-doubt-channel.md:213`.
- Standalone backward optical-flow propagation on the analysis-by-synthesis side moved the chest
  62.83 mm away from the original forward estimate within 30 steps and was not adopted. That
  differs from the "local reconnection with stored detections held fixed" proposed here. The
  difference was measured against an internal estimate, not human ground truth.
  See: `work/three_body/reverse_motion_v1/RESULT.md`.
- The adjacent-frame motion term in the current `identity_dp.py:226` is the squared chest
  displacement. There is a forward-backward DP, but the motion term itself has no history of
  past or future velocity direction or acceleration.
- `temporal_refine.py:66` compares the 6 correspondences at the boundaries on both sides and also
  evaluates velocity and acceleration. It is, however, a missing-observation recovery that
  optimises the coordinates themselves, and differs from the direction diagnosis here, which
  holds the coordinate set fixed. The idea of using boundaries on both sides is not itself new
  to this plan.
- There is an actual case where removing a 174 mm jump still left a quiet exchange in 3–4 mm
  steps. "Fewer jumps" alone is not grounds for adoption.
  See: `docs/decisions/2026-09-01-doubt-channel.md:264`.

## Exploratory check on the existing 20 s CG interval

Target: C34 lambda 0.767, `75270:78590`, 166 fps. Without modifying the original chest
coordinates, each step was compared with the component-wise median of the displacement vectors
over 17/33/50 frames before and after. The step under test and the 2 steps on either side of it
were excluded from the reference windows. The angle was left undefined when the reference speed
was below 0.05 mm/frame. The future-side direction is compared with its sign aligned to forward
time.

Examples with the 17-frame (about 0.10 s) reference window:

| source frame / slot | displacement | angle vs past side | angle vs future side |
| --- | ---: | ---: | ---: |
| 76930 / 1 | 4.48 mm | 130.0° | 114.8° |
| 76235 / 0 | 14.61 mm | 32.1° | 19.7° |

Of the 21 large displacements, those at 80° or more numbered 10 against the past side, 7 against
the future side, and 10 in the union. Within these 21, the backward direction alone picked up
nothing new. Widening the range to displacements over 1 mm gave 85 candidates on the past side
and 99 in the past-future union. Of the 14 additional candidates, 6 had an undefined past side
because of low speed or similar. These are slot-step counts that include overlap between
consecutive frames, not a count of anomalous events, a count of ground-truth events, or a
detection accuracy. They may have picked up genuine turns, and are not interpreted as an
improvement until checked against the three-view source images.

On this result, angle and distance are kept as separate candidate-extraction conditions. 80° is
an initial hypothesis based on the user's observation, not a confirmed biological constraint.
Source data, settings, all candidates and provenance are stored in
`work/aquarium_cg/direction_probe_20s/`.

## Experimental procedure

### 1. Compare anomaly detection without repair

Compare the following on the same full set of native frames. Compute before decimation to 30
fps.

1. Distance only: over 4 mm, as in the current CG.
2. Distance + past direction: displacements deviating by 80° or more from the past stable
   interval.
3. Distance + past-and-future direction: displacements showing a direction inconsistency against
   either the past or the future.

So that angle alone does not pick up large numbers of tiny wobbles, report alongside it the
absolute displacement, its ratio to the typical displacement before and after, the lateral
displacement, and the spread of direction within each reference window. At low speed or while
station-keeping, angle is not a basis for the decision; decide on distance, position-prediction
error and image support. Body axis and heading are distinguished: the mouth-to-chest orientation
is auxiliary information, and the tail is treated as weak evidence.

Reference windows of 0.1/0.2/0.3 s and angles of 60/80/100° are all reported as a sensitivity
comparison. Do not select only a convenient setting after looking at the 20 s and claim accuracy
from it. The 1 mm / 0.05 mm conditions of the exploratory check are also provisional, and are
frozen only after looking at false detections in normal intervals. Direction is not estimated
from a window that contains an isolated spike; robust velocity is estimated outside an interval
that merges adjacent candidates. The candidate steps themselves are not smoothed.

### 2. Separate individual-correspondence errors from position errors

Handle all three fish jointly and compare the 6 possible correspondences across the candidate
interval. No-repair is always included as a candidate. Use not only endpoint positions but also
the predicted direction and speed on both sides, mouth-chest consistency, and detection and
reprojection support in the three views.

- Detections correctly placed, only the ID wrong: correct the permutation without moving the
  coordinate set.
- Exchanged for a single frame and then reverted: evaluate a permutation confined to the
  interval, without propagating it outside the interval.
- The 3D position itself is wrong: an ID permutation does not fix it. Separate which view or node
  is responsible, and consider, as a separate option, re-triangulation from the supported rays.
  The effects of position repair and ID repair are not mixed.
- Insufficient support, disagreement between past and future correspondence, or a long dropout:
  do not repair automatically; retain as a review candidate.

Candidate extraction picks up an inconsistency on either side alone, but automatic repair is
limited to cases where past and future support the same correspondence and both the margin over
the runner-up and the image support are sufficient. Past-future agreement is not independent ID
ground truth either. Start with cases whose anomalous interval is 0.5 s or shorter. Do not bridge
long intervals by constant-velocity extrapolation. A sustained exchange is recorded and evaluated
as a link to the later stable tracklet. The permutation is applied consistently to `source`, all
nodes and the related ID arrays. Avoid designs in which linking to a missing observation has
zero motion cost. There is a record of this loophole actually producing an error in which a
normal fish was assigned to a missing slot for a single frame
(`labs/20260909_three_body/ITERATION_LOG.md:455`). Keep the elapsed real time since the last
direct observation and the uncertainty of the missing stretch, and do not treat a path through
missing observations as smooth ground truth.

If the implementation is extended to a DP, simply re-running the current first-order 6-state DP,
which has no direction history, is not enough. First estimate direction from the short tracklets
before and after and evaluate the link; if needed, consider a second-order model that retains the
preceding permutation pair.

### 3. Controlled evaluation against source images and CG

Validate first on the present 20 s and on known cases. Besides 76930, short out-and-back
exchanges and position spikes, include the earlier mis-repair cases 82k/90k/212k, and normal
sharp turns, fast movement, close crossings and station-keeping. Compare the CG before and after
repair with the three-view raw images at the same native frame. Cases picked up only by adding
the direction condition are counted separately.

Detection is evaluated by the capture rate of human-confirmed anomalous intervals, false
detections in normal intervals, and time needing review. Repair is evaluated by exchanges,
mis-repairs, unresolved cases, and dropout and coverage, based on independent correspondence
checks. Improvement in step, velocity and acceleration is a secondary metric. Keep the existing
298-interval female evaluation; male-male cases are evaluated only on intervals where a human
could confirm the before/after correspondence. Unconfirmed intervals are not treated as ground
truth, and agreement between both directions is not made a ground-truth label either.

### 4. Freeze, then move to unused intervals

State explicitly that the 20 s and the known cases are for development. After freezing the
settings, widen the checked range in order: an unused contiguous 1 minute, the existing arms at
all 7 lambdas, then K_1_1/K_1_3. An improvement on only one arm or one interval is reported as
limited to that scope.

Adoption requires: an improvement, relative to distance-only / past-direction-only, in detection
of confirmed anomalies or in confirmed ID repairs; no confirmed new ID mis-repairs; and no
degradation of normal turns or observation coverage. The unconfirmed portion and the small sample
size are always stated. If only jumps decrease while IDs stay the same or get worse, it is not
adopted as a tracking repair, and its value for prioritising review is judged separately.

## First deliverables

For the current 20 s: candidate lists for distance only, past direction, and past-and-future
direction, with comparison CG; a raw-image review of each candidate; and a classification table
of repairable / coordinate error / hold. Proceed to a local-reconnection prototype only if this
stage is passed. The existing C34 results and the in-progress analysis-by-synthesis outputs are
kept.
