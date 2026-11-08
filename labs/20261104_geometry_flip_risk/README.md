---
hypothesis: Bidirectional motion at the ends of short geometry warning intervals can improve identity connections and concentrate manual review.
stage: linker
expires: 2026-11-18
decision_rule: >
  Carry a correction rule forward only if it reduces wrong identity time in all
  seven lambda arms and introduces no reference-wrong endpoint connection.
  Carry a warning rule forward only if it contains at least 95 percent of
  sustained reference-change brackets while occupying at most 10 percent of
  post-preparation time in every arm. Otherwise report the tradeoff and misses;
  do not call unflagged intervals identity-verified.
---

# Geometry and motion warning intervals on P-1-1

Owner: P-1-1 research team; execution by Codex. Registered 2026-11-04 before
running this experiment. This continues the existing P-1-1 identity analysis.
The input is `id_row_mapping_20261102_r1`, all seven C34 arms, with the frozen
manual reference already used in that analysis. No image classifier, new pose
inference, reconstruction changes, or reference-label edits are included.

## Fixed experiment

Use the existing annotation immediately as evaluation evidence. Do not require
another human review or whole-track purity before starting. Reference labels
are excluded from warning detection and correction. Scoring uses the shared
`medaka3d.eval` calculations and the existing fixed 298--300 s anchor.

Warnings combine proximity, large steps, non-identity nearest assignments,
three-ray residual spikes, ray-count changes, local-track changes, and missing
positions. Compare proximity thresholds 5, 10, 20 mm; pad by 0.05 s and merge
overlapping/touching intervals only. Also report the existing reconstruction
margin <1.5 mm as an explicit broader ablation. Native acquisition times govern
all intervals. A long ambiguous interval remains long; it is never split just
to meet a desired clip length. Missing data remain declared separately.

At the primary 10 mm warning intervals, compare position-only, bidirectional
constant-velocity, and position-plus-velocity-consensus endpoint associations.
Only intervals <=0.75 s are eligible. Use 0.1 s of consecutive observations at
each endpoint, without crossing a local-track change or missing position;
require at least 10 samples spanning >=0.08 s. Fit linear motion in actual
seconds. A decision requires a >5 mm gap between best and second-best full
permutations; velocity must select the same permutation in both directions.
Consensus additionally requires the position assignment to agree. These are
fixed hypotheses, not optimized against P-1-1 labels. Corrections start at the
first endpoint after the warning; the warning interior stays unresolved for
accepted-identity purposes even though exploratory fixed-anchor scoring retains
all original observations. Reordering never modifies coordinates or sources.

Report local endpoint correct/wrong/unknown before and after; whole-recording
corrected/newly-broken fish-time; and results in fixed recording-time blocks
[300,900), [900,1500), [1500,end). No random-frame split, fitting, or selection
on test results. These are retrospective within-recording comparisons, not an
independent population validation. The previously inspected 932.879 s failure
is a known diagnostic case, not new validation.

For warning evaluation, include every observed reference correspondence change
and a separate sustained subset (>=0.1 matched seconds and >=10 observations
on each side). Group overlapping sustained change brackets for review, keeping
the constituent fish-level changes and gaps. Report both any-overlap and full
bracket containment. A gap limits onset localization; overlap alone is not a
captured exact event. These are reference-derived transitions, not newly
adjudicated physical swaps. Evaluate unknown reference time and encounter
coverage explicitly. No warning score is a calibrated error probability.

## Expected failure modes

Constant velocity can fail on real turns, long encounters, reflection geometry,
or contaminated endpoints. A smooth swapped trajectory may have no motion
alarm; proximity may be the only warning and may produce long intervals.
Persistent wrong physical naming outside warnings is possible until a boundary
has been checked. The reference shares upstream geometry; agreement cannot
establish independent spatial accuracy. This recording contains one trio.

## Output and status

Versioned outputs belong under the configured shared P-1-1 directory. Source,
configuration, input hashes, tables, and provenance accompany exports. Results
and the verdict will be appended after execution. No production promotion.

## Follow-up registered after the first experiment, before the second run

The first run completed on 2026-11-04. At lambda 0.767, primary warnings cover
all 21 sustained reference-change groups, but require 1,649 intervals and
401.022 s (22.28%). Ray-count churn alone occupies 50.591 s before padding.
Position bridges worsen fixed-anchor wrong time from 848.417 to 2,878.434
fish-seconds; bidirectional and consensus bridges make no corrections in this
arm. Both full direction disagreement and incomplete endpoint histories occur.
These findings do not establish that geometry cannot improve.

The second version separates observation-quality alerts from identity warnings.
Retain the same 10 mm proximity and 0.05 s padding, comparing (a) removal of
ray-count churn alone and (b) identity core: missing positions, >10 mm steps,
non-identity nearest assignment, and proximity. Both rules are fixed now and
evaluated at all seven arms with the original capture/time decision rule.
They were motivated by this recording; time blocks are descriptive checks,
not pristine held-out validation. Long close encounters are preserved whole.

Also test the historical permissive velocity condition inside C30 by setting
only `VELOCITY_NEEDS_MEDIAN_MM` from 2.5 to 0.0. Keep the remaining v4 logic,
its zone-bridge threshold, C33 geometry, and the original C34 microflip pass
unchanged. Run all seven arms. This directly tests whether the existing median
prerequisite rejects useful motion evidence; prior K-1-2 regressions are already
documented, so an improvement here would remain a P-1-1 research result. Use the
same manual-reference scorer and fixed anchor, with local reference-transition
counts alongside persistent wrong-ID time. No threshold search is included.

The second run stopped at its invariant check: 96 post-preparation matched
fish-frames at lambda 0 were incorrectly added by the comparison adapter,
which omitted the original label output's frame-level `n_views_used >= 2`
mask. Coordinate multisets were identical. Inspection also found that the
adapter passed C30's partially finite frame set to C34, whose original tool
requires all three positions finite. Version 3 restores both existing stage
contracts, reruns all seven candidates, and preserves version 2 as a failed
attempt. No warning thresholds or geometry were changed to repair this.
