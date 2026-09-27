# Whole-time identity-assignment optimization over fixed 3D trajectories

2026-09-09. A separate experiment recorded before running the DP on real data.

## Hypothesis under test

The version that used colour to re-select even the pose candidates lowered the contour and
colour losses over 666 frames, while direct support fell from 1948 to 1946 individual-frames.
This time the completed 3D geometry is fixed and the 6 identity permutations are compared over
the whole time. The aim is to examine the contribution of image information to identity
assignment; this is not an experiment showing an increase in support rate. The original poses,
observations, sources and holds are not changed, and the proposed assignment is a separate
output.

The full 200, 666 and 9960 frames obtained by the original persistent-mouth-missing version with
1 initialization and 0 external mid-run resets are used. This identity stage is offline
processing that uses the past and the future; no new body fit or external mid-run seed is added.
It is a global minimum restricted to fixed candidates of 6 states and a first-order temporal
cost, and is not claimed to be an optimum searched over all poses in continuous 3D space.

## Inputs and initial appearance

The permutation of each state is `P[identity] = raw geometry slot`. The existing body width,
skeleton and visible contribution belong to the raw geometry and are not replaced with an
identity-dependent shape. The fixed first state is [0,1,2].
This anonymous identity order comes from the registered initial pose and is not ground truth for
sex or for the biological naming of the males.

Appearance is enrolled only from the then-current online states of the first 166 frames of the
previously run appearance version, and the frozen template, audited by recomputation from raw
images, is used. The same seed, calibration, SLEAP and configuration as the geometry run, and the
native BGR, all independent D2 masks and row order of the first 166 frames, are verified. On the
same 200/666 intervals, path/SHA agreement of all inference inputs is also required. When the 666
version's template is used for the one-minute geometry, the actual content of the first 166
frames is verified. Because the background PNG SHA differs by interval, this transfer is allowed
only in detectron2 mode, which does not use the background, and all inputs are not presented as
identical. It is not applied to background-dependent modes.

If the provisional identities at enrollment are wrong, that error may remain in the template. No
retroactive relabelling of the template to fit the final trajectory, and no selection of colours
by ground-truth annotation, is done. Old trajectories and GT values used for evaluation are not
used in the identity cost; only the required upstream geometry is read, as an explicitly
declared input.

## The first objective, fixed

The union of all independent D2 regions in the top view is eroded with the default 5×5 ellipse,
and at most 512 pixels are chosen with a fixed seed derived from the native frame. Colour-scoring
pixels are not chosen by position or by SLEAP support. The default refractive renderer is
applied once to each frame's fixed geometry, and the raw slots of its visible contributions are
mapped in the 6 ways. 8 Gaussians, width_ratio=0.06, colour sigma=0.1, background_cost=2,
outlier_fraction=0.1. The colour coefficient is 1. The colour cost of the first 166 frames is 0,
separating the enrollment period from the scoring period.

The temporal cost uses only directly supported pectoral-fin positions in adjacent native frames.
`0.2 / 3 * sum_identity(distance_mm² / 0.8²)` is added for the assignments where both time points
are directly supported. The last past position or a predicted position is not substituted for a
missing observation. The denominator is always 3; the average is not re-taken according to the
support count. No speed limit or acceleration constraint is added, as these have not been
measured.

This is an uncalibrated engineering cost. The temporal cost of a link involving a missing
observation is 0, so an assignment that avoids supported high-cost links may be favoured. This
limitation is stated explicitly, and the number of supported pairs and the selected link for
each transition are saved. A small cost obtained by exploiting missing observations is not taken
as proof of identity. In the first experiment, no additional missing-observation penalty or
post-hoc weight adjustment is made to hide this problem.

## Uncertainty saved, and evaluation

From the 6-state forward/backward minimum costs, the minimum-cost differences are computed for
each state at each time point and for all paths in which each identity is in a different slot.
Alternative paths that return to the same assignment at the ends of the interval are also
included as competitors. Numerical ties, infeasible paths, initial constraints and the absence of
a feasible alternative are distinguished. An inf due to the absence of an alternative is not
treated as strong observational evidence, and cost differences are not converted into
probabilities of being correct.

Per-frame colour cost, visible contribution, number of scored pixels, direct support and the
original review are saved alongside the proposed identity assignment. The original review is not
cleared and original files are not replaced. Unobserved positions stay NaN, and when identities
change, not only poses but also sources, observations and the original review are mapped
through the same raw-slot reference.

First, on all 200/666 frames, invariance of inputs and geometry, the exact DP solution and the
template audit are checked. That the total observation count of this method equals that of the
original method is a property of the construction, not an accuracy improvement. If the
assignment worsens on the known supported female annotations, the cause is investigated and the
method is not treated as adopted.
The one minute is evaluated to compare this method's identity assignment and ambiguity. There is
no human ground truth for all 3 individuals yet, and zero swaps are not reported on the basis of
female endpoint agreement, large model-cost differences or agreement with the original method
alone.

Same colour, overlap, nearly invisible bodies, illumination changes, template contamination,
duplicated 3D bodies and geometry that follows reflections are limitations. Failures that need
position reconstruction are distinguished from failures that identity assignment can improve.

## Comparison added before running (07:24 JST)

To isolate the colour contribution, a "motion only" DP with all colour unaries set to 0 is also
computed at the evaluation stage, using the same saved geometry, temporal cost and initial
state. No image reprocessing is needed. Its DP arrays and proposed assignment are also saved
separately, and the original fixed identities, motion only, and motion + colour are compared with
the same annotations and metrics. Weights are not tuned by looking at ground truth. Some paths
are chosen deterministically from ties, so agreement with the original identities or with another
method is not taken as grounds for correct identity.
