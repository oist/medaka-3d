# Whole-time identity-assignment optimization that retains the direct observation before a gap

2026-09-09. Conditions recorded before running this method on real data.

## Problem found in the previous method

The adjacent 6-state DP over fixed geometry completed 200 frames and its CPU recomputation from
raw images agreed, but in an early interval where S2's pectoral fin is missing it chose a
1-frame exchange-and-return of the S0/S2 assignment 6 times. The number of identities whose
movement is evaluated dropped from 2 fish to 1, avoiding the movement cost of the directly
observed S0. All of this was in the colour enrollment period, with unary 0. This is a flaw of an
objective that evaluates movement only at the two adjacent ends, not a colour or GPU numerical
error. This version is not extended to 666 frames or the one minute, and the original results are
kept.

## Inputs and verification conditions kept

The 3D geometry of the original persistent-mouth-missing version, direct pectoral-fin
observations, independent D2, refractive calibration, raw images and the causal appearance
template enrolled from 166 frames are used. Full native intervals of 200, 666 and 9960, 166 fps.
The original geometry had 1 initialization and 0 external resets. The first identity permutation
is [0,1,2] only. The pectoral fins of all 3 fish are required to be directly observed at the first
frame; a missing observation at the first frame is not filled with a latent value.

Top view, at most 512 pixels, 8 Gaussians, width_ratio=.06, colour sigma=.1, background_cost=2,
outlier_fraction=.1, colour coefficient 1. The colour unaries of the 166 enrollment frames are
all 0. No identity fixing or manual seed is added after the first frame. Raw-image scoring uses
the existing verified pure helper, and the geometry is not refit.
GT and the identity sequences of completed reference methods are not input to the objective.

## State, and movement across a gap

Each candidate's state is the current permutation P and the ordered triple of the events at which
each identity was last directly observed. P[id]=raw geometry slot. An event is identified by
native frame offset and raw slot. Only when the raw pectoral fin currently assigned to that
identity is directly supported is the movement cost from the last event to now added and the
event updated. While missing, the past event is kept and no observed value is created.

With d the number of frames since the last direct observation and dx the 3D displacement, the
per-identity cost is
`0.2 / 3 * (sum(dx**2) / (0.8**2 * d) + 3 * log(d))`.
This is twice the negative log density of a 3D isotropic Brownian transition, with the common
constants removed. Because the 3 first-frame observations and the total number of later
observations are common to all paths, the common term per observation event does not affect the
comparison. At d=1 it equals the direct adjacent movement cost of the previous method.

This is not a model validated as fish motion. It includes no constant-velocity or acceleration
prediction, and no probabilistic model of the support-dropout process or of occlusion. Weights
and scale are not changed after seeing these results. The displacement of an identity not
re-observed by the end remains not evaluated, so the last event and missing age are stated
explicitly, and that identity is not confirmed as correctly kept through the missing period.

## Exact path comparison and computational limits

Paths that reach the same P and the same ordered event history have identical subsequent
computation, so they can be merged into the minimum prefix. When all 3 fish are directly observed,
the current events coincide and the states merge naturally into 6. Cumulative costs and past
paths are carried over. This merging is not reinitialization.

To compare alternative paths, all states and transitions of each layer are kept and
forward/backward is computed. The minima over all event histories with the same P are collected
to obtain the T×6 minimum marginal costs and the per identity→slot differences. Out-and-back
paths that return to the same P at the ends are also included as alternatives. Initial
constraints, no alternative path, and numerical ties are distinguished, and differences are not
presented as probabilities of correct identity.

Limits: 100000 states per layer, 1000000 states in total, 6000000 transitions. The number of
reachable states is first measured from the support mask alone. Exceeding a limit stops the run
as an incomplete exact computation; no silent beam pruning or temporal decimation is done. If a
change is needed, the reason and new conditions are registered separately.

## Comparison and next stages

On small sequences, all 6-permutation paths are enumerated and the objective value, MAP and the
minima conditioned on P/identity at each time point are checked against them. The checks confirm
that the result equals the previous method when all 3 fish are supported in every frame, that
different missing histories are not merged early on the same current P alone, and that past
alternative paths can still be evaluated after merging on all 3 fish being supported.

On the real 200, how the earlier 6 out-and-back exchanges that exploited missing observations are
handled, and each identity's re-observation links, are compared. All original geometry, sources,
support, review and NaN are kept, and only the identity assignment is a separate output. A
motion-only comparison (all colour unaries 0) is also computed under the same conditions to
isolate the appearance contribution. After the implementation and objective audit on 200, 666 is
evaluated, and if the known supported female endpoints do not worsen, the one minute. Independent
ground truth for all 3 individuals has not been received, and identity correctness or achievement
of the final goal is not claimed on the basis of endpoint agreement, preserved support counts or
fewer out-and-back exchanges alone.

Duplicated bodies, fits to reflections and geometry that cannot represent the fish itself cannot
be fixed at this identity stage.
