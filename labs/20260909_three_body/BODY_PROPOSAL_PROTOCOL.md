# Body hypotheses after the fixed-geometry identity experiment

2026-09-09. This protocol precedes both new real-image fitting experiments.
The existing persistent minute, identity inference and comparisons are immutable.
The goal of this cycle is to distinguish missing body candidates from inadequate
body representation and inadequate scoring. It does not certify biological IDs.

The current persistent geometry was independently remeasured at thirteen native
frames: **79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394, 79396, 79405,
79547, 82719, 82720**. These are inspected development frames, not a new temporal
holdout. At 79372 two bodies explain the same top instance while another top
instance and a side1 instance remain almost unexplained. Current measurements,
not the old v1 diagnosis alone, motivate the experiments below.

## A: mask-volume positions and orientations

Use the existing calibrated 2 mm cell-centre grid and engineering bounds
[0,192] x [0,192] x [-6,186] mm. Project it through the unchanged refractive rig.
For each independent top-camera D2 instance, retain voxels whose projection is
inside it and also supported by at least one side-camera union. Use native
bilinear samples with threshold .5. An unavailable view or outside-image pixel
is unknown; an available empty mask is known background in this spatial
diagnostic. Record two-view support, three-view support and a known third-view
contradiction separately. This does not change the tracker's empty-mask loss.

Use six-neighbour components, recording singleton and degenerate PCA components.
The pool holds at most three components per top instance and twelve overall,
with content-stable round-robin selection across instances. Record omissions;
neither fine structures nor the true body are guaranteed to survive this grid.
Only a component with a defined principal axis supplies oriented starts.
This arm requires a top D2 instance; a fish missed there cannot generate one of
these volume proposals. Raw top-contrast foreground is not used in this arm.

Offer each such component to each of the three current raw bodies, including
bodies with directly supported SLEAP nodes. Preserve that body's three-node
shape. Align its eight-Gaussian volume-weighted centroid with the volume centre,
and its longitudinal direction with either sign of the principal axis at four
quarter-turn rolls. This gives at most 288 trials plus the unchanged scene.
A volume centre is not a measured pectoral node. Only one body is replaced per
trial; joint replacement of multiple bodies is outside this initial experiment.

The control has the same nominal number and raw-slot order of trials, using
isotropic quaternion rotations and Gaussian translations with sigma 12 mm,
seed `native_frame * 7 + 9101`. Both arms retain their original scene. Keep and
count all outside-box trials, exclude them without clipping or redrawing them.
Thus nominal attempt budgets match; admissible fit counts can differ and must
be reported. Do not call the comparison equal measured computation.

For each admissible trial recompute the nearest exclusive SLEAP plan, retain
that plan during the same fourteen Adam steps (learning rate .25 mm) and score
with the frozen joint objective and initial body-length priors. Reconstruct
the chosen previous state's velocity and missing counts from the saved native
backtrace, using the original recurrence, for the common motion prediction.
The unchanged scene receives the same fresh assignment and fit in both arms.
This is a new local comparison, not a replay of the old discarded proposal
plans and not a continuously tracked or reinitialized output trajectory.

Retain scores before and after fitting, all five original objective parts,
sources/support, boundary hits, proposal origin and runtime. Use a batch cap
of 24 on the RTX5070Ti. Verify the unchanged-scene control and scalar scoring
against the frozen objective. Do not adjust weights to make the image starts
win. A candidate that covers the missing region but loses on the current
objective is evidence to examine scoring, not permission to relabel IDs.

## B: a latent tail curve with all three observed-node locations fixed

Use the new curved renderer: mouth-to-pectoral remains straight; the tail
centreline is a quadratic Bezier curve with control point equal to the
pectoral/tail midpoint plus an offset. The three input node coordinates,
widths and axial/radial Gaussian scales are fixed to the original scene.
The zero-offset renderer must reproduce the frozen renderer exactly.

Optimize only the three bodies' offsets. Project each offset onto the plane
perpendicular to its current tail chord. Bound its norm by that chord length
and shrink it by one scalar if needed to keep the control point in the unchanged
box. Do not clip coordinates separately and thereby change perpendicularity.
This bounded single curve cannot express every U shape, S shape or torsion.
B adds curvature without maintaining a constant anatomical arc length; an
initial-arc-length-constrained model remains untested. Fixed chord-derived
widths can themselves be inadequate. Failure here cannot reject curved-body
representations in general.

Start at zero, take 100 Adam steps with learning rate .25 mm, and retain the
best finite fitting iterate including zero. The objective is the unchanged
weighted foreground union error on the exact original 1024 sampled pixels per
view. Optimize all three offsets jointly, with fixed geometry and rendering
scale; introduce no new node support or identity acceptance from this fit.
Record offsets, chord/arc lengths, numerical arc integration disagreement,
curve degeneracy, boundary constraint activity and fitting time.

## Spatial evaluation shared by A and B

Use the original fitting pixels unchanged. Before any optimization, construct
evaluation pools from the same D2 union: foreground, the original 31x31 dilation
ring, and all background. Exclude EVERY pixel appearing in the fitting set.
With seed `native_frame * 7 + view_index + 6137`, sample without replacement up
to 768 foreground, 768 ring and 512 background pixels. Use per-stratum total
weights 1, .75 and .25, divided by the actual nonzero count. Empty pools remain
explicit, with zero weight; a detector-empty view is neutral to fitting and
evaluation loss. Save actual pixels, strata, counts, native coordinates and
weights. Ring and all-background are intentionally overlapping sampling
strata, as in the old objective; their weighted contributions are distinct.

Evaluate every retained candidate on the saved evaluation pixels; choose
rankings using fitting scores only. Report foreground and background error
separately for each camera and frame, alongside per-instance unexplained and
shared rendered mass for selected candidates. Show all thirteen frames,
including failures. Lower held-out-pixel error indicates only improved
within-frame image explanation. These pixels share the same detector and image;
they are not new animals, new time windows or independent identity truth.

No numerical result in this cycle clears the old review flags, overwrites a
coordinate file or supplies human truth. Compare original, refitted original,
mask starts and random starts explicitly; curve-only results remain a separate
arm. A/B integration and a new native continuous run require a further
documented mechanism choice after these diagnostics. All-three-animal manual
review remains outstanding; no swap-free minute or thirty minutes is claimed.
