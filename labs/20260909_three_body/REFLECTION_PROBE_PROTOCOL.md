# Fixed reflection-path probe, registered before real rendering

This is a CPU-only geometric response diagnostic, with no fitting or identity update.
The opened development frames are fixed to 79316, 79318, 79322, 79355, 79370,
79372, 79381, 79394, 79396, 79405, 79547, 82719 and 82720. The primary previously
noticed region is top-camera D2 instance row 3 at 79381; it is a hypothesis about
a reflected image, not a true fish/identity label. All frames, views, and instances
are reported, including contradictory changes.

Inputs are the sealed persistent_mouth_continuous_one_minute_v1 geometry,
its original D2 masks, SLEAP and calibration, and the already saved spatial
sample coordinates from curve_body_spatial_13frames_v1. Fixed mouth/pectoral/tail
coordinates and all 8 Gaussian bodies per slot are reused with width ratio .06,
opacity .85, and the original axial/radial scale construction. No coordinates,
wall, opacity, indices, or detection parameters are optimized. Review/support
flags remain unchanged. Biological GT and later baseline data are not opened.

The old renderer is checked against an independent NumPy float64 implementation
of its peak response on [0, infinity), with maximum alpha/union absolute error
at most 1e-12. For each existing Gaussian, depth becomes
clip(d' Q (c-o)/(d' Q d), 0, maximum_distance_mm). Its response remains
.85 exp(-.5 (c-o-depth*d)' Q (c-o-depth*d)); Gaussian unions remain symmetric.

The nominal physical water box is [0,192] x [0,192] x [0,186] mm. These are fixed
plane hypotheses in the existing calibration gauge, not newly surveyed planes. The engineering
z=-6 allowance is never used for reflection paths. Fixed bodies are not clipped;
any original points or centers outside the physical box are explicitly counted.
The direct segment stops at its first nominal wall. Only a unique first y=192
wall allows the specified reflection paths. Edge hits are retained as direct
finite rays but cannot create a reflection. `reflection_paths` supplies exact
branch validity, reason, next exit, and maximum geometric (not optical) distance.
The inner wall is water/glass at y=192; the outer wall is glass/air at y=196,
with a 4 mm parallel glass layer. Other/multiple reflections and roughness are
not added. Each view uses its own recorded water/glass/air refractive indices.

For branch b and body f, first compute alpha_bf by union over its 8 Gaussians.
The branch-unit combined response is 1-product_bf(1-alpha_bf). The weighted
response is 1-product_bf(1-w_b*alpha_bf): direct weight 1; reflection weights
are supplied by the Fresnel path helper. Its outer weight preserves s/p
polarization through the outward transmission, reflection, and return
transmission. The helper assumes unpolarized light at the existing water-ray
origin; camera-entry losses/polarization are unmodeled. Invalid branches have
zero response. No cross-branch depth/occlusion sorting is performed. These
unions combine an engineering alpha with ray-power weights: they are not
radiometric image likelihoods, calibrated segmentation probabilities, or ID
confidence. Unit response is an upper bound only on the same branches with
weights in [0,1], not on actual brightness or true visibility.

Contrasts are old unbounded direct; finite direct; finite direct plus inner;
finite direct plus outer; finite direct plus both. The latter three are reported
with unit and Fresnel weights. Added response is measured on all D2 foreground
pixels, each individual mask, and its fixed 5x5 elliptical eroded core. Masks
may merge animals, miss them, or include reflections; no instance-to-fish
assignment, detection filter, or new acceptance gate is made.

The original 1024/view fitting pixels are retained only as an unchanged
diagnostic sample. Spatial evaluation excludes ALL those coordinates and uses
the frozen body_pixel_evaluation sampler: seed frame*7+view_index+6137, up to
768 foreground, 768 31x31 dilation-ring, 512 all-background pixels, without
replacement within each pool; stratum weights 1, .75, .25 normalized by the
actual counts. Ring and all-background intentionally overlap. The same saved
sample coordinates/targets/weights are checked exactly before rendering. D2
sampling uses dummy images for native shape only and never their content.
Original calibrated-ray validity defines evaluation denominators: rejecting a
physical path must not silently discard its foreground target. Missing/empty
detector views remain neutral. Per-view/stratum squared error, response outside
D2, new foreground support and path-reason counts are all saved. No test pixel,
frame or successful case selects a fitted model: there is no fit.

The primary visual is 79381 top-camera raw plus D2 contours and direct/inner/
outer coverage. The BGR image is sought at its exact native frame and checked
against its saved SHA. Display response uses a fixed 4-pixel native grid, clearly
labeled; a detail panel uses the primary mask bbox plus 140 native pixels of context.
Quantitative foreground and evaluation measurements use the full
registered pixel sets. Three-slot colors are arbitrary, not biological labels.
Input/source/snapshot/output hashes and a successful separate completion seal
are required. Parent tracking and source inputs are immutable.
