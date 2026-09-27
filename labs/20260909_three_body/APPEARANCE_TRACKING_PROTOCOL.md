# Comparing 3D candidates by pixel colour enrolled from the first observations

2026-09-09. In a diagnostic at 12 fixed time points, pixel colours could be compared using the
visible contribution of the 3D bodies even when a D2 region merged several fish images. Even
after the identity changed at a confirmed female endpoint, the colour correspondence with the
initial template still left a cue. However, this was a diagnostic conditioned on existing shapes,
and a slight contribution from an unseen shape cannot be used as identity evidence.

This arm is a separate experiment that builds that observation path into tracking. Candidate
generation restricted to a persistently missing mouth, and the contour, SLEAP, flow and 14-step
body fit including 3-camera refraction, are unchanged. An additional pixel-colour loss is given
to post-fit candidates to select the beam. The colour here is not put into the continuous pose
optimization; it is reflected in the candidate ranking and the next frame's state. Old results,
existing appearance templates and human identity annotations are not read into inference.

## Initial enrollment

Each of the first 166 native frames is enrolled once, from the online best state at that time.
The colour loss during this period is 0. Slots under review are excluded, and the default core6
features are collected when these conditions are met: direct support of mouth and pectoral fin
with a correspondence in each view, mouth and pectoral fin within the same independent D2 region,
and no other latent mouth or pectoral fin inside that region. After the first 166 frames the
median template is fixed.
A view can be used only when all 3 slots reach 5 samples each in that view.

An enrollment label is a provisional model identity carried by tracking up to that time from the
slot order of the first registered 3D pose. It is not a ground-truth label for sex or for the two
males, and may include early correspondence errors. Templates are not updated from later images,
and past enrollment labels are not rewritten by a later beam backtrace. Even if the final
trajectory selects a different past branch, the template is not automatically relabelled to fit
that trajectory. A view whose enrollment failed stays invalid.
No body reinitialization or external position injection is done.

For all 166 time points, the then-current online best pose, node source, direct-observation
support observed and identity review actually passed to enrollment are saved with the native
frame number. These are not substituted with poses or support sliced from the final backtrace.
Saved input states are linked to features, reasons and D2 region numbers, so that review
exclusion, direct mouth/pectoral-fin support, corresponding rows in the same view, containment
in the region and exclusion of other fish can be audited. Time points not enrolled are not
omitted either, and the first 166 contiguous frames and the length and order of each state array
are checked.

## The colour loss, fixed

The added colour here is restricted to top_cam only, because in the independent diagnostic
top-view colour information was more promising than side-view. The 3-camera contours, nodes and
flow are kept.

The union of all D2 regions is eroded once with a 5×5 ellipse, and at most 512 pixels are chosen
with a fixed seed based only on the native frame number. Detections are not truncated to 3, and
pixels are not chosen by SLEAP support or candidate pose. The 6 features of each pixel are the
r/g colour ratios, scaled S cos H and S sin H, S, and V. Because the computation differs from the
aggregated enrollment features, this is a fixed surrogate loss.

The visible_alpha of each candidate's existing refractive renderer is used. With q_f a Gaussian
kernel on colour distance and q_0 a constant for unexplained pixels, a fixed outlier component is
added to the mixture of the 3 visible fish and the unexplained part.
`PixelAppearanceConfig(sigma=0.1, background_cost=2.0, outlier_fraction=0.1)` is used, and the
negative log averaged over the number of valid foreground pixels common to all candidates is
added to the original loss with coefficient 1.0.
The values are an uncalibrated engineering setting, not identity probabilities. Invalid rays are
excluded. A view whose pixels or templates are unusable is a neutral 0, and merely hiding a fish
body does not make the valid-foreground loss 0.

Same-coloured individuals, unobserved bodies, wrong 3D shapes, reflections and errors in the
provisional identities during enrollment remain limitations. Identity review is not cleared by
minimum colour loss; the original support decisions, holds and backtrace are kept.

## Running and screening

`run_appearance_tracking.py` is a separate runner and allows only the same registered full 200,
666 and 9960 intervals. All native frames, the same initial pose once, 0 external resets. The
current persistent one-minute GPU run takes priority; no new GPU tracking for this arm starts
until it finishes.

First, on CPU, the following are checked: agreement with the original persistent output at
coefficient 0, causal equivalence during enrollment, fixed enrollment, colour ranking, and empty
views, invalid rays and holds being kept. Then the same 200 and 666 intervals are compared.
The pre-set minimum conditions are: on 200, direct not below 586/600 and accepted not below
417/600; on 666, keep accepted 1948/1998 and the two female annotations matching the initial S1.
The number of frames in which the added colour was actually used is also shown. Equivalence on
intervals where it is not used is not taken as colour effectiveness.

The equivalence check during enrollment is an independent verification that processes only the
first 166 frames with the same input and initial pose and then stops, comparing the original 21
arrays (pose, support, source, review etc.) with the original persistent. It also audits the
then-current online state passed to enrollment, and that colour scoring starts at the 167th
frame or later. This verification is not a run option that shortens this arm's full-interval
200/666/9960 CLI.

On full intervals of 200 frames or more, later colour losses may make the final backtrace select
a different past branch. Therefore agreement between the first 166 frames of the arrays returned
by a full-interval run and the final trajectory of the original persistent is not a pass
condition. This difference alone does not mean that future information was used in enrollment.
The subject of the causality audit is the saved online state at enrollment and the update order.
The one minute is compared only after both short intervals pass. This is screening on opened
development intervals; meeting the criteria or a small image loss is not treated as all 3
individuals' identities being correct.

In addition to the original 21 arrays, geometric_cost, appearance_cost, appearance_enabled and
appearance_pixels are saved per frame. costs is after the addition; loss_parts is the original 5
geometric components. The features, reasons, region numbers, counts and templates for all 166
enrolled time points, plus the then-current online pose, source, observed and review, are saved
to a separate NPZ and JSON, and inputs, sources and outputs are bound by SHA.
Automatic results, human checks and future corrections are kept separately.
