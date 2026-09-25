---
hypothesis: Independent segmentation masks backprojected through the calibrated refractive rig supply useful 3D candidates beyond SLEAP keypoints.
stage: associator
expires: 2026-10-08
decision_rule: >
  First gate is feasibility only: reproduce independent mask inputs with provenance,
  verify CPU/CUDA support parity, retain missing-view semantics, and inspect all
  predetermined frames for supported fish and ghost components. Do not force exactly
  three components, identify centroid with a SLEAP anatomical node, or claim ID accuracy.
  Continue to explicit three-body hypothesis comparison if the mask evidence and
  refractive sampling are usable. Production promotion requires independent all-fish
  identity evaluation, accepted coverage and continuous tracking under the research plan.
---

# Independent masks through refractive projection

This is the first, bounded feasibility gate of the
[research plan](../../docs/MASK_VOLUME_RESEARCH_PLAN.md), registered before inspecting
the volume results. It does not update the completed generative tracking experiment.

The frame-selection source is an existing **unfiltered Detectron2** pilot at
frames 132000:137000. Here unfiltered means before SLEAP/track-based postprocessing;
the cache can still have detector score/NMS filtering with incompletely recorded settings.
Ten frames are selected by equal spacing over the cache's shared available extent,
using frame indices only. Requiring stored groups in all three views can exclude
failed detections, so this sample cannot estimate detector recall.
All three raw camera images are extracted at exactly those indices. This is a
cache-selected development diagnostic, not a held-out identity benchmark.

The preserved dedicated Detectron2 environment passed RTX 5070 Ti CUDA operator
smoke checks. Before volume evaluation, run fresh inference for these 30 images
with the preserved top/side models, a recorded score threshold of 0.5 and full-image
inputs. Use these fresh masks for the main volume diagnostic; retain the old masks
only as the selection source and a separately labelled historical artifact. Record
the resolved configurations, weight hashes and runtime, without retraining.

The first volume uses a 2 mm grid, x/y in [0,192], z in [-6,186] mm. The 6 mm floor
tolerance is the previous engineering sensitivity setting, not a measured floor
correction. Thin bodies can fall between grid points; absence of a component is
not proof that a fish is absent. Finer sampling is a later sensitivity test.

The union of all detected instance masks is sampled through `RefractiveCamera.project`.
Mask identity, count and detector scores are retained in the input. The first
diagnostic aggregates foreground support by view; it is not a likelihood calibrated
for Detectron2 and does not claim the masks are statistically independent.
Two-view and three-view support volumes are reported separately. Neither is a hard
veto on SLEAP. Missing views remain unknown. Connected components are candidates
and can include multiple fish, fragments, reflections or cross-view ghost volumes.

No SLEAP point, existing trajectory, female annotation or biological ID is used to
construct the volume. Segmentation training provenance, inference thresholds and
any source uncertainty are recorded in the input manifest. The next gate can
compare these independent observations with SLEAP and render three explicit bodies;
the present implementation alone does not perform that comparison or track identities.

Outputs live in `work/mask_volume/`, with source hashes, frame indices, masks,
summaries and review images. Synthetic geometry, missing-view behavior and CPU/CUDA
parity are checked separately from the qualitative real-image inspection.

The first float32 sampler run is retained as `feasibility_v1`. It exposed one
CPU/CUDA threshold disagreement in each of three of the twenty frame/mode cases.
The next numerical revision changes sampling precision, keeping all input images,
mask models, grid cells and thresholds fixed. It is a numerical reproducibility
check, not a second independent biological validation.
