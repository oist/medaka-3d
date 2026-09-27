# Next cycle: evaluating three-fish 3D hypotheses from independent masks

2026-09-08. A new development plan drawn up after the previous cycle's results were unsealed.
The user accepts a small amount of human checking and correction, and computation on an RTX 5070
Ti. The goal of this plan is usable trajectories that keep the three fish's IDs, not good image
reproduction.

The first 10-frame, 3-view feasibility run is done.
[The result and next decision](../labs/20260908_mask_volume/FEASIBILITY_RESULT.md) records new
Detectron2 inference, refraction-aware support regions, the comparison with top background
subtraction, and a fix to GPU numerical precision.
It has not yet reached a comparison test of keeping the three fish's IDs.

The previous [contiguous 1-minute result](../labs/20260908_generative_tracking/REACQUISITION_RESULT.md)
had a pectoral-fin observation rate of 99.76% and an acceptance rate of 3.56%. Neither the
observation rate nor the acceptance rate is an ID accuracy. The previous cycle used background
subtraction and RGB histograms, not independent Detectron2 per-individual masks. It is not a test
of the present proposal.

## Hypotheses and implementation order

1. **Secure independent image observations.** Store SLEAP's 3 points, Detectron2 per-individual
   masks, and top_cam brightness / background subtraction separately. Do not give the mask
   detection existing 3D IDs or trajectories. Errors from occlusion or reflection in the same
   image can be shared, so statistical independence is not assumed. Retain the cases where two
   fish overlap in one mask, where one fish splits into several, missed detections, and
   reflections.
2. **Build refraction-aware 3D support regions from the masks.** Project candidate points inside
   the tank into each camera with the existing refraction model, and sample support inside and
   outside the masks. This is not a method that triangulates by matching contour points to each
   other. A missing observation is treated as missing, and a hard three-view AND does not delete
   all candidates. The number of connected components of the support region is not necessarily
   the number of fish, and no ID is assigned here.
3. **Compare image-generative models that represent the three fish explicitly.** Start by
   representing each fish with a few ellipsoids / a bending body, and check the silhouettes
   projected into the three views against the real masks. Besides foreground being present
   where a fish projects, also evaluate how much of the image foreground the three fish explain.
   Handle pixel-to-fish attribution, occlusion by nearer objects, and reflection outliers, and
   deal with solutions in which two IDs converge on the same fish. Mouth, pectoral fin and tail,
   body shape, movement and optical flow are added as separate support information.
4. **Keep several correspondence candidates over time.** An instantaneous mask match alone cannot
   distinguish permutations of the three IDs. Retain each individual's last supported position,
   orientation, velocity and shape, and re-evaluate candidates using images from before and after
   a close approach as well. Do not keep training a wrong assignment into the appearance
   template. Initialisation happens once, at the start. Recovery from a hold does not use
   ground-truth IDs or reinitialisation by an existing method.
5. **Once the shape representation is confirmed to be insufficient, compare per-fish 3D
   Gaussians.** Limit the number of Gaussians and keep the constraints of attribution to the same
   fish and of deformation. Do not use a standard pinhole renderer as is; implement and validate
   either an approximation via the Jacobian of the refractive projection or rendering along
   refracted rays. Free image optimisation of the whole scene is not, by itself, to be called ID
   estimation of biological individuals.

## Candidates compared and narrowing down

| Candidate | Information added | Question that decides adoption |
|---|---|---|
| A | stored SLEAP, the existing 7 arms, the previous prototype | control on the same interval and initial conditions |
| B | 3D support from independent masks, assignment of observations to the same object | can it separate and recover the correct fish when point detections are missing? |
| C | silhouette generation including the three fish's finite volume and occlusion | does it reduce duplicate explanations and false correspondences from reflections? |
| D | persistent per-individual Gaussian shape and appearance | does it improve the shape and appearance errors left by C, in terms of both ID and compute time? |

Proceed in order from B, comparing one change at a time, each matched to a cause. Do not
introduce everything at once. On the development interval, compare masks only, with SLEAP, with
motion, and with each view left out. Reprojection error, mask agreement and smoothness are for
diagnosing causes and do not replace the ID pass criteria.

## External evaluation and pass criteria

- Top priority: **accepted wrong IDs** against independent three-fish annotation. F–M and M1–M2
  are separated. Matching entry and exit can still miss an out-and-back swap in between, so the
  three-view video is also checked during close approaches.
- Report **acceptance rate, missing time, rate of simultaneous output of all three fish, number
  of held intervals, and time spent on checking** together. A method that holds everything does
  not pass. Automatic output and human-corrected output are scored separately.
- The interim engineering promotion criterion: on independently confirmed intervals, 0 accepted
  wrong IDs, a fish-frame acceptance rate of 90% or more, and 80% or more for each fish. This is a
  condition on a finite sample and is not a guarantee of zero swaps in unconfirmed intervals. The
  number of scenes acceptable for human checking has been asked of the user; no answer is not
  treated as approval.
- The previously unsealed 666/200fr, the 4 validation intervals, and the 1 minute 76620:86580 are
  treated in the next cycle as known development / regression tests. They are not again called
  "unseen validation".
- Once a candidate qualifies for promotion, select and fix new close-approach / occlusion /
  ordinary intervals without looking at the candidate's output. Exclude known intervals, and do
  not split a close-approach event from its before/after context. No random split of adjacent
  frames. Save the selection list, input hashes, settings and code before unsealing.
- Proceed in the order: pass on the new short intervals → a contiguous 1 minute with the same
  settings (no reinitialisation) → 30 minutes. If held intervals are looked at to make
  improvements, move those intervals to the development side and secure a separate next unseen
  evaluation. Validation within one recording is not interpreted as generalisation to other
  individuals or other days.
- Human annotation does not adopt a candidate's lines or IDs as ground truth; it records the fish
  correspondence on the raw images. Evaluation annotation is not fed back into inference. An
  experiment that includes operational human corrections records their number and positions.

## What proceeds autonomously and what needs human judgement

Implementation, GPU use, cause diagnosis on short intervals, comparisons, rejection of failed
candidates, and record-keeping proceed autonomously. Changing the objective or the ground-truth
labels to fit one's own predictions does not count as improvement. Human judgement is needed for
hard-to-distinguish male correspondence, the checking burden per 30 minutes, how to leave in the
deliverables the points that cannot be confirmed even from video, and facts about the recording
setup such as the physical tank-floor position. The first test does not mix in a renewed
widening of the floor tolerance or a re-optimisation of the calibration.

The first run is a mask-support-region feasibility on 10 frames selected by time alone from the
old Detectron2 pilot cache 132000:137000. This is a development diagnostic chosen from the data
assets, not unseen validation of ID. Audit that the Detectron2 runtime environment and weights
still exist, and until masks can be obtained, do not report background-subtraction tests as a
Detectron2 evaluation. The GPU is used for mask sampling and batch evaluation of candidates;
measure numerical agreement with the CPU and the wall-clock time including loading.

## Gaussian Splatting: evidence and interpretation

[Dynamic 3D Gaussians (3DV 2024)](https://dynamic3dgaussians.github.io/) keeps the Gaussians'
appearance, size etc. fixed and estimates translation and rotation, obtaining temporal point
correspondence through local rigidity constraints. A blanket denial such as "Gaussian Splatting
does not track" is therefore inappropriate. On the other hand, it has not yet been demonstrated
that it keeps the IDs of three medaka seen by three cameras through refraction.

[GaussianObject (SIGGRAPH Asia 2024)](https://gaussianobject.github.io/) uses masks and a visual
hull as the structural initialisation, even from 4 images, and optimises Gaussians. It is image
reconstruction of a static object, not validation of moving-fish IDs. Images filled in by a
diffusion model are an inferential prior and cannot serve as independent observations that verify
a fish's identity. Taking mask-based shape initialisation and a persistent object representation
from it is an implementation-level inference made here.

[3DGUT (CVPR 2025)](https://research.nvidia.com/labs/toronto-ai/3DGUT/) projects representative
points of each Gaussian through a nonlinear camera model to estimate its 2D extent, and also
supports secondary rays for reflection and refraction. It is a concrete reference for connecting
the existing refractive projection to Gaussian rendering.
[Through the Water (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/38197) is a
Gaussian reconstruction that handles reflection and refraction at the water surface. Neither
demonstrates keeping the three fish's IDs here, but both are considered as primary sources for
implementing rendering beyond the usual pinhole assumption.
