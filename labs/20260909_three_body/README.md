---
hypothesis: Jointly fitting three persistent 3D bodies to independent masks and ungrouped SLEAP nodes improves tracking through detection mixing and occlusion.
stage: linker
expires: 2026-10-09
decision_rule: >
  Compare the same native-rate development windows and single initial poses with
  mask loss disabled, independent Detectron2 masks enabled, and top contrast as
  an observation ablation. Preserve old runs and all seven fixed C34 baselines.
  Record candidate coverage, accepted coverage, independent female anchors,
  three-animal encounter truth when available, review burden, and runtime.
  A smoother path, smaller render loss, fixed slot count or self-generated label
  agreement does not establish identity. Freeze a promising configuration before
  new-window and uninterrupted-minute evaluation. Do not promote without independent
  all-three-animal identity evidence and full-recording validation.
---

# Three persistent bodies through refractive image formation

2026-09-09. The user explicitly requests autonomous iterative improvement toward
three fish with persistent identities, allowing a small amount of human review.
This lab is the next cycle in [the research plan](../../docs/MASK_VOLUME_RESEARCH_PLAN.md).
The completed 20260908 experiments and their outputs remain intact.

## First development cycle

Use 333519:333719 (200 frames) first, then 76620:77286 (666 frames), at the native
166 fps. These are already inspected development windows. Only the first pose
comes from the recorded C34 lambda0.767 seed; all later observations are untracked
SLEAP detections and raw images/independent Detectron2 masks. The initial seed is
optimistic because the historical baseline was computed offline over the full video.
No later baseline pose or identity annotation can reset the tracker.

Represent each fish by its mouth, pectoral and tail and a small number of oriented
Gaussian body elements. Render onto **water rays from the existing calibration**,
so pose gradients do not require replacing the calibrated refractive camera with
a pinhole camera. This is an approximate silhouette renderer, not a photorealistic
Gaussian-splatting reconstruction or a validated anatomical mesh.

Fit all three bodies jointly. The foreground term measures their union, including
unexplained observed foreground; it is not three independent line-to-mask scores.
SLEAP assignments are exclusive within each view/node and may be missing. Detection
rows do not carry identity. Body lengths, smooth motion and non-duplication provide
constraints; their satisfaction is not independent identity truth. Foreground from
reflections and segmentation failures must remain possible outliers.

Retain competing temporal correspondences instead of treating every recovery as
permanently resolved or permanently unresolvable. Output the latent pose separately
from independently supported nodes and accepted identity claims. Score gaps and
review separately. A missing detection must not update an appearance template as
though a new identity were observed.

## Evidence, review and iteration

GPU is authorized. Reuse the preserved Detectron2 CUDA environment and current
PyTorch environment with explicit input hashes at their NPZ/video boundary. Keep
the full selected native-rate raw frames losslessly. Known-empty inference and
missing/unprocessed frames remain distinct. Segmentation scores are not calibrated
pixel probabilities and mask failure is not a hard veto on SLEAP.

Use a model-agnostic three-view review clip to obtain the missing independent
three-animal correspondence when necessary. There is confirmed female-click truth
and some partial male-related annotation in the archives; none is automatically
treated as a verified three-animal correspondence across this encounter. Predictions
or geometry-based suggestions must not become ground truth.

Each revision records its diagnosed failure, the changed mechanism, code and input
hashes, and the same-window ablation. New holdout windows are not opened to tune the
candidate. Human corrections are a separate, explicitly counted evaluation arm.
Coverage gates remain 90% overall and 80% per fish, alongside zero observed accepted
identity errors on independently adjudicated intervals; those finite tests do not
prove universal zero swaps.

The first implementation can fail this gate. Preserve the failure and change the
identified mechanism, rather than changing the success metric or calling rendered
agreement identity accuracy. Continue to the uninterrupted minute and approximately
30 minutes when the corresponding earlier gates have supporting evidence.
