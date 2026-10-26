# Within-trial identity: segmentation, appearance and block association

Date: 2026-09-30. Status: literature and local-code review; recommendations are
inferences to test, not measured medaka improvements. No tracking run, package
installation or method promotion was performed.

Scope: three medaka, synchronized three-view video, existing SLEAP poses and
refractive calibration. One individual must retain one ID throughout a trial;
missing or unresolved intervals are allowed. Evaluate usable time after 300 s,
as specified in the [delivery decision](2026-09-30-dataset-delivery.md).

## Recommendation

Segmentation is useful, but it is not the only route to identity. The strongest
practical candidate for this project is a combination of **reliable local 3D
continuity, appearance evidence aggregated over multiple images per segment,
and consistent identity assignment across the whole trial**. Use masks where
they improve the appearance evidence, rather than assuming mask tracking itself
proves identity.

The most directly relevant ready-made block-matching baseline found is
**idtracker.ai v6 with idmatcher.ai**. The most useful alternative to compare
locally is a **pose-aligned image embedding**, with and without mask-based
background suppression, attached to the existing identity-assignment machinery.
There is no evidence yet that either beats the current method on this dataset.

## Three different meanings of segmentation-based identity

| Meaning | What information it supplies | Implication for separated blocks |
|---|---|---|
| Binary silhouette / mask shape | Outline, area, width, shape | Potential morphology cue; changes with pose and projection. Does not retain texture or color. |
| Pixel appearance inside a mask | Body intensity, color, texture and shape | Suitable input to an identity classifier or embedding. Segmentation is preprocessing, not the identity decision. |
| A video segmentation model's object ID | Temporal mask propagation from a prompt or detection | Useful local continuity evidence. A persistent output label alone does not validate the animal after an ambiguous crossing. |

Mask overlap has little identifying power when an animal has moved far during a
gap. This is a reason to test appearance and global constraints, not proof that
every appearance model will work. Also distinguish the *clustering silhouette
score* reported by idtracker.ai from the animal's binary silhouette.

## Primary-source evidence and limits

### idtracker.ai v6 and idmatcher.ai

The new idtracker.ai learns visual representations from same-fragment positive
pairs and temporally coexisting-fragment negative pairs. Its aligned,
background-suppressed identification images retain grayscale animal intensities;
it is not a silhouette-only method. Appendix 5 directly tests disconnected blocks:
14 zebrafish/fruit-fly videos were split with a 200-frame gap, tracked independently,
then matched. The authors report no block identity-matching errors in that test.
This does not establish minute-scale gap performance or medaka accuracy.
[eLife paper, Appendix 5](https://elifesciences.org/articles/107602v1).

idmatcher.ai cross-identifies the image collections in both directions and solves
a one-to-one assignment. Its documentation requires compatible image sizes and
recommends matching segmentation settings and software versions. Direct use
requires idtracker.ai session artifacts; it is not a generic SLEAP-fragment
importer. Assignment margins and agreement remain evidence requiring validation.
[Official documentation](https://idtracker.ai/latest/user_guide/idmatcherai.html).

For a continuous trial, jointly learning one appearance space from its reliable
regions is also worth comparing with separate models per block. Independent
block training is not a requirement imposed by the scientific question.
This is a design inference; the official FAQ allows multiple files to be treated
as one continuous recording.
[Official FAQ](https://www.idtracker.ai/latest/user_guide/FAQs.html).

### Closely related three-camera fish workflows

O'Shaughnessy et al. combine SLEAP 3D skeletons with top-view idtracker.ai identity.
They propagate identity through gaps using 3D skeleton correspondence, retaining
segments when the final assignment agrees with the identity after the gap and
excluding failed segments. This is a close precedent for supported IDs with
abstention, on two zebrafish. Endpoint agreement alone can miss two internal
swaps; that last limitation is our inference, not the paper's validation claim.
[PRX Life 2024, Methods M–O](https://journals.aps.org/prxlife/pdf/10.1103/PRXLife.2.043006)
and [author code](https://github.com/liamshock/Dynamics_of_dominance).

Deligkaris et al. publish 173 five-hour zebrafish-pair recordings using SLEAP,
idtracker.ai and three-camera correspondence, retaining N/A values for rejected
observations. This demonstrates a related dataset-delivery workflow; it is not
proof of zero identity error or of transfer to three medaka.
[Scientific Data 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066468/)
and [author code](https://github.com/deligkarisk/Zebrafish_3D_Tracking_Workflow).

The 3D-ZeF work associates multi-view tracklets using geometric and temporal
compatibility. It supports an appearance-independent route where observations
bridge a gap, but its short benchmark sequences do not demonstrate whole-trial
identity after long simultaneous occlusion.
[Paper](https://arxiv.org/html/2006.08466)
and [author code](https://github.com/mapeAAU/3D-ZeF).

### Pose-guided visual re-identification without masks

DeepLabCut's re-ID transformer learns from visual features associated with pose
and tracklet-derived triplets, then contributes appearance similarity to global
tracklet stitching. It is not merely a comparison of skeleton coordinates. This
supports testing a separate SLEAP-guided image encoder while preserving existing
pose detections; using DLC's implementation directly entails an integration cost.
[Nature Methods paper](https://www.nature.com/articles/s41592-022-01443-0)
and [official API](https://deeplabcut.github.io/DeepLabCut/dev/3.0/reference/deeplabcut/pose_tracking_pytorch/apis/).

SLEAP's development documentation also describes embeddings combined with
geometry, and distinguishes output tracks from global identities. Its reported
tests are gerbils, not medaka. Locally, package metadata reports sleap-nn 0.3.1
and sleap-io 0.9.2; `inference/embedding.py` is absent and the installed tracker
has no `appearance_weight` option. Therefore the development guide is not a
drop-in command for this environment.
[SLEAP development guide](https://nn.sleap.ai/dev/guides/embedding-tracking/).

### Other alternatives and their priority

| Candidate | Evidence / capability | Priority for this project |
|---|---|---|
| Trial-wide crop clustering | Fang et al. use bounding boxes, a known animal count and self-supervised visual learning; no pixel masks required. | Exploratory alternative. Evidence is pigeons/calves, not fish; initial pseudo-matches may be wrong. |
| SAM2Long / SAM 3.1 | Long-video mask propagation, multiple hypotheses or improved multi-object memory. | Candidate for better observations/local tracking; no reviewed medaka evidence that it solves trial-wide identity by itself. |
| Supervised individual-ID head | SLEAP supports jointly learned pose and visually distinguishable identities. | Useful if independently supported labels can be obtained; not an automatic solution for nearly identical males. |
| 3D morphology | Body-part lengths/width and multi-frame shape statistics can complement images. | Cheap auxiliary test. First show between-fish differences exceed measurement and pose variation. |

Sources: [Fang et al., January 2026 preprint](https://arxiv.org/html/2601.09663v1),
[SAM2Long paper](https://arxiv.org/abs/2410.16268),
[SAM 3 official repository](https://github.com/facebookresearch/sam3),
[SLEAP supervised-ID guide](https://nn.sleap.ai/latest/guides/supervised-id/).
The morphology row is a project-specific proposal, not a published medaka result.

Fang's strongest accuracy and cheapest frozen-backbone configurations should not
be conflated: its table reports 97.6% versus 86.6% for full versus frozen ResNet18
on one pigeon sequence. The author-linked code/data repository exists, but the
exact reproduction entry point was not verified in this review.
[Author-linked repository](https://huggingface.co/datasets/tonyFang04/8-calves/tree/main).

A recent study also tests transfer between portions of the same recording with
controlled visual changes and finds reduced identity consistency. This supports
testing actual separated time blocks and image-condition changes. Its consistency
metric and transfer protocol are not a direct test of idmatcher.ai's bidirectional
block matching.
[iScience 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469807/).

## What the local repository already contains

- [Mask appearance features](../../labs/20260909_three_body/appearance_diagnostic.py)
  use six RGB/HSV statistics from mask-core pixels.
  [The appearance tracker](../../labs/20260909_three_body/appearance_tracker.py)
  enrolls provisional templates over an initial 166-frame prefix, then freezes
  them and reranks candidates using top-camera appearance. This is not a learned
  multi-view re-ID model or independently established identity anchoring.
- [Identity DP](../../labs/20260909_three_body/identity_dp.py) already evaluates
  six permutations on fixed geometry. Its adjacent-motion version records a
  missing-cost avoidance defect; do not simply reuse that objective unchanged.
  [Memory DP](../../labs/20260909_three_body/identity_memory_dp.py) preserves
  last observed pectoral evidence across gaps but remains an unvalidated cost
  model, not a probability of correct identity.
- [Prior findings](../PRIOR_FINDINGS.md), especially sections 2.7, 2.9 and 4.4,
  record earlier idtracker/contrastive/SAM2 failures, label circularity and
  train/test leakage. These archived outcomes are setup-specific evidence to
  recheck, not a universal ranking that rules out appearance methods.
- [Identity evidence inventory](../../labs/20260909_three_body/IDENTITY_EVIDENCE.md)
  records partial male-related evidence but no confirmed independent all-three
  encounter correspondence set within its inspected scope. Do not state that
  no male annotations exist at all.

Thus the next experiment should improve the identity evidence and evaluation,
not rebuild segmentation or six-permutation assignment from scratch. The physical
color descriptors in [cohort.py](../../src/medaka3d/cohort.py) were assigned during
close examination and cannot be assumed readable in an individual video frame.

## Bounded comparison proposed before integration

Keep the SLEAP detections, 3D geometry, source rows, candidate fragments and
evaluation windows fixed. Compare the existing geometry/memory baseline and
existing color baseline against one learned visual identity model. For that
model, compare pose-aligned crops with versus without mask-based background
suppression. A binary-shape-only diagnostic can establish whether the silhouette
itself contributes; it need not become another full pipeline.

Build a per-trial gallery from several independently supported clean intervals
covering varied poses, depths and lighting. Aggregate temporally spaced images
within a block; do not count hundreds of adjacent frames as independent votes.
Begin with camera-specific features and combine their identity evidence through
verified geometric correspondence; top and side appearances are not automatically
interchangeable. Reliable same-fish cross-view pairs may support later learning.

Do not make every pair of same-frame detections a negative example: reflections
or duplicated detections can represent the same fish. Do not make uncertain
cross-gap matches positive training labels. Low-quality/missing masks remain
unknown rather than deleting an otherwise valid pose observation.

Use early/late and well-separated time blocks with temporal buffers for fitting,
threshold selection and held-out evaluation. Review all-three-fish correspondence
using source video context, including accepted and rejected links. Do not score
against the tracker labels that generated the training targets. Where humans
cannot establish a correspondence, mark it undecidable, not correct.

The comparison must report wrong accepted block links, unresolved links,
trial-ID-supported fish-time after 300 s, longest/typical usable intervals, and
review/computing effort. Male–male association must be visible separately. Compare
retained time at a common error criterion, rather than maximizing assignment rate.
A small audit with no errors does not certify a low population error rate.

Only intervals connected by accepted correspondence evidence to a documented
common trial identity anchor/gallery count as trial-ID-supported time. Locally
correct but unconnected blocks do not count. The same trial ID cannot be assigned
to two physical fish at the same time; absolute male naming is still a separate
claim from consistent anonymous trial IDs.

For accepted three-fish blocks, six permutations are tractable. The existing DP
enumerates complete permutations of three raw slots; missing/unassigned identity
and open-fragment handling require additional logic or an export/acceptance
wrapper, not merely switching on an existing option. Reuse the supported machinery
with that adaptation and inspect competing solutions. Geometric constraints help
short gaps and surviving views;
long all-view gaps need persistent appearance/morphology or independent identity
evidence. If neither continuity nor distinguishable individual evidence survives,
the correct correspondence is not identifiable from these inputs alone.

Freeze this limited comparison before running it. Promote only a measured gain
in supported trial-wide identity time; if the visual signal does not separate
the males, prioritize review of the few consequential joins and retain abstention
instead of starting another open-ended architecture search.
