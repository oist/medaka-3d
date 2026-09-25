# v1: motion proposals, with the objective and admission thresholds held fixed

This revision is specified before its real-video runs. It responds to the v0
transient's abrupt turn, where the predicted body left the fish and local mask
gradients did not relocate it reliably. The v0 three-arm outputs remain frozen.

Use Lucas-Kanade image features selected inside the preceding projected body and
independent mask. Reject texture shared by two projected bodies, poor texture,
forward/backward discrepancies, large photometric errors, and inconsistent
multiview reconstructions. The motion module uses neither current SLEAP rows nor
future poses, baseline trajectories, or identity labels. A flow-warped anatomical
node is a local approximation, not a directly observed SLEAP node.

Compute motion from the best preceding beam state, recording that proposal context.
Keep the ordinary constant-velocity proposal from every retained parent. Add up to
two joint proposals: the reliable flow-warped nodes with a translation fill for
missing nodes, and a body-translation-only version. Both keep the same preceding
ID's ordinary motion prior and use the same current SLEAP assignment and joint
mask objective as v0. This changes candidate generation, not acceptance criteria.
Flow does not fill the direct-observation mask or clear an identity-review flag.

Compare native 333519:333719 and 76620:77286 using their same registered first
poses. First run v1 with Detectron2 masks against its frozen v0 counterpart.
If motion proposals improve availability without new identity-disagreement,
shape, or motion defects, extend the fixed setting to the other observation
arms and the uninterrupted minute. Do not change success metrics based on these
development results. All three-animal identity claims still require independent
human evidence; baseline agreement and a low loss do not provide it.

The union-only renderer optimization may be included if values and pose/width
gradients remain identical and an isolated GPU comparison shows a useful speed
gain. It changes no physical approximation or optimizer setting. Also preserve
explicit permutation ambiguity when pectoral centers tie but head/tail directions
differ; geometric lexicographic tie-breaking is not identity evidence.
