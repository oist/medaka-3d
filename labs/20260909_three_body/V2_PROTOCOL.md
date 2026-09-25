# v2: optimize the short missing episode using both temporal boundaries

Specified before running this refinement. The first target is the first >=4-frame
body-support gap in the v1 Detectron2 transient, expanded to include supported
mouth/pectoral observations for all three fish on both sides. Bound the initial
development window to at most 50 native frames. This is a known development
failure, not a held-out sequence or independent identity annotation.

The v1 forward pass remains unchanged and has exactly one external initialization.
Refinement is an offline optimization of its uncertain interval, using the same
independent SLEAP, mask and raw-contrast observations. Boundary poses are internal
estimates from that pass; they are not new external seeds, truth or ID resets.
Save a separate refinement result and retain the original complete trajectory.

Consider all six possible correspondences to the unordered scene at the right
boundary. Optimize the three bodies over all frames of the window together, with
the same refractive ray likelihood and approximate Gaussian silhouette union.
Add explicit temporal velocity/acceleration penalties to prevent a large return
jump from being interpreted as smooth physical motion. Keep supported boundary
mouth/pectoral nodes fixed; unobserved boundary nodes remain latent. Reassign
untracked node observations during optimization, exclusively within each view/node.

For each boundary correspondence use two starts: linear interpolation between
boundaries, and the original unordered scenes assigned to that interpolation,
with unsupported bodies replaced by the interpolation. This reduces, but does
not eliminate, dependence on a local minimum. Record every correspondence's
best score, geometry, support, boundary residual, displacement and source usage.
Use 256 fixed pose-independent pixels per view/frame for this development fit;
record this difference from the forward tracker's 1024 samples explicitly.

Do not clear an existing identity-review flag solely because one local optimum
has a smaller score. A cost margin is neither a probability nor independent ID
evidence. The first evaluation asks whether the scene is reconstructed more
coherently across the missing interval and whether competing correspondences
remain viable. Human confirmed raw correspondence is scored separately with the
same shared evaluator used for any method. Universal or 30-minute swap-free
tracking is not established by a successful short refinement.
