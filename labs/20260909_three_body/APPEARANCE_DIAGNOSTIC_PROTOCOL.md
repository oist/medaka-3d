# Colour within independent instances: a diagnostic conditioned on saved slots

Conditions fixed before computing on real data, 2026-09-09. The target is the already-opened
v1_interval_detectron2 (76620:77286, 666 native frames) and the same D2 staging. Model,
thresholds, tracking and identity assignment are not changed. The saved slots are provisional
labels, not ground truth for the physical fish.

Templates use the 166 frames of 76620:76786; validation uses the 50 fixed time points
76786,76796,...,77276. This is not a tracking evaluation over 500 contiguous frames. Every
requested time point across all 3 views and 3 slots is included in the denominator, and
unobserved, view unavailable, zero mask, non-unique, shared with another fish and insufficient
core are tallied separately. Slots under review are not excluded either; the review flag is
saved separately.

In each view, cases where mouth and pectoral are directly supported and both projected points lie
within the same D2 instance are examined. Coordinates are rounded to native pixels (NumPy rint),
and no distance tolerance is searched. If several masks contain both points, none is used, and a
mask is not used if it contains the mouth/pect of any other latent slot. Unobserved competitors
are also kept. If a competitor's projection is unknown, uniqueness cannot be guaranteed and
extraction for that view is held. This is a geometry-conditioned mask correspondence and is
distinguished from independent inference of the mask itself.

The mask is eroded with a 5×5 elliptical kernel (radius 2px, 0 outside the image), and a core of
at least 12 pixels is required. For colour, the following 6 components are used, from b,g,r (the
source BGR uint8 divided by 255) and HSV h (radians) and s,v (0–1). Whole-image histograms and
background colour are not mixed into the templates.

```
c = max(b + g + r, 1e-8)
descriptor = [median(r/c), median(g/c),
              (1 + mean(s*cos(h)))/2, (1 + mean(s*sin(h)))/2,
              median(s), median(v)]
```

For each view and provisional slot, the per-component median over at least 5 valid training
samples is the template. No scale is learned; the nearest template by Euclidean distance over all
6 components is recorded. If the 3 templates are not all available, a feature is missing, or the
shortest distance is a tie within 1e-12, the point is unscored. No post-hoc search or adoption
decision is made over the distance threshold, colour space, erosion radius or number of features.

Saved outputs are the feature arrays, inter-template distances, within-training distances, the
validation own-slot and other-slot distances, a 3×4 confusion matrix (last column unscored), the
SHAs of raw images, masks, candidate, calibration and code, and ROI examples. The fraction
classified to the same provisional slot is conditional consistency, not identity accuracy.
Because the labels use the slots of a saved trajectory committed from all 666 frames, the label
side is not an independent forward-prediction validation either. A true colour change and a swap
of saved slots cannot be distinguished from this data alone.

The old 20260908 observations._appearance was an RGB 8-bin × 3 histogram of the pixels where the
tube of the 3 predicted nodes overlapped the background-subtraction foreground. This measurement
uses the eroded core of independent D2 instances and 6 components, so no direct comparison with
the old histogram's values or distances, and no improvement over them, is claimed. Neither fully
excludes reflections or erroneous masks. Whether this is usable for physical-fish identity needs
separate verification with independent human correspondences.
