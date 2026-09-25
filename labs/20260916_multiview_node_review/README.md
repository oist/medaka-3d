---
hypothesis: Within the previously flagged native-frame windows, explicit camera-row and node exchanges can explain cross-view and body-cohesion errors that whole-3D-slot exchanges cannot.
stage: associator
expires: 2026-10-30
decision_rule: >
  Reevaluate all previous bidirectional episode windows in the 19-case panel,
  preserve the accepted C34 tracks, and deliver source-pixel comparisons and
  common-unit geometry charts. Candidate fit improvements are descriptive,
  not identity accuracy. No promotion or biological identity correction without
  independent image adjudication and no coverage regression. Rank candidates
  with fixed geometric, body and optional exterior motion terms; report missing
  references, ambiguity, every changed source row/node and held-view errors.
---

# Three-view and node correspondence review

Authorized by the user on 2026-09-16. This is a separate CPU experiment using
`work/bidirectional_reconnect_v1/analysis_v2` and the same checksummed C34,
detection caches, calibration and source videos. No old lab imports; no source
or concurrent analysis-by-synthesis output changes.

## Fixed candidate experiment before execution

- Test the union of all padded bidirectional episodes in each of the previous
  19 cases (20-second primary,12historical,3false-repair controls,3unlabelled
  controls). Deduplicate overlapping episodes within a case. Frame-local
  detection row numbers have no temporal identity meaning.
- Start with each slot's selected detection in each camera. Enumerate all
  6^3=216 whole-instance view permutations. Also consider single-view whole-row
  substitutions from unused detections, including frames with2,4,5detections.
  Never reuse the same row/node for two fish. Missing detections stay missing.
- On the baseline and best whole-instance candidate, enumerate one-camera
  pair exchanges of mouth, pectoral, tail or mouth+pectoral. This is a bounded
  local search, not exhaustive over arbitrary simultaneous node exchanges.
- Reconstruct all nodes with the same refractive ray triangulation. Preserve
  frozen C34 geometry separately. Compute common triangulated baseline, ray
  RMS(mm), own-view reprojection(px), and leave-one-view-out prediction(px).
  Held-view prediction requires all3views;2-view cases remain unscored for it.
- Learn pooled mouth–pectoral and pectoral–tail reference lengths from outside
  all tested frames in that case, with finite nodes and ray RMS<=1mm. Require
  at least30 eligible fish-frames; otherwise use the primary20s reference.
  Scale=max(3*MAD,0.75mm). This prior does not determine biological identity.
- Fixed dimensionless candidate cost: mean min((rayRMS/1mm)^2,100), plus mean
  Huber(length deviation/reference scale,delta2), plus0.25 times the average
  available forward/backward Huber(pectoral prediction error/3mm,delta2), plus
  0.05 per changed source view/node, plus50 per missing reconstructed node.
  Invalid/nonphysical projections and loss of total2Dsupport cannot improve
  a candidate. Huber(z)=0.5*z^2 for|z|<=2, else2*(|z|-1).
- Exterior motion is fitted independently to17frames before/after the merged
  suspect interval, using robust pairwise slopes separated by>=3frames,
  >=80% finite pectorals, median fit error<=1.5mm and no>4mm reference step.
  Absent flanks are marked, not filled. Motion terms rank local candidates;
  they do not certify long-range identity or repair an already-wrong flank.
- Show the best candidate even if weak, explicitly labelled unaccepted.
  Mark it review-worthy only if cost improves by>=25% and>=0.5, all baseline
  reconstructed nodes remain available, no supported camera/node is dropped
  in total, all segments fall within3reference scales, and all candidate nodes
  lie in the192x192x186mm tank. These are review filters, not acceptance gates.
- Keep original/candidate geometry,2Dcoordinates and per-node source rows,
  cost components, alternative margin, missingness, and reference validity.
  Shared observed-motion evaluation is descriptive; no custom identity score.

## Delivery

Every primary episode and historical center gets a clean/overlay comparison
sheet. Representative changed and unresolved cases get native-frame slow clips.
Standalone scientific PNG graphs show common metrics on the same native-frame
axis. An offline gallery indexes all cases and detailed numeric records. All
artifacts carry source/code hashes and readback checks. Assistant visual
inspection remains separate from human labels.
