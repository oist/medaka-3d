---
verdict: parked
expires: 2026-11-13
blocker: Human swap/no-swap review received for eight clips on 2026-09-16; reviewed panel and exact identity intervals remain unspecified, and node-candidate safety is not established.
---

# Re-evaluation of three-view and body-part correspondence (2026-09-16)

**Addendum after the human review was received (2026-09-16):** ID swap: 76235, 76930, 76988,
31479. No ID swap: 76791, 323819, 77415. 30735 was judged as S0 temporarily jumping to S1 in
side2 and returning. In particular, the counterpart at 76988 is S1/S0, so the improvement in the
S1/S2 tail residual below does not indicate a repair of that ID problem. The geometric
evaluation below is kept as the original measurement. The review was reflected in the
[record of the verbatim text and reviewed videos](../../docs/reviews/2026-09-16_multiview_identity_review.json)
and the
[2D tracking investigation and comparison plan](../../docs/decisions/2026-09-16-custom-2d-tracking-plan.md).
No candidate was accepted automatically.

The requested re-evaluation and the comparison material built from source images, videos and
graphs are complete. The scope was the current 20 s, the 12 known cases and 6 control
intervals, and candidates were searched only in the intervals designated by the previous
forward-backward diagnosis. The original C34 and the outputs of parallel work were not changed.

Application to production is `parked`. Initially, independent image judgement and short-interval
ID annotation had not been obtained. The ID review of the 8 scenes above has now been received,
but the safety of the candidates' body-part correspondence is not confirmed. The re-evaluation
deadline is 2026-11-13. The candidates are not output as accepted tracking.

## Main result

Re-evaluated the 55 candidate intervals of the 20 s, 348 native frames. Taking the union of
overlapping intervals gives 51 windows. A detection correspondence different from the original
was chosen in 129 frames. 24 frames passed the pre-specified check conditions on geometry,
segment length, 3D support and so on.

An additional audit of supporting views per body part found that in 18 of the 24 frames one
fish's tail dropped from 3 supporting views to 2. Even when the whole-fish observation count is
preserved, moving an observation to another fish hides this problem. **6 frames meet the
conditions without reducing the supporting-view count of any body part**: f75398, 75399, 76791,
76989, 76990, 76992.

These 6 are a post-hoc priority set for visual inspection, not an accepted count or a correct
count under modified pre-specified conditions. At f76791 the references in both directions are
valid for all slots. In the other 5 frames one of the references is insufficient, and ID
retention across before and after is not guaranteed.

Over the whole 20 s, pectoral-fin moves over 10 mm number 3 in both the original and the
candidates. Neither a swap reduction nor removal of teleports has been demonstrated. The
candidates are per-frame local search results; correspondence switches on linking, boundaries
and IDs need to be checked separately.

## What was tried

- Whole-fish correspondence in each of the 3 views: 6^3=216 combinations.
- When there are 4 or more fish and similar cases, substitution of one individual in one view by
  a detection not currently used.
- Local candidates, starting from the original correspondence and the best whole correspondence,
  that exchange the mouth, pectoral fin, tail, or mouth + pectoral fin between a pair of fish in
  one view.
- Each candidate evaluated by the shared refractive triangulation, ray residual, reprojection,
  leave-one-view-out prediction, segment length, and the available past and future motion
  prediction.
- Biological ID ground truth is not used for candidate selection. Candidates that use the same
  detected node for more than one fish, or that lower the apparent residual by reducing total
  observation support, are invalidated.

The search does not exhaust all simultaneous node exchanges. Where the 2D detections in a view
themselves contain reflections, missing observations or mixed body parts, reassignment alone
does not resolve them.

## Examples

| frame | Result and remaining problem |
|---|---|
| 75398 | Exchanges the S1/S2 tail correspondence in the top view. Body-part support kept; in the images the offset between the side-view projection and the fish body looks smaller. Past-side reference insufficient. |
| 76791 | Exchanges the S1/S2 pectoral-fin correspondence in side 2. Support kept, past and future references valid. The fish overlap in the side view, so judgement from the source images is ambiguous. |
| 76988 | S1 tail residual 8.29→0.69 mm, but tail support 3→2 views. S2 goes 2→3, tail residual 0.32→1.21 mm. Not interpreted as recovery of three-view agreement. |
| 76989–76992 | A different candidate uses a previously unused 2D detection, and the supporting-view count does not drop. To be checked separately from the missing-observation shift at f76988. |
| 76235 | Changing mouth and tail in side 1 lowers some residuals, but the pectoral-fin jump and an out-of-tank node remain. Conditions not passed. |
| 76930 | The candidate that fills the missing tail has a pectoral fin–tail length of about 37.29 mm. Despite the low combined cost, held on the fish-body coherence condition. |
| 30735 | A candidate that moves the 3D tail dropout to another slot. Per-part coverage is lost; conditions not passed. |
| 31479 | Some mouth and pectoral-fin residuals worsen, and past and future references are also insufficient. Conditions not passed. |
| 323819 | At the centre frame the original correspondence remains, as does a ray inconsistency of about 13–14 mm on mouth, pectoral fin and tail. Target for re-checking the detections themselves, reflections and so on. |

The image observations are qualitative checks by the assistant, not independent ID ground truth.

## Comparison caveats and additional audits

`support_audit.json` stores per-part support counts for all 19 cases. Comparing only the 134
corresponding nodes that keep the same 3 views in the main interval's "24 frames that passed the
original check conditions", the mean ray residual is 0.842→0.478 mm (12 improved, 4 worse, 118
unchanged). The 46 nodes that keep the same 2 views are all unchanged. This compares fit on
already-selected candidates and is not an accuracy evaluation on unused data.

For the S1 tail at f76988, the original leave-one-view-out prediction's maximum error of
116.36 px becomes undefined under the candidate. The comparison graphs do not replace missing
values with 0 or with interpolation. The per-part support counts need to be read, not only the
total 2D observation count.

`refit_audit.json` also checks the effect of re-triangulation. For unchanged source nodes the
coordinate difference is at most 0.0000105 mm and the availability change is 0. The difference
between the saved C34 and the all-node reconstruction is numerically tiny in this comparison,
but residuals from the shared triangulation and projections of the saved coordinates were
recorded as clearly distinct.

## Artifacts for human review

- [Comparison page](../../work/multiview_node_review_v1/index.html): the 6 priority frames,
  comparison videos, support reductions, and all images and graphs can be reviewed in order.
- 81 comparison images: representatives of the 55 main candidate intervals, the 14 additional
  backward-direction candidates, the 12 known scenes, and supplements for focus scenes.
  Duplicate representative frames are shared.
- 9 slow-motion videos: 166 native frames each, 1 s of real time played as about 5.53 s at
  30 fps. Raw images are used for the top/bottom and left/right comparisons, with no frames
  skipped.
- 29 graph images: all 19 cases, the 9 local video intervals, and a dedicated figure for the
  tail support count at f76988.
- The NPZ/JSON for all 19 cases store per-part detection rows, original and candidate 2D/3D
  coordinates, all cost components, top candidates, reference validity, missing observations
  and the original verdicts.
- Hashes of source data, code and outputs, read-back of all video frames, read-back of PNGs, and
  checks of HTML reference targets are recorded.

Applying the 24 that passed the original conditions automatically in bulk is not recommended. In
particular, the 18 that move supporting views to another fish cannot be justified as corrections
by low residual alone. The material is arranged so that the 6 support-preserving frames and the
f76988 control example can first be compared on the same raw images.

## Implementation verification

34 tests pass in total: 10 for refractive geometry, 15 for candidate generation and rejection,
and 9 for the baseline-preserving Associator. The lab's structure and isolation are also checked.
The Associator performs only within-frame diagnostics and returns the original object. The
search that uses past and future references is isolated in this lab's runner and has not been
introduced into production.

Filter interaction in the browser was not verified automatically. Images, videos and graphs can
be opened directly, and local link targets are verified.
