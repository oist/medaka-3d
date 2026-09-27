---
verdict: parked
expires: 2026-10-30
blocker: Independent current-C34 short-interval identity annotations are unavailable for the added directional candidates and the unresolved historical failures; assistant image inspection cannot establish repair precision or safe promotion.
---

# Forward-backward jump detection and local linking test (2026-09-16)

The development test was completed on the current 20 s and the 12 known scenes. The
forward-backward direction check extracted additional candidates, but 0 link proposals met all
pre-specified conditions and 0 automatic corrections were accepted. A reduction in swaps has not
been demonstrated. The existing C34 output was preserved, and the analysis-by-synthesis
implementation and runs were not changed.

`parked` is the verdict on promotion to production. The implementation, the analysis and the
review artifacts for this round are complete. Independent short-interval ID annotation of the
additional candidates is an unmet external condition, and AI observations of the source images
are not used in its place. The re-evaluation deadline is 2026-10-30.

## Scope and results

- Main interval: f75270–78589, 166 fps, 20 s.
- 12 known cases: about 1 s before and after f21232, 30735, 31479, 67142, 72143, 76930, 99209,
  105541, 260434, 302762, 323819, 333619.
- Controls for past erroneous corrections: f82121–83265, 90200–90687, 212276–213061.
- Unannotated controls: 2 s each from f50000, 150000 and 280000. These are not ground-truth data
  asserted to be normal.

| Detection method (main interval) | Candidates (individual × adjacent frame) | Merged candidate intervals |
|---|---:|---:|
| Distance only | 21 | 12 |
| Distance + past direction | 96 | 47 |
| Distance + past and future direction | 110 | 55 |

The main condition is a 17-frame reference and 80°; the direction test applies to moves over
1 mm, and distance alone to moves over 4 mm. For 6 of the 14 additional candidates the past
reference direction is undefined. The candidate count is neither a swap count nor a count of
correct detections. All 9 conditions (17/33/50 frames × 60/80/100°) were saved, and the main
threshold was not re-chosen after seeing the results.

Persistent link verdicts for the 55 main-interval candidates: original correspondence preferred
37, reference interval insufficient 15, ambiguous 3. Preferring the original correspondence is
not proof of correct ID either. Including temporary 1–3-frame exchanges, 0 proposals passed the
conditions in the link checks over all 19 intervals.

At f76930, S0↔S1 was preferred in both directions. The summed boundary error over the three fish
was 10.93 forward / 10.77 mm backward for the original correspondence, against 8.07 forward /
6.59 mm backward for the exchanged correspondence. However, the margins over the runner-up,
2.86 / 4.18 mm, are below the required 5 mm, and the per-individual maximum errors under the
exchange, 5.68 / 3.46 mm, exceed the 3 mm limit. The exact exchange time was not determined
either, and the exchange was not accepted.

## Source-image check

For the 14 additional candidates and the 12 known cases, comparison sheets placing the three
cameras' source images and node projections side by side were made and inspected. Each sheet
holds 5 native frames and is not a human ID annotation from consecutive images.

- The additional candidates include scenes where proximity, turning, node position jitter or
  partially mixed correspondence is suspected. The 14 additional candidates are not counted as
  new confirmed swaps.
- The side images at f31479 contain an observation suggesting a whole-fish slot exchange.
- At f30735 and 323819, projections that lie near the same fish in the top view are offset from
  the fish body in the side views; an anomaly in cross-view correspondence or depth
  reconstruction is suspected.
- Around f76235 and 77415, the correspondence of mouth/pectoral fin and tail may be mixed.
- Some scenes flagged under the earlier C30 do not reproduce an obvious whole-individual exchange
  in the current C34 images. The rate of detections near the old notes is not used as the
  current recall.

All of these are qualitative observations by the assistant and are not used as independent ID
ground-truth labels. Slot names from the original notes are not carried over to C34 as they are
either.

## Existing independent female annotation

The 299 adjacent intervals of the 300 annotations were fixed and scored with the shared
evaluation function and the original C34 availability: **295 kept / 2 swapped / 2
undeterminable**. No correction was accepted, so before and after are identical.

The annotation intervals where a swap is confirmed are f76698–77285 and f130760–138892. These
are not the true exchange times. The denominator differs from the 298 intervals of the old
tally, which deletes missing annotations and joins the neighbouring intervals, so the two are
not to be confused. The evaluation by availability under geometric support was saved in a
separate field.

## Artifacts and verification

- [Review page](../../work/bidirectional_reconnect_v1/index.html): links to the videos, all
  candidate images, all 19 intervals and the sensitivity analysis.
- [20 s source video + CG](../../work/bidirectional_reconnect_v1/review/cg20s_raw_and_measured_cg.mp4):
  600 frames, 30 fps, 20.000 s.
- [Slow-motion videos](../../work/bidirectional_reconnect_v1/review/manifest.json) of f76235 and
  f76930: 332 native frames each, 30 fps, 11.067 s. The original 2 s is stretched about 5.53×,
  with no intermediate frames omitted.
- [Final tally](../../work/bidirectional_reconnect_v1/analysis_v2/summary.json), the errors and
  rejection reasons for all 6 correspondences in each interval, and provenance with hashes of the
  source data and code.
- 33 tests covering direction, linking, the Linker adapter and the tally, plus structure and
  isolation checks for the new lab. Videos were verified by reading back every frame, PNGs by
  pixel equality. HTML reference targets were verified for local file existence; interactive
  verification in a browser was not performed.

Three implementation issues found in review were fixed and `analysis_v2` was generated: 53 poses
before and after for the 50-step boundary reference; review time with overlapping padding
tallied as a union; the residual of the original triangulation distinguished from the residual
of the saved C34 pectoral-fin coordinates. The main threshold was not changed, and generating the
review page checks that all core poses, sources and main diagnostic values of the `analysis`
used for the videos match the final version exactly. The first-run record is retained.

## Next verification targets

Narrow the intervals with the direction test, and there re-evaluate the three-view detection
correspondence and the grouping of mouth, pectoral fin and tail. Using the stable intervals
before and after as boundary conditions, treat individual-ID exchanges separately from node and
view mix-ups. Errors that cannot be resolved by swapping whole individuals' pectoral-fin 3D
coordinates alone need to be verified using reprojection image evidence.

Promotion to production requires independent short-interval ID annotation and evaluation on
untuned intervals. Reducing jumps alone is not judged a successful correction.
