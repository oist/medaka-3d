# Where to start — the current state, 2026-09-08

**2026-09-18 completed comparisons:** Stages A-C of the variable-count plan
and the subsequent SLEAP adapter comparison are complete. Start with the
[variable-count result](../labs/20260917_variable_count_tracklets/RESULT.md)
and [SLEAP adapter result](../labs/20260918_sleap_pipeline_adapter/RESULT.md).
The adapter separates provisional histories from confirmed fragments and keeps
every detection. Across the main 20 s, contiguous runs are S2 208, P1 11,615,
G0 593, history3 417, and margin0.02 397. These are continuity measures, not
identity accuracy; all 12 evaluated intervals were already known. The adapter
is parked, not promoted. The planning note below records the earlier handoff
and no longer describes an unimplemented task.

**2026-09-17 next-session plan (planning only):** Start with the
[variable-count independent 2D tracklet plan](decisions/2026-09-17-variable-count-2d-tracklet-plan.md).
The user will execute it in a new session. Three physical fish do not imply
three detections or three camera-local tracklets: preserve 0/1/2/3/4+ detections,
reflection candidates, open endpoints and unresolved identities. Continue clear
pectoral tracks independently of other fish disappearing or mouth/tail absence.
First implement and compare stages A–C on the current 20 s and known scenes,
including per-view videos and an HTML report. Separate row-faithful SLEAP,
OKS-scale repair and independent tracking. Stage D is subsequent 3D/gap joining.
No implementation or tracking run for this new plan has occurred yet.
This is the next task; the older next-step lists below are historical context.

**2026-09-17 direct 2D video comparison:** The requested camera-by-camera
SLEAP/P/N/F comparison is in [the new HTML report](../work/clean2d_video_comparison_v1/comparison_report.html)
and [lab result](../labs/20260917_clean2d_video_comparison/RESULT.md).
42 videos compare identical raw frames and detections, with no 3D layer.
Main20s retains all 29,386 detections; contiguous 2D runs are saved C27 224,
P 323, N 1,257, F 1,324 (joint numeric relaxation: 1,028). Mouth is absent in
743 rows while pectoral is present in every row. Several human-clear scenes
already have identical 2D links; f21232 side2 shows unnecessary N/F splits.
Current SLEAP's two-node OKS also degenerates for mouth-missing rows and can
continue zero-score assignments. Cropped history1/5/10 does not solve large
steps. A C24/C27 returned-row ordering defect was reproduced separately, but
not established as the cause of the highlighted jumps; no production files or
prior results were changed. **Parked:** separate pectoral continuity from
mouth/tail quality; do not claim a validated swap-rate improvement.

**2026-09-17 clean-fragment follow-up:** The user's original architecture is now
implemented in [a separate lab](../labs/20260917_clean_tracklets/README.md):
ignore SLEAP temporal IDs, build independent conservative per-view fragments,
then verify a separate join layer from two other clean views. Pectoral-only,
node-coherence and node-plus-image-flow conditions retain every detection.
See [the result](../labs/20260917_clean_tracklets/RESULT.md) and
[the new HTML report](../work/clean_tracklets_v1/clean_tracklets_report.html).
Final arrays are `work/clean_tracklets_v2`; the complete v1 is preserved.
The primary 20 s retains 29,386 detections with 94.7–96.1% incoming clean-edge
coverage, but the new 3D observation rate is only 47.9% and all three female
intervals are unscored. Its six core joins are adjacent-fragment connections,
not six verified swap fixes. 50 focused tests and both six-case audits pass.
**Parked, not promoted:** improve 3D clean-fragment assignment/reacquisition;
do not describe gate-induced disappearance of large steps as ID improvement.

**2026-09-17 implementation update:** The custom 2D plan through step 3 is
implemented and compared in [the isolated lab](../labs/20260917_custom2d_comparison/RESULT.md).
The [HTML report](../work/custom2d_comparison_v1/comparison_report.html) compares
saved SLEAP (A), bidirectional link cuts (B), and custom bidirectional tracking
with short deferred hypotheses (C), with videos and native-frame graphs.
Main 20 s final steps >10 mm are 3 / 4 / 3; all methods retain the same female
endpoint change at 76698→77285. f31479 improves, while the human no-swap control
f323819 worsens in C (4→6 large steps). Final results are `analysis_v3`,
not the preserved first attempt or conservative v2 ablation. No production
promotion: all-three-fish ID superiority is unproven. The bounded deferred
check adds two 2D links at 90576→90578 without changing 3D coordinates/sources.

**2026-09-16 review update:** The experimenter reviewed eight clips from the
20-second / known-failure multiview review. Four have reported ID swaps, three
have no ID swap, and one has a transient side2 jump. See the
[verbatim review](reviews/2026-09-16_multiview_identity_review.json) and
[custom 2D tracking investigation and comparison plan](decisions/2026-09-16-custom-2d-tracking-plan.md).
The review and provenance are archived under configured `asset_root/groundtruth/swap_event_gt/`
and registered in MANIFEST (3303 entries; all previous entries retained).
Reviewed left/right panel and exact event intervals remain unspecified.
No node proposal or new 2D tracker has been promoted. In particular, f323819
is human-reviewed **no ID swap**, and f76988 is **S1/S0**, not the S1/S2 pair
examined by the earlier tail-node proposal. The older status below is retained
as the 2026-09-08 record.

**Read first:** [`TRACKING_PIPELINE.md`](TRACKING_PIPELINE.md) — the chain as
it stands, its numbers, its instruments, and what is still wrong. Then the two
decision records that carry every pre-registration and verdict of the
2026-09-01/03 work:
[2026-09-01 slot stitching](decisions/2026-09-01-slot-stitching.md) (the join
rule, v1 → v4) and
[2026-09-01 doubt channel](decisions/2026-09-01-doubt-channel.md) (doubt
signals, demotion variants, ray rejection, micro-flip repair, the annotator's
proposals). **Status: nothing is promoted.** `c03` still owns the fixed
output paths. `n_groups = 1`.

## What stands

Production chain on K_1_2: `c27 → c28 → c33 → c30 v4 → c34`. At λ 0.767 the
slot labelling keeps the female's identity in **296 of 298** annotated
intervals (0 breaks, 2 swaps — both recorded refusals); the gated tracker is
unchanged at 272 / 26 / 0. The two swaps and the 29 remaining big-step scenes
are classified in the pipeline page. Its render is
`work/c29_full_render/chain-c28-c33-c30v4-c34/`, not yet reviewed by eye.

## The one habit that produced everything above

Every improvement in this arc came from the experimenter watching the render
and naming a mechanism — the FP-as-third-ray spikes, the wave-corrupted top
camera, the body-length wobble, the node exchange, the displacement *vector*
against the surrounding 0.1–0.3 s of motion, the "flag the scene and connect
the clean ends". Each was measured before it was built, and each measurement
is in the decision records. **Test the experimenter's proposal directly, as
stated, before generalising it**: the one generalisation made without asking
(demoting all views, not just `top_cam`) is the one that failed its gate.

## What to do next, in order

1. **Review the v4 render.** The v4 thresholds (medians must lean 2.5 mm
   before velocity may decide; a zone correction needs 30 mm) were calibrated
   on the twelve annotated events; the review is their first test. Add new
   swap notes to `assets/groundtruth/swap_event_gt/` in the same format —
   frame, slots, kind, verbatim note — and say which video they were made on.
2. **Node-granular demotion**, pre-registered before it runs: mask only the
   exchanged node's ray at frames where the node-exchange signal fires
   (1.3 % of frames, 88 % capture, names view and node). Same reading rules
   as every demotion variant: all seven λ, `c09` gated swaps may not exceed
   0/0/0/0/0/1/3, scored as a full chain against the production chain's label
   swaps 30/5/4/3/2/2/4. Whole-view demotion is parked; its three verdicts are
   in the doubt-channel record and should not be re-run.
3. **Blank, don't interpolate**, the 2-ray contaminated frames the doubt
   channel flags — turn a wrong point into a visible gap. Measurement-only
   first: how many of the 40 residual-blind big steps it removes.
4. **Transfer to K_1_1 and K_1_3.** No annotation there; the doubt channel and
   the step census are the instruments. Run the production chain and report
   the grid-wide quantities beside K_1_2's.
5. `c35`, apply human verdicts from `swap_event_gt` as corrections — cheap,
   and closes the review loop for the two declared refusals.

## Housekeeping owed

* Everything since 2026-08-02 is **uncommitted**, including the tools, the
  tests and these documents. Commit before deleting anything else.
* `MANIFEST.json` was regenerated on 2026-09-08 with hashes: 3,301 assets,
  27.70 GB, `swap_event_gt` registered. The `manifest_registration: PENDING`
  line inside `swap_events.json` is now stale but is left as written, because
  editing the file would invalidate the checksum just recorded for it; the
  manifest is the authority.
* The HITL-v2 benchmark under `benchmarks/` is still outside the manifest.

## Reading rules that do not change

* Pre-register the rule and the reading rules, run all seven λ, then append
  results; a claim is carried only if it holds at every arm.
* `c09` with `--npz` per arm, then bare to restore the baseline (269 / 19 / 8).
  Swap counts of 0–8 rank nothing.
* Output names carry the chain (`--tag`); partial runs carry their range.
* Human-made data goes to the bucket at once, verbatim, with provenance.

## Historical documents kept on purpose

`NEXT_SESSION_stitching.md` (2026-08-18) and `NEXT_SESSION_uncapped_followup.md`
(2026-08-26) are cited by decision records and tool docstrings and stay as the
plans those records executed or retracted. `NEXT_SESSION_confidence_weighting.md`
(2026-08-24) is the only record of a 14-scene human review of refused links.
None of the three is a starting point.
