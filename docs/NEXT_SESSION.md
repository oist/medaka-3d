# Where to start — the current state, 2026-09-08

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
