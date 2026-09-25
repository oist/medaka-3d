---
hypothesis: Independent motion estimates before and after a suspicious interval expose fixed-detection identity mismatches missed by distance or past direction alone.
stage: linker
expires: 2026-10-16
decision_rule: >
  This development experiment compares distance-only, forward-direction and
  bidirectional detectors on the existing 20-second CG interval and all 12
  historical human-noted failure neighbourhoods, plus past false-repair and
  normal-motion controls. Preserve coordinates and source assignments in the
  accepted baseline. Geometry proposals remain explicitly unaccepted pending
  independent image/identity evidence. Report sensitivity, candidate burden,
  known-event proximity, abstentions and shared female-anchor scores. Fewer
  jumps alone cannot justify acceptance. No new independently confirmed ID
  error, no lost coverage and improvement in reviewed detection or identity
  correspondence are required before any promotion; this development panel
  cannot establish out-of-sample accuracy.
---

# Bidirectional jump audit and bounded reconnection

Authorized 2026-09-16; implements `docs/BIDIRECTIONAL_JUMP_PLAN.md` as a
separate CPU experiment. Existing C34 and analysis-by-synthesis runs are inputs,
never overwrite targets. No calibration, tracking model or frozen output changes.

## Fixed primary diagnostic before execution

- Baseline: local C34 lambda0.767 with source hash
  `a025af23eab568620ca1e2ae5f2e2fa941a788a17c881c70de94936e56930de0`.
- Development movie: frames75270:78590. Historical event cases: all12 rows of
  the preserved `swap_events.csv` (the notes refer to an earlier C30 render,
  not automatically to errors remaining in C34). Use one-second context either
  side and 52 additional frames for direction estimates. Add three documented
  v3 regression intervals82121:83266,90200:90688,212276:213062 as C34 controls.
- Normal controls fixed before scoring: frames50000:50332,150000:150332,
  280000:280332; they are unlabeled controls, not asserted normal ground truth.
- At166fps, do not smooth candidate edges. Reference directions use componentwise
  median step vectors over17frames, excluding two steps either side of each
  tested edge. Require at least80% finite steps and reference norm0.05mm/frame.
- Compare distance>4mm, that rule OR (step>1mm AND past-angle>=80deg), and
  that rule OR either-angle>=80deg. Report all9 combinations of17/33/50frame
  references and60/80/100deg thresholds. Low-speed angle is unavailable, not zero.
- Candidate edges across fish are grouped when separated by at most3frames,
  then padded by2frames to obtain suspect pose intervals. Record all raw edges.
- Bounded reconnection uses full observed three-fish flanks of17frames, robust
  pairwise slopes, median fit error<=1mm, no>4mm reference step. Intervals over
  83frames are refused. All six right-to-left slot matchings are scored in each
  temporal direction. A geometry proposal requires the same nonidentity winner
  in both directions, margins to runner-up>=5mm summed over fish, and each
  fish's forward/backward boundary residual<=3mm. No threshold is relaxed after
  reading outcomes. Transient proposals must additionally agree at entry/exit.
- Pointsets remain fixed. Detection-row and pose permutations travel together.
  Missing support is a refusal and never a zero-cost shortcut. A proposal is
  not an accepted biological ID correction. No long-range velocity bridge.

## Outputs and evidence

`work/bidirectional_reconnect_v1/` stores protocol/input hashes, candidate and
episode tables, all matching scores and refusals, derived poses, raw+CG clips,
cropped raw review sheets, and a readable report. Shared
`medaka3d.eval.generative` computes observed-motion and female-anchor evidence;
historical note proximity remains a descriptive diagnostic, not a recall score.
Image inspection by the assistant remains separate from human identity truth.

This is a diagnostic/linker proposal experiment; the adapter exposes the Linker
protocol without publishing unreviewed permutations as accepted tracks.

## Implementation audit correction (2026-09-16)

The original preregistration above remains intact. The final `analysis_v2` uses
53 context poses per side, because the largest 50-step sensitivity window plus
two excluded steps and the tested edge requires 53 preceding poses. Review
duration now counts the union of padded intervals. Exported residuals are named
`triangulated_source_residual_mm` to distinguish source-ray triangulation from
the preserved C34 pectoral coordinates; no residual was used as an acceptance
gate. All primary thresholds and cases are unchanged. The original `analysis`
is retained as provenance for the raw review renders; core poses, source rows
and primary flags are checked for exact equality between revisions.
