---
hypothesis: Separating SLEAP provisional per-track histories from confirmed adjacent observation fragments avoids self-generated endpoint competition while retaining all observations and recording ambiguous alternatives for downstream 3D association.
stage: linker
expires: 2026-10-18
decision_rule: Pass all observation, one-to-one, row-order, gap-separation, and recovery contracts before interpreting real-data results. Compare a fixed default and every predeclared one-factor setting on all twelve existing intervals and three views. Fragmentation is a continuity diagnostic, not identity accuracy. Do not promote a setting based only on fragmentation without verified individual correspondence.
---

# SLEAP adapter for the medaka pipeline

User-authorized continuation on 2026-09-18 of the completed variable-count 2D
comparison. This experiment improves the actual installed SLEAP tracker through
local subclasses. Neither the installed package nor production tracking changes.
The camera-local observation contract is intentionally separate from the
production `Linker` protocol, which consumes already associated 3D instances.

## Design fixed before real-data execution

- Use `sleap-nn==0.3.1` `Tracker.track`, per-track local queues, mean similarity,
  Hungarian matching, and native source-instance object correspondence.
- Score the observed pectoral node with SLEAP `compute_oks`, fixed area
  `120**2` (top) / `100**2` (side), and sigma `0.025`. Missing mouth/tail alone
  cannot reject a link. This is a declared scoring change from S2 (mouth + P).
- Add a private unmatched option for each detection. A low similarity may
  create a new provisional history; insufficient confidence margin does not
  create a new history. Soft confidence refusal splits only the output fragment.
- Retain up to five observations per provisional track. Expire each history
  observation by native-frame age, with a default maximum of 17 frames
  (`ceil(0.10 * 166)`). Track IDs never recycle after expiry.
- Do not discount distance by prediction uncertainty. This adapter uses observed
  history scores, with no optical flow, Kalman filter, or fabricated observations.
- Commit an output link only when the selected provisional predecessor is in
  the immediately preceding native frame, its P step is at most `0.30 * span`,
  and the selected score beats alternative tracks and alternative detections
  by the declared margin. Include the unmatched threshold in alternatives.
- Gap associations may update a provisional history but remain separately
  recorded reconnection candidates. They never join confirmed fragments.
- All detections, including missing-P and surplus rows, receive nonnegative
  confirmed fragment IDs (singletons where needed). Missing-P rows may have
  provisional ID `-1`. Original points, masks, scores, and global rows survive.
- Provisional IDs are private association hypotheses, not fish identities or
  confirmed continuity. A confident fragment is not a reflection classification.

SLEAP's history update after an ambiguous provisional match can still influence
future decisions. Separating fragment output prevents that ambiguous edge from
becoming a confirmed identity claim; it does not establish that the provisional
history is correct. Crossing identity remains an independent evaluation target.

## Fixed parameter study

No parameters are selected per scene or after looking at these results.

| Condition | Only change from G0 |
|---|---|
| G0 | history 5, match threshold 0.05, confidence margin 0.05, maximum age 17 frames |
| G_history3 | history 3 observations |
| G_history9 | history 9 observations |
| G_match025 | unmatched similarity threshold 0.025 |
| G_match10 | unmatched similarity threshold 0.10 |
| G_margin02 | confidence margin 0.02 |
| G_margin10 | confidence margin 0.10 |
| G_age8 | maximum age 8 native frames |
| G_age33 | maximum age 33 native frames |

The prior saved C27, S1, S2, P0, and P1 results are unchanged controls. An additional
`SP` control changes S2's scoring nodes from mouth + P to P only, using the same
fixed area, five-observation mean, uncapped native queues, and Hungarian matching.
SP keeps ordinary SLEAP labels, including gaps; these are not confirmed G fragments.
SP versus G0 includes the state/commitment design changes and is not a one-factor
ablation. The nine G conditions isolate parameter sensitivity within the declared
adapter design. Correct continuity and wrong identity links cannot be measured
from unlabelled fragment counts.

## Evaluation and integration scope

Use the same twelve intervals, three views, raw C01 detections, and 332-frame
context as `work/variable_count_tracklets_v1/protocol.json`. Every interval has
already been inspected: this is development/regression evaluation, not a new
holdout, and all recordings concern the same trio.

The core continuous-run definition and strata are frozen from the previous
comparison, with their source hashes recorded. No competing definition of
identity accuracy is introduced. The small one-fish, one-position-change recovery
test that exposes P1's self-competition is required, alongside variable count,
missing nodes, ties, gaps, arbitrary input order, and history expiry.

The existing C28/C25 interface excludes rows whose labels are negative and uses
frame-local source offsets. Therefore export confirmed fragment labels, retain
the explicit global-row mapping, and check source offsets before any later
integration. These interval outputs do not replace C27's full-recording files.

Inputs, code, configuration, numerical outputs, and verification have SHA-256
and `medaka3d.provenance` records in `work/sleap_pipeline_adapter_v1/`.
`RESULT.md` records the result and the unresolved adoption criteria.

Run entry points with `.venv/Scripts/python.exe`. Preserve all prior results.
