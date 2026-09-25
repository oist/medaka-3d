---
hypothesis: Direct per-view comparison on identical detections can separate missing input observations, conservative fragmentation, association changes, and parameter sensitivity without confusing them with 3D abstention.
stage: linker
expires: 2026-10-17
decision_rule: Preserve the saved SLEAP baseline and frozen clean2D algorithm; measure all predeclared variants, inspect historical clear-2D scenes, and do not claim identity superiority from smoothness or fewer fragments.
---

# Direct 2D tracking comparison

User-requested follow-up to the clean-fragment experiment. This experiment only
compares camera-local 2D links. No 3D correspondence, projection, join, or slot ID
enters estimation, metrics, or video overlays. All methods receive exactly the
same C01 detections, including their original missing nodes and within-instance
grouping. Saved C27 labels are the SLEAP baseline; negative labels, if present,
are unassigned observations and not a shared track.

`tracking.py` and `flow.py` are byte-for-byte frozen copies of the previous lab,
not a modified tracker. The protocol records their source hashes. Existing P/N/F
results are reused byte-for-byte when the source interval matches; other scenes
use the same configuration and 332 native frames of context on either side.

The primary 20 s and seven historical notes describing clear/easy 2D motion
(21232, 30735, 31479, 105541, 260434, 302762, 333619) are evaluated, along with
the later no-swap review at 323819. Previously examined control_50000 and
holdout20s are regression checks, not fresh holdouts. All are the same trio.

Sensitivity settings are declared in `run.py` before running: independent
relaxation of motion limits, dynamic node coherence limits, competition limits,
and flow residual limits; their combined relaxation; and the opposite combined
strict setting. Missing-node requirements, adjacent-frame-only association,
bidirectional agreement, input detections, and LK extraction remain fixed.
These tests measure parameter sensitivity, not a search for a winning setting.
P versus N versus F is a feature ablation, not a parameter-only comparison.

Metrics separate detector count/node absence, unassigned rows, unique IDs,
contiguous same-ID runs, adjacent links, gap links, changed predecessors,
and large node displacements. A same-ID link is not known biological truth.
SLEAP IDs persist across gaps, so unique-ID counts are not comparable to clean
fragments without the accompanying contiguous-run count. Videos show local IDs
and every detected row; a fragment birth is not a missing detection.

`gap_links` counts available same-ID links with a native gap greater than one
whose endpoint is after the first core frame and inside the core. An origin may
be in the 332-frame warm-up context; origins before the loaded context are not
counted. Adjacent-link and continuous-run statistics exclude the core-entry
link and describe the core interval itself.

Run `run.py --init`, then `run.py --view top_cam` (and both side views), then
`run.py --summarize`. Outputs go to `work/clean2d_video_comparison_v1`.
