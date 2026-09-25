# Result: direct camera-local 2D comparison

**Verdict: parked.** The requested comparison and review artifacts are complete;
the strict N/F tracking design is not promoted as superior to SLEAP.

The experiment compares the saved full-session C27 output with the exact frozen
clean2D P/N/F arrays on the same C01 detections. There is no 3D reconstruction,
projection, or cross-view joining in the comparison. New scenes use the same
frozen tracking and optical-flow code; existing arrays remain identical.

## Main 20-second interval

Native frames `[75270,78590)`, 166 fps, one trio, all three cameras. All 29,386
detections remain in all methods and no row is unassigned. Mouth is absent in
743 rows, pectoral in zero, tail in 1,558. These are node observations within
detected instances, not counts of invisible fish. The previous 47.9% result was
3D acceptance, not 2D detection retention.

| Method | Contiguous adjacent-frame runs | Gap links | Pectoral steps >30 px |
|---|---:|---:|---:|
| Saved full C27 / SLEAP | 224 | 211 | 39 |
| Pectoral-only P | 323 | 0 | 0 |
| Node-coherence N | 1,257 | 0 | 0 |
| Nodes + optical flow F | 1,324 | 0 | 0 |
| F, all predeclared numeric limits relaxed | 1,028 | 0 | 0 |

F cuts 1,120 saved adjacent edges and changes 9 predecessors; 720 of these
1,129 differences include a missing-primary-node reason. All missing primary
nodes here are mouths. Reasons are nonexclusive. More fragments are not more
missing detections; fewer large steps are not proof of correct identities.

All six numeric sensitivity variants were reported. Joint relaxation reduces
fragment count by 22.4%, whereas joint tightening gives 18,452 fragments. This
is a sensitivity diagnostic, not a selected winning setting. Mandatory finite
mouth/pectoral support, adjacent-only links and bidirectional agreement remain
unchanged across those six variants.

## Historical clear-2D scenes

The original human comments were traced to the archived swap-event CSV for the
earlier C27/C28/C33/C30 video. They locate scenes; their slot IDs are not 2D
identity truth. Seven clear/easy notes and a later no-swap control were included.
Detailed case conclusions are in `case_findings.json`; metrics separate the
17-native-frame anchor vicinity from the full 166-frame video excerpt.

- f21232 side2: SLEAP/P retain three continuous runs in the 17-frame vicinity;
  N/F produce 15. Raw three-frame crops show a stable pectoral trajectory and
  absent mouth in the fish whose F ID changes every frame.
- f30735 side2: the single detected instance remains one track in every method.
  The earlier 3D slot jump is not established as an input 2D swap. Top is also
  unchanged locally; side1 acquires extra N/F cuts.
- f31479 side1/side2, and f302762 side1/side2: local links agree across methods.
  Top gains cuts. This is not evidence that the independent 2D part fixed the
  earlier 3D correspondence failure.
- f105541 top: F changes six predecessors, but raw fish proximity, overlapping
  nodes and changing detections prevent a confident all-fish identity verdict.
- f260434 top: P/F fragment heavily during close, rapidly changing motion;
  continuity improvement is not demonstrated even after numeric relaxation.
- f333619 top: saved IDs move between separated fish with up to 233 px pectoral
  steps. P maintains the nearby trajectories in the inspected frames; F keeps
  two and fragments the mouth-missing fish. Current same-setting cropped SLEAP
  also removes this jump, so the custom algorithm's unique advantage is not
  established. Both side cameras already agree locally.
- f323819 no-swap control: F introduces extra side-view cuts. They must not be
  counted as verified swap corrections.

## SLEAP configuration and replay findings

Saved C27 uses mouth/pectoral OKS, mean over each ID's last five observations,
LocalQueueCandidates, uncapped tracks and forward Hungarian assignment. Five
observations do not mean native-frame age expiry. Current implementation has
no configured no-match alternative, next-best margin or displacement gate.

Under the current implementation, a mouth-missing M/P pair has bounding-box
area zero. At selected real frames all candidate OKS scores underflow to zero,
but assignment continues. Fixed-history sigma .0125/.025/.05/.1 rescoring
does not rescue the mouth-missing rows. This is score diagnostics, not a full
tracking replay at changed sigma. N/F instead split those rows; neither
behavior satisfies continuous tracking from a clear, observed pectoral alone.

Current cropped SLEAP replay with identical history and row-faithful output was
compared at windows 1/5/10 on four cases × three views (36 arrays). Main20s runs
are 219/217/221 and the maximum pectoral step is 403.5 px in each. This is a
controlled current-runtime comparison, not exact historical full-run replay.

The C24/C27 output wrapper also assumes returned instances retain input order.
Synthetic and real cropped replay show otherwise: legacy and row-faithful
readouts differ at 24 core rows over seven frames/six case-view runs. This is
not established as the cause of the highlighted jumps. In particular, side1
235/276 px links reproduce with either readout and no nearby reordering;
top333619 does not reproduce with either. Full history and historical runtime
differences remain unseparated. Existing C24/C27 files and saved outputs were
not edited as part of this diagnostic.

## Artifacts and verification

Output root: `work/clean2d_video_comparison_v1`.

- `comparison_report.html`: self-contained interactive report with case/view
  selection, direct videos, inspection crops, local and full metrics, parameter
  comparison and all tested sensitivity settings.
- `review/manifest.json`: 42 original videos + 42 PNG stills. Three 20-second
  overview videos plus 13 anchor clips × three views; 8,274 output frames all
  decoded. Overview is time-sampled; local clips retain all 166 native frames
  and play at 30 fps without interpolation.
- `report_media/manifest.json`: same-resolution JPEG derivatives of the stills
  for the HTML; original PNG and MP4 hashes remain unchanged.
- `summary.json`, `review_windows.json`, `parameter_audit.json/.md`,
  `c27_sensitivity/summary.json`, `case_findings.json`: inspectable evidence.
- `audit.json`: all 330 method/case/view arrays retain exact C01 rows, points,
  frames and missing nodes; saved C27 labels are exact; earlier frozen P/N/F
  arrays and both earlier HTML reports remain unchanged.
- `review_receipt.json`: independent recomputation of all 330 core blocks,
  390 local-window blocks and 66 stated numerical claims passes. Gap-link counts
  can include a warm-up-context origin; adjacent-link and run counts cannot.
- 13 focused metric/renderer tests pass, as does Ruff for this lab and tests.

The immediate design implication is to separate pectoral-based continuity from
mouth/tail observation quality, with explicit uncertainty and meaningful
unmatched options. Missing nodes must remain missing. No all-fish ID accuracy
or swap-rate improvement is claimed, and this is still one previously examined
trio rather than an independent benchmark.
