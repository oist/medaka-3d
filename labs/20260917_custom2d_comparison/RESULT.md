---
verdict: parked
expires: 2026-11-17
blocker: No demonstrated reduction in biological ID swaps; improvements coexist with regressions.
---

# Custom 2D tracking: implementation and comparison results (2026-09-17)

Stages 1–3 of the plan were implemented, and saved SLEAP (A), cutting by forward-backward audit
(B), and custom bidirectional 2D tracking + short-term two-hypothesis re-evaluation (C) were
compared on the same detections. The scope was 20 intervals in total: the main 20 s, known
cases, controls, and a different 20 s from the same group.
Final numbers are in `work/custom2d_comparison_v1/analysis_v3/`.
Adoption for production is **parked**. Neither a swap reduction nor an advantage of full
replacement has been demonstrated.

[HTML report with embedded videos and graphs](../../work/custom2d_comparison_v1/comparison_report.html)

## Measured values, main 20 s

f75270:78590, 3320 native frames, 166 fps. The main comparison uses λ=0.767.

| Diagnostic | A saved SLEAP | B link audit | C custom + short-term re-evaluation |
|---|---:|---:|---:|
| Links with a large 2D move | 18 | 0 | 0 |
| 2D tracklets in the interval (3-view total) | 16 | 192 | 185 |
| Moves > 10 mm, right after C28 | 18 | 15 | 15 |
| Moves > 10 mm, after post-processing | 3 | 4 | 3 |
| Moves > 4 mm, after post-processing | 21 | 19 | 17 |
| 3D missing (individual × frame / 9960) | 2 | 2 | 2 |
| female endpoint intervals (kept / changed) | 2 / 1 | 2 / 1 | 2 / 1 |

No detection was deleted. Large 2D moves use diagnostic thresholds of top 60 and side
50 px/native frame. Cutting alone lowers the large-move count, so it is reported alongside
fragmentation. 3D moves are counted per individual over adjacent frames. If an exchange makes
2 individuals jump, that counts as 2, so this is not a count of swap events. The female's changed
interval stays f76698→f77285 under every method.

## Review scenes

- **f31479:** the maximum move in the 1 s clip, 174.92 mm, is 12.81 mm under B/C.
  Moves > 10 mm: 6 / 3 / 2. B changes the slot assignment in 18 frames, and the set of
  three-view correspondences in each frame is the same. The correct ID is not confirmed.
- **f30735:** moves > 10 mm: 2 / 2 / 2. A cross-view correspondence problem remains.
- **f323819 (no swap in human review):** C changes the correspondence in 8 frames, and
  moves > 10 mm rose 4→6. There is a risk of erroneous correction.
- **f76791 / f77415 (no swap):** C changed the correspondence in 1 / 10 frames respectively.
  The number of changes is not treated as a number of ID improvements.

The 8 clips are 166 native frames each. f76930 and f76988 overlap by 108 frames.
The original review did not specify the reviewed side or the exact exchange interval. The
centre numbers are not the true onset times.

## Short-term candidate re-evaluation

The v2 custom method only split ambiguous links conservatively, so v3 added plan §3's
short-term hypothesis retention and selection by subsequent observations. Details in the
[fixed protocol](deferred_protocol.md). A minimal implementation restricted to 2 fish,
2 possible continuations, 3 observations on each side, gaps of at most 5 native frames, and
fragments that end and start at the same time.
Three-fish contention, insufficient candidates and insufficient margin are not force-linked.
Intermediate detections keep their original fragment membership.

In the main 20 s, 10 boundary candidate pairs were rejected and 0 accepted. Across all 20
intervals, one candidate in regression_90200 side2, f90576→90578, added 2 links, and the
tracklet count within that view's evaluation core went 9→7 (17→15 including the context before
and after).
The common support is the pectoral fin only. Prediction-error scores 25.0322 vs 50.7135 px,
margin 25.6813 px (required margin 4 px). This is not a verification of correct ID.
Changes to 3D source/xyz/pose/uv from the added processing: 0 in all 20 intervals.
Only the internal C28 state numbers in the changed interval differ.

## λ sensitivity comparison and reproducibility

With the main 20 s 2D labels fixed, 7 values of λ were compared. For A/B/C alike, recomputation
at λ=.767 matches all 24 saved arrays, the tally, female and the prefix alignment exactly.

| λ | 0 | .1 | .25 | .5 | .767 | 2 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A final moves > 10 mm | 18 | 7 | 6 | 4 | 3 | 2 | 2 |
| B | 9 | 7 | 6 | 4 | 4 | 3 | 3 |
| C | 14 | 6 | 5 | 3 | 3 | 2 | 2 |

At λ=0, slot namings with the same geometric cost tie, and that flicker is also included in the
move counts. The table above is not a count of biological swaps, nor the result of selecting an
optimal λ.

A's final detection correspondence and coordinates match the saved C34 exactly in the evaluation
core of all 20 intervals. The C28 stage is reproduced exactly in 15 intervals; in known_30735,
known_72143, known_99209, known_323819 and regression_82121 the short-window solution differs.
For the main 20 s and f31479, C28 also matches exactly. Agreement over the whole context region
is not claimed.

In the separate 20 s f240000:243320, moves > 10 mm are 0 under every method. The female endpoint
annotation covers only 1 interval, kept under every method. This is the same group of fish, not
ID ground truth for all 3 fish and not an independent biological replicate.
Move counts from the overlapping 20 intervals are not summed as independent events.

## Verification and artifacts

74 tests pass in total: 62 for tracking, short-term hypotheses, geometry, tallying and
rendering, and 12 lab checks. An independent audit of the saved arrays confirmed A/B byte
equality, detection retention, within-frame one-to-one assignment, retention of the original
C fragment membership, validity of the selected detection indices, and a recount of moves and
missing observations. Runs including the first-run bug were kept in a separate directory with a
code snapshot. Thresholds were not changed to fit performance on real data.

The 10 videos and 9 stills show all 3 views and all 3 fish. Decoding of all 2094 video frames and
the SHA of the 12 input NPZ and 19 output files were checked. Time-series PNGs for the 8 scenes
were also made. The overview shows the 20 s in real time, subsampled to 30 fps; the local videos
show all 166 native frames in slow motion. No coordinate interpolation or smoothing was applied.

Evidence: `analysis_v3/summary.json`, `review_windows.json`, `validation.json`,
`sensitivity_v3/summary.json`, `review_v3/manifest.json`,
`event_figures/manifest.json` and each provenance record.
Production C27/C28/C34, the source detections and the raw video were not changed.
