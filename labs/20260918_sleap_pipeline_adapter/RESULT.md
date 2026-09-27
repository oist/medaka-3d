# SLEAP-based 2D tracking: design and parameter comparison

**Verdict: parked (not promoted).** Implementation of the improved version, runs of
all settings, the compatibility audit and the comparison HTML are complete. The
persistent fragmentation from P1's self-competition was resolved. Because ground
truth for physical-fish individual correspondence is lacking, the acceptance
criterion of not increasing wrong links is undetermined. Deadline 2026-10-18.

[HTML report](../../work/sleap_pipeline_adapter_v1/report.html)

## Main 20 s results (three views combined)

All conditions retain 29,386 rows. The adjacent link rate's denominator is 29,377
observations; it is not ID accuracy.

| Condition | Contiguous intervals | Adjacent link rate |
|---|---:|---:|
| S1 | 217 | 99.29% |
| S2 | 208 | 99.32% |
| SP | 211 | 99.31% |
| P0 | 323 | 98.93% |
| P1 | 11,615 | 60.49% |
| G0 | 593 | 98.01% |
| G_history3 | 417 | 98.61% |
| G_history9 | 1,356 | 95.41% |
| G_match025 | 482 | 98.39% |
| G_match10 | 952 | 96.79% |
| G_margin02 | 397 | 98.68% |
| G_margin10 | 1,199 | 95.95% |
| G_age8 | 586 | 98.04% |
| G_age33 | 600 | 97.99% |

SP is standard SLEAP with the mouth removed from S2, computing fixed-area OKS on the
pectoral fin only. G0 is a design comparison adding history expiry, an unmatched
option and a commit test to SP; it is not a single change. The 9 G settings are
pre-specified one-factor changes. No best setting was selected and no per-scene
tuning was done. G/SLEAP use past history; P0/P1 use past and future history. The 332
input frames on each side are shared.

## What the design and tuning established

- Runs SLEAP's actual Tracker, per-track histories, mean OKS and Hungarian
  assignment.
- Separates provisional IDs from committed fragments. Cutting a fragment for
  insufficient candidate margin does not multiply provisional histories.
- A lone fish over 120 frames with a single 9px change, which P1 split into 105
  fragments, is 1 fragment under G0. A case held temporarily by a strict candidate
  margin and a case that creates a new history on an 80px change also recover to
  continuation.
- The 584 G0 unlinked observations in the main 20 s: candidate margin 284, dummy 162,
  gap candidate 133, non-mutual 5. 273 insufficient-margin cases fall below the
  effective commit floor of 0.10 from dummy 0.05 + margin 0.05.
- At side1 f75286, despite a preceding displacement of 4.64px and 317px to the other
  detection, history 5→9 lowers the mean OKS 0.166→0.092. Independently recomputed
  from the distances to past positions; the values agree.
- History 3 and candidate margin 0.02 are comparison candidates with high continuity.
  The correctness of the added links is not evaluated, however. A condition applying
  both together was not run.

## Pipeline compatibility and limits

Every detection carries a nonnegative committed fragment ID (singleton fragments
included). Provisional IDs of rows with missing P may be -1. Original points, dropout
mask, detection score and global_row are not changed. Committed G fragments are
adjacent-only, and provisional correspondences across a gap are separated into
reconnection candidates. Reflections are unclassified; no physical-fish ID names are
assigned.

The output was actually passed to the existing C25 compact_layout, verifying that all
rows are retained and that source offsets round-trip. This covers input compatibility
for 3D candidates only; it does not verify any improvement in 3D geometry or
physical-fish ID. The current C25 takes at most 12 detections and raises an explicit
error beyond that, and there is a separate constraint of at most 3 slots per 3D
state. Existing caveats also remain for C24's returned-row order correspondence and
for C28's offsets on sub-intervals with a nonzero start. The NPZ from this run must
not be substituted unconditionally into whole-recording C27; use an explicit
global_row correspondence.

## Verification and reproduction

- 12 known intervals × 3 views × 10 new conditions = 360 arrays; with the 5 existing
  conditions, 540 comparison records.
- 54 tests, Ruff, and checks of all input arrays, one-to-one assignment, 0 committed
  G links across a gap, and execution with changed row order in 3 views.
- SHA-256 match confirmed between 4 relevant files of official SLEAP v0.3.1 and the
  installed environment.
- Verified the SHA-256 of the 3 source videos, the data for 36 frame views, the 36
  graphs and the HTML.
- Checked in a browser: source-video playback, frame navigation, method switching and
  display of 0/5 detections.
- All intervals were already investigated; this is not a new holdout. The same 3
  fish, 1 independent group.

```powershell
.venv/Scripts/python.exe labs/20260918_sleap_pipeline_adapter/run.py --init --output <new-output>
.venv/Scripts/python.exe labs/20260918_sleap_pipeline_adapter/run.py --view top_cam --output <new-output>
# Run side1_cam / side2_cam the same way, then --summarize last
```

The existing output is frozen. run.py refuses a change in code SHA.
audit.py/build_report.py/diagnose.py/recovery_check.py verify and render this fixed
output.
Settings, verification, numbers, evidence JSON and each provenance record are linked
from the HTML.
