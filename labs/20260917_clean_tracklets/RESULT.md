# Result — parked

The requested ID-independent 2D fragment builder and cross-view completion
layer are implemented and executable. They are not promoted over the existing
tracking chain: the new 3D estimator leaves too many observations/identities
unresolved in the primary difficult sequence to claim fewer swaps.

The final comparison is `work/clean_tracklets_v2`; the initial complete run is
retained in `work/clean_tracklets_v1`. The
[HTML report](../../work/clean_tracklets_v1/clean_tracklets_report.html) contains
eight reviewed scenes, a 20-second overview and native-frame graphs.

## Primary 20 seconds, native f75270–78589

Every one of the 29,386 detector rows is retained by all P/N/F conditions.
No SLEAP temporal ID is an input. SLEAP's within-frame M/P/T grouping remains
an input assumption. Conditions N/F associate mouth and pectoral; tail can
veto inconsistent bundles but cannot settle an identity tie.

| View | P fragments | N fragments | F fragments | F incoming clean edges / detection rows |
|---|---:|---:|---:|---:|
| top | 72 | 372 | 385 | 96.13% |
| side1 | 99 | 412 | 424 | 95.72% |
| side2 | 152 | 473 | 515 | 94.71% |

These are fragmentation and coverage diagnostics, not identity purity.
Incoming edges ending inside the core include its first entry from context.

The final estimator independently verifies two-view support before recording
joins in a separate layer. Six joins have both endpoints inside the primary
20 seconds; all six reconnect adjacent-frame fragments. They are not six
restored occlusions or six corrected swaps. The known f31479 interval includes
one gap-spanning join in top_cam, f31460→31463, crossing two intermediate
unassigned view observations. Original pixels and raw 2D components remain
unchanged. Synthetic tests separately demonstrate 20-frame missing-view support.

| Sequence | Saved C34 missing fish-frames | New missing fish-frames | Possible fish-frames | New observed but ID-review | Joins with both endpoints in core |
|---|---:|---:|---:|---:|---:|
| primary 20 s | 2 | 5,191 | 9,960 | 3,778 | 6 |
| known 30735 | 1 | 457 | 1,032 | 508 | 0 |
| known 31479 | 0 | 269 | 999 | 78 | 5 |
| known 323819, human no-swap | 0 | 10 | 1,017 | 283 | 13 |
| control 50000 | 0 | 231 | 996 | 0 | 0 |
| secondary 20 s | 0 | 197 | 9,960 | 3,134 | 28 |

The primary 3D observation rate is 4,769/9,960 = 47.88%; 991 observed
fish-frames have no estimator review flag, which is not biological ID truth.
All three primary female endpoint intervals are unscored. The secondary
20-second interval preserves its one scored female endpoint interval. No
male-versus-male or all-fish ID accuracy claim is available.

Every new native step is ≤4 mm because the estimator explicitly gates recent
motion at 4 mm/native frame. Its >4/>10 mm counts are therefore not independent
evidence of improved tracking. Missing observations and unscored female
intervals must accompany any such plot. Two-view reconstructions, third-view
predictions, and observed third-view pixels are stored separately.

## Observation-level consensus revision

The initial run used the complete three-view source tuple as the unit of
abstention. A synthetic third-view competitor reproduced an unnecessary loss
of two agreed measured views. `multiview_consensus.py` preserves the sources
common to **every** competitive joint assignment, only when at least two
remain and their independent triangulation passes the same gates. Identity
competition with no common pair remains unresolved.

All P/N/F NPZ and metadata files are byte-identical between v1 and v2. Numerical
thresholds, objective, motion model and calibration are unchanged. Winner and
margin parity are tested. v1 remains executable; its primary observation rate
was 47.10%. v2 preserves 84 observations through source consensus in the core,
but its sequential history changes mean the net gain is 78 fish-frames, not
84. This fixes one failure mechanism without solving the main coverage loss.

## Verification and remaining limits

- 50 focused tests pass: independent 2D association, variable cardinality,
  local node swaps, crossing isolation, forward/backward/row-order invariance,
  real synthetic-video optical flow, cross-view joins, source consensus and
  independent audit failure cases. Ruff passes for pipeline and tests.
- Both complete six-case outputs pass independent audits: original C01 rows
  and points retained, components contiguous, no detection reused by two fish,
  no projection written as a detection, join unions exactly reconstructed,
  every accepted bridge recomputed from the other two views, and summary
  metrics matched to saved arrays. Inputs/code have provenance and hashes.
- Joins can only be considered where the current 3D trajectory already has
  enough continuous two-view support. The auxiliary join layer does not run
  a second global 3D optimization. The 3D temporal estimator is forward-only;
  only the initial 2D association uses forward/backward agreement.
- The first owner of a clean fragment remains an anchor. The current objective
  gives a soft bonus to exactly one clean-view vote and constrains two-view
  votes rather than using a monotone support score. These are explicit
  remaining design choices, not established optimal tracking rules.
- Long unanchored 3D reacquisition sets a persistent ID-review flag. Improving
  the geometry observation count alone cannot remove this uncertainty. The
  next algorithmic work should evaluate clean-fragment 3D assignment and
  reacquisition with independent identity evidence, without relaxing gates
  merely to make these known clips look better.
- Eight human notes refer to earlier videos, not new per-frame truth. There
  is one trio; some windows overlap; the secondary interval is not a fresh
  holdout. The previous pipeline, reports and manual annotations are intact.
