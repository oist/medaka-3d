# Geometry, motion, and localized identity warnings on P-1-1

Date: 2026-11-04. Owner: P-1-1 research team; execution by Codex.
Question: can existing geometry/motion improve connections, and can likely
flip intervals be isolated for checking their clean endpoints?

## Findings

**Motion improves some local connections, but the tested changes do not safely
improve trial-wide naming. Geometry concentrates the review queue substantially.**
No appearance classifier, new pose inference, or new human annotations were used.
The existing manual annotation is evaluation evidence throughout, including
sections of mixed local tracks. Additional video adjudication was not a starting
requirement. Original extraction outputs and manual labels remain unchanged.

At the previously displayed lambda 0.767:

| Quantity | Result |
| --- | ---: |
| Full post-preparation duration | 1,800.266 s |
| Geometry-only warning intervals | 129 |
| Warning duration, including 0.05 s padding | 90.795 s (5.043%) |
| Warning duration median / 95th percentile | 0.193 / 1.062 s |
| Warnings at most 1 s long | 121 of 129 |
| Longest warning | 32.432 s |
| Sustained reference-change brackets, grouped by overlap | 21 |
| Groups entirely contained by warnings | 21 of 21 |
| All observed fish-level correspondence changes entirely contained | 94 of 95 |
| All observed changes overlapping a warning | 95 of 95 |
| Lower-risk local intervals | 130, totaling 1,709.468 s |
| Longest lower-risk local interval | 110.803 s |

The 21 groups are not 21 independently adjudicated crossings or physical swap
events. Sustained changes require at least 0.1 matched seconds and 10 observations
on both sides. Missing observations retain an onset bracket. Short changes are
also exported, rather than disappearing from the evaluation. The one incomplete
raw bracket is M1 at 369.089--369.192 s: it overlaps a warning, but is not fully
contained. All seven lambda arms retain this distinction.

The geometry-only core uses separation below 10 mm, adjacent steps above 10 mm,
non-identity nearest assignments, or missing positions. It pads by 0.05 s and
unions overlapping intervals. Reference labels never enter this detector.
Separating ray-count churn from identity warnings reduced the initial queue
from 1,649 intervals / 401.022 s to the 129 intervals / 90.795 s above.
Ray-count changes and residual-only alerts remain observation-quality evidence.

Across all seven arms, warning duration is **90.144--94.789 s (5.01--5.27%)**.
Full containment of sustained groups is **96.97--100%**: lambda 0 contains 64
of 66, with both remaining groups overlapping warnings; the other arms contain
every group. These are retrospective results on one recording. The narrower
rule was motivated by the first comparison on this recording, so the reported
time blocks are not independent held-out validation.

## What the motion comparison actually establishes

The C30 ablation changes only `VELOCITY_NEEDS_MEDIAN_MM` from 2.5 to 0.0,
retaining the other v4 rules, fixed C33 positions, and the original C34 pass.
This is the historical permissive velocity condition, not a parameter fitted
to the known 932.879 s failure. All seven arms were run.

| Lambda | Original wrong-ID fish-s | Permissive C30 wrong-ID fish-s | Corrected endpoint edges | Newly wrong endpoint edges |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3,271.779 | 3,038.189 | 4 | 0 |
| 0.1 | 3,883.358 | 3,349.457 | 4 | 0 |
| 0.25 | 2,678.673 | 1,611.095 | 4 | 0 |
| 0.5 | 2,935.217 | 3,018.590 | 4 | 0 |
| 0.767 | 848.417 | 3,612.210 | 6 | 0 |
| 2 | 4,285.441 | 3,579.723 | 5 | 1 |
| 8 | 4,133.675 | 3,692.955 | 4 | 0 |

Edges here connect successive matched observations of the same reference fish;
adjacent observations and gap endpoints are separate in the exports. They are
not independent biological replicates or physical event counts. At 0.767, the
six improved edges are the two males around **539.71, 932.879, and 1467.647 s**.
The male-1 edge near 539.71 s crosses a missing observation. There are 886,117
assessable fish-level endpoint edges, including ordinary continuous frames;
their overwhelming easy majority must not be presented as crossing accuracy.

Fewer incorrect connections can coexist with more wrong-name time. Some earlier
errors are accidentally cancelled by later errors. Correcting only some of
those later edges changes the net permutation while leaving earlier errors.
For example, at 600.017 s the baseline reference-to-slot vector is `[0,1,2]`,
while the candidate is `[2,1,0]`; it retains an earlier wrong naming even though
the local continuity audit records no new wrong endpoint edge in this arm.
This is why both connection and fixed-anchor measures are necessary. At 0.767,
the candidate repairs 229.873 fish-s but newly misnames 2,993.665 fish-s.

The first experiment also compared clean-end bridges on the broader warning
set. At 0.767, position-only bridging increased wrong-ID time to 2,878.434
fish-s. Forward/backward velocity agreement and position-plus-velocity agreement
made no corrections in this arm. On the seven-arm comparison they also failed
the stated adoption rule. These findings do not establish a general limit of
geometry; they reject these particular changes as a safe automatic replacement.

## Where to inspect

Ten of the 129 core warnings contain the 21 sustained reference groups. This
retrospective prioritization uses the existing annotation and is not available
to a detector on an unlabeled recording. It is useful for the present P-1-1
review:

| Recording time, s | Duration, s | Reference groups |
| --- | ---: | ---: |
| 510.108--527.893 | 17.785 | 1 |
| 536.364--536.605 | 0.241 | 1 |
| 544.498--544.829 | 0.331 | 1 |
| 559.156--559.307 | 0.151 | 1 |
| 613.681--646.113 | 32.432 | 12 |
| 932.831--932.963 | 0.133 | 1 |
| 1158.516--1158.625 | 0.108 | 1 |
| 1467.593--1467.707 | 0.114 | 1 |
| 1499.531--1499.639 | 0.108 | 1 |
| 1972.851--1972.996 | 0.145 | 1 |

The two long close encounters occupy 50.217 s of the queue. They were not
artificially divided into short clips. Stable separated endpoints, not a clip
length target, should govern any further subdivision. The queue CSV includes
native frames and 0.5 s review context on each side; 90.795 s is warning time,
not a measured human review time or total context-inclusive playback time.

All 130 lower-risk intervals have at most one matched reference identity per
slot in the observations available inside that interval. Some reference coverage
is incomplete (one slot/interval has none), which the CSV reports. These are
local continuity candidates, not verified trial-wide names: **815.558 wrong-ID
fish-s still lie outside the warnings** in the original fixed-anchor output.
The purpose is to check/correct correspondence at the boundaries and then carry
supported names through the locally consistent sections. Unsupported boundaries
must stop accepted identity propagation; their positions remain available.

The reference-endpoint oracle is only a diagnostic. It composes links where all
three exact endpoint matches exist and does not reset names where evidence is
missing. On the narrower warning set, it worsens 0.767 to 3,110.186 wrong-ID
fish-s. It is not an attainable-accuracy claim or an upper bound. Future endpoint
review should establish supported correspondence over clean context, and retain
unresolved links explicitly, rather than propagate names through skipped checks.

## Inputs, artifacts, and checks

The frozen source is `id_row_mapping_20261102_r1`, starting from repository
revision `fd8d86f81715239b39ef8496299768b43d0679c4` plus the documented working
changes. Every consumed extraction array is checked against its saved artifact
manifest. The manual reference SHA-256 is
`4f188db9b7ac32a282234ef3ed2c84529f6949f1ff2e9b35c181e525f80c26f0`.
Native top-camera acquisition timing and the existing same-index camera alignment
are retained. Nothing is inferred from an assumed 166 fps conversion. The full
expected denominator remains 5,400.797 fish-s; the initial partial frame remains
unprocessed rather than being silently removed.

Outputs are under the configured data parent, `jingyi/P-1-1/`:

- `geometry_motion_risk_20261104_r1`: first bridge and broad-warning comparison.
- `geometry_motion_risk_20261104_r2`: preserved failed comparison attempt.
- **`geometry_motion_risk_20261104_r3`**: valid narrower-warning/C30 comparison.
- `r3/report_v1/review_queue.html`: searchable warning and lower-risk tables.
- `r3/report_v1/geometry_motion_review.html`: executed analysis and figures.
- `r3/audit_v1/review_queue_lam*.csv`: exact times, native frames, reasons and
  retrospective reference-group evidence, for every arm.
- `r3/lam*/lower_risk_intervals.csv`: intervals, per-slot reference coverage,
  label diversity/purity, and fixed-anchor correct/wrong time.
- `r3/audit_v1/c30_connection_changes_lam*.csv`: paired local connection changes.

`r2` stopped because the comparison adapter omitted the original label output's
frame-level availability mask. Inspection also found a C30/C34 finite-frame policy
mismatch. `r3` restores both original contracts, reruns every arm and verifies
identical matching availability. Coordinates and contributing camera sources are
exact permutations of the pinned C33 inputs. Warning duration and full-bracket
containment were independently recalculated from native-frame union masks.

The focused calculation/C30/C34/doubt tests pass: **33 tests**, with two existing
all-NaN synthetic-zone warnings. The broader lab check also exposed two unrelated
expired labs without results (`20260908_mask_volume`, `20260909_three_body`);
they were not modified. Full-suite success is not claimed. Ruff passes for the
new scope and the executed notebook. The notebook ran top-to-bottom in a fresh
project-virtualenv kernel; both saved figures were visually inspected. Windows
Jupyter emitted its nonfatal event-loop fallback and local transport notices.

## Decision and next step

Keep the original automatic connections. Carry forward the core warning detector
as a P-1-1 review aid; do not promote it as a general identity guarantee. The
existing annotation already identifies the ten warning intervals above as the
first informative checks. Establish/verify correspondence across those boundaries
using clean context, preserve the two long ambiguous stretches, and propagate
supported identities across the exported lower-risk intervals. Evaluate new
automatic connection rules against both local errors and propagated wrong-ID
time. Appearance remains untested and outside this task.
