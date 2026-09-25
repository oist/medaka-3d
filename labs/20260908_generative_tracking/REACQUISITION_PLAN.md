# Reacquisition cycle: registered before new tracker results

Authorized 2026-09-08. The user requested implementation and an uninterrupted
one-minute run, including GPU use where available. This document and
[reacquisition_windows.json](reacquisition_windows.json) fix the evaluation
locations before running the new candidate. Existing v1/v2/v3 code, inputs and
results remain reproducible and are not overwritten.

## Question and bounded change

Can a fish lost by the v3 local association search be recovered from current
three-view observations and connected to its earlier identity without assigning
an unsupported ID? Separate absence from the proposal set, wrong score ordering,
and pruning of a useful branch in the two previously examined development windows.
Then test a bounded broad 3D reacquisition mechanism and short trajectory
consistency. Do not retrain SLEAP, learn a new identity classifier, or use baseline
or human labels to reset a lost fish.

Development-only diagnosis also found low ray-residual observations at approximately
z=-1.4 to -4.4 mm near frames 77104–77152, outside v3's hard z>=0 domain. The
calibration manifest does not certify the absolute floor/world-origin gauge.
Therefore compare nominal z>=0 with a bottom-only engineering tolerance z>=-6 mm
on the two development windows before freezing the validation configuration.
The 6 mm value was selected from already examined development deviations; it is
not an independently measured floor location or a biological ground truth.
Keep x/y limits and the upper optical interface domain unchanged. Do not refit
calibration or clamp/translate the output coordinates. Record observations outside
the nominal tank, per-animal counts, and minimum observed and latent z separately.
State which floor policy is selected before the four fixed validation clips and
the minute; the tolerance comparison must not become hidden parameter selection
on those results.

The exact tracker configuration and source snapshots are recorded for each run.
Freeze the candidate before the four validation windows. If those results lead
to a change, record the change and call subsequent results development reuse;
they cannot remain a fresh validation set. GPU/parallel image preprocessing must
be checked against the reference inputs and its numerical differences recorded.
Faster execution is not evidence of better identity.

The new cycle selects CUDA background preprocessing with three CPU view workers
and `flow_max_side=320`. This explicitly changes v3's 640-pixel optical-flow
resolution. In the separate eight-frame performance check, CPU/CUDA at matching
resolution produced identical maps, whereas 640→320 changed sampled flow vectors
(median difference approximately 0.494 px, p95 1.478 px). The eight-frame tracker
path was unchanged; that does not guarantee whole-window equality. Development
comparisons must include this declared observation change. A single producer uses
a queue holding at most two frames, plus one being prepared, while the main thread
fits the current frame. It must preserve frame order, previous-frame relationships
and exception propagation.

## Fixed intervals

All intervals are half-open native frame indices in `20251225_K_1_2_FCMC`,
166 frames/s. The recording contains 348600 frames. All four validation windows
are disjoint from one another, the earlier two development windows, and the
one-minute interval. The one-minute interval intentionally includes the first
development window and the known 76930 failure; it is a continuous stress test,
not an independent held-out sample. There is one animal group.

| Role | Start | Stop | Frames | Seconds | Selection basis |
|---|---:|---:|---:|---:|---|
| Development interval | 76620 | 77286 | 666 | 4.012 | Previously run female-anchor failure |
| Development transient | 333519 | 333719 | 200 | 1.205 | Previously run loss of S2 and transient corruption |
| Validation swap A | 99070 | 99900 | 830 | 5.000 | Existing event annotation 99209–99213; female marks 99156, 99870 |
| Validation swap B | 105085 | 106080 | 995 | 5.994 | Existing event annotation 105541; female marks 105304, 106021 |
| Validation occlusion | 323660 | 324490 | 830 | 5.000 | Existing annotation 323819–323825 mentions occlusion/reflection; female marks 323691, 324489 |
| Validation temporal control | 165950 | 166780 | 830 | 5.000 | Time-position control with female marks 166025, 166503; no behavior inferred from that designation |
| Continuous one minute | 76620 | 86580 | 9960 | 60.000 | User-requested no-reset stress test containing 76930; 14 female marks |

The event table was inherited and already known to this project. The three
event-selected validation clips are newly evaluated for this candidate, not
previously unknown failures or a random sample. Candidate outputs were not used
to choose any location. Female annotation **frame locations**, not candidate
matches, were used to make short-interval evaluation possible. Existing event
slot names do not supply a new independent male-identity ground truth.

Before inference, initialization feasibility alone was checked using frozen
C34 lambda 0.767 sources and the calibrated camera rays. The rule was the latest
frame at or within 166 frames before the preselected start with all three baseline
slots accepted, nine finite nodes each supported by at least two views, all nodes
inside [0,192] × [0,192] × [0,186] mm, and nonzero segment lengths. Stop frames
were fixed. Requested start 105250 required moving back to 105085; its extra
165 frames preserve both female marks and approximately 2.75 seconds before the
event. Every other start passed without adjustment. The JSON stores the exact
source rows, initial poses, baseline hash, segment lengths and geometric residual.
This is a feasibility check, not independent certification of the seed identity.

## Initialization and input restrictions

Each short run has exactly one initialization. The one-minute run instantiates
one tracker once at frame 76620 and passes all 9960 frames to one `track` call.
It must not concatenate separately seeded windows. Reacquisition from current
raw images/SLEAP detections is allowed; baseline positions, baseline IDs, or
ground-truth labels after initialization are not tracker inputs. The initial
baseline result itself used the whole recording, so this remains an optimistic
initialization protocol. Slot 0/1/2 are anonymous and are not declared female/M1/M2.

All three videos are read at native rate with sequential frame-position checks.
This checks decoder positions, not independent acquisition timestamp synchronization.
Fixed background sample frames are 20000, 40000, 60000, 80000, 100000, 160000,
220000, 280000, 340000, excluding samples inside each target interval plus a
332-frame guard on either side. Backgrounds and appearance use no identity GT.
Any future video used to form these backgrounds is declared offline image
context, not a baseline identity reset.

## Comparisons and distinct outcomes

Use the same shared mouth-source female scorer for the candidate and all seven
frozen C34 arms. Preserve unavailable anchors in the denominator and never bridge
across an unavailable mark. Evaluate raw and review-excluded candidate output
separately. Female endpoint agreement cannot count transient swaps that return
between marks and cannot validate either male ID. Sparse human labels remain a
material limitation; no automatic or visual surrogate is renamed independent
three-animal ground truth.

Record observed and accepted pectoral coverage, each animal's coverage, all-three
coverage, missing/review episode lengths, reacquisition attempts and accepted
returns, observed-only jumps, latent body lengths, female interval outcomes,
solver/search diagnostics, image support contributions and runtime. Reacquisition
counts and coverage are availability diagnostics, never identity accuracy.
Keep raw observations, latent predictions and accepted coordinates distinct.
Use fixed preview times and the annotated event neighborhoods for visual audit;
log disagreements rather than assuming either method is correct.

There are three separate decisions:

1. **Technical completion:** all requested frames return in order from one tracker
   invocation, with one initial seed, matching input/source hashes, valid output
   shapes and honest missing/review masks. Completion does not imply quality.
2. **Provisional availability gate:** accepted pectoral coverage at least 90%
   overall and at least 80% for each individual in each validation clip. Report
   the same criteria for the minute. These thresholds prevent success by leaving
   one fish indefinitely missing; they do not certify any ID.
3. **Identity assessment:** report every available female interval and any
   independently verifiable accepted ID error. Any confirmed accepted ID error
   rules out a successful swap-free result. All-three identity quality remains
   **unverified** without sufficiently dense, independent three-animal identity
   annotations, even if female endpoints and availability gates pass.

The user's new instruction explicitly requests the minute run. Therefore execute
it as a **diagnostic continuous run even if short-clip quality gates fail**.
Do not call that progression a quality pass. A software failure, invalid inputs
or unavailable resources must be reported and corrected before a claimed complete
run. Do not silently shorten the minute, change frame rate, reset identities or
fill missing output to obtain a successful status.

No human correction enters the automatic runs. If human-assisted correction is
later evaluated, use a separate output and report correction count, time, and
post-correction errors. Do not pool it with the automatic result.

## Runtime, provenance and final record

At the previous approximately 0.66–0.69 frames/s, the minute alone would require
about four hours. GPU and CPU parallelism are authorized; record hardware,
backend, numerical parity and wall time instead of promising a speedup. Frozen
baseline arrays are not timed, so no candidate/baseline speed comparison is made.

Use `benchmark_reacquisition.py`, a separate command from frozen `benchmark.py`.
Store input/window/config/source hashes, runnable code snapshots, progress,
candidate arrays, common evaluation JSON and view overlays. Refuse existing
output destinations. Preserve failed runs. Append the actual result and any
deviation to a separate result record without rewriting this registration to
match the outcome. Thirty-minute/full-recording tracking and broad biological
generalization remain beyond this cycle.
