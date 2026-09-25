# Raw v2 configuration freeze and sealed validation

The raw configuration decision and source edits were fixed on 2026-09-08 before
inference on the four registered validation windows and the one-minute interval.
This addendum and JSON record were written immediately after the minute process
started (11:47:36 JST), while its outcomes remained unopened. Thus the decision
preceded execution, but this written addendum did not. The run's own startup
code snapshots and hashes confirm the selected source. The locations and seeds
in `reacquisition_windows.json` remain unchanged. Selection used **development
results only**.

## Fixed raw inference

- Use `ReacquisitionTracker` with `RecoveryConfig()` defaults, replacing only
  `config.local.lower_mm` with `(0, 0, -6)`. Keep the optical x/y/upper domain.
- Use `FastFitTracker`'s equivalent sparse finite-difference Jacobian and batched
  image-score projection. Three recorded development frames showed identical
  source/observed/review with pose differences of approximately 1.67e-11 mm;
  this is a numerical regression, not a whole-recording equality guarantee.
- After sustained missing support or selected reacquisition, retain raw ID review
  even for one lost fish. A return of 3D observations alone does not clear identity.
  The raw geometric candidate objective and measurement rules are unchanged.
- Use CUDA background processing, three CPU view workers, flow maximum side320,
  and the ordered capacity2 prefetch queue. No native frames are omitted.
- Initialize once from the registered initial pose; use no subsequent baseline
  or annotation reset. The minute is exactly 76620:86580, 9960 frames.

Floor tolerance6 was selected because the development interval's observed rate
rose from78.83% to98.75% and its female endpoints became matched to the same slot;
the transient's observation/source/review assignments were unchanged. The floor
is not surveyed by this comparison. There were911 observed nodes below nominal
z=0 in the floor-tolerant interval, and the minimum touched approximately-6mm.
Keep that sensitivity visible and do not widen the floor for validation outcomes.
The accepted output and suspected male exchange in v1 remain documented failures.

## Identity postpass and information barrier

The six-permutation, fixed-geometry identity postpass is a separate algorithm and
command with separate provenance. It may be developed and configured using only
the two previously examined development intervals. Raw inference is independent
of that postpass, so validation and the minute may compute while it is finalized.

Until the postpass configuration is frozen and its source/config hashes recorded:

1. Do not open validation/minute `candidate.npz`, `comparison.json`, or previews,
   and do not inspect their ID, geometry, quality or coverage outcomes.
2. It is permissible to monitor progress, process errors, completion, output
   existence and hashes. Runner stdout exposes only progress/technical completion.
3. Do not change raw v2 in response to those sealed runs. Preserve failed outputs;
   any necessary technical correction must be explicitly versioned.

After the postpass freeze, evaluate all four validation windows and the complete
minute with the same shared metrics, including unsuccessful and unscored cases.
Preserve raw and postpass arrays separately. The postpass cannot modify coordinates
except by a recorded permutation, restore missing geometry, or use female labels
inside inference. Ground truth is read for scoring only after the postpass finishes.

## Completion and timing

The user requires the minute run even if the short-window availability/identity
assessment fails. Completing9960 frames is not a quality pass. Three-animal identity
remains unverified without independent dense annotations; female endpoints do not
resolve that limitation. Registered coverage thresholds stay90% overall and80%
for each individual, with unavailable output retained in the denominator.

Validation and minute computation may overlap to finish the authorized task.
Record that CPU/GPU contention affects wall time; do not label concurrent timing
an isolated hardware benchmark. The preceding measured rate of about1.7frames/s
implies roughly98minutes for9960 frames before any new speedup or resource contention.

Runnable snapshots and hashes of raw source are stored in every output. The
registration, this addendum and `reacquisition_v2_freeze.json` are durable records.
The JSON records the already-selected source/config hashes after minute startup,
without inspecting validation outcomes; startup snapshots remain the contemporaneous
record of the code actually used.

## Postpass freeze metadata correction

The postpass defaults were fixed without opening validation/minute outcomes.
Before any standard `benchmark_reconnect.py` run began, its developer corrected
diagnostic metadata once: the physically-feasible-global-path flag now also
includes the individual-gap audit, and adjacent/block feasibility is reported
separately. Inference arrays and configuration were unchanged. Final `reconnect.py`
SHA256 is `8bb6f013ba9cb1f86bc1e78c84054f6e5f69ecca838921410e955ba5904157db`,
superseding `865c19c66685af9587ad0b2f8c34a5aefaba0ec5a60c5137db002c62d933d9cd`.
The metadata correction and its recorded time are retained in the postpass
developer's freeze-history record; no raw-source file changed in this correction.

## Postpass wrapper storage correction

After the first six standard postpass runs (four development runs and validation
swap A/B) completed, and while every standard postpass process was idle, only
`benchmark_reconnect.py` and its wrapper tests were corrected. The inference core
and `ReconnectConfig()` remained frozen. Missing transitive source dependencies
were added to future snapshots. An actual all-paths-infeasible result reproduced
the JSON `global_cost=inf` serialization error; JSON diagnostics now use null for
nonfinite values and list the affected fields. NPZ arrays and the infeasibility /
review decisions are preserved. The two remaining validation postpasses ran with
this corrected wrapper; the six earlier outputs remain untouched.

The corrected wrapper passed 3 focused tests and Ruff before these last two runs.
This correction note was recorded on 2026-09-08 JST, after those runs finished.
Final wrapper SHA256:
`9c86fd51c47c8f96f5cc1b6bb447f32525d2f042cf7fd5cfba3f66bff81b6213`.
The raw one-minute inference was running throughout and no raw hash target changed.
