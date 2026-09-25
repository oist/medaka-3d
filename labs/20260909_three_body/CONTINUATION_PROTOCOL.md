# Preserve a continuous trajectory across process boundaries

2026-09-09. Operational extension for the registered thirty-minute goal. This is
not a new identity model, a promotion of the current model, or a claim that the
minute or thirty-minute biological identities are correct. Actual human review
JSON has not yet been supplied. The original frozen tracker and runs remain intact.

## State and numerical contract

The new `continuous_tracker.py` separates the persistent loop into begin,
advance-one-native-frame and final backtrace. The frame's numerical operation
order is copied from `persistent_node_tracker.py`. Proposals, fit, source gates,
beam ordering, review rules, motion, node ages and final 21 arrays are unchanged.
Exactly one logical initial pose is supplied. A fresh process can restore a
completed-frame snapshot; restoring state is recorded separately from seeding.

A snapshot contains all ordered active beam states and every retained history
record with its parent index. Active velocity, missing/body review, cumulative
cost and node histories are retained. The complete preceding FrameObservation,
including raw BGR and independent masks with None distinct from an empty array,
is preserved when flow is enabled. Restored flow uses that frame and the same
current first beam pose. Source frame numbers, not chunk offsets, seed sampling.

Only a fully processed frame can be checkpointed. A fit/step exception makes
that in-memory tracker unusable; recovery must use an earlier saved snapshot.
Empty checkpoints are not supported. Configuration, implementation/backend,
initial geometry and actual rig parameters must match on restore. Every history
parent and the derived active velocity/age/support/cost state are revalidated.
No final chosen trajectory is substituted for the complete hypothesis history.

NPZ/JSON checkpoints contain no pickle or object arrays. The codec verifies
array shape, dtype and bytes, file digests, metadata and caller-bound inputs and
source versions. A success manifest is written last in a new directory. The
runner binds that manifest to its own provenance and preserves source snapshots.
GPU fit graph caches are rebuilt, since they do not carry temporal inference
state; equivalence of a cold cache is an experimental question checked below.

`resumable_observations.iter_from` seeks within the same sealed complete input
descriptor. A staged FFV1 clip uses a relative decode offset, whereas a source
MP4 uses its absolute native frame. It never renumbers masks/SLEAP/frames or
rewrites a shorter manifest. Previous flow evidence comes from the snapshot.

## Registered development comparison

First use the completed persistent 200-frame window [333519,333719), at the
original native166fps with the original seed/configuration/Detectron2/raw inputs.
Use RTX5070Ti, float64 fitting and the original two Torch CPU threads. No human
marks or baseline coordinates after the initial seed enter tracking. Completed
producer bytes may be hashed before inference, but its candidate arrays are
loaded for comparison only after the new full trajectory has been saved.

Compare one fresh uninterrupted process with a chain of three fresh processes:
stop after85 completed frames, restore and stop after121 completed frames, then
restore and finish200. These split positions are fixed before this comparison.
Each process starts with an empty CUDA graph cache. Require all21 output arrays
to agree in dtype, shape and bytes with the uninterrupted process and original
completed persistent output. Also compare recovery counters. Record actual
processed ranges, one logical initialization, two restores, per-phase wall time,
checkpoint size and input/source/output hashes. A mismatch is a failed operational
gate and must be diagnosed before extending this execution path.

Independent CPU tests cover parent-branch reversal, missing-node thresholds,
sticky review, full history, previous-frame flow, snapshot copying, invalid
context/parent/index data, source offsets and interrupted writes. They are
software evidence, not animal identity ground truth.

## Scope and remaining long-run work

The first checkpoint format stores the full retained history. Its file size,
serialization work and restore memory grow with processed frames. Repeated
snapshots do not make history streaming or give a thirty-minute memory/runtime
guarantee. The eventual snapshot frequency must be chosen from measured costs.

The registered long source interval is [49800,348600). Its first-frame seed
audit is not yet a completed runner registration; full independent mask staging
and the actual long tracker run remain to be completed. The background policy
also differs from the frozen minute. Passing the200-frame continuation comparison
permits this operational path to be considered for further development; it does
not establish long-run performance, sufficient manual-review budget or zero swaps.
