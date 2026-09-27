# Observation-recovery candidates restricted to a persistently missing mouth

2026-09-09. The immediate-addition version reduced pectoral-fin support from 586 to 565/600 over
200 contiguous frames, with acceptance unchanged at 417/600. Because the mouth was supported
just before the first mouth candidate, this separate arm stops adding candidates for transient
prediction misses. The immediate version, the original v1, and the existing results and sources
are not changed.

PersistentNodeConfig(enabled=True, minimum_missing_frames=4,
minimum_pectoral_frames=4) is fixed. A current mouth candidate is allowed only for a fish that,
over the native frames each beam parent has completed so far, meets both conditions: the mouth
directly unsupported for 4 or more consecutive frames, and the pectoral fin directly supported
for 4 or more consecutive frames. That the mouth is currently still directly unsupported and the
pectoral fin directly supported is checked with the existing helper. The initial age is 0, so
addition becomes possible at the 5th input frame at the earliest.

node_missing and direct_pectoral_age are propagated explicitly from each beam parent. A mouth
supported by an unselected candidate does not clear the age of the selected branch. Mask-based
body support is not counted toward the direct pectoral-fin age. Body-missing age, velocity and
identity review keep the original processing order, and mouth recovery does not clear review.
The track loop is a copy that explicitly records the original frozen implementation as the base
it derives from, adding age initialization, update, addition conditions and saving. Parents are
not inferred from pose contents or memory addresses.

The original 18 candidates are retained and at most 6 mouth candidates are added. beam 6, the
14-step fit, flow, D2, refractive calibration, the image and node objectives, and the body-length
and ray/box conditions of NodeRecoveryConfig are fixed. The already-verified CUDA Graph fitter is
used. The 4 frames here are an engineering development setting, equivalent to about 24 ms at
166 fps. They are not a biologically validated threshold or an identity confidence.

The CLI `run_persistent_node_recovery.py` is a separate file based on the immediate-version
runner. It accepts only --sequence / --input-root / --output, and uses all native frames of the
same registered 200, 666 and 9960 intervals, the same initial pose once, and 0 external resets.
The 19 original arrays and 2 kinds of age array are saved, and the ages are checked to match
values recomputed from the exact observation support of the final backtrace. The original v1,
all 7 C34 and the female annotations are opened after tracking completes, and all
inference-input SHAs, the original configuration and the seed are verified. The new source and
this protocol are snapshotted, and the before/after SHAs of inputs and code are verified.

The following screening conditions are fixed before running.

1. Over all 200 frames of 333519:333719, direct pectoral-fin support does not fall below 586/600
   and acceptance does not fall below 417/600.
2. After passing, over all 666 frames of 76620:77286, acceptance does not fall below 1942/1998,
   and the two female annotations match the initial S1. Unscored is not counted as kept.
3. Only if both pass, the contiguous minute 76620:86580 is compared without mid-run
   reinitialization.

All of these are already-opened development intervals. Meeting the criteria does not mean
correct identities or achievement of the 30 minutes; it is only the condition for running the
next computation. The one-minute acceptance criteria of 90% overall and 80% per fish, and the
conditions for independent identity evaluation, are not changed. On failure, the results are
kept and the cause is investigated; if thresholds are changed to fit the results, that is
recorded as yet another development version. Zero swaps are not claimed without independent
correspondences for all 3 fish.
