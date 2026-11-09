# Result: parked pending independent warning validation

Date: 2026-11-04. Verdict: **parked**, not promoted. Expiry remains 2026-11-18.

The P-1-1 experiment and handoff are complete. The external blocker to promoting
the warning detector is an independently annotated recording not used to design
the narrower warning rule. This does not block use of the present P-1-1 review
queue or require the user to redo its existing annotations.

The core rule occupies 5.01--5.27% of post-preparation time and entirely contains
96.97--100% of sustained reference-change groups across the seven arms. It meets
the registered within-recording warning criterion, but not an external validation
claim. At lambda 0.767 there are 129 warnings / 90.795 s and 130 lower-risk
intervals. One short observed change bracket is not entirely contained; it does
overlap a warning. Two long close encounters remain 17.785 and 32.432 s long.

The tested automatic corrections do not meet the registered adoption rule.
The permissive C30 variant improves six fish-level endpoint edges at 0.767 with
no new wrong edge in that arm, yet increases fixed-anchor wrong-ID time from
848.417 to 3,612.210 fish-s because remaining earlier naming errors persist.
Lambda 2 also gains one wrong local endpoint edge. Position-only bridges worsen
0.767; strict bidirectional/consensus bridges make no corrections in that arm.
Keep these variants as comparison evidence inside this parked warning experiment,
not as production code or a claim that geometry has reached its limit.

See [the complete analysis result](../../analyses/20261031_p11_identity_evaluation/GEOMETRY_MOTION_RESULT.md)
for inputs, unsuccessful attempts, exact times, denominators, validation, and
artifact locations. No appearance classifier was used. No original extraction
or human labels were overwritten.
