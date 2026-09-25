# Fixed-pose SLEAP association control

2026-09-09, registered before the new real-data rescore. The completed A body
pool and its thirteen development frames are fixed. No new fit, frame, detector
threshold, motion weight, mask term, acceptance rule or animal label is added.

Current per-node assignment can join different SLEAP rows into one body and
share one row between different bodies' anatomical nodes. The new control
tests that feasible set, without assuming the detector's grouping is correct.

Use all 1,384 admitted **post-fit** A scenes and their frozen non-node scores.
Retain three distinct columns: original score with its fixed starting plan;
fresh exact per-node assignment; fresh exact coherent-row assignment. Both
fresh controls use the same robust forward-half-ray distance, the original
float32 ray cache converted to float64, sigma .7 mm, missing cost 9, weights
[1,1,.35] and denominator 9*sum(weights). Original assignment generation used
infinite-line distances; its saved fitting score used the half-ray objective.
Do not attribute a change from saved to fresh solely to row coherence.

For each view, coherent assignment maps a body to zero or one SLEAP row, and
each real row to at most one body. Null assignments are unrestricted. Within
an assigned row each node may be an outlier (the missing-cost alternative).
The row stays exclusive even if some nodes are dropped. Row numbers are local
to the camera and current frame; equal row numbers across views have no role.
Enumerate all improving partial injections, without truncating the row pool.
Canonicalize rows with no improvement over all-missing to null. Count exact
ties; representative choices are deterministic but do not establish identity.

For every pose, coherent node score must be no lower than the fresh nodewise
minimum, within numerical tolerance. Recompute the original fixed-plan node
score against the saved score (1e-12 absolute/relative tolerance). Verify row
permutation invariance of minimum cost and node/row exclusivity in synthetic
tests. Preserve finite-pool alternatives even when they score poorly.

For original/refitted, mask and random pools report original, relaxed and
coherent winners, their unchanged spatial-evaluation errors, row conflicts,
anatomical direct support and score components. Select only by fitting score;
the same-frame disjoint pixels remain diagnostic, never selection truth.
Existing A per-instance regions may be inspected after selection, but must
not be substituted into this registered score.

This control cannot remove the constant cost of a truly missed fish, improve
an inaccurate body representation, remove reflections or fix a wrong SLEAP
instance grouping. It can only expose or prevent combinations that join rows.
Even if successful on these opened development frames it must undergo a new
continuous run and independent all-three-animal evaluation. No old trajectory,
review flag or accepted coordinate is changed by this rescore.
