# Deferred two-hypothesis comparison, revision 3

Frozen on 2026-09-17 before running this refinement on real data. This is an
addition to the preserved revision-2 tracker; `tracking.py` and its numerical
thresholds remain unchanged. The v2 lambda sensitivity remains a separate result.

The public API is `refine_deferred(frame_idx, points, base_result, *, view,
config=None) -> TrackingResult`, with `DeferredConfig`. It accepts the saved v2
labels/events/metadata directly; no SLEAP rerun, image, 3D coordinate, reference
permutation, or human identity annotation is used.

1. Treat each v2 consensus tracklet as fixed. Never remove an existing link or
   relabel a subset of a tracklet. Every input detection retains a nonnegative
   label, and every label remains unique within a frame.
2. Consider only an ambiguity/disagreement logged by v2, bracketed by two stable
   left tracklets ending on the same native frame and two stable right tracklets
   beginning on the same later native frame. Other forms of ambiguity remain
   declined. Synchronous boundaries are an explicit limitation of this first
   implementation.
3. Each flank needs three consecutive native observations: its boundary point
   plus two past/future observations. A hypothesis is the complete continuation
   of the two left histories into the three observations of the two right
   tracklets, with either of the two one-to-one mappings. The reverse prediction
   supplies a symmetric check. Each hypothesis has separate histories; competing
   histories are never mixed into one active tracker queue.
4. The left boundary to the last right support observation, and the first left
   support observation to the right boundary, must each span at most **5 native
   frame intervals**. Thus, with three consecutive support observations, a
   bridge spans at most 3 native intervals (at most 2 missing/ambiguous frames).
   A sparse array must not turn a longer time gap into a short one.
5. Use only nodes finite in every support observation of both fish on both
   flanks. This identical support set is used for both hypotheses; at least one
   of mouth/pectoral is required. Tail is excluded. The existing 2:1 pectoral:
   mouth weighting is retained. Flank velocity is the median of its two native
   finite-difference velocities. No prediction crosses a rejected hard-speed
   observation within a support flank.
6. For each mapping, predict the three right observations from each left flank,
   and the three left observations from the corresponding right flank. Average
   their six weighted positional errors for each fish; sum the two fish scores.
   Both candidates have the same observation count and node support. Accept only
   when the runner-up minus best score is at least the unchanged **0.04 body**
   margin (top 4.8 px, side 4 px), all individual prediction errors satisfy the
   unchanged **0.20 body × [1 + 0.25 × (native distance - 1)]** gate, and bridging
   node speeds satisfy the unchanged **0.50 body/frame** hard gate. Body scales
   remain top 120 px / side 100 px. No parameter is tuned to real-data outcomes.
7. Build boundary candidate components using the same physical and prediction
   gates. Only a component with exactly two left and two right tracklets and two
   possible bijections is eligible. Three-way components are refused. Also refuse
   if three or more detections lie within the two candidates' gated motion
   neighbourhood at any frame spanning the supports, so an ongoing third fish
   cannot silently be dropped from a collision. A far, unrelated third fish is
   retained without entering this component.
8. A clear winner joins only the corresponding fixed flank tracklets. Uncertain
   detections inside the gap keep their existing separate labels. A tie remains
   a refusal; no minimum-fragment or smoothest-output fallback is allowed. If
   accepted proposals compete for the same incoming/outgoing endpoint, refuse
   all competing proposals rather than choosing by iteration order.
9. Keep the v2 result, candidate hypotheses, scores, support nodes, reasons,
   deadlines and accepted/refused decisions auditable. Record a distinct v3
   algorithm revision and new output directory. All improvements remain
   hypotheses about tracking, not validated biological identity accuracy.

Required synthetic checks: later observations resolve a two-fish ambiguity;
true ties remain declined; a third participant forces refusal but a distant
third fish stays untouched; native-time expiry and missing support are respected;
time reversal and row order preserve the inferred partition; original confirmed
links and every detection survive. These checks precede any real-data execution.
