# Scope for using human checks of a few scenes to correct identities

2026-09-09, implementation in stages. A pure-CPU helper that builds endpoint correspondence
constraints has been implemented.
There is not yet any correction code for dense trajectories, corrected output or human
ground-truth labels.
This document is a design applied to a separate output after completion, and does not change
the running tracking, the fixed automatic evaluation, the original candidate, raw images or the
review JSON.

The user has agreed to checking and correcting a few scenes. Additional approval or
adjudication by another person is not required. Whether short positional gaps are acceptable
has not been answered, so identity is provisionally prioritized, and positions and identity
correspondences without backing are left as NaN. Mathematically error-free identity,
maintenance over the full 30 minutes, and achieving this with a few corrections cannot yet be
guaranteed.

## 1. What the human JSON shows

The input is `raw_encounter_review_v1`, produced by [review_truth.py](review_truth.py).
A/B/C are anonymous names a human assigned at the entry of this clip, and do not automatically
correspond to A/B/C of another clip, fixed IDs for the whole video, or the female, the white
male and the red male. Biological names depend on separate `biological_naming` evidence. Names
are not inferred from courtship roles or existing slots.

`confirmed` on each individual's `entry_to_after` and `after_to_exit` is used as a
correspondence in which a human stated, on grounds such as direct visual continuity, that "the
fish at both checkpoints is the same".
The mouth clicks at 3 time points do not annotate the candidate ID in each frame of the interval
or the time of a switch.
Even with the same entry and exit ID, the possibility remains that the fish swap partway and
swap back.

The template saved at present is `unreviewed_template_not_ground_truth` and cannot be used as
ground truth. The proposed time points are 333519, 333621 and 333718, and are not committed
until a human selects and confirms them.
An actually saved human JSON is required; the assistant does not fill in blanks.
A status such as `draft` saved by the human is kept as is, as provenance, and the
per-correspondence confirmed is used. The old text at the top of `review_truth.py` that suggests
additional adjudication is not applied as a new approval requirement. This handling matches the
implementation of [evaluate_review.py](evaluate_review.py).

## 2. Conditions for committing a correspondence at a checkpoint

The same rules as the existing [shared evaluation](../../src/medaka3d/eval/encounter_review.py)
are used, and nothing is forced onto an arbitrary nearest neighbour far from the human check.

1. The review's manifest SHA, experiment, 166 fps native frame number and 1200×1200 mouth
   coordinates are checked against the source images. The candidate is complete, and the
   calibration, input and output SHAs are linked to it.
2. At each probe, the latent mouths of all 3 slots are projected with refraction. Competing slots
   that are unobserved or under review are also kept, and one-to-one correspondences within
   5 px in every view the human clicked are examined. Only an individual that is the same in
   every assignment matching the maximum number is unique. Uniqueness is not manufactured by
   removing competing slots first.
3. The matched mouth is required to be directly supported by detection rays from at least 2 of
   the original cameras. Mask support, finite predicted coordinates or the assumption of 3 fish
   alone do not substitute for this. Even if the mouth check succeeds, no position support is
   added for the pectoral fin or tail.
4. The automatic method's review is applied after the correspondence computation. When the
   correspondence for a correction proposal is examined, the raw-side unique correspondence may
   be used, but the original accepted-side evaluation and hold records are not changed.

Probes that are non-unique, far, unprojectable, or whose mouth is not observed are left
unmatched. Even if a human can confirm the physical fish, if no candidate tracklet represents
that fish, relabelling identities alone cannot recover it. A check at another supported time
point, or improved position tracking, is needed.

## 3. The unit of correction, and uncertain intervals

A correction is saved as a link "confirmed earlier tracklet → later tracklet confirmed as the
same fish".
Raw slots are not swapped wholesale for the rest of the video. Tracklet references that retain
position, detection source and observation flags are held separately from the identity
correspondence over their time range.

Let p be the earlier probe and q the later probe. Given a confirmed correspondence and unique
mouth correspondences at both probes, the correspondence of the same fish at p and q can be
saved as a constraint. Only when the entry ID connects to an earlier confirmed link is it bound
to the same anonymous ID on the whole-video side. If the entry itself is unresolved, only an
A/B/C correspondence within the clip is obtained at first, and it is not treated as identity
recovery for the whole video.

If the switch time is unknown, the corrected identity-labelled positions of that fish for
p<frame<q are NaN.
A switch time is not decided from interpolation, velocity minimization, the nearest time point
or a low model loss and then automatically accepted.
The gap can be narrowed only when there are additional human time-point checks or the like that
bound it before and after.
This principle does not change even in the case "entry and exit are the same slot". 3 probes give
at most 2 links, but they are not dense identity annotation over all 6 individual-intervals.

Identity propagation beyond q is recorded separately as **conditional machine propagation**
starting from the human-confirmed correspondence at q. This does not mean a human checked each
frame. Correspondences are attached to existing observed coordinates only within the range of
the later tracklet delimited by the risks of the next section. Nodes whose position is not
directly supported stay NaN. The original latent pose is saved separately for diagnostics.

## 4. Limited clearing of sticky review

A review that has once become true is not simply rewritten to false. A separate ledger is
created that distinguishes old risks resolved by a human from unresolved or subsequent risks.
The output carries at least
`raw_review`, the review SHA and probe that ground the resolution, the start and end of
application, unresolved risk IDs, and
`identity_origin` (automatic / human probe correspondence / subsequent machine propagation /
unresolved).

The first implementation uses the fixed rules of [review_events.py](review_events.py).
A body-unsupported episode of 4 or more frames is a risk from the start of the dropout, not only
from the 4th frame, and repeated dropouts already under sticky review are counted every time.
A finite `score_margin <= 1` episode is a separate risk. A scalar margin does not identify the
pair concerned, so rather than restricting it to one pair by guesswork, all 3 IDs are
conservatively treated as affected. A margin of +inf means "no other candidate compared" and is
not proof of safety.
A break in native frames is also a link boundary.

What can be cleared at q is only the affected IDs of risks that ended before q and are covered by
the present confirmed correspondence. Nothing is cleared if q lies inside a risk, q's mouth is
unobserved, an unresolved risk connecting to the entry remains, or the mapping is not
one-to-one. Continuous machine propagation stops just before the start of the next applicable
risk and holds again from that start. Geometric non-acceptance and missing direct observations
are not cleared either.

A recovery that became body-supported through the mask alone is no substitute for a human mouth
correspondence.
Under the existing risk settings, a position gap of fewer than 4 frames is not a sufficient
condition for an identity break, but its coordinates are NaN, and the identity bridging it is
not presented as densely guaranteed by a human.

This risk detection itself cannot detect every swap. A swap can occur even when detection-row
exclusivity, a smooth trajectory and agreement with the mask all hold. The interval after
clearing is therefore "propagation assuming there is no undetected risk", and is verified at
independent separate scenes and checkpoints. The existing sticky flag and scalar margin alone
cannot fully reconstruct past competitors or reasons, and an unknown target is not
unconditionally narrowed toward the safe side.

## 5. Partial confirmation, overlap and contradiction

| Situation | Scope of application |
|---|---|
| Only A confirmed, B/C ambiguous | Only A's unique link is retained. B/C are not decided from the remaining slots. Nothing is cleared either if A cannot be matched exclusively against unresolved competing slots. |
| entry→after ambiguous, only after→exit confirmed | The local tracklet link for the second half can be recorded. It is not carried back to the entry or to whole-video IDs. The existing evaluation keeps it unscored for lack of continuity to the entry. |
| A new risk starts while an earlier risk is ongoing | The union of affected IDs and time intervals is held as unresolved. Not cleared until a new probe lies after both and resolves every required correspondence. |
| Another review assigns the same tracklet to a different fish | Not overwritten by the JSON read later. The conflicting connected component and its dependants are held, and the contradiction is recorded. No new adjudication or approval is requested automatically; parts that existing information cannot resolve are left unresolved. |
| One scene has A↔slot0, another scene also A↔slot0 | A is a separate name in each clip. They are not the same fish unless there is a correspondence at an overlapping time point or an already established tracklet link. |
| All fish missing, or the mouth correspondence at recovery is non-unique | Recovery IDs are not decided from the count of 3 fish or from predicted positions. The affected identity-labelled output is missing until a human confirmed correspondence and a supported later probe are both available. |
| Another risk after a confirmed interval | The earlier check does not resolve the new risk. It is held again as an independent unresolved risk. |

The scene that groups up to 166 frames and the context of 83 frames before and after are display
units that group the material to look at, not units of resolution. All risks within the same
scene are not counted as resolved by one decision. Committing one ID does not automatically name
the other two IDs by exclusion.

## 6. Required data and limits of the current JSON

What the current JSON provides is up to 9 mouth time-point checks, 2 correspondences per fish,
and provenance of reviewer, time and grounds. There is not yet an actually saved human JSON.
Applying it requires, in addition to the existing source images, calibration and completed
candidate, the following derived information.

- A risk ledger over the whole target time, and time-delimited tracklet IDs. Repeated risks,
  unknown targets and risks truncated at interval ends are kept. The JSON contains neither this
  ledger nor switch times.
- The raw one-to-one correspondence table and observation support for each probe, the earlier and
  later tracklets of each link, the scope of anonymous names, and grounding SHAs.
  This makes it possible to trace which human correspondence later machine propagation depends
  on.
- To connect to whole-video IDs, the existing identity grounds of the entry tracklet. Without
  them, a check on the corresponding earlier raw images is needed. A/B/C within the current clip
  alone cannot substitute for this.

The minimal cases that may need additional clicks are "one time point after recovery where a
unique mouth is visible" when the candidate mouth at q is unobserved or non-unique, or
"correspondences just before and just after the uncertain interval" when the gap is to be
narrowed. If no check is possible, ambiguous and the gap are left. Success with a few corrections
is not reported while implicitly assuming more human work. If biological names are wanted,
evidence that confirms that fish's appearance and sex is needed, separately from the links.

## 7. Evaluation, and checks before implementation

The fully automatic original output, correspondences confirmed by human probes, and corrected
machine propagation are compared as separate evaluation arms. Agreement with the same probes
used for correction is application consistency, not independent accuracy validation.
Generalization after correction is verified on later probes not used for correction and on
other scenes. Unscored intervals among the 6 individual-intervals are not removed from the
denominator, and the position observation rate, identity-labelled acceptance rate, unresolved
risks, and the number of human scenes, correspondence decisions and clicks and the time actually
required are reported alongside. If the review time was not entered, it is unknown.

The small regression tests at implementation include: 3 probes with a swap only partway, partial
confirmed, unresolved entry, overlapping risks, re-dropout after clearing, a margin with unknown
target, 2 contradictory reviews, all fish missing, and NaN competitors and unobserved competitors.
When IDs are permuted, not only pose but sources, observed, review and tracklet references are
all handled by the same correspondence, and no source data or unobserved positions are
fabricated.
Frozen sources, original outputs and human JSON are not changed.

## 8. The first scope implemented

[review_constraints.py](review_constraints.py) takes the bytes of the actual review and manifest,
reuses the shared input validation and raw unique correspondence, and retains all 9 endpoints and
6 requested correspondences.
A constraint between endpoints is created only when confirmed and both mouths are uniquely and
directly supported. A check on the second half only is retained as a local constraint on the
second half, and is not tied to an unconfirmed entry or to whole-video IDs.

Reviews are not joined merely by the name A in another review. Endpoint graphs with the same
candidate, calibration and projection source are joined, and a contradiction in which several
slots at the same time become the same fish, or a mismatch in endpoint content, holds the
connected component and every link depending on it.
The last review read does not overwrite, and the 2 unconfirmed fish are not inferred from the
remaining assignment. 21 CPU tests and Ruff passed.

This helper performs no I/O and computes the SHAs of the original review and manifest bytes
itself.
Checking the SHAs of candidate and calibration files and of the projection implementation must
be done by a future caller. That a declared SHA was passed in is not taken to mean that the
actual file has been verified. No actual review JSON has been applied yet, and tracklet identity
propagation, reordering of dense trajectories and clearing of review are not implemented.
