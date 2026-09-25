# Dense export from sparse human correspondences

2026-09-09. This implementation addendum follows HUMAN_CORRECTION_PROTOCOL.md.
The original automatic result and review documents are immutable. No real
human review has yet been received. Synthetic reviews belong only to tests.

The new pure `review_correction.py` compiles actual endpoint constraints into
conditional intervals. `run_review_correction.py` verifies completed input,
native frames, SLEAP sources, calibration, raw review probe pixels and every
input/source digest before writing a separate output directory. It rejects
unreviewed templates. An empty review list is a zero-review control.

## Identity convention and evidence

G0/G1/G2 mean the initial raw slots. They are conditional anonymous global
names; they do not name the female or either male. An initial slot receives a
global anchor only when its initial tracklet has neither a recorded preceding
risk nor an initial review flag. Later data blocks are not automatically seeded.
Clip-local A/B/C names are never merged across documents by spelling.

Actual confirmed links and uniquely matched, directly supported mouths provide
the human evidence. A link p -> q reaches a global name only if the left
endpoint already has one uniquely supported global name, by an initial
conditional interval or earlier confirmed links. Otherwise it remains local.
The existing extractor's raw competitor matching, source gate and content
hashes remain in force. This exporter does not replace missing body candidates.

## Intervals and geometry

For a usable link, the same identity's previous machine continuation ends after
p. Its unannotated interior p < frame < q is unknown, including when the two
endpoints use the same raw slot. An additional independently connected human
endpoint inside that interval may supply that endpoint alone. This is not
interpolation or an inferred switch time.

At q a confirmed identity may be attached to the existing node coordinates
only if q is outside the finite active risk intervals. Following q, machine
propagation is restricted to that raw slot's current risk-bounded tracklet.
It starts at q and ends before the next applicable risk or source-data boundary.
It never extends backwards from q to the beginning of its tracklet.

A q inside an active risk can retain endpoint identity evidence for a later
confirmed link, but emits no accepted coordinates there and does not resume
automatically when that risk ends. An explicit later confirmed endpoint is
needed. Unknown-extent boundaries are connection barriers, not geometrical
episodes lasting forever. They can be crossed by a confirmed link with a
known left endpoint and eligible right endpoint. The earlier cause remains
unknown and the interior stays unfilled. The current real-data loader/extractor
accepts consecutive source sequences; a sparse source-gap review interface
has not been added.

The target raw slot's pre-q historical labels are not inherited as the new
global identity's history. The complete old risk ledger is preserved separately,
and each human claim records the crossed boundaries/ended episodes it addresses.
The risk detector can miss swaps: all following intervals remain conditional
machine propagation, never densely verified human truth.

Every pose, source row and observed flag is selected with the same mapping.
Unobserved nodes remain NaN in measured coordinate arrays. A mapped latent
pose is saved only as a diagnostic. Unknown identities have mapping -1,
sources -1, observed false and coordinate NaNs. No new SLEAP assignment or
3D fit is performed. Original automatic coordinates and reviews are retained
in a byte-identical copy of the original candidate NPZ.

## Partial confirmation and conflicts

Human A at q can contradict a different identity's machine continuation on
that raw slot. The unsupported machine identity is withheld from p+1; it is
not assigned to a remaining slot. Its dependent claims are revalidated, so a
later claim whose left global identity depended on the invalidated continuation
also becomes unresolved. A later human endpoint can remain valid if it is
connected to an anchor preceding that cut; its following machine continuation
is separately limited by the new evidence.

Human-versus-human contradictions withhold the conflicting connections and
their dependent propagation. All links ending at one frame are handled as a
set; JSON order is not priority. Duplicate documents do not add evidence.
There is no last-writer override, naming by elimination, or new approval role.

## Output and evaluation

The separate output contains `raw_geometry.npz`, exact copies of supplied
review JSONs, extracted constraints, `correction.npz`, correction metadata,
summary, source snapshots and a success/failure provenance record. Mapping,
identity origin, claim and tracklet indices accompany coordinates. Identity
origin distinguishes initial automation, a human endpoint, machine propagation
after a human endpoint, and unresolved data. Metadata retains all links,
dependency cuts, raw risk history and unavailable/local-only correspondences.

The zero-review control can reduce availability because the existing ledger
cuts at risk onset rather than the fourth missing frame or sticky flag alone.
It must not create new accepted positions or identity permutations. Such a
reduction is risk-policy bookkeeping, not improved tracking accuracy.

The real corrected-versus-automatic comparison requires actual human JSON.
Correction probes measure application consistency, not independent accuracy;
different probes/windows are required to validate identity preservation. No
zero-swap minute, thirty-minute result or sufficient manual-review budget is
claimed by passing software tests or running the zero-review control.
