# V3 development: mask-grid paths as additional temporal starts

The 100- and 400-step v2 experiments leave S2 outside foreground at frame 333544.
Spatially resampled image error improves little and still favours the competing
endpoint correspondence on image terms alone. The independent-mask grid supplies
distant alternatives, but a globally ranked 48-point pool can omit supported
regions. These observed failures motivate this additional-start experiment.

Use the same 31 native frames, original first anatomy, all six endpoint
correspondences, and the exact saved 256 fitted pixels per view/frame from the
400-step v2 run. Do not modify its twelve results. Generate one extra start for
each endpoint correspondence and optimize it for 400 steps with the same v2
objective, weights, learning rate and fixed internal boundary constraints.
Recompute the frozen twelve costs under the current objective before comparison;
a numerical mismatch invalidates the comparison. Keep every result, including
unsuccessful new starts, in a separate artifact.

Generate at most 48 spatially diverse candidates from every saved voxel with at
least two-view mask support, including known-third-view contradictions. Select
24 using image support and distance from already selected points, then up to 24
using the additional soft penalty near the other supported fish. The latter is
conditioned on the current SLEAP-supported scene, not an independent identity
measurement. This finite pool still cannot guarantee coverage of every region.

Use v2's scene-to-linear assignment for each endpoint correspondence. This first
prototype permits at most one physically unsupported pectoral per frame. Offer
translations of that body's whole three-node pose to the selected grid points,
plus the original scene fallback. Reject translations outside the unchanged
engineering bounds, without clipping anatomy. Record every rejection. Spatial
candidates are optimizer starts, not anatomical landmark measurements.

Find a first-order dynamic-programming path through the finite full-scene states.
Its image emission is 3 times one minus mean known mask support at the proposed
centre; fallback emission is 3. Its transition is the existing soft speed term,
0.05 times the mean across fish of squared pectoral displacement divided by
3 mm squared. This proposal objective omits acceleration and full body rendering;
the subsequent joint v2 optimization restores those terms. Preserve the first
and last two scenes' original supported boundary coordinates exactly.

Compare all eighteen hypotheses by fresh full-objective scores, direct node
support, latent jumps, body shape and raw-image previews. Reuse the saved disjoint
pixel evaluation without resampling after looking at the result. Do not change
accepted coordinates or clear ID review from any grid/path/fit score. This known
short development interval does not establish one-minute or thirty-minute IDs.
