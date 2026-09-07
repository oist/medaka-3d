# Three-node skeleton: `mouth → pectoral → tail`

**Date:** 2026-08-04
**Requested by:** the experimenter
**Status:** seed built and shipped; **annotation not started**, so nothing here
is a result yet.

## What was asked

Extend the production skeleton from two nodes to three — insert a **pectoral
fin (centre)** node between `mouth` and `tail` — and prepare 5–10% of the round-8
training set as a HITL bootstrap, with the new node pre-placed by interpolating
between the existing two, for annotation in the three-view GUI.

## What was decided, and by whom

| Decision | Value | Who |
|---|---|---|
| Seed position | `t = 0.30` from mouth toward tail | experimenter, offered 0.5 / 0.3 / 0.25 |
| Sample size | 10% of round-8 train | experimenter |
| Node name | `pectoral` | experimenter |
| Extend the GUI | yes | experimenter |
| Partial instances | use a hidden node's coordinates as-is and interpolate from them | experimenter |
| Node order | `(mouth, pectoral, tail)` | this session — see below |
| Video layout | one mp4 per `(experiment, view)` | this session — forced by a bug, see below |

## Deliverable

```
Z:\StephensU\medaka\annotations\sleap_pectoral_seed\
    pectoral-seed-20260804T103622JST-3d5f46b7c830\
```

139 moments × 3 views = 417 view-frames, 1,251 instances, all frames embedded.
Built by [`tools/s01_pectoral_seed.py`](../../tools/s01_pectoral_seed.py);
pinned by [`tests/test_pectoral_seed.py`](../../tests/test_pectoral_seed.py).

The GUI is a **separate, self-contained bundle**:

```text
Z:\StephensU\medaka\annotations\multiview_annotator_3node\    (465 KB)
```

The first attempt shipped only the annotator `.py` next to the seed, with a
launcher whose instructions said "copy this file into your existing annotator
bundle". That assumed a machine already set up for the 2-node pass. On a PC
that had never run the annotator there was no bundle, so `uv run` started a
bare interpreter and the GUI died on
`ModuleNotFoundError: No module named 'numpy'` — reported by the experimenter,
not caught here, because every test had run against the one machine that
already had the dependencies.

The bundle now carries `pyproject.toml`, `uv.lock`, `annotator_helpers.py` and
`calibration/`, and installs everything on first run. Two further defects were
found only by running it the way a user would:

- **the `.bat` files had LF line endings and non-ASCII characters.** `cmd.exe`
  mis-parses `goto :label` and parenthesised blocks without CRLF; the launcher
  executed its own comments as commands. Both files are now CRLF and ASCII, and
  that is checked rather than assumed.
- **`"%~dp0"` as an argument silently swallows the next one.** It ends in a
  backslash, and a backslash before a closing quote escapes the quote on
  Windows, so the test runner passed 4 arguments where 5 were written. Fixed
  with a trailing `.`.
- **The launcher printed cmd.exe recovery instructions, and they were pasted
  into PowerShell.** `%USERPROFILE%` does not expand there, so `robocopy`
  cheerfully created a folder literally named `%USERPROFILE%`; `cd /d` is not a
  PowerShell command at all. The message now prints paths already expanded by
  `cmd`, so the same text works in both shells, and the bundle ships
  `setup_local.ps1` — a one-step PowerShell setup that removes the choice of
  shell entirely and rejects a destination still containing a `%`.

A general lesson worth keeping: **an error message is an interface.** These
three defects were all in text or scripts that only run when something has
already gone wrong, which is exactly the code least likely to be exercised
before a user meets it. All three now have tests.

### Mouse-wheel zoom ignored the pointer (inherited defect, fixed 2026-08-04)

Also reported by the experimenter, and **not** introduced by the skeleton work
— the diff against the pristine bundle touches no zoom, pan or focus code.

`ViewPane.set_image` pinned the scene rect to the 1200×1200 frame.
`QGraphicsView.centerOn` can only scroll within the scene rect, so once the
scaled image fitted inside the pane — every ordinary working zoom, including
the one `fit_zoom` picks — the reachable centre range collapsed to a point and
**centerOn silently became a no-op**. The pane rendered the image centred
regardless of the focus it was handed. The arithmetic in `on_pane_wheel` was
correct; the renderer threw it away.

Measured: **55–66 px of drift below zoom ~1.0 and 0 px above it**, the boundary
falling exactly where the scaled image stops fitting the viewport.

Fixing it exposed two more, both previously masked by the same clamping:

1. **Drag-to-pan was clamped identically** — up to 47 px of error, with the
   focus sitting up to 269 px off the pane centre.
2. **`back_project_view_click` could return `NaN`.** `abs(nan) < 1e-6` is
   `False`, so a degenerate Jacobian slipped past the determinant guard into
   `np.linalg.solve`, and `np.clip` preserves `NaN`. While `centerOn` was
   clamped this was harmless; with the focus made effective, one wheel click
   over the grey margin outside the frame would have thrown the pane somewhere
   meaningless. It now returns the previous focus, and the wheel anchors to the
   nearest point on the image when the cursor is off it.

This is worth recording beyond the fix itself: **the clamping bug was hiding a
latent NaN bug, and repairing the first would have shipped the second as a
regression.** The only reason it did not is that the fix was measured across
the zoom range and all three views rather than eyeballed once.

`tests/test_zoom.py` in the annotator bundle pins all three. It fails 5 of its
6 checks against the pre-fix annotator and passes against the current one — a
regression test that does not fail on the buggy version proves nothing, so that
was verified explicitly.

Also verified after the experimenter's second report: paths containing spaces
(their username does) work through `setup_local.ps1`, the launcher and the test
runner.

It also refuses to build its virtualenv on a network drive: 1–2 GB over SMB,
and it would break the checksums of the folder it landed in.

Verified end to end on a clean local copy with no prior setup —
`uv sync` from scratch, then all 48 checks passing.

The seed run directory keeps `annotator/MOVED.txt` pointing at the bundle, so
anyone who navigates to where the GUI used to be is told where it went rather
than finding nothing.

The pristine 2-node annotator is preserved at
`Z:\StephensU\medaka\annotations\multiview_annotator_source\bundle-2node-20260722\`.
It previously existed **only** under `C:\Users\machi\Downloads`, in two copies
that differ; the 2026-07-22 one is canonical — it is newer, it carries the
refractive calibration JSONs the June copy lacks, and it is the one named in the
locked HITL-v2 benchmark's provenance.

## Five things established by looking rather than assuming

1. **Round-8 top and side train are exactly synchronised.** 1,382 top frames and
   2,764 side frames form **1,382 moments with all three views present** — no
   moment is missing a view, and no view is duplicated. That is what makes a
   three-pane annotation pass possible without dropping anything.

2. **`b03_heldout_comparison`'s video re-attachment is ambiguous for the side
   packages.** It maps `frame_idx → video` by scanning `source_inds` with
   `setdefault`, but side1 and side2 carry *identical* frame indices, so every
   side frame resolves to side1 and side2 disappears. Using this, the moment
   inventory reported `('side1_cam', 'top_cam')` and no side2 at all. The direct
   `frame.video` attribute is correct and unambiguous — checked on all 4,146
   labelled frames, 0 inconsistent. **This does not invalidate any B0.3 result**:
   that tool uses the mapping only to derive a canonical `(leaf video, frame_idx)`
   key per view, and it processes top and side separately. It is recorded because
   the next person to reuse that helper for a cross-view purpose will be wrong.

3. **"Hidden but positioned" points do not exist in this data.** The rule given
   for partial instances is implemented and unit-tested, but it has **zero**
   cases to act on: across all 24,878 mouth and tail points in round-8 train,
   every one is either finite-and-visible or NaN-and-invisible. The 5.9% of
   instances called "partial" are missing *coordinates*, not hidden. 112 of the
   1,251 seeded instances therefore carry no pectoral position at all and need
   one placed by hand.

4. **Cardinality is not exactly 3 per view-frame.** 4,143 of 4,146 view-frames
   carry three instances; 2 carry four and 1 carries two. The "exactly 3 is a
   free label" claim used elsewhere holds for the *top* package and is off by
   three view-frames across all three views. Small, but it is the kind of
   assumption this project has been burned by.

5. **A 26-video package breaks the annotator's quick save.** See below.

## The video-layout bug, and why the seed was rebuilt

The first build reused round-8's own `Video` objects, giving **26 videos** —
round-8's internal partitioning of the same nine recordings, which carries no
meaning for annotation. That package opened, drew and edited correctly, but its
**quick save produced a file that could not be reloaded**: 25 videos written
against frames referencing index 25, `IndexError` on read.

That is not cosmetic. `Ctrl+S` is the quick save, and the *auto-resume* path
reads that same file back at startup — so an annotator would have lost a
session's work with no warning until they reopened the GUI.

**It was not a regression in the GUI changes.** Isolated across three
combinations: the pristine 2-node annotator fails identically on the 3-node
package, and both annotators round-trip a 9-video package. The fault is the
package shape.

Fixed at the source: each labelled frame is now attached to the **original mp4**
for its `(experiment, view)` — exactly `3 × n_experiments = 9` videos, one hop
of provenance. Every frame is re-decoded and required to be **byte-identical**
to what round 8 embedded, so changing the image source cannot quietly change an
image. The builder now also performs a no-embed re-save and reload before
declaring success, which is precisely the operation that failed.

## Node order: `(mouth, pectoral, tail)`

Anatomical, matching the requested topology. It moves `tail` from positional
index 1 to index 2 — and index 1 is now the pectoral, a plausible point on the
same animal. Any code reading by position therefore returns a wrong answer that
looks right, which is why the annotator's readers were converted to lookup by
node name and why a test asserts explicitly that the tail is *not* the index-1
point.

No existing tool is affected: nothing in the repo reads this file, and the
round-8 packages are untouched. Anything written against the returned
annotations must index by name.

## The GUI was hardcoded to two nodes

`NODE_NAMES` was a 2-tuple, `FishKeypoints` carried four explicit dicts, the
add-instance flow was a two-click state machine, and every draw / drag / delete
/ save path named the two nodes literally. A 3-node file would have opened,
drawn two of three nodes, and saved what it had.

It now reads its node list from the seed's skeleton, so a 2-node seed behaves
exactly as before (verified against the locked HITL-v2 benchmark) and the
3-node seed is editable. One latent defect was found and fixed on the way: the
**ID swap** exchanged mouth and tail by name and would have left a third node
behind — a silent half-swap in the one command that exists to correct identity
errors.

48 checks pass: 24 headless on the data model and 24 driving the real window
offscreen. Nobody has yet annotated with it.

## What this does **not** establish

- **n_groups = 1.** Every frame is Group K. One trio on three consecutive days.
- **`t = 0.30` is a prior, not a measurement.** There is no pectoral label
  anywhere in this project to fit it to. It is anatomy, asserted.
- **The seed is model-derived at one remove.** Its mouth and tail come verbatim
  from round-8 labels, ~48% of which are byte-identical to round-7 output. The
  pectoral is a deterministic function of those two points. Leaving a seeded
  point untouched therefore records a straight line, not an observation — the
  same failure mode as round-8 seeding from round-7, with arithmetic in place
  of a model.
- Nothing about whether a third node improves detection, tracking or identity.

## Next, once annotation is returned

1. **Measure the corrected `t` distribution**, per view. It says exactly how
   wrong 0.30 was, and it is free. The animal projects ~17% smaller in the side
   views, so do not pool them.
2. **Count untouched seeds.** A pectoral still at exactly `t = 0.30` was either
   correct or not looked at, and those are not the same thing. Report the number
   rather than assuming the first.
3. Register the seed and the returned annotations in `MANIFEST.json`. The locked
   HITL-v2 benchmark is still unregistered too — the same defect, already
   recorded against Gate A.
4. Only then consider a 3-node training run, and note it needs new top *and*
   side models; no 3-node checkpoint exists anywhere.
