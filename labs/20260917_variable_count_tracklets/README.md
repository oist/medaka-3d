---
hypothesis: Pectoral-first bidirectional links with explicit candidate competition and short-lived open endpoints preserve clear continuity with variable detection counts without forcing fish identity.
stage: linker
expires: 2026-10-17
decision_rule: Preserve every C01 observation and missing node, pass the structural and row-order contracts, and reduce unnecessary fragmentation without new verified wrong links in annotated clear scenes. Fragment counts or smoothness alone cannot establish identity superiority. Report every predeclared condition and leave identity claims unconfirmed where annotations do not support them.
---

# Variable-count independent 2D tracklets

Implements stages A-C of `docs/decisions/2026-09-17-variable-count-2d-tracklet-plan.md`.
This is a camera-local 2D experiment. The production Linker protocol consumes
3D instances; it cannot represent these unassigned 2D observation fragments.
The explicitly requested 2D comparison therefore remains in this isolated lab
with a common input/metric contract across all conditions, rather than making
an artificial 3D adapter. No production stage or other lab is changed.

Conditions are saved full-session C27 (reference only), row-faithful current
SLEAP S1, S2 changing only OKS scale to fixed squared body spans (top 120 px,
side 100 px), frozen P0, P1, and P1 with optional coherent M/T support.
P1 margin sensitivities are fixed at 0.015 and 0.035 body spans, compared with
the base 0.025. They are diagnostics, never scene-specific selected winners.
Every method gets the same C01 rows and 332 native frames on both sides.
SLEAP is causal; independent P0/P1 use forward and backward context.

The original eleven cases and three views are required. Additional temporal
validation is selected using metadata before viewing its tracking results,
then run after the settings are frozen. This remains one trio from one
recording, not independent-group generalization. Human reviews are event
anchors with limited view/panel scope, not all-fish framewise 2D truth.

Outputs live in a new `work/variable_count_tracklets_v1/`. Points, observed
mask, original global row and detection score remain intact. Predictions and
gap reconnection candidates are separate; gaps never merge P1 tracklets.
Real-fish/reflection interpretation starts unknown. A confident time link
does not establish that an image is a real fish.

The final `RESULT.md` records observed results, limitations and verification.
The HTML includes comparison movies, frame inspection, static images, native
frame diagnostics, machine-readable evidence and all settings.

## Entry points

- `prepare_audit.py`: metadata exclusions and SHA snapshot of earlier artifacts.
- `run.py --init --unseen-selection work/variable_count_tracklets_v1/selection_audit.json`:
  freeze inputs/settings/code. An existing protocol is intentionally refused.
- `run.py --view top_cam` (also `side1_cam`, `side2_cam`): all conditions/cases;
  `run.py --summarize`: shared metrics, strata and numerical manifest.
- `tracking.py`: independent P1 API; `baseline.py`: current SLEAP controls;
  `frozen_p0.py`: byte-identical prior pectoral tracker.
- `render.py`, `case_findings.py`, `review_audit.py`, `build_report.py`: media,
  scientific evidence and HTML. `audit.py` independently checks every output.
- `serve.py`: loopback preview with MP4 byte ranges; the report is at
  `http://127.0.0.1:8765/work/variable_count_tracklets_v1/report.html`.
- `finalize.py`: verify completed audits and browser/test receipts, write verdict
  and final SHA manifest. Receipts record performed checks, never inferred passes.

Use `.venv/Scripts/python.exe` for these entry points. To rerun numerical work,
choose an unused output with `run.py --output ...`; never overwrite the frozen
protocol or earlier results. P1 output tables reference C01 global rows. Frame
intervals are half-open. The report's auxiliary graphics are derived solely
from the saved outputs and do not change the frozen tracker.
