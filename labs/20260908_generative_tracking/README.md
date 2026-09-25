---
hypothesis: Refractive 3D prediction fitted to raw images and ungrouped SLEAP nodes preserves identity through detection corruption better than the current stitched correspondence chain.
stage: linker
expires: 2026-11-08
decision_rule: >
  This first implementation is a feasibility study, not a production promotion.
  Compare identical native-rate intervals against all seven existing C34-label arms.
  Record female anchor-interval preservation and abstention separately, observed
  coverage, raw-image support, steps, and runtime. Retain as a research candidate
  only if functioning on real synchronized images with independently tested geometry;
  do not claim ID superiority from synthetic tests, fewer jumps, a single lambda,
  or window initialization. Production promotion additionally requires held-out
  all-three-animal encounters and full-recording validation with no per-window reset.
---

# Refractive generative tracking feasibility experiment

**Verdict: parked; not promoted.** See [RESULT.md](RESULT.md) for the completed
two-window comparison, failed v2 runs, v3 image ablation and remaining limitations.
Independent identity ground truth for all three fish is required to evaluate
promotion; the present implementation also demonstrably needs better re-acquisition.
Expiry extended once from 2026-10-08 to 2026-11-08 with the completed experiment.

The subsequent **reacquisition cycle** is recorded separately in
[REACQUISITION_RESULT.md](REACQUISITION_RESULT.md), with its
[registered protocol](REACQUISITION_PLAN.md) and
[raw v2 configuration freeze](REACQUISITION_V2_FREEZE.md).
Four fixed validation windows and the authorized uninterrupted 9960-frame minute
are complete. The minute used one initialization and took 3582.26 seconds with
CUDA image preprocessing. Pectoral observation coverage was 99.76%, but accepted
coverage was only 3.56%; offline ID reconnection reduced it to 3.02%. All 13 raw
female endpoint intervals were preserved, while review-excluded output left all
13 unscored. Persistent identity review and unsuccessful reconnection prevent
promotion or a swap-free claim. The original v3 code and result below remain
available for comparison.

The new executable is `benchmark_reacquisition.py`; `reacquisition.py` adds
observation-driven 3D recovery and persistent identity review, `fast_fit.py`
accelerates the same continuous fitting objective, and
`accelerated_observations.py` uses CUDA image preprocessing with three CPU view
workers. [ACCELERATION.md](ACCELERATION.md) separates image-only, fit-only, and
end-to-end timing and documents the deliberate optical-flow resolution change.

`benchmark_reconnect.py` evaluates the frozen `reconnect.py` postpass in a new
output directory. It reads only raw geometry, image features and the original
seed before inference; annotations enter afterward for scoring. It records the
per-frame ID permutation without refitting or filling coordinates. This postpass
failed to improve the fixed short-window results and is an experimental comparator.
`render_review_video.py` renders a completed run or a selected subinterval for
review; `plot_review_timeline.py` compares availability and review masks. Preview
frame sampling changes playback only; all tracking runs consume native frames.

User-authorized 2026-09-08. The observation/association/identity loop is new;
calibration, SLEAP predictions, provenance and shared evaluation are reused.
The existing chain and inputs are read-only comparators.

## Model and inference

Three persistent latent fish have three 3D nodes each (mouth, pectoral, tail),
velocity, soft segment lengths and view-specific appearance templates. Each step:

1. Predict each fish in water and forward-project it through the calibrated
   air/glass/water or air/water interfaces and lens distortion.
2. Propose injective per-node observation assignments, including missed
   observations EVEN when exactly three detections exist. SLEAP's instance
   grouping is not a hard constraint; nodes from a mixed instance may be separated.
3. Fit the projected skeleton to observed nodes and image foreground, with
   soft dynamics/shape terms. Compare view-conditioned masked appearance and flow.
4. Keep a bounded beam of alternative assignments and 3D trajectories. Backtrack
   after the final frame, so subsequent image evidence can select earlier history.

This is an approximate beam search with local continuous fitting, not an exact
whole-recording optimum or a calibrated Bayesian posterior. Beam alternatives may
be pruned before ambiguity resolves. The baseline's `min emission + 6 lambda`
pruning proof is NOT used. A small beam gap is a review signal, not an ID probability.

The `GenerativeLinker` adapter implements the existing Linker protocol with bound
raw observations; the first frame's three skeletons initialize it. Subsequent
`Instance3D` seeds are not consulted. The broader image-driven design deliberately
revisits association within linking, as explicitly requested by the user.

## Inputs and honest outputs

Only the first frame's baseline sources initialize geometry; no subsequent
baseline coordinates, IDs, or human annotations enter inference. A local-window
run is consequently a **window-initialized recovery experiment**, not evidence
of uninterrupted 35-minute tracking. IDs are anonymous 0/1/2 until independently
anchored; sex and male names are not inferred from slot numbers.

`pose_xyz` stores the latent state, including predicted missing nodes. `xyz`
stores only pectoral nodes supported by at least two assigned camera observations;
predictions are not reported as measured positions. `observed`, `node_source`,
`review`, and the uncalibrated beam-cost `margin` remain separate. No post-hoc
interpolation fills measurement gaps.

`xyz` is a raw observation-supported diagnostic and may still contain a mistaken
position flagged for review. Use `accepted_xyz` for downstream positions: it also
excludes every `review` flag. The Linker adapter returns this masked output.
Even an accepted position has not been independently certified as correct.

Image evidence uses deterministic background subtraction, soft foreground support,
masked color histograms and optical flow. It requires no new trained checkpoint.
It is not a photorealistic renderer, learned re-identification system, or dynamic
water-surface reconstruction. Merged fish masks are not independent ID evidence;
projected overlap suppresses appearance template updates. Background subtraction
cannot, by itself, tell reflections from real fish.

## Predeclared comparison

Development smoke: frames 76900:76960 (half-open), then female-anchor interval
76698:77286 if runtime allows. A separate transient-corruption diagnostic is
333519:333719. These are known development failures, not held-out examples.
Baseline: `c27 -> c28 -> c33 -> c30v4 -> c34`, all available lambda arms.
Templates/backgrounds use video pixels only; evaluation annotations stay outside
the inference module. Record initial source arm and frame explicitly.

Tests cover refractive forward/backward consistency, exclusive assignment with
clutter/missing observations, actual image evidence affecting fitted state,
shape/dynamics, honest missing outputs, future hypothesis selection, and common
evaluation invariance to a fixed permutation of ID numbers. Test-scene success
is software evidence only. Timing and real-data outcomes belong in RESULT.md.

Before running image comparisons, initialization feasibility was checked without
using identity outcomes. Frame 76900 has missing seed nodes; use 76898:76960.
Frame 76698 has an out-of-tank reconstructed tail; use 76620:77286, the latest
preceding complete, in-bounds seed, to retain the original female anchor interval.
These adjustments concern initial geometry only, not measured tracker success.

The image fit is one-sided foreground support along projected body segments,
not a full photometric renderer. It does not enforce exclusive ownership of
foreground pixels. The current conservative assignment admission also requires
the robust pixel cost to be below the missing cost (about 15.5 px with defaults),
even inside the nominal 60 px search gate. Large-gap re-acquisition is unvalidated.

### Review-rule correction after the first smoke

The initial 62-frame smoke marked every fish for review because its raw beam-cost
gap compared omitted-tail variants, not competing animal identities. Independent
code review identified this mismatch before the outcome was inspected. Version 2
keeps that filtering gap as a diagnostic but derives review from conflicting
identity permutations along the final surviving paths, unsupported mouth/pectoral,
solver nonconvergence, or a <4 mm pectoral encounter with no surviving identity
alternative. This is an uncalibrated heuristic; an absent alternative is not proof
of identity. It changes review flags, not inference or raw identity assignments.
Preserve the original smoke results. Validate review burden separately from accuracy.

## Reproduce a run

From the repository root, using the project virtual environment. Supply the
mounted medaka root explicitly; no machine-specific drive letters are built in.
The local staging directory contains checksummed copies of arrays, calibration
and annotations. Raw videos are streamed from their existing location.

```powershell
.venv/Scripts/python.exe labs/20260908_generative_tracking/stage_inputs.py --source-root '\\bucket.oist.jp\bucket\StephensU\medaka' --out work/generative_tracking_inputs
.venv/Scripts/python.exe labs/20260908_generative_tracking/benchmark.py --input-root work/generative_tracking_inputs --raw-root '\\bucket.oist.jp\bucket\StephensU\medaka\medaka_202512' --start 76620 --stop 77286 --out work/generative_tracking/anatomy_v3_interval_76620-77286
.venv/Scripts/python.exe labs/20260908_generative_tracking/benchmark.py --input-root work/generative_tracking_inputs --raw-root '\\bucket.oist.jp\bucket\StephensU\medaka\medaka_202512' --start 333519 --stop 333719 --out work/generative_tracking/anatomy_v3_transient_333519-333719
```

Commands refuse to overwrite an existing destination. A remote-share permission
is needed in sandboxed environments; no credentials are embedded or collected.
The current baseline bundle is intentionally limited to K_1_2. Changing experiment
requires a separately identified baseline bundle and its evaluation annotations.

Each run emits `candidate.npz`, `comparison.json`, a provenance sidecar, exact
source-code snapshots, and a zoomed three-camera `preview.html`. Hollow nodes in
previews are latent predictions. The baseline preview shows its existing pectoral
coordinates; female annotation scoring uses mouth observations for BOTH methods.

`female_identity_raw` describes the best retained path before review exclusions;
`female_identity_accepted` additionally excludes flagged identities. Neither counts
male swaps or swaps that return between annotation endpoints. The raw-image fit
is not dense 3D ground truth: reprojection and smoothness remain diagnostics.
The initial seed was selected by the baseline using the full recording, so this
local experiment is optimistic even though later baseline predictions never enter
the new tracker. Baseline compute time is not measured by reading cached outputs.

### Core objective and approximation

For fixed node assignments the local fit minimizes robust pixel reprojection,
distance from projected body samples to foreground, deviation from predicted 3D
motion, and deviation from initial body-segment lengths. A missing node observation
has an explicit cost. Appearance (same-view Hellinger histogram distance) and
confidence-weighted optical flow rank the fitted hypotheses. Appearance templates
freeze at the first sufficiently clear observation, without labels from GT.

Only a bounded list of assignments is fitted. A coherent all-view ID alternative
has a reserved proposal when feasible, preventing cheaper tail omissions from
crowding it out. Finite-difference Jacobians batch all perturbations through the
refractive projection, with steps directed inside the tank bounds. No baseline
lambda-dependent pruning bound is reused for this different objective.

The retained beam is backtracked after the final requested frame. This is not a
bidirectional continuous smoother: future evidence can choose a previous discrete
history, but cannot recreate a hypothesis already pruned or continuously refit all
earlier poses. All such limitations apply also to a longer invocation.

### Anatomical-model correction after the first two windows

The first complete-window implementation (v2) failed: one pectoral-to-tail segment
grew from 17.49 to 174.29 mm in the transient window, and the female's exit anchor
was lost in the 666-frame interval. Both outcomes remain in the original run
directories. This is a model defect, not a successful zero-swap result.

Two independent synthetic regressions reproduced 5-10x segment elongation with
all nodes still called observed. The global soft-L1 loss downweighted anatomical
violations, while independent velocity forecasts let missing nodes drift away.
The corrected implementation (v3) therefore keeps robust losses on image/node
observations ONLY, uses quadratic motion/anatomy terms, and predicts a shared
body translation. A fit with any segment >2x its seed length is refused for that
fish, retaining a latent body prediction and clearing its measured observations.
This generous upper bound is an engineering guard, not validated medaka anatomy;
a strongly curved or biased seed may cause false abstention. Rerun BOTH windows
into new `anatomy_v3_*` destinations; do not replace or hide the failed runs.

### Additional control, declared before inspecting v3 outcomes

Run the same v3 tracker, configuration, initial seed and both complete windows
with `FrameInput.evidence=None` using the separate `run_without_images.py` command.
This removes foreground, appearance and optical-flow evidence together while
retaining SLEAP observations, refractive projection, body constraints and search.
It measures whether raw images change recovery in these development windows;
it does not identify the contribution of each image term or establish held-out
superiority. No weights are retuned for either condition. Both outcomes are
reported even if the image-free control performs better. Its timing excludes
video reading/preprocessing and cannot be compared as equal-work throughput.

```powershell
.venv/Scripts/python.exe labs/20260908_generative_tracking/run_without_images.py --input-root work/generative_tracking_inputs --start 76620 --stop 77286 --out work/generative_tracking/ablation_no_images_interval_76620-77286
.venv/Scripts/python.exe labs/20260908_generative_tracking/run_without_images.py --input-root work/generative_tracking_inputs --start 333519 --stop 333719 --out work/generative_tracking/ablation_no_images_transient_333519-333719
```

These destinations now contain the completed comparisons. Choose a fresh output
directory for a reproduction; original results are never overwritten.
