# Asset Manifest — medaka mating pipeline reset

Separation of the old repo (`medaka-mating-pipeline`) into what survives and what does not.
Compiled 2026-08-01. All sizes measured directly, not estimated.

**Governing rule:** an asset survives only if it *cannot be regenerated*, or if regenerating it
would cost human annotation time or scarce cluster time. Everything else is derived output and
is deliberately left behind — derived artifacts are the main source of the confusion this reset
is meant to clear.

---

## Tier 0 — Irreplaceable source data

Cannot be recreated at any price. The fish are gone; the recordings are the experiment.

| Asset | Location | Size | Notes |
|---|---|---|---|
| Raw recordings | `Z:\StephensU\medaka\medaka_202512\<exp>\{top_cam,side1_cam,side2_cam}\cam*.mp4` | **1.05 TB** (1074 GB, 234 mp4 measured) | **71 experiment folders**, 3 views each, + calibration/test recordings. Dates 2025-12-20 → 2026-01-18 |
| Frame timing sidecars | same dirs: `frametimes.npy` (5.6 MB), `frametimes.mat` (2.1 MB), `metadata.csv` (~1.2 KB) | ~600 MB | `metadata.csv` `totalFrames` is the authoritative frame count used for HPC chunking |
| Calibration recordings | `Z:\...\medaka_202512\cali_251217_way{1,2,3,4}\` | 72 MB × 4 | The 190 mm 49-bead cube, 4 orientations. Source for the production calibration |
| Calibration recordings (12-ball) | `Z:\...\medaka_202512\calib_12balls\` | 1.5 GB | 2026-07-09 refinement filming |
| Calibration recordings (air) | `Z:\...\medaka_202512\calib_air\` | 14 GB | Air/water interface work |
| Test recordings | `Z:\...\medaka_202512\{G_test,I_TEST}\` | small | Keep for completeness |

**Verified counts (three independent signals agree):** 71 experiment directories, 71
`experiment_metadata.json` files, 71 rows in `catalog.jsonl`.

> An earlier automated pass reported "76 experiments" — that was wrong. 71 is correct.

### ⚠️ A second raw dataset the old repo never touched

| Asset | Location | Size |
|---|---|---|
| Pilot series, Aug–Sep 2025 | `Z:\StephensU\medaka\medaka_202508\` | **535 GB**, 84 mp4, 28 sessions |

This is an **earlier pilot series** (2025-08-26 → 2025-09-05) that appears nowhere in the old
repo: `paths.local.toml` points only at `medaka_202512`, and it has no catalog entry, no
`experiment_metadata.json`, and no SLEAP output.

Session names are free-form and inconsistent — spaces, mixed case, embedded notes
(`20250827three fish with room light off`, `20250901 fish3hi exp2 0630pm adapt tank light off`,
`20250905 groupg 1f2m background w`). They encode real experimental variables — light on/off,
background colour, group letter, adaptation time, HDR settings, feeding — that the structured
`202512` naming convention does not capture.

**Decision needed:** in scope for the new repo or not? It is a third again as much data as the
main series. If in scope, the naming must be normalised into the catalog schema first — and
whatever those condition variables mean should be recorded while it is still remembered.

### Action
**Do nothing.** This data already lives on `/bucket` (= `Z:` = `\\bucket.oist.jp\bucket`), which is
OIST's backed-up long-term storage. Do not copy, do not move, do not stage a second copy.
The new repo points at it read-only via its own path config.

---

## Tier 1 — Human labor

Regenerable in principle, but only by spending human hours again. This is the most valuable
recoverable category.

### 1a. Experiment catalog & outcome data

| Asset | Location | Size |
|---|---|---|
| Catalog | `Z:\...\medaka_202512\catalog.{csv,jsonl,html}` | 124 KB / 234 KB / 62 KB |
| Experiment list (Google Sheet cache) | `Z:\...\medaka_202512\experiment_list\` — 9 files incl. `Experiment_List.xlsx` | 2.5 MB |
| Per-experiment metadata | `Z:\...\medaka_202512\<exp>\experiment_metadata.json` | ~4.7 KB × 71 |
| BORIS manual ethograms | `G:\.shortcut-targets-by-id\14OI1k7IaNOdC5yaC9GnYyCi2u-gfGLEs\OIST 1215-0122\boris_events\` | 26 CSVs, ~1 MB |

`catalog.jsonl` is the lossless form (one manifest per line); the CSV and HTML are renders of it.

> **Coverage caveat that must carry forward:** roughly **22–26 of 71** experiments have BORIS
> annotation, and the exact number is disputed across sources — the catalog reports 22 with
> `has_boris=True`, a 2026-06-30 cohort build parsed 23 CSVs, and 26 CSVs sit in the Drive folder.
> At least one is a join failure on a date mismatch (BORIS `20260112_I_2_3_FCMC` vs catalogued
> `20260113_I_2_3_FCMC`). **Recount in the new repo.**
>
> BORIS is the only independent human ground truth in the project, so every claim validated
> "against BORIS" is validated on at most a third of the dataset.

### 1b. Hand-made ground truth living inside the old repo

| Asset | Location | Git status |
|---|---|---|
| Fish sex ground truth | `annotations/fish_sex_gt/` | tracked |
| Contact episode ground truth | `annotations/contact_episode_gt/` | tracked |

~37 MB total on disk. **These are inside the repo and would be lost with it — extract them.**

### 1c. SLEAP training labels (annotated keypoints)

These embed thousands of hand-corrected frames. Losing them means re-annotating.

| Asset | Location (under `Z:\StephensU\medaka\sleap_models\`) | Size | Keep |
|---|---|---|---|
| Round-8 cumulative superset | `retrain_20260722_round8\combined_labels.pkg.slp` | **5.1 GB** | ✅ primary |
| Round-8 top train / val | `retrain_20260722_round8\top\{train,val}_labels.pkg.slp` | 2.1 GB / 31 MB | ✅ |
| Round-8 side train / val | `retrain_20260722_round8\side\{train,val}_labels.pkg.slp` | 3.1 GB / 46 MB | ✅ |
| Round-7 top labels (n=1295) | `retrain_20260704\top\labels.pkg.slp` | 2.0 GB | ✅ reproducibility |
| Round-7 side labels (n=2590) | `retrain_20260704\side\labels.pkg.slp` | 2.9 GB | ✅ reproducibility |
| Round-9 side HITL review labels | `retrain_20260728_round9_side\source_labels\side_ci_review_210_labeled_complete_full.pkg.slp` | 505 MB | ✅ one copy only (duplicated across 2 dirs) |
| Round-8 active-learning seed + edits | `Z:\...\runs\round8_active_learning\20251225_K_1_2_FCMC\pilot_120_{seed,edits_full}.pkg.slp` | 474 MB × 2 | ✅ |

Rounds 3/4/5 labels (`retrain_20260508`, `retrain_20260611`, ~8.3 GB) are **subsumed** by the
round-8 cumulative merge. Keep only if exact round-4/5 reproduction ever matters; otherwise drop.

**Subtotal Tier 1c keeper: ~16.6 GB.**

### 1d. Detectron2 training data (COCO)

Currently **only inside the old repo, untracked** — at risk.

| Asset | Location | Size |
|---|---|---|
| Top-view COCO | `outputs\20251225_K_1_2_FCMC\detectron2_coco\` (chunk1, chunk2, train_chunks1_2) | 406 MB |
| Side-view COCO (merged) | `outputs\20251225_K_1_2_FCMC\side_coco_merged_firstprompt\` | 149 MB |

**Must be moved to `Z:` or they die with the repo.**

---

## Tier 2 — Trained models

Regenerable from Tier 1 + GPU time, but that time is real. Keep production + one fallback.

### 2a. SLEAP — production is SPLIT ACROSS TWO ROUNDS

This is the single most important correction to carry into the new repo. The old repo's
documentation does not state it clearly anywhere.

| View | Production model | Source of truth |
|---|---|---|
| **top** | round 7 (`retrain_20260704`) | `retrain_20260704\EVAL_SUMMARY.md`; every shipped 3D run is tagged `round7_260704` |
| **side** | round 8 (`retrain_20260722_round8`) | `report/20260729/sleap_side_round9b_snn031/final_summary.json` → `"production_centroid":"Round8"`, `"production_centered_instance":"Round8"`, `"promote_retained_round9b": false` |

All models are top-down (centroid + centered_instance). `EVAL_SUMMARY.md` for round 7:
*"use TOP-DOWN (centroid + centered_instance) for production, both cameras."*

| Keep | Path (under `Z:\StephensU\medaka\sleap_models\`) | Size | Verified |
|---|---|---|---|
| round7 top centroid | `retrain_20260704\top\runs\round7_260704.centroid_n1295\best.ckpt` | 90 MB | ✅ exists |
| round7 top CI | `retrain_20260704\top\runs\round7_260704.centered_instance_n1295\best.ckpt` | 810 MB | ✅ exists |
| round7 side centroid | `retrain_20260704\side\runs\round7_260704.centroid_n2590\best.ckpt` | 90 MB | ✅ exists |
| round7 side CI | `retrain_20260704\side\runs\round7_260704.centered_instance_n2590\best.ckpt` | 810 MB | ✅ exists |
| round8 top centroid | `retrain_20260722_round8\top\runs\round8_260723_strictmouth.centroid_n1382\best.ckpt` | 90 MB | ✅ exists |
| round8 top CI | `retrain_20260722_round8\top\runs\round8_260723_strictmouth.centered_instance_n1382\best.ckpt` | 810 MB | ✅ exists |
| round8 side centroid | `retrain_20260722_round8\side\runs\round8_260723_strictmouth.centroid_n2764\best.ckpt` | 90 MB | ✅ exists |
| round8 side CI | `retrain_20260722_round8\side\runs\round8_260723_strictmouth.centered_instance_n2764\best.ckpt` | 810 MB | ✅ exists |

Round 7 **side** is kept purely so existing `round7_260704` reconstructions remain reproducible.
Round 8 **top** is a validated-but-never-shipped candidate — kept as the fallback/upgrade path.

**Copy the whole run directory, not just `best.ckpt`.** Each contains (verified):
`training_config.yaml` + `initial_config.yaml` (the HPC submitter validates the former),
`labels_gt.{train,val}.0.slp` — **the exact train/val split**, which makes the run genuinely
reproducible — `training_log.csv`, `training_console.log`, and for round 8 also
`metrics.{train,val}.0.npz`.

Exclude `labels_pr.*.slp` from each run dir: those are prediction dumps, regenerated at eval time.

**Subtotal: ~3.6 GB.**

#### Drop (~19 GB)
Rounds 3/4/5 checkpoints (`retrain_20260508` ~9 GB, `retrain_20260611` ~1.9 GB);
`retrain_20260724_sideopt`, `retrain_20260725_sideopt_phase4_ohkm_pair`,
`retrain_20260728_round9_side`, `retrain_20260728_round9b*`,
`retrain_20260729_round9b_ci_moks_snn031` — **all explicitly rejected** by the 2026-07-29
decision; `eval_round4_vs_round5`, `smoke_bottomup`; bottom-up baselines;
`retrain_20260704\trt_engines\` (A100-specific, cheap to re-export).

### 2b. Detectron2 segmentation

| Asset | Location | Size |
|---|---|---|
| Top model | `Z:\...\medaka_202512\detectron2_models\top_detectron2_model\model_final.pth` + `detectron2_config.yaml` | 351 MB |
| Side model | `Z:\...\medaka_202512\detectron2_models\side_detectron2_model_firstprompt\model_final.pth` + `detectron2_config.yaml` | 351 MB |

Already on `Z:`. The repo-local training runs under `outputs\20251225_K_1_2_FCMC\` have the
**same byte size** (351,018,161) — almost certainly the same files, but this was *not*
hash-verified. Checksum before deleting the local copies. Drop
`outputs\...\detectron2_model_smoke\` (smoke test) unconditionally.

### 2c. Appearance identity classifiers — **archive only, not wired in**

Per the decision to start geometry-only, these are preserved as assets for a later phase.

Location: `Z:\StephensU\medaka\runs\udmt_validation\segid\<exp>\<model>\model.pt` (45 MB each).

> ⚠️ **Structural finding — verify this before planning phase 2.** These classifiers are
> **trained per experiment, not globally**, and the naming is inconsistent across experiments:
>
> | Experiment | Models actually present |
> |---|---|
> | K_1_1 | `model_female_clean_v2`, `model_male_clean_v2` |
> | K_1_2 | `model_female_clean`, `model_male_clean`, `model_top_female`, `model_top_male` (+ 11 experimental) |
> | K_1_3 | `model_female_clean_v2`, `model_male_clean`, `model_male_clean_v2` |
>
> Only K_1_2 has the full four-model set. There is **no global appearance model** — which means
> the appearance approach as built does not scale to 71 experiments without per-experiment
> training. Treat this as a design constraint, not a detail.

Keep: the `*_clean*` and `model_top_*` sets above (~8 files, ~360 MB) plus their
`*_report.json` metrics and the pooled crop training datasets in the same directory
(`pooled_sides_corrected`, `pooled_all`, `male_sides_*`).

Drop the ~11 experimental variants under K_1_2 (`model_all`, `model_resnet_*`, `*_masked`,
`*_smoke`, `model_female`, `model_male_identity_classifier_masked`).

### 2d. Calibration models — tiny, git-tracked, keep wholesale

| Asset | Location | Size |
|---|---|---|
| **Production** (fixed-interface refractive) | `calibration_results\fixed_refractive_192x192x186\` — manifest + 3 camera bundles | ~5 KB |
| Legacy polynomial / pinhole-DLT | `calibration_results\cali_20251217_manual\` | ~86 MB total dir |
| Oldest per-way polynomial fits | `models\calibration\cali_251217_way{1,2,3,4}\` | 16 joblib, ~2 KB each (`way4` is empty) |

`calibration_results/` is 306 git-tracked files, kilobytes each. **Copy the whole directory** —
it is free and it is the measurement foundation. The user has chosen to inherit the calibration
*code* (`scripts/calibration/`, 42 files) along with these artifacts.

---

## Tier 3 — Derived output: leave behind

Everything here is reproducible from Tier 0–2. It is also where the misleading partial results
live, so carrying it forward would defeat the purpose of the reset.

| Asset | Location | Size | Verdict |
|---|---|---|---|
| SLEAP predictions | `Z:\...\sleap_output\<exp>\round7_260704\` | see note | **Decision needed** — see below |
| Detectron2 mask shards | `Z:\...\detectron2_output\{K_1_1,K_1_2,K_1_3}\` | **~82 GB** | **Drop** — regenerate on saion when needed |
| Local run outputs | `<repo>\runs\` | **45 GB** | Archive to `Z:`, then drop |
| Local pipeline outputs | `<repo>\outputs\` | **20 GB** (20 GB of it is K_1_2 alone) | Extract COCO (Tier 1d) first, then archive |
| Reports / figures / decks | `<repo>\report\` | **6.2 GB** | Archive to `Z:` — contains the DECISION.md files that justify Tier 2 choices |
| SAM2 pretrained weight | `<repo>\checkpoints\sam2_hiera_large.pt` | 898 MB | Drop — upstream download, not yours |
| Git history | `<repo>\.git` | 1.6 GB | Stays with the old repo |
| Virtualenvs | `.venv`, `.venv-detectron2`, `.venv-idtrackerai` | ~50k files | Drop |

### The one genuinely hard call: existing SLEAP predictions

`Z:\StephensU\medaka\sleap_output\` holds inference output for **73 experiments, 65 of them
tagged `round7_260704`** — that is most of the dataset already run through the cluster.

It is derived output, so by the governing rule it should be dropped. But it represents roughly
65 × 3 = 195 full-length view-inferences of GPU time, which is the scarce resource.

**Measured (K_1_2 as representative): ~777 MB per experiment → ~50 GB across 65.** But that
double-counts: each view keeps both the 8 per-chunk `.slp` (~15 MB each) *and* the merged
`<view>.slp` (~125 MB). The chunks are redundant once merged and are pure HPC scratch residue.

- Merged `.slp` only, all 3 views: ~375 MB/exp → **~24 GB**
- Merged `.slp`, **top camera only**: ~125 MB/exp → **~8 GB**

Eight gigabytes for a validated head start on 65 experiments is cheap.

The complication is the **split production rounds**:

| View | Model used for the existing round-7 output | Still production? |
|---|---|---|
| top | round 7 | ✅ yes — output is current |
| side ×2 | round 7 | ❌ **no** — superseded by round 8 |

So two thirds of that cluster time (the side views) produced output from a model that has since
been replaced. Re-running side inference with round 8 is needed regardless, if the new pipeline
is to use the current models.

**Recommendation:** keep the round-7 `top_cam` predictions as a free head start and a
regression baseline; plan to re-run side inference under round 8. Confirm the round-8 side model
actually improves the downstream 3D result on one experiment *before* committing ~130
view-inferences of cluster time to it — the round-8 decision was made on SLEAP metrics, not on
3D outcomes.

**Extract before archiving `report/`:** the DECISION and summary files are the evidence base for
the "which model is production" answers above —
`report/20260708/round7_bakeoff/DECISION.md`, `report/20260729/sleap_side_round9b_snn031/final_summary.json`,
`report/20260630/k12_appearance/bakeoff/DECISION.md`, `runs/calib_12balls/REPORT.md`.

---

## Tier 4 — Code and knowledge

**No analysis code is inherited.** ~175,000 lines of Python (`scripts/` 126.5k,
`tests/` 24.6k, `sam2_id_experiment/` 22.1k, `UDMT/` 2.8k) stay behind by explicit decision.

Exceptions, in order of certainty:

| What | Why it travels |
|---|---|
| **OIST HPC knowledge** → `OIST_HPC.md` | The explicit carve-out. Distilled to a standalone reference, not copied as scripts |
| **Calibration code** (`scripts/calibration/`) | User decision: measurement infrastructure, not the buggy analysis layer |
| **Prior findings** → `PRIOR_FINDINGS.md` | Carried as *hypotheses to re-verify* + dead ends not to repeat |
| **This manifest** | The map back to the assets |

### Known-stale things that must NOT be copied forward verbatim

Found during this audit — each is a live source of confusion in the old repo:

1. `configs/sleap/model_paths.yaml` points at round-1/2 models (`sleap_top_mouth/models/260313_*`)
   that **no longer exist** under `sleap_models_dir`. Production never used this file; the round
   directory is passed on the CLI instead.
2. `scripts/hpc/submit_experiment.sh:62-65` hardcodes the same dead `/bucket/...` model paths.
3. `README.md` references `scripts/sleap/run_inference_slurm.sh` — **the file does not exist.**
4. `MEMORY.md` was last updated 2026-06-06 and describes the polynomial calibration as
   production; `AGENTS.md`/`paths.local.toml` say the refractive backend is. `MEMORY.md` also
   points at a stale user path (`c--Users-Makoto-Hiroi-...`).
5. `AGENTS.md` documents the round-4 recipe as canonical while production moved to round 7/8.
6. Saion **partition names have been renamed** upstream: `largegpu` → `gpu-a100`, and `gpu` was
   split into `gpu-v100` / `gpu-p100`. Every script in `scripts/hpc/` uses the old names.

---

## Summary

| Tier | Keep | Size | Where it must end up |
|---|---|---|---|
| 0 Source data | video, frametimes, calibration recordings | ~1.07 TB | already correct on `/bucket` — leave alone |
| 1 Human labor | catalog, BORIS, GT annotations, SLEAP labels, D2 COCO | ~17.2 GB | consolidate on `Z:` (COCO + `annotations/` must be rescued from the repo) |
| 2 Models | SLEAP ×8, D2 ×2, appearance ×8, calibration | ~4.7 GB | consolidate on `Z:` under one models root |
| 3 Derived | — | 0 | archive `runs/`+`report/`+`outputs/` (~71 GB) to `Z:`, drop 82 GB of D2 masks |
| 4 Code | HPC doc, calibration code, findings doc | KB | seed the new repo |

**Net: ~22 GB of durable assets + ~1.07 TB of untouched source data.**
Against ~175k lines of code and ~153 GB of derived output that do not travel.
