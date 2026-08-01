# Migration plan — old repo → fresh repo

Companion to `ASSET_MANIFEST.md` (what travels), `OIST_HPC.md` (cluster knowledge),
`PRIOR_FINDINGS.md` (claims to re-verify).

---

## Decisions already made

| Question | Decision |
|---|---|
| Where does the new repo start? | **From raw video** — full rebuild: SLEAP inference → 3D → identity → behavior |
| Prior scientific findings | Carried as **hypotheses to re-verify**, never as settled facts |
| Calibration | **Artifacts + code both inherited** — measurement infrastructure, not the buggy layer |
| Appearance/segmentation identity | **Deferred.** Geometry-only first; classifier model files archived for phase 2 |
| Model generations | Production + one fallback (round 7 & 8); ~19 GB of rejected checkpoints dropped |
| Detectron2 mask shards (82 GB) | **Dropped** — regenerate on the cluster when needed |
| Old repo | Local derived data (~71 GB) archived to `Z:`; repo itself not deleted |
| HPC | saion + deigo, sourced from `C:\Users\machi\Desktop\OIST_HPC\` |

---

## Why the old repo became unusable — and the structural fixes

Worth stating explicitly, because a fresh repo with the same habits decays the same way.

| Failure mode in the old repo | Evidence | Fix to build in from day one |
|---|---|---|
| **605 loose scripts, no package** | `sam2_id_experiment/scripts/00_…52_`; three parallel `scripts/experiments/*` benchmark trees | One importable package. No numbered scripts — a number is a decision you refused to make |
| **Config drifted from reality silently** | `configs/sleap/model_paths.yaml` and `submit_experiment.sh:62-65` point at models deleted months ago; nothing failed loudly | Config paths validated at load; a missing model is a hard error, not a fallback |
| **Two of everything** | legacy clip-CSV vs `.slp` workflow; polynomial vs refractive calibration; `scripts/` vs `src/medaka_pipeline/` scaffold | One path. Retire by deleting, not by adding "legacy" to the name |
| **A flag silently changed the algorithm** | `--num-candidates 6` selects a different (60× faster, validated) code path; omitting it runs an unvalidated one | No flag may change which algorithm runs. Different algorithm = different command |
| **Results couldn't be traced to inputs** | which round, which calibration backend, which bind source — recoverable only by reading `AGENTS.md` prose | Every output writes a provenance record: code rev, model paths + hashes, calibration ID, full argv |
| **Docs asserted a state that had passed** | `MEMORY.md` (2026-06-06) still names the polynomial calibration production; `README.md` links a script that doesn't exist | Docs describe *how*; machine-readable manifests describe *what is current*. Don't duplicate state into prose |
| **Human-labor assets lived in gitignored dirs** | D2 COCO training data (555 MB) and `annotations/` only ever existed inside the repo | Anything a human made goes to `/bucket` immediately, never only in a working dir |

---

## Sequence

### Phase 0 — Rescue (do first; small, at-risk, irreversible if lost)

These exist **only** inside the old repo working directory. Everything else is already on `Z:`.

| Item | From | Size |
|---|---|---|
| Fish-sex ground truth | `annotations/fish_sex_gt/` | small |
| Contact-episode ground truth | `annotations/contact_episode_gt/` | ~37 MB total |
| Detectron2 top COCO | `outputs/20251225_K_1_2_FCMC/detectron2_coco/` | 406 MB |
| Detectron2 side COCO | `outputs/20251225_K_1_2_FCMC/side_coco_merged_firstprompt/` | 149 MB |
| Decision records | `report/20260708/round7_bakeoff/DECISION.md`, `report/20260729/sleap_side_round9b_snn031/final_summary.json`, `report/20260630/k12_appearance/bakeoff/DECISION.md`, `runs/calib_12balls/REPORT.md` | KB |

Verify by checksum after copying. `Z:` has 46 PB free — space is not a constraint.

### Phase 1 — Consolidate the asset root on `Z:`

Give the new repo **one** root to point at, so `paths.local.toml` has one entry instead of
seven scattered ones. Physical copies for what's scattered; leave the 1.05 TB of video in place
and reference it.

```
Z:\StephensU\medaka\assets\
  models\
    sleap\round7_260704\{top,side}\{centroid,centered_instance}\   best.ckpt + training_config.yaml
    sleap\round8_260723\{top,side}\{centroid,centered_instance}\
    detectron2\{top,side}\                                          model_final.pth + config
    appearance\<exp>\                                               phase-2 archive, not wired in
    calibration\                                                    from calibration_results/
  training\
    sleap\round8_combined_labels.pkg.slp                            (5.1 GB, the superset)
    sleap\round7\{top,side}_labels.pkg.slp
    sleap\round9_side_ci_review_210.pkg.slp
    detectron2\{top,side}_coco\
  groundtruth\
    fish_sex\  contact_episodes\  boris\                            (BORIS copied off G: — Drive is not durable storage)
  MANIFEST.json                                                     paths + sizes + sha256 + provenance
```

Write `MANIFEST.json` with a checksum per asset. It is the contract the new repo validates
against at startup — the direct fix for failure mode #2 above.

### Phase 2 — Archive the old repo's derived output

`runs/` 45 GB + `report/` 6.2 GB + `outputs/` 20 GB → `Z:\StephensU\medaka\archive\medaka-mating-pipeline\`
preserving relative paths (same prefix-substitution convention used for the earlier
`C:` → `Z:` consolidation, so old absolute paths in logs remain mechanically translatable).

Then reclaim: `.venv*` (~50k files), `checkpoints/sam2_hiera_large.pt` (898 MB, upstream
re-downloadable), and the 82 GB of `Z:\...\detectron2_output\` mask shards.

Leave `.git` (1.6 GB) and the tracked source alone — the old repo stays readable.

### Phase 3 — Seed the new repo

Starting content, before any pipeline code:

```
docs/OIST_HPC.md          inherited cluster knowledge (saion + deigo)
docs/PRIOR_FINDINGS.md    every prior claim, marked hypothesis, with a re-verification recipe
docs/ASSET_MANIFEST.md    the map back to inherited assets
configs/paths.example.toml
```

Then calibration (the one inherited code module), then stage by stage.

### Phase 4 — First milestone: reproduce one experiment end to end

Target **K_1_2** (20251225_K_1_2_FCMC) — the most-studied experiment, so prior numbers exist
to compare against.

Success = raw video → 3D trajectories with geometry identity, with a provenance record, and a
documented comparison against the archived prior result. Where the new number disagrees with
the old, that disagreement is the finding — the old pipeline is not the reference.

---

## Sequencing risks

- **Phase 0 before anything else.** The COCO datasets and GT annotations are the only assets
  with a single copy. Everything else is on backed-up `/bucket`.
- **BORIS lives on Google Drive**, not `/bucket`. Drive-for-Desktop is a sync client, not
  archival storage. Copy the 26 CSVs into the asset root.
- **Don't delete the 82 GB of D2 masks until the new pipeline can regenerate them** —
  demonstrated on one experiment, not assumed.
- **Saion partition names changed** (`largegpu` → `gpu-a100`, `gpu` → `gpu-v100`/`gpu-p100`).
  Verify with `sacctmgr show assoc user=$USER` before the first cluster submission; do not
  copy the old scripts' names forward.
- **Appearance identity does not scale as built** — the classifiers are per-experiment, and
  only K_1_2 has a complete set. Re-examine that design before phase 2, don't inherit it.
