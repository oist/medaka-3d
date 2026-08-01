# Handoff pack — medaka pipeline reset

Everything that survives the retirement of `medaka-mating-pipeline`, in four documents.
Compiled 2026-08-01 against the repo at commit `b506a69`.

| Document | What it answers |
|---|---|
| **`ASSET_MANIFEST.md`** | What data and models survive, where they are, what gets dropped and why |
| **`OIST_HPC.md`** | The only inherited *skill*: how to run work on saion and deigo |
| **`PRIOR_FINDINGS.md`** | Every prior scientific claim, marked as a hypothesis, with a recipe to re-verify it — plus dead ends not to repeat |
| **`MIGRATION_PLAN.md`** | The order of operations, and the structural habits that made the old repo decay |

## The reset in one paragraph

The old repo accumulated ~175,000 lines of Python across 605 scripts, ~153 GB of derived
output, and a documentation layer that had drifted out of sync with reality — configs pointing
at deleted models, a README linking a script that never existed, a handoff file naming a
calibration backend that had been replaced. The science underneath is probably sound but was no
longer separable from the artifacts of partial runs and late-discovered bugs. The reset keeps
the irreplaceable things (1.05 TB of recordings, hand-annotated training labels, trained
models, human ground truth) and the transferable skill (OIST HPC), and rebuilds the analysis
from raw video with provenance tracking built in from the first commit.

## What was already done

Everything lands under `Z:\StephensU\medaka\archive\medaka-mating-pipeline\`, preserving the
repo-relative layout so old absolute paths in logs stay mechanically translatable.

- ✅ **At-risk assets rescued** — 2,892 files, 591 MB, verified byte-exact (file count *and*
  total bytes). These existed in only one place, inside the repo's gitignored working
  directories, and would have been lost with it: `annotations/` (fish-sex and contact-episode
  ground truth) and both Detectron2 COCO training sets.
- ✅ **`report/` archived** — 5,868 files, 6.29 GB, 0 files and 0 bytes failed. (robocopy
  reported 7 failed *directories*; all are permission-denied `pytest_tmp` / `item56_verify_tmp_*`
  scratch dirs containing nothing of value.)
- ⏳ **`runs/` + `outputs/` archiving** (~65 GB) — running in the background.
- ✅ Old repo confirmed **clean and fully pushed** — nothing uncommitted, nothing unpushed.
  Safe to leave in place as a searchable reference.
- ✅ This handoff pack copied to `...\archive\medaka-mating-pipeline\handoff\` so it survives
  the session.

> **Why `runs/` matters more than it looks.** `runs/` is gitignored (`.gitignore:213`), so every
> calibration report, every `FINDINGS*.md` under it, and the entire dual-line evidence chain
> **exists only on this machine and has never been in git history**. `report/` is tracked; `runs/`
> is not. This was the single highest-risk gap found in the audit.

## What needs a decision before proceeding

1. **Name and location of the new repo.**
2. **`medaka_202508`** — a 535 GB / 28-session pilot series from Aug–Sep 2025 that the old repo
   never touched and never catalogued. In scope or not?
3. **Existing SLEAP predictions** for 65 experiments — keep the top-camera output as a head
   start, or start inference clean? (See `ASSET_MANIFEST.md` → Tier 3.)
4. Whether to run the bulk archive (~71 GB of `runs/`, `report/`, `outputs/`) now.

## Facts worth knowing before reading further

- Production SLEAP is **split across two rounds** — round 7 for the top camera, round 8 for the
  sides. This is stated nowhere in the old repo's documentation and is the kind of thing the
  reset exists to fix.
- The appearance-based identity classifiers are trained **per experiment**, not globally, and
  only K_1_2 has a complete set. As built, that approach does not scale to 71 experiments.
- Nearly all analysis to date rests on **3 experiments out of 71** (K_1_1, K_1_2, K_1_3), and
  only ~26 experiments have any BORIS human annotation. Generalization is untested.
- Saion **renamed its partitions**: `largegpu` → `gpu-a100`, `gpu` → `gpu-v100` / `gpu-p100`.
  Every script in the old repo uses the old names.
