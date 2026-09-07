# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Three-camera 3D behaviour tracking for medaka mating experiments. One female + two males per
trial, three synchronised cameras at 166 fps, 20×20×20 cm tank. The research question is which
behavioural factors determine female mate choice — courtship, male–male aggression, or their
interaction.

**This is a deliberate rebuild.** It replaces `medaka-mating-pipeline`, retired 2026-08-01.
Read [docs/HANDOFF.md](docs/HANDOFF.md) first — it explains what was inherited and why almost
nothing was.

## Read before making claims about prior results

[docs/PRIOR_FINDINGS.md](docs/PRIOR_FINDINGS.md) records every prior scientific claim **as a
hypothesis with a re-verification recipe**, never as established fact. Three constraints bound
everything that came before:

- Nearly all analysis rests on **one trio of fish**. `K_1_1`, `K_1_2`, `K_1_3` are the *same
  three fish on three consecutive days* — repeated measures, not three replicates. The prior
  "N=3 mate choice" result is n=1. The dataset holds **24 trials across 14 groups**, 22 with
  all three days; 23 of the 24 were never analysed.
- Only ~22–26 experiments have any human (BORIS) annotation.
- Ground truth labels **only the female**. There has never been a male-vs-male ground truth set,
  so every M1/M2 claim is currently unvalidated.

Use `medaka3d.cohort` for anything involving experiment structure. The independent statistical
unit is `Trial` (group + trial), **never** the folder. The experimenter's physical descriptors
for each fish (group K: female *yellow*, males *white* and *red*) live in `group_list.csv` and
are loaded by `cohort.load_groups` — M1/M2 denote specific animals, not arbitrary slots.

Do not repeat a prior number as settled. Cite it as a hypothesis and say what would confirm it.

## Design rules — each one has a scar behind it

1. **One importable package** (`src/medaka3d/`). No loose scripts, no numbered scripts.
2. **Config fails loudly.** `paths.py` existence-checks every root at load. Never add a silent
   fallback — the old repo's config pointed at models deleted months earlier and nothing complained.
3. **Assets are checksummed.** Everything inherited is declared in `MANIFEST.json` under
   `asset_root`, verified before use. Regenerate with `tools/build_asset_manifest.py`.
4. **Every output carries provenance.** Wrap stages in `medaka3d.provenance.provenance(...)`.
5. **No flag may change which algorithm runs.** A different algorithm gets a different command.
6. **One way to do each thing.** Retire by deleting, not by renaming to "legacy".
7. **Human-made data goes to `/bucket` immediately.** In the old repo, hand-made ground truth and
   two COCO training sets existed only in a gitignored folder on one laptop.
8. **Labels are not models.** Never delete a model directory without enumerating its label files
   first — 7.44 GB of human annotation was nearly lost that way during the migration.

## Facts the old repo got wrong

- ~~**Production SLEAP is split across rounds**: round 7 for **top**, round 8 for **side**.~~
  **RETRACTED 2026-08-03. Production is round 8 for BOTH views.** The "split" was a
  hardcoded constant in `tools/build_asset_manifest.py` that `MANIFEST.json` echoed
  back, so an assertion read as a measurement. Round-8 top and side both trained
  2026-07-23; the retained corpus predictions are round 7 only because they were
  produced 07-07 to 07-13, before round 8 existed — their `.slp` provenance says so.
  Consequences, because they are large: every B0.3/B0.4/k01 number describes round 7,
  **93 of the 113 "held-out" frames are inside round-8's training data**, and only
  **20** frames are held out from both checkpoints. On those 20, round 8 emits
  **0** false positives and **0** shared tails against round 7's 7 and 3.
  See [docs/decisions/2026-08-03-k-detection-quality.md](docs/decisions/2026-08-03-k-detection-quality.md).
- **Saion renamed its partitions**: `largegpu` → `gpu-a100`, `gpu` → `gpu-v100` / `gpu-p100`.
  See [docs/OIST_HPC.md](docs/OIST_HPC.md), which flags provenance on every fact.
- `medaka_202512` holds **71** experiments, not 76.
- Appearance identity classifiers were trained **per experiment**, not globally. That approach
  does not scale to 71 experiments as built. Deferred to phase 2; models archived.

## Environment

```powershell
uv sync --python 3.11              # or: uv pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
```

Always use the project virtualenv: `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python`.

`configs/paths.local.toml` is gitignored and machine-specific. Copy from `paths.example.toml`.
Never hardcode `Z:` or `G:` — they differ per machine.

## Storage

`Z:` = `\\bucket.oist.jp\bucket` = `/bucket` on the clusters. Same bytes, three names.

| Path | Contents |
|---|---|
| `Z:\StephensU\medaka\medaka_202512` | 71 experiments, 1.05 TB of source recordings. Read-only in practice |
| `Z:\StephensU\medaka\assets` | Inherited models, training labels, ground truth + `MANIFEST.json` |
| `Z:\StephensU\medaka\archive\medaka-mating-pipeline` | Frozen output of the retired repo. **Read `report/` before concluding something has never been measured** — it holds a full round-7 vs round-8 evaluator and the 2026-07-27 report naming round 8 production |
| `Z:\StephensU\medaka\benchmarks\sleap_hitl_v2` | **The locked HITL-v2 benchmark.** 40 source frames × 3 views, 360 human instances, all three K days, frame-disjoint from every training split. **Not in `MANIFEST.json` — register it.** 30% of its instances are verbatim round-8 output, so bracket it |
| `Z:\StephensU\medaka\medaka_202508` | 535 GB pilot series, **out of scope**, untouched |

`/bucket` is read-only from cluster compute nodes. This drives most of the HPC architecture.

## Status and what to do next

**Start here: [docs/NEXT_SESSION.md](docs/NEXT_SESSION.md)**, then
[docs/TRACKING_PIPELINE.md](docs/TRACKING_PIPELINE.md) for the chain as it stands.
[docs/ROADMAP.md](docs/ROADMAP.md) holds the phase structure; its Phase C text predates
the tracking work and the status block at its top says so.

As of 2026-09-08: one experiment (K_1_2) runs end to end from raw video through a
tracking chain (`tools/c27 → c28 → c33 → c30 → c34`) whose slot labelling keeps the
female's identity in 296 of 298 annotated intervals. **Nothing is promoted** — the numbered
tools under `tools/` are the experiments (a standing tension with design rule 1; promotion
into `src/medaka3d/` is owed), `c03` still owns the fixed output paths, and every number is
`n_groups = 1`. Each stage has a decision record under `docs/decisions/` holding its
pre-registration, its runs and the variants that were rejected. Where a new number disagrees
with the archived prior result, the disagreement is the finding — the old pipeline is not the
reference.

## Work rules

- Investigation and reading: proceed freely.
- Creating or overwriting files, installing packages: confirm first, unless the user's request
  already asked for that specific change.
- Deleting data: enumerate what would be lost, verify the keeper copy, then delete. Never the
  other order.
