# medaka-3d

Three-camera 3D behaviour tracking for medaka (*Oryzias latipes*) mating experiments.

**Research question.** How do female mate choice, male courtship, and male–male aggression
interact — and which behavioural factors actually determine the female's mating decision?
One female and two males per trial, filmed from three synchronised cameras at 166 fps.

**Status.** Foundation only. No pipeline code yet.

---

## This is a rebuild

It replaces `medaka-mating-pipeline`, which was retired in August 2026 after accumulating
~175,000 lines across 605 scripts, ~153 GB of derived output, and documentation that had
drifted out of sync with what the code actually did. The science underneath is probably
sound, but it was no longer separable from the artifacts of partial runs and
late-discovered bugs.

Almost nothing was inherited. What was:

| Document                                        | Contents                                                             |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| [docs/HANDOFF.md](docs/HANDOFF.md)               | Index of the reset — what survived and what was done                |
| [docs/ASSET_MANIFEST.md](docs/ASSET_MANIFEST.md) | Every inherited data and model asset, with paths and sizes           |
| [docs/OIST_HPC.md](docs/OIST_HPC.md)             | The one inherited*skill*: running work on saion and deigo          |
| [docs/PRIOR_FINDINGS.md](docs/PRIOR_FINDINGS.md) | Prior results —**as hypotheses to re-verify**, plus dead ends |
| [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md) | Order of operations, and the habits that made the old repo decay     |

No analysis code was carried over.

### Read this before trusting any prior number

- Nearly all previous analysis rests on **3 experiments out of 71**, and it is not
  established that they are three independent females.
- Only about **22–26 experiments** have any human (BORIS) annotation.
- Ground truth labels **only the female**. There has never been a male-vs-male ground truth
  set, so every M1/M2 claim is currently unvalidatable.

Details and per-claim re-verification recipes are in [docs/PRIOR_FINDINGS.md](docs/PRIOR_FINDINGS.md).

---

## Design rules

Each one exists because its absence caused a specific failure in the old repo.

1. **One importable package.** No loose scripts, and no numbered ones — a number in a
   filename is a decision someone refused to make.
2. **Config fails loudly.** Every configured path is existence-checked at load. A model that
   has been deleted is a hard error, never a silent fallback.
3. **Assets are checksummed.** Inherited models and labels live under `asset_root` with a
   `MANIFEST.json` recording size, sha256 and provenance, verified before use.
4. **Every output carries provenance.** Code revision, inputs, resolved model paths, full
   argv — written to a sidecar next to the artifact, on failure as well as success.
5. **No flag may change which algorithm runs.** A different algorithm gets a different
   command. (In the old repo, omitting `--num-candidates` silently selected a different,
   60× slower, unvalidated code path.)
6. **One way to do each thing.** Retire by deleting, not by prefixing the old thing with
   "legacy".
7. **Human-made data goes to `/bucket` immediately** — never only into a working directory.

---

## Setup

```bash
uv sync --python 3.13          # or: pip install -e ".[dev]"
cp configs/paths.example.toml configs/paths.local.toml   # then edit for your machine
pytest -q
```

```python
from medaka3d import load_paths, load_catalog

paths = load_paths()
video = paths.video("20251225_K_1_2_FCMC", "top_cam")
models = load_catalog(paths.asset_root)
centroid = models.get("sleap.round8.top.centroid")
```

## Hardware

Three cameras — cam0 top (XY), cam1 side (XZ), cam2 side (YZ). Tank 20×20×20 cm,
166 fps, three fish (1 female + 2 males).
