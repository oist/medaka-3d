# Repository instructions for AI agents

These instructions apply throughout this repository. Read this file at the start
of a task, including after a fresh clone. `CLAUDE.md` adds project background;
it does not replace these shared instructions.

## Language

Use English for all repository-authored text: documentation, notebook Markdown,
code comments and docstrings, plot labels, output messages, configuration notes,
commit messages, and issue/PR descriptions. Preserve original source data and
verbatim evidence; explain them in English rather than translating the source.
Conversation with the user may follow the user's language.

## Start here

- For student onboarding, QC, and downstream analysis, read
  [analyses/README.md](analyses/README.md) first.
- For the current delivery objective, read
  [the dataset delivery plan](docs/decisions/2026-09-30-dataset-delivery.md).
- For extraction development, read [docs/NEXT_SESSION.md](docs/NEXT_SESSION.md)
  and the decision/result record for the exact method being used. Older dated
  sections are historical evidence, not a current execution instruction.
- [The source checkpoint](docs/decisions/2026-09-25-source-checkpoint.md)
  records existing validation failures. Check the current state before making
  claims; do not assign unrelated baseline repairs to a student analysis task.

Do not assume that a fresh clone includes videos, model files, derived tracks,
or local HTML reports. Do not describe experimental tracking as promoted or
identity labels as verified merely because a command completes.

## Where work belongs

| Location | Purpose |
| --- | --- |
| `analyses/YYYYMMDD_question/` | One scientific question or QC task, with a notebook, configuration, and concise findings |
| `analyses/_template/` | Copyable, runnable synthetic tutorial; never evidence about real fish |
| `src/medaka3d/` | Reviewed reusable loading, calculation, and pipeline code |
| `labs/` | Algorithm experiments governed by `labs/README.md` |
| `docs/decisions/` | Accepted methodological decisions and their evidence |
| Configured shared storage | Research inputs, large outputs, and backed-up human annotations |

Keep routine analysis outside `labs/`; its experiment lifecycle rules still
apply to algorithm experiments. Do not create parallel packages or copy pipeline
implementations into notebooks. Move reusable calculations into `medaka3d` when
needed, with checks appropriate to the calculation.

## Collaboration and analysis records

1. Use one short topic branch and PR per question or coherent change. The
   supervisor reviews changes before merging into `main`; work in progress can
   be saved in a draft PR. Do not commit, push, change access, or configure GitHub
   unless the user requests that action. Preserve unrelated working-tree edits.
2. Copy the analysis template for a new question. Keep its `README.md` current:
   question, owner, input run/version, method, findings, limitations, and next
   step. Record unsuccessful approaches and material parameter changes too.
3. Pin research inputs to an explicit extraction revision, run ID, checksum,
   provenance record, and documented schema. An upstream improvement creates a
   new input version; compare before switching. Never silently select a latest
   file or overwrite the input used for a result.
4. Keep settings in the analysis configuration and machine-specific roots in
   `configs/paths.local.toml`. Use the project virtualenv and the locked
   environment. Never hardcode a person's drive letters or install packages
   inside notebook cells.
5. Restart the kernel and run the notebook top-to-bottom before handoff. Keep
   small, readable figures and tables in the saved notebook; put large results
   in configured storage. Record code/config/input provenance alongside exports
   using `medaka3d.provenance.provenance(stage, out_dir)`.
6. In an Issue, a brief update of work done, findings, and blockers is enough.
   Use PRs for review and the analysis README for the durable result. A new
   notebook each day is unnecessary; Git records revisions.

## Scientific constraints

- Exclude each recording's first 300 seconds from analysis coverage. Verify
  timestamp origins, synchronization, native frame mappings, and sampling
  before converting frames to seconds; do not assume every file is 166 fps.
- Separate valid position, observed versus interpolated position, local segment
  ID, supported trial-wide identity, and physical F/M1/M2 naming. A numeric
  tracker slot does not establish the same individual across a trial.
- Keep unsupported identities unresolved. Preserve their positions for review,
  but exclude them from accepted identity-bearing trajectories. Keep missing,
  rejected, and unprocessed time in the declared coverage denominator.
- Break trajectory plots and derivatives at gaps, segment boundaries, and
  unresolved identity. Never turn missingness into zero or interpolate through
  it without a separately declared method and mask.
- Use `medaka3d.cohort` for experimental structure. Trial (group + trial), fish,
  recording, and frame are different units. Repeated days of the same fish and
  adjacent video frames are not independent biological replicates.
- Review whether encounters/courtship are disproportionately excluded. Low
  geometric residuals, smooth tracks, or a small error-free sample do not prove
  identity accuracy. Label exploratory findings and unexecuted proposals.
- Back up human-made labels and reviews to the curated shared asset storage
  immediately. Preserve raw inputs. Generated data belongs outside Git; small,
  explicitly synthetic tutorial fixtures are the exception.

## Completing a task

Validate the changed scope and report what actually ran. A template running on
synthetic data does not validate a real export, storage access, a tracking
method, or the whole test suite. Report unavailable checks and missing inputs
explicitly; never invent results to fill the handoff.
