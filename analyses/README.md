# Student analysis workspace

Start here after cloning `medaka-3d`. This is the shared home for trajectory QC
and downstream scientific analysis. Use English for all repository-authored
text, including notebooks, figures, comments, and PR descriptions.

AI assistants must read [the root AGENTS.md](../AGENTS.md) before making changes.
Claude Code should also read [CLAUDE.md](../CLAUDE.md). These files are tracked so
the same instructions travel with a student's clone.

## First session

1. Agree with the supervisor on one question, the recording(s), the starting
   code revision, the input extraction run, and what will count as completion.
   [The delivery plan](../docs/decisions/2026-09-30-dataset-delivery.md) requires
   consistent within-trial identities and excludes each recording's first 300 s.
2. Prepare the environment and run the copied tutorial below. It needs no video,
   GPU inference, or shared-storage access. It uses invented coordinates and
   demonstrates the workflow only; it is not a real trajectory result.
3. Obtain the supervisor-selected research export, schema, units/axes, body
   reference, time/frame mapping, quality/observation masks, identity evidence,
   and extraction provenance. Raw SLEAP rows and old identified CSVs are not
   automatically compatible with this template.
4. Verify local storage access before switching to research data. Copy
   `configs/paths.example.toml` to `configs/paths.local.toml` and edit the roots
   for this machine. That local file is ignored by Git. All required roots are
   checked by `medaka3d.load_paths`; missing storage is an error.

If no suitable export is available, record the blocker and work on input
inventory or source-video review with the supervisor. Do not silently substitute
an older run, synthetic data, or assumed identities for a research result.

## Setup and first run

From a clean clone, create a task branch and install the locked environment:

```bash
git clone https://github.com/oist/medaka-3d.git
cd medaka-3d
git switch -c analysis/trajectory-qc
uv sync --locked --python 3.13 --extra dev --extra analysis
```

Python 3.13 matches `.python-version`. This installs the project's normal
dependencies plus Jupyter, its execution tools, and Matplotlib. Do not upgrade
SLEAP packages or rebuild tracking models just to run the tutorial.

Copy the template once for the question. Replace the example date/name with the
start date and a descriptive question; the date is not a daily notebook counter.

Windows PowerShell:

```powershell
Copy-Item -Recurse analyses/_template analyses/20261026_trajectory_qc
.venv\Scripts\python.exe -m jupyterlab
```

Linux/macOS:

```bash
cp -R analyses/_template analyses/20261026_trajectory_qc
.venv/bin/python -m jupyterlab
```

Open the copied `analysis.ipynb`. In VS Code, select the project `.venv` as the
notebook kernel. In JupyterLab, check the printed `sys.executable` points to the
same environment. Restart the kernel and run all cells in order. The tutorial
prints its synthetic status and writes exports under the copied folder's
gitignored `_outputs/` directory.

For a noninteractive check from the repository root:

```bash
uv run --locked --extra analysis --extra dev jupyter nbconvert --execute --to notebook --inplace analyses/20261026_trajectory_qc/analysis.ipynb
```

Then open the saved notebook and inspect the figures and takeaways. Execution
alone does not establish that the figures are readable or the interpretation is
valid. Do not use the shared `_template/` directory for an ongoing research task.

## Layout and records

```text
analyses/
  README.md
  _template/
    README.md                  # Copyable question/method/findings record
    analysis.ipynb             # Executable QC example
    config.toml                # Explicit inputs and sampling assumptions
    synthetic_tracks.csv       # Tiny synthetic teaching fixture
  YYYYMMDD_question/
    README.md
    analysis.ipynb
    config.toml
```

Keep one folder per question or coherent QC task. Git stores successive attempts;
record failed approaches and meaningful changes in the folder README. Start a
new question folder when the question changes. A second focused notebook is fine
when it makes the calculation easier to understand; avoid copies named `final2`.

| Where | What belongs there |
| --- | --- |
| Analysis README | Question, owner, input/run references, methods, findings, limitations, next step |
| Notebook | Inspectable calculations, small figures/tables, and interpretation |
| Configuration | Input identity, time window, thresholds, and explicit assumptions |
| `src/medaka3d/` | Reviewed calculations that are reused across analyses |
| `labs/` | Tracking/algorithm hypotheses under its existing lifecycle rules |
| Shared storage | Research data, large exports, videos, and backed-up manual reviews |

The fixture's CSV contract is an example for one body-reference point per local
segment and recording frame. It is documented in [the template](./_template/README.md#input-table).
An adapter to a real export must explicitly map and validate its columns and
preserve source frames, timestamps, and masks. This change does not implement or
promote a dataset-wide export format.

## Switching to research inputs

Edit the copied README and configuration together. Set `input.kind = "research"`,
select `input.root = "work_dir"` or `"asset_root"`, and give a relative CSV path
under that configured root. Record the exact extraction commit, run ID, SHA-256,
schema, body reference, provenance, and identity-evidence references. Replace
the synthetic recording/window/sampling settings with documented values. Set
the owner and question. Remove the unused synthetic fixture from the research
folder when it is no longer needed.

The notebook checks the checksum and input contract. It never infers identity
support from a smooth trajectory, a non-null slot, or a low residual. Research
support flags must come from the documented upstream evidence and review policy.
It deliberately rejects irregular sampling: adapt its counting method before
using timestamp-weighted data. The tutorial's 1 Hz grid is not the acquisition
frame rate; do not carry it into a 166 fps recording.

Inputs stay fixed while the supervisor improves extraction. A replacement run
gets a new run ID/checksum and a recorded old/new comparison before adoption.
Never choose the newest file automatically or overwrite a prior input. Existing
`medaka3d.provenance` records are reused for the notebook's exports, together with
a frozen configuration and notebook-source hash. Real exports go under
`work_dir/analyses/<analysis slug>/<run ID>/`; curate/back up the final handoff
on shared storage. A local scratch directory is not a backup.

## Working with the supervisor

- Use a short task branch. Open a draft PR early if feedback would unblock you;
  merge after supervisor review. Do not directly push to `main`. Configure branch
  protection on GitHub separately if enforcement is needed.
- In the task Issue, leave a brief update: work done, findings, blockers/next
  step. Use the README for the durable outcome and the PR for change review.
- Assign one active editor per notebook to reduce merge conflicts. Review the
  scientific choices as well as the Python: units, denominator, masks, sampling,
  ID evidence, and whether the question matches the design.
- At the first session, reproduce one small example together. Review the first
  real-data QC result before expanding to another recording or adding a new
  behavioral measure. Keep daily check-ins short and inspect plots together.

The initial two-week milestone is one reproducible pilot whose usable range and
limitations can be explained, followed by one bounded exploratory question if
the inputs support it. It is not a promise to complete identity-method research
or dataset-wide behavioral inference in two weeks.

## Before requesting review

- Restart and run all cells; inspect the saved plots and tables. Include the
  exact command and explain any data, environment, or execution gap.
- Keep the input version, schema, code/config provenance, and output run ID
  traceable. Only small, purposeful notebook outputs belong in Git.
- Keep missing/rejected/unprocessed time in coverage denominators. Separate
  observed positions, interpolation, local continuity, and supported trial IDs.
- Show gaps and unresolved boundaries in plots. Do not calculate derivatives
  through them or treat missing values as zero.
- Use trial/fish structure from `medaka3d.cohort`; frames and repeated recording
  days are not independent animals. Label exploratory comparisons accordingly.
- Check whether encounters or courtship are selectively missing. Back up manual
  annotations/reviews to shared asset storage with recording/frame, reviewer,
  date, label, and undecidable cases.
- Update the README's findings, failed attempts, limitations, and next step.
  Passing this tutorial does not validate any tracking method.

For changes to reusable calculations, run the relevant focused tests. Historical
full-suite failures are documented in
[the source checkpoint](../docs/decisions/2026-09-25-source-checkpoint.md); inspect
the current baseline and distinguish existing failures from new regressions.
