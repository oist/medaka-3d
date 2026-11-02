# Running the test suite

Use the project's locked virtualenv. Synchronize it before validation;
`--no-sync` alone does not check whether installed versions match the lockfile.
From the repository root:

```bash
uv sync --locked --extra dev --extra analysis
uv run --no-sync pytest tests -q
uv run --no-sync ruff check .
```

`conftest.py` chooses a fresh, short directory under the ignored `work/` folder
for pytest temporary files. Several audit tests intentionally reject paths
outside the repository; using the OS temporary directory would violate their
contract. The fresh path also avoids replacing existing work and keeps long
calibration fixture names within ordinary Windows path limits. An explicit
`--basetemp` is respected; choose a fresh path inside this repository when
running the audit tests.

The frozen generative-tracking lab uses flat imports such as `observations`
and `tracker`. Its test modules declare the lab path in their source.
`LabImportScope` isolates their aliases and search paths during both collection
and the complete setup/call/teardown protocol. It retains the same module graph
between that lab's tests, so shared classes and monkeypatch targets stay stable.
Do not fix collisions by overwriting another lab's `sys.modules` entries or by
reordering test files.

Tests of saved-output qualification run their orchestration in clean subprocesses
when absence of an imported model runtime is part of the real contract. The
preloaded-runtime rejection guard is tested separately; it is not disabled.

Most tests use synthetic inputs and require no video, model, or archived run.
Original-run audits declare their entry points with
`@pytest.mark.requires_artifacts(...)`. They run automatically when those paths
exist, and skip with the missing path in the reason otherwise. Present but
corrupted evidence still fails. To run only these audits after restoring their
original inputs, use `uv run --no-sync pytest -m requires_artifacts -rs`.
Inspect the skip report; a skipped archive audit is not validation of research
artifacts. Do not use the marker to hide a numerical failure or a missing source
file. Metadata mutation and decode-failure tests use self-contained synthetic
inputs so those regressions are checked in normal CI too.

Numerical tests distinguish differentiable values from a clamp's boundary
subgradient. They check analytic values/derivatives away from the boundary and
the installed PyTorch scalar convention at equality. Bucket-adapter execution
is byte-exact against an independent reference at the same padded shape; a
separate tightly bounded FP64 comparison checks the unpadded reference. These
synthetic checks do not qualify CUDA or real-video identity accuracy.

The lab lifecycle checks are intentional: an expired lab without `RESULT.md`
needs an evidence-backed verdict. Do not move its expiry date, invent a result,
or skip that check merely to make CI green. Historical source-formatting debt
is separate from behavioral test failures.

## Validation snapshot: 2026-11-02

Reviewed the 24-file test-maintenance change on top of `ba6211e`. No tracking
algorithm or scientific result was changed. The existing source-checkpoint
failures are recorded in
[the checkpoint](../docs/decisions/2026-09-25-source-checkpoint.md).

| Check | Result |
| --- | --- |
| Full `pytest tests -o addopts= -q -ra --tb=short` | 3,728 passed, 21 skipped, 2 failed |
| All 330 cases in the 21 changed test modules, after dependency synchronization | 330 passed |
| `ruff check .` with locked Ruff 0.16.7 | Passed |
| Formatting of the 22 changed Python files | Passed |
| Repository-wide `ruff format --check .` | 246 existing files still require formatting |

The two failures are unchanged lifecycle checks: `20260908_mask_volume` expired
on 2026-10-08 and `20260909_three_body` on 2026-10-09, both without `RESULT.md`.
They require research verdicts; this maintenance change does not invent those
verdicts, extend deadlines, or suppress the checks. The skips were 19 CUDA checks
and two mounted-asset checks, not validation of those capabilities.

The full run used the project Python 3.13.11 / pytest 9.1.1 environment. Its
installed runtime versions remained unchanged when `uv sync --locked --extra
dev --extra analysis --inexact` subsequently aligned Ruff and the editable
package registration; `--inexact` preserved unrelated installed extras. The
corresponding `--check` then reported no changes, and the affected tests were
repeated. The full suite was not repeated after this tooling-only alignment.

Full-run warnings included two existing all-NaN median warnings in slot-stitch
fixtures and two local pytest-cache permission warnings. The focused rerun
disabled the cache provider. Local logs, JUnit reports, and environment snapshots
are under the ignored `work/20261102_test_portability/` directory; fresh clones
do not include these generated files. Neither this review nor the passing
synthetic tests establishes real-video tracking accuracy or a fully green CI.
