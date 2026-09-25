# Research source checkpoint, 2026-09-25

The outstanding implementation is committed by research context: refractive
projection, aquarium visualization, generative tracking, independent masks,
three-body fitting and recovery, identity and review, long-run execution,
bidirectional reconnection, multiview node review, the successive 2D comparisons,
and the SLEAP adapter. This checkpoint preserves the experiments; it does not
promote them into the production tracking pipeline.

## Source and artifact boundary

Some existing tests import implementation directly from `work/three_body/`.
Those Python and PowerShell sources are explicitly tracked at their existing
paths so the repository does not contain tests without their implementation.
The other small, authored `work/` helper scripts are retained with their related
experiment. Source files are enumerated individually; source snapshots inside
run directories, downloaded packages, caches, videos, arrays and model files are
not included. The `work/` ignore rule remains in place. New implementation
belongs in `src/` or `labs/`, rather than extending this historical layout.

Frozen experiment sources are preserved without mass formatting or relocation.
Full numerical results and HTML report assets remain in the configured/local
artifact storage. Links into `work/` in experiment result documents therefore
require those existing artifacts and do not imply that Git hosts the videos.
Machine-local configuration, including `.claude/settings.local.json`, is ignored.

## Checks at the checkpoint

| Check | Observed result |
|---|---|
| Nine non-three-body labs, aquarium and forward projection, each in an independent pytest process | 424 passed, 9 skipped (CUDA-dependent checks) |
| Lab isolation and lifecycle checks | 16 passed |
| Shared calibration and SLEAP I/O round-trip checks | 32 passed |
| Three-body-related tests in one separate process, `--maxfail=3` | 145 passed, 3 failed; remaining tests not run |
| Full `tests/` invocation, `--maxfail=3` | Collection stops with 3 import errors caused by conflicting lab-local `observations` module names |
| `ruff check .` | 7 findings in `tests/test_bucketed_full_fit_graph.py` |
| `ruff format --check .` | 249 files would be reformatted; 352 already formatted |
| Syntax parsing of 951 lab and three-body helper Python sources | Passed |

The three numerical test failures are:

- `test_body_point_fitter.py::test_point_mass_above_one_is_not_clipped_and_q_one_keeps_clamp_derivative`:
  the gradient is 0 where the test expects 0.4.
- Both parameterizations of
  `test_bucketed_full_fit_graph.py::test_all15_objectives_and14_updates_unpadded_prefix_byteexact`:
  byte-exact floating-point comparisons differ.

These failures and the import/formatting issues remain open. A successful
checkpoint or push must not be reported as a successful full CI run. The task
does not change the numerical algorithms to resolve them. Detailed local logs
and the explicit commit file manifest are under `work/commit_push_20260925/`.
