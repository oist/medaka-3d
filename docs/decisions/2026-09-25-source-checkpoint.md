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

## Addendum 2026-11-08: full suite on a CUDA build of torch

Since 2026-11-07 the Windows virtual environment installs torch from the
PyTorch cu130 index (`pyproject.toml`, `[tool.uv.sources]`), so the machine's
RTX 5070 Ti is visible to the tests. A full `pytest` run on 2026-11-08 gave
8 failed, 3808 passed. Two are the lab-hygiene failures already open
(`test_labs.py::TestLabHygiene::test_lab_is_well_formed` for
`20260908_mask_volume` and `20260909_three_body`, expired without a
RESULT.md). The other six assert that `torch.cuda.is_initialized()` is false
or raise their own "requires no CUDA" / "must remain CPU" guards:

- `test_evaluate_body_point_identity200.py::test_failed_input_preserves_receipt_and_does_not_solve`
- `test_fast_tail_five_minute_pixels.py::test_real_small_metadata_binds90_masks_without_mask_pose_or_cuda`
- `test_fast_tail_two_minute_pixels.py::test_metadata_only_real_seals_never_open_arrays_or_cuda`
- `test_fixed_body_width_graph_fit.py::test_cpu_constructor_rejected_without_cuda_use_and_plain_renderer_cannot_be_captured`
- `test_paired_rescore_mixed4.py::test_actual_small_summary_plan_manifest_contract_without_numeric_payload`
- `test_prepare_body_point200_packets.py::test_input_failure_persists_failed_seal_without_attempting_packet_work`

All six pass when their files run in their own process (78 passed). In the
full run, CUDA is first initialised by
`test_accelerated_observations.py::test_cuda_foreground_matches_v3_all_maps_and_flow[random]`,
a test that runs on the GPU whenever one is available, and
`torch.cuda.is_initialized()` stays true for the rest of the process. On the
CPU-only build it could never become true, which is why these guards held
before. Note also that on this torch (2.14.1) any `torch.optim` step, even on
CPU parameters, initialises CUDA through
`Optimizer._accelerator_graph_capture_health_check`, so the guards cannot be
made order-independent by keeping other tests on the CPU. The same run with
`tests/test_contrastive_appearance_lab.py` excluded fails identically, so the
appearance lab is not the cause; its training tests are nevertheless pinned
to `device="cpu"`.

Open. Options, none taken here: have the six tests compare
`is_initialized()` before and after the stage under test rather than against
false; run them in a subprocess; or run the suite on a GPU machine with
`CUDA_VISIBLE_DEVICES` hiding the device. The choice belongs to the owners of
those labs. The numerical failures listed above remain as recorded; the
2026-09-25 import and formatting findings were resolved by later commits.
