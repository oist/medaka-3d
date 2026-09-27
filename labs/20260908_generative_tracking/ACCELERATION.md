# Independent verification of the acceleration (2026-09-08)

Image preprocessing on the RTX 5070 Ti, and a reorganisation of the projection and Jacobian
computation that keeps the same objective function, were measured separately. What follows is
a component-level verification on short intervals, not a demonstration of overall throughput
over one minute or of ID accuracy. For long-interval speed, use each run's
`comparison.json: tracking_fps`.

## GPU image preprocessing

Real CUDA kernels were confirmed to run with PyTorch `2.13.0+cu130` / CUDA 13.0 in the
existing `.venv`. The GPU is an RTX 5070 Ti 16 GB. OpenCV 5.0.0 reported 0 CUDA devices, so
`accelerated_observations.py` does RGB differencing, illumination-difference correction via
integer histograms, and the channel maximum on the GPU, and does connected components, the
distance transform and Farneback flow as CPU processing parallel across the 3 cameras.
There is no implicit CPU fallback when the GPU is unavailable. No additional drivers or
trained models were needed.

Preprocessing measured on real images `76620:76628`, after warm-up:

| Processing | Flow max side | Median s/frame | Preprocessing fps |
|---|---:|---:|---:|
| CPU sequential | 640 | 0.941 | 1.06 |
| CPU, 3 cameras in parallel | 640 | 0.354 | 2.82 |
| GPU differencing + CPU parallel | 640 | 0.313 | 3.21 |
| CPU, 3 cameras in parallel | 320 | 0.186 | 5.40 |
| GPU differencing + CPU parallel | 320 | 0.131 | 7.57 |

Image loading and background preparation are excluded; GPU transfer and synchronisation are
included. The measuring process ran under cProfile. fps is computed from the total time over
8 frames; the times in the table are medians. The main improvement is CPU parallelisation
across cameras and is not to be confused with a GPU-only effect. The median of the CUDA
section was about 0.026 s.

At the same 640 setting, all image maps from CPU sequential, CPU parallel and GPU matched
exactly on the 8 real frames, and the tracking output of sequential CPU and GPU also matched
exactly. At the 320 setting the CPU/GPU maps also match exactly.
However, **640→320 is a change of observation setting**, not an equivalent implementation
optimisation. The flow-displacement difference at 213 real SLEAP points, in source image
coordinates, has median 0.494 px, 95th percentile 1.478 px and maximum 2.143 px. The selected
trajectory, source and review did not change on these 8 frames, but invariance of tracking
results in general is not guaranteed. No temporal subsampling of the original 166 fps is done.

Run, numerical QA and provenance:
[`profile_image_acceleration.py`](profile_image_acceleration.py),
`work/generative_tracking/acceleration_76620-76628/acceleration.json`.
The 8 tests in `tests/test_accelerated_observations.py` that actually use the GPU also passed.

## Thread investigation and projection computation

OpenBLAS was found at 24 threads each for NumPy/SciPy, torch CPU at 16 and OpenCV at 32.
Only `threadpoolctl==3.6.0` was added to the local venv, without dependency changes, and the
comparison was made with a limited scope. The 8-frame fit measurement using fixed CUDA320
images took 4.079 s by default, about 4.004 s with BLAS at either 1 or 2 threads, and 4.082 s
with BLAS1 + torch1 + OpenCV1. All output differences were 0, but the improvement was about
2%; threads were not the main cause. These numbers also carry cProfile overhead. Of the
8.157 s total over 16 frames, cumulative refractive projection time was 5.233 s, `_fit` was
5.600 s and the image score was 2.339 s.
Record: `work/generative_tracking/fit_threads_76620-76628/`.

`fast_fit.py` exploits the structure in which each residual row depends only on one fish's
9 coordinates, compressing finite differences from 28→10 batches. The solver that solves all
27 coordinates jointly, the objective function and the stopping criteria are kept. Node and
body-axis samples are also merged into a single projection per camera. In the image score,
the three groups current pose, body axis and previous pose were likewise merged into one.

Median of 3 alternating repetitions on the saved real images `76620:76623`, after the other
tracking processes had finished, without cProfile, using the same precomputed CUDA320 images:

| Computation inside ReacquisitionTracker | 3 frames, s | fit-only fps |
|---|---:|---:|
| Original fitter + original image score | 0.877 | 3.42 |
| Original fitter + image score with batched projection | 0.827 | 3.63 |
| FastFit + image score with batched projection | 0.539 | 5.57 |

The last condition is about 38.5% shorter. Across the whole tracking the maximum coordinate
difference is `1.67e-11 mm`, and the source, review, appearance and quality differences are
0. The image score alone, 60 calls, went from 0.567→0.442 s, about 22% shorter, and the
projection, score, template, feature and quality values all differed by 0.
In an independent GenerativeTracker comparison, FastFit also went from 0.868→0.590 s, about
32% shorter. Jacobian difference 0, maximum cost difference of individual fits `3.20e-14`,
source, observed and review differences 0.
These are numerical agreement on a finite set of verification cases; bit-level agreement over
the whole video is not claimed. The 7 tests in `tests/test_fast_fit.py`, which cover missing
observations, near-boundary cases, images and invalid projection points, passed.

Records of the correct comparisons are below. Each directory holds a code snapshot, input
hashes and provenance.

- `work/generative_tracking/batched_projection_76620-76623_standalone_explicit_fit/`
- `work/generative_tracking/fast_fit_76620-76623_standalone_batched/`
- Images for rerun: `work/generative_tracking/fast_fit_76620-76623_contended/raw_frames.npz`

The initial measurements carry two limitations and corrections.
`fast_fit_76620-76623_contended` was measured while contending with 2 other processes and with
Jacobian compression only; its reduction of about 10% was treated as a reference value only.
In `batched_projection_76620-76623_standalone` the parent class at run time already used
FastFit, so the tracking row named `frozen` could not be used as a comparison with the
original fitter. `interpretation_correction.json` was saved, and that comparison was replaced
by the comparisons above, with the fitter made explicit. Those records were not deleted or
overwritten.
For GPU synchronisation semantics see the
[official PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html);
for thread control, [threadpoolctl (official)](https://github.com/joblib/threadpoolctl).
