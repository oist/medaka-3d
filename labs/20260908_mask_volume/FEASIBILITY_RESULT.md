# First result of fusing independent masks through refractive projection

2026-09-08. **The implementation that computes a 3D support volume from masks, and its
real-image test, are complete. It has not yet been judged whether separation of the three
fish or ID retention improved.**
Next, this support information is used together with SLEAP to evaluate the body shape of the
three fish and their correspondence over time.

[Result image index](../../work/mask_volume/feasibility_v2/index.html) /
[Run record](../../work/mask_volume/feasibility_v2/summary.json) /
[Independent audit](../../work/mask_volume/feasibility_v2/independent_audit.json) /
[Research plan](../../docs/MASK_VOLUME_RESEARCH_PLAN.md)

## What was run

- Reused the Detectron2-specific environment left in the old repository.
  Real inference on the RTX 5070 Ti succeeded. The environment was not reinstalled and
  nothing was retrained.
- Selected 10 points, by time alone, from the commonly stored frames of the old pilot cache:
  132000, 132555, 133110, 133665, 134220, 134774, 135329, 135884, 136439, 136994.
  This is not a contiguous interval; it is a development diagnostic on still frames within
  132000:137000.
- Newly applied the stored top/side models to the 3 raw images at each time point.
  Score threshold 0.5, input the original full frame. The model's standard internal resize
  is as configured. The full SHA256 of the models and configs was checked against the asset
  ledger, and the configs actually used were saved.
- Compared 20 cases in total: Detectron2 for all 3 views, and top alone replaced with fixed
  background subtraction. SLEAP, existing 3D trajectories, individual IDs and ground truth
  annotations were not used to generate the 3D support volume.
- Projected the centres of 2 mm-spaced cells, 96×96×96 = 884,736, through the existing
  refraction model. x/y are 0–192 mm, z is −6–186 mm. The 6 mm floor tolerance is an earlier
  engineering setting, not a measured correction.
  Cases where support inside the masks agrees in all 3 views and cases where it agrees in
  2 or more views were saved separately.

68 masks were obtained from 30 images. Detections per frame: top 1–2, side1 1–3, side2 1–4.
Masks, bboxes and scores for the selected frames matched the old cache.
This is a reproduction check on a short sample, not grounds for treating detection counts as
fish counts, and not a verification of ID.
Because frames were chosen where the original cache had a stored group, there is a selection
bias that excludes detection failures, and detection recall is not estimated from this test.
Independence from the models' training frames is not guaranteed either.

## What was observed

A "component" below is a region of connected adjacent support cells; one component is not one
fish. Face-adjacent 6-connectivity was used, and no component was removed by size or count.

| Input | 3-view support cells / frame | 3-view support components | ≥2-view support components |
|---|---:|---:|---:|
| Detectron2, 3 views | 109–164 | 1–5 | 1–4 |
| top background subtraction + side Detectron2 | 120–191 | 1–5 | 3–26 |

In the reprojection onto the raw images, support regions remained on fish-like shapes and near
overlaps. Replacing top with background subtraction increased the 3-view support cells in all
10 frames, but small scattered regions also increased. For example, at 134220 3-view support
went from 128→191 cells, while ≥2-view components went from 1→21. This is not counted as
"recovering fish". Support also gathers in the bottom corner at 136994, but it cannot be
confirmed from a still image that the three fish were reconstructed separately.

A test covers a case in which even perfect masks of three synthetic fish produce an extra,
fourth intersection region. In real images, reflection, overlap within the same view,
incomplete masks and grid coarseness can merge or split components further. Therefore
**the approach of picking the three largest components is not adopted**.
A single-frame union mask is unchanged when IDs are permuted, so on its own it cannot
determine the temporal identity of the two males.

## Numerical precision was diagnosed and fixed

In the first float32 version, CPU and GPU support counts differed by one cell in 3 of 20
cases. That output and its code snapshot were saved in `feasibility_v1`.
With inputs, thresholds and grid unchanged, sampling was changed to float64 and all 20 cases
were rerun in `feasibility_v2`.

In the final version, CPU/GPU disagreements in observability and support count are 0 in every
case. The maximum absolute difference in support value was 1.72×10⁻¹³. This is numerical
agreement on this sample, not ID accuracy. The old float32 mode is kept for reproduction.

## GPU and speed

Detectron2 ran in the existing dedicated environment, Python 3.11.15 / torch 2.10.0+cu130 /
Detectron2 0.6; the support-volume computation ran on this repo's torch 2.13.0+cu130. Both
used the RTX 5070 Ti. The two environments are connected through hashed NPZ files and the raw
images.

- Detectron2 on 30 images: 4.93 s including initialisation and NPZ saving. Excludes PNG
  pre-loading and model checksums.
- Final float64 support evaluation: median 0.314 s with GPU, 1.091 s for the CPU version, per
  case. A speed difference of about 3.5× on the evaluation part at the same precision.
  Excludes refractive-projection precomputation and image loading.
  Connected-component tallying and similar steps run on CPU in both versions.
- Final 20-case runner: 39.43 s including the CPU cross-check and image output.
  This is not the throughput of a continuous three-fish tracker, nor of a whole 30-minute
  video.

## Verification and next decision

The 56 related tests pass, with no CUDA tests skipped, and Ruff is clean on the added code.
They check geometry using refracted rays and silhouettes of a sphere, extra intersection
regions, missing views, unknown pixels, independent CPU/CUDA sampling, threshold boundaries on
a 1200 px image, and I/O. Hashes of the input PNGs, masks, model configs and code snapshots
were also verified.

**The next candidate is "independent masks + SLEAP + a body model of the three fish".**
The support volume alone is not promoted as a finished fish detector or ID tracker.
The next short-interval test evaluates whether the three fish jointly explain the foreground
in the images, whether multiple individuals avoid explaining the same image region twice, and
the consistency of motion and shape before and after occlusion.
A small number of Gaussians is included among the candidate body representations, but image
agreement alone is not taken as an ID improvement.

The decision to adopt or reject a method reports, side by side, accepted wrong IDs against
independently confirmed correspondences for the three fish, the acceptance rate, and the
number of scenes reviewed and the time required. The one minute opened last time becomes a
regression test, and new unused intervals are reserved for evaluation after the settings are
frozen. Results with human corrections are kept separate from automatic results.

## Rerun

With the fresh mask/raw inputs present, specify a new output name.

```powershell
.venv/Scripts/python.exe -B labs/20260908_mask_volume/run_feasibility.py --masks work/mask_volume/fresh_masks --raw work/mask_volume/inputs/raw_v1 --calibration work/generative_tracking_inputs/calibration/fixed_refractive_192x192x186/phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration --out work/mask_volume/feasibility_reproduction --device cuda
```

Completed run folders are not overwritten. For inference, staging and image extraction, see
each CLI in the same lab and the corresponding manifest/provenance.
