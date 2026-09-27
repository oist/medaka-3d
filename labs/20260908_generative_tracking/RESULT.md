---
verdict: parked
date: 2026-09-08
expires: 2026-11-08
blocker: Independent identity ground truth for all three fish through encounters is required before promotion can be assessed.
---

# Tracking by predicting images from 3D hypotheses: implementation and comparison

**Implemented and compared on real video, but not at the stage of replacing the current
method.** There were scenes where image evidence kept the mouth and tail observed. However, in
a short interval the loss of one fish was never resolved, and at the end of another interval
the pectoral fin was missing for all three fish. Even in the interval where the female's ID
can be evaluated, the end point was lost, and swap-free tracking of 3 fish over 30 minutes is
not achieved. The scope of this comparison is 2 known difficult intervals, 866 frames in total
(about 5.22 s).

Verdict `parked`; not adopted for production. Independent ID ground truth for the three
individuals is lacking, and building it is required to verify exchanges among all
individuals. Separately, this implementation's reacquisition capability is clearly
insufficient. More ground truth alone would not fix the failures seen here. The implementation
and the failure cases are kept as input for the next decision, and the expiry is updated to
2026-11-08.

## What was implemented

A new `GenerativeTracker` was implemented. It holds the 3D positions of the mouth, pectoral fin
and tail of each of the three fish as internal state, projects them into the 3 cameras
including refraction, and updates position, pose and correspondence candidates to fit the
actual SLEAP points and images. This is not parameter tuning of the existing chain.
Calibration, existing SLEAP output, input/output provenance management and evaluation
annotations are reused.

Each frame performs 3D motion prediction, projection into each camera, generation of
correspondence candidates with detected points, local optimisation of the 3D pose, and
comparison of the candidates. When observations are insufficient, missing can be chosen.
SLEAP's per-instance grouping is not held fixed; correspondence is reconsidered per node.
From the images, agreement with the foreground, the colour distribution near the fish body and
optical flow are used. In views with overlap, the confidence of colour and flow is lowered.

Up to 4 history candidates are retained, and the best history is selected by tracing back from
the end of the interval. This is an approximate search with pruning. Candidates discarded
earlier are not revived, and 3D poses at all times are not jointly re-optimised. The projection
onto images is likewise a comparison of foreground support, colour and motion along the
skeleton, not a full image generator that renders the fish surface or occlusion.

The internal predicted pose `pose_xyz`, the pectoral-fin observation candidate supported by
2 or more views `xyz`, the per-node observation state `observed` and the needs-review flag
`review` are saved separately. `accepted_xyz` is the pectoral-fin coordinate with needs-review
entries further removed. The adapter to the existing Linker also returns this. "Accepted" is
this implementation's judgement and does not mean a human has vouched for its correctness.
Sex is not inferred; IDs are anonymous 0/1/2 carried forward from the initial state.

## Comparison conditions

The targets are 2 known difficult intervals of `20251225_K_1_2_FCMC`. All frames at 166 fps are
used, without temporal subsampling.

| Interval | Frames (end exclusive) | Length | Female annotations usable for evaluation |
|---|---|---:|---|
| Interval with approach and ID change | 76620:77286 | 666 frames, about 4.01 s | 76698, 77285: 2 points = 1 interval |
| Interval with transient detection breakdown | 333519:333719 | 200 frames, about 1.20 s | 1 point at 333597, no interval comparison |

The current method is the saved results of `c27 → c28 → c33 → c30v4 → c34`.
All 7 conditions λ = 0, 0.1, 0.25, 0.5, 0.767, 2, 8 were compared.
Neither the current method nor the raw video was changed for the comparison.

The new method is initialised, on the first frame of each interval only, with the three fish's
skeletons reconstructed from the λ=0.767 correspondence. After that, no coordinates or IDs from
the current method, and no human annotations, are passed to inference. The initial
correspondence itself, however, is output of the current method run over the whole recording.
This comparison is therefore a short-interval recovery experiment given favourable initial
conditions. It does not show performance on unseen data, or the ability to follow 30 minutes
without reinitialisation.

The female's ID is scored as the ID to which the detected point corresponding to the human
mouth position belongs, using an evaluator shared by both methods. If both ends have the same
ID the interval is kept; if they differ, changed; if even one end is unavailable, not
evaluable. Not evaluable is not counted as successful retention. Swaps that happen and revert
in between, and exchanges between males, cannot be counted with these annotations. A direct
comparison of the current method's pectoral fin with the new method's mouth is also avoided.

The denominator of the observation rate, acceptance rate and needs-review rate is
"frames × 3 fish". The current method's output rate is coordinate availability, not accuracy
itself. The current method has no needs-review judgement equivalent to the new method's, so
needs-review rates cannot be compared directly.

## Failures found in the initial version, and fixes

In the first 62-frame trial run, small cost differences caused by tail-missing candidates were
misinterpreted as ID ambiguity, and all data were marked needs-review. v2 was fixed to use
history differences corresponding to an ID swap, missing mouth and pectoral fin, and similar
signals. Only the review judgement changed.

The subsequent v2 run on the 2 intervals revealed a more serious problem. When detection broke
down, some nodes kept drifting away from the fish body; in the 200-frame interval, pectoral
fin–tail stretched from 17.49 mm to 174.29 mm. Even when the mouth or tail was lost, the
pectoral fin alone moved smoothly, so the anomaly could not be detected from the count of large
coordinate jumps alone.

An independent synthetic test that moves a false detection little by little also reproduced
stretching to about 5–10× the initial length. The causes were that the robust loss meant for
observation outliers was also applied to the body-shape constraints, and that the velocity of
unseen nodes kept being extrapolated independently.

v3 uses the robust loss only for images and detected points, and squared loss for the
body-shape and motion constraints. Prediction is a common translation of the whole fish body,
and a candidate whose inter-part distance exceeds 2× its initial value is not accepted as an
observation of that fish. This 2× cap is an engineering guard against the anomaly, not a value
validated as the anatomical tolerance of medaka.

| v2 measurement | 666-frame interval | 200-frame interval |
|---|---:|---:|
| Pectoral-fin observation candidates | 1486/1998 (74.37%) | 417/600 (69.50%) |
| Needs review | 642/1998 (32.13%) | 366/600 (61.00%) |
| Accepted, excluding needs-review | 1356/1998 (67.87%) | 234/600 (39.00%) |
| Female annotation interval | Mouth not found at the end point; 1 interval not evaluable | 1 annotation point, no interval comparison |
| Processing time (loading, image processing, inference) | 949.10 s | 326.20 s |

The v2 result is not a success in swap prevention.

## v3 and the no-image control

The no-image control uses the same initial state, settings and 2 intervals, with only the image
evidence disabled. It removes foreground, colour and flow together and does not measure each
contribution individually. This additional comparison was written into the README before the v3
results were seen. No per-condition retuning is done.

| Interval, method | Pectoral-fin observation candidates | Accepted, excluding needs-review | Needs-review rate | Female annotation interval |
|---|---:|---:|---:|---|
| 666 frames, v3 with images | 1442/1998 (72.17%) | 1357/1998 (67.92%) | 32.08% | 1 interval not evaluable |
| 666 frames, v3 without images | 1490/1998 (74.57%) | 1342/1998 (67.17%) | 32.83% | 1 interval not evaluable |
| 200 frames, v3 with images | 407/600 (67.83%) | 396/600 (66.00%) | 34.00% | 1 annotation point, no interval comparison |
| 200 frames, v3 without images | 405/600 (67.50%) | 236/600 (39.33%) | 60.67% | 1 annotation point, no interval comparison |

In the 666-frame interval, adding images changed the acceptance rate by 0.75 points, and the
number of pectoral-fin observation candidates was higher without images. Mouth and tail support
improved, but in both conditions the female corresponds to S1 at entry 76698 and has no mouth
correspondence at exit 77285. For both the raw candidates and the results excluding
needs-review: kept 0, changed 0, not evaluable 1. In the final 179 frames, 77107–77285, all
three fish are needs-review. At the final frame 77285 the pectoral fin is missing for all three
fish, and the comparison images also confirm that the predicted skeletons have moved away from
the physical fish. Partial tail support is not interpreted as tracking having recovered.

In the 200-frame interval, S0's mouth observations rose from 20 without images to 180 with
images, and its tail observations from 32 to 162. The acceptance rate was also 26.67 points
higher. On the other hand, S2's pectoral fin was obtained in only 17/200 frames in both
conditions and did not recover. This is evidence that image evidence affected node support and
this implementation's acceptance judgement. It does not mean that the proportion of correct IDs
improved by 26.67 points.

The maximum inter-part distance for v3 with images was within 1.295× the initial length in the
666-frame interval and 1.235× in the 200-frame interval. The stretching of about 9.978× in v2 on
the 200-frame interval was resolved. However, the internal prediction of the lost S2 also falls
within this range. A plausible body shape alone is not evidence that the fish is being tracked.
v2→v3 changed the loss, the velocity prediction and the cap at the same time, so the
improvement cannot be interpreted as the effect of the cap alone. On real video, v3's cap
rejected candidates 0 times.

### Comparison with the 7 conditions of the current method

Results on the same 666-frame interval, evaluated with the same female annotations.
"Changed" is an interval judgement that the IDs at the two ends differ, not a count of exchange
events.

| Current method λ | Pectoral-fin coordinate output rate | Female annotation interval | Adjacent-frame displacements > 10 mm |
|---:|---:|---|---:|
| 0 | 1998/1998 (100%) | kept 1 | 3 |
| 0.1 | 1998/1998 (100%) | changed 1 | 4 |
| 0.25 | 1998/1998 (100%) | changed 1 | 3 |
| 0.5 | 1995/1998 (99.85%) | changed 1 | 1 |
| 0.767 | 1995/1998 (99.85%) | changed 1 | 1 |
| 2 | 1995/1998 (99.85%) | kept 1 | 0 |
| 8 | 1980/1998 (99.10%) | kept 1 | 0 |
| New method v3 with images | observed 1442/1998 (72.17%) | not evaluable 1 | 0 |

Even though the new method has no large displacements, a result that lost the end point is not
counted as better ID retention than the current method. In the current method too, λ=0, 2, 8
keep this 1 interval. Drawing a conclusion favourable to the new method by comparing against
λ=0.767 alone was avoided. The optimal λ cannot be chosen from this 1 interval either.

In the 200-frame interval, all 7 current conditions had a pectoral-fin coordinate output rate of
100%. An ID corresponding to the one female annotation is obtained, but with 0 intervals no
method's ID retention can be scored. There are no male-vs-male ground truth labels in this
comparison either.

### Processing time and application to the full recording

v3 with images took 967.76 s (0.688 fps) for 666 frames and 304.57 s (0.657 fps) for 200 frames.
This includes video loading, image processing and tracking; background construction took about
8 s separately. Without images the times were 103.40 s and 27.91 s respectively, but video
loading and image processing are omitted. For the current method only the saved results were
read; its compute time was not measured. Some processing ran in parallel, so these are not
strict speed benchmarks either.

Because reacquisition already failed on short difficult intervals, this version did not proceed
to running the full 30 minutes. Besides improving the candidate search, image-processing speed
and the memory used to retain histories are issues to address before applying it to the full
recording. Camera frame numbers and video metadata were checked, but sub-frame synchronisation
accuracy and the effect of the moving water surface were not newly verified.

## What the next development must solve

Candidate generation that can reacquire a fish that has drifted away from its observations is
the top priority. Currently an effective condition restricts ordinary node candidates to within
about 15.5 px of the predicted pixel, and nothing widens the search range or the motion
uncertainty according to how long the fish has been lost. Adding image terms alone cannot turn
distant unused detections into new recovery candidates.
This is an interpretation from the code, not a determination that isolates the cause of every
failed frame.

The next version first verifies regenerating wide-area 3D candidates from the 3 views after a
dropout and connecting them to reliable trajectories before and after. After that, it widens
the range over which correspondence and 3D pose are re-optimised forward and backward in time.
A fish-body and occlusion model that also accounts for duplicate pixel use and reflection is to
be decided on after seeing those results and the no-image control. These failures alone do not
lead to the conclusion that the direction itself is impossible.

Validating production adoption requires ground truth IDs independently confirmed for each of
the three fish before and after encounters, and evaluation intervals separate from the known
intervals. If a small amount of human checking is built in, the number and length of the
intervals checked and the exchanges remaining after correction must also be measured.
What was implemented here stops at the needs-review flag and the viewing artefacts; a correction
UI and re-optimisation workflow that propagate human-corrected IDs through the full recording
are not yet implemented.

## Reproduction and artefacts

Commands, settings and constraints are in [README.md](README.md).
Large inputs and experiment outputs are in `work/`, outside Git.
The 18 small input files are copied locally with SHA-256 checked against the originals; raw
video is read from the original shared storage. Each run saves a complete code snapshot,
checksums, settings and start/end times, and refuses to overwrite an existing output location.

| File | Purpose |
|---|---|
| [tracker.py](tracker.py) | Prediction from 3D hypotheses, projection into images, update of correspondence and pose |
| [observations.py](observations.py) | Foreground, colour and flow image evidence |
| [benchmark.py](benchmark.py) | Comparison run using raw video and the 7 current conditions |
| [run_without_images.py](run_without_images.py) | Control experiment with image evidence removed |
| [stage_inputs.py](stage_inputs.py) | Copying only the required inputs and checking their integrity |
| [calib.py](../../src/medaka3d/calib.py) | 3D→pixel projection accounting for refraction and lens distortion |
| [eval/generative.py](../../src/medaka3d/eval/generative.py) | Shared ID evaluation; missing-observation, motion and skeleton diagnostics |

Software verification is kept separate from tracking correctness. The tests check agreement of
refractive projection with the physical formula, missing and mixed detections, that images
influence the 3D estimate, history selection from future observations, rejection of abnormal
body stretching, prevention of needs-review data leaking into normal output, and so on.
Their passing is not a guarantee that the actual IDs of the three fish are correct over the full
recording.

In the final state, repository-wide pytest gave **385 passed, 2 skipped**.
There were 2 All-NaN warnings from the existing `c30_slot_stitch.py` and no test failures.
The Ruff check on the added and changed Python files and `git diff --check` also passed.
For both v3 runs, the saved copies, recorded hashes and current versions of the 5 code files were
confirmed to match, and accepted coordinates, observation counts and maximum skeleton lengths were
recomputed from the saved NPZ and checked against the JSON.
Agreement of the initial correspondence, initial 3D pose, settings and code with the no-image
control was also confirmed.

```powershell
.venv/Scripts/python.exe -m pytest -o addopts='' -q --basetemp work/pytest_generative_final -p no:cacheprovider
```

The main run results are below. Links go from this report to
`work/generative_tracking/` in the repository.

| Result | Numbers | Images |
|---|---|---|
| v3 with images, 666 frames | [comparison.json](../../work/generative_tracking/anatomy_v3_interval_76620-77286/comparison.json) | [3-camera comparison](../../work/generative_tracking/anatomy_v3_interval_76620-77286/preview.html) |
| v3 with images, 200 frames | [comparison.json](../../work/generative_tracking/anatomy_v3_transient_333519-333719/comparison.json) | [3-camera comparison](../../work/generative_tracking/anatomy_v3_transient_333519-333719/preview.html) |
| v3 without images, 666 frames | [comparison.json](../../work/generative_tracking/ablation_no_images_interval_76620-77286/comparison.json) | — |
| v3 without images, 200 frames | [comparison.json](../../work/generative_tracking/ablation_no_images_transient_333519-333719/comparison.json) | — |
| Failed v2, 666 frames | [comparison.json](../../work/generative_tracking/interval_76620-77286/comparison.json) | [3-camera comparison](../../work/generative_tracking/interval_76620-77286/preview.html) |
| Failed v2, 200 frames | [comparison.json](../../work/generative_tracking/transient_333519-333719/comparison.json) | [3-camera comparison](../../work/generative_tracking/transient_333519-333719/preview.html) |

The [v2/v3 comparison figure of fish body, observation and needs-review (200 frames)](../../work/generative_tracking/transient_v2_v3_diagnostic.png)
shows internal prediction as dashed lines and observation support as solid lines, separately.

The [666-frame comparison figure](../../work/generative_tracking/interval_v2_v3_diagnostic.png),
to the same specification, also shows the final period in which all three fish are
needs-review. Each figure comes with SVG, CSV and JSON files and a record of the plotting script.

The actual motion can be checked in the [666-frame comparison video](../../work/generative_tracking/comparison_videos/interval_76620-77286/index.html)
and the [200-frame comparison video](../../work/generative_tracking/comparison_videos/transient_333519-333719/index.html).
The top row is the current λ=0.767, the bottom row v3. They are 4× slow motion drawn every
4 frames, show the original frame numbers, and the top and bottom panels of the same view share
a crop fixed over the whole interval. Inference ran on every frame; subsampling is for display
only. The output MP4 is 1440×960 at 10.375 fps; all 167 / 50 images were re-decoded and checked.
