# Continuous native-rate minute: v1 Detectron2 plus image-flow proposals

Registered before staging or tracking results for the full minute are available.
The input staging process has begun; no output from this minute has been inspected
to select or change the tracking settings.

Process every source frame in `[76620, 86580)`: 9960 frames at 166 fps, exactly
60 seconds. Use the existing registered first pose at 76620, one tracker instance,
one call to `track`, and no later external seed or reset. Keep the existing
Detectron2 models, threshold 0.5, FP32 batch size 1, and all independent detections.
Use v1 `detectron2` mode with `flow_proposals=True`, all other TrackerConfig
defaults, CUDA float64 body fitting, and the existing frozen refractive calibration.

This condition is selected from the two already opened development windows:
it improved direct support over v0 Detectron2 in both, achieved 97.20% accepted
pectoral support in the 666-frame window, and outperformed v1 top contrast there.
The 200-frame transient still has an unresolved missing episode and sticky ID
review; this experiment does not assume that problem has been solved.
The minute overlaps previous development work and is a technical stress test,
not a held-out estimate of identity accuracy.

Keep original source frames, lossless staged images, independent packed masks,
per-frame decoded-image hashes, input/model/code hashes, progress, and all result
arrays. A completed candidate file alone is insufficient: the final summary and
provenance must both confirm successful evaluation and unchanged inputs/code.
There is no checkpoint resume in this version. If interrupted, preserve the
failed output and repeat from the original first frame in a new directory.

After tracking returns, compare all seven frozen C34 conditions and the previous
generative minute using the shared evaluator. Report direct support, accepted
support, per-fish missing episodes, geometrical jumps, source conflicts,
identity-review episodes and grouped review scenes. The previously specified
90% overall / 80% per-fish accepted-support gates are engineering criteria only.
Report all requested female anchor intervals, including missing or unscored
ones. Do not infer either male's identity from female-only labels or low loss.

The separate v2 short-window refinement is not incorporated into this forward
minute and cannot automatically clear review flags. Independent human three-fish
correspondences remain a separate evaluation input. No universal, dense,
one-minute, or thirty-minute swap-free verdict is possible from coverage alone.
