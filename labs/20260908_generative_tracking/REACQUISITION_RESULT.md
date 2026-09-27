# Results of the reacquisition cycle

**The one minute without reinitialisation ran to completion technically, but the goal of
producing usable trajectories that keep the IDs of the three fish is not met. Against an
observation rate of 99.76%, the acceptance rate was 3.56%.**
The 2 known intervals and the 4 validation intervals run after the settings were frozen also
ran to completion.
On the 4 fixed intervals, against a position observation rate of 99.80–100%, accepted output
(output on which ID confirmation is not held) stayed at 23.01–67.83%, and the pre-set
usable-rate criterion was not met on any interval.
The one minute likewise had an acceptance rate of 3.56% against an observation rate of 99.76%.
The female's raw endpoints kept 13 intervals, but after acceptance all 13 intervals are not
evaluated, and the quality needed to output ID-labelled trajectories continuously has not
been reached.

The pre-registration is [REACQUISITION_PLAN.md](REACQUISITION_PLAN.md); all start frames and
initial poses are in [reacquisition_windows.json](reacquisition_windows.json). Source video and
existing results were not changed; each run's new folder holds input hashes, a snapshot of the
executed code, NPZ, the shared evaluation and previews.
All of it is the same single group of medaka. The initial frame is an optimistic seed taken
from the existing method over the full recording, but there is no subsequent reinitialisation
from existing-method predictions or GT.

## Floor sensitivity comparison on the known intervals

"Observation rate" is the fraction of pectoral fins supported by an assignment in 2 or more
views. "Acceptance rate" is that fraction with review targets removed; it is not ID accuracy.
The denominator is frames × 3 fish.

| Interval | Method | Pectoral-fin observation rate | Acceptance rate | Accepted frames S0 / S1 / S2 | Female raw / post-acceptance evaluation | Inference time |
|---|---|---:|---:|---|---|---:|
| 76620:77286 (666fr) | old v3 | 72.17% | 67.92% | 444 / 426 / 487 | 1 interval not evaluated / 1 interval not evaluated | 967.76 s |
| same interval | reacquisition v1, nominal floor z≥0 | 78.83% | 69.12% | 450 / 444 / 487 | 1 interval swap / 1 interval not evaluated | 385.77 s |
| same interval | reacquisition v1, floor tolerance z≥−6 | 98.75% | 85.09% | 514 / 651 / 535 | 1 interval kept / 1 interval kept | 402.11 s |
| 333519:333719 (200fr) | old v3 | 67.83% | 66.00% | 180 / 200 / 16 | 1 female mark, no interval | 304.57 s |
| same interval | reacquisition v1, nominal floor z≥0 | 98.17% | 92.67% | 171 / 200 / 185 | 1 female mark, no interval | 120.19 s |
| same interval | reacquisition v1, floor tolerance z≥−6 | 98.17% | 92.67% | 171 / 200 / 185 | 1 female mark, no interval | 116.99 s |

Old and new use the same frames and the same initial pose. Besides reacquisition, the new cycle
includes CUDA background processing, CPU parallelisation over views, optical-flow max side
640→320, and order-preserving prefetch.
Times include video I/O, image processing and tracking, but some runs executed in parallel with
other intervals, so the table does not represent the standalone speed effect of each change.
Compute time of the existing 7 arms was not measured.

With the nominal floor, the female was S1 at entry 76698 and S2 at exit 77285. The exit was
review and not accepted, so the post-acceptance evaluation is "unknown". With the 6mm floor
tolerance, both ends were S1.
This is the result of a single female endpoint interval and does not exclude round-trip swaps
in between or swaps between males.

Among the existing 7 arms passed through the same shared evaluation, λ0/2/8 kept this female
interval and λ0.1/0.25/0.5/0.767 were swap. In every arm both female marks are usable.
It is not shown that the new approach is better than every existing setting.

## Recovered positions, unresolved IDs

In the 200fr interval, old v3 observed S2's pectoral fin in 17/200fr. Reacquisition v1 raised
this to 193/200fr, and S2's accepted output returned after a 15fr gap at 333535:333550. The
selected reacquisitions were 333537(S2), 333539(S0), 333540(S2), 333542(S0), 333543(S2),
333545(S0), 333547(S2), 333552(S0), 333580(S0), 333600(S0). This is a record of recovered
position support.

However, in the 3-view previews at 333558, 333638 and 333718, the candidate's S0/S2 correspond
to the opposite fish positions from the existing method. Matching by positional proximity alone
over the 190fr in which raw pectoral fins exist for all three fish, the correspondence with the
existing slots was (0,1,2) for the first 18fr and (2,1,0) for the 172fr from 333547 on.
At 333547 the 3-point RMS distance of this correspondence is 0.319mm. The initial slots agree,
so a male exchange at reacquisition is suspected. Because there is no independent male ground
truth annotation, this disagreement is not counted as a confirmed number of male wrong IDs.
Even so, an acceptance rate of 92.67% with identity_review=0 cannot be called a success.
source/observation/review were identical with and without the floor tolerance.

With this v1 result kept, v2 holds the ID as unresolved even after a single dropout and
reacquisition, separating position reacquisition from ID reconnection. Linking by the
6 permutations, using trajectories before and after and initial image information, is
evaluated as a separate post-process and does not re-estimate or interpolate the original 3D
coordinates.

As a cause diagnosis on the development interval only, added after the post-process settings
were frozen, a comparison with `appearance_weight=0` on the same v1 transient was also saved.
The final correspondence changed from the default's [2,0,1] to [2,1,0], and the raw evaluation
ID corresponding to the female mark returned from the default's 2 to the initial 1.
However, the accepted count went from 57 to 56, and both female marks are review. In this local
example the appearance term is thought to have contributed to an inappropriate reassignment of
the female, but this is not proof of improved ID accuracy.
The default settings for the formal 4 validation intervals and the one minute were not changed.
Record: [appearance ablation](../../work/generative_tracking/appearance_ablation_v1_transient/README.md).

## The floor tolerance is not a guarantee of calibration accuracy

Diagnosis on the known intervals found observations whose triangulated z was about −1.4 to
−4.4mm even with a small ray residual, and candidates were being eliminated by the nominal z≥0
constraint. Because the absolute floor position of the calibration is not sufficiently
guaranteed, a sensitivity comparison that tolerates 6mm on the floor only was pre-registered.

In the 666fr floor-tolerance run there were 911 observed nodes below the nominal floor,
351 / 336 / 224 by fish. On the accepted side, excluding review, there were 532
(104 / 336 / 92). The minimum z is about −5.99999mm, touching the new tolerance boundary as
well. In the 200fr interval there were 0 observations outside the nominal floor.
This is neither the result of re-estimating the optical calibration nor of translating
coordinates after the fact.

For the fixed validation, the 6mm floor tolerance was adopted as a process setting. The
reasons are that recovery of observations improved on the known intervals and that the
correspondence in the interval away from the floor was the same.
It was not measured that the floor position is off by 6mm, and widening the floor tolerance
further to get future intervals through is not part of this validation.

## Storage and audit

The following 4 folders in `work/generative_tracking/` are retained.

- `recovery_v1_nominal_dev_interval`
- `recovery_v1_nominal_dev_transient`
- `recovery_v1_floor6_dev_interval`
- `recovery_v1_floor6_dev_transient`

Each `comparison.json`, `candidate.npz` and `benchmark.provenance.json` is the evidence.
The independent audit, `<run name>_audit.json`, checked the recorded hashes and the
hierarchically saved code snapshot, the frame sequence, the shapes of the added
appearance/quality arrays, the review and accepted coordinates, and the observation counts.
All 4 runs completed with a single initialisation, with 0 fit failures and 0 anatomy refusals.
The software running to completion and the IDs being correct are separate judgements.

The first 6 runs of the standard postpass (the 4 development runs and validation swap A/B)
snapshot the main part of the executed code, but some indirect imports such as `fast_fit.py`
are not included. Reproducing them requires the same repo environment or the corresponding
additional dependencies; the snapshot is not claimed to be self-contained on its own.
While all postpasses were stopped, only the wrapper's set of saved files was corrected, and
later runs added the dependencies. The inference core and settings were not changed, nor were
existing outputs.
During the same stop, the JSON save failing on `global_cost=inf` when all paths are rejected
was reproduced with the existing code, and only the wrapper's save step was corrected. In JSON,
non-finite values become null and the affected fields are listed. feasibility=false and review
are kept, and the NPZ values are not changed.

## The 4 validation intervals after the settings were frozen

raw v2 used the 6mm floor tolerance, CUDA background processing, flow max side 320 and the same
tracking settings on every interval, and ran to completion with only the seed at the start.
Intervals and seeds were fixed before seeing the new candidate's results, and the raw results
were not opened until the post-process settings were frozen. swap A/B and occlusion were chosen
from earlier event annotations, so this is not a validation on randomly sampled unknown failure
scenes. control is an interval within the same recording chosen by temporal position.
The distinction between when the settings were decided and when the appended document was saved
is recorded in [REACQUISITION_V2_FREEZE.md](REACQUISITION_V2_FREEZE.md).

Each female column evaluates **one endpoint interval** joining 2 existing manual mouth
annotations. raw is the assignment regardless of review; post-acceptance is the output with
review removed.
"Kept" means both ends had the same ID; round-trip swaps in between and exchanges between males
cannot be checked.

| Fixed interval | Frame range | Pectoral-fin observations / denominator | Accepted / denominator (rate) | Accepted frames S0 / S1 / S2 | Female raw / post-acceptance | Inference time |
|---|---|---:|---:|---|---|---:|
| swap A | 99070:99900 | 2485 / 2490 (99.80%) | 1017 / 2490 (40.84%) | 830 / 138 / 49 | kept / kept | 348.06 s |
| swap B | 105085:106080 | 2983 / 2985 (99.93%) | 1136 / 2985 (38.06%) | 995 / 77 / 64 | kept / kept | 433.69 s |
| occlusion | 323660:324490 | 2486 / 2490 (99.84%) | 573 / 2490 (23.01%) | 39 / 492 / 42 | kept / not evaluated | 375.11 s |
| temporal control | 165950:166780 | 2490 / 2490 (100%) | 1689 / 2490 (67.83%) | 504 / 602 / 583 | kept / kept | 374.05 s |

The pre-set accepted-output rate criterion is at least 90% overall and at least 80% for each
individual. It was not met on any of the 4 intervals.
The female's post-acceptance evaluation is kept on 3 intervals and not evaluated on 1; this is
not "zero errors on 4 intervals".
Raw unresolved-ID animal-frames are, in order, 1471, 1847, 1913 and 797.
v2's policy of holding ID confirmation even when observation returns after a dropout is
working, but it does not sufficiently recover the unresolved IDs. The ID accuracy of the output
on which confirmation was not held also cannot be estimated from this small GT alone.
If the hold continued indefinitely, output containing wrong IDs would be reduced, but usable
trajectories would be lost too.
This increase in permanent holds is not evaluated as "zero swaps achieved". No experiment was
run in which a human corrects IDs midway, and it is not yet demonstrated whether a small number
of human corrections would suffice or how much time that would require.

All 7 baseline arms, passed through the same frames and shared score, are shown without
omission. Each cell is "pectoral-fin outputs / denominator; female endpoint evaluation". In
every arm both female marks are usable.
The baseline output states and the candidate's review policy differ, so a larger output count
is not read as ID accuracy.

| C34 λ | swap A | swap B | occlusion | temporal control |
|---:|---|---|---|---|
| 0 | 2487/2490; kept | 2985/2985; swap | 2490/2490; swap | 2490/2490; kept |
| 0.1 | 2487/2490; kept | 2985/2985; kept | 2490/2490; kept | 2490/2490; kept |
| 0.25 | 2487/2490; kept | 2985/2985; swap | 2490/2490; kept | 2490/2490; kept |
| 0.5 | 2487/2490; kept | 2985/2985; kept | 2490/2490; kept | 2490/2490; kept |
| 0.767 | 2481/2490; kept | 2985/2985; kept | 2490/2490; kept | 2490/2490; kept |
| 2 | 2481/2490; kept | 2985/2985; kept | 2490/2490; kept | 2490/2490; kept |
| 8 | 2481/2490; kept | 2985/2985; swap | 2487/2490; kept | 2490/2490; kept |

At the female endpoints the candidate does better than some baseline arms, but
λ0.1/0.5/0.767/2 also keep all 4 intervals, so it is not shown to be better than every setting.
No method has independent GT for each male individual.

### The fixed post-process linking before and after

The 6-permutation post-process is saved in a folder separate from raw, and its 3D coordinates,
source and observation state were checked to match a recorded permutation of the original
arrays exactly. No coordinates are re-estimated or interpolated.
Settings and model were not changed after seeing the validation results, and results that did
not improve are shown as they are.

| Fixed interval | Post-process accepted / denominator (rate) | Accepted frames S0 / S1 / S2 | Female raw / post-acceptance | Overall geometry/gap constraint judgement |
|---|---:|---|---|---|
| swap A | 328 / 2490 (13.17%) | 141 / 138 / 49 | kept / not evaluated | not satisfied |
| swap B | 774 / 2985 (25.93%) | 633 / 77 / 64 | swap / not evaluated | satisfied |
| occlusion | 572 / 2490 (22.97%) | 39 / 492 / 41 | kept / not evaluated | not satisfied |
| temporal control | 1589 / 2490 (63.82%) | 504 / 502 / 583 | kept / not evaluated | satisfied |

The post-process reduced the accepted count on every interval, and on swap B it changed the
female's raw endpoint assignment. All 4 post-acceptance female intervals are not evaluated.
Therefore no grounds were obtained for adopting this before-after linking as an improvement.
"Constraint satisfied" is a judgement within this model; as swap B shows, it does not mean the
female ID is correct. On the development intervals too, both 666fr runs (nominal floor and
floor tolerance) gave 892 accepted and both 200fr runs gave 57 accepted; ID reconnection was not
sufficiently recovered.

### Image and numerical audit

The independent review examined 15 fixed 3-view previews from the 4 intervals.
At the occlusion end frame 324489, raw S0/S2 and C34 S2/S0 correspond inversely in all 3 views:
agreement at the start has become disagreement at the end. A male exchange is suspected, but
with no male ground truth annotation it cannot be determined which method is correct.
Female S1 corresponds to the same endpoint fish, but is review and not accepted.
In temporal control as well, all IDs are review at the end frame 166779, so endpoint retention
up to the last female mark 166503 cannot be interpreted as extending to the end.

At 99235 in swap A and 105542 in swap B there were images in which the predicted tail is off the
physical fish. Those tails are saved as observed=false, and the whole skeleton drawn on the
image should not be interpreted as measurement. The audit images and evidence are saved in
[independent image QA](../../work/generative_tracking/validation_visual_qa/visual_qa.md).
This is not new GT or a visual inspection of every frame.

For all 4 raw runs, the frame sequence, the single initialisation, input hashes, snapshot
hashes, the accepted mask and agreement of the summaries were confirmed. All recorded raw code
hashes match the current frozen code.
Fit failures and anatomy refusals were 0 in each run, and the maximum ratio to initial length
for each skeleton segment was at most 1.185 across all 4 intervals. The total image/flow cost is
nonzero in each run, but the fraction of cases in which individual image evidence correctly
supported the ID was not measured.

In swap B there were 247 observed nodes below the nominal floor (all S1), and the minimum z of
−5.99999mm touched this tolerance boundary. In the other 3 intervals there are 0 observed nodes
outside the nominal region. The floor constraint was not relaxed further after seeing this
result. Throughput was 2.21–2.38fps per run, but measured while sharing CPU/GPU with the other
intervals and the one-minute run. No speed comparison with the baseline was made.

Storage is `work/generative_tracking/recovery_v2_validation_{swap_a,swap_b,occlusion,temporal_control}`
and the corresponding `reconnect_fixed_validation_*`. Besides each comparison/provenance/NPZ,
numbers and hashes are saved in the aggregated
[fixed_validation_summary.json](../../work/generative_tracking/fixed_validation_summary.json)
and in each raw run's audit JSON.

## One minute without reinitialisation

**76620:86580, 9960fr (166fps, 60 s), ran to completion with a single initialisation.**
There was no stop/restart midway and no human ID correction. After the seed at initial frame
76620, there is no reinitialisation from baseline predictions or ground truth annotations.
Reacquisition based on current observations continued as part of the tracker's processing.
Settings are the same as for the 4 validation intervals above.
This minute is a continuous diagnostic interval fixed in advance, and it contains the failure at
76930 that was known during development. It is distinguished from a generalisation test on
unknown intervals alone.

Pectoral-fin observations were 29807/29880 animal-frames (99.76%), but accepted output
excluding review was 1064/29880 (3.56%). Pectoral-fin observations for all 3 fish are present
in 9887/9960fr; accepted output for all 3 fish is present only in the first 267fr.

| Initial slot | Pectoral-fin observed frames / 9960 | Accepted frames / 9960 (rate) | Review frames | Longest observation gap | Accepted-output gap lasting to the end |
|---|---:|---:|---:|---:|---|
| S0 | 9932 (99.72%) | 267 (2.68%) | 9693 | 7fr | 76887:86580, 9693fr |
| S1 | 9952 (99.92%) | 310 (3.11%) | 9650 | 7fr | 76930:86580, 9650fr |
| S2 | 9923 (99.63%) | 487 (4.89%) | 9473 | 13fr | 77107:86580, 9473fr |

Intervals in which the S0/S1/S2 pectoral-fin observation was temporarily lost and then returned
numbered 12/2/10. At the end all 9 nodes are observed, but every ID is review. Unresolved IDs
came to 9692/9649/9472fr, 28813 animal-frames in total. The gap between reacquiring the
observation and confirming that it is the same individual in order to resume output remains
unresolved. The pre-set acceptance-rate criterion of 90% overall and 80% per individual is not
met, and increasing permanent holds is not treated as having achieved zero swaps.

The female's manual mouth marks are the 14 points 76698, 77285, 78083, 78799, 79547, 80215,
80657, 81410, 82121, 83265, 83853, 84700, 85625, 86350. In raw all correspond to the same S1,
and 13 endpoint intervals were kept. On the other hand, only the first, 76698, was usable after
acceptance, and **all 13 post-acceptance intervals are not evaluated**. Raw endpoint retention
is not a guarantee of anything about round-trip swaps in between, exchanges between males, or
the ID from the last mark 86350 to the end 86579.

All 7 baseline arms on the same frames were also evaluated with the shared score.
In every arm all 14 female marks are usable, and 0 endpoint intervals are not evaluated.

| Method | Pectoral-fin observations/outputs / 29880 | Accepted outputs / 29880 | Female raw endpoints kept / swap / not evaluated | Female post-acceptance kept / swap / not evaluated |
|---|---:|---:|---|---|
| raw v2 | 29807 | 1064 | 13 / 0 / 0 | 0 / 0 / 13 |
| fixed post-process | 29807 | 901 | 13 / 0 / 0 | 0 / 0 / 13 |
| C34 λ0 | 29880 | — | 10 / 3 / 0 | — |
| C34 λ0.1 | 29880 | — | 10 / 3 / 0 | — |
| C34 λ0.25 | 29877 | — | 11 / 2 / 0 | — |
| C34 λ0.5 | 29871 | — | 11 / 2 / 0 | — |
| C34 λ0.767 | 29871 | — | 12 / 1 / 0 | — |
| C34 λ2 | 29850 | — | 13 / 0 / 0 | — |
| C34 λ8 | 29823 | — | 13 / 0 / 0 | — |

The endpoint evaluation in the baseline rows is based on the states each arm declared as
outputtable. The baselines do not have the same review model as the candidate, so rather than
fabricate a column with the same meaning as the candidate's acceptance rate, "—" is shown.
The raw candidate's female endpoints improve on some arms, but λ2/8 also keep 13 intervals, so
it cannot be concluded that the new approach is better than every arm. The number of
endpoint-swap intervals cannot be read as the actual number of swap occurrences either.

The fixed post-process accepted 901 (3.02%), 276/310/315fr by ID, 163 fewer in total than raw.
The original 3D coordinates, source and observation states were not changed other than by the
recorded permutation. Female raw endpoint retention is the same, but all 13 intervals remain
not evaluated after acceptance.
The post-process exceeded the process's movement cap at 5 per-ID links across gaps, so the
overall geometry/gap-constraint-satisfied flag was false. This is the result of linking fixed
trajectories by 6 permutations failing to achieve sufficient ID recovery, and is not evaluated
as an improvement from the post-process.

Reacquisition candidate search ran on 821fr, candidate fits numbered 5993, and selected
reacquisitions were 55 animal-frames. The first were 76889(S0), 76937(S1) and 77109(S2); the
full frame list is saved in `candidate.diagnostics.recovery_events` inside the raw
`comparison.json`.
These are reacquisitions selected from observations, not reinitialisations from external IDs.
Pending confirmation was 433 animal-frames; fit failures and anatomy refusals were 0.
The maximum length ratio to initial length for each segment is S0 1.537/1.144, S1 1.214/1.276,
S2 1.808/1.307 (mouth–pectoral fin / pectoral fin–tail, in that order). The longest, 1.808×,
includes latent prediction; this is not an assessment that the skeleton was always correct.

Observed nodes below the nominal floor, z<0, numbered 9707 (1648/4047/4012 for S0/S1/S2).
The minimum z is −5.99999mm, touching the tolerance boundary. Observed nodes excluding review
include no points below the nominal floor, but since acceptance is confined to the beginning
this does not show that the floor model is correct.
The total image/flow cost was 32183.13, nonzero, but ground truth annotation to directly measure
the usefulness of image evidence for ID is lacking. No calibration re-estimation or further
relaxation of the floor constraint was done.

Raw tracking time was 3582.26 s (59.70 min), a throughput of 2.780fps. Setup of 9.11 s is
separate; tracking time includes video I/O, image processing and inference. The first part
shared CPU/GPU with the 4 validation intervals, so this is not a measurement of standalone
hardware performance. The post-process took 2.104 s on CPU (linking computation only).
For the baseline, saved outputs were evaluated, so compute time was not compared.
The GPU used was an NVIDIA GeForce RTX 5070 Ti, with 9960 CUDA image-preprocessing calls.
The GPU was used for image background processing; per-view processing, flow and optimisation
also include CPU processing.

The independent audit confirmed the contiguous frame sequence of 9960, the single
initialisation, agreement with the pre-set seed, a monotonic progress log of 201 entries,
hashes of the 18 input files, input/output hashes of raw/postpass, hashes of the run snapshot
and of the current frozen code, the review and accepted masks, and permutation agreement of the
post-process coordinates and source.
No rerun with changed validation code or settings was done. No human ID correction midway was
done either, and whether a small number of corrections would suffice is not verified.

The evidence is in the following saved artefacts.

- [raw comparison JSON](../../work/generative_tracking/recovery_v2_floor6_continuous_one_minute/comparison.json)
- [fixed post-process comparison JSON](../../work/generative_tracking/reconnect_fixed_continuous_one_minute/comparison.json)
- [raw numerical audit](../../work/generative_tracking/recovery_v2_floor6_continuous_one_minute_audit.json)
- [independent audit of raw and post-process](../../work/generative_tracking/continuous_minute_independent_audit.json)
- [timeline of observation, acceptance and hold](../../work/generative_tracking/timeline_continuous_one_minute_precise/review_timeline.png)
- [one-minute 3-view comparison video](../../work/generative_tracking/review_videos/recovery_v2_minute/index.html)

The comparison video is 2490 images showing every 4th frame of the source video, about 4 minutes
at 4× slow motion. Decoding of all 2490 images and the input hashes were checked, but this is
not a visual annotation of all 9960fr.
The top row is C34 and the bottom row raw v2; hollow skeleton nodes are unobserved predictions.

The full test suite gave 432 passed / 2 skipped, and Ruff was clean (2 All-NaN warnings from the
existing c30). These are checks of the implementation, not tests that guarantee biologically
correct IDs.

What was achieved here is processing a minute fixed in advance without reinitialisation and
evaluating it, failures included. Quality sufficient to output accepted IDs for the 3
individuals continuously, zero swaps for all 3 individuals, and applicability to 30 minutes
are not confirmed.
