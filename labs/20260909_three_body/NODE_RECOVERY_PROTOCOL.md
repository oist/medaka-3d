As a new algorithm arm, all candidates of the original v1 are retained, and mouth-only
reacquisition candidates are added for slots where, in the leading parent, the current pectoral
fin is directly supported and the mouth is not directly supported. This is not an auxiliary
input-acquisition step or ground-truth label generation.

The remaining defaults of `TrackerConfig(mode="detectron2", device="cuda", flow_proposals=True)`
are fixed. beam=6, opt_steps=14 and the original max_hypotheses=18 are not changed. After the
original candidates, the NodeRecoveryConfig default of at most 2 candidates × 3 slots is added,
so the actual number of candidates is at most 24. The original 18 candidates are not
reconfigured to 24. `AcceleratedNodeRecoveryTracker`, which combines the candidate generation
of `NodeRecoveryThreeBodyTracker` with the already-verified Graph version of `_fit`, is used.

NodeRecoveryConfig defaults: RMS≤1.5 mm, 0.5–1.5 × the initial mouth–pectoral-fin length, the
process box (0,0,-6)–(192,192,186) mm, at most 2 candidates per slot, at most 4096
combinations. In the added starting candidates, coordinates other than the mouth and the plan
are not changed, and the mouth rows of other slots are protected even when they are not
directly supported. In the subsequent existing joint fit, all nodes may move as before. The
final observation and review decisions are recomputed by the original tracker. A candidate plan
is a trial correspondence, not an observation-confirmed source. Direct pectoral-fin support,
ray agreement and candidate score do not mean the identity is correct.

The CLI accepts only `--sequence`, `--input-root` and `--output`. The permitted full intervals
are 333519:333719 (200 frames), 76620:77286 (666 frames) and 76620:86580 (9960 frames). All are
contiguous native frames at 166fps; prefix selection and mid-run reinitialization are not
allowed. The previously fixed first registered 3D pose is used exactly once, and external
resets are 0. Intermediate human GT, old baseline trajectories and old v1 candidates are not
input to inference.

The old v1 output is opened only after inference has fully returned and the new candidate has
been saved. Agreement of the same initial seed, the resolved paths and SHAs of all inference
inputs, and the original TrackerConfig is verified, and the comparison uses the same shared
candidate_metrics. The GT mouth annotations and all 7 C34 arms are also evaluated only after
inference, and unscored endpoints are kept. All 19 original arrays are saved, and the
cumulative mouth-reacquisition diagnostics are saved separately to the summary. The new arm's
summary is made compatible with the completion, native frames, initialization count and reset
conditions of the existing `evaluate_minute.load_completed`, but is distinguished from the
original v1 by algorithm_arm.

Snapshots/SHAs of the code and protocol, calibration, registered seed, raw frames, masks,
background and C01 inputs are recorded and re-verified before saving finishes. An existing
output directory is refused, and the output of a run that failed partway is not given a success
seal. The original run_experiment, tracker, Graph code, existing candidates and human JSON are
not changed. runtime measures only the new arm's loading and tracking, excluding D2
preparation and evaluation.

No tracking is run at the implementation-preparation stage. When run, each output is saved to a
separate directory. Adding candidates, recovering geometric support or increasing the output
rate alone does not count as successful identity maintenance. This is a development comparison
on three already-opened intervals, not an unseen evaluation. Missing observations other than the
mouth, mixed masks, identity mismatches on the reference side, and male-to-male identities that
a few female endpoints alone cannot verify remain limitations.
