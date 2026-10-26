# Dataset delivery with explicit identity support

Date: 2026-09-30. Status: working direction and delivery plan; no new tracking
run, algorithm promotion, or dataset-wide accuracy claim.

## User-defined objective

Analyze `/bucket/StephensU/medaka/medaka_202512` using SLEAP poses and the
tracking/reconstruction work already developed. Deliver 3D trajectories that
collaborators can use for subsequent analysis. Trustworthy identity is essential;
where complete coverage is unattainable, maximize the duration of trustworthy
tracking intervals. Exclude the first five minutes of every recording session as
preparation time.

**User clarification:** The same individual must have the same ID throughout each
trial. Independent segment IDs alone do not satisfy the delivery requirement.
Missing frames are acceptable; silently changing the individual denoted by an ID
after an interruption is not. The immediate unit here is one recording session;
links across separately recorded days are a distinct, unestablished claim.

This changes the next milestone from repeated tracker refinement to a usable,
auditable dataset. It supersedes the older next-task ordering in
`NEXT_SESSION.md`, `ROADMAP.md`, and `DEVELOPMENT_STRATEGY.md`. Historical results
and their limitations remain evidence, not automatically promoted methods.

Here, "reliable interval" means a temporal tracking segment, not a statistical
confidence interval. Numerical reliability thresholds have not yet been
established. Within-trial ID consistency is required; cross-day identity evidence
must be represented separately.

## Analysis time

- Each recording session has its own preparation exclusion, including separate
  days in a multi-day trial. The analysis window is `[300 s, recording end)` on
  the recording's documented common time axis.
- Preserve both recording time and experiment time (`recording_time_s - 300`),
  original frame indices, and the mapping to each camera's native frames.
  Establish timestamp origins and camera synchronization before selecting rows.
- Use acquisition timing where available. An FPS-derived conversion must record
  its assumption; at verified constant 166 fps with frame 0 at time zero, the
  first included frame is 49,800. Do not hardcode that index for every recording.
- Preparation frames may supply tracker context, but do not count toward delivered
  analysis coverage. Missing predictions, missing cameras, processing failures,
  and ambiguous identity within the analysis window remain in the accounting;
  do not improve coverage by silently shortening its denominator. Record sessions
  with no post-preparation time explicitly.

## Identity and position are separate claims

Keep camera-local fragments, 3D continuous segments, and identities linked across
segments distinct. A numeric slot or provisional tracker ID does not establish
physical identity. Neither a low geometric residual nor absence of an automatic
review flag proves correct identity.

The export retains local segments as supporting data while leaving uncertain
relations between them unresolved. Only segments linked with supporting evidence
to the same trial-wide individual count toward the primary usable-time objective.
Do not join across an ambiguous crossing just
to lengthen a track. An unaffected fish can continue through another pair's
ambiguity. Later clear observations can start a new segment, but cannot restore
the old fish identity without supporting evidence. Also do not discard reliable
links merely because mouth/tail observations are missing.

Record segment IDs, nullable linked fish IDs, the scope of each identity claim,
link evidence, and whether support is automatic, independently reviewed, or
unresolved. Physical names such as female/M1/M2 and correspondence across days
need their own evidence; anonymous local A/B labels must not imply that naming.
Establish a documented identity anchor and accepted correspondence evidence for
each trial-wide ID. Anonymous A/B/C can denote stable individuals within one trial
if supported throughout the accepted segments; physical M1/M2 naming is a separate
claim. Where segment-to-individual correspondence remains unresolved, retain the
segment for review with a null trial-wide ID, exclude it from the accepted
identity-bearing trajectories, and keep it in coverage denominators. Flag review
needs when automated evidence cannot resolve a link. Neither a fresh local ID nor
the re-use of an old numeric label resolves the required correspondence.

Preserve original observations and missingness. Keep predicted/interpolated
positions separate from observed reconstruction. Distinguish position validity,
node validity, and identity validity; a valid position can have unresolved ID.
Scores are diagnostic values unless calibrated against independent identity
evidence; do not label arbitrary margins as probabilities of correctness.

## What to measure

Within-trial identity consistency is a constraint; duration and continuity of
usable positions attached to those same trial-wide IDs are the primary improvement
targets. Local segment coverage is a supporting diagnostic, not a substitute.
Freeze the acceptance rule before evaluating a
candidate, and report the evidence and unreviewed scope instead of inventing a
percentage target.

For every session, record:

- Total post-preparation duration; reconstructed fish-time; duration passing the
  declared position and local-continuity criteria; and duration linked to a
  supported whole-session identity. Keep these quantities separate.
- Usable segment duration distribution, median and longest segment, plus joint
  coverage for all three fish. Report per-identity coverage only where identities
  are supported; otherwise report segment-level quantities explicitly.
- Missing/withheld time by reason, unresolved joins, and processing completeness.
  Use all expected fish-time in the window as the denominator for pooled
  three-fish coverage, not just detected or successfully processed rows.
- Independently reviewed correct/wrong/undecidable identity links with their
  denominators, female and male coverage, and the sampling protocol. Review both
  accepted links and flagged boundaries; short difficult clips alone do not
  validate a whole recording. Known scenes remain regression data.
- Where exclusions occur in time, especially at encounters. Collisions and
  courtship may be disproportionately excluded; collaborators need the missingness
  information when interpreting behavior in retained intervals.

No observed swaps on a small reviewed sample is not a zero-swap guarantee.

## First deliverable, then expansion

1. Inventory the dataset read-only: recording IDs, durations, timing, views,
   SLEAP model/node provenance and output completeness, calibration, and existing
   derived tracks. Treat archived corpus counts as historical until checked.
2. Use the already studied K_1_2 recording as the first engineering pilot over
   its complete post-preparation window. Select and record one runnable chain
   using the existing work; assess the SLEAP adapter through 3D reconstruction
   before adopting it. Resolve source-row/frame-offset defects before integration.
   Preserve prior outputs and identify incomplete/provisional runs explicitly.
3. Produce a versioned pilot package: 3D node trajectories indexed by supported
   trial-wide IDs and a documented body
   reference point, frame/time mappings, observation/quality masks, segment and
   identity-link tables, exclusions/review reasons, and a coverage summary. Include
   a 3D trajectory plot and synchronized source-video review material that visibly
   breaks at gaps or unresolved identity boundaries.
4. Include a data dictionary (units, axes, node meanings, ID scope and missing
   values), machine-readable provenance (input/model/calibration/code hashes,
   configuration and completion state), and a small reproducible loading example.
   Preparing a handoff package does not itself send data to collaborators.
5. Check different groups/dates using the frozen rules before dataset-wide
   execution. Process all eligible sessions or record a specific failure/exclusion
   for each; aggregate without treating repeated days of the same fish as
   independent animals.

The pilot is complete when its full analysis window is accounted for, the package
can be loaded and plotted, and all accepted segments have documented support for
consistent trial-wide IDs, with evidence and limitations inspectable. A segmented
export whose relationships are all unresolved is an intermediate artifact, not
completion of the requested identity task. The pilot need not have 100% usable
frames. Algorithm changes must address a measured
delivery blocker or improve validated usable time/continuity; another isolated
parameter comparison is not the default next task.

## Existing implementation and limits

- [SLEAP adapter result](../../labs/20260918_sleap_pipeline_adapter/RESULT.md):
  provisional histories, committed adjacent fragments, and separate gap
  candidates are implemented. The adapter is parked; input compatibility with
  C25 does not establish 3D identity accuracy. Nonzero-start/global-row mapping
  requires care.
- [Clean-fragment result](../../labs/20260917_clean_tracklets/RESULT.md):
  two-view-supported joins and preserved source observations are reusable, but
  substantial missing/ID-review time remains. Fragment counts are not ID accuracy.
- [Long continuation usage](../../labs/20260909_three_body/LONG_CONTINUATION_USAGE.md):
  an existing K_1_2 plan already targets `[49800,348600)` with resumable state.
  That document reports a completed first 10-second run, not a finished 30-minute
  result. It is a separate execution option, not evidence that adapter integration
  already works. A candidate prefix may change before the full sequence finishes.
- [Tracklet ledger](../../labs/20260909_three_body/tracklet_ledger.py) and
  [identity output](../../labs/20260909_three_body/identity_output.py) preserve
  uncertainty and source geometry. They are reusable patterns, not an existing
  dataset-wide certified export.

On 2026-09-30, the configured Windows `Z:` dataset path was not accessible in this
session, and its documented UNC equivalent returned access denied. Thus this
planning pass did not inventory the live dataset or run tracking. Storage access
must be established for the inventory; this does not establish that the dataset
is missing.
