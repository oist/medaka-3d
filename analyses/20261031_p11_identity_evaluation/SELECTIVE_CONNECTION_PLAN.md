# Selective geometry/motion connections: registration draft

Date: 2026-11-04; revised 2026-11-05. Owner: research supervisor; preparation by
Codex (2026-11-04) and Claude Code (2026-11-05 revision).
Status: **draft, not frozen, no new extraction or model run authorized by this
document alone**. The associated `selective_connection_protocol.draft.toml`
deliberately disables execution until the pending choices and source manifests
are filled. It does not mark files as sealed by access control or claim that a
holdout has never previously been seen.

Question: can geometry/motion evidence safely automate a useful fraction of
connections between locally consistent segments, while sending unsupported
connections to human review? Appearance is outside scope.

## Revision of 2026-11-05: limited manual effort

The supervisor cannot supply hundreds of new manual identity corrections. The
design is reorganized accordingly, with the supervisor's approval:

1. **Development needs no new manual labelling.** The six existing ID-pass
   reviews are the development reference. Reference conflicts become unknown
   instead of requiring new adjudication.
2. **Nested leave-one-recording-out cross-validation replaces the earlier
   four-recording fit / two-recording calibration split**, so out-of-fold
   decisions from all six recordings are evaluated instead of two.
3. **The first holdout is K_2 only.** P_2 stays reserved as a conditional second
   stage, decided before any K_2 reference exists (Phase 5).
4. **The blinded raw-image audit is limited to about 20 decisions.** It detects
   gross reference failure; it cannot certify a reference error rate below 1%.

The ~300 figure from planning is the number of *reference-labelled automatic
passes* needed by an IID illustration (see workload), not a number of manual
corrections. Whole-recording ID-pass labels every decision in a recording at
once; the reviewer corrects only the reviewed tracker's errors, 5--34 per
recording in the existing six.

## What is known, and what remains to verify

`medaka3d.cohort` identifies K as one trio and P as another, across trials and
days. Thus six recordings are repeated measurements of two trios, not six
independent biological replicates. K_2 assesses new recordings of the same
individuals as K_1. Possible C/P male reuse must be resolved before treating C
as a new-individual holdout.

The inventory below was **read directly from the configured bucket on
2026-11-04** during a Claude Code session (directory listings, ID-pass summaries,
session logs and inference logs). File hashes are not yet pinned; that is a
Phase 0 item. No holdout ID-pass, extraction output or video content was opened.

| Key | Folder | ID-pass corrections | ID-pass GUI open time | Detector run | Development status |
| --- | --- | ---: | ---: | --- | --- |
| K_1_1 | `20251225_K_1_1_FCMC` | 12 | 2.50 h | s16 (S14 weights) | Prior extraction; ID labels unused |
| K_1_2 | `20251225_K_1_2_FCMC` | 18 | 1.15 h | s16 (S14 weights) | Used for chain development |
| K_1_3 | `20251225_K_1_3_FCMC` | 34 | 1.56 h | s16 (S14 weights) | Prior extraction; ID labels unused |
| P_1_1 | `20260112_P_1_1_FCMC` | 13 | 2.23 h | S14, cap 6 | Used for warning-rule development; also has the Jingyi reference |
| P_1_2 | `20260113_P_1_2_FCMC` | 18 | 1.11 h | S14, cap 6 | Chain not yet run |
| P_1_3 | `20260112_P_1_3_FCMC` | 5 | 1.08 h | S14, cap 6 | Chain not yet run |

- ID-pass files are under the data parent's
  `review_packages/<folder>/<package>/id_pass/`. All six were reviewed on the
  same host on 2026-10-30--11-01 at up to 2x playback speed; the reviewer is not
  recorded. Correction counts concern the reviewed `m3dfull` tracker, not C34.
- `--max_instances 6` is confirmed in all three per-view inference logs of P_1_2
  and P_1_3, and in the P_1_1 top-camera log. The K cap of 8 is documented in
  `docs/TRACKING_PIPELINE.md`; `tools/s16_saion_inference_bu.py` sets no explicit
  cap, so verify it from the cached outputs. Both runs name the S14 bottom-up
  weights from `retrain_20260810_3node_reviewed`; verify by hash.
- `experiment_list/exp_list.csv` lists `20260112_P_1_2_FCMC` and
  `20260113_P_1_3_FCMC`, whose set-dates disagree with the actual folders above.
  Pin actual folder names in manifests.
- The K_2 folders (`20260109_K_2_{1,2,3}_FCMC`) have review packages but no
  `id_pass` directory. Their detector run is listed as
  `s14_bottomup_3node_reviewed` (like P), not K_1's s16 run, so K_1-to-K_2 also
  changes detector stratum. Verify the cap and hashes before freezing.

Common review dates/host do not establish independent reviewers or correctness.
An unused label file does not erase previous algorithm development on the same
recording. K/P transfer is also confounded with detector cap/version; recording
the detector stratum does not separate this effect from fish differences.

## User decisions recorded

- **2026-11-04:** `epsilon = 0.01` per automatically accepted connection, a
  development goal, not an achieved error rate or an external guarantee.
- **2026-11-05:** the limited-effort reorganization above. Additional new-trio
  holdout groups remain undecided and are outside the current workload; without
  them, this study makes no claim about unseen trios. The K_2 ID-pass reviewer
  and the audit judge remain undecided.

## Pending choices required for freezing

1. **Risk unit.** Proposed: one fully automatic clean-end mapping per warning,
   wrong if any asserted correspondence is wrong (see below). Needs confirmation.
2. **Threshold-selection rule.** Proposed: inside each training set, the largest
   automatic coverage whose one-sided 95% Clopper--Pearson upper bound on
   inner out-of-fold wrong-among-accepted known mappings is at most epsilon;
   accept nothing when no threshold qualifies (Phase 4). Needs confirmation.
3. Holdout manifests: exact K_2 and reserve P_2 folder names, documented fish
   reuse, prior exposure, custodian, and a read/access log.
4. Named K_2 ID-pass reviewer and blinded audit judge.
5. The merged extraction commit, immutable input paths, detector-weight and
   cache hashes, ID-pass and source-tracker hashes, calibration, native timing,
   and schema. Current HEAD is `fd8d86f81715239b39ef8496299768b43d0679c4`; the
   row repair is still a working-tree change, not a merged row-repair revision.
6. Holdout reporting method for the K_2 fixed collection (proposed default in
   Phase 5) and the development go/no-go rule (proposed in Phase 4).

The new algorithm lab is planned as `labs/20261104_selective_geometry_connections/`.
Create its frozen README/configuration and Linker adapter before execution;
put shared scoring in `medaka3d.eval`. This draft remains with the existing
analysis so an unimplemented proposal is not mistaken for a runnable lab.

## Evaluation unit and output contract

Keep four distinct objects: warning interval, proposed clean-end correspondence,
fish-level edges, and reference-derived identity-change brackets. A warning is
a **decision/review unit**, not necessarily one physical crossing or one swap.
The existing 32.432 s P_1_1 warning contains 12 sustained reference groups.

The proposed primary risk unit is a **fully automatic clean-end mapping** for
the relevant visible animals. A mapping is wrong if any asserted correspondence
is wrong. A partly resolved mapping still needs human review and does not count
as a fully automatic pass. Report individual edges separately, preserving their
shared warning/recording/trio IDs; do not count M1 and M2 as independent trials.

A correct end-to-end mapping does not prove correct identity inside the warning.
Until separately supported, keep warning-interior positions but withhold their
identity labels. An accepted bridge can connect the supported segments on either
side. Unaffected fish can retain independently supported continuity, reported
separately. On an unresolved bridge, stop propagation of affected trial identities;
do not silently carry the previous name into the next segment.

Report original full-chain errors beside these supported-output masks so apparent
improvement cannot be achieved merely by withholding data. Missing, withheld,
unprocessed, and unknown reference time remain in the full post-300 s denominator.

## Phase 0: source and exposure freeze

- Freeze the row-repaired chain after the requested merge and pin exact inputs.
  Primary lambda is the inherited 0.767; it is not a newly blind choice, since
  its previous reference results are known. Use the other six arms only for
  sensitivity analysis with the same frozen decision procedure.
- Freeze detector differences as found; verify actual weight hashes, node
  ordering, count cap, and truncation policy rather than relying on model names.
- Record native time origins and synchronization before any seconds conversion.
  Exclude each recording's first 300 s from scientific coverage.
- Reserve K_2 (stage 1) and P_2 (conditional stage 2) using cohort metadata, not
  observed tracking quality. A separate custodian can record checksums without
  exposing content to developers. Do not claim a machine-enforced seal until
  access isolation actually exists.
- Fix model variants, endpoints, normalization, label support, sample gates,
  folds, threshold rule, audit sampling, go/no-go and holdout reporting before
  running.

## Phase 1: reference adapters and quality (no new manual work)

There are five additional primary recording references, **plus the P_1_1 ID-pass
adapter needed to compare with Jingyi**: six ID-pass sources in total. Preserve
both P_1_1 sources separately, with the Jingyi reference remaining the inherited
primary comparison. Do not silently replace it by a consensus.

Pin exact files beneath `review_packages/<folder>/.../id_pass/`; never choose a
directory merely because it is latest. Before replaying swaps, establish their
source tracker/version, initial identity convention, index base, native frame
grid, interval endpoint convention, and operation ordering. Preserve raw logs,
reference-support flags, original source positions, and missingness. Do not
invent watched/analysed/node-quality fields merely to imitate a CSV schema.
Unknown reviewer or missing quality fields stay explicitly unknown.

Replay corrections onto the pinned source tracker, then spatially match C34
positions to the resulting reference using the inherited 5 mm distance / 2 mm
mutual margin rules. Report the existing 2/5/10 mm sensitivity without choosing
the best score. Danger/unusable intervals have unknown identity support; positions
and time are retained. Curate versioned references with source hashes and
`medaka3d.provenance`; never overwrite raw human annotations.

Compare P_1_1's two reviews, and compare K_1_2 female correspondence against the
exact historical evidence set. The reported 298 scored intervals and the larger
source CSV are not interchangeable: pin which rows and timing were used. Distinguish
original human clicks from tracker-linked/snapped estimates. **A conflict is
recorded and its decisions become unknown**: excluded from supervised losses and
kept in the coverage tables. Video adjudication of conflicts is optional, not a
prerequisite; if done later, it creates a separately versioned reference.

These comparisons are computational. Report the conflict counts to the
supervisor, who decides whether the references are adequate for development
before reference outcomes are attached to the Phase 2 ledger. Agreement between
two tracker-assisted references is not independent truth.

## Phase 2: label-blind candidate and feature export

Run the pinned chain for K_1_1--3 and P_1_2--3; reuse P_1_1 only if its code/input
manifest is identical to the chosen source revision or compare explicitly first.
The earlier ~85-minute P_1_1 run is a single historical measurement, not a scheduling
guarantee for five different inputs or a reason to change detector settings.

Keep the core detector fixed: <10 mm proximity, >10 mm adjacent steps,
non-identity nearest assignment, or missing position, with 0.05 s padding and
overlap union. Do not split a long warning merely to reduce displayed clip length.
Export all geometry-derived warnings and ordinary segment connections before
attaching reference outcomes. Warning misses inside a continuous track must also
be visible when the reference is evaluated later.

For each clean-end pair, compare all six complete mappings on an equal footing;
missing/unmatched animals allow an unresolved alternative. Current C34 names and
the identity permutation receive no privileged score. Candidate ties abstain.
Do not include timestamps, physical names, reference flags, or lambda labels as
predictors.

Proposed feature vector (four families, eight scalars):

| Family | Scalars |
| --- | --- |
| Bidirectional motion fit | Forward and backward normalized prediction error; agreement of the two winning mappings |
| Competition | Best-to-second-best cost margin, normalized consistently with prediction error |
| Uncertainty | Maximum endpoint predictive uncertainty |
| Missingness/support | Gap duration; pre-context support duration; post-context support duration |

Fit endpoint motion on native seconds without crossing local segment boundaries
or gaps. Both directions share data and count as correlated features, not independent
probabilities. Learn any global uncertainty scale or feature standardization from
training folds only. Deterministic local feature computation may use the event's
own unlabeled pre/post observations; it may not use that recording's reference
to tune the scale. Check arbitrary slot-renaming invariance, time-direction
consistency, gap boundaries, exact source-row preservation, and abstention.

Two proposed model variants differ only in motion context (0.1 s versus 0.3 s).
Use the same eight features and L2 logistic model. Exact covariance estimator,
minimum endpoint support, penalty normalization, and out-of-range handling must
be specified in the frozen implementation; they are not yet fixed by this draft.

## Phase 3: attach reference outcomes and apply the sample gate

The proposed candidate is correct/wrong/unknown according to supported reference
correspondence at both contexts. Use context support, not a single potentially
missing endpoint. Do not require purity of an entire local track; assess the
specific supported pre/post pieces. Preserve all reference conflicts and unknown
targets in the coverage tables, outside supervised losses.

The execution gate requiring error counts belongs **after label attachment**, not
in the label-blind phase. Proposed engineering floor for **every outer and inner
training set**: at least 20 known-correct and 20 known-wrong decisions, with each
outcome present in at least two recordings. This floor is a conservative design
choice for this small model, not a theorem of sufficiency or an independence
assumption. If a training set fails, report insufficiency and fixed-rule
risk/coverage descriptively; do not borrow held-out labels, pool adjacent frames,
or treat lambda copies as new examples. Freeze the numerical gate before seeing
these counts.

## Phase 4: nested cross-validation and the frozen procedure

The procedure below is applied identically at every level. Fit only on the
primary lambda; the six other arms are scored with the same frozen procedure for
sensitivity. Recording-level folds keep every fish, candidate and lambda copy of
a scene together. Normalization, standardization, variant selection and
threshold selection use only the current training set.

**Procedure on a training set T of recordings:**

1. Inner loop: for each recording s in T, fit both variants on T without s and
   score s. This gives inner out-of-fold scores for every decision in T.
2. For each variant, choose the threshold by the proposed rule: maximum automatic
   coverage such that the one-sided 95% Clopper--Pearson upper bound on
   wrong-among-accepted known mappings is at most epsilon. This is an
   IID engineering selection rule, deliberately conservative; it is not the
   study's inferential claim. If no threshold qualifies, the variant accepts
   nothing.
3. Choose the variant with larger qualifying coverage; a tie selects 0.1 s.
4. Refit the chosen variant on all of T and attach its threshold.

**Outer loop (internal estimate):** for each of the six recordings r, run the
procedure on the other five and apply the result to r. Pool the six outer
results, and report them per recording and per trio as well. This estimates the
procedure at five-recording training size on these two trios; the six outer
folds are not six independent biological tests.

**Transfer (descriptive):** run the procedure on K_1 alone (three inner folds)
and apply it to P_1, then the reverse. The destination trio contributes nothing
to fitting, normalization, variant or threshold. Detector cap/version remains a
confound, not an isolated fish-transfer effect.

**Frozen model:** run the same procedure on all six recordings (inner loop over
six). This is the model, variant and threshold frozen for Phase 5. The outer
estimate describes the procedure, not this exact six-recording model; that model
is assessed directly only in Phase 5.

Report automatically passed mapping count and wrong count, human-review count,
partial decisions, unknown reference, warning misses, reference coverage, usable
fish-time, and propagated wrong-ID time. For reviewed bridges, run two declared
scenarios: unresolved support stops naming; supported reference resolves the
bridge. The latter is reference-assisted and not automatic performance. Do not
interpret a logistic output as a certified per-event probability. Do not loosen
epsilon after viewing performance.

**Proposed go/no-go before opening K_2:** proceed only if (i) the pooled outer
automatic passes have empirical wrong-among-accepted at most epsilon, with the
achieved IID-illustrative upper bound reported beside it, and (ii) in every
recording, propagated wrong-ID fish-time in the unresolved-stops-naming scenario
does not exceed the original chain's wrong-ID fish-time. Otherwise report the
development result and keep K_2 unopened for a revised procedure.

## Phase 5: K_2 holdout, conditional P_2 extension, and the limit of the claim

Freeze the complete model, feature code, scales, threshold, candidate detector,
source inputs, cohort manifest, audit sampling and reporting method before
opening holdout predictions or new reviews.

1. Run the frozen chain and rule on `20260109_K_2_{1,2,3}_FCMC`. Count automatic
   passes; this count is label-blind.
2. **Before any K_2 ID-pass exists**, the supervisor records whether to add P_2
   as stage 2, using only this label-blind count. Under the IID illustration,
   fewer than 299 automatic passes cannot demonstrate 1% even with zero errors.
   Never decide the extension from errors or reference outcomes.
3. K_2 ID-pass is made on the base tracker output without risk scores, accepted
   mappings, or referral status shown to the reviewer.
4. Blinded raw-image audit of **20 decisions**, proposed as 10 automatic passes,
   5 human referrals and 5 warning-negative intervals, uniform within stratum
   with a seed fixed at freeze. Use [the existing A/B/C evidence procedure](../../labs/20260909_three_body/IDENTITY_EVIDENCE.md),
   hiding tracker identities, scores and stratum from the judge. Every
   disagreement with the ID-pass is reported individually and investigated.
   Twenty decisions detect gross reference failure only; they cannot establish
   reference error below epsilon or the warning miss rate. Anonymous letters do
   not establish physical M1/M2 names. Do not replace ambiguous cases by easy ones.
5. If stage 2 was chosen, P_2 follows the same steps and is reported as a
   separate stage with no change to the frozen procedure.

**Proposed reporting for the K_2 fixed collection:** per-recording counts of
automatic passes, known wrong, unknown reference, referrals, warning misses and
propagated wrong-ID time; the pooled exact one-sided 95% bound labelled as an
IID illustration; and a sensitivity bound counting unknown automatic passes as
wrong. Claim "consistent with at most 1%" only if that bound is at most epsilon;
otherwise report the achieved bound and "1% not established".

The estimand is the fixed K_2 collection: new recordings of trio K, which also
appears in development. K_2 and P_2 cannot increase the number of unseen trios,
so no bound here transfers to a future independent trio. A failed holdout is not
reused for tuning; a revised procedure needs a new assessment set.

## Workload and arithmetic

- **Development: no new manual annotation.** The existing six ID-passes took
  9.6 GUI-open hours in total (1.08--2.50 h per recording) and contain 100
  tracker corrections.
- **Holdout stage 1:** ID-pass of three K_2 recordings. From the existing
  open times, roughly 3.2--7.5 h; open time is not a measured effort for K_2.
- **Audit:** 20 decisions; time unmeasured.
- **Conditional stage 2:** P_2, of similar size to K_2.

Exact one-sided 95% Clopper--Pearson requirements, under an **IID illustration
only**:

| Target | 0 errors | 1 error | 2 errors |
| --- | ---: | ---: | ---: |
| 1% | 299 | 473 | 628 |
| 2% | 149 | 236 | 313 |
| 3% | 99 | 157 | 208 |

These are counts of reference-labelled automatic passes, not manual corrections
or recording requirements for the present correlated data. P_1_1 had 129 core
warnings at lambda 0.767; per-recording warning counts and the automatic-pass
fraction for other recordings are unmeasured until Phases 2--4.

No extraction, reference conversion, model fitting, held-out inspection, or
source-video adjudication was performed for this draft. No new lab, notebook,
Git commit, push, merge, or access change was created.

## Methodology references

- [Cawley and Talbot, model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html):
  fitting and selection must be included inside the evaluation boundary.
- [Lee, Barber, and Willett, hierarchical inference](https://arxiv.org/abs/2306.06342):
  repeated/grouped data require appropriate hierarchical assumptions; this paper
  is background, not a claim that its coverage theorem controls this selector's
  accepted-connection error rate.
- [NIST exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm):
  binomial bounds do not make dependent within-recording decisions IID.
