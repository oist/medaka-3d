# Roadmap — ordered phases with gates

> **2026-09-30 current priority:** Follow the
> [dataset delivery plan](decisions/2026-09-30-dataset-delivery.md) for next steps.
> Its input inventory, complete pilot handoff, and subsequent expansion supersede
> the phase ordering below. Consistent IDs throughout each trial are required; 100% frame
> coverage is not required. All coverage excludes each recording's first 300 s.

> **Status note, 2026-09-08.** The Phase C text below predates the tracking work of
> 2026-08-13 → 2026-09-03. The chain that exists, its numbers and its open failure classes
> are in [TRACKING_PIPELINE.md](TRACKING_PIPELINE.md); the entry point is
> [NEXT_SESSION.md](NEXT_SESSION.md). This file is kept for the phase structure and the
> gates, which still hold; nothing here has been promoted.


Working order for `medaka-3d`. Rationale lives in
[DEVELOPMENT_STRATEGY.md](DEVELOPMENT_STRATEGY.md); this is the checklist.

**Rules of the road**

- Phases are ordered by dependency. Do not start a later phase because an earlier
  one is dull — the whole reset exists because the retired repo ran experiments it
  could not conclude.
- Every phase has a **gate**. If the gate is not met, stop and report rather than
  proceeding on an assumption.
- Anything speculative goes in `labs/` with a decision rule written *before* it
  runs. Production code lives in `src/medaka3d/`.
- Enumerate, don't extrapolate. Two claims in the inherited docs were wrong
  because someone generalised from one sample.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Orientation and environment

- [x] **0.1** Read `CLAUDE.md`, then `docs/HANDOFF.md`, then the **Corrections**
      section at the top of `docs/PRIOR_FINDINGS.md`. The corrections overturn two
      claims repeated later in that same file.
- [x] **0.2** Environment: `uv pip install -e ".[dev]"`, then `pytest -q` and
      `ruff check .`. *(2026-08-02: 93 passed, 1 skipped; ruff clean.)*
- [x] **0.3** Verify the inherited assets resolve and are intact:

      ```python
      from medaka3d import load_paths, load_catalog
      p = load_paths()
      c = load_catalog(p.asset_root)
      print(len(c), c.meta["production_models"])
      print(c.verify_all())          # must be []
      ```

**Gate 0:** tests green, `verify_all()` returns `[]`. If `Z:` is not mounted,
stop — nothing downstream works without it.

---

## Phase 0.5 — The cheap question, asked before months are spent

- [x] **0.5.1** Test whether the study's conclusion is reachable **without
      tracking at all**. `PRIOR_FINDINGS.md` §1.7 suggests the lab sheet's
      `female_accept` field and the BORIS courtship asymmetry may agree in every
      experiment where both exist — an N=7 result from two independent human
      record sources, no ML. It is flagged there as an *unverified derivation*.
- [x] **0.5.2** If it holds: verify annotator independence (were the BORIS coder
      and the sheet's author the same person, or was one derived from the other?).

**Gate 0.5: ANSWERED 2026-08-02 — NO.** The conclusion is *not* reachable without
tracking. The two "independent" record sources are one annotator pass: the sheets
carry identical counts (7/7), `female_accept` is reproduced from the wrap events in
the same BORIS file (8/9), and 3 of 26 files were coded on the recording day itself.
The independent channel — the physical egg record — is per-experiment yes/no and
cannot discriminate between males. The unit is the group, so this is **n = 3 trios
(E, K, O)**, not 7, 8 or 9; and the one behaviourally interesting row (E day 3,
where the female switched males) flips depending on whether courtship is counted
by event or by duration — 7.7 s vs 7.4 s. Proceed to B0.
See [`decisions/2026-08-02-phase05-accept-vs-courtship.md`](decisions/2026-08-02-phase05-accept-vs-courtship.md);
reproduce with `tools/phase05_accept_vs_courtship.py`.

---

## Phase B0 — Fix SLEAP at crossings *(first real engineering work)*

Reported by the experimenter: round 8 degrades exactly at crossings — one tail
joined to two mouths, unstable detection in X-crossings. A fish that is never
detected cannot be recovered by any downstream cleverness, so this comes first.
**No new annotation is needed for B0.1–B0.3.**

- [x] **B0.1 Diagnose the training data.** Count labelled frames in
      `assets/.../round8 combined_labels.pkg.slp` containing two instances closer
      than 100 px. `sleap-io==0.9.2` and `sleap-nn==0.3.1` are pinned together.
      **DONE 2026-08-02 — the inherited “never seen” claim is rejected.** Primary
      definition is **complete instances only** (both nodes labelled): **2,107 of
      3,974 eligible frames = 53.02%** below 100 px; 172 of 4,146 frames are
      ineligible for having fewer than two complete instances. Sensitivities, which
      answer different questions and are **not** interchangeable with the primary:
      all-visible-centroid 2,603/4,146 (62.78%), minimum visible-node 3,325/4,146
      (80.20%). ~9% of instances are missing a node, which is why the definition
      matters. §2.10's "0% closer than 100 px" measured the Detectron2 crop set and
      3D masks — both gated to exclude close contact — and never covered the SLEAP
      labels. Rejecting "never seen" does **not** establish that coverage is
      adequate for the failure regime. Re-confirmed on the latest labels
      (`round8.top.train` + **round9b** side, 4,500 frames). 0 training frames share
      a node; predictions do. See
      [`decisions/2026-08-02-b01-crossings-in-round8-training.md`](decisions/2026-08-02-b01-crossings-in-round8-training.md).
- [x] **B0.2 Build a crossing-frame evaluation set.** Retained round-7 `top_cam`
      predictions for all 65 experiments are at `paths.sleap_output_dir`. Link
      them roughly, then use `medaka3d.eval.find_encounters` to locate crossings.
      Report the count and the annotation-minute estimate from `summarise()`.
      **DONE 2026-08-02 — full census of all 65 experiments with retained
      predictions, reported by group (`n = 12 groups`):**
      **14,684 link-free encounters = 367.1 h** of annotation; 30,558 counting all
      three pairs = 764.0 h. Coverage gap: the recording tree has 71 experiments in
      14 groups, so **groups B and C have no retained predictions at all**, along
      with `20260115_N_2_3` and the two `_P2` continuations.
      Limits that stand: counts are **top-view only** (depth collapsed → upper
      bound on true 3D encounters), and the mm/px figure is an **operational
      screening proxy**, not calibration. Detection failures across the corpus:
      <3 instances in 3.2–18.2% of frames (median 5.7%), >3 in 1.0–9.9%
      (median 3.0%).
      *The earlier 9-experiment projection (~270–425 h) did bracket 367.1 h, but
      understated per-experiment variability several-fold — the scale spread is
      **19.7%** across 65, not the 4% seen in E/K/O, and group G alone spans
      0.1864–0.2131 within one group.* See
      [`decisions/2026-08-02-b02-crossing-eval-set.md`](decisions/2026-08-02-b02-crossing-eval-set.md).
- [x] **B0.3 Re-score existing checkpoints on crossing frames only.** Round 7
      trained both top-down and a bottom-up baseline; the production choice was
      made over *all* frames, which are overwhelmingly separated fish. A model can
      win on average and lose exactly where it matters. Both checkpoints are on
      `/bucket` — this is inference only, no training.
      **DONE 2026-08-02 on frames held out from BOTH checkpoints** — 113 top and
      113 side frames whose canonical `(leaf video, frame_idx)` key is absent from
      the round-7 packages, criterion fixed in code before the run, primary =
      both-node instance recall@20px after Hungarian matching. All frames are
      `20251225_K_1_2_FCMC`: **n_groups = 1**.

      **Governing finding: the round-8 labels are not independent of top-down.**
      47.8% (top) / 48.4% (side) of GT instances are byte-identical to a round-7
      top-down prediction; **0.0%** match bottom-up. Results are therefore
      bracketed by two arms — ALL (favours top-down) and CORRECTED (favours
      bottom-up). Crossing recall, bottom-up − top-down:

      | view | ALL arm | CORRECTED arm |
      |---|---|---|
      | top | 73.33 → **80.00** = **+6.67** | 56.72 → **72.39** = **+15.67** |
      | side | 77.08 → 77.08 = +0.00 | 61.47 → 64.22 = +2.75 |

      Top clears the frozen +5.0 margin at **both** ends; side clears neither.
      **Shared-node predictions — the reported "one tail joined to two mouths" —
      are top-down 5 (top) / 3 (side) and bottom-up 0 / 0.** Precision at
      crossings: top-down ~73%, bottom-up ~92%. See
      [`decisions/2026-08-02-gate-b0.md`](decisions/2026-08-02-gate-b0.md).
- [x] **B0.4 Geometric plausibility filter.** ~~Body length is physical and bounded
      (~25–35 mm) and calibration makes it measurable.~~ Reject instances sharing a
      node, or with implausible 3D length. Cheap post-process; catches the
      reported failure directly.
      **DONE 2026-08-02 — ADOPT for top-down.** Two rules, both derived from human
      labels: drop instances sharing a node (< 1 px; human labels never do this),
      and drop spans outside the 0.5–99.5 percentile of the labelled distribution,
      **[35.1, 152.4] px** top and **[15.9, 142.4] px** side. Bounds are in
      **pixels, not mm** — the ~25–35 mm body length is the wrong reference because
      the `tail` node sits at the tail base, and B0.2's mm/px is an operational
      proxy, not calibration. Effect at crossings: **every shared-node prediction
      removed (5→0 top, 3→0 side), precision +0.8 to +2.2 points, recall −0 to −1.0
      point.** No 3D length test — round-7 retained only `top_cam`, so 3D is not
      available. See [`decisions/2026-08-02-gate-b0.md`](decisions/2026-08-02-gate-b0.md).
- [ ] **B0.5 Retrain with crossing frames** *(needs annotation — coordinate with
      Phase A.2 so one pass serves both)*.
      **DO NOT RUN AS SPECIFIED — the premise is false.** B0.1 shows the training
      set is already 53.02% crossing frames under the primary definition, so it is
      not short of them; and B0.3 shows the failure is a decoder/grouping behaviour
      (shared nodes appear in top-down, never in bottom-up on the same frames, and
      never in the human labels), not a missing-examples problem. Also blocked on
      annotation that does not exist. If a retrain is wanted anyway, the unexploited
      asset is **round9b** — 3,118 side frames, all with three complete instances,
      **no trained model anywhere on `Z:`** — but expect less than its size suggests.

- [x] **B0.6 Replace suppression with assignment** *(2026-08-03, added after
      `k01` showed suppression destroys evidence)*. B0.4's shared-node rule
      raises precision, but in **15 of 47** false-positive frames only 2 of the 3
      animals are covered, so deleting the shared-node loser discards the only
      detection carrying a third animal. Reframed as *assign tails to mouths
      one-to-one*, with abstention allowed.
      **DONE — the first specification was REJECTED on its own pre-registered
      rule** (coverage 88.79 → 78.47%, recall 74.93 → 60.77%). Cause, measured:
      a span-deviation cost is a *plausibility* signal and not a *discriminative*
      one — 45.3% of foreign mouth–tail pairings sit inside the acceptance band,
      so reassignment ran **29 correct→wrong against 2 the other way**. Also one
      threshold error of my own: 68.7 px is the ground-truth minimum edge distance
      for a *centroid* and was applied to *mouths*, whose true minimum is 28.8 px
      (worth 3.5 of the 10.3 lost points).
      **A corrected, origin-preferred cost is PARKED**: shared tails **28 → 0**
      structurally, false positives **52 → 41** (better than B0.4's 43), coverage
      −0.59 and both-node recall −2.95 — the last two miss the pre-registered
      tolerance. It recovers exactly **9** tails by reassignment, the ceiling k01
      predicted. Blocker: both-node recall scores an honest mouth-only abstention
      and a silently swapped tail identically, and the whole argument is that
      they differ downstream. **That needs Gate A.**
      See [`decisions/2026-08-03-k-detection-quality.md`](decisions/2026-08-03-k-detection-quality.md)
      and [`labs/20260803_mouth_anchored_assignment`](../labs/20260803_mouth_anchored_assignment/).
- [x] **B0.7 Size the annotation that is actually worth doing** *(2026-08-03)*.
      Not more crossing frames (B0.5's premise, already refuted). **464 partial
      instances** — 221 in `labels.round7.top`, 243 in `labels.round8.top.train`,
      5.7–5.9% of instances — across 426 frames, **≈3.9 h** at 0.5 min each
      against the 367.1 h a full encounter pass would take. 334 of the 464 sit
      within 100 px of a complete instance. The model reproduces this gap as
      **36.5%** of its held-out false positives. Work list at
      `work_dir/k06_partial_instances/`; regenerate with
      `tools/k06_partial_instance_worklist.py`. **The annotation itself is not
      done and is a human judgement.**

- [x] **B0.8 Capacity retrain of the top centered-instance head** *(2026-08-03,
      approved; supersedes the dropped B0.5)*. `max_stride` 64 → 32,
      `filters` 24 → 32, intensity augmentation on. Trained on the round-7
      package so the evaluation frames stay held out; centroid head unchanged.
      **DONE — pre-registered rule PASSED, verdict PARKED.** Mouth-only 59 → 45,
      both-node recall 74.93 → 78.17%, raw shared tails 28 → 11, and overfitting
      *narrowed* (in/out FP ratio 14.1× → 11.4×) on **44% of round-7's
      parameters** — 31.4 M vs 70.8 M, so it is a reallocation, not a bigger net.
      `val/loss` finished better and was flat for 80 epochs while these moved;
      it decided nothing, as pre-registered.
      **Not promoted: the gain does not reproduce on the locked benchmark** (see
      B0.9), where k04 is not better than round 7. The 113 frames are the subset a
      human chose to correct, so they are enriched for difficulty. See
      [`labs/20260803_centered_instance_capacity`](../labs/20260803_centered_instance_capacity/).
- [x] **B0.9 Two inherited claims corrected, both by looking rather than assuming**
      *(2026-08-03, prompted by the experimenter)*.
      1. **Round 8 is production for BOTH views, not "round 7 top / round 8 side".**
         The split was a hardcoded constant in `tools/build_asset_manifest.py` that
         `MANIFEST.json` echoed back. Round 8 trained 07-23; the corpus predictions
         were produced 07-07→07-13 and their provenance names round 7, so they are
         round 7 by chronology. The archive's own 2026-07-27 report states
         *"Round 8 remains the production model for both top and side views."*
      2. **A locked human benchmark already exists** (see Gate B0 limitation 1),
         so "no unbiased evaluation set" was false.
      Consequences: **93 of the 113 "held-out" frames are inside round-8's
      training data**; k01's 0.46 FP/frame is a round-7, worst-case number
      (round 7 scores 0.125 FP/frame on the locked benchmark); and the reported
      shared-tail failure is largely round-7's — round 8 emits 1 across 40
      frame-disjoint frames. **No model separation is established**: on the only
      leakage-free arm — 16 locked-benchmark frames containing no verbatim
      round-8 instance — round 7, round 8 and k04 all score **97.92% coverage**
      and differ by one instance on recall. Round 8's apparent lead comes from
      having trained on 93 of the 113 frames and from being 30% of the locked
      benchmark's ground truth. An earlier draft of this entry claimed "labels
      beat architecture"; that is withdrawn as unsupported.

**Gate B0:** a detection-quality number at crossings versus overall, for both
architectures. That number decides retrain-vs-switch-architecture. Do not retrain
before it exists.

**Gate B0 status 2026-08-02 — MET for group K (`n_groups = 1`).**

Full decision: [`decisions/2026-08-02-gate-b0.md`](decisions/2026-08-02-gate-b0.md).

- **Switch architecture: NO**, under the frozen rule. Top clears the +5.0 margin
  at both ends of the bracket; side clears neither; rule 2 says views disagree →
  retain top-down and report per view. **The rule was not relaxed after seeing the
  result** — bottom-up wins decisively on precision (~92% vs ~73%) and emits zero
  shared-node predictions, but those are pre-specified *secondaries*. That is an
  argument for revising the rule **before** the next run, not for overriding it now.
- **Retrain on crossing frames (B0.5): NO.** The premise is false.
- **Adopt the B0.4 geometric filter for top-down: YES** — ~~globally~~
  **SCOPED TO GROUP K, 2026-08-03.** Measured on all 65 experiments in 12 groups
  by the cardinality anchor: outside K the filter destroys **7.6 points of
  exact-3** and more than doubles the miss rate (`<3` 6.44% → **14.58%**), with
  per-group damage up to **−18.6 points (H)**, −15.0 (I), −14.9 (N), −12.5 (E).
  The **span rule** is 93% of the mechanism (97,523 of 104,809 deletions) and its
  bounds `[35.1, 152.4] px` are the 0.5–99.5 percentile of *Group-K labels*;
  K has the lowest drop rate of any group precisely because the bounds came from
  it. Group medians are similar, so this is an upper-tail effect, not a scale
  shift. **Do not apply the span rule outside K.** The shared-node half is
  unaffected and stays. B0.6's assignment reaches a comparable `>3` reduction for
  **+0.18** points of `<3` against b04's **+8.14**, because it converts an
  out-of-band instance into a mouth-only anchor instead of deleting the animal.
- **FLAGGED:** architecture-independent decoder work — every model falls below the
  80% crossing-recall floor in at least one arm.

**Three things this does not establish, which bound everything downstream:**

1. ~~**No unbiased evaluation set exists.**~~ **CORRECTED 2026-08-03 — one exists
   and this rebuild had not looked for it.** The retired repo built a locked,
   human-annotated benchmark on 2026-07-26/27 and it was preserved during the
   migration: `benchmarks/sleap_hitl_v2/phase1-hitl-v2-preserve-.../final/
   shared_benchmark_40_hitl_v2_locked.pkg.slp` — 40 source frames × 3 views,
   **360 human instances, exactly 3 per view-frame**, across **all three K days**,
   with **zero** frame overlap with round-7 training, round-8 train or round-8 val.
   It is **not in `MANIFEST.json`**, which is a defect to fix.
   The original point survives in weakened form: **frame-disjoint is not
   annotation-independent.** 30% of the locked benchmark's instances are
   byte-identical to *round-8* output — the same seeding error one round later —
   so it too must be bracketed, and on the arm independent of round 8 all three
   top models are indistinguishable. Phase A still has work: this benchmark is
   `n_groups = 1`, and A.3 requires ≥ 3 groups.
2. **`n_groups = 1`.** Every held-out frame is group K, day 2.
3. **Round 8 — the model actually reported as failing — is still unmeasured at
   crossings.** Top-down only, and its saved validation splits hold 5 crossing
   frames (top) and 6 (side).

Phase C is not started.

---

## Phase S — Three-node skeleton *(added 2026-08-04, experimenter request)*

Extend the production skeleton from `mouth → tail` to
`mouth → pectoral → tail`. Runs alongside Phase A rather than blocking it.

- [x] **S.1 Build the HITL bootstrap seed** *(2026-08-04)*. 10% of round-8
      train — **139 moments × 3 views = 417 view-frames, 1,251 instances** —
      with `pectoral` pre-placed at **t = 0.30** along mouth→tail. At
      `Z:\StephensU\medaka\annotations\sleap_pectoral_seed\pectoral-seed-20260804T103622JST-3d5f46b7c830`;
      rebuild with `tools/s01_pectoral_seed.py`.
      **`t = 0.30` is a prior, not a measurement** — no pectoral label exists
      anywhere in this project to fit it to. Sampling is deterministic (stable
      hash) with a **166-frame minimum separation**, because labelled frames in
      this corpus arrive 4 frames — 24 ms — apart and an unconstrained sample
      buys near-duplicates. **112 of 1,251 instances carry no seed**: their
      mouth or tail has no coordinates to interpolate from.
- [x] **S.2 Generalise the multi-view annotator** *(2026-08-04)*. It was
      hardcoded to two nodes and would have opened a 3-node file, drawn two of
      the three, and saved what it had. Now reads its node list from the seed;
      a 2-node seed is unchanged. Fixed a latent defect on the way: **the ID
      swap exchanged mouth and tail by name and would have left a third node
      behind** — a silent half-swap in the one command that exists to correct
      identity errors. 48 checks pass, 24 of them driving the real window.
      **Shipped as a self-contained bundle** at
      `Z:\StephensU\medaka\annotations\multiview_annotator_3node` after the
      first attempt failed on a fresh PC: it carried only the annotator `.py`
      and assumed the machine already had the 2-node bundle's dependencies, so
      the GUI died on `ModuleNotFoundError: numpy`. Two further defects came
      out of running it the way a user does — `.bat` files need **CRLF and
      ASCII** or `cmd.exe` executes their comments, and `"%~dp0"` passed as an
      argument **swallows the next one** because its trailing backslash escapes
      the closing quote. Verified from a clean local copy with no prior setup.
- [ ] **S.3 Annotate.** Human work; not started.
- [ ] **S.4 Measure the corrected `t` distribution**, per view — it says how
      wrong 0.30 was and costs nothing. **Count untouched seeds separately**:
      a pectoral still at exactly 0.30 was either correct or not looked at, and
      those are not the same thing.
- [ ] **S.5 Register the seed and the returned annotations in `MANIFEST.json`.**
      Same defect as the locked HITL-v2 benchmark, already open under A.3.

**Gate S:** a corrected-`t` distribution per view, with the untouched-seed
count stated. Do not train a 3-node model before it exists — and note that
doing so needs new **top and side** models, since no 3-node checkpoint exists.

**n_groups = 1.** Round-8 train is Group K only, so the seed is too.

Full record:
[`decisions/2026-08-04-three-node-skeleton.md`](decisions/2026-08-04-three-node-skeleton.md).

---

## Phase A — Build the ruler *(blocks all method work)*

Nothing downstream is meaningful without a way to measure a swap.

- [ ] **A.1** Encounter candidates for a colour-separated trial with all three
      days — start with **K_1**. Report count, duration distribution, and
      annotation budget.
- [ ] **A.2** Annotation tool. Requirements, each non-negotiable:
      - the clip **starts and ends in separation** — omitting this invalidated the
        retired repo's audit, because an annotator cannot say who left if they
        never saw who arrived;
      - annotator marks three animals at entry and again at exit, preserving
        correspondence (the "scrub back and forth" operation);
      - female may be named directly; **males are anonymous-but-consistent** —
        never ask which male is *white*, that judgement is not available in-video;
      - `confident=False` is always available and excludes the encounter from
        scoring. Never record a guess.
- [ ] **A.3** Frozen benchmark. `dev` (iterate freely) and `test` (touched only to
      report a final number). Span **≥3 different groups**, not three days of one —
      a group is one trio, so three days of K is n=1.
      **Start from what exists (found 2026-08-03):** the locked HITL-v2 benchmark
      is 40 source frames × 3 views, 360 human instances, three K days,
      frame-disjoint from every training split. Two jobs, not one:
      (a) **register it in `MANIFEST.json`** so it is checksummed and findable —
      it is human-made data living outside the catalogue, which is design rule 3
      and rule 7 both;
      (b) **extend it to ≥ 3 groups**, because it is `n_groups = 1` and that is
      the binding limit, and annotate the new frames **without** seeding from any
      model — 30% of the existing instances are verbatim round-8 output, which is
      exactly the mistake round 8's own labels made against round 7.

**Gate A:** one command scores any registered component and emits a `Scorecard`.
**Phase C is forbidden until this exists.**

---

## Phase B — Collect the data the methods actually need

- [ ] **B.1** Deliberately sample **<20 mm** episodes — the frames every previous
      filter discarded.
- [ ] **B.2** All three cameras, not side1 only.
- [ ] **B.3** Split by **time window**, never randomly. Random splits leak
      near-duplicate adjacent frames and inflate every score (§4.4).
- [ ] **B.4** Full provenance per crop: source video, frame, mask index, bbox,
      score.

**Gate B:** a dataset card stating counts per regime, per view, per split, and
proving no time window appears in two splits.

---

## Phase C — Methods, one stage at a time

Each is a lab implementing one stage `Protocol`, scored by the shared harness.

- [x] **C.0 3D reconstruction and tracking of group K** *(2026-08-13,
      experimenter request — **ran ahead of Gate A, knowingly**)*. The s16
      bottom-up 3-node predictions were reconstructed in 3D through the promoted
      refractive calibration and linked by three competing methods. **3D was
      chosen over 2D**, and **`gated`** over `nearest` and `predictive`.
      Reconstruction now covers **95.9–97.1%** of frames per day at a
      **0.38–0.43 mm** point-to-ray residual — the calibration's own accuracy.
      New library code: `medaka3d.calib`, `medaka3d.reconstruct`,
      `medaka3d.track`, `medaka3d.eval.tracking`.
      **What running without Gate A cost, exactly as the gate predicts:** the
      primary metric could not separate the methods (it compares a tracker
      against the geometry all three are built on), so the choice rests on a
      secondary one, and no statement about *identity* — as opposed to
      geometry — is available from any of it. The tracks are usable for
      encounter segmentation and for C.1; they are not validated.
      Full record:
      [`decisions/2026-08-13-tracking-3d.md`](decisions/2026-08-13-tracking-3d.md),
      which also prices the M1/M2 ground truth Gate A needs: **390 encounters
      across ≥3 groups, ~780 marked frames, ~10 h of annotation.**
      **n_groups = 1.**
- [x] **C.0b Top-down 3-node, scored as a production candidate** *(2026-08-15,
      experimenter request)*. Full-length top-down inference (`c05`, 3,137,400
      frames on saion) put through the identical reconstruction and `gated`
      tracker, against a pre-registered five-clause rule. **Top-down cleared one
      clause (coverage, +0.23 pt) and failed four.** Shared tails **900–2,100×**
      bottom-up's, `tail_swap` **1.9–5.8×** worse, track jumps **11×** worse.
      **Bottom-up stays.** The tracking evidence *confirms* the 2026-08-09 and
      2026-08-13 decisions rather than overturning them: part-mixing is **not**
      absorbed by three cameras — it propagates all the way to impossible track
      motion. Two by-products worth keeping: `sleap-nn/0.3.1` was **withdrawn
      from saion** and the 0.3.1→0.3.3 runtime change was measured to be
      bit-identical over 75,000 frames (see [`OIST_HPC.md` §2a](OIST_HPC.md));
      and top-down places a mouth 50× more reliably than bottom-up
      (0.021–0.029% missing vs 1.065–1.507%), which makes bottom-up's mouth gap
      the cheapest known improvement. Full record:
      [`decisions/2026-08-15-topdown-3node-tracking.md`](decisions/2026-08-15-topdown-3node-tracking.md).
      **n_groups = 1.**
- [ ] **C.1 3D body-length anchor** *(do first — cheapest, and it unblocks
      cross-day validation)*. Separate the males by continuity within one day,
      measure each one's 3D mouth–tail distribution over thousands of frames, and
      test whether the ordering is stable across days and trials. Length is
      physical and view-invariant, so it owes nothing to appearance — which is the
      only way to validate an appearance model without checking it against itself.
      Either outcome is publishable-grade information about the method.
- [ ] **C.2 Appearance identifier, cross-trial validated** (`identifier`). Train on
      trial 1, test on trial 2 — same animals, weeks apart, provably no
      within-video shortcut. `Trio.holdout_by_trial()`. Report `naming_accuracy`.
- [ ] **C.3 Segmentation-instance linker** (`linker`). SAM2 video propagation.
      Masks also give the appearance model clean pixels instead of a
      background-contaminated bbox. Constraint (§2.7): segmentation is a
      *confidence signal*, never a hard veto.
- [ ] **C.4 Global fragment path-cover** (`identifier`). The best-designed
      *unexecuted* proposal in the retired repo: explicit edge costs, multiple
      independent anchors, **unanchored paths left unlabelled**.
- [ ] **C.5 Reflection-suppressing scene model** (`detector`). Addresses the
      entry-phantom bug (§4.6).

**Gate C:** every promoted component beat the production baseline on the frozen
`test` split, under a decision rule written before it ran.

---

## Standing questions to resolve with the experimenter

- [ ] Is group **P** group **C**'s males with a new female? (C: "give up female at
      20260105"; P: set 20260105, "only change female", same two male
      descriptors.) If yes, C and P are not independent units.
- [ ] Which camera view shows the male morphological difference best? Decides
      where appearance crops come from.
- [ ] Do reference photographs of individual fish exist from setup? They would be
      a direct cross-day anchor and would make C.2 verifiable immediately.

---

## Things that will be tempting and are wrong

- Treating `K_1_1/K_1_2/K_1_3` as N=3. It is one trio, three times. Use
  `cohort.Trio`; the independent unit is the **group**.
- Reporting a frame-wise identity accuracy. At 166 fps it is dominated by easy
  separated stretches and reads >99% while swapping at every crossing.
- Reading a high `survival_rate` as "identity solved". It is relabelling-invariant
  by design; naming is a separate metric.
- Assuming the colour descriptors are usable in-video. They are not — they were
  assigned under close examination at setup.
- Deleting a model directory without enumerating its label files first. 7.44 GB of
  human annotation was nearly lost that way during the migration.
