# Roadmap — ordered phases with gates

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

- [ ] **0.1** Read `CLAUDE.md`, then `docs/HANDOFF.md`, then the **Corrections**
      section at the top of `docs/PRIOR_FINDINGS.md`. The corrections overturn two
      claims repeated later in that same file.
- [ ] **0.2** Environment: `uv pip install -e ".[dev]"`, then `pytest -q` and
      `ruff check .`.
- [ ] **0.3** Verify the inherited assets resolve and are intact:

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

- [ ] **0.5.1** Test whether the study's conclusion is reachable **without
      tracking at all**. `PRIOR_FINDINGS.md` §1.7 suggests the lab sheet's
      `female_accept` field and the BORIS courtship asymmetry may agree in every
      experiment where both exist — an N=7 result from two independent human
      record sources, no ML. It is flagged there as an *unverified derivation*.
- [ ] **0.5.2** If it holds: verify annotator independence (were the BORIS coder
      and the sheet's author the same person, or was one derived from the other?).

**Gate 0.5:** a written answer either way. This costs hours and determines how
much tracking precision the paper actually needs. Do not skip it because tracking
is more interesting.

---

## Phase B0 — Fix SLEAP at crossings *(first real engineering work)*

Reported by the experimenter: round 8 degrades exactly at crossings — one tail
joined to two mouths, unstable detection in X-crossings. A fish that is never
detected cannot be recovered by any downstream cleverness, so this comes first.
**No new annotation is needed for B0.1–B0.3.**

- [ ] **B0.1 Diagnose the training data.** Count labelled frames in
      `assets/.../round8 combined_labels.pkg.slp` containing two instances closer
      than 100 px. §2.10 predicts approximately zero. Needs `sleap-io`
      (not yet a dependency — add it).
      *Expected finding: the model has essentially never seen a crossing.*
- [ ] **B0.2 Build a crossing-frame evaluation set.** Retained round-7 `top_cam`
      predictions for all 65 experiments are at `paths.sleap_output_dir`. Link
      them roughly, then use `medaka3d.eval.find_encounters` to locate crossings.
      Report the count and the annotation-minute estimate from `summarise()`.
- [ ] **B0.3 Re-score existing checkpoints on crossing frames only.** Round 7
      trained both top-down and a bottom-up baseline; the production choice was
      made over *all* frames, which are overwhelmingly separated fish. A model can
      win on average and lose exactly where it matters. Both checkpoints are on
      `/bucket` — this is inference only, no training.
- [ ] **B0.4 Geometric plausibility filter.** Body length is physical and bounded
      (~25–35 mm) and calibration makes it measurable. Reject instances sharing a
      node, or with implausible 3D length. Cheap post-process; catches the
      reported failure directly.
- [ ] **B0.5 Retrain with crossing frames** *(needs annotation — coordinate with
      Phase A.2 so one pass serves both)*.

**Gate B0:** a detection-quality number at crossings versus overall, for both
architectures. That number decides retrain-vs-switch-architecture. Do not retrain
before it exists.

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
