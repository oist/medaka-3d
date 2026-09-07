# PRIOR FINDINGS — carried forward as HYPOTHESES, not facts

**Source repo:** `medaka-mating-pipeline` (169 commits, `main` @ `b506a69`, snapshot 2026-08-01)
**Purpose:** The old repo is being abandoned because incomplete predictions and code bugs made it
unclear what was actually true. Everything below is recorded so the new repo can **re-derive** it,
not inherit it. Nothing in this document is an established fact.

---

## ⚠️ CORRECTIONS — 2026-08-02

Two framing facts below were **wrong**, and both were resolved from the
experimenter's own records in `experiment_list/` (now under
`asset_root/catalog/experiment_list/`). Corrections are recorded here rather
than edited in place, so the original reasoning stays auditable.

> **Five further corrections, C3–C7, are recorded at the end of [§1.7](#17-egg-laying--outcome-link-a-corroboration-channel-that-was-found-but-never-written-up).**
> §1.7's "two independent human record sources", its p = 0.0078, and its 7/7 count
> do not survive re-derivation: there is **one** annotator pass, not two channels;
> the egg record cannot discriminate between males; the unit is the group, so
> **n = 3 trios**; and the one interesting row flips with the choice of metric.
> Phase 0.5 is answered — the conclusion is **not** reachable without tracking.
> Full working: [`decisions/2026-08-02-phase05-accept-vs-courtship.md`](decisions/2026-08-02-phase05-accept-vs-courtship.md).

### C1. "The three trials may not be three independent females" — resolved: they are ONE trio

Framing fact 2 recorded this as INCONCLUSIVE. It is not. The folder convention
is `<set_date>_<GROUP>_<trial>_<day>_<paradigm>`, confirmed by the sheet's own
legend (`ExpStartTime_Group_Times_Day_Procedure`) and by the catalog's
`experiment.group` / `.trial` / `.day` columns:

| folder | group | trial | day | actual recording date |
|---|---|---|---|---|
| `20251225_K_1_1_FCMC` | K | 1 | 1 | 2025-12-25 |
| `20251225_K_1_2_FCMC` | K | 1 | 2 | 2025-12-26 |
| `20251225_K_1_3_FCMC` | K | 1 | 3 | 2025-12-27 |

**K_1_1, K_1_2 and K_1_3 are the same three fish on three consecutive days.**
The date in the folder name is the *set* date, not the recording date — which
is why an earlier pass reported them as "all recorded 2025-12-25".

Consequence: the headline result in §1.1 — "female mate choice in all 3 trials",
association 0.61 / 0.71 / 0.80 — is **n=1 trio with three repeated measures**,
not n=3. Every prior analysis in this repository used this one trio. Reporting
it as N=3 would not survive review.

The independent unit is the **trial**, not the folder. `medaka3d.cohort.Trial`
exists to make that a type rather than a convention.

**The dataset is far richer than the prior work used: 24 distinct trials across
14 groups, 22 of them with all three days.** Generalisation was never tested
because 23 of 24 trials were never analysed.

### C2. "The two males are not visually distinguishable" — false

Repeated below as justification for treating the M1/M2 letters as provisional.
The experimenter recorded a physical descriptor for every animal in
`group_list.csv`:

| group | female | male 1 | male 2 | separating cue |
|---|---|---|---|---|
| K | yellow | **white** | **red** | colour |
| M | smallest | tail yellow | tail red | colour |
| I | biggest | yellow, small | white, big | colour + size |
| L | yellow | small | big | size only |

Across all 24 trials: **14 separated by colour + size, 8 by colour, 2 by size
only** (L_1, L_2). No group has indistinguishable males — enforced as a test in
`tests/test_cohort.py`.

Consequences:

1. **M1 and M2 are not arbitrary labels.** They denote specific animals. A
   consistent global relabelling is a real error, so continuity and naming are
   scored separately (`Scorecard.survival_rate` vs `.naming_accuracy`).
2. **A learnable appearance signal exists**, and it is colour — not a subtle
   texture. The prior conclusion that geometry had hit a ceiling and appearance
   was marginal was reached without using this.
3. **Size-only groups are weak targets.** Apparent size varies with pose, depth
   and camera, and the animals grow between days. Prefer colour-separated
   groups; treat L as a stress test, not a starting point.

---

## READ THIS FIRST — five framing facts

> Facts 2 and 5 are superseded by the corrections above. Read them as a record
> of what was believed, not as current state.

1. **Nearly all science rests on 3 experiments.** K_1_1, K_1_2, K_1_3 (all recorded 2025-12-25,
   group "K", all `FCMC`). The corpus is ~71–76 experiments. **Generalization is completely
   untested.** Sources: `runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md`,
   `report/20260708/round7_bakeoff/DECISION.md`.

2. **The three trials may not be three independent females.** A cross-trial appearance re-ID probe
   returned a *sex-inconsistent* grouping ("a fish cannot change sex across trials"), so it could
   not confirm whether the same individuals were reused. Verdict recorded as INCONCLUSIVE; "the
   experimental design records are the authority."
   Source: `runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md` (script
   `runs/dual_line/extend_k11_k13/cross_trial_identity.py`). **N=3 cannot currently be claimed as
   3 independent replicates OR as a within-subject design.**

3. **BORIS coverage is partial and the exact number is disputed across sources.** The 06-30 cohort
   build parsed **23 BORIS CSVs** (`report/20260630/boris_cohort/data/boris_cohort_key_metrics.json`:
   `input_file_count: 23, parsed_file_count: 23, missing_files: []`); the experiment catalog reports
   **22/71 with `has_boris = True`** (`scripts/common/build_catalog.py` →
   `Z:\StephensU\medaka\medaka_202512\catalog.csv`). The briefing for this handoff says 24–26 of 76.
   One known join miss: BORIS file `20260112_I_2_3_FCMC` vs catalogued folder `20260113_I_2_3_FCMC`
   (date mismatch). **Treat "roughly 22–26 of ~71–76 annotated" as the honest range and recount in
   the new repo.**

4. **The `runs/` tree is gitignored** (`.gitignore:213`). Every calibration report, every
   `FINDINGS*.md` under `runs/`, and the whole dual-line evidence chain **exists only on this
   machine and is not in git history**. The `report/` tree IS tracked. If the old machine dies,
   most of the evidence below is gone. **Highest-priority preservation action.**

5. **The ground truth labels only the female.** Repeated verbatim across
   `report/20260717/windowed_tracklet_stitching/*_verifier_report.md`,
   `report/20260731/round8_mouth_tracking_ablation/COMPARISON_REPORT.md`,
   `report/20260718/k12_jump6_strict_mouth/*`: *"the current GUI ledger labels only the female and
   cannot independently detect male-to-male swaps."* Total human female annotations: **300**
   (`report/20260716/female_sex_gt_consistency/analysis_summary.json`). There has **never** been a
   male-vs-male ground truth set. Every M1/M2 number in this document is therefore unvalidatable
   with current data.

---

# 1. Biological / scientific claims

Confidence scale: **A** = corroborated by ≥2 genuinely independent channels; **B** = single
pipeline + one weak external check; **C** = single pipeline, no external check.

---

## 1.1 Female mate choice — association index above chance in all 3 trials

**Claim (verbatim):** "in every trial the female associates 1.6–4× more with one male"; association
index (preferred male's share of time within 30 mm) = **0.61 / 0.71 / 0.80** for
**K_1_2 / K_1_3 / K_1_1**. Chance = 0.50.

| exp | assoc index | time <30 mm M1 vs M2 | female nearer M1 in courtship-range frames | valid F–M frames |
|---|---|---|---|---|
| K_1_2 | 0.61 | 286.5 s vs 181.6 s | 67.4% vs 32.6% | 345,211 (99.0%) |
| K_1_3 | 0.71 | 639.9 s vs 263.2 s | 73.7% vs 26.3% | 347,890 (99.8%) |
| K_1_1 | 0.80 | 563.3 s vs 138.4 s | 82.6% vs 17.4% | 347,943 (99.8%) |

**N:** 3 trials, 348,600 frames each (2100 s @ 166 fps), 1 gravid female + 2 same-size males,
20×20×20 cm tank, 3 calibrated cameras.

**Artifacts:**
- `runs/dual_line/lineB_fulltimeline/analysis_debounced/mate_choice_report.md` (K_1_2, debounced)
  and `runs/dual_line/lineB_fulltimeline/analysis/mate_choice_report.md` (interp variant)
- `runs/dual_line/extend_k11_k13/K_1_1/analysis/mate_choice_report.md` + `mate_choice_per_frame.csv`
  + `mate_choice_bouts.csv` + `mate_choice_temporal.csv`
- `runs/dual_line/extend_k11_k13/K_1_3/analysis/…` (same layout)
- Tracks: `runs/dual_line/lineB_fulltimeline/step2/round5_260611_3d_individual_id_interp_debounced.csv`;
  `runs/dual_line/extend_k11_k13/K_1_{1,3}/round5_260611_3d_individual_id_interp_debounced.csv`
- Roll-up: `runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md`
- Numbers are also frozen in code at `report/20260630/build_figures.py` L70-104 (hard-coded `DATA` dict)

**Corroboration — this is the strongest claim in the project:**
1. **Second, independent backbone** (round7 SLEAP + M3 appearance) reproduces it: **0.62 / 0.71 /
   0.80** — `report/20260708/round7_bakeoff/DECISION.md`.
2. **Second, independent K_1_2 track** (OLD committed vs NEW `recon_nofill`) both give **0.61**;
   closer-to-preferred-male 69.9% vs 70.0% — `report/20260630/k12_appearance/bakeoff/DECISION.md`.
3. **Manual BORIS ethogram** agrees on direction in all 3 (§1.5).
4. **Spawn outcome** agrees in all 3 (§1.3).
5. **Lab experiment sheet** independently records `female_accept = m2` for all three
   (`Z:\StephensU\medaka\medaka_202512\experiment_list\lay_eggs_table.csv`, joined by
   `scripts/common/build_catalog.py`).

**CONFIDENCE: A** (for "the female preferred one male"). **B** for the exact index values.

**CONTAMINATION RISK:**
- The *appearance classifier* that names the female was trained on a dataset later audited as
  **circular and leaky** (§4.4) — the historical appearance metrics were formally withdrawn as
  biological-ID evidence. The index survives only because it is anchored on the *independent* spawn
  landing + BORIS + gravid montage, not on classifier confidence.
- The index depends on which fish is called "the female". A **cross-video female mislabel**
  (§4.2) originally put a MALE in the female role for K_1_1 and K_1_3; the published 0.71/0.80 are
  the **post-fix** values. If the fix is wrong, both are wrong.
- Frames are not independent (166 fps); no proper statistical test was run — the "above chance"
  claim has **no p-value anywhere in the repo**.
- Interpolation bridges the contact "hard core" (~1% of frames), which is exactly the regime the
  index is most sensitive to.

**RE-VERIFICATION RECIPE (to promote from hypothesis to fact):**
1. Rebuild 3D tracks for K_1_1/K_1_2/K_1_3 from raw `.slp` on the new repo's calibration.
2. Establish female identity by an **independent** anchor that is not the appearance classifier —
   the gravid-belly morphology at ≥30 human-labelled frames spread across the timeline, plus the
   BORIS spawn-wrap gate (female must be ≤~12 mm from a male at the annotated spawn frame).
3. Recompute the association index. **Report the whole distance-threshold sweep (30/20/12/8 mm),
   not one number.**
4. Add a real statistical test: bout-level or block-bootstrap over temporal blocks (frames are
   autocorrelated at 166 fps). Report a CI.
5. **Extend to ≥3 more experiments outside the K group** before claiming anything general.

---

## 1.2 Preference sharpens toward contact

**Claim:** preference ratio rises as the proximity threshold tightens 30 → 8 mm.

| exp | <30 mm | <20 mm | <12 mm | <8 mm |
|---|---|---|---|---|
| K_1_2 | 1.58× | 1.83× | 1.79× | 1.88× |
| K_1_3 | 2.43× | 2.54× | 2.51× | 3.68× |
| K_1_1 | 4.07× | 5.34× | 6.79× | **10.82×** |

**Artifacts:** the three `mate_choice_report.md` + `mate_choice_per_frame.csv`; hard-coded in
`report/20260630/build_figures.py` L78/88/98.

**Corroboration: NONE.** Single pipeline per experiment. The round7 bake-off re-checked only the
assoc index and spawn distance, **not** the sharpening ratios.

**CONFIDENCE: C.**

**CONTAMINATION RISK: HIGH — this is the claim most exposed to the identity bug.** Sub-12 mm is
precisely the close-contact regime where (a) SLEAP fragments, (b) geometry ID swaps, (c) the
appearance dataset has **zero coverage** (`runs/dual_line/phase0/FINDINGS.md`: *every* clean-mask
frame has 3D distance >20 mm; **zero coverage below 20 mm**), and (d) the track is
interpolation-bridged. A female↔male swap in contact inflates the ratio for whichever male the
tracker happened to be holding.

**RE-VERIFICATION RECIPE:** Do not attempt until close-contact identity is independently validated.
Then: hand-label identity in ≥200 randomly-sampled frames drawn *only from* <12 mm episodes,
measure the per-frame identity error rate in that regime, and propagate it as an error bar on the
ratio. If the contact-regime error rate exceeds ~5%, the <12 mm and <8 mm columns are not reportable.

---

## 1.3 She spawns with the preferred male (contact at spawn)

| exp | spawn frame / time | d(F, preferred M) | d(F, other M) |
|---|---|---|---|
| K_1_2 | 135888 / 818.6 s | 7.4 mm | 11.5 mm |
| K_1_3 | 121602 / 732.5 s | 4.9 mm | 124.1 mm |
| K_1_1 | 82288 / 495.7 s | 5.4 mm | 110.5 mm |

Independent round7/M3 rerun: **6.4 / 7.4 / 5.9 mm** (`report/20260708/round7_bakeoff/DECISION.md`).
K_1_2 pre-identity geometry-only check (2026-06-22): nearest male in ±1 s = ID_3, median 7.1 mm vs
13.1 mm for the other — `report/20260622/behavior_comparison/behavior_trajectory_summary.md`.

**Spawn frame ground truth = manual BORIS `Spawn` point event.** 15 spawn-coded observations
cohort-wide. `report/20260630/boris_cohort/data/boris_cohort_key_metrics.json` +
`boris_cohort_experiment_summary.csv`.

**Corroboration: STRONG.** Human BORIS timestamp + two independent tracking pipelines. Docs call
spawn-landing "the strongest GT" (`FINDINGS_N3_mate_choice.md`; `report/20260630/README.md`).

**CONFIDENCE: A** for "she spawns in contact with one male, and that male is the one she associated
with more." **C** for the precise mm values.

**CONTAMINATION RISK:**
- ⚠️ **Frame-number inconsistency:** K_1_2 spawn is **135888** in the mate-choice reports but
  **135883** (818.572 s) in BORIS / behavior_comparison. 5-frame (30 ms) offset, unresolved.
- ⚠️ **Doc-stated blocker:** *"Spawn-pileup male identity (M1 vs M2) is not resolvable at the exact
  spawn frame"* — all 3 fish are mutually <15 mm for 67% of the ±1 s window, males 3–11 mm apart,
  male `id_conf` = interp/ambiguous (`report/20260630/k12_appearance/bakeoff/DECISION.md`). So the
  spawn corroborates the *female*, not the male letter.
- The spawn distance is also the **validation gate** used to fix the female mislabel — using it
  again as evidence is partly circular. It is a valid gate; it is not independent confirmation of
  the track that was tuned to pass it.

**RE-VERIFICATION RECIPE:** Recompute spawn-window distances from a fresh reconstruction. Keep the
spawn gate as a *pass/fail filter* on the identity assignment, and then find a **separate**
outcome to corroborate the choice (e.g. total wrap-contact duration per male, computed on frames
far from the spawn). Resolve the 135883/135888 discrepancy by re-deriving BORIS `Image index` →
frame alignment from scratch.

---

## 1.4 Courtship ≫ aggression, and the courtship channel (not aggression) is asymmetric

**Claim:** courtship : aggression time ≈ **19–25×** in all three; the preferred male courts
**1.7–4.3×** more.

| exp | courtship bouts/total | preferred vs other courtship | ratio | aggression bouts/total | courtship:aggression |
|---|---|---|---|---|---|
| K_1_2 | 483 / 435.2 s | 270.9 s vs 164.3 s | 1.65× | 171 / 17.7 s | 24.57× |
| K_1_3 | 718 / 855.9 s | 614.4 s vs 241.5 s | 2.54× | 378 / 44.4 s | 19.28× |
| K_1_1 | 532 / 672.0 s | 544.7 s vs 127.3 s | 4.28× | 233 / 30.8 s | 21.84× |

**The asymmetry control (important):** male–male aggression *initiator* is near chance in all three
— 97/171 (K_1_2, 56%), 179/378 (K_1_3, 47.4%), 106/233 (K_1_1, 45.5%). So the courtship channel is
1.7–4.3× asymmetric while the aggression-initiation channel is ~50/50. **This contrast is the
actual quantitative content of the "courtship, not aggression" claim.**

Operationalization: aggression = close+fast bouts (<25 mm & >150 mm/s); courtship = sustained
female-proximity bouts (<30 mm). **These are proxies invented in this repo, not validated ethogram
definitions.**

**Artifacts:** the three `mate_choice_report.md` + `mate_choice_bouts.csv`.

**CONFIDENCE: C** — single pipeline, and **backbone-sensitive**.

**CONTAMINATION RISK: HIGH.**
- ⚠️ **The ratio is not stable across backbones.** On the same K_1_2 data, the committed track
  gives **24.5×** but the NEW `recon_nofill` backbone gives **15.2×**
  (`report/20260630/k12_appearance/bakeoff/DECISION.md`). The association index was invariant
  across the same two backbones; the courtship:aggression ratio was **not**. That is a red flag
  that the ratio is measuring tracker behaviour as much as fish behaviour.
- The aggression proxy (>150 mm/s at <25 mm) is exactly the signature an **ID swap** produces.
  Speed-spike QC was shown to be blind to swaps (§2.2), so aggression counts may be inflated by
  residual swaps — or suppressed by the teleport-debounce step that was added to remove them
  (`per_frame_resolve.debounce_teleport_swaps`, K_1_2: 184 → 71 teleport events, commit `c15be12`).
- Male-directed metrics depend on the M1/M2 letter, which is explicitly **provisional** (§1.7).

**RE-VERIFICATION RECIPE:**
1. Define courtship and aggression against the **human BORIS categories**, not against invented
   distance/speed thresholds — then report the thresholded proxy only as a *validated surrogate*
   with a measured agreement rate against BORIS.
2. Recompute on ≥2 independent backbones and report both. If the ratio moves >30% between
   backbones, it is not reportable.
3. Sensitivity-sweep the thresholds (distance 20/25/30 mm × speed 100/150/200 mm/s) and show the
   conclusion is threshold-invariant.

---

## 1.5 BORIS ethogram independently agrees on which male was chosen

| exp | BORIS m1→f | BORIS m2→f | asymmetry | tracker's preferred male | agree? |
|---|---|---|---|---|---|
| K_1_1 | 9.52 s | 118.19 s | **12.41×** toward m2 | M1 (0.80) | ✅ M1 ≡ m2 |
| K_1_2 | 41.82 s | 103.56 s | **2.48×** toward m2 | M1 (0.61) | ✅ |
| K_1_3 | 6.75 s | 99.16 s | **14.70×** toward m2 | M1 (0.71) | ✅ |

**Artifacts:** `report/20260630/boris_cohort/data/boris_cohort_experiment_summary.csv` (columns
`m1tf_duration_s`, `m2tf_duration_s`, `boris_male_courtship_ratio`, `boris_more_courtship_subject`);
source CSVs on `G:\…\OIST 1215-0122\boris_events\20251225_K_1_{1,2,3}_FCMC.csv` (131/173/179 raw
rows → 68/89/92 parsed events).

**CONFIDENCE: A** for direction. **This is the genuine independent human corroboration.**

**CONTAMINATION RISK:**
- ⚠️ **Doc-stated caveat, verbatim (all three `mate_choice_report.md`):** *"BORIS m1/m2 labels are
  the annotator's and not directly mappable to appearance M1/M2, but the per-male ASYMMETRY should
  agree."* So the agreement is **directional only** — it is not an identity-level match, and it
  cannot detect a systematic swap that inverts both channels together.
- ⚠️ **Cohort-level annotator skew:** across all 23 BORIS files, m2→f totals **1006.05 s / 380
  events / 18 exps** vs m1→f **655.37 s / 179 events / 14 exps**. If the annotator tends to label
  the more active male "m2", then "the chosen male is m2 in all three K trials" is partly a
  labelling convention, not a finding.
- With N=3 and a binary agreement outcome, this is p = 0.125 under a fair coin — **not
  significant on its own.**

**RE-VERIFICATION RECIPE:** Recount the m1/m2 skew across all annotated experiments; if the skew is
real, ask the annotator how m1/m2 were assigned. Then test agreement on the **7 experiments where
the lab sheet records a resolved `female_accept ∈ {m1, m2}`** (see §1.6) — that is a larger and
partly independent sample.

---

## 1.6 Behaviour ↔ geometry ladder, and wrap acceptance duration

**K_1_2, `report/20260622/behavior_comparison/behavior_trajectory_summary.md`:** median
female-to-nearest-male distance by BORIS category —

| state | median distance | n |
|---|---|---|
| baseline | 40.4 mm | — |
| Courtship | 27.5 mm | 52 |
| Touch | 16.8 mm | **2** |
| Wrap | 8.0 mm | **3** |

Wrap acceptance (K_1_2): accepted (≥10 s) **n=2, median 28.0 s**; rejected (<10 s) **n=1, median
3.6 s** — matching the lab write-up `results_EN_260616.docx` (~27 s vs ~3 s).

**Cohort replication of the wrap-duration claim (much stronger):** across 23 BORIS files, wrap
accepted **n=23, median 25.74 s** vs rejected **n=26, median 3.00 s** —
`report/20260630/boris_cohort/data/boris_cohort_key_metrics.json`. Both K_1_2 backbones in the
bake-off reproduce the 28.0/3.6 s durations (`report/20260630/k12_appearance/bakeoff/DECISION.md`).

**CONFIDENCE:**
- Wrap accepted-vs-rejected duration bimodality: **A** (n=49 wraps across 16 experiments, and it is
  purely a BORIS-derived number — no tracker involved).
- The 40 → 28 → 17 → 8 mm ladder: **B**. It is a genuine cross-modal check (human category ↔
  measured mm), but **n=2 for Touch and n=3 for Wrap within K_1_2**. Do not quote medians on n=2.

**CONTAMINATION RISK:** The ladder file predates the M1/M2 resolution and explicitly treats the two
males as an undistinguishable **dyad** (verbatim L3: *"the two males (ID_1, ID_3) are not
appearance-distinguishable, so male-directed events are validated as a dyad"*). It therefore
validates *female-to-any-male* geometry, which is the robust part, and says nothing about male
identity.

**RE-VERIFICATION RECIPE:** Recompute the ladder pooled over **all** BORIS-annotated experiments
that also have 3D tracks, not just K_1_2 — that turns n=2/n=3 into a real sample. Report per-event
distances, not just medians. The wrap-duration bimodality can be re-derived from BORIS alone and
should be promoted first, since it needs no tracking at all.

---

## 1.7 Egg-laying / outcome link (a corroboration channel that was found but never written up)

The lab experiment sheet (`Z:\StephensU\medaka\medaka_202512\experiment_list\lay_eggs_table.csv`,
joined into `catalog.csv` by `scripts/common/build_catalog.py` via
`scripts/common/experiment_list.py::load_lay_eggs_table`) records per experiment: `eggs`,
`female_accept`, `male_contest`.

- **K_1_1, K_1_2, K_1_3: `eggs = yes`, `female_accept = m2`, `male_contest = yes`** — a *third*,
  human-recorded channel naming m2 as the accepted male in all three, matching BORIS (§1.5).
- Cohort-wide: `eggs` yes 29 / no 22 / blank 18 (of 71). `female_accept` populated for 28 (m2 ×4,
  m1 ×3, "no" ×11, "?" ×10).
- **7 experiments have a resolved `female_accept ∈ {m1,m2}`**: 20251220_E_1_1 (m1), E_1_2 (m1),
  E_1_3 (m2), 20251225_K_1_1/K_1_2/K_1_3 (m2), 20260109_O_1_1 (m1). In **7/7**, BORIS
  `boris_more_courtship_subject` matches the accepted male (ratios 1.41×, 1.75×, 2.67×, 12.41×,
  2.48×, 14.70×, 23.05×).

**⚠️ SOURCE STATUS: this 7/7 concordance is a derivation made during this handoff research by
joining `boris_cohort_experiment_summary.csv` with `catalog.csv`. It is NOT asserted in any repo
document. Treat as unverified until recomputed.** If it holds, 7/7 under a fair coin is p = 0.0078
— and it needs **no tracking at all**, only two human record sources.

**CONFIDENCE: C (unverified derivation), but potentially the cheapest strong result available.**

**CONTAMINATION RISK:** Both channels (BORIS coding and the lab sheet) may have been filled in by
the same person with knowledge of the other — check for annotator independence before claiming it.

**RE-VERIFICATION RECIPE — DO THIS FIRST IN THE NEW REPO.** It is pure bookkeeping, uses no
tracking, and either yields a publishable N=7 result or kills a false lead cheaply:
1. Re-parse `lay_eggs_table.csv` and all BORIS CSVs independently.
2. Confirm the 7 resolved rows and the 7/7 match.
3. Establish whether the BORIS annotator saw the egg-laying record (independence check).
4. Expand: chase the 10 `?` and 11 `no` rows for resolvable outcomes.

### ⚠️ CORRECTION — 2026-08-02: this was executed, and §1.7 above is wrong

Recorded here rather than edited in place, so the original reasoning stays auditable.
Full working: [`decisions/2026-08-02-phase05-accept-vs-courtship.md`](decisions/2026-08-02-phase05-accept-vs-courtship.md).
Reproduce: `tools/phase05_accept_vs_courtship.py`.

**C3. "A *third*, human-recorded channel" and "two human record sources" — false.
There is one source.**

- `lay_eggs_table.csv` and `boris_events.csv` carry **identical** courtship counts in
  **7/7** rows populated in both. The lab sheet is a partial copy of the BORIS
  aggregation, not a second observation.
- `female_accept` is defined as *"accept wrap"*, and the wrap events are in the **same
  BORIS file** as the courtship events. Predicting it by *which male wrapped first*
  reproduces it in **8/9** rows. It is a second behaviour category from one coding
  pass, not a second channel.
- **3 of the 26 BORIS files were coded on the recording day itself** (E_1_1
  2025-12-20, E_1_2 2025-12-21, K_1_1 2025-12-25); the rest in later waves. For those,
  coder and sheet-author are the same person on the same occasion.

**Consequence: the p = 0.0078 above does not hold.** It assumes independent channels
and a fair coin. Neither applies. The concordance measures the internal consistency of
one coding scheme.

**C4. The independent channel — the physical egg record — cannot discriminate between
males.** `eggs` / `eggs_after` are per *experiment* (yes/no); `Egg_Laying_by_pair.csv`
is per *group* percentages despite its name. All nine scorable rows are `eggs = yes`,
so the field has zero variance within the scored set. Nothing in the sheets records
*which* male fertilised the eggs. Without paternity data this channel is silent on choice.

**C5. "7 experiments … 7/7" — wrong scope, wrong unit, and metric-dependent.**

- **The unit is the group.** The scorable rows span only groups **E, K and O**:
  **n = 3 trios**, not 7, 8 or 9. Days within a group are repeated measures.
- Scope: `boris_events.csv` also resolves O_1_2 and O_1_3, giving 9 scorable rows.
  K_1_2 is `m2m1`, which the sheet's own legend resolves to **m2**.
- **E_1_3 flips with the metric.** By event count m1 courted more (6 vs 5) → miss; by
  duration m2 did (**7.7 s vs 7.4 s**) → match. Per-experiment totals are therefore
  **8/9 by count, 9/9 by duration**. The ratios quoted in §1.7 (1.41×, 1.75×, 2.67× …)
  are neither the sheets' counts nor these durations, so the quantity behind the
  original 7/7 remains unidentified. A 0.3 s margin over 2100 s is undetermined, not
  a result.

**C6. Preference is not stable within a trio.** Group E accepted **m1 on days 1 and 2,
then m2 on day 3**. K and O are internally consistent; E is not. One of three trios
switched — which bears directly on §1.8 ("preference is stable in time"), whose
evidence is drawn entirely from group K.

**C7. `female_accept` must not be used as ground truth for a tracking method.** It is
derived from the same video-coding pass a tracker would be scored against. It is a
label, not a referee.

**Standing: Phase 0.5 is answered — the study's conclusion is NOT reachable without
tracking.** Proceed with B0.

---

## 1.8 Preference is stable in time (not a late switch)

| exp | early bin | late bin | slope/bin | bins favouring the other male |
|---|---|---|---|---|
| K_1_2 | 0.65 | 0.66 | −0.001 | **1 ("a switch is present")** |
| K_1_3 | 0.73 | 0.65 | −0.010 | 0 |
| K_1_1 | 0.75 | 0.88 | **+0.023 (builds)** | 0 |

n = 10 temporal bins/trial. `…/analysis[_debounced]/mate_choice_temporal.csv`.

**CONFIDENCE: C.** Single pipeline, n=10 bins, no test. Note the poster claim "the female favours
the eventual mate from the start (not a late switch)" **softens K_1_2's own report line**, which
says 1 bin favours the other male and "a switch is present". Do not carry the softened version.

**RE-VERIFICATION RECIPE:** This maps directly onto README Goal 2 ("does mate choice occur early or
require observing male-male aggression?"). Re-derive with a proper changepoint test, and align the
temporal profile against the *timing of male-male aggression bouts* — that is the actual research
question and it was never answered.

---

## 1.9 The called female is biologically the gravid female

`runs/dual_line/lineB_fulltimeline/FINDINGS_step6_gt.md`: across 6 frames (f67799, f142060, f171107,
f201144, f237164, f277889) the called F is the deeper-bodied / rounder-abdomen (gravid) fish, and
she occupies **different geometry slots** in those frames — so the appearance line tracks the
biological female, not a geometry slot. Artifact: `runs/dual_line/lineB_fulltimeline/step2/gt_montage.png`
(also `report/20260630/reused/gt_montage.png`).

**Doc's own caveat, verbatim:** *"this is a qualitative spot-check, not a quantitative GT label
set."* And: *"M1 vs M2 are NOT visually distinguishable… Visual GT can corroborate the FEMALE
anchor but cannot easily confirm M1≠M2 by eye."*

**CONFIDENCE: B** for the female. **The M1/M2 letter is explicitly NOT proven** —
`report/20260708/round7_bakeoff/DECISION.md` verbatim: *"M3 and the gold agree ~98% but are
correlated appearance pipelines. The robust conclusions are 'the female prefers one male and spawns
with him'; which letter that male carries is not proven."*

**RE-VERIFICATION RECIPE:** Build a real female GT set: ≥200 frames stratified over the whole
timeline and over contact/non-contact regimes, labelled by a human on raw video without tracker
overlays. The existing 300-annotation ledger
(`report/20260716/female_sex_gt_consistency/analysis_summary.json`) is a starting point but was
model-assisted.

---

# 2. Methodological findings that should shape the new design

## 2.1 Geometry-only identity has a hard ceiling — it cannot name the female

Three independent quantifications:

- **Human ledger (decisive).** 300 human-confirmed female annotations, frames 4,439–348,388: the
  female's geometry slot distributes **ID_1 43.3% / ID_2 22.8% / ID_3 33.9%**. Best possible fixed
  geometry slot = **0.4329 accuracy**; **26 observed ID switches over 297 adjacent pairs**.
  `report/20260716/female_sex_gt_consistency/analysis_summary.json`.
- **Round7 bake-off.** Geometry method M1 has excellent *track quality* (coverage 0.843–0.874, **0
  speed-spikes >8 mm**, 96–97% top-bind) but its implied-female matches the gold only **43% / 47% /
  54%** within 20 mm (median error 35 / 24 / 10 mm). Verbatim: *"Geometry alone cannot hold
  biological identity — the project's core thesis, confirmed on round7."*
  `report/20260708/round7_bakeoff/DECISION.md`.
- **Slot wandering.** The female occupies all three raw 3D slots across windows in every
  experiment (`runs/dual_line/lineB_fulltimeline/FINDINGS.md`,
  `runs/dual_line/extend_k11_k13/FINDINGS_c0_gate.md`), and flips repeatedly *within* a single
  5,000-frame window (`runs/dual_line/lineB_w64000/FINDINGS.md`).

**Design implication:** the new repo must treat "3D geometry" and "biological identity" as two
different objects from day one. Geometry gives trajectories; identity must come from an independent
anchor. Never let a geometry slot index leak into a biological label without an explicit,
recorded assignment step.

## 2.2 Track-quality QC metrics are BLIND to identity swaps

Verbatim, `report/20260708/round7_bakeoff/DECISION.md`: *"Swaps trade two same-position fish,
leaving no speed signature, so swapped frames are still labeled `high`. **Do not use `id_conf` to
find swaps.**"* Independently: `scripts/experiments/tracklet_id_benchmark/FINDINGS.md` reports
**speed_spike = 0.0 for every method in every window** — the metric cannot discriminate.

Corollary from round8 QC (`report/20260730/round8_3d_tracking/round8_3d_quality_report.md`):
"0 steps >8 mm" is *expected* because the linker gate **is** 8 mm — it is *"bounded-continuity
evidence, not an independent identity proof."*

**Design implication:** budget for a real identity-error metric with human labels from the start.
Smoothness, coverage, residual and confidence are all diagnostics, never evidence.

## 2.3 Appearance closes the identity gap — but the classifier pipeline was contaminated three ways

**It works, in principle:** per-frame female-vs-male P-spread 0.911, global AUC **0.982**, and the
production dual-line track (M3) places the female **0.18–0.27 mm median** from an independent gold,
98.8–99.7% within 20 mm, with 100% biological validity (exactly 1 F + 1 M1 + 1 M2 every frame).
`runs/dual_line/phase1b/FINDINGS.md`, `report/20260708/round7_bakeoff/DECISION.md`.

**But all three of these bit:**

### (a) Crop distribution must match training — AUC 0.982 vs ~0.5
Same masks, same model, same 20 events; only the crop builder differed:

| metric | `33._square_crop` (bbox→resize, unmasked) | `crop_from_mask` (raw 128 px box, black mask) |
|---|---|---|
| within-frame F-vs-M P-spread | **0.911** | 0.025 |
| frac P(female) > 0.7 | 0.338 (≈ ideal 1F:2M) | 0.856 |
| global AUC | **0.982** | ~0.5 |

Confirmed by a no-Detectron2 control: shrinking *in-distribution* crops onto black reproduces the
collapse (baseline 0.908 / AUC 1.00 → shrunk-to-36 px 0.08 / 0.86).
`runs/dual_line/phase1a/FINDINGS.md` (superseded), `runs/dual_line/phase1b/FINDINGS.md`,
`docs/dual_line_tracking_plan.md` L16-19.

**Design implication:** assert train/inference crop-builder identity (framing + masking + resize)
as an automated **test**, not a convention. This one mismatch produced a full NO-GO decision that
was wrong.

### (b) Cross-window label swaps masquerade as classifier weakness
Label-noise-free embedder on K_1_2 side1 crops:

| | 3-way ARI | male-male ARI | male-male acc |
|---|---|---|---|
| within-window (avg of 4) | **0.82** | **0.67** | 0.87 |
| pooled across windows | **0.24** | **0.11** | 0.66 |

Cross-window nearest-centroid: Hungarian-**REMAP ≈ 1.00** while **RAW ≈ chance (0.00–0.41)**. That
gap is direct proof the 3D-bind `global_id` is **permuted between windows** — appearance persists
perfectly, geometry *naming* is swapped. After swap-correction, female LOSO went **0.78 → 0.99**;
male-vs-male went from "chance" (balacc 0.52–0.62, ARI ≈ 0) to **AUC 0.994** vs a position-only
baseline of **0.477**. `docs/per_video_id_plan.md`,
`runs/dual_line/lineB_fulltimeline/FINDINGS_step1b_male.md`.

⚠️ **Window 176300–181300 is a known "bad apple"** (3-way REMAP 0.71, male-male 0.58) — and it is
the window the production `model_female` leaned on, *and* the window used for the UDMT test *and*
the whole identity-recovery benchmark.

**Design implication:** never pool geometry-derived labels across time windows without an
appearance-bijection alignment step first.

### (c) Classifiers must be trained PER VIDEO, and validated by an external gate
The story reversed twice. Final position (`runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md`,
confirmed on the round7 backbone in `report/20260708/round7_bakeoff/DECISION.md`): the naive K_1_2
female model applied to K_1_1/K_1_3 **confidently mislabelled a MALE as the female in both**
(real female P≈0.07, a male P≈0.9), systematically at aligned-2 in both. See §4.2.

**Mandatory design rules extracted:**
- Per-experiment retrain (`model_female_clean_v2`, `model_male_clean_v2`), no cross-video transfer.
- **External gate: the called female must land in the BORIS spawn wrap (≤~12 mm from a male at the
  annotated spawn frame).** Post-fix: 0.6 mm (K_1_3), 2.6 mm (K_1_1).
- **Keep the tell:** if male-vs-male separability comes out *too high* (0.95–0.99 vs a genuine
  ~0.91), suspect that the female is mislabelled and the "male model" is really doing female-vs-male.

## 2.4 The `--num-candidates` flag silently selects a different algorithm

`AGENTS.md` L117-132 / `CLAUDE.md` L106-112:
- `--num-candidates 6` (> `--num-individuals 3`) → global mutually-exclusive candidate-set selector
  `triangulate_extra_candidates.py`, tqdm label **"3D best-assign"**, ~400–900 frame/s, ~10–15 min
  for 348k frames. **This is the path behind every validated metric.**
- Omitting it → `triangulate.py`, tqdm label **"3D candidates"**, ~10 frame/s, ~9.7 h — *"both far
  slower and a different algorithm that does not match the validated numbers."* ~60× difference.

**Design implication:** no silent algorithm fallbacks. If two algorithms exist, they get two
explicit subcommands and their outputs carry the algorithm name in provenance. Same class of bug:
`--max-residual-px2` (polynomial, px²) vs `--max-ray-residual-mm2` (refractive, mm²) — *"never
carry the historical polynomial value 5000 px² into a refractive run"*; `--max-residual-per-fish`
is a deprecated alias that silently means the polynomial one.

## 2.5 Mouth/tail node swaps → the visible-node-centroid anchor (and its own trap)

SLEAP round4 shows mouth↔tail keypoint swaps in the p99 tail. `--use-visible-node-centroid`
(centroid of visible mouth+tail) is **swap-invariant** and is the production choice.

Concrete failure mode of the alternative, `report/20260731/round8_mouth_tracking_ablation/COMPARISON_REPORT.md`:
at frames 34547→34548 the top-view mouth moved **0.2361 px** while the visible-node centroid moved
**56.2087 px** once the tail became unavailable — under the fixed 8 mm gate the straight
continuation became infeasible and the *crossed* continuation stayed feasible.

**But strict-mouth is not a safe replacement.** Full-length ablation A0 (centroid) vs A1/A2/A3
(strict mouth): female best-fixed **221/300 (73.7%)** for A0 vs 180 / 192 / **94** (31.3%) for
A1/A2/A3; identity segments 3 / 10 / 10 / **117**. A1–A3 are *smoother* (jumps ≥7.5 mm: ~166 vs
A0's 597; residual p95 1.02 vs 3.32 mm²) — verbatim: *"smoothness and calibration residuals are not
biological identity labels."* Decision: keep A0; A1 experimental; **reject A3, A4, A5**.

⚠️ Related loader bug: *"the prior loader silently fell back to the remaining visible node when the
requested mouth anchor was absent"* — with a tail-only instance population of 0.32% / 0.85% / 0.65%
(top/side1/side2). Any "mouth-anchored" result predating `--anchor-node mouth --require-anchor-node`
is partially tail-anchored. `report/20260716/mouth_tracking_ablation/THREE_WAY_TRACKING_COMPARISON.md`.

⚠️ **Window-initialized identity hides prior global label permutations:** window-initialized A1–A3
hit 9/9 female consistency on a held-out set, but the *same 9 frames* read from frame-zero-linked
tracks give only **6/9**, while A0 stays 9/9.

## 2.6 Association / linking: motion beats position; global smoothing is a trap

`runs/round7_full_qc/20251225_K_1_2_FCMC/heldout_association_f0_15000/association_benchmark_decision.md`:

| Method | held-out identity acc | switches | >8 mm | decision |
|---|---|---|---|---|
| existing pre-motion labels | 72.48% | 25 | 36 | baseline |
| conservative layered repair | **78.65%** | 10 | 9 | keep |
| global six-state Viterbi | **53.64%** | 6 | 0 | **reject** |

Verbatim: *"Global Viterbi demonstrates the central ambiguity: a trajectory can be perfectly smooth
while carrying the wrong biological identity."*

Anonymous association on clean mouth observations: position/residual-only Hungarian **47.29%** vs
velocity+tangent **88.79%**. Learned pair-cost weights (normalized to displacement = 1):
velocity-prediction error **0.93**, tangent discontinuity **2.90**, acceleration **0.80** — but the
acceleration term is **not justified** (fixed 1/2/0 weights give the same result), and the
max-reprojection and side-1 coefficients have counterintuitive signs from collinearity and **must
not be deployed**.

Also useful: centroid-to-mouth offset is **12.32 mm median / 14.92 mm max**; centroid is better
position-only (57.4% vs 47.3%) but mouth is far better motion-aware (88.8% vs 57.4%).

## 2.7 Segmentation should be a *confidence signal*, never a hard veto

- **Binary hard veto: REJECTED.** Fallback frames 12→4 looks good, but tracklet nodes **330 → 836**
  and resolved retained 2D rows **98.88% → 28.06%**.
  `runs/tracklet_first_identity/…/ablation_comparison_f10000-14999/FULL_WINDOW_ABLATION_REPORT.md`.
- **Three-state (support/unknown/contradiction): geometry-safe but insufficient.** The strict
  contradiction gate **never fired**; zero XYZ differences across 14,986 rows; consolidates 131→110
  components and resolves exactly one extra node. On held-out data none of the joined components
  reaches the seed identity set. Verbatim: *"Geometry is usable; biological identity propagation is
  not."* Window expansion was **paused**.
- Mask quality itself is fine (median consecutive-mask IoU 0.943/0.947).
- Verbatim design rule: *"use segmentation as confidence and a targeted veto — not as an
  unconditional row-deletion/tracklet-splitting rule"*; missing segmentation stays neutral.

## 2.8 Tracklet-first does not scale and does not generalize

Tracklet-first fixed the focused K_1_2 window (all-3 coverage 73.54% → 94.96%; resolves 122/143
components vs the prior pilot's 49/143). But: the full-length run **exceeded one hour, ~5.6 GB RAM,
and had written no final CSV when stopped**. On K_1_3 only **30 of 1,132 components resolve**
(57.6% of selected slots synthetic). On a held-out window, median fragment length is **4–7 frames**
and only **1.01%** of 2D rows resolve. Verbatim: *"a single clean seed frame cannot label hundreds
of disconnected components."*
`runs/tracklet_first_identity/TRACKLET_FIRST_IDENTITY_REPORT.md`,
`report/20260716/mouth_tracking_ablation/THREE_WAY_TRACKING_COMPARISON.md`,
`runs/tracklet_first_identity/…/ablation_comparison_three_state_f74720-79719/HELDOUT_BENCHMARK_SUMMARY.md`.

**The recommended successor formulation** (written but never executed) is in
`runs/tracklet_first_identity/…/identity_pipeline_audit_20260713/IDENTITY_PIPELINE_AUDIT_AND_PLAN.md`:
a constrained global fragment **path-cover** with explicit edge costs (displacement /
velocity-prediction / angle / reprojection / silhouette / appearance / gap), production gate
`min(8 mm × Δt, 80 mm)`, **multiple independently trusted biological anchors**, and unanchored
paths left **unlabeled**. Structural benchmark: 131→49 paths (dev), 540→110 (held-out), exact HiGHS
optimum, XYZ untouched. **This is the single best design carry-over in the repo.**

## 2.9 Blinded human audit is the only primary truth — and both audits failed everything

`runs/identity_recovery_benchmark_20260713/`. Decision rule locked **before** ratings
(`DECISION_RULE.md`). First audit (`portable_report_final_audit/evidence/FINAL_DECISION.json`):

| method | pos. recall @80 px | all-3 coverage | audited crossings passed | verdict |
|---|---|---|---|---|
| multi-view constrained contrastive | 0.9667 | 0.9026 | 2/7 | not recovered |
| idtracker.ai v6 | 0.8541 | 0.6221 | 3/7 | localization gate FAILED |
| native UDMT | 0.0733 | 0.0138 | 0/7 | localization gate FAILED |

Then **the audit itself was invalidated** (`AUDIT_REVISION_NOTICE.md`, 2026-07-14): clips used only
83 frames of context so many began after the fish were already close, and under a 100 px / 166-frame
sustained-separation rule the "7 crossings" were really **2 encounters**. Replacement
encounter-level audit (`portable_report_separation_context_final/evidence/FINAL_REVISED_DECISION.json`):
**every method 0/2 end-to-end successes.**

**Design implications:** (a) lock the decision rule before unblinding; (b) define encounter units by
a *sustained-separation* rule and start/end every clip in separation; (c) geometry-proxy IDF1 and
model confidence are diagnostics only. **Note that as of the last audit, NO method has passed a
blinded human identity audit.**

## 2.10 The contact regime is structurally absent from every training set

`runs/dual_line/phase0/FINDINGS.md`: **every clean-mask frame has 3D distance >20 mm; zero coverage
below 20 mm.** `docs/detectron2_appearance_classifier_test.md`: the concrete-tracklet gate removes
close-contact frames, so male crops are **always** separated — inter-male distance min **117 px**,
median 386 px, **0% closer than 100 px**.

Also: the male cannot-link graph **fragments into isolated pairs (largest component = 2)** because
tracklets break synchronously at crossings — constraint propagation alone cannot 2-colour the males.

**Design implication:** the decisive regime is the one every filter throws away. The new repo must
deliberately *collect and label* close-contact data instead of gating it out.

## 2.11 Calibration

**What is actually production (and the docs disagree — fix this immediately):**

| source | claims production is |
|---|---|
| `AGENTS.md` §Calibration Notes | `calibration_results/fixed_refractive_192x192x186/` (promoted 2026-07-15) ✅ operative |
| `CLAUDE.md` L206-210 | `cali_20251217_manual/…/imCoords_to_XYZ.joblib` (0.90 mm mean / 1.00 RMSE / 2.15 max, 228 pts) ❌ stale |
| `MEMORY.md` L51 | same stale polynomial model ❌ stale (MEMORY.md last updated 2026-06-06) |
| `docs/calibration.md` | nothing — 100 lines of GUI workflow only, no model, no accuracy, no limits |

**Production model accuracy** (`calibration_results/fixed_refractive_192x192x186/` manifest,
`schema_version: fixed_refractive_calibration_manifest_v1`, score unit **mm²**):
inner span 192×192 mm, water height 186 mm, glass 4 mm, n_water 1.3325, validated 25–26 °C.

| validation | n | RMS | median | p95 |
|---|---|---|---|---|
| training point-to-ray | 16,392 | 0.4697 mm | 0.2248 | 0.6623 |
| held-out wet-10 mm board | 2,100 | **0.2367 mm** | 0.1681 | 0.4480 |
| held-out wet-20 mm board | 1,440 | **0.4298 mm** | 0.3025 | 0.6769 |
| held-out rigid 3-view Kabsch | 13 poses | **0.7922 mm** | 0.5432 | 1.5634 |

Manifest guardrail, verbatim: *"Held-out rigid-object results validate shape and scale. A training
cube pose fixes the world gauge, so the manifest does not claim absolute surveyed-marker accuracy."*
Stage-2 interface corrections are **`"status": "disabled"`** in the shipped bundle.

**Error budget (later, better analysis):** on 12,369 complete three-view triplets, held-out
calibration RMS 0.436 mm vs propagated detector localization median 0.157 mm → **calibration is
88.6% of clean-track variance**. `runs/calib_12balls/physics_qc/REPORT.md`. ⚠️ This **reverses** the
earlier claim in `runs/calib_12balls/PHYSICS_MODEL_ANALYSIS.md` ("detection is the bottleneck,
4:1"), which used raw fish reprojection residual as detector sigma. **Both statements are still in
the repo — use the later one.** Multi-mm tail errors come from discrete tracking failures (swaps,
reflections), not Gaussian noise.

**Refinements tried:** 12-ball rigid-object polynomial (−34% Kabsch on held-out, wins 5/5 CV folds,
prescribed as a **hybrid** — refined forward model only, keep base reverse/pairwise because refined
reverse is 1.5–1.8× worse in pixels); projective DLT full-set (best held-out Kabsch, **0.720 mm**);
flat refractive (0.777 mm). **The shipped fixed-refractive bundle scores 0.792 mm on the held-out
rigid cube — i.e. slightly WORSE than the DLT comparator it was promoted over.** No head-to-head
write-up justifying that choice exists in `runs/`. The rationale must have been
transferability/physicality; **state it explicitly or re-run the comparison.**

**Known limitations to carry forward:**
- **Not valid for fine appendages (<2 mm).** Fin extension is 2–5 mm ⇒ error is 12–30% of the
  feature; individual fin rays and mouth gape "cannot be precisely localised in 3D"
  (`calibration_results/protocol.md`).
- **Z is weaker than X/Y** — constrained by the two side cameras only.
- **Water height was never measured.** The 186 mm plane and 192³ volume are provisional
  (`runs/calib_fullset/water_height_evidence/` holds only ruler frames, no unambiguous meniscus).
- **Shape/scale ≠ absolute localization.** Kabsch removes pose error, so it is a *lower bound*. The
  fit-free pairwise ruler only spans ≤86.6 mm. **No surveyed absolute-accuracy validation was ever
  acquired.**
- **Interpolation only** — held-out poses fall inside the training bounding box; polynomial
  extrapolation degrades rapidly. Refilming should prioritize upper far-wall edges near (192,192,192).
- Legacy bead cube has physical defects: `beads_way1_008` off by 1.4–2.2 mm in all 4 ways.
- Top camera is less trustworthy ("water ~1 cm low") — prefer side-driven triangulation
  (`report/20260630/K12_TRACKING_HANDOFF.md`).

## 2.12 SLEAP model status

**Production: Round 8**, tag `round8_260729_snn031_m6`. Rejected in its favour: Round 9b and every
side-optimization sweep candidate.

Two corrections to common belief:
- **There is no trained "Round 9" model.** "Round 9" was only a side-view *data split* that had 8
  historically-exposed validation frames; it was replaced by a leakage-safe Round 9b split
  (`report/20260728/sleap_side_round9b/README.md`). Only Round 9b was trained.
- **The decisive metric is OKS**, not mOKS or PCK. mOKS was only the internal checkpoint monitor.
  `report/20260729/sleap_side_round9b_snn031/REPORT_SOURCE_NOTES.md:67` warns the four metric
  families "must not be compared as if identical."

Rejection numbers (`report/20260729/sleap_side_round9b_snn031/final_summary.json`, 240 GT fish):

| candidate | matched-only OKS | per-GT OKS (miss=0) | Δ vs R8 [95% CI] |
|---|---|---|---|
| **Round 8** | **0.6760** | **0.6451** | — |
| Round 9b e32 | 0.5952 | 0.5704 | −0.0746 [−0.1065, −0.0420] |
| fresh mOKS e0 | 0.5653 | 0.5394 | −0.1057 [−0.1371, −0.0741] |

Round 7 → Round 8 was also a real gain (top OKS 0.751→0.805; side 0.599→0.689).

⚠️ **Caveats on the rejection:** the HITL-v2 benchmark (40 synchronized frames, 120 view frames, 360
human instances) is only *partially* held out — annotation used "human-reviewed Round8-model-assisted
coordinates," so *"subpixel localization may therefore favor Round8."* Round 9b's epoch was also
selected **on the benchmark itself** (optimistic, and it still lost). CIs are 200k paired cluster
bootstraps over only **40 independent moments** — descriptive, not population-randomized.

⚠️ **Counter-signal never resolved:** jitter-trained models are far more robust to crop-anchor
error. At a fixed ±8 px oracle offset, Round 8 collapses (OKS 0.269) while Round 9b holds 0.555–0.636.
Round 8 wins only because it is much better at the *correct* crop (0.7515 vs 0.5908). **Crop-anchor
robustness is an open design axis.**

⚠️ Round 8 is a **pilot-only fine-tune**: 20 held-out source frames + 93 usable training source
frames. The planned corrected 240-frame expansion package (`report/20260721/round8_active_learning/expansion_240.csv`)
was **never executed**. Manual visual QC ledger `manual_visual_qc.csv` is **0/60 reviewed**.

⚠️ `report/20260724/sleap_topdown_optimization/PHASE1_REPORT.md` is the audit that invalidated the
earlier round comparisons: *"The audit found no evaluation cohort genuinely unseen by both models"*
and *"Do not promote Round 7 or Round 8… from these numbers alone."* Also found an
annotation-reference bias: at a 0.01 px threshold Round 7's predictions **equal** the pilot
reference for 84/107 side mouths — the pilot was seeded from Round 7 predictions.

`docs/sleap.md` is a 9-line stub. `docs/sleap_retrain_session_handoff.md` (310 lines) is a **stale
Round-1-era brief** that contradicts current reality (single-node skeleton, max_instances=3, TRT) —
mark historical or delete.

---

# 3. Dead ends — do not repeat

| # | Thing tried | Why it failed | Date | Source |
|---|---|---|---|---|
| D1 | **TensorRT / ONNX export on RTX 5070 Ti (Blackwell)** | TRT export fails — `peaks_castOut` op has no TRT 10.16 tactic; ONNX gives no worthwhile speedup. Always `--mode track`. `phase6_execution_manifest.json` encodes `"tensorrt_prohibited_on_this_machine": true`. ⚠️ `MEMORY.md:95` still lists TRT under *Completed* — that line is stale and contradicts `MEMORY.md:49`. | ongoing | `AGENTS.md:41-43`, `MEMORY.md:49` |
| D2 | **Native UDMT (ST-Net / TrDiMP / DiMP), Detectron2-seeded, on saion** | Localization collapse: position recall **0.0733 @80 px**, all-three coverage **0.0138**, 0/7 then 0/2 audited encounters — it was never on a fish. Motion proxy detects ~95–99%. Failure is the native DiMP tracker, not the seed (exhaustively verified). | 2026-06-23 → 2026-07-14 | `runs/identity_recovery_benchmark_20260713/portable_report_*/evidence/FINAL_*DECISION.json`, `docs/udmt_saion_runbook.md` |
| D3 | **UDMT bidirectional repair pass** | No ID benefit. Raw `id_switches` roughly doubled (top 46→102) but that is an **evaluation artifact**: `repair_tracks_with_backward` only overwrites x/y of anomaly rows and **never changes `track_id`**. Restricted to untouched rows, forward == repaired in all 3 views. Report forward-pass only. | — | `docs/udmt_individual_tracking_test.md` |
| D4 | **UDMT motion proxy as an identity solution** | Detects all 3 fish ~95–99% but males **fragment** (2–3 tracks each) and the female swaps on side2. IDF1-like 0.60/0.80/0.76. Scored *by* the swap-prone 3D proxy, so male numbers mean "agrees with the 3D label", not "correct". Continuity aid only. | — | `docs/udmt_individual_tracking_test.md` |
| D5 | **idtracker.ai v6** | Self-report is high but unvalidated: `estimated_internal_accuracy = 0.9538`, best silhouette **0.7040 vs target 0.91 — not reached**, plus its own `"warning": "internal estimates are not independent biological identity accuracy"`. Independent scoring: position recall **0.8541**, all-3 coverage **0.6221**, geometry-proxy IDF1 **0.5988**. Blinded audit **3/7 → 0/2**; localization gate FAILED. ⚠️ The "96.2% self-reported / 58.7% agreement" figures from the memory notes are **not findable verbatim in this repo** (June-2026 logs are gitignored) — the closest in-repo numbers are 95.38% internal and 59.88% IDF1. | 2026-07-13/14 | `runs/identity_recovery_benchmark_20260713/idtracker_v6_run_summary/idtracker_run_summary.json` |
| D6 | **SLEAP Round 9b + all side-optimization sweeps** | Lost to Round 8 on held-out OKS by −0.075 to −0.106 (see §2.12). `sigma15__control` had the best localization but detect 0.938 < R8's 0.954. The crop448/rot90/strict-mouth/OHKM-off bundle improved *oracle* tail p95 but **dropped end-to-end OKS**. Crop-position jitter ±8/16 rejected earlier. | 2026-07-25 → 07-29 | `report/20260727/sleap_side_optimization_hitl_v2_evaluation/EVALUATION_REPORT.md`, `report/20260729/…/final_summary.json` |
| D7 | **sleap-io 0.8.0 for the annotator bundle** | The bundle's unpinned `sleap-io` let a fresh `uv sync` pull 0.8.0 while the pipeline+seeds were built on 0.7.0. Under 0.8.0 the interactive **save writes a view-scramble** into `edits.pkg.slp` (side1←side2, side2←top; keys stay valid so no orphans), which auto-resume then reloads and re-saves, **propagating the corruption**. Fixed by pinning `sleap-io==0.7.0`. ⚠️ Repo is now internally inconsistent: bundle pins 0.7.0 but `environment/requirements.txt:54` says `sleap-io>=0.8.0,<0.9.0` and the 2026-07-29 training ran on **sleap-io 0.9.2**. **Re-validate the save→reopen round trip on any new version.** | 2026-07-04 | commit `bf62361`, `scripts/sleap/make_annotator_bundle.py:70-73` |
| D8 | **M2 — SAM2-segmentation-assisted identity** | **Strictly dominated.** Seed identities *are* the geometry `final_fish_id`, so it also produces arbitrary IDs. SAM2 anchor covers only **2,442 frames (0.7%)**; bind falls back to geometry 99.3% of the time. Implied-female output is **byte-identical to the geometry baseline M1** on all three experiments. Even a full stride-1 rebuild caps at ~7% coverage. | 2026-07-08 | `report/20260708/round7_bakeoff/DECISION.md` §M2 |
| D9 | **Segmentation binary hard veto** (row deletion + tracklet splitting) | Resolved retained 2D rows **98.88% → 28.06%**; tracklet nodes 330→836. Superseded by three-state, which is safe but adds ~nothing. | 2026-07-13 | `…/ablation_comparison_f10000-14999/FULL_WINDOW_ABLATION_REPORT.md` |
| D10 | **Temporal-geometry annotation of disconnected components (variants D/E/F)** | D changed **11,109 XYZ rows** and produced 2,815 conflict rows; E and F preserve XYZ but produce 3,138 / 3,398 conflicts. *"Local geometry IDs are not reliable biological IDs across the whole window."* Retained only behind a disabled-by-default flag. | 2026-07-13 | `…/ablation_comparison_three_state_f74720-79719/HELDOUT_BENCHMARK_SUMMARY.md` |
| D11 | **Global six-state Viterbi identity smoothing** | 53.64% held-out accuracy vs 78.65% for conservative layered repair, despite 6 switches / 0 jumps. Perfectly smooth, wrong identity. | — | `heldout_association_f0_15000/association_benchmark_decision.md` |
| D12 | **Windowed/checkpointed full-recording stitcher ("Item 1") and Items 2–6** | Engineering wins, scientific losses. Item 1: 348,600 frames in 575 s, 1.63× faster, all seams exact — but female best-fixed **0.4276 vs 0.5235**, female-ID changes **81 vs 11**, all-3-direct **0.8943 vs 0.9816**; **84.63% of 92,426 fragments are singletons**. Item 2 (multiframe cross-view grouping): collapses all-3-direct coverage **84.8% → 0.3%** — "antagonistic". Item 3: 378 links compress 491 fragments into **113 restart segments, not three fish**. Item 4 (tail vectors): 6 eligible edges, **none selected**, zero decisions changed, and it breaks the frozen-graph hash contract. Item 5 (exact MILP): only a 310-frame feasible slice; the 1,000-frame graph is exact-three **infeasible**. Item 6 (deferred decisions): a no-op after an offline global solve; with Item 2 leaves **744 of 840 identity rows blank**. | 2026-07-17 | `report/20260717/windowed_tracklet_stitching/{optimized_baseline_report.md, final_recommendation.md, item*_verifier_report.md}` |
| D13 | **Blind mask dilation for fin-inclusive appearance** | Radius 2 adds ~22% mask area with **no change** in SAM agreement (IoU 0.775→0.776) while 57–63% of added pixels lie outside the SAM reference. r12 is clearly worse (IoU 0.403). Keep body-only masks. | 2026-07-13 | `…/identity_pipeline_audit_20260713/IDENTITY_PIPELINE_AUDIT_AND_PLAN.md` |
| D14 | **Cross-video appearance embedder for individual re-ID across trials** | Sex-inconsistent grouping — it pairs the K_1_1/K_1_3 female with a K_1_2 *male*. "A fish cannot change sex across trials." Embedder coherence saturates at 1.0 (flagged uninformative in phase 0). Verdict INCONCLUSIVE. | 2026-06-26 | `runs/dual_line/extend_k11_k13/FINDINGS_N3_mate_choice.md` |
| D15 | **Side-trained appearance models applied to top-cam crops** | Collapse: top female P-spread **0.018**, male **0.011** (vs side 0.91 / 0.52); "agreement" 0.49–0.72 is chance-level noise. Fix is a **top-inclusive model auto-labeled from the validated side track**, not transfer. | — | `runs/dual_line/lineB_fulltimeline/FINDINGS_step3_top.md` |
| D16 | **Geometry-contrastive appearance embedding as an ID system** | AUROC only **0.7413** (CI 0.697–0.785), EER 0.3315; at the frozen 1%-FAR point TAR is **0.3502**. Confident on **5 of 170** candidate edges; XYZ exactly unchanged; no structural coverage gain. Decision matrix: 3× FAIL_POSITIVE, 1× UNAVAILABLE, execution BLOCKED (1 of 3 experiments). | 2026-07-13 | `runs/appearance_geometry_contrastive/multiexperiment_20260713/{CONCLUSION.md, decision.json, DECISION_MATRIX.csv}` |
| D17 | **`tracklet_bind` / `hybrid` confidence gating** | Over-flags: up to **100% unknown** on a suspect window, reconnection 0.03. And all geometry-derived "methods" share labels (`label_agreement = 1.00`) — they are one tracker with three confidence policies, not four trackers. | — | `scripts/experiments/tracklet_id_benchmark/FINDINGS.md` |
| D18 | **6 mm vs 8 mm jump gate** | **Inconclusive; the earlier "prefer 8 mm" was withdrawn.** 6 mm: best-fixed-ID 44.63% vs 52.35%. But only 14/26 gate branches meet even the structural proxy and the cohort is observation-quality confounded. Bidirectional tangent audit is mixed. | 2026-07-18 | `report/20260718/k12_jump6_strict_mouth/RESULTS.md` |
| D19 | **Legacy 1.5-body-length tracklet separation gate** | Retained only **1.9% / 12.6% / 10.8%** of fish-frames. Replaced by 0.5 body lengths (76–85% retained). | — | `report/20260716/mouth_tracking_ablation/THREE_WAY_TRACKING_COMPARISON.md` |
| D20 | **GPU port of endpoint/triangulation kernels** | Kernels are 100× faster in isolation (0.0798 s → 0.0005 s for 1M endpoints) but the pipeline is dominated by cache/window orchestration, triangulation setup, merge and I/O — **no end-to-end win**. Also 16 workers is slower than 8. | 2026-07-17 | `optimized_baseline_report.md` |
| D21 | **UDMT's in-repo adapter as a tracker** | The `UDMT/` package contains only post-processing (score/refine/repair/crossings/resolve); the real ST-Net/DiMP stack is Linux/py3.8-only and its pinned torch 2.1.1/cu121 cannot run on Blackwell at all. | — | `docs/udmt_individual_tracking_test.md`, `docs/udmt_saion_runbook.md` §0 |
| D22 | **3D→2D reprojection tier-2 swap adjudication** | Unavailable — **13/20 events flagged `reproj_uncalibrated`**: a coordinate offset between the round5 3D track and the calibration model means the appearance-female does not land on reprojected ID_2 even when the fish are separated. Still open. | 2026-06-23 | `runs/dual_line/phase1a/FINDINGS.md` |
| D23 | **Refractive bundle-adjustment with jointly-fit intrinsics** | Did **not converge** — joint intrinsics + refraction is ill-conditioned; plus a camera-convention bug in back-projection (30 mm error even with refraction disabled). Fixed by fixing dry intrinsics + measured pane geometry and fitting only extrinsics/poses. Wet-only parameter identifiability remains a named unresolved limitation. | 2026-07-09/10 | `runs/calib_12balls/PHYSICS_MODEL_ANALYSIS.md`, `runs/calib_12balls/physics_qc/REPORT.md` |

---

# 4. Known bugs and data-integrity issues found late

Ordered by scientific damage.

## 4.1 Crop-builder mismatch (`crop_from_mask` vs `33._square_crop`)
**Corrupted:** every Phase-1a measurement. Produced a **false NO-GO** decision ("the appearance head
does not generalize") that would have forced an unnecessary full-timeline retrain and blocked the
saion fan-out. **Magnitude: AUC 0.982 vs ~0.5.**
**Found:** 2026-06-23, same day, by re-running with the production builder.
**Downstream:** `runs/dual_line/phase1a/FINDINGS.md` is **superseded in full**. `phase1a/p1a_detect.py`
is explicitly marked "a latent bug"; `phase1b/p1b_detect.py` is the corrected reference.
`docs/dual_line_tracking_plan.md` L175-205 carries the correction.

## 4.2 Cross-video female mislabel in K_1_1 and K_1_3 (worst scientific bug)
**Corrupted:** the round5 dual-line tracks for K_1_1 and K_1_3 — a **MALE was confidently labelled as
the female in both**, systematically at aligned-2. The real female scored P≈0.07, a male P≈0.9. The
C0 gate reported "GREEN on all three lines" with female P 0.922/0.822 — that was **the model's
confidence in the wrong fish**.
**Detection:** the called female sat **65–245 mm** from the spawning pair at the BORIS spawn frame
(108 mm / 120 mm in the round7 re-test).
**The tell — keep this heuristic:** male-vs-male LOSO AUC came out **too high** (K_1_3 0.99, K_1_1
0.95 vs a genuine male-male ~0.91) because with the female mislabelled, the "male model" was really
separating female-from-male.
**Found:** 2026-06-26, during the N=3 extension (commit `c15be12`).
**Fix + now-mandatory gate:** per-experiment `model_female_clean_v2` / `model_male_clean_v2` + the
female must land in the BORIS spawn wrap (post-fix 0.6 mm K_1_3, 2.6 mm K_1_1).
**Downstream:** `runs/dual_line/extend_k11_k13/FINDINGS_c0_gate.md` conclusions are **invalidated**.
`report/20260630/README.md` states it plainly: *"the earlier C0 male-male AUCs (0.948 / 0.990) are
not shown as separability: they were the tell of a female-mislabel bug."* **Every K_1_1 and K_1_3
number published before 2026-06-26 is suspect.** Round7 confirmed the mislabel is an
**appearance-model** limitation, not a backbone one.

## 4.3 Cross-window `global_id` permutation in the 3D-bind labels
**Corrupted:** the entire first generation of appearance models and their reported metrics — pooled
3-way ARI 0.24 (vs within-window 0.82), male-male ARI 0.11 (vs 0.67), and the conclusion
"appearance is a female anchor only" in `docs/detectron2_appearance_classifier_test.md`.
**Found:** 2026-06-23 via the RAW-vs-REMAP diagnostic.
**Downstream:** all metrics in `docs/detectron2_appearance_classifier_test.md`
§"Not achieved — distinguishing the two males" and the `model_female`/`eval_male` reports are
superseded by the swap-corrected models (`43_build_corrected_segid_dataset.py`,
`sam2_id_experiment/src/id_model/window_align.py`, commit `ced9fbb`). **Window 176300–181300 is
contaminated** and is auto-dropped — but it is also the window used for the UDMT test *and* the
identity-recovery benchmark.

## 4.4 The appearance training dataset is circular and leaky (audit 2026-07-13)
`runs/tracklet_first_identity/…/identity_pipeline_audit_20260713/IDENTITY_PIPELINE_AUDIT_AND_PLAN.md`:
- **19,887 corrected crops, all `side1_cam`, 0 side2, 0 top ⇒ no valid cross-camera test.**
- Conservative sample audit: **0 clean / 18,962 ambiguous / 925 reject.**
- **457 duplicate-mask→same-tracklet/frame groups (925 crops)** = reflection / secondary-blob
  label contamination.
- **Severe scene/time leakage:** 4,403 of 4,421 female validation crops and 3,041 of 4,038 male
  validation crops share frames with training.
- The "corrected" labels were produced by an appearance model run over the **same crop pool before
  splitting** ⇒ circular.
- The builder **discarded** mask index, bbox, mask score, link mode, mask path, source video and the
  exact command — provenance cannot be reconstructed without replaying the historical linker.
- **Explicit withdrawal, verbatim:** *"The historical appearance metrics must also be withdrawn as
  biological-ID evidence."* The male 0.884/1.0 balanced accuracies "measure recovery of
  appearance-derived targets under a leaky random-tracklet split"; the top-camera 0.826 is
  "side-model teacher agreement."
**Downstream:** every headline appearance number in `docs/detectron2_appearance_classifier_test.md`,
`docs/per_video_id_plan.md`, all `runs/dual_line/*/FINDINGS.md`, and the M3 bake-off inherits this.
The M3 *biological* conclusions survive only insofar as they are anchored on independent evidence
(spawn landing, gravid montage, BORIS), **not** on classifier metrics.

## 4.5 sleap-io 0.8.0 annotator view-scramble
**Corrupted:** `edits.pkg.slp` annotation files — the interactive save wrote side1←side2, side2←top.
Keys stayed valid so there were **no orphans and no error**; auto-resume reloaded and re-saved,
**propagating** it. Presented as "side1↔side2 flipped + edits lost" in round 6.
**Found:** 2026-07-03/04. The seed was always correct (side1 = cam1, 0.0 px); only the live save
under 0.8.0 was wrong. **Not reproduced headlessly** — the environment difference is inferred.
**Fix:** pin `sleap-io==0.7.0` in the generated bundle (commit `bf62361`).
**Downstream:** any round-6 annotation saved before the pin. ⚠️ Still inconsistent today:
`environment/requirements.txt:54` says `sleap-io>=0.8.0,<0.9.0`; the 2026-07-29 training ran on
**sleap-io 0.9.2** under Python **3.13.11** (repo standard is 3.11, same as CI). **Re-validate the
save→reopen round trip before trusting any bundle on a new version.**

## 4.6 Entry-phantom: reflection FPs + unbounded interpolation
Reflection/duplicate SLEAP false positives during the staggered fish-entry period seeded phantom 3D
slots that an **unbounded linear fill bridged across thousands of empty frames**, drawing a fish
before it entered the tank — a continuous **2,224-frame** phantom on K_1_2.
**Found & fixed:** commit `3026930`. Fixes: geometric FP rejection at ingestion in `load_2d_slp`
(collapsed instances, aligned/flipped duplicates) + `max_gap_frames` cap and an `is_interpolated`
provenance column in `fill_missing_3d`.
**Downstream:** any 3D track produced before `3026930`. The commit states *"mate-choice conclusions
unaffected (phantom is pre-courtship)"* — **verify that claim rather than inheriting it.**

## 4.7 Velocity gate scaled without a ceiling → accepted the wrong fish after long gaps
The per-frame velocity gate scaled linearly with gap: at gap = 95 frames and `max_jump_mm = 30` the
effective threshold was **2850 mm — larger than the tank**. On re-acquisition after a long NaN
stretch the Hungarian matched a *different* fish (observed: ID_3 "jumped" 237 mm).
**Fixed:** commit `456d49e`, adding `max_jump_hard_cap_mm` (effective gate = `min(max_jump_mm × gap,
hard_cap)`).
**Downstream:** every 3D track built before this commit. Current production values are
`--max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100`.

## 4.8 SLEAP loader silently substituted the wrong anchor node
*"the prior loader silently fell back to the remaining visible node when the requested mouth anchor
was absent"* — tail-only instances are 0.32% / 0.85% / 0.65% and mouth-only 1.90% / 2.45% / 2.98%.
**Any "mouth-anchored" result predating `--anchor-node mouth --require-anchor-node` is partially
tail-anchored.** Also noted: *"the current provenance records calibration but not all tracking
options."* `report/20260716/mouth_tracking_ablation/THREE_WAY_TRACKING_COMPARISON.md`.

## 4.9 Spline-fill clipping defect vs `--no-shift-to-positive`
Before the fix, spline gap-filling **clipped coordinates and silently violated the flag**
(`report/20260730/round8_3d_tracking/round8_3d_quality_report.md`). Related: the default
shift-to-positive step **chose different arbitrary output origins for the 6 mm and 8 mm runs**, so a
constant translation had to be removed before the two could be compared at all
(`report/20260718/k12_jump6_strict_mouth/RESULTS.md`).

## 4.10 Human-audit context defect (invalidated a completed blinded audit)
The 21-clip blinded audit used only **83 frames of context**, so many clips began after the fish were
already close; and under a sustained-separation rule the "7 independent crossings" were really **2
encounters**. The entire pre-audit report and the 6/7 decision rule were superseded.
**Found:** 2026-07-14, `runs/identity_recovery_benchmark_20260713/AUDIT_REVISION_NOTICE.md`. Cost: a
full blinded audit round had to be re-rendered and re-rated.

## 4.11 Round8 identity-segment continuity defect
A full-recording defect where a row carried `identity_segment = 0`, *"so a retired label could
silently resume as an existing identity"*, plus a denominator defect, an A5 schema defect and a
blocking enumerator defect. Caught before scoring/freeze.
`report/20260731/round8_mouth_tracking_ablation/INDEPENDENT_AUDIT.md` L1018-1022.

## 4.12 `split_pkg_by_view` view-classification bug
The view classifier *"returned the wrapper path (no view tag) instead of descending to the mp4"*, so
views were mis-bucketed. Fixed by descending the JSON chain to the leaf mp4
(`scripts/sleap/split_pkg_by_view.py:81-86`, `_classify_path`). **Corrupted:** round-5 training data
splits. Found during round-5 prep (2026-06-11).

## 4.13 NaN `time_bin` in the frame miner
`write_report` in the training-frame miner crashed/mis-binned on NaN time bins; found and fixed
during round-6 seed construction (2026-06-16). Affects the *selection* of mined training frames, not
their labels. Current code guards with `n_bins = max(1, n_time_bins)`
(`scripts/sleap/mine_3d_bad_frames.py:168`) and a clamped bin index
(`scripts/sleap/mine_round8_hard_frames.py:1239-1240`).

## 4.14 Stale configuration and documentation pointing at retired artifacts
These are live in the tree right now and will mislead anyone starting from the docs:
- `CLAUDE.md:206-210` and `MEMORY.md:51` name the **retired polynomial** calibration as production;
  `AGENTS.md` names the fixed-refractive bundle. ~6 weeks stale.
- `MEMORY.md:95` lists "TensorRT export path for ~2.5× inference speedup" under **Completed**, while
  `MEMORY.md:49` says TRT is broken on this GPU.
- `MEMORY.md:35` and `AGENTS.md:10` reference `src/medaka_pipeline/`, which **was removed** (commit
  `a0f536d`).
- `MEMORY.md:117` points at a memory index under a `c--Users-Makoto-Hiroi-…` path — wrong user.
- `environment/requirements.txt:53-54` pins `sleap-nn==0.3.0` + `sleap-io>=0.8.0,<0.9.0`;
  `pyproject.toml:34` says `sleap-nn==0.3.1`; the training configs stamp `sleap_nn_version: 0.3.0`;
  `scripts/sleap/finalize_round8_labels.py:372` hardcodes `"0.3.0"`. Only the report says 0.3.1.
- CI reproduction on Python 3.11 + `environment/requirements.txt`: **687 passed, 9 failed** — all 9
  are a pre-existing **`pyarrow` dependency gap** (tests call Parquet I/O; requirements omits
  pyarrow). Still open.
- ⚠️ **`pytest` is not installed in `.venv`** on this machine (noted in the storage-consolidation
  memory) — tests may not have been run locally in a long time.
- `docs/calibration.md` documents no model, accuracy or limitation. `docs/sleap.md` is a 9-line stub.
  `docs/sleap_retrain_session_handoff.md` is a stale Round-1-era brief that contradicts current reality.

## 4.15 Data-loss hazard (UDMT), if that code is ever revived
`UDMT`'s `create_new_project` calls `os.remove(video)` if `VideoReader` cannot open the file — **it
deletes the source video** (`new.py:157`, per `docs/udmt_saion_runbook.md` §2). Must use
`copy_videos=True` against scratch copies, never a `/bucket` master. Also `set_ini_value` **silently
drops** any fish whose seed centroid lands on a background pixel.

## 4.16 Operational gaps that blocked verification
- **Node/npm/bun/deno unavailable on this machine**, so the canonical portable-HTML report builder
  cannot run. Recorded at least four separate times as a blocker
  (`report/20260716/female_sex_gt_consistency/REPORT_PACKAGING_BLOCKER.md`,
  `report/20260718/…/REPORT_BLOCKER.md`, and two others).
- `.pytest-tmp-*` directories in the repo **cannot be removed** — Windows ACL denies the repo owner.
- Video decode over the `Z:` share runs **~4 fps** vs ~100 fps locally; script `33`
  gzip-decompresses a whole 43,575-frame mask chunk per window (fixed by pre-slicing: 772 s → ~95 s
  per view, byte-identical crops).
- `20260115_N_2_3_FCMC` raw mp4 is **corrupt/unfinalized** (no moov atom, no metadata.csv) — the one
  experiment that could not be inferred in the round7 full-dataset run.

## 4.17 Gates that were opened and never closed
These are results that were *reported* but whose validation step was never executed:
1. **12-ball calibration hybrid downstream non-regression on fish tracks** — verbatim *"Downstream
   non-regression on fish tracks is NOT yet verified."* Four named criteria (ID-swap agreement at
   BORIS events, mate-choice reproduction 0.61/0.71/0.80, residual/speed-spike, eye review) — none run.
2. **DLT calibration adoption rule** required stratified windows on all three experiments; **only
   K_1_2 was ever run.** The DLT-scored-assignment arm (changed 4.58% of frames) was deferred and
   never done.
3. **No head-to-head justifying fixed-refractive over the DLT comparator** that beat it on held-out
   Kabsch RMS (0.720 vs 0.792 mm).
4. **SLEAP Phase-6 fixed-clip 3D + identity-QC promotion gate** — the folder contains only a
   36-step execution manifest. **No results, no report.** Self-labelled
   `scientific_status: engineering_and_known_failure_gate_not_independent_promotion`.
5. **The genuinely-locked multi-experiment SLEAP test** recommended by Phase 1 (111 eligible view
   rows / 37 synchronized frames from 21 K_1_2 events) was **never annotated**.
6. **The corrected round8 expansion-240 annotation package does not exist**; Round 8 is a pilot-only
   fine-tune with a `manual_visual_qc.csv` that is **0/60 reviewed**.
7. **Contact-episode human GT** (`report/20260719/contact_episode_gt/`): only **32 of 366 episodes
   annotated (8.7%)**, and **100% of those are marked `contact_validity = uncertain` and
   `tracker_assessment = uncertain`.** Do not cite it as validation — it is the open workstream.
8. **Water height never measured** ⇒ absolute Z and the 192³ working volume are provisional.

---

# 5. Open questions the new repo should answer first

Ranked by (value × cheapness) and by how much downstream work is blocked.

**1. Preserve the evidence before anything else.** `runs/` is gitignored and exists only on the old
machine. Copy `runs/dual_line/`, `runs/calib_*/`, `runs/identity_recovery_benchmark_20260713/`,
`runs/tracklet_first_identity/` and `annotations/` into durable storage (Z: archive) with a manifest
of checksums. Zero science, total insurance.

**2. Can the study conclusion be reached WITHOUT tracking?** The `female_accept` field in the lab
sheet plus the BORIS courtship asymmetry appear to agree in **7/7** experiments (§1.7) — an N=7
result from two human record sources, no ML. **Verify this derivation, check annotator
independence, and if it holds, it is the strongest and cheapest result in the project.** It also
tells you how much tracking precision the paper actually needs.

**3. What is the female-identity error rate, per regime, against real human labels?** Every claim
in §1 is a function of this one number, and it has never been measured in the close-contact regime.
Build a stratified human GT set (≥200 frames, half from <12 mm episodes, labelled on raw video with
no tracker overlay). Until this exists, §1.2 and §1.4 are not reportable.

**4. Re-derive the association index end-to-end on a clean rebuild, with a real statistical test.**
Frames at 166 fps are massively autocorrelated; there is **no p-value anywhere in the repo**. Use
block bootstrap or bout-level units. Report the full 30/20/12/8 mm sweep, not one number.

**5. Does the result generalize beyond the K group?** Everything rests on three experiments recorded
on one day. Pick ≥3 experiments from different groups and dates that have both BORIS and usable
video, and run the whole chain. **If it does not replicate outside K, nothing else matters.**

**6. Which calibration is production, and does the choice change any biological number?** Resolve
the `AGENTS.md` vs `CLAUDE.md`/`MEMORY.md` contradiction, then close the never-closed gate: run the
same experiment through the polynomial, 12-ball-hybrid, DLT and fixed-refractive backends and show
the association index is invariant. If it is not, calibration is a confounder.

**7. Can M1 vs M2 be established at all, or should the paper avoid per-male identity?** There has
never been a male-vs-male ground truth. The letter is explicitly provisional; the two males are not
visually distinguishable by eye. **Decide early whether the paper needs per-male identity** — the
"preferred vs other" framing (which only needs the female anchor plus a consistent partner
assignment) may be sufficient and is far more defensible.

**8. What identity architecture actually works?** No method has yet passed a blinded human audit
(§2.9, all 0/2). The best-designed unexecuted proposal is the **constrained global fragment
path-cover with explicit edge costs and multiple independently trusted biological anchors, leaving
unanchored paths unlabeled** (`identity_pipeline_audit_20260713/IDENTITY_PIPELINE_AUDIT_AND_PLAN.md`).
Build that, and design the blinded-audit protocol (locked rule, encounter units by sustained
separation, clips starting in separation) **before** running it.

**9. Collect the data the appearance model actually needs.** Every existing crop dataset is
side1-only, contact-free, leaky and circular (§4.4, §2.10). A fresh dataset must span all three
cameras, deliberately include <20 mm frames, split by *time window* not randomly, and record full
provenance (mask index, bbox, score, source video, command).

**10. Is the courtship:aggression ratio real or a tracker artifact?** It moved 24.5× → 15.2× between
two backbones on the same data (§1.4). Define the behaviours against BORIS categories, measure the
proxy's agreement with human coding, and sweep the thresholds.

**11. Answer README Goal 2 — early choice vs post-aggression choice.** This is a stated research
goal that no analysis has addressed. It needs the temporal preference profile (§1.8) aligned to the
timing of male-male aggression bouts, with a changepoint test.

**12. Fix the reproducibility surface.** Pin one sleap-nn/sleap-io/Python combination and enforce it
in CI; add `pyarrow` (9 tests currently fail on it); install `pytest` in the venv; delete or mark
the stale docs (`docs/sleap.md`, `docs/sleap_retrain_session_handoff.md`, the `src/medaka_pipeline`
references, the TRT line in MEMORY.md); and add automated tests for the two silent-fallback classes
that caused the worst bugs (crop-builder identity; algorithm selection by flag).
