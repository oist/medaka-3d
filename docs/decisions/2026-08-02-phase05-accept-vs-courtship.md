# 2026-08-02 — Phase 0.5: can the conclusion be reached without tracking?

**Verdict: no.** The cheap route does not replace tracking, and the reason is not
sample size — it is that the two "independent" record sources are one source.

**Reproduce:** `.venv\Scripts\python.exe tools/phase05_accept_vs_courtship.py`

---

## Read this before the numbers

**n = 3.** The rows that can be scored come from groups **E, K and O** — three
trios of fish. Days and trials within a group are repeated measures of the same
three animals, so the nine scorable experiments are **three independent units,
not nine and not eight**. Any statement of the form "8/9" or "7/8" is a count of
repeated measures and must not be reported as a result.

Three trios is not enough to conclude anything about mate choice, whichever way
the numbers fall.

---

## 1. The headline check, re-derived

Does the male the female accepted also court her more?

| experiment | accept | courtship events m1 : m2 | courtship seconds m1 : m2 |
|---|---|---|---|
| 20251220_E_1_1 | m1 | 3 : 2 ✓ | 6.8 : 3.0 ✓ |
| 20251220_E_1_2 | m1 | 3 : 2 ✓ | 6.3 : 3.0 ✓ |
| 20251220_E_1_3 | **m2** | **6 : 5 ✗** | **7.4 : 7.7 ✓** |
| 20251225_K_1_1 | m2 | 1 : 36 ✓ | 1.8 : 73.8 ✓ |
| 20251225_K_1_2 | m2 | 7 : 45 ✓ | 13.1 : 71.0 ✓ |
| 20251225_K_1_3 | m2 | 4 : 44 ✓ | 5.0 : 71.7 ✓ |
| 20260109_O_1_1 | m1 | 18 : 0 ✓ | 27.0 : 0.0 ✓ |
| 20260109_O_1_2 | m1 | 13 : 0 ✓ | 19.4 : 0.0 ✓ |
| 20260109_O_1_3 | m1 | 21 : 0 ✓ | 35.6 : 0.0 ✓ |

By event **count**: 8 of 9. By event **duration**: 9 of 9. The prior claim of
7/8 or 7/7 differs only in which rows were in scope and which quantity was used.

**Every one of those framings is wrong to report, because the unit is the group:
n = 3.**

### K_1_2's `female_accept` is "m2m1", and resolves to m2

`boris_events.csv` records `m2m1`. The `lay_eggs_table.csv` legend fixes the
reading — *"during the egg-laying, the second wrap male will not be regarded as
'accepted'"* — so the first male named is the accepted one, m2. The experimenter
resolved it the same way: `lay_eggs_table.csv` independently records `m2` for
that row. Excluding this row instead of resolving it is what turns 8/9 into 7/8.

---

## 2. The one miss is the interesting row, and it is metric-dependent

**Group E accepted m1 on days 1 and 2, then m2 on day 3.** Preference is not
stable across three consecutive days within a single trio. K and O are internally
consistent; E is not. One of three trios switched.

E_1_3 is also the only row where the answer depends on how courtship is measured:

- by event **count**, m1 courted more (6 vs 5) and the row is a **miss**;
- by event **duration**, m2 courted more (**7.7 s vs 7.4 s**) and the row agrees.

A 0.3-second margin across a 2100-second recording, and a one-event margin, are
both within the resolution of a single annotator decision. This row is
**undetermined**, not a success and not a failure. It should not be scored either
way, and a metric chosen after seeing which way it falls is not a measurement.

This is the row that most needs tracking, and it is exactly the row the cheap
method cannot resolve.

---

## 3. Independence — the crux, and the reason the answer is "no"

`PRIOR_FINDINGS.md` §1.7 treats the lab sheet's `female_accept` and the BORIS
courtship counts as *"two human record sources"* and a *"third, human-recorded
channel"*. They are not. Three lines of evidence, all from the files themselves:

1. **The two sheets are the same numbers.** For all 7 experiments populated in
   both `boris_events.csv` and `lay_eggs_table.csv`, the courtship counts are
   **identical, 7/7** — as are the wrap counts, save one transcription
   discrepancy at K_1_3 (`m2` wrap: 2 in one sheet, 1 in the other).
   `lay_eggs_table.csv` is a partial copy of the BORIS aggregation, not a second
   observation.

2. **`female_accept` is derived from the same coding pass.** It is defined as
   "accept wrap", and the wrap events live in the same BORIS file as the
   courtship events. Taking simply *which male wrapped first*, `female_accept` is
   reproduced in **8 of 9** experiments. The single exception is K_1_2, where m1
   wrapped first but m2 is recorded as accepted — consistent with the sheet's
   note that the wrap column includes *attempts*, so the annotator judged which
   wrap succeeded. That judgement is another read of the same video by the same
   coder.

3. **Three of the 26 BORIS files were coded on the recording day itself**
   (E_1_1 on 2025-12-20, E_1_2 on 2025-12-21, K_1_1 on 2025-12-25); the rest were
   coded in later waves (2026-01, 03, 04, 07). Same-day coding is the experimenter
   working from the live session, which is also when the egg record was filled.
   For those rows the two "sources" are one person, one occasion.

**Conclusion: two behaviour categories scored by one annotator in one pass over
one video. The concordance measures the internal consistency of a coding scheme,
not the agreement of independent channels.** Under a fair coin it is not p =
0.0078; the events are not independent and no p-value of that form is available.

---

## 4. The egg record cannot rescue it

The genuinely independent channel is the physical egg outcome — it is a thing
observed in the tank, not a video judgement. It does not discriminate between
males:

- `eggs` / `eggs_after` in `lay_eggs_table.csv` are **per experiment**, yes/no.
- `Egg_Laying_by_pair.csv` is **per group percentages** (`Group, no, yes`) —
  despite the name, no male-pair breakdown.
- All nine scorable rows have `eggs = yes`, so within the scored set the field
  has no variance at all.

The egg record confirms **that** a mating occurred. Nothing in the sheets records
**which** male fertilised the eggs. Without paternity data, the physical channel
is silent on choice.

---

## 5. Join defects — every failing row, none dropped

Validated with `cohort.ExperimentId.parse` against the 69 parseable folders in
`raw_data_dir` (71 directories including two `_P2` continuations).

**`boris_events.csv` — 12 rows, 1 defective:**

| sheet value | defect | resolves to |
|---|---|---|
| `20260109_K_1_3_FCMC` | set-date wrong; no such folder | `20251225_K_1_3_FCMC` via (K, 1, 3) |

Recoverable, and unambiguously — but note `20260109_K_2_3_FCMC` *is* a real
folder, so the bad name is one character from silently naming a different
experiment of the same group.

**`lay_eggs_table.csv` — 66 rows, 14 defective:**

| sheet value | resolves to | note |
|---|---|---|
| `20250106_E_2_1_FCMC` | `20260106_E_2_1_FCMC` | year typo 2025→2026 |
| `20250106_E_2_2_FCMC` | `20260106_E_2_2_FCMC` | year typo |
| `20250106_E_2_3_FCMC` | `20260106_E_2_3_FCMC` | year typo |
| `20250106_N_1_1_FCMC` | `20260106_N_1_1_FCMC` | year typo |
| `20250106_N_1_2_FCMC` | `20260106_N_1_2_FCMC` | year typo |
| `20250106_N_1_3_FCMC` | `20260106_N_1_3_FCMC` | year typo |
| `20250106_H_1_1_FCMC` | `20260106_H_1_1_FCMC` | year typo |
| `20250106_H_1_2_FCMC` | `20260106_H_1_2_FCMC` | year typo |
| `20250106_H_1_3_FCMC` | `20260106_H_1_3_FCMC` | year typo |
| `20251220_F_1_3_FCM3` | `20251220_F_1_3_FCMC` | **paradigm typo** `FCM3` |
| `20260112_I_2_3_FCMC` | `20260113_I_2_3_FCMC` | set-date off by one |
| `20260112_P_1_2_FCMC` | `20260113_P_1_2_FCMC` | set-date off by one |
| `20260112_M_1_2_FCMC` | `20260113_M_1_2_FCMC_P2` | **ambiguous, see below** |
| `20260112_M_1_3_FCMC` | `20260113_M_1_3_FCMC` | set-date off by one |

None of these touch the nine scored rows except the K_1_3 entry above.

**Two `(group, trial, day)` keys are not unique in the recording tree**, so
resolution by that key is not safe for them:

- `(E, 2, 1)` → `20260106_E_2_1_FCMC` **and** `20260106_E_2_1_FCMC_P2`
- `(M, 1, 2)` → `20260112_M_1_2_FCMC` **and** `20260113_M_1_2_FCMC_P2`

`exp_list.csv` records that `20250106_E_2_1_FCMC_2` is a *continuation* recording
made immediately after the first — so `_P2` is a second segment of one session,
not a separate experiment. The resolver reports these rather than guessing; the
two `_P2` rows above should be confirmed with the experimenter before use.

**`exp_list.csv` disagrees with `lay_eggs_table.csv` on folder names** for the
same experiments (e.g. `20260113_I_1_3_FCMC` vs `20260112_I_2_3_FCMC` — a
different *trial*, not just a different date) and carries its own year typos
(`20160108`–`20160114`). The recording tree is the only authority.

**One manifest hygiene item:** `MANIFEST.json` declares 27 assets under
`groundtruth/boris/`, of which one is a Windows `desktop.ini`. 26 are real
ethograms.

---

## What this changes

1. **Phase 0.5 does not let the project skip tracking.** The cheap route yields
   n = 3 trios from a single annotator pass, with the one behaviourally
   interesting row undetermined. Proceed with B0 as planned.
2. **`PRIOR_FINDINGS.md` §1.7 is corrected** — see the note appended there. The
   "two independent human record sources" framing and the p = 0.0078 do not
   survive.
3. **`female_accept` is not usable as ground truth for a tracking method**, since
   it is derived from the same video-coding pass any tracker would be scored
   against. It is a label, not a referee.
4. **The metric definition must be fixed before Phase C**, not after. Count and
   duration disagree on the one row where the biology is interesting.

## Open questions for the experimenter

- Who coded the BORIS files, and did that person have the lab sheet open? The
  same-day coding of E_1_1, E_1_2 and K_1_1 suggests yes for at least those.
- Is there any paternity or clutch-assignment record that would make the egg
  outcome male-discriminating? That is the one thing that would turn this into a
  real independent channel.
- Confirm `_P2` folders are continuations of the same session, and whether their
  events are included in the BORIS coding of the parent experiment.
- The 10 `?` and 11 `no` rows in `female_accept` were not chased. Worth doing
  only if the independence problem above can be solved — otherwise it enlarges a
  measurement that does not mean what §1.7 claimed.
