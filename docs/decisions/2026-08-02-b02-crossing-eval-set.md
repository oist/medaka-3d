# 2026-08-02 — B0.2: crossing-frame evaluation set and its annotation budget

> **SUPERSEDED BY A FULL CENSUS — 2026-08-02.** All 65 experiments with retained
> predictions have now been enumerated; see the "Measured corpus census" section
> at the end. Headline: **14,684 link-free encounters = 367.1 h across 12 groups.**
> The projection below (~270–425 h) did bracket it, but its stated precision was
> not earned — see the scale-spread correction in that section.
>
> **SCOPE AND SCALE CAVEATS on the 9-experiment numbers below.**
>
> 1. **This section covered 9 of 65 experiments** (groups E, K, O). The 47 h is
>    measured and correct *for those nine*; the corpus figure below it was a
>    projection. Use the census instead.
> 2. **The 0.182–0.190 mm/px scale is an operational screening proxy**, derived
>    from the extent the animals happen to cover — not physical calibration. Its
>    consistency across 9 recordings shows the estimator is stable, **not** that it
>    is accurate. The inference below that the `tail` node sits at the tail base
>    rather than the fin tip does not follow from it, and B0.4's threshold must be
>    settled against the calibration model instead.
> 3. **Top view only**, so depth is collapsed and every count is an upper bound on
>    true 3D encounters.

**1,888 encounters across 9 experiments (3 trios) ≈ 47 hours of annotation.
Projected to the 65 experiments with retained predictions: ~340 hours. The
benchmark cannot be built by annotating everything — it has to be sampled, and
Phase A.3 needs a sampling rule written before anyone starts.**

**Reproduce:** `.venv\Scripts\python.exe tools/b02_crossing_eval_set.py <experiment>...`

---

## Scope

The 9 experiments carrying BORIS annotation and a resolved `female_accept` —
groups **E, K, O**, three days each. **n = 3 trios**, not 9; days are repeated
measures. These were chosen because they are the same rows Phase 0.5 scored, so
the two analyses share a cohort.

## Results

| experiment | mm/px | detections <3 | detections >3 | linked | link-free | median dur | max dur |
|---|---|---|---|---|---|---|---|
| E_1_1 | 0.1820 | 35,750 | 18,514 | 564 | **188** | 3.48 s | 121.2 s |
| E_1_2 | 0.1859 | 21,817 | 13,625 | 553 | **263** | 2.91 s | 44.5 s |
| E_1_3 | 0.1881 | 19,690 | 9,858 | 496 | **254** | 2.96 s | 47.6 s |
| K_1_1 | 0.1887 | 19,782 | 13,630 | 461 | **233** | 3.27 s | 140.5 s |
| K_1_2 | 0.1896 | 22,939 | 10,951 | 514 | **185** | 4.66 s | 100.1 s |
| K_1_3 | 0.1875 | 25,377 | 14,095 | 524 | **167** | 4.67 s | 162.0 s |
| O_1_1 | 0.1880 | 25,221 | 10,749 | 487 | **172** | 6.31 s | 83.1 s |
| O_1_2 | 0.1901 | 24,900 | 9,065 | 374 | **224** | 3.86 s | 74.6 s |
| O_1_3 | 0.1832 | 21,247 | 11,757 | 280 | **202** | 4.09 s | 48.1 s |
| **total** | | | | **4,253** | **1,888** | | |

Each experiment is 348,600 frames (2,100 s at 166 fps).

**Annotation budget** at the harness's 1.5 min/encounter:

- link-free: **1,888 encounters → 2,832 min = 47.2 h**
- linked (all three pairs counted separately): 4,253 → 6,379 min = **106.3 h**

The **link-free** number is the one to plan against. Phase A.2's protocol marks
*all three* animals at entry and exit, so one close-approach window is one clip
regardless of how many pairs it involves. The linked count is 2.3× higher because
a single three-fish cluster generates up to three pairwise encounters.

---

## The pixel scale, independently confirmed

The per-experiment scale estimates span **0.1820–0.1901 mm/px** — a 4% spread
across 9 recordings measured independently. That consistency is itself evidence
the method works: it is derived from the extent the animals cover against the
calibrated 192 mm inner span (§2.11), not assumed.

At ~0.188 mm/px the tank occupies ~1,020 of the 1,200 px, and B0.1's median
labelled mouth–tail span of 102 px is **19.2 mm**. This confirms B0.1's open
question: the labelled span is *not* the animal's 25–35 mm body length, so the
`tail` node most likely sits at the tail base rather than the fin tip.

**B0.4 must key its plausibility filter to ~19 mm, not ~30 mm.** A filter written
against the true body length would reject most correct instances.

---

## Detection failures, which B0.4 should target

Per experiment, out of 348,600 frames:

- **fewer than 3 detections in 19,690–35,750 frames (5.6–10.3%)** — at least one
  fish missing. E_1_1 is the worst at 10.3%.
- **more than 3 detections in 9,065–18,514 frames (2.6–5.3%)** — phantom or
  duplicate instances.
- zero detections in 2–485 frames.

A concrete instance of the reported failure, from `K_1_1` frame 1000 of the first
chunk: four predicted instances, two of whose tails are **3.4 px apart**
(`686.7, 822.5` and `690.1, 822.6`), one of them with a **NaN mouth**. That is
"one tail joined to two mouths" in the raw production output. B0.1 found **zero**
such node-sharing in the training labels, so this is generated at inference.

---

## Three honesty constraints on these numbers

1. **Top view only.** The retained predictions are `top_cam`. A top-down
   projection collapses depth, so two fish at the same (x, y) at different depths
   read as coincident. **These counts are an upper bound** on true 3D encounters.
   The real figure needs side views, which for round 7 were deleted (only
   `top_cam` was retained, 7.94 GB of 65 experiments).
2. **The predictions are untracked** (`len(labels.tracks) == 0`), so the greedy
   nearest-centroid linker here is not a tracker. That is why every experiment is
   also measured link-free; where they disagree, trust link-free.
3. **Round-7 top predictions are the *production* model's output, but B0.3 could
   not score that model** — its saved validation predictions are empty files. So
   the detection-quality of the very predictions used here is currently unmeasured.

---

## What this changes

1. **Phase A.3's frozen benchmark cannot annotate everything.** Even three trios
   cost 47 h. Projecting the per-experiment range (167–263 encounters) across the
   65 experiments with retained predictions gives roughly **11,000–17,000
   encounters, ~270–425 h** — this is a projection from 9, flagged as such, not a
   measurement.
2. **A sampling rule must be written before annotation starts**, per the
   `labs/` discipline: how many encounters per trio, stratified by what (duration?
   minimum distance? number of participants?), and which trios. Gate A.3 requires
   ≥3 different groups; E, K and O are exactly three colour-separated groups with
   BORIS coverage, so they are the natural starting cohort.
3. **Encounters are long enough to be affordable per clip but numerous.** Median
   3–6 s plus 1 s of context each side gives ~5–8 s clips. The cost is count, not
   length.
4. **B0.4's threshold is now derivable** — ~19 mm labelled span at ~0.188 mm/px.

## Open questions

- Should the benchmark sample encounters uniformly, or over-sample the short
  ones? The long tail (max 44–162 s) is a small number of clips carrying a large
  share of total ambiguous time.
- Do the 65 retained `top_cam` predictions cover E, K and O for all three days?
  All 9 requested experiments resolved here, but the full 65-experiment coverage
  against the 71-folder tree has not been enumerated.
- Is it worth re-running round-7 side inference to get true 3D encounter counts,
  or does the top-view upper bound suffice for building a benchmark? This is
  HANDOFF's open question 2 in a different form.

---

## Measured corpus census — all 65 experiments (2026-08-02)

Reproduce:

```powershell
.venv\Scripts\python.exe tools/b02_crossing_eval_set.py            # all experiments
.venv\Scripts\python.exe tools/b02_aggregate_census.py TRANSCRIPT [TRANSCRIPT ...]
```

**Reported by group, because the group is the independent statistical unit.** The
experiment count is a workload figure, never a sample size.

| group | exps | link-free | linked | link-free hours | scale mm/px |
|---|---|---|---|---|---|
| D | 3 | 857 | 1,489 | 21.4 | 0.1781–0.1870 |
| E | 6 | 1,222 | 3,305 | 30.6 | 0.1820–0.1912 |
| F | 3 | 910 | 1,232 | 22.8 | 0.1838–0.1875 |
| G | 6 | 1,696 | 3,192 | 42.4 | **0.1864–0.2131** |
| H | 6 | 1,090 | 2,722 | 27.2 | 0.1863–0.1917 |
| I | 6 | 1,515 | 2,956 | 37.9 | 0.1833–0.2088 |
| K | 6 | 1,050 | 3,257 | 26.2 | 0.1836–0.1896 |
| L | 6 | 1,418 | 2,593 | 35.5 | 0.1873–0.2021 |
| M | 6 | 1,417 | 2,810 | 35.4 | 0.1844–0.1990 |
| N | 5 | 819 | 2,035 | 20.5 | 0.1827–0.1922 |
| O | 6 | 1,224 | 2,214 | 30.6 | 0.1814–0.1901 |
| P | 6 | 1,466 | 2,753 | 36.6 | 0.1868–0.1905 |
| **total** | **65** | **14,684** | **30,558** | **367.1** | 0.1781–0.2131 |

**n = 12 groups.** Link-free: **14,684 encounters = 367.1 h**. Linked (all three
pairs counted separately): 30,558 = 764.0 h.

## Corpus coverage — what the census does *not* reach

The recording tree holds **71 experiments across 14 groups**; `sleap_output` holds
**65 across 12**. Six experiments have no retained predictions:

- **groups B and C entirely** — `20251223_B_1_1`, `20251223_C_1_1`, `20251223_C_1_2`
  (both are "give up" groups in `group_list.csv`);
- `20260115_N_2_3`, and the two `_P2` continuation recordings
  (`20260106_E_2_1_FCMC_P2`, `20260113_M_1_2_FCMC_P2`).

So the census is complete over the *retained predictions*, not over the corpus.

## The projection was bracketed but overconfident

| quantity | projected from 9 experiments | measured over 65 |
|---|---|---|
| link-free encounters | 11,000–17,000 | **14,684** |
| annotation hours | ~270–425 h | **367.1 h** |
| scale mm/px spread | 0.182–0.190 (**4%**) | **0.1781–0.2131 (19.7%)** |
| frames with <3 detections | 5.6–10.3% | **3.2–18.2%** (median 5.7%) |
| frames with >3 detections | 2.6–5.3% | **1.0–9.9%** (median 3.0%) |

The encounter and hour projections did contain the true values, but the
per-experiment variability was understated several-fold in every case. The
9-experiment scale spread of 4% was presented as evidence the estimator was
stable; over 65 it is **19.7%**, and group G alone spans 0.1864–0.2131 *within one
group*. That is a second reason — beyond the absence of calibration — not to treat
this scale as physical.

## What this changes for Phase A

**367 h of annotation to cover every candidate is not a plan.** Phase A.3 must
sample, and the sampling rule has to be written before annotation starts. The
per-group hours above are the budgeting input: three colour-separated groups with
BORIS coverage (E, K, O) total **87.4 h** of candidates, so even the minimum
Gate A.3 cohort needs a sampling rate, not exhaustive coverage.
