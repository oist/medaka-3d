# Development strategy

How we get to reliable 3D tracking of three medaka with minimal identity swaps.

Companion to [PRIOR_FINDINGS.md](PRIOR_FINDINGS.md) (what was already tried) and
[HANDOFF.md](HANDOFF.md) (what was inherited).

---

## The problem, stated honestly

Three facts from the retired repo determine everything below.

**No method has ever passed a blinded identity audit.** The 2026-07-13 benchmark
tested multi-view constrained contrastive, idtracker.ai v6, and native UDMT under
a decision rule locked before unblinding. After correcting for clip-context bias,
**all three scored 0/2 encounters end-to-end**. This is not a tuning problem.

**The decisive regime has no training data.** Every clean-mask frame has 3D
separation >20 mm; male crops are never closer than 100 px (median 386, 0% under
100). Every quality filter discarded exactly the frames where identity is decided
(§2.10). Methods didn't fail at contact by accident — nothing had seen contact.

**Swaps were never measurable.** No male-vs-male ground truth has ever existed.
Total human identity annotation is 300 female labels on one experiment. And §2.2
records that track-quality QC metrics are structurally blind to swaps: a swapped
track and a clean track score identically.

The bottleneck is therefore **not** which tracker or classifier we choose. Adding
methods before fixing measurement is what produced 53 undead experiments.

---

## Decisions taken

| Question | Decision | Consequence |
|---|---|---|
| Per-male identity (M1 vs M2)? | **Required** | The hardest open problem stays in scope. Male-male GT must be built — it has never existed |
| Annotation budget | **A few hours** | Forces encounter-level annotation. Frame-level labelling is not affordable and never was |
| Structure | `labs/` quarantine, CI-enforced | Experiments cannot silently accumulate |

### Why a few hours is enough

Only because the unit of work is the **encounter**, not the frame.

Encounters are rare. The retired repo found 924 hard-core events, but only 67
lasting ≥0.5 s and 20 lasting ≥1 s — and its "7 crossings" collapsed to **2
encounters** under a correct sustained-separation rule. At ~1.5 min per encounter
(watch the clip, mark three animals at entry, mark them again at exit), a few
hours covers several experiments.

`eval.summarise()` reports `annotation_minutes_estimate` so the budget is known
before anyone starts labelling.

The ground truth is **method-agnostic**: it records only what the human saw, never
a tracker's output. Every future method is scored against it without
re-annotation. The retired repo's audits were method-specific, so both had to be
thrown away. These hours are spent once.

---

## Phases

### Phase A — build the ruler (blocking)

Nothing downstream is meaningful without this.

1. **Candidate encounters** — run any tracker good enough to give approximate
   positions, segment with `eval.find_encounters()`, report the annotation budget.
2. **Annotation tool** — plays a clip that *starts and ends in separation*, and
   captures three clicks at entry and three at exit. The context is not optional:
   omitting it is precisely what invalidated the previous audit.
3. **Frozen benchmark** — ≥3 experiments from **different groups and dates**.
   Everything to date rests on three experiments recorded on one day, so
   generalisation is untested. Split into `dev` (iterate freely) and `test`
   (touched only to report a final number).

**Exit criterion:** a single command scores any registered component and emits a
`Scorecard`. Until that exists, Phase C is forbidden.

### Phase B — collect the data the methods actually need

The contact regime is absent from every existing training set, and every
appearance dataset built so far was circular and leaky (§4.4).

- Deliberately sample **<20 mm** episodes — the frames every previous filter threw away.
- All three cameras, not side1 only.
- Split by **time window**, never randomly: random splits leak near-duplicate
  adjacent frames and inflate every score.
- Record full provenance per crop: source video, frame, mask index, bbox, score.

### Phase C — methods, one stage at a time

Only now. Each experiment is a lab, implements one stage Protocol, and is scored
by the shared harness.

Priority order, by expected value against swaps:

1. **Segmentation-instance linking** (`linker`). SAM2 video propagation is
   genuinely stronger than what was tried. Constraint from §2.7: segmentation is a
   *confidence signal*, never a hard veto.
2. **Contact-trained appearance identifier** (`identifier`). Only meaningful after
   Phase B — the previous classifiers were trained on data where the males were
   never within 100 px.
3. **Global fragment path-cover** (`identifier`). The best-designed *unexecuted*
   proposal in the old repo: constrained path-cover with explicit edge costs and
   multiple independent biological anchors, **leaving unanchored paths unlabelled**.
   Abstention is scored as a first-class outcome precisely to make this viable.
4. **Reflection-suppressing scene model** (`detector`). Addresses the
   entry-phantom bug (§4.6), where reflections produced false fish.

### On 3D Gaussian splatting

Not recommended as a tracking approach. 3DGS reconstructs static scenes from many
viewpoints; we have three fixed cameras and fast, non-rigidly deforming subjects.
Dynamic/4D variants need far more view coverage than three cameras give, and none
of them output identity — it does not address swaps.

One defensible narrow use: a static splat of the **empty tank** as a background
and reflection model, feeding the `detector` stage (priority 4 above). That is a
detector improvement, not a tracking strategy, and it should be scoped and judged
as such.

---

## Structural rules

Full detail in [../labs/README.md](../labs/README.md). Enforced by
`tests/test_labs.py`, not by discipline.

- Four stages: `Detector → Associator → Linker → Identifier`.
  **Linker and Identifier are separate on purpose** — temporal continuity and
  biological naming are different claims with different failure modes, and the old
  repo fused them so the two were indistinguishable in the output.
- One hypothesis per lab, with a `decision_rule` written **before** running anything.
- `labs/` may import `medaka3d`; `medaka3d` may never import `labs/`.
- Every lab expires. Past expiry with no `RESULT.md`, the build fails.
- Promotion moves code into `src/medaka3d/` and **deletes the lab directory**.
  Git remembers; the working tree should not.
- No lab writes its own scoring code.

---

## What we measure

Headline: **encounter survival** — the share of human-adjudicated encounters
crossed without a swap.

Reported alongside, never pooled:

- `precision_when_committed` — accuracy on encounters where the method answered.
- `abstention_rate` — how often it declined. Declining is safe, not correct.
- `swaps_by_pair` — F-M and M1-M2 counted separately. Different problems,
  different difficulty, different scientific cost.

Two properties worth internalising:

- A **consistent global relabelling is not a swap.** The M1/M2 letters are
  arbitrary and were always provisional; only a *change* in assignment across an
  encounter counts.
- Therefore **sexing errors are invisible to this metric.** A high survival rate
  is not evidence the female was correctly identified. That needs its own measure.

---

## Open questions

1. **Can the study conclusion be reached without tracking at all?** The
   `female_accept` field and the BORIS courtship asymmetry may agree across all
   experiments where both exist (§1.7, flagged as an unverified derivation). If it
   holds, it is the cheapest and strongest result in the project — and it tells us
   how much tracking precision the paper actually needs. **Check this before
   investing months in tracking.**
2. Does anything replicate outside the K group? All prior analysis is one day's
   recordings.
3. Does the calibration backend change any biological number? A gate opened and
   never closed (§4.17).
4. Is the courtship:aggression ratio real? It moved 24.5× → 15.2× between two
   backbones on identical data.
