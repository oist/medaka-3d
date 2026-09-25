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
| Per-male identity (M1 vs M2)? | **Required** | It is the experiment's purpose: male–male competition, and which male she accepts. Not optional |
| Cross-day linkage of M1/M2 | **Highly desirable** | Turns identity into a re-identification problem — and into the validation instrument (below) |
| Annotation budget | **A few hours** | Forces encounter-level annotation. Frame-level labelling was never affordable |
| Structure | `labs/` quarantine, CI-enforced | Experiments cannot silently accumulate |

### Why per-male identity is now the *most promising* part, not the hardest

Two discoveries from the experimenter's own records (see
[PRIOR_FINDINGS.md](PRIOR_FINDINGS.md) §Corrections) invert the earlier outlook.

**The males differ by colour.** `group_list.csv` records a physical descriptor
for every animal. Group K: female *yellow*, males *white* and *red*. Across 24
trials, 14 are separated by colour + size, 8 by colour, only 2 by size alone.
The retired repo concluded that appearance was marginal and geometry had hit a
ceiling — without ever using this. Colour is about the strongest signal a CNN
can be handed.

**The same three fish are recorded on three consecutive days.**
`K_1_1 / K_1_2 / K_1_3` is one trio observed three times. That is the design the
whole identity problem should be built around.

### The three-day design is the validation instrument

Every appearance dataset in the retired repo was circular and leaky (§4.4):
trained and tested within the same video, so a model could score well by
memorising a background patch or a lighting gradient rather than the animal.

The repeated-measures design removes that possibility outright:

> **Train on day 1. Test on day 3.**

Different session, different lighting, different behaviour, same animals. A
model that transfers has learned the individual — there is no within-video
shortcut available to it. `Trial.holdout_split(test_day=3)` implements exactly
this and is the default protocol for any `identifier` experiment.

This is also the answer to "is cross-day linkage possible?" — it is the same
measurement. Cross-day accuracy *is* the re-identification result.

### What ground truth is actually obtainable (experimenter, 2026-08-02)

This constrains the whole design and corrects an over-reading of `group_list.csv`.

| Judgement | Obtainable? |
|---|---|
| Which fish is the female, from a **single frame** | **Yes**, near-perfectly |
| Which male is *white* vs *red*, from a single frame | **No** — the descriptors were assigned by close morphological examination at setup, not applied to video |
| Which male is which **across a crossing** | **Yes**, by scrubbing back and forth around the intersection |

So the colour descriptors are *not* a usable in-video labelling cue, and my earlier
claim that they hand a CNN a strong signal was too optimistic. They remain the
physical referent for M1/M2, and evidence that a real morphological difference
exists — a CNN may still exploit pixel statistics a human cannot, which is
exactly the idtracker.ai precedent. But they cannot be assumed.

**The design already matches what is obtainable.** `EncounterTruth` asks for
exactly the operation the expert can perform: mark the animals at entry, mark
them at exit, preserve the correspondence. Nothing else is required of them.

Two consequences:

1. **Male labels in ground truth are anonymous but consistent.** The annotator
   asserts "this male leaving is that male arriving", not "this is the white
   one". Chaining adjudicated encounters across a video yields globally
   consistent males for that day — which is the stated *minimum* requirement.
2. **`naming_accuracy` is scoreable for the female, not yet for the males.**
   Reported per identity, never as one pooled number. Do not read male naming
   accuracy until an anchor exists.

### The cross-day anchor problem, and a physical way out

Within a day, identity comes from continuity. **Across days there is no
continuity** — the fish are removed and returned — so cross-day linkage must come
from something intrinsic. If the expert cannot name a male from appearance, human
ground truth cannot validate an appearance model directly, and we would be
validating a model against itself.

A way out that does not depend on appearance at all: **measured 3D body length.**

Calibration gives metric 3D, and mouth–tail distance is a physical measurement
that is view-invariant, pose-correctable and independent of colour, lighting and
camera. `group_list.csv` records a size difference for 16 of 24 trials.

The test is cheap and self-validating:

1. Within one day, use continuity ground truth to separate male A from male B.
2. Measure each one's 3D length distribution over thousands of frames.
3. If the distributions separate, size is a usable anchor.
4. Check the ordering is stable across days and across trials of the same group.

If it holds, cross-day linkage gets an anchor that owes nothing to appearance —
and an appearance model can then be *validated against it* rather than against
itself. If the distributions overlap, that is an equally valuable negative result,
obtained cheaply.

### One trap to avoid

If part of how the males are told apart in practice is **behavioural** — one is
bolder, one courts more — then using that to label ground truth would make the
mate-choice analysis circular: identity inferred from behaviour, then behaviour
compared between identities.

Ground truth must rest on **morphology only**: colour, markings, body
proportion. If an annotator cannot tell them apart from a still frame without
watching what the fish does, that frame is not usable ground truth.

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

### Phase B0 — fix SLEAP at crossings (highest practical leverage)

Reported by the experimenter, 2026-08-02, on round 8: **detection degrades exactly
at crossings.** Instances connect one tail to two mouths; in an X-shaped crossing
one fish is found and the other becomes unstable.

This is upstream of everything. A wrong or missing detection at a crossing
poisons association, linking and identity in turn, and no amount of downstream
cleverness recovers a fish that was never detected. Fix it first.

**Diagnosis — three causes, all addressable:**

1. **The training set contains essentially no crossings.** §2.10: every clean-mask
   frame has 3D separation >20 mm, and male crops are never closer than 100 px.
   Labels were mined from frames the pipeline already handled, so the model has
   effectively never seen an X-crossing. This is almost certainly the main cause,
   and it is self-inflicted rather than a limitation of SLEAP. *Verify first* by
   counting labelled frames with two instances closer than 100 px.
2. **Top-down is structurally weak here.** The pipeline is centroid → crop →
   keypoints. When two fish overlap, the crop contains both, and
   `centered_instance` has no mechanism to decide which mouth belongs to which
   tail. "One tail, two mouths" is the exact signature of that failure.
3. **Top-down vs bottom-up was decided on the wrong distribution.** Round 7 trained
   a bottom-up baseline and `EVAL_SUMMARY.md` chose top-down for production — but
   over all frames, which are overwhelmingly well-separated fish. **A model can win
   on average while losing precisely where it matters.** Bottom-up part-affinity
   fields explicitly model which limb belongs to which instance, which is the
   problem at hand.

**Actions, cheapest first:**

- **Re-evaluate the existing round-7 top-down and bottom-up models on crossing
  frames only.** No training, both checkpoints are on `/bucket`. May flip the
  production choice, or justify running both and merging.
- **Mine crossing frames with `eval.find_encounters()` and annotate them.** The
  encounter detector already locates exactly these moments from existing tracks.
  A few hundred labelled crossing frames should move this more than any
  architectural change.
- **Add a geometric plausibility filter.** With mouth+tail and calibration, body
  length is a physical quantity with a known range (~25–35 mm). Two instances
  sharing a node, or a 3D length far outside distribution, is detectable and
  rejectable as a cheap post-process — it catches the reported failure directly.
- **Use multi-view consistency as a detection gate.** A 2D instance that cannot
  triangulate coherently against the other two views is suspect. Three cameras
  are an advantage no single-view tracker has; the old pipeline used them for
  reconstruction but never fed the result back to filter detections.

> **One annotation pass serves two purposes.** The frames needed to retrain SLEAP
> at crossings are the *same frames* needed for identity ground truth, and the
> same ones the expert can adjudicate by scrubbing. Annotate encounters once, and
> use the result for both. Given a budget of a few hours, this is the difference
> between the plan being affordable and not.

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

Priority order, revised now that the colour cue and the three-day design are known:

1. **Colour-anchored appearance identifier, validated cross-day** (`identifier`).
   The highest-value experiment in the project. Train on day 1 of a
   colour-separated trial (K, M, F, P), test on day 3. Success is a genuine
   re-identification result and directly delivers the cross-day linkage the
   study needs. Reported as `naming_accuracy`, not survival.
2. **Segmentation-instance linking** (`linker`). SAM2 video propagation is far
   stronger than what was tried, and masks give the appearance model clean
   colour pixels rather than a background-contaminated bbox. Constraint from
   §2.7: segmentation is a *confidence signal*, never a hard veto.
3. **Global fragment path-cover** (`identifier`). The best-designed *unexecuted*
   proposal in the old repo: constrained path-cover with explicit edge costs and
   multiple independent biological anchors, **leaving unanchored paths unlabelled**.
   With a working colour anchor it becomes far more tractable than it was.
4. **Reflection-suppressing scene model** (`detector`). Addresses the
   entry-phantom bug (§4.6), where reflections produced false fish.

### On the idtracker.ai precedent

The precedent is real: it routinely separates 3+ animals that humans cannot
distinguish, by training a CNN on crops taken from *global fragments* — stretches
where every animal is simultaneously visible and separated — then propagating
those identities through crossings.

The retired repo ran it and it failed its localisation gate (§2.9). That is
worth understanding rather than dismissing, because three things were working
against it and all three are fixable:

- It was fed 3D-cleaned **top-camera-only** input. We have three synchronised
  views, so there is 3× the appearance evidence per frame that idtracker.ai's
  single-view design cannot use.
- Its crops came from pipelines that filtered out close contact, so the model
  never saw the regime it had to resolve.
- Nobody had established that a visual cue existed, so a poor result was
  indistinguishable from an absent signal. We now know the cue is colour.

**Adopt the mechanism, not the tool**: global-fragment mining plus a CNN is the
right architecture, extended to three views and anchored on a known cue. Keep
idtracker.ai itself as a baseline to beat, since it gives a published reference
point.

### On 3D Gaussian splatting

Revised 2026-09-08: the earlier blanket dismissal was too broad.
[Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) explicitly studies
tracking through persistent Gaussian appearance and locally constrained motion.
[GaussianObject](https://gaussianobject.github.io/) uses masks and a visual hull
to initialize static object reconstruction from four images. These are relevant
mechanisms; neither establishes biological identity preservation for three moving
fish seen through refractive interfaces by three fixed cameras.

The next experiment should test independent masks and a constrained, per-fish 3D
representation, using the existing refractive projection. A Gaussian representation
is a candidate if a simpler body model's limitations justify it. Render quality,
point persistence and smooth trajectories are not sufficient evidence of correct
fish identity. See the [mask-volume research plan](MASK_VOLUME_RESEARCH_PLAN.md)
for comparisons and independent evaluation gates. An empty-tank background model
remains a separate possible use, not the only defensible use.

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
