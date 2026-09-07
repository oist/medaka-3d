# 2026-08-03 — Group K: what the detector's false positives actually are

**86.5% of false positives at approaches are geometrically classifiable and
removable without touching the model. Two categories are 77% of the total:
shared-node instances (40.4%) and partial instances (36.5%). The honest
false-positive rate is 0.46 per frame — 13.8× what the pooled number suggests.**

Reproduce: `.venv\Scripts\python.exe tools/k01_misdetection_taxonomy.py`

---

## Why Group K can be measured without new annotation

- **Every labelled top frame in the corpus is Group K** — 1,382 round-8 training
  frames (K_1_1 125, K_1_2 1,132, K_1_3 125) plus 20 validation. So FP status can
  be determined exactly by matching predictions against ground truth.
- **Cardinality is exact.** Every labelled frame holds exactly 3 instances, so
  `n > 3` implies at least `n − 3` false positives in any unlabelled frame too.
- All six K experiments have retained round-7 `top_cam` predictions.

## The rate, and why the pooled number is wrong

| frames | n | false positives | rate |
|---|---|---|---|
| in round-7 training | 1,288 | 43 | 0.033 /frame |
| **held out from round 7** | **113** | **52** | **0.460 /frame** |

**The held-out rate is 13.8× the in-sample rate.** 1,288 of the 1,401 labelled K
frames are round-7's own training data; scoring there measures memorisation.
**Use 0.46 FP/frame.** Pooling the two gives 0.068 and is meaningless.

## Taxonomy — held-out frames only (52 FPs)

Categories are applied in priority order, so each FP lands in exactly one.

| category | n | share | what it is | remedy |
|---|---|---|---|---|
| **A shared_node** | 21 | **40.4%** | shares a node (<5 px) with another instance — the reported "one tail joined to two mouths" | B0.4 shared-node rule, **already removes 100%** |
| **E partial** | 19 | **36.5%** | one node missing | completeness rule + label fix (below) |
| B duplicate | 3 | 5.8% | second detection of the same animal, no shared node | centroid NMS at ~0.5 body length |
| D collapsed | 2 | 3.8% | mouth and tail nearly coincident (span 0–3 px) | B0.4 span lower bound |
| C chimera | 0 | 0.0% | span above the labelled 99.5th pct | — does not occur |
| F out_of_arena | 0 | 0.0% | centroid outside the arena | — **not a clean negative, see caveats** |
| **G unclassified** | 7 | **13.5%** | none of the above | **needs model work or annotation** |

**Geometrically addressable: 45/52 = 86.5%.**

## Where they occur — FPs concentrate at contact and crossing

Pooled across all 1,401 frames, regime from the closest pair of complete
ground-truth instances, in body lengths:

| regime | frames | FPs | FP/frame | A shared_node |
|---|---|---|---|---|
| contact (<0.25 body) | 184 | 15 | 0.082 | 7 |
| crossing (0.25–1) | 669 | 54 | 0.081 | 21 |
| approach (1–2) | 328 | 12 | 0.037 | 2 |
| separated (>2) | 200 | 9 | 0.045 | 4 |

**The FP rate roughly doubles at contact and crossing**, and **80% of all
shared-node failures (28 of 35) occur there**. The reported bug is confirmed as
approach-specific, not a uniform background error.

## Root cause of the second-largest category

Partial instances are not a decoding accident — the model was trained on them:

| label set | both nodes | mouth only | tail only | empty |
|---|---|---|---|---|
| round7 top | 94.3% | 1.7% | 3.9% | 0.1% |
| round8 top train | 94.1% | 1.7% | 4.1% | 0.1% |
| round8 side train | 88.6% | 5.0% | 5.9% | 0.5% |
| round9b side | 89.6% | 4.6% | 5.3% | 0.5% |

**5.9% of round-7 top training instances are incomplete**, and the model
reproduces that behaviour as 36.5% of its held-out false positives. Note round9b
did **not** fix this on the side view — it completed *instances per frame*, not
*nodes per instance*.

**This is the highest-leverage annotation task available.** Completing the ~244
partial instances in `labels.round8.top.train` is hours of work, against the 367 h
that annotating every encounter candidate would take.

## Visual review — what the crops show

Rendered with `tools/k02_render_misdetections.py` into
`work_dir/k02_misdetections/` (per-category montages, a combined sheet, and one
tile per FP). Red = the false positive, blue = other predictions, green = ground
truth.

**A_shared_node (21) is confirmed by eye**: in almost every tile the red instance
lies on top of a blue one, on the *same animal*, sharing a node. These are second
detections of an already-detected fish, not detections of nothing.

**D_collapsed (2) is unambiguous** — spans of 0 px and 2 px, mouth and tail on the
same pixel.

**G_unclassified is not mysterious model failure — it is largely periphery.**
Measured distance of the instance centroid from the nearest frame edge:

| set | n | median | min | below 60 px |
|---|---|---|---|---|
| ground truth | 339 | 204.4 px | 68.7 px | 0 |
| matched predictions | 301 | 204.5 px | 32.6 px | — |
| A_shared_node | 21 | 187.3 px | 102.0 px | 0/21 |
| E_partial | 19 | 174.6 px | 42.7 px | 2/19 |
| **G_unclassified** | **7** | **92.4 px** | **36.2 px** | **3/7** |

G instances sit at **less than half** the typical distance from the frame edge,
and **3 of 7 are closer to the edge than any labelled fish has ever been**
(ground-truth minimum is 68.7 px). Those three are outside the region where
animals actually occur and are consistent with wall/edge reflections — the
PRIOR_FINDINGS §4.6 phantom class.

**Correcting an overstatement**: on first viewing I read 6 of the 7 as sitting on
tank-edge reflection bands. The measurement supports only 3 as provably outside
the observed animal region; the remaining 4 are at 92–150 px, well within the
range fish occupy, so the bright bands they overlap are *interior* tank edges
rather than the frame periphery. The effect is real but smaller than it looked.

**So the F rule was mis-specified, not falsified.** Defining the arena as labelled
positions padded by one body length swallows the whole frame and can never fire.
Re-specified as *"reject instances closer to the frame edge than the ground-truth
minimum (68.7 px)"*, it would catch 3 of the 7 unclassified FPs plus 2 partials,
lifting the geometrically addressable share from **86.5% to roughly 92%**.

## Selection: why suppression is the wrong operation

The top-down heads use **`anchor_part='mouth'`** (both `centroid` and
`centered_instance` training configs). An occluded mouth therefore yields a
spurious anchor that claims the *same tail* as a real instance.

**Confidence cannot resolve it.** Across the 21 shared-node pairs, the false
positive has the lower instance score in only **16/21**. And it fails exactly on
the hard cases — montage tile 15 (frame 274709):

| | false positive | partner |
|---|---|---|
| instance score | 0.831 | 0.825 |
| mouth score | 0.988 | 0.980 |
| tail score | 0.448 | 0.500 |

Both mouths are strong — both animals *are* correctly anchored — and the two weak
tails are the same tail. A 0.006 score difference is noise.

**Worse, suppression destroys evidence.** Of the 47 held-out frames containing a
false positive, **15 (32%) have only 2 of the 3 animals covered by matched
instances**. Frame 274709 is one: 3 animals, 4 predictions, 2 matched. The
duplicated tail coexists with a *missed* fish, so deleting the loser leaves two
animals, not three.

**Therefore B0.4's rule A must not be applied as suppression at crossings.**
Removing shared-node instances raises precision, which is what B0.4 measured, but
in a third of these frames it discards the only detection carrying a third animal.

### The mouth-anchored assignment reframing

State the problem as *assign tails to mouths one-to-one*, not *pick the better
instance*. Mouth-count on the held-out frames (score > 0.5):

| mouths in frame | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| frames | 2 | 21 | **66** | 24 |
| after dropping mouths within 69 px of the frame edge | 2 | 26 | **70** | 14 |

So ~60% of frames are well-posed (exactly 3 anchors), ~20% have a genuinely
occluded mouth, and ~12–21% carry an extra anchor — mostly glass reflections, of
which the edge rule removes about 40%.

**Note this frame set is enriched for difficulty by construction** — it is the
subset a human chose to correct — so these are worst-case proportions, not
corpus-wide ones.

## Would more labels fix the fully-visible crossings? No.

The question is fair: tiles 15 and 31 have **all three ground-truth instances
complete** — nothing is occluded — so the failure is in principle learnable, and
more labels of that configuration ought to help. The data says otherwise.

Define the **decoy ratio** for a labelled instance as
`distance(mouth → nearest foreign tail) / distance(mouth → own tail)`. Below 1.0
means a competitor's tail is *closer to the anchor* than the true tail — the hard
case the model must resist.

**Training already contains the hard case in bulk.** In `labels.round8.top.train`
(3,883 instances with a neighbour):

| decoy ratio | instances | share |
|---|---|---|
| < 1.0 (decoy nearer than own tail) | 1,308 | **33.7%** |
| < 1.5 (ambiguous) | 2,220 | 57.2% |
| 10th percentile of ratio | 0.50 | |

**The cited failures are easier than typical training data:**

| case | decoy ratio | easier than |
|---|---|---|
| tile 15, inst1 | 1.14 | **44.2%** of training instances |
| tile 31, inst1 | 1.19 | 46.3% |
| tile 31, inst0 | 1.34 | 52.5% |
| tile 15, inst2 | 1.65 | 60.4% |

In tile 15 the true tail was **79.9 px** from the anchor and the decoy **90.8 px**
— *the correct tail was the nearer one and the model still took the other.*

**The model fails on configurations less ambiguous than roughly half its training
set, having seen ~1,300 strictly harder examples.** That is a decoder/architecture
limit, not a coverage gap. More labels of the same configuration should not be
expected to help.

### ~~Where the non-annotation leverage is: crop size~~ — RETRACTED

`crop_size: 512` against a median fish of 113 px is 4.5× the animal, and over the
**training population** shrinking the crop looks helpful: decoy-tail inclusion
falls from 70.2% at 512 to 54.8% at 320 while every true tail is retained.

**That reasoning was wrong, and the experiment refutes it.** Failures are not a
random sample of training instances — they are concentrated exactly where the two
tails are close, so a population statistic says nothing about them. Measured on
the 18 shared-node failures directly:

| crop | shared-node failures fixed |
|---|---|
| 320 | **0 / 18** |
| 256 | 1 / 18 |
| 192 | 8 / 18, but only 21.7% of true tails survive |

The wrongly-taken tail sits at median **91.8 px** (max 134.6), well inside any
crop that still contains the correct tail. In **8 of 18** the decoy is *nearer*
than the true tail, so no crop size can separate them even in principle. **Crop
tuning is not a lever for this failure.**

### What the assignment method can and cannot recover

When two mouths claim one tail, a one-to-one constraint **consumes** the tail —
the winner keeps it, the loser never reuses it. What the loser can do depends on
whether its own tail exists in the output at all. Measured over the 19 shared-tail
failure frames:

| | frames | outcome under assignment |
|---|---|---|
| every ground-truth tail present in the output | **9 / 19** | loser takes the unclaimed tail — **animal recovered** |
| a true tail never detected anywhere | **10 / 19** | no candidate exists — loser emits **mouth-only** |

**The missing tails are genuinely absent, not threshold-suppressed.** Sweeping the
decoder 20× lower recovers almost nothing:

| setting | instances | tail coverage | previously-missing tails recovered |
|---|---|---|---|
| production (0.2) | 70 | 81.5% | — |
| 0.05 | 70 | 81.5% | 0 / 10 |
| 0.01 | 76 | 83.3% | **1 / 10** |
| 0.01 + max_instances 6 | 76 | 83.3% | 1 / 10 |

So a confmap-based candidate pool will not rescue those frames, and the realistic
ceiling for assignment on this set is **9 of 19 recovered, 10 converted to honest
mouth-only**. Eliminating the shared-tail category structurally is still worth it —
a mouth-only instance keeps position and identity continuity and loses only
orientation, whereas a swapped tail corrupts the identity chain silently.

*Limitation:* this tests the standard decode path, which emits one tail per crop
(the argmax). It does not prove the tail confmap has no secondary peak at the true
location — that needs `Predictor.layer` internals. But nothing reachable by normal
inference produces it.

**This is the measured case for a capacity retrain**, and it is the argument the
confmap loss could not make: for those 10 tails the model does not represent the
keypoint at all, at any threshold.

**Where labels *would* help** is the other failure: 5.9% of top training instances
are missing a node, and the model reproduces that as 36.5% of held-out false
positives. Completing ~244 instances is hours of work against a category twice the
size of what crop tuning addresses.

## Caveats

- **`n_groups = 1`.** Everything here is Group K. It is also the only group with
  labels, so the detector itself was trained only on K — which is a live concern
  for the planned rollout, not a footnote.
- **Only 113 frames give an honest rate.** The taxonomy shares are from that
  sample; treat single-category percentages as ±several points.
- **F out_of_arena = 0 is not a clean negative.** The arena bound is the labelled
  positions padded by one body length, which is generous, and these frames may not
  include the entry period where the §4.6 reflection phantom occurs. The category
  is **untested**, not disproven.
- **C chimera = 0 is expected**, not a surprise: a top-down centered-instance model
  crops around one centroid, so it structurally cannot join two distant animals.
  This category would matter for a bottom-up model and should be retained.
- FP status is defined against ground truth that is itself ~48% verbatim top-down
  output (see Gate B0). That inflates *matches*, so the true FP count is a **lower
  bound**.
