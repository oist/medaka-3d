# Evidence usable for independent evaluation of the 3 individuals, and the minimum unit of additional checking

On 2026-09-09, all file names in `assets/groundtruth`, the columns and contents of the
CSV/JSON/JSONL files, the old repository's `annotations/`, and the annotation-related file names
in `report/` and `runs/` were examined.
Within the scope examined, **no confirmed, method-independent set of before/after-encounter correspondences for all 3 individuals was found**.
This does not mean every file in the whole archive was scrutinized.
File SHA256s and tallies were saved to
[groundtruth_inventory.json](../../work/three_body/groundtruth_inventory.json).

| Existing data | Evidence actually present | Limitation for this evaluation |
|---|---|---|
| `female_sex_gt.csv` | Female selections: 300 confirmed, 1 ambiguous | The correspondence of the other 2 fish is not determined. `click_x/y` is a human selection; `mouth_x/y` and the geometry ID are linked estimates |
| `contact_episode_gt.csv` | 32 rows; clicks on `female` and `anchor_male` at each entry and exit | No clicks for the third individual. `contact_validity` and `tracker_assessment` are `uncertain` in every row. This does not contradict `evidence_quality=usable`, and is separate from a confirmed judgement. Automatic correspondence proposals are not promoted to human judgements |
| `swap_events.csv/json` | Human swap reports against one C30 render; includes male-side evidence | Depends on that render's slot names. Male-to-male reports are not exhaustive. Cannot be transcribed directly onto another method's slots |
| BORIS 26 CSV | Behaviour times and subject. K_1_2 has 173 rows, including `m1tf`, `m2tf` etc. | No raw-image coordinates or before/after-encounter correspondences. Naming individuals by which one courts and then verifying that behaviour is circular |
| SLEAP / Detectron2 / HITL image annotations | Keypoints and contours per image | The existence of pose ground truth does not imply the existence of temporal male-to-male identity ground truth |

The 32 old contact rows are not worthless. They retain the original human clicks and, kept apart
from the automatic mapping and SLEAP-snapped coordinates, can be the target of independent
re-checking if needed.
However, they do not include this `333519:333719`; their entry and exit frames lie in
13323–32635.
The earlier simplified statement that "there is no human male information at all" is not
accurate.

## The raw-image review page prepared for this

The [review page](../../work/three_body/review_truth/transient_333519-333719/index.html) reads
only the new FFV1 raw-image staging. It does not load model identities, poses, masks or automatic
correspondences.
The source BGR pixel SHA256s of 600 images were verified.
The 3-view video of all 200 frames can be played at 20 fps, and a native-size JPEG of each frame
allows exact frame stepping and mouth clicks. The initial 3 time points also have lossless PNGs.
The videos and JPEGs are recompressed for display, not the lossless raw frames themselves.

The minimum human work is as follows.

1. At the entry, call the 3 fish A, B and C arbitrarily, and mark one point on each fish's mouth in a view where it is easy to see.
2. Also mark the same fish's mouths just after the reported anomaly and at the exit. The minimum total is 9 points.
3. For each fish, answer whether "entry→after" and "after→exit" can be confirmed as the same fish, are ambiguous, or are not visible. 6 judgements in total.
4. If the female can be identified from the video, record that additionally. Naming the males white and red is not required.

The initial time points are 333519, 333621 and 333718. These are review candidates bracketing
the known tracking-anomaly report 333619–333620, and do not assume that the physical fish made
contact. The human can change the time points after viewing the raw images.
If continuity cannot be determined from these 200 frames alone, the item is left ambiguous and
additional video before and after is requested.
Image interpretation by the assistant is not written in as human ground truth.

A, B and C are anonymous correspondences within this clip and are not linked to the same letters
in another clip or on another day.
Confirming path continuity by playing back and forth is allowed. Inference from courtship
tendency or dominance is not allowed.
This follows the separation in `DEVELOPMENT_STRATEGY.md` between "encounter correspondence of
anonymous males" and "biological naming".
The "morphology only" sentence further on is not used to mean that visual temporal continuity is
also prohibited.

The JSON is initially entirely unconfirmed, and remains `human_draft_requires_adjudication` after
saving.
Checks for missing clicks and structural validation do not guarantee that the human judgements
are true.
The raw human input, the judge, the time taken and the ambiguous items are retained, and the
final version is saved to `/bucket`.
Results run after human correction are reported as a separate condition from fully automatic
results.

The 4 generator and input-boundary tests passed, and Ruff passed. Decoding of all video frames
was confirmed.
Opening `file://` in the browser was refused by security policy, and working around it by another
route was prohibited, so **clicking and JSON saving in a real browser are not verified**. No
claim is made that the UI has been checked for practical use.

## Proposed selection rule for the next unused evaluation windows — not yet extracted or opened

Results are not viewed before the settings are frozen. The old 2 development windows, the old 4
validation windows, the one minute 76620:86580 and the 132000:137000 mask-volume pilot are
already opened and are not reused for the next unused validation.
This exclusion is applied not only to the individual frames used but to the contiguous intervals
including the viewed frames before and after.

It is proposed to fix the following order and then select four 3-second windows from the same
video.

1. Save the list of used intervals and the list of source frames of all training and validation images of the SLEAP / Detectron2 models actually used.
   Training data whose mapping to source images is unknown is marked "unknown", and strict training non-overlap is not claimed.
2. Build 498-frame windows from start candidates at 1-second steps. Exclude the used intervals and all training frames with 332 frames before and after.
   Also include in the list the images already read as background samples, and keep background images out of the windows.
3. Sort the remaining candidates in lexicographic order of the SHA256 of the fixed string `three-body-validation-20260909-v1` with the experiment name and start frame.
   Take 4 in that order, at least 1660 frames apart from each other.
   Detection count, tracking score, presence of GT and fish separation state are not selection criteria. If there are too few, report the shortfall.
4. Open the video only after initialization, thresholds and stopping rules are frozen. Do not replace a window whose raw-image human correspondence is ambiguous with another, easier window.
   Do not silently drop the not-evaluated portion from the denominator.

This rule verifies "the results of the new tracking candidate have not been seen" and "does not
overlap the detector's training images" separately.
Adding windows within K_1_2 does not make a new independent group of individuals. A later stage
that claims generalization needs a different group.
What partial annotation confirms is the correspondence between the specified time points; it is
not a guarantee that no transient exchange ever occurs in between.
The observation rate, acceptance rate, fraction with confirmed correspondence, exchange reports
and not-evaluated fraction for the 3 fish are reported separately.

The current shared `score_encounter` also evaluates biological `Identity` names, so anonymous
A/B/C must not be reread as M1/M2 to compute naming accuracy.
When a future shared evaluation is added, a correspondence evaluation invariant to arbitrary
permutation of the initial slots and a naming evaluation for the distinguishable female are
verified separately.
