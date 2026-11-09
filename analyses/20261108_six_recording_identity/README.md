# Automatic identity against human correction on the six K_1 and P_1 recordings

Date: 2026-11-08. Owner: research supervisor; execution by Claude Code.
Status: **registered 2026-11-08 before any reference array, score or new
appearance model for this study existed; all four phases executed the same
day.** Findings are under "Findings"; one reference conflict (P_1_2) awaits
a human video check.

## Question

The downstream analysis follows the 3D trajectories of the three fish (one
female, two males) up to behavioural contact and needs each trajectory to
carry the right individual. For the six recordings that have a human
identity review, how close does automatic identity come to the human-corrected
identity, on the quantities that analysis consumes, when identity is taken
from

- **A0**: the geometry chain alone (`c27 -> c28 -> c33 -> c30 v4 -> c34`),
  slot labels carried from the first human assignment without any later
  correction;
- **N / S / G, per recording**: the chain plus appearance naming of
  Registration 4 of `labs/20261106_contrastive_appearance_link` (clean
  crops, one side model, silhouette stopping), encoder trained on that
  recording's own label-free fragments;
- **N / S / G, per group**: the same pipeline with one encoder per trio
  trained on the pooled fragments of its three recordings (K_1_1--3,
  P_1_1--3), applied to each recording.

"Human-corrected" is the ID-pass review of each recording, with the Jingyi
review as a second reference on P_1_1 and the female clicks as a consistency
check on K_1_2 (see "References").

## Inputs, fixed and read-only

All paths are relative to the data parent (the parent of the configured
`raw_data_dir`). Files are hashed when the scripts read them and recorded in
each run's provenance; nothing here is modified.

| Recording | Chain output (`work/c34_microflip/<rec>__chain-c28-c33-c30v4-c34/`, lambda 0.767) | Detection caches | Review package (`review_packages/<rec>/<pkg>/`) |
| --- | --- | --- | --- |
| `20251225_K_1_1_FCMC` | written 2026-10-30 | `work/c01_s16_predictions/` | `s16_bottomup_3node_uncapped_m3dfull` |
| `20251225_K_1_2_FCMC` | written 2026-09-03 | `work/c01_s16_predictions/` | `s16_bottomup_3node_uncapped_m3dfull` |
| `20251225_K_1_3_FCMC` | written 2026-10-30 | `work/c01_s16_predictions/` | `s16_bottomup_3node_uncapped_m3dfull` |
| `20260112_P_1_1_FCMC` | written 2026-10-28 | `work/c01_s16_predictions/bottomup/` | `s14_bottomup_3node_reviewed_m3dfull` |
| `20260113_P_1_2_FCMC` | written 2026-10-30 | `work/c01_s16_predictions/bottomup/` | `s14_bottomup_3node_reviewed_m3dfull` |
| `20260112_P_1_3_FCMC` | written 2026-10-30 | `work/c01_s16_predictions/bottomup/` | `s14_bottomup_3node_reviewed_m3dfull` |

Also: `reconstruction_3d/<rec>/<pkg>/<pkg>_3d.csv` (the 3D positions the
ID-pass GUI displayed; SHA-256 equal to the package's `review_manifest.json`
entry, checked for K_1_2 and P_1_1 on 2026-11-08), `jingyi/P-1-1/reviewed_3d.csv`
(SHA-256 `4f188db9...`), `assets/groundtruth/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv`,
raw videos and `top_cam/frametimes.npy` of each recording, and the two
existing Registration 4 appearance runs
(`jingyi/P-1-1/appearance_clean_20261107_r1`,
`work/appearance/20251225_K_1_2_FCMC/appearance_clean_20261107_k12`).

All six chain outputs predate the 2026-11-02 row-mapping repair; on P_1_1
that repair changed two labels at frame 20 and nothing downstream. The K
recordings were detected with the s16 run and the P recordings with the S14
run capped at 6 instances; detector stratum and trio are confounded.

## What the review package is, established before registration

Each package's `<pkg>_reconstruction_provenance.json` states: pipeline
"medaka-3d full chain (c27 -> c28 -> c33 -> c30 v4 -> c34)", identity
"C34-label slot columns as ID_1..ID_3", xyz "pectoral anchor from the chain;
gaps up to 100 frames interpolated in the CSV". Checked on 2026-11-08 for all
six recordings: 99.76--99.97% of the package's finite positions lie within
1 mm of the chain slot with the same index, and every frame with three
finite positions matches the slots as the identity permutation. **The ID-pass
is therefore a human correction of the chain's own slot labels**, which is
exactly the comparison this study needs: "without manual correction" is the
chain's slot labelling, "with manual correction" is the same positions
relabelled by the review.

An ID-pass consists of `id_segments.csv` (time segments, each with the
mapping from F/M1/M2 to ID_1..ID_3), `human_events.csv` (initial labels,
`swap` events, `danger` and `pose_bad` intervals, `analysis_start`) and
`id_pass_summary.json`. Gaps between segments are the reviewer's danger or
unusable intervals; `danger_frames` inside a segment are machine-flagged
frames the reviewer still assigned. One reviewer, host `DESKTOP-9MO13KR`,
2026-10-30 to 2026-11-01, playback up to 2x; the reviewer's name is not
recorded.

## References

- **ID-pass reference (primary, all six).** The review names the package's
  ID_1..ID_3, which are the production chain's slots. Each finite package
  position is matched to the slot of the chain the recording's appearance
  run consumes, at the same frame, within 1 mm and mutually unique; that
  slot is named by the segment containing the frame, and frames outside
  every segment and before `analysis_start` are unknown. For five
  recordings that chain is the production output and the match is the
  identity; for P_1_1 the appearance run of Registration 4 consumed the
  row-repaired rerun chain of 2026-11-02
  (`jingyi/P-1-1/id_row_mapping_20261102_r1/pipeline/c34_microflip/...`),
  whose slot numbering differs from the production chain's after some
  stitches, so P_1_1 is scored on that chain. (Correction 2026-11-08, same
  day, before any appearance score was recorded: run r1 read the review by
  slot index onto the production chain for all six and was discarded; it
  stays on disk as the record of the mistake.) Flags kept per frame: inside a `danger`
  interval, inside a `pose_bad` interval, position interpolated in the
  package CSV, position missing in the chain. The adapter also replays the
  `swap` events from the initial labels and requires the result to equal
  `id_segments.csv`; a mismatch stops the build.
- **Jingyi reference (P_1_1 only, second).** Per-frame pectoral positions of
  F/M1/M2 from `reviewed_3d.csv` matched to chain slots with the inherited
  rule (mutual nearest, at most 5 mm, margin at least 2 mm;
  `medaka3d.eval.review_identity.match_positions`), valid where not
  `uncertain`, not `bad3d` and finite. Agreement with the ID-pass reference is
  reported per fish; frames where the two disagree are counted and, in a
  sensitivity row, scored as unknown.
- **Female clicks (K_1_2 only, check).** The 299 confirmed female
  identifications mapped to slots as in Registration 4; agreement with the
  ID-pass's F is reported. Males on K_1_2 are checked by the ID-pass only.

## Candidates and how their names are obtained

- **A0** names slot k for the whole recording by its majority review label
  over the first review segment that reaches the analysis start (on five
  recordings this is the human's assignment of the production slots in that
  segment; on P_1_1 it is read through the position match); nothing later
  is corrected. This is the chain "without manual correction".
- **N_carry, S, G (per recording)** are the per-frame slot names in
  `evaluation/identity_tracks_{N_carry,S,G}.npz` of a Registration 4 run.
  For K_1_2 and P_1_1 the existing runs are read as they are (P_1_1's crops
  come from Registrations 1--3 of the same pipeline). For the other four,
  `run_appearance_clean.py --config config_appearance_six_v1.toml` runs
  bundle -> extract -> prepare -> train -> evaluate with Registration 4's
  settings; crops are staged on local disk (the network share runs at about
  10 MB/s) and the run directory is copied to `work/appearance/<rec>/`
  with a hash manifest afterwards. The names of a run's N/S/G are per
  recording (the three cluster labels are arbitrary); they are aligned to the
  reference on the first review segment that has frames at or after the
  analysis start (restricted to those frames), by the permutation of the
  three codes that agrees most with the review there. A candidate is thus
  credited only for keeping identity consistent with the human's assignment
  at the start of the scored time, never for guessing physical names.
  (Correction 2026-11-08, before any appearance score was recorded: the
  registration first said "the first segment", which on K_1_2 and P_1_1
  ends before 300 s where candidates name nothing; the first scoring run
  with that window failed its own check and was discarded.)
- **N_carry, S, G (per group)** use one encoder per trio trained on the
  pooled clean fragments of its three recordings (fragment = (recording,
  interval, slot); positives only within a recording), then embed each
  recording's crops with that encoder and run the unchanged naming, bridging
  and gap steps per recording. Stopping is Registration 4's rule applied to a
  pooled validation sample with three clusters; if the 0.91 target is not
  reached the patience rule or the 6000-step cap stops it. As a label-free
  diagnostic, the three per-recording centroids are matched across the three
  days of a trio by embedding distance and the margin between the best and
  second-best assignment is reported; no cross-day identity is asserted.

## Quantities, per recording, on recording time >= 300 s

Durations are the acquisition clock's left-held frame durations
(`interval_weights`); a fish-second is one fish for one second.

- **Q1 identity time.** For each candidate: correct, wrong and unknown
  fish-seconds against the ID-pass reference, per fish and pooled, over
  frames where the reference knows the identity and the chain has a
  position. Also the reference's own unknown time (danger and unusable) and
  the candidate's withheld time, in the same denominator.
- **Q2 name changes.** Count of times a slot's candidate name changes between
  consecutive reference-known frames, split into changes that the reference
  also makes (correct corrections) and changes it does not.
- **Q3 pair identity.** For the pairs F--M1, F--M2 and M1--M2: the fraction
  of reference-known time on which the candidate's labelled pair distance
  equals the reference's (difference under 2 mm; positions are identical,
  so this isolates labelling). This is the quantity that decides which male
  is near the female.
- **Q4 contact episodes.** Reference episodes where the nearer male's
  pectoral is within 15 mm of the female's for at least 0.1 s (gaps under
  0.1 s merged). For each episode: whether the candidate attributes the
  nearer male to the same individual on at least half of the episode's
  known frames, and whether it keeps the female. Candidate-only episodes were
  planned but are not evaluated by the current scorer; its counts concern
  reference episodes only. The 15 mm and
  0.1 s are proxies declared here; they are not the behavioural definition
  of contact, which is the downstream analysis's choice.
- **Q5 label-free diagnostics** of each appearance run: warning seconds,
  decided and undecidable boundaries, intervals named globally, partially or
  not at all, cluster silhouette and purity at the stop.

Reported against the delivery objective's thresholds, as a description and
not as an adoption decision: at most 1% wrong of named fish-time and at
least 90% of reference-known fish-time named, per recording and per
candidate. The per-group variant is compared with the per-recording variant
on Q1 and Q3 with the same thresholds. No threshold or setting is changed
after the first scores exist; a change is a new registration.

## Phases

1. Reference adapter and A0 scoring on all six recordings
   (`build_references.py`, `score_candidates.py`). No video decode. Done.
2. Appearance per recording on K_1_1, K_1_3, P_1_2, P_1_3 (four runs), then
   scoring of N/S/G on all six. Done.
3. Appearance per group (two encoders, `train_group_encoder.py`), scoring on
   all six, cross-day centroid diagnostic (`cross_day_centroids.py`). Done.
4. Report (`report_study.py`): tables per recording and per trio, identity
   timelines, and the list of contact episodes with their verdicts. Done;
   staged runs copied to the share by `archive_staged_runs.py`.

Outputs: `work/six_recording_identity/<run_id>/` under the data parent, with
`medaka3d.provenance` records, and appearance runs as described above.

## Limits declared now

- Six recordings are two trios on three consecutive days each: repeated
  measures of two groups of fish, not six independent replicates.
- The ID-pass is a tracker-assisted correction by one reviewer at up to 2x
  speed; its own error rate is unknown. Agreement with it is not truth.
  The Jingyi comparison on P_1_1 bounds this on one recording only.
- P_1_1 has informed every earlier appearance registration; K_1_2 informed
  the chain. Only K_1_1, K_1_3, P_1_2 and P_1_3 are new to both.
- Danger and unusable time of the review is excluded from the scored
  denominator and reported separately; the hardest moments may be there.
- A0's initial assignment is taken from the human review, so A0's wrong
  time measures drift after the start, not the cost of an unknown start.
- Added after execution: geometry and appearance indicate that the reviewer
  missed two male--male slot exchanges on P_1_2 (Phase 2; a hypothesis until
  the video check, softened 2026-11-09), so the review's error rate on
  male--male events is probably not zero and the agreement figures above are
  agreement with this reference, not accuracy. The cross-day consistency of
  the reviewer's male names is an observation about this reviewer.
- Added 2026-11-09: the scored (review-known, chain-valid) time is
  94.2--99.2% of the planned post-300 s fish-time (three fish times the
  duration), per recording; the named fractions in the tables are fractions
  of scored time, not coverage of the recording.

## Findings

### Phase 1 (2026-11-08): references built, geometry alone scored

Run `work/six_recording_identity/six_identity_20261108_r2/` (`references/`,
`scores/phase1_A0/`); r1 is the discarded first build (see References). All six review packages coincide with the chain's
same-index slots (fraction 1.0000 at 1 mm); every swap replay reproduced
its segment table. Review-known fish-time after 300 s: 94.2--99.2% of the
expected 5,401 fish-seconds per recording; the rest is the reviewer's danger
or unusable time (3.7--62.5 s of recording time) plus 1.6--6.5 fish-s the
chain has no position for. On P_1_1 the Jingyi review and the ID-pass agree
on 5,176 of the 5,184 fish-seconds both know (99.85%; F 99.77%, M1 99.77%,
M2 100%); Jingyi names 180 fish-s the ID-pass leaves unknown. On K_1_2 the
review names F at 249 of the 255 scored female clicks, a male at none, and
leaves 6 unknown.

A0, the chain's slot labels carried from the review's assignment in the
first scored segment (no correction afterwards):

| Recording | Correct fish-s | Wrong fish-s | Wrong fraction | Pair F--M1 / F--M2 / M1--M2 identity | Contact episodes correct / wrong male / mixed (of n) |
| --- | ---: | ---: | ---: | --- | --- |
| K_1_1 | 835 | 4,459 | 84.2% | 0.11 / 0.29 / 0.15 | 5 / 141 / 9 (155) |
| K_1_2 | 2,868 | 2,301 | 44.5% | 0.48 / 0.39 / 0.83 | 91 / 98 / 20 (209) |
| K_1_3 | 3,230 | 1,857 | 36.5% | 0.48 / 0.60 / 0.86 | 77 / 98 / 19 (194) |
| P_1_1 | 4,451 | 753 | 14.5% | 0.79 / 0.79 / 1.00 | 128 / 21 / 0 (149) |
| P_1_2 | 2,603 | 2,697 | 50.9% | 0.68 / 0.39 / 0.47 | 49 / 90 / 8 (147) |
| P_1_3 | 3,581 | 1,779 | 33.2% | 0.52 / 0.62 / 0.88 | 58 / 34 / 9 (101) |

A0 names everything (0 withheld) and meets the delivery thresholds on no
recording. The wrong time is drift after a slot swap that is never undone;
once swaps have permuted all three slots, every later second counts as
wrong, which is why K_1_1 reads 84%. Contact episodes (nearer male within
15 mm of the female for at least 0.1 s) number 101--209 per recording.

### Phase 2 (2026-11-08): appearance per recording, all six

`scores/phase2_per_recording/`. Registration 4's pipeline with one encoder
per recording (existing runs for K_1_2 and P_1_1; new runs
`appearance_six_20261108_{k11,k13,p12,p13}`, extract 21 min, prepare 4 min,
train 5--7 min, evaluate 3--8 min each with crops on local disk), scored
against the ID-pass on review-known time after 300 s:

| Recording | Candidate | Correct fish-s | Wrong fish-s | Withheld fish-s | Named | Wrong / named | Episodes correct / wrong male / withheld (of n) | Thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| K_1_1 | N_carry | 5,092 | 0.0 | 202 | 96.2% | 0.00% | 122 / 0 / 33 (155) | met |
| K_1_1 | S | 5,289 | 0.0 | 4.7 | 99.9% | 0.00% | 153 / 0 / 2 (155) | met |
| K_1_1 | G | 5,257 | 0.0 | 37 | 99.3% | 0.00% | 154 / 0 / 1 (155) | met |
| K_1_2 | N_carry | 5,046 | 3.0 | 119 | 97.7% | 0.06% | 183 / 0 / 26 (209) | met |
| K_1_2 | S | 5,157 | 2.2 | 9.3 | 99.8% | 0.04% | 207 / 0 / 2 (209) | met |
| K_1_2 | G | 5,157 | 7.2 | 4.3 | 99.9% | 0.14% | 207 / 1 / 1 (209) | met |
| K_1_3 | N_carry | 4,937 | 1.2 | 149 | 97.1% | 0.02% | 164 / 0 / 30 (194) | met |
| K_1_3 | S | 5,084 | 1.2 | 1.5 | 99.97% | 0.02% | 193 / 0 / 1 (194) | met |
| K_1_3 | G | 5,052 | 1.2 | 33 | 99.3% | 0.02% | 190 / 0 / 4 (194) | met |
| P_1_1 | N_carry | 5,034 | 0.0 | 170 | 96.7% | 0.00% | 134 / 0 / 15 (149) | met |
| P_1_1 | S | 5,204 | 0.0 | 0.1 | 100.0% | 0.00% | 149 / 0 / 0 (149) | met |
| P_1_1 | G | 5,140 | 0.0 | 65 | 98.8% | 0.00% | 147 / 0 / 2 (149) | met |
| P_1_2 | N_carry | 5,056 | 189 | 55 | 99.0% | 3.6% | 123 / 8 / 16 (147) | not met |
| P_1_2 | S | 5,106 | 191 | 4.0 | 99.9% | 3.6% | 138 / 9 / 0 (147) | not met |
| P_1_2 | G | 5,109 | 191 | 0.5 | 99.99% | 3.6% | 138 / 9 / 0 (147) | not met |
| P_1_3 | N_carry | 5,286 | 0.0 | 73 | 98.6% | 0.00% | 76 / 0 / 25 (101) | met |
| P_1_3 | S | 5,359 | 0.0 | 0.2 | 100.0% | 0.00% | 101 / 0 / 0 (101) | met |
| P_1_3 | G | 5,360 | 0.0 | 0.1 | 100.0% | 0.00% | 101 / 0 / 0 (101) | met |

Pair identity (Q3) is 0.998--1.000 on every pair of every recording except
P_1_2's two female--male pairs (0.946). The female is never wrong on any
recording against ID-pass. Against the Jingyi review, P_1_1's N_carry is wrong 0.048 fish-s
(Registration 4 reported 0.042 with its own matching), S 7.9 and G 0.85
fish-s; against the ID-pass with the 7.8 fish-s of review disagreement set
to unknown, all three are wrong 0.0. Candidate name changes coincide with
review segment changes everywhere except 4 male-only changes each on K_1_2
and K_1_3 and 4 on P_1_2 (below).

Added 2026-11-09, after an independent reading of the stored arrays (Q1
recomputed from the identity tracks for all 24 recording x encoder x
candidate conditions and found equal to the stored scores): G does not
withhold uniformly more than S. G names 5.0 fish-s more than S on K_1_2
(4.99 fish-s additional wrong time, inside gaps), 3.5 more on P_1_2 and 0.1 more on P_1_3,
and 32--65 fish-s less on K_1_1, K_1_3 and P_1_1; summed over the six
recordings S names 125.2 fish-s more correctly and 5.0 fish-s less wrongly
than G. Against the Jingyi review of P_1_1 the order reverses inside gaps:
S is wrong on 3.08% of its named gap time (7.9 fish-s) and G on 0.40%
(0.85 fish-s), consistent with Registration 4's gap-only verdict on this
recording (S 1.70% with the lab's own matching, above the registered 1%).
A preference for S as a default therefore rests on the trio-wide ID-pass
figures, not on a passed gap-only rule, and a decision to adopt it should
record that scope. Two implementation facts matter for reading the
candidates: `decide` has no margin threshold (a fully described interval
is named whenever its costs are finite, however small the margin), and
intervals named from two described fragments plus elimination
(`partial_two`, 9--12 per recording) count as anchors for S and G; G names
the interior of a gap with no evidence nodes exactly as S does.

**P_1_2, 1417.3--1512.8 s: a reference conflict, not yet adjudicated.** All
of P_1_2's wrong time is one episode in which the candidates name the two
males the other way round from the review (the female agrees throughout).
At both ends the chain's own geometry raises the identity-core warning
`large_step, nonidentity_nearest` (0.11 s each), the signature of a slot
exchange, and the appearance decides the male--male permutation at both
boundaries with margins 1.81 and 1.82; the review's segment 1349--1751 s
records no swap there. The chain positions favor the exchange interpretation:
between frames 235,245 and 235,246 (1417.322 to
1417.328 s, 6 ms) the two male slots move 9.9 and 19.4 mm if the slots are
kept but 3.6 and 6.5 mm if they are exchanged, with the two males 16 mm
apart; between frames 251,092 and 251,093 (1512.80 s) the figures are 22.7
and 14.2 mm kept against 5.2 and 3.8 mm exchanged, the males 19 mm apart.
A 20 mm displacement in 6 ms is 3 m/s. Geometry and appearance therefore
point the same way: the leading hypothesis is that the chain exchanged the
two males' slots at 1417.3 s and exchanged them back at 1512.8 s, and that
the single reviewer missed both male--male events (the review's own
`danger_frames` show the GUI flagged risk inside that segment). It is a
hypothesis until the video check: a reconstruction error at the two
crossing frames is not excluded, and the two encoders share the crops and
the processing structure, so their agreement is not an independent check
(wording softened 2026-11-09 after an independent reading of the arrays).
The score counts the review as right, as registered, and the reference
file is not changed; the two instants are recorded for a video check. If
the review is corrected there, P_1_2 meets the thresholds with S and G at
0.0 wrong, and no recording fails. Of the 190.7 wrong fish-s, 187.5 lie
inside anchors shared by S and G and 3.2 inside gaps (added 2026-11-09):
the conflict is about the interval naming at those two boundaries, not
about gap handling, so no gap method decides it.

### Phase 3 (2026-11-08): appearance per group, and identity across days

`scores/phase3_per_group/`, runs `appearance_group_20261108_*` (one per
recording, sharing a trio's encoder). Training on the GPU with crops on
local disk: K_1 top 217,936 clean crops in 1,508 fragments, stopped on the
0.91 target at step 500 (silhouette 0.957 on a pooled three-day validation
sample with three clusters); K_1 side 450,527 crops, 1,538 fragments, step
1000, 0.925, cross-camera median 0.039; P_1 top 247,723 crops, 1,003
fragments, step 700, 0.961; P_1 side 455,373 crops, 997 fragments, step
1200, 0.925, cross-camera 0.035. Both encoders took about 14 min for the
two models; each evaluate 3--8 min.

Within each recording the per-group encoder gives the same identity as the
per-recording encoder: correct, wrong and withheld fish-seconds of N_carry
and S are identical on all six recordings to the tenth of a second, and G
differs by at most 2 fish-s of withheld time (K_1_1 36.9 against 36.7,
P_1_1 63.3 against 65.0). The P_1_2 male--male episode is named the same
way by the group encoder, so two independently trained encoders agree
with the geometry against the review there.

Pooled over each trio (`report/trio_summary.md`):

| Trio | Candidate | Correct fish-s | Wrong fish-s | Withheld fish-s | Named | Wrong / named | Episodes correct / wrong male / withheld (of n) | Recordings meeting thresholds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| K_1 | A0 | 6,932 | 8,618 | 0 | 100% | 55.4% | 173 / 337 / 0 (558) | 0 of 3 |
| K_1 | N_carry | 15,076 | 4.3 | 470 | 97.0% | 0.03% | 469 / 0 / 89 (558) | 3 of 3 |
| K_1 | S | 15,531 | 3.4 | 15.5 | 99.9% | 0.02% | 553 / 0 / 5 (558) | 3 of 3 |
| K_1 | G | 15,467 | 8.4 | 75 | 99.5% | 0.05% | 551 / 1 / 6 (558) | 3 of 3 |
| P_1 | A0 | 10,635 | 5,229 | 0 | 100% | 33.0% | 235 / 145 / 0 (397) | 0 of 3 |
| P_1 | N_carry | 15,376 | 189.5 | 299 | 98.1% | 1.22% | 333 / 8 / 56 (397) | 2 of 3 |
| P_1 | S | 15,669 | 190.7 | 4.3 | 100.0% | 1.20% | 388 / 9 / 0 (397) | 2 of 3 |
| P_1 | G | 15,610 | 190.7 | 64 | 99.6% | 1.21% | 387 / 9 / 1 (397) | 2 of 3 |

N_carry and S agree between variants; G differs slightly in withheld time
and episode attribution (see `report/trio_summary.md` for both variants). All of
P_1's wrong time and all nine wrong-male episodes lie inside the P_1_2
conflict described under Phase 2.

**Identity across the three days of a trio (label-free).** With one
encoder per trio, the three per-recording centroids (mean embedding of the
clean lower-risk crops under each N_carry name) were matched across every
pair of days and every camera view (`report/cross_day_centroids.json`). In
all 18 pairings the best one-to-one assignment costs 0.03--0.12 (sum of the
three matched centroid distances; single distances 0.003--0.093) while the
second-best assignment costs 1.78--2.24 more, and the smallest separation
between two centroids within one day is 0.91--1.13. The same three
appearance clusters therefore recur on all three days of each trio with a
margin about twenty times the matching cost. Read through the review's
labels, every one of the 18 assignments maps F to F and each male to the
reviewer's same name (M1 to M1, M2 to M2): the reviewer named the males
consistently across days, and the appearance agrees. This is a diagnostic,
not a scored claim; a video check of a few crops per day is the cheap
confirmation.

### Phase 4 (2026-11-08): report

`report/summary_table.md` (one row per recording, variant and candidate),
`report/trio_summary.md`, `report/<recording>_timeline.png` (one row per
candidate, green where all three review-known slots are named correctly,
orange where any is wrong, blue where any is withheld, grey where the
review does not know; review segment edges marked), and
`scores/*/contact_episodes.csv` with every contact episode and its
verdict. The P_1_2 timeline shows one orange block, 1417--1513 s, on every
appearance row.

### Conclusion

On the quantities the downstream contact analysis consumes, geometry alone
is far from the human-corrected identity: after the first slot exchange its
labels drift and never recover, so 14.5--84% of the review-known fish-time
is wrong and only 408 of 955 contact episodes have the right male. Adding
appearance naming on the chain's own slots (Registration 4's pipeline,
trained label-free per recording or per trio) reproduces the human
correction on five of the six recordings with 0--7 fish-seconds wrong and
96--100% of the time named; S names 99.8--100% with at most 3.4 fish-s wrong
per trio where the review is right, and attributes 941 of 955 episodes
correctly, 5 withheld, with the 9 "wrong" all inside the P_1_2 conflict. On
the sixth recording the only disagreement is a 95 s male--male episode where
geometry and two encoders agree against the single reviewer. The reviewer's
names are also consistent across the three days of each trio, and the
per-trio encoders recover that cross-day identity label-free.

On 2026-11-09 the supervisor selected S as the experimental default and G as
a separate optional command/output. S's withheld identities must never be
filled from G. The [operating decision](../../docs/decisions/2026-11-09-separate-s-g-outputs.md)
documents the evidence, limits, and the separate `export_identity_s.py` and
`export_identity_g.py` commands. Each exports exactly one checksummed frozen
candidate to a new directory, preserving its bytes and withheld values.
Historical comparison runs still produce all registered candidates; these
exports do not recompute or combine them. This study changes no registration
of the appearance lab and promotes nothing. Video adjudication of P_1_2 and
P_1_1 reference disagreements remains outstanding.

## Artifacts

Study run `work/six_recording_identity/six_identity_20261108_r2/` under the
data parent: `references/` (per-recording reference arrays, female anchors
of K_1_2, `summary.json`, provenance), `scores/phase1_A0/`,
`scores/phase2_per_recording/`, `scores/phase3_per_group/` (each with
`scores.json`, `summary_table.csv`, `contact_episodes.csv`, provenance),
`report/`. The discarded first build is `six_identity_20261108_r1/`.
Appearance runs: `appearance_six_20261108_{k11,k13,p12,p13}` and
`appearance_group_20261108_{k11,k12,k13,p11,p12,p13}` under
`work/appearance/<recording>/` (K_1_2, P_1_1 group runs under their
recording's usual parent, `jingyi/P-1-1/` for P_1_1). Runs computed on local
disk were copied there with `staged_copy_manifest.json` (SHA-256 per file);
their intermediate crop arrays (`crops_all/`, `crops/`, `warning_crops/`,
`masked/`, `warning_masked/`) stayed local and are regenerable from video
with the recorded code; the prepared `clean/` and `clean_warning/` inputs,
models, embeddings and evaluations were copied.

Portability update, 2026-11-09: `config.toml` now reads the fixed shared copies
of the eight staged runs. All 24 N/S/G candidate files were checked against
their local counterparts and `staged_copy_manifest.json` hashes before this
location change. Run IDs and candidate bytes are unchanged. For local training
or archiving, set `appearance_staging_dir` in `configs/paths.local.toml` to the
existing staging directory; the training configurations no longer embed a
personal path. Existing run snapshots and shared outputs are not rewritten.

Commands, in order, from the repository root with the project virtualenv:

```powershell
# Phase 1
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\build_references.py
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\score_candidates.py --variant per_recording --candidates A0 --tag phase1_A0
# Phase 2: for each of k11 k13 p12 p13, steps bundle extract prepare train evaluate
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py <step> --recording <key> --config analyses\20261031_p11_identity_evaluation\config_appearance_six_v1.toml
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\score_candidates.py --variant per_recording --tag phase2_per_recording
# Phase 3: for each of g_k11 g_k12 g_k13 g_p11 g_p12 g_p13, steps bundle prepare; then per group; then evaluate per key
.venv\Scripts\python.exe analyses\20261031_p11_identity_evaluation\run_appearance_clean.py <step> --recording <key> --config analyses\20261031_p11_identity_evaluation\config_appearance_group_v1.toml
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\train_group_encoder.py --group K_1 --config analyses\20261031_p11_identity_evaluation\config_appearance_group_v1.toml
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\score_candidates.py --variant per_group --tag phase3_per_group
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\cross_day_centroids.py --scores-tag phase3_per_group
# Phase 4
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\report_study.py --tags phase2_per_recording phase3_per_group
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\archive_staged_runs.py --skip-crops
```

No extraction output, review file, reference or earlier run was modified.
