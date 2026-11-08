# Separate S and G identity candidate outputs

Date: 2026-11-09. Status: supervisor-approved operating policy for experimental
outputs; no identity algorithm promotion or new accuracy certification.

## Decision

Use S as the experimental default candidate. Keep G as an optional comparison,
through a separate command and a separate output directory/file. Never fill S's
withheld identities with G, merge the candidates, or fall back to another
candidate when the requested output is unavailable. A withheld code remains -1.

The historical appearance runners continue to produce the registered comparison
artifacts. The operational entry points export one frozen candidate each; they
do not retrain, recompute identity, or rewrite historical runs:

```powershell
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_s.py --run '<explicit evaluated run>' --sha256 '<recorded S SHA-256>' --output '<new S export directory>'
.venv\Scripts\python.exe analyses\20261108_six_recording_identity\export_identity_g.py --run '<same explicit evaluated run>' --sha256 '<recorded G SHA-256>' --output '<different new G export directory>'
```

Take each expected hash from the frozen appearance run's
`appearance_clean_evaluate.provenance.json`, under
`outputs.identity_tracks_S.sha256` or `outputs.identity_tracks_G.sha256`.
The commands read `evaluation/identity_tracks_S.npz` and
`evaluation/identity_tracks_G.npz`, respectively. Each writes a byte-identical
NPZ, `identity_policy.json`, and its own provenance record. Existing output
directories, changed checksums, malformed arrays, and missing candidates fail
explicitly. There is no algorithm-selection or fill-from-G flag. G can be absent
when no anchors disagree and the registered switch rate is zero.

These are candidate exports, not the completed collaborator-ready dataset.
Physical F/M1/M2 naming, correspondence across days, observation/quality masks,
and accepted identity support still require the delivery evidence described in
[the dataset plan](2026-09-30-dataset-delivery.md). Preserve full post-300 s
coverage accounting, including unreviewed, missing, withheld and unprocessed
time. An export does not resolve a reference conflict or certify inherited
physical names. In accepted identity-bearing trajectories, unsupported
correspondence must remain null; the experimental candidates retain evidence
for review.

## Evidence and limits

The [six-recording study](../../analyses/20261108_six_recording_identity/README.md)
compares two trios on three days each. The independent reading on 2026-11-09
recomputed Q1 from the stored NPZ arrays for all 24 combinations of six recordings,
two encoders, and S/G, reproducing the saved values. No video adjudication or
retraining was performed.

Against ID-pass, per-recording S has 125.197 more correct fish-seconds and 4.989
fewer wrong fish-seconds in total than G. G nevertheless names more time on
K_1_2 (+5.025 fish-s), P_1_2 (+3.500), and P_1_3 (+0.108). Almost all of K_1_2's
net gain is additional disagreement with the reference (+4.989 fish-s).

Against the alternative Jingyi reference on P_1_1, the current study scorer gives
S 7.935 wrong fish-s and G 0.850. Restricting these same arrays to gaps gives
3.078% and 0.395% wrong among named gap time. Reference conflicts therefore matter
to the comparison; the zero errors against ID-pass are not established accuracy.
The earlier Registration 4 scorer gave S 1.70% wrong on gap time and did not
adopt it under the registered 1% limit. This operating choice does not rewrite
that result. The original G rule required more named gap time than S and at
most 1% wrong among named gap time, not zero additional wrong time relative to S.

P_1_2's 190.700 fish-s of disagreement remains unadjudicated. Of this, 187.531
fish-s is in anchors shared by S and G, with 3.169 in gaps. Geometry favors a
male-slot exchange at each reported boundary, but reconstructed positions and
encoders trained on overlapping data are not independent video adjudication.
Keep the original reference and scores; a correction requires a versioned review.

Implementation details relevant to interpretation:

- Finite naming costs yield a minimum-cost assignment even at zero margin;
  there is no ambiguity-margin rejection. Two described slots can name the
  third by elimination, and those intervals are anchors for both S and G.
- G's 0.99 cutoff is a conditional node posterior using pseudo-labelled
  temperature fitting and fixed anchors. It is not an independently calibrated
  99% correctness claim. Matching nodes propagate names between samples; a gap
  with no evidence nodes also inherits names when the anchors agree.
- Review-known time covers 94.2--99.2% of full expected fish-time. Named fractions
  within that subset are not full-window verified coverage. Contact Q4 scores
  attribution within reference episodes; candidate-only episode false positives
  are not evaluated by the current implementation.

Before extending trusted output, freeze the procedure and audit both accepted
and withheld cases, including elimination anchors, agreeing-anchor gaps and close
encounters. K_2/P_2 test later recordings of the same animals, not new trios.
P_1_1 reference conflicts and the P_1_2 boundaries remain targeted review needs.

## Portable storage configuration

The study now names the fixed shared copies of the eight locally staged runs,
using paths relative to the configured data parent. Their candidate checksums
must agree with the staged-copy manifests; this changes location, not run ID or
research input version. Executed run snapshots remain untouched. Future local
training resolves `appearance_staging_dir` from `configs/paths.local.toml`; no
personal staging directory is embedded in the tracked analysis configurations.

## Validation

Regression checks cover separate exports, S abstention preservation, missing-G
failure without fallback, checksum rejection, refusal to overwrite existing
outputs, and invalid or ambiguous names. Both CLI commands also exported the
fixed P_1_2 candidates with byte-identical results. The
[checkpoint addendum](2026-09-25-source-checkpoint.md#addendum-2026-11-09-identity-research-and-separate-sg-exports)
records notebook, source-copy, lint, lockfile and full-suite checks, including
the eight existing full-suite failures and the formatting backlog.
