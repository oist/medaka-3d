# Continuation runs on long inputs

`run_long_continuation.py` advances from the registered initial pose at 49800 to the end of the
completed input chunks with the same tracking state. The full target is `[49800,348600)`,
166 fps, 298800 frames. The independent-D2 storage unit is 1660 frames (10 s). Poses are not
initialized at file boundaries.

The first real-data trial was limited to the 1660 frames of `[49800,51460)`. It is not a trial
representing completion of the full 30 minutes, or confirmation of ground truth for the
biological identities of the 3 fish. The run conditions were fixed in
[long_tracking_protocol.json](long_tracking_protocol.json) before tracking.

## Command for the completed first trial

From PowerShell at the repository root, specify a new output location that does not overwrite
existing outputs. The output location below is the record of the first run; do not re-run under
the same name.

```powershell
& .venv/Scripts/python.exe -B labs/20260909_three_body/run_long_continuation.py `
  --seed-root work/three_body/long_seed_registration_49800_v1 `
  --input-stage-root work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --through-frame 51460 `
  --output work/three_body/long_continuation_first1660_v1
```

Preprocessing uses the dedicated Detectron2 environment; tracking uses this repository's
CUDA-enabled environment. No new package installation is needed. Because the SHAs of the
initial-value registration, SL, refractive calibration, source video, D2 model, background rule,
saved masks and implementation are checked, the first progress display takes a while.

## Continuing to later intervals

The command that prepares only the next 10 seconds of independent masks is as follows. Chunks
that already pass the checks are reused, and the next incomplete chunk is saved to a new
attempt. If a single limited run does not complete all 180 chunks, the parent process returns
`PreparationPaused` / `incomplete` and exit code 1. This means something different from the seal
of an individual successful chunk.

```powershell
& ../medaka-mating-pipeline/.venv-detectron2/Scripts/python.exe -B `
  labs/20260909_three_body/prepare_chunked_source_masks.py `
  --protocol labs/20260909_three_body/long_input_protocol.json `
  --source-cache work/three_body/source_cache `
  --out work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --max-new-chunks 1
```

Once the first run's success seal and the next chunk `[51460,53120)` are both present, the
following can be run. If the same output location exists, use a different name. Do not repeat
this command before verification passes.

```powershell
& .venv/Scripts/python.exe -B labs/20260909_three_body/run_long_continuation.py `
  --seed-root work/three_body/long_seed_registration_49800_v1 `
  --input-stage-root work/three_body/inputs/long_source_masks_49800-348600_v1 `
  --through-frame 53120 `
  --resume work/three_body/long_continuation_first1660_v1 `
  --output work/three_body/long_continuation_through53120_v1
```

`--through-frame` accepts only the end of a registered chunk. The fixed input binding stays the
same as future chunks are added. If the seal, content or order of chunks the parent consumed
changes, the run is refused. Resumption from a failed parent, or from saved candidate
coordinates alone, is not accepted.

## Reading the outputs

- `continuation.provenance.json`: success/failure of the whole phase, and the SHAs of inputs,
  implementation and outputs.
- `checkpoint/`: saves every candidate's parent history, cumulative cost, velocity, missing and
  hold state, and previous image.
- `candidate.npz`: the 21 arrays selected at the current end. Earlier parts may also change with
  later video.
- `consumed_chunks.json`: the input evidence up to this run. On restore, an exact match with the
  earlier prefix is required.
- `summary.json`: tracking time, observation and acceptance counts, initialization and restore
  counts, and whether the result is provisional.
- `progress.jsonl`: processed native frames and elapsed time.

`full_sequence_complete` is true only when the run has completed through 348600.
For a result with `prefix_backtrace_provisional` true, the winning path alone is not reused as
the next initial value. Restore is done from the checkpoint containing all candidates. The cost
of saving, verifying and restoring history grows with processed time.

`accepted_pectoral_count` is a conditional count of outputtable points, not a count of correct
identities. Scenes where identity became ambiguous are checked against actual human records
obtained with the [raw-frame review procedure](../../work/three_body/minute_reviews/REVIEW_INSTRUCTIONS.md).
The fixed 9 background frames differ from the background of the old one-minute experiment, so
an algorithm improvement rate is not computed from this new long interval and the old
one-minute values alone.
