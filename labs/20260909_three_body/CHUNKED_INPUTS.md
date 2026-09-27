# Preparing resumable independent mask inputs

`prepare_chunked_source_masks.py` saves per-raw-image Detectron2 inference into independent
chunks. It does not read SLEAP, initial poses, individual IDs or human ground-truth labels.
It also does not run tracking, reinitialize it or restore tracking state.

## CLI and API

The following example creates only the first chunk in a new output location. It names the
protocol to run explicitly and uses the existing dedicated Detectron2 environment.

```powershell
& ..\medaka-mating-pipeline\.venv-detectron2\Scripts\python.exe -B labs/20260909_three_body/prepare_chunked_source_masks.py --protocol work/three_body/chunked_input_smoke_protocol.json --source-cache work/three_body/source_cache --out work/three_body/inputs/new_chunked_smoke --max-new-chunks 1
```

To continue, drop `--max-new-chunks 1` from the same command. The Python API is
`run(protocol_path, source_cache, out, max_new_chunks=None)`.
`max_new_chunks` limits only the number of newly created chunks; reuse of verified chunks is
not counted. If the whole is incomplete, `PreparationPaused` is raised and the CLI returns the
intended exit code 2. `incomplete.json`, `final_sequence_path: null` and an incomplete run
provenance remain. Completion records of saved chunks are kept. The outer exit code of
PowerShell etc. and the Python exit code are recorded separately.

The full 30-minute conditions are registered in [long_input_protocol.json](long_input_protocol.json).
49800:348600, 166 fps, 298800 frames are split into 180 chunks of 1660 frames each.
At the time of writing, neither the full 30 minutes nor the cost measurement on the first
10 seconds of the registered long window had been run.
See [LONG_INPUT_PROTOCOL.md](LONG_INPUT_PROTOCOL.md) for protocol details.

## Storage and reuse

```text
out/
  chunks/f333519-333521/attempt_0001/
  chunks/f333521-333523/attempt_0001/
  runs/run_0001/incomplete.json
  runs/run_0002/summary.json
  assemblies/attempt_0001/manifest.json
```

No attempt is overwritten. Reuse requires the 3 completed views, the exact contiguous frame
range, the protocol, the SHAs of inputs, model, configuration and code, the saved execution
environment, and consistency of all outputs.
Failed attempts and attempts under different conditions are kept, and a new one is created
under the next number. If configuration or code is changed, existing chunks are not reused as
the same experimental condition.

Only once all chunks are present are they joined, keeping array values, order and dtypes.
Only `frame_offsets` and `mask_byte_offsets` are rewritten to cumulative positions.
All detections in each frame are saved; the number of detections is not limited by the
assumption of 3 fish. Zero detections are also kept as empty observations with
`view_available=True`.

## Conditions for reading a completed input

`final_sequence_path` in a successful `runs/run_NNNN/summary.json` is the path of the completed
assembly relative to `out`. It is treated as complete only when the run and assembly provenance
are `status: ok` and the SHAs of the manifest, chunk index, referenced chunks and every input and
output match. A few completed chunks alone do not make the whole complete.

The final manifest is in a `SourceSequenceReader(sequence_path, sleap_root)`-compatible format
with `raw_storage: original_source_video`.
It verifies the source-video frame positions, 166 fps, 1200×1200 and the SHA of each BGR image,
and reads only the specified range. Because the reader itself can also read unsealed
descriptors for internal verification, the caller must check the completion record first. This
does not mean it has been integrated automatically into the existing tracking CLI.

## What was checked on real video

6 frames × 3 views of 333519:333525 were newly inferred as 3 chunks of 2 frames each.
The run first stopped after 1 chunk, then created the remaining 2 chunks, and finally reused all
chunks. The checksums of the first chunk and of the completed assembly did not change.

- The 18 native BGR images matched the old FFV1 input byte for byte.
- 55 detections; all 13 packed mask arrays matched in dtype, shape and every byte.
- top has 3 detections every frame, side1 has 4 detections every frame, side2 has
  3,2,2,2,2,2 detections.
- The background matched the median of the 9 registered frames 5000,10000,…,45000 in all 3 views.
  It differs from the old background in all 3 views. This change is stated explicitly as a
  separate condition.
- The before/after-audit SHAs of all inputs and code matched. 0 bytes of video payload were
  copied.

Results: [audit summary](../../work/three_body/chunked_source_smoke_qa_v1/summary.json) and
[audit provenance](../../work/three_body/chunked_source_smoke_qa_v1/audit.provenance.json).
The 75 implementation-side CPU tests and 3 independent additional regression tests also pass.
This small real window had no zero-detection frame, so retention of zero detections was
confirmed by CPU tests. This is not a result showing the detector's general accuracy or
long-duration individual-ID maintenance.

## Operational notes

The source video is a hardlink, so it shares the same file body as its link target. Always treat
it as read-only and never write to it through either path. Videos are not re-encoded or copied.

Full SHAs are computed at start, at each chunk seal, at joining and at call completion, so there
is a cost of reading about 15.8 GB of source input. The elapsed time of a short window is not
called D2 inference speed. This CPU audit took 54.60 s, of which image decode and comparison
took 1.43 s. Concurrent execution also has an effect, so the cost of long intervals is measured
separately.
