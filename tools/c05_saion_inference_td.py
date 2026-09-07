"""C05 — full-length **top-down** 3-node inference for K_1_{1,2,3}, all three views.

The mirror of `s16`, which ran the bottom-up arm. Same 3 experiments x 3 views x
348,600 frames = **3,137,400 inferences**, same corpus, same chunking, so the two
architectures can be compared frame for frame rather than sample against sample.

=============================================================================
WHY THIS RUNS AT ALL — the decision it re-opens, stated before it runs
=============================================================================

`s16` ran bottom-up **only**, on the grounds that the 2026-08-10 reviewed labels
improved that arm and made the top-down arm *worse*. Two decisions stand behind
that and this run re-opens both deliberately, at the experimenter's request:

* **2026-08-09.** Top-down 3-node's shared-node rate on the side views was
  5.6-7.4% against bottom-up's 0.00-0.01%, and its 1st-percentile tail-to-tail
  distance was 0.45 px — two predicted instances placing a tail on the same
  animal. Bottom-up cannot do this: peak NMS plus PAF grouping floors it.
* **2026-08-13.** Retrained on the same 543 reviewed moments these checkpoints
  come from, top-down's mouth-linked swap rate *rose* in all three views
  (significantly on side1, p=0.044) while bottom-up's fell 73% on top.

**What this run can and cannot settle.** It measures whether those part-mixing
failures propagate into *identity* failures once three cameras and a 3D
correspondence are in the loop — a question neither earlier decision could ask,
because neither had a reconstruction. Two predicted instances sharing a tail in
one camera may be absorbed by the other two cameras, or may poison the
correspondence; that is a measurement, and it is new evidence.

It is **not** ground truth. There is no human M1/M2 track anywhere in this
project. A tracking metric that reads better for top-down does not overturn a
detection-quality measurement; it stands beside it, and where the two conflict
the conflict is the finding. `c06` fixes that clause in its decision rule before
any number exists.

=============================================================================
THE MODELS
=============================================================================

Top-down is **two** checkpoints, not one: round 8's centroid head locates the
animals and the `s14` 3-node centered-instance head places the nodes. The
centroid is shared with the production 2-node model, which is why top-down's
cardinality is identical to round 8's and only node placement can differ.

Bottom-up is one checkpoint. That asymmetry is a property of the architectures,
not of this comparison.

=============================================================================
CLUSTER CONSTRAINTS, VERIFIED ON THIS ACCOUNT
=============================================================================

Unchanged from `s16` and re-stated because each cost a failed submission:
partition **`largegpu`** (`gpu-a100` is rejected for this association),
**MaxWall 12 h** (over-asking pends forever on
`AssocMaxWallDurationPerJobLimit`), **`/bucket` read-only from compute nodes**,
and `#SBATCH` directives are read by Slurm rather than the shell so a
`${VAR:-default}` in one is a literal.

Measured locally on an RTX 5070 Ti, top-down 3-node runs at **23.8 fps (top)**
and **24.6 fps (side)** at batch 4 — roughly 60% of bottom-up's rate, because
every frame runs two networks. At 24 fps a task's 392,175 frames take **4.5 h**,
inside the 12 h wall.

Run::

    .venv\\Scripts\\python.exe tools/c05_saion_inference_td.py --plan
    .venv\\Scripts\\python.exe tools/c05_saion_inference_td.py --submit
    .venv\\Scripts\\python.exe tools/c05_saion_inference_td.py --status
    .venv\\Scripts\\python.exe tools/c05_saion_inference_td.py --collect
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from medaka3d import load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance

EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
N_FRAMES = 348_600
CHUNK = 25_000
TASKS = 8

PARTITION = "largegpu"
BUCKET = "/bucket/StephensU/medaka"
WORK = "/work/StephensU/medaka3d/c05_td_inference"
#: Where the predictions land on the bucket, mirroring s16's layout so `c01`
#: reads both runs with the same code.
DESTINATION = f"{BUCKET}/sleap_output_3node/c05_td_K_1_all"
#: **`sleap-nn/0.3.1` no longer exists on saion.** `s16` loaded it on 2026-08-11;
#: by 2026-08-14 the modulefile was gone and the install directory had been
#: emptied to 2.5 MB with no `bin/`. 0.3.2 (15 GB) and 0.3.3 (16 GB) are intact.
#: **0.3.3 is the current version and is what runs here.** Pinning to the
#: nearest survivor of the withdrawn 0.3.1 was considered and rejected: chasing a
#: version that no longer exists buys nothing, and 0.3.3 is what the cluster will
#: still have next month.
#:
#: That leaves a **runtime confound**: the bottom-up predictions this run is
#: compared against were produced by 0.3.1. It is controlled by measurement
#: rather than assumed away — `--control` re-runs bottom-up on three chunks under
#: 0.3.3 and `c06` compares them against the 0.3.1 originals frame by frame,
#: exactly as `s10` verified that batch size was a throughput knob. The same
#: control also proves the output is readable by the local `sleap-io==0.9.2`
#: pin, which a newer writer is not obliged to guarantee.
PYBIN = "/apps/unit/ReiterU/sleap-nn/0.3.3/tools/sleap-nn/bin/python"
SLEAP_NN_VERSION = "0.3.3"
#: The interpreter is addressed absolutely and `module load` is **not** used.
#: It never put python on `PATH` even when it existed — `s16` worked because of
#: the absolute path, not because of the module — and under `set -euo pipefail` a
#: module that has since been withdrawn kills the job before it starts. That is
#: how this run first failed.
BOTTOMUP_MODELS = {
    "top_cam": [
        f"{BUCKET}/sleap_models/retrain_20260810_3node_reviewed"
        f"/top/s14.bottomup_3node.reviewed_top_f24_ms64_paf8"
    ],
    "side1_cam": [
        f"{BUCKET}/sleap_models/retrain_20260810_3node_reviewed"
        f"/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8"
    ],
    "side2_cam": [
        f"{BUCKET}/sleap_models/retrain_20260810_3node_reviewed"
        f"/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8"
    ],
}

ROUND8 = f"{BUCKET}/sleap_models/retrain_20260722_round8"
REVIEWED = f"{BUCKET}/sleap_models/retrain_20260810_3node_reviewed"

#: Centroid first, centered-instance second — `sleap_nn.predict` composes them in
#: the order given, and reversing it is not an error it reports.
MODELS_BY_VIEW = {
    "top_cam": [
        f"{ROUND8}/top/runs/round8_260723_strictmouth.centroid_n1382",
        f"{REVIEWED}/top/s14.topdown_3node.reviewed_top",
    ],
    "side1_cam": [
        f"{ROUND8}/side/runs/round8_260723_strictmouth.centroid_n2764",
        f"{REVIEWED}/side/s14.topdown_3node.reviewed_side",
    ],
    "side2_cam": [
        f"{ROUND8}/side/runs/round8_260723_strictmouth.centroid_n2764",
        f"{REVIEWED}/side/s14.topdown_3node.reviewed_side",
    ],
}
VIEW_TO_MP4 = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4",
               "side2_cam": "cam2_side.mp4"}
#: 4, for the reason s10 measured on the bottom-up head and s16 carried: it is a
#: throughput knob, verified not to change the arithmetic. One value for every
#: model so no arm gets different treatment.
BATCH = 4


def build_chunks() -> list[dict]:
    """Frame ranges, round-robin across tasks — identical geometry to `s16`.

    Chunk *i* goes to task ``i % 8`` rather than contiguous blocks, because top
    runs slower than side and a contiguous split would strand the slow work on
    one task while the rest idle.
    """
    chunks = []
    for experiment in EXPERIMENTS:
        for view in VIEWS:
            for start in range(0, N_FRAMES, CHUNK):
                end = min(start + CHUNK, N_FRAMES)
                chunks.append({
                    "experiment": experiment, "view": view,
                    "start": start, "end": end,
                    "video": f"{BUCKET}/medaka_202512/{experiment}/{view}/"
                             f"{VIEW_TO_MP4[view]}",
                    "models": MODELS_BY_VIEW[view],
                    "out": f"{WORK}/predictions/{experiment}__{view}"
                           f"__frames_{start}-{end}.slp",
                })
    for index, chunk in enumerate(chunks):
        chunk["task"] = index % TASKS
    return chunks


RUNNER = '''"""Run one array task's chunks of top-down 3-node inference."""
import json
import os
import sys
import time
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
task = int(os.environ["SLURM_ARRAY_TASK_ID"])
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4

mine = [c for c in manifest["chunks"] if c["task"] == task]
print(f"task {task}: {len(mine)} chunks, "
      f"{sum(c['end'] - c['start'] for c in mine)} frames", flush=True)

import sleap_io as sio
import sleap_nn
import torch

print(f"cuda {torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)

done = skipped = 0
for chunk in mine:
    out = Path(chunk["out"])
    if out.is_file() and out.stat().st_size > 0:
        skipped += 1
        print(f"  SKIP {out.name} (exists)", flush=True)
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = list(range(chunk["start"], chunk["end"]))
    video = sio.Video.from_filename(chunk["video"])
    blank = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=i, instances=[])
                        for i in frames],
        videos=[video],
        skeletons=[sio.Skeleton(["mouth", "pectoral", "tail"])],
    )
    started = time.time()
    predicted = sleap_nn.predict(
        blank, model_paths=chunk["models"], device="cuda", batch_size=batch)
    # Write to a temporary name and rename, so an interrupted task never leaves
    # a half-written file that the resume check would treat as done.
    tmp = out.with_suffix(".partial")
    sio.save_slp(predicted, str(tmp), embed=False)
    tmp.replace(out)
    elapsed = time.time() - started
    done += 1
    print(f"  {out.name}: {len(frames)} frames in {elapsed/60:.1f} min "
          f"({len(frames)/elapsed:.1f} fps)", flush=True)

print(f"task {task} COMPLETE: {done} written, {skipped} already present", flush=True)
'''


def sbatch_script() -> str:
    return "\n".join([
        "#!/bin/bash -l",
        "#SBATCH --job-name=c05_td_infer",
        f"#SBATCH --partition={PARTITION}",
        f"#SBATCH --array=0-{TASKS - 1}",
        "#SBATCH --gres=gpu:a100:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        # The association MaxWall. Asking for more pends forever.
        "#SBATCH --time=12:00:00",
        f"#SBATCH --output={WORK}/logs/%x-%A_%a.out",
        f"#SBATCH --error={WORK}/logs/%x-%A_%a.err",
        "set -euo pipefail",
        "",
        'echo "task $SLURM_ARRAY_TASK_ID on $(hostname)"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        f'MANIFEST="${{1:-{WORK}/manifest.json}}"',
        f'"{PYBIN}" {WORK}/run_inference.py "$MANIFEST" {BATCH}',
    ]) + "\n"


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "saion", f"bash -lc '{command}'"],
        check=check, capture_output=True, text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--plan", action="store_true", help="Report the split only.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true", help="Queue state and chunk count.")
    parser.add_argument(
        "--control", action="store_true",
        help="Re-run three BOTTOM-UP chunks under this runtime, to measure the "
             "0.3.1-vs-0.3.3 confound rather than assume it away.",
    )
    parser.add_argument("--collect", action="store_true",
                        help="Copy /work predictions to the bucket and verify sizes.")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Measured local throughput, for the time estimate only.")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "c05_saion_inference_td"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks()
    total = sum(c["end"] - c["start"] for c in chunks)
    per_task = {t: sum(c["end"] - c["start"] for c in chunks if c["task"] == t)
                for t in range(TASKS)}
    groups = sorted({ExperimentId.parse(e).group for e in EXPERIMENTS})

    if args.status:
        queue = ssh("squeue -u $USER -o '%.10i %.12P %.14j %.2t %.10M %R'", check=False)
        print(queue.stdout or queue.stderr)
        counted = ssh(f"ls {WORK}/predictions 2>/dev/null | wc -l; "
                      f"ls {WORK}/predictions/*.partial 2>/dev/null | wc -l", check=False)
        lines = counted.stdout.split()
        if len(lines) >= 2:
            print(f"chunks written: {lines[0]}/{len(chunks)}   .partial: {lines[1]}")
        return

    if args.control:
        # One chunk per view, the same frame ranges s16 already wrote under
        # 0.3.1. Identical model, identical hardware class, identical batch —
        # only the runtime differs, which is the whole point.
        control = [
            dict(c, models=BOTTOMUP_MODELS[c["view"]], task=index,
                 out=f"{WORK}/control/{c['experiment']}__{c['view']}"
                     f"__frames_{c['start']}-{c['end']}.slp")
            for index, c in enumerate(
                next(x for x in chunks if x["view"] == view and x["start"] == 100_000)
                for view in VIEWS
            )
        ]
        manifest = {"chunks": control, "architecture": "bottomup_3node_reviewed",
                    "runtime": SLEAP_NN_VERSION, "batch_size": BATCH}
        (out_dir / "control.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out_dir / "run_inference.py").write_text(RUNNER, encoding="utf-8", newline="\n")
        (out_dir / "c05_infer.sbatch").write_text(sbatch_script(), encoding="utf-8", newline="\n")
        ssh(
            f"mkdir -p {WORK}/logs {WORK}/control && "
            f"cp {BUCKET}/work/c05_saion_inference_td/control.json {WORK}/ && "
            f"cp {BUCKET}/work/c05_saion_inference_td/run_inference.py {WORK}/ && "
            f"cp {BUCKET}/work/c05_saion_inference_td/c05_infer.sbatch {WORK}/"
        )
        result = ssh(
            f"cd {WORK} && sbatch --array=0-2 --job-name=c05_control "
            f"--time=01:00:00 {WORK}/c05_infer.sbatch {WORK}/control.json"
        )
        print(result.stdout.strip())
        print(f"3 bottom-up chunks of {CHUNK:,} frames under sleap-nn {SLEAP_NN_VERSION}")
        return

    if args.collect:
        copy = ssh(
            f"mkdir -p {DESTINATION}/predictions {DESTINATION}/logs && "
            f"cp -n {WORK}/predictions/*.slp {DESTINATION}/predictions/ && "
            f"cp -n {WORK}/logs/* {DESTINATION}/logs/ 2>/dev/null; "
            f"ls {DESTINATION}/predictions | wc -l"
        )
        print(f"chunks on the bucket: {copy.stdout.strip()} (expected {len(chunks)})")
        # Byte-for-byte size check against /work, the same verification s16 ran.
        check = ssh(
            f"for f in {WORK}/predictions/*.slp; do "
            f"  b={DESTINATION}/predictions/$(basename $f); "
            f'  if [ ! -f "$b" ] || [ "$(stat -c%s $f)" != "$(stat -c%s $b)" ]; then '
            f'    echo "MISMATCH $(basename $f)"; fi; done; echo CHECKED'
        )
        print(check.stdout.strip())
        return

    print(f"experiments : {len(EXPERIMENTS)}  views: {len(VIEWS)}  "
          f"n_groups: {len(groups)} {groups}")
    print("architecture: top-down 3-node (round-8 centroid + s14 reviewed CI head)")
    print(f"chunks      : {len(chunks)} of <= {CHUNK} frames")
    print(f"frames      : {total:,}")
    print(f"tasks       : {TASKS}  ({min(per_task.values()):,}"
          f"-{max(per_task.values()):,} frames each)")
    print(f"estimate    : {max(per_task.values()) / args.fps / 3600:.1f} h per task "
          f"at {args.fps:.0f} fps  (MaxWall is 12 h)")
    print(f"destination : {DESTINATION}")

    manifest = {
        "chunks": chunks, "tasks": TASKS, "chunk_size": CHUNK,
        "experiments": list(EXPERIMENTS), "views": list(VIEWS),
        "n_frames_per_video": N_FRAMES, "batch_size": BATCH,
        "architecture": "topdown_3node_reviewed",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "run_inference.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    (out_dir / "c05_infer.sbatch").write_text(sbatch_script(), encoding="utf-8", newline="\n")
    print(f"\nwrote {out_dir}")

    if args.plan or not args.submit:
        print("\nnot submitting (pass --submit).")
        return

    with provenance("saion_inference_td", out_dir, name="c05") as record:
        record.params_from(args)
        record.param("partition", PARTITION)
        record.param("frames_total", total)
        record.param("chunks", len(chunks))
        record.param("models", MODELS_BY_VIEW)
        record.param("n_groups", len(groups))
        ssh(
            f"mkdir -p {WORK}/logs {WORK}/predictions && "
            f"cp {BUCKET}/work/c05_saion_inference_td/manifest.json {WORK}/ && "
            f"cp {BUCKET}/work/c05_saion_inference_td/run_inference.py {WORK}/ && "
            f"cp {BUCKET}/work/c05_saion_inference_td/c05_infer.sbatch {WORK}/"
        )
        result = ssh(f"cd {WORK} && sbatch {WORK}/c05_infer.sbatch")
        print(result.stdout.strip())
        record.param("submission", result.stdout.strip())


if __name__ == "__main__":
    main()
