"""S16 — full-length bottom-up 3-node inference for K_1_{1,2,3}, all three views.

3 experiments x 3 views x 348,600 frames = **3,137,400 inferences**, run on
saion `largegpu` as an 8-task array.

The models are the `s14` bottom-up arms trained on the 2026-08-10 reviewed
labels. `s15` measured them against their predecessors on 20,000 frames per
view and found the improvement is real and confined to this architecture:
mouth-linked tail_swap 0.0818 -> 0.0224% on top (p=9.3e-06), tail_jump down in
all three views, and 15-20% fewer frames missing an animal. The top-down arm
did not improve, which is why only bottom-up runs here.

=============================================================================
HOW THE WORK IS SPLIT, AND WHY LIKE THIS
=============================================================================

Nine (experiment, view) units are cut into fixed-size frame chunks, and chunk
*i* goes to array task ``i % 8``. Round-robin rather than contiguous blocks
because the three views are not equally fast and a contiguous split would leave
one task holding all the slow work while the others idle.

**Chunks are frame ranges, not files**, so a task that dies loses one chunk. Its
output name carries `(experiment, view, start, end)`, which is also the only
safe key in this project: `side1_cam` and `side2_cam` share every `frame_idx`,
and anything keyed on the index alone silently crosses them.

**8 tasks is the ceiling, not a choice.** The association allows
`gres/gpu=8` concurrently on `largegpu`, so a ninth task would queue rather
than add throughput.

=============================================================================
CLUSTER CONSTRAINTS, VERIFIED ON THIS ACCOUNT
=============================================================================

* Partition **`largegpu`** — `gpu-a100` is rejected for this association
  ("Invalid account or account/partition combination"), despite `sinfo`
  listing it. See `s14`.
* **MaxWall 12 h.** Over-asking does not fail at submit; the job pends forever
  on `AssocMaxWallDurationPerJobLimit`.
* **`/bucket` is read-only from compute nodes.** Videos are read from it;
  every prediction is written to `/work` and copied back from a login node.

Run::

    .venv\\Scripts\\python.exe tools/s16_saion_inference_bu.py --plan
    .venv\\Scripts\\python.exe tools/s16_saion_inference_bu.py --submit
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
WORK = "/work/StephensU/medaka3d/s16_bu_inference"
MODULE = "sleap-nn/0.3.1"
PYBIN = "/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python"
MODELS = f"{BUCKET}/sleap_models/retrain_20260810_3node_reviewed"
MODEL_BY_VIEW = {
    "top_cam": f"{MODELS}/top/s14.bottomup_3node.reviewed_top_f24_ms64_paf8",
    "side1_cam": f"{MODELS}/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8",
    "side2_cam": f"{MODELS}/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8",
}
VIEW_TO_MP4 = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4",
               "side2_cam": "cam2_side.mp4"}
#: 4, not 16. Measured on the 3-node bottom-up head at 1200x1200: batch 16
#: peaks at 14.29 GiB and thrashes to 2.8 fps where batch 4 peaks at 3.80 GiB
#: and runs 18.9 fps. Instance counts identical, median coordinate delta
#: 0.0002 px. See the s10 decision document.
BATCH = 4


def build_chunks() -> list[dict]:
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
                    "model": MODEL_BY_VIEW[view],
                    "out": f"{WORK}/predictions/{experiment}__{view}"
                           f"__frames_{start}-{end}.slp",
                })
    for index, chunk in enumerate(chunks):
        chunk["task"] = index % TASKS
    return chunks


RUNNER = '''"""Run one array task's chunks of bottom-up 3-node inference."""
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
        blank, model_paths=[chunk["model"]], device="cuda", batch_size=batch)
    # Write to a temporary name and rename, so an interrupted task never
    # leaves a half-written file that the resume check would treat as done.
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
        "#SBATCH --job-name=s16_bu_infer",
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
        f"module load {MODULE}",
        'echo "task $SLURM_ARRAY_TASK_ID on $(hostname)"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        # Manifest as $1 so a smoke run needs a different argument, not a
        # sed over this script: the first attempt rewrote the log paths too
        # ("s16_bu_inference" -> "s16_smokeence") and the job died in 1 s
        # with no output, because the output directory did not exist.
        f'MANIFEST="${{1:-{WORK}/manifest.json}}"',
        f'"{PYBIN}" {WORK}/run_inference.py "$MANIFEST" {BATCH}',
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--plan", action="store_true", help="Report the split only.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Assumed A100 throughput, for the time estimate only.")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "s16_saion_inference_bu"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks()
    total = sum(c["end"] - c["start"] for c in chunks)
    per_task = {t: sum(c["end"] - c["start"] for c in chunks if c["task"] == t)
                for t in range(TASKS)}
    groups = sorted({ExperimentId.parse(e).group for e in EXPERIMENTS})

    print(f"experiments : {len(EXPERIMENTS)}  views: {len(VIEWS)}  "
          f"n_groups: {len(groups)} {groups}")
    print(f"chunks      : {len(chunks)} of <= {CHUNK} frames")
    print(f"frames      : {total:,}")
    print(f"tasks       : {TASKS}  ({min(per_task.values()):,}"
          f"-{max(per_task.values()):,} frames each)")
    print(f"estimate    : {max(per_task.values()) / args.fps / 3600:.1f} h per task "
          f"at {args.fps:.0f} fps  (MaxWall is 12 h)")

    manifest = {
        "chunks": chunks, "tasks": TASKS, "chunk_size": CHUNK,
        "experiments": list(EXPERIMENTS), "views": list(VIEWS),
        "n_frames_per_video": N_FRAMES, "batch_size": BATCH,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "run_inference.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    (out_dir / "s16_infer.sbatch").write_text(sbatch_script(), encoding="utf-8", newline="\n")
    print(f"\nwrote {out_dir}")

    if args.plan or not args.submit:
        print("\nnot submitting (pass --submit).")
        return

    with provenance("saion_inference_bu", out_dir, name="s16") as record:
        record.params_from(args)
        record.param("partition", PARTITION)
        record.param("frames_total", total)
        record.param("chunks", len(chunks))
        record.param("models", MODEL_BY_VIEW)
        record.param("n_groups", len(groups))
        stage = (
            f"mkdir -p {WORK}/logs {WORK}/predictions && "
            f"cp {BUCKET}/work/s16_saion_inference_bu/manifest.json {WORK}/ && "
            f"cp {BUCKET}/work/s16_saion_inference_bu/run_inference.py {WORK}/ && "
            f"cp {BUCKET}/work/s16_saion_inference_bu/s16_infer.sbatch {WORK}/"
        )
        subprocess.run(["ssh", "saion", f"bash -lc '{stage}'"], check=True)
        result = subprocess.run(
            ["ssh", "saion", f"bash -lc 'cd {WORK} && sbatch {WORK}/s16_infer.sbatch'"],
            check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        record.param("submission", result.stdout.strip())


if __name__ == "__main__":
    main()
