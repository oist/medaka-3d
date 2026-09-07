"""S14 — train the four 3-node arms on saion, in parallel, on the reviewed labels.

Four models, one per (architecture, view):

    topdown centered-instance x {top, side}
    bottomup                  x {top, side}

all on ``--pectoral-mode reviewed-only`` — the 543 moments a human reviewed on
2026-08-10, where every pectoral is an observation rather than a fill.

=============================================================================
WHY THIS TOOL EXISTS RATHER THAN RUNNING s03/s07 ON THE CLUSTER
=============================================================================

`s03` and `s07` resolve their inputs through `medaka3d.paths`, which is a
Windows drive-letter config and existence-checks every root at import. On a
Linux compute node that fails before it does anything useful. So the *decisions*
stay here — the hyperparameters are imported from `s03`/`s07` so there is one
definition of each, not a second copy that drifts — and only the resolved
config, in POSIX paths, is shipped to the cluster.

=============================================================================
CLUSTER FACTS, VERIFIED ON THIS ACCOUNT RATHER THAN INHERITED
=============================================================================

**The partition is `largegpu`.** `docs/OIST_HPC.md` says saion renamed
`largegpu` to `gpu-a100` and to prefer the new name. That is right in general
and **wrong for this account**: `sinfo` lists both, but the association for
`makoto-hiroi` carries `largegpu` and not `gpu-a100`, and a `--test-only`
submission to each settles it —

    -p largegpu  -> "Job ... to start at ... on nodes saion-gpu23"
    -p gpu-a100  -> "Invalid account or account/partition combination specified"

So `PARTITION` defaults to `largegpu` here. Override with the environment
variable if the association changes.

**`largegpu` allows 12 h per job and 8 GPUs at once** for this association
(`MaxWall 12:00:00`, `GrpTRES cpu=128,gres/gpu=8,mem=1T`). Four single-GPU
jobs fit inside that. Asking for 24 h does not fail at submit — it pends
forever on `AssocMaxWallDurationPerJobLimit`, which looks like a queue and
is actually a dead job.

**`/bucket` is read-only from compute nodes.** Inputs are read from it
directly; every checkpoint and log goes to `/work/StephensU`, which is saion's
scratch — not `/flash`, which is deigo's. Results are copied back to `/bucket`
from a login node afterwards.

**The environment is `module load sleap-nn/0.3.1`**, verified to carry
sleap-nn 0.3.1 / sleap-io 0.9.2 / torch 2.13.0+cu130 — the same pins as this
repo, so a checkpoint trained there loads here.

Run::

    .venv\\Scripts\\python.exe tools/s14_saion_train_3node.py --dry-run
    .venv\\Scripts\\python.exe tools/s14_saion_train_3node.py --submit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s03_train_bottomup_3node import ARMS
from s03_train_bottomup_3node import ROUND8 as BU
from s07_train_topdown_3node import ROUND8_CI as CI

from medaka3d import load_paths
from medaka3d.provenance import provenance

#: Verified for this account; see the module docstring. Not `gpu-a100`.
PARTITION = "largegpu"
BUCKET = "/bucket/StephensU/medaka"
WORK = "/work/StephensU/medaka3d/s14_3node"
MODULE = "sleap-nn/0.3.1"
NODE_ORDER = ["mouth", "pectoral", "tail"]
EDGES = [["mouth", "pectoral"], ["pectoral", "tail"]]
#: round-8's arm, matching the centered-instance capacity the 2x2 compared.
BU_ARM = "round8"
#: The 3-node bottom-up models s10 measured used these; kept so the only thing
#: that changed against them is the training data.
PAF_SIGMA, PAF_OUTPUT_STRIDE = 8.0, 2

DATASET = "{bucket}/sleap_models/retrain_20260804_3node/{view}/reviewed-only"


def dataset_paths(view: str) -> tuple[str, str]:
    root = DATASET.format(bucket=BUCKET, view=view)
    return (
        f"{root}/train_labels.3node_reviewed.pkg.slp",
        f"{root}/val_labels.3node_reviewed.pkg.slp",
    )


def common(view: str, run_name: str, batch_size: int, epochs: int) -> dict:
    train, val = dataset_paths(view)
    return {
        "train_labels_path": [train],
        "val_labels_path": [val],
        "validation_fraction": 0.0,
        "user_instances_only": True,
        "data_pipeline_fw": "torch_dataset_cache_img_disk",
        "cache_img_path": f"{WORK}/cache/{run_name}",
        "delete_cache_imgs_after_training": False,
        "ensure_rgb": True,
        "max_height": 1200,
        "max_width": 1200,
        "scale": 1.0,
        "use_augmentations_train": True,
        # Round 8 ships every intensity probability at 0.0. Held there so the
        # only difference between these arms and the ones s10 measured is the
        # labels they were trained on.
        "intensity_aug": {
            "uniform_noise_min": 0.0, "uniform_noise_max": 0.04, "uniform_noise_p": 0.0,
            "gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.02, "gaussian_noise_p": 0.0,
            "contrast_min": 0.9, "contrast_max": 1.1, "contrast_p": 0.0,
            "brightness_min": 0.9, "brightness_max": 1.1, "brightness_p": 0.0,
        },
        "geometry_aug": {
            "rotation_min": -180.0, "rotation_max": 180.0, "rotation_p": 1.0,
            "scale_min": 0.7, "scale_max": 1.3, "scale_p": 1.0,
        },
        "batch_size": batch_size,
        "shuffle_train": True,
        "num_workers": 8,
        "ckpt_save_top_k": 1,
        "ckpt_save_last": True,
        "trainer_accelerator": "auto",
        "enable_progress_bar": False,
        "min_train_steps_per_epoch": BU["min_train_steps_per_epoch"],
        "visualize_preds_during_training": False,
        "keep_viz": False,
        "max_epochs": epochs,
        "seed": 0,
        "save_ckpt": True,
        "ckpt_dir": f"{WORK}/runs/{view}",
        "run_name": run_name,
        "optimizer": "Adam",
        "learning_rate": BU["learning_rate"],
        "lr_scheduler": {
            "reduce_lr_on_plateau": {
                "threshold": 1e-6, "threshold_mode": "abs", "cooldown": 3,
                "patience": 5, "factor": 0.5, "min_lr": 1e-8,
            }
        },
        "early_stopping": True,
        "early_stopping_min_delta": 1e-8,
        "early_stopping_patience": 40,
    }


def topdown_config(view: str, epochs: int) -> tuple[str, dict]:
    run = f"s14.topdown_3node.reviewed_{view}"
    cfg = common(view, run, CI["batch_size"], epochs)
    cfg.update({
        "crop_size": CI["crop_size"],
        "min_crop_size": CI["min_crop_size"],
        "backbone_config": {"unet": {
            "in_channels": 3, "kernel_size": CI["kernel_size"],
            "filters": CI["filters"], "filters_rate": CI["filters_rate"],
            "max_stride": CI["max_stride"], "stem_stride": None,
            "middle_block": True, "up_interpolate": True, "stacks": 1,
            "convs_per_block": CI["convs_per_block"],
            "output_stride": CI["output_stride"],
        }},
        "head_configs": {"centered_instance": {"confmaps": {
            "part_names": NODE_ORDER, "anchor_part": CI["anchor_part"],
            "sigma": CI["sigma"], "output_stride": CI["output_stride"],
            "loss_weight": 1.0,
        }}},
        # Present for the centered-instance arm only, exactly as s07 sets them.
        "online_mining": True, "hard_to_easy_ratio": 2.0,
        "min_hard_keypoints": 2, "loss_scale": 5.0,
    })
    return run, cfg


def bottomup_config(view: str, epochs: int) -> tuple[str, dict]:
    spec = ARMS[BU_ARM]
    run = (f"s14.bottomup_3node.reviewed_{view}"
           f"_f{spec['filters']}_ms{spec['max_stride']}_paf{int(PAF_SIGMA)}")
    cfg = common(view, run, spec["batch_size"], epochs)
    cfg.update({
        "min_crop_size": BU["min_crop_size"],
        "backbone_config": {"unet": {
            "in_channels": 3, "kernel_size": BU["kernel_size"],
            "filters": spec["filters"], "filters_rate": BU["filters_rate"],
            "max_stride": spec["max_stride"], "stem_stride": None,
            "middle_block": True, "up_interpolate": True, "stacks": 1,
            "convs_per_block": BU["convs_per_block"],
            "output_stride": BU["output_stride"],
        }},
        "head_configs": {"bottomup": {
            "confmaps": {"part_names": NODE_ORDER, "sigma": BU["sigma"],
                         "output_stride": BU["output_stride"], "loss_weight": 1.0},
            "pafs": {"edges": EDGES, "sigma": PAF_SIGMA,
                     "output_stride": PAF_OUTPUT_STRIDE, "loss_weight": 1.0},
        }},
    })
    return run, cfg


RUNNER = '''"""Train one 3-node arm from a JSON config. Shipped to saion by s14."""
import json
import sys

config = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(f"run_name : {config['run_name']}")
print(f"train    : {config['train_labels_path']}")
print(f"ckpt_dir : {config['ckpt_dir']}")
import torch
print(f"cuda     : {torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
from sleap_nn.train import train as run_training
run_training(**config)
print("TRAINING COMPLETE")
'''


def sbatch_script(jobs: list[tuple[str, str]]) -> str:
    """One array-free script per job, submitted separately, so a failure in one
    arm does not cancel the other three."""
    lines = [
        "#!/bin/bash -l",
        "#SBATCH --job-name=s14_3node",
        # SBATCH directives are read by Slurm, not the shell: \ is a
        # literal here and fails with "invalid partition specified". Override
        # per submission with  sbatch -p <name>  instead.
        f"#SBATCH --partition={PARTITION}",
        "#SBATCH --gres=gpu:a100:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=120G",
        # 12 h is the association MaxWall for largegpu on this account. 24 h
        # is accepted at submit and then pends forever as
        # AssocMaxWallDurationPerJobLimit. Read from sacctmgr, not guessed.
        "#SBATCH --time=12:00:00",
        f"#SBATCH --output={WORK}/logs/%x-%j.out",
        f"#SBATCH --error={WORK}/logs/%x-%j.err",
        "set -euo pipefail",
        "",
        "# saion needs a login shell for `module` to exist at all; the -l above",
        "# is load-bearing, not decoration.",
        f"module load {MODULE}",
        "",
        "CONFIG=\"$1\"",
        "# The module puts a uv-managed tool on PATH; its interpreter is the one",
        "# carrying sleap_nn, so call it directly rather than a bare `python`,",
        "# which the login environment does not define.",
        "PYBIN=/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python",
        'echo "node   : $(hostname)"',
        'echo "config : $CONFIG"',
        'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true',
        f'"$PYBIN" {WORK}/run_training.py "$CONFIG"',
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--submit", action="store_true",
                        help="Stage to saion and sbatch. Without it, nothing leaves "
                             "this machine.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "s14_saion_train_3node"
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        ("topdown", "top"): topdown_config,
        ("topdown", "side"): topdown_config,
        ("bottomup", "top"): bottomup_config,
        ("bottomup", "side"): bottomup_config,
    }
    configs: dict[str, dict] = {}
    for (arch, view), build in builders.items():
        run, cfg = build(view, args.epochs)
        configs[run] = cfg
        print(f"{arch:<9} {view:<5} -> {run}")
        print(f"            batch {cfg['batch_size']}  "
              f"train {cfg['train_labels_path'][0].rsplit('/', 1)[-1]}")

    with provenance("saion_train_3node", out_dir, name="s14") as record:
        record.params_from(args)
        record.param("partition", PARTITION)
        record.param("module", MODULE)
        record.param("runs", sorted(configs))
        for run, cfg in configs.items():
            path = out_dir / f"{run}.config.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            record.output(run, path)
        (out_dir / "run_training.py").write_text(RUNNER, encoding="utf-8")
        script = out_dir / "s14_train.sbatch"
        script.write_text(sbatch_script(list(builders)), encoding="utf-8", newline="\n")
        record.output("sbatch", script)
        print(f"\nwrote {out_dir}")

        if args.dry_run or not args.submit:
            print("\nnot submitting (pass --submit).")
            return

        staged = f"{WORK}"
        remote_cfgs = " ".join(f"{staged}/configs/{run}.config.json" for run in configs)
        setup = (
            f"mkdir -p {staged}/configs {staged}/logs {staged}/runs {staged}/cache && "
            f"cp {BUCKET}/work/s14_saion_train_3node/*.config.json {staged}/configs/ && "
            f"cp {BUCKET}/work/s14_saion_train_3node/run_training.py {staged}/ && "
            f"cp {BUCKET}/work/s14_saion_train_3node/s14_train.sbatch {staged}/"
        )
        print(f"\nstaging:\n  {setup}")
        subprocess.run(["ssh", "saion", f"bash -lc '{setup}'"], check=True)
        for run in configs:
            cmd = (f"cd {staged} && sbatch --job-name=s14_{run.split('.')[1][:12]}_"
                   f"{'top' if '_top' in run else 'side'} "
                   f"{staged}/s14_train.sbatch {staged}/configs/{run}.config.json")
            result = subprocess.run(
                ["ssh", "saion", f"bash -lc '{cmd}'"],
                check=True, capture_output=True, text=True,
            )
            print(f"  {run}: {result.stdout.strip()}")
        print(f"\nsubmitted {len(configs)} jobs to {PARTITION}")
        print(f"remote config list: {remote_cfgs}")


if __name__ == "__main__":
    main()
