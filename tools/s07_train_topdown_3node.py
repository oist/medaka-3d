"""S07 — train a top-down centered-instance head, 3-node or 2-node control.

Bottom-up has been settled: 0 shared nodes on the locked benchmark, but so was
round 7's 2-node bottom-up, so the floor was already reached and the skeleton
change could not be seen there. Top-down is where the reported failure lives —
the centered-instance head linking parts from different individuals — so it is
the arm where a third node has something to fix.

=============================================================================
SPECIFICATION — fixed before the run
=============================================================================

**Only the centered-instance head is trained.** The centroid head predicts one
anchor point (`anchor_part='mouth'`) and knows nothing about how many nodes the
skeleton has, so round 8's centroid is reused unchanged. That halves the
compute and, more importantly, removes the centroid from the comparison
entirely: any difference measured downstream belongs to the CI head.

**Every hyperparameter is round 8's.** crop 512, UNet filters 24 /
filters_rate 2.0 / max_stride 64 / convs_per_block 2 / output_stride 2, confmap
sigma 2.5, anchor_part mouth, batch 16, Adam 1e-4, 200 epochs, online hard
keypoint mining on. Copied from
`round8_260723_strictmouth.centered_instance_n1382/training_config.yaml`, not
from memory.

**The 2-node control is not optional.** Round 8's CI trained on 1,382 frames;
these train on 409 (top) / 248 (side). Comparing a 3-node model on 409 frames
against round 8 confounds the skeleton with a 3.4x difference in data, and this
project has already published one conclusion that had to be withdrawn for
exactly that reason. `--nodes 2` trains on the *same frames, same mouth/tail,
pectoral removed*, so the pair differs in the skeleton and nothing else. Round 8
stays in the comparison as the production reference, not as the control.

**What the comparison can and cannot show.** The locked benchmark has no
pectoral, so the 3-node model is scored on mouth and tail only. That is the
right scope: part-mixing is about whether mouth and tail belong to one animal,
and the pectoral's job here is to change how the crop is *interpreted*, not to
be scored itself.

**n_groups = 1.**

Run::

    .venv\\Scripts\\python.exe tools/s07_train_topdown_3node.py --view top --nodes 3
    .venv\\Scripts\\python.exe tools/s07_train_topdown_3node.py --view top --nodes 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sleap_io as sio

from medaka3d import load_paths

DATA_ROOT = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260804_3node")
ROUND8 = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260722_round8")

# Read from round 8's own training_config.yaml, not from memory.
ROUND8_CI = {
    "crop_size": 512, "min_crop_size": 100, "kernel_size": 3, "filters": 24,
    "filters_rate": 2.0, "max_stride": 64, "convs_per_block": 2, "output_stride": 2,
    "sigma": 2.5, "anchor_part": "mouth", "batch_size": 16, "max_epochs": 200,
    "learning_rate": 1e-4, "min_train_steps_per_epoch": 200,
}
CENTROIDS = {
    "top": ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centroid_n1382",
    "side": ROUND8 / "side" / "runs" / "round8_260723_strictmouth.centroid_n2764",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=("top", "side"), default="top")
    parser.add_argument("--nodes", type=int, choices=(2, 3), default=3)
    parser.add_argument("--pectoral-mode", choices=("annotated-only", "interpolate"),
                        default="annotated-only")
    parser.add_argument("--epochs", type=int, default=ROUND8_CI["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=ROUND8_CI["batch_size"])
    parser.add_argument("--cache", default="C:/sleap_cache/s07_td")
    parser.add_argument("--ckpt-dir", default="C:/sleap_cache/s07_runs")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = DATA_ROOT / args.view / args.pectoral_mode
    suffix = "annotated" if args.pectoral_mode == "annotated-only" else "interp"
    if args.nodes == 2:
        suffix += "_2node"
    train = data_dir / f"train_labels.3node_{suffix}.pkg.slp"
    val = data_dir / f"val_labels.3node_{suffix}.pkg.slp"
    for path in (train, val):
        if not path.is_file():
            raise SystemExit(
                f"missing {path}\nBuild it first:\n"
                f"  .venv\\Scripts\\python.exe tools/s02_merge_pectoral.py "
                f"--view {args.view} --pectoral-mode {args.pectoral_mode} "
                f"--val-moments 30{' --drop-pectoral' if args.nodes == 2 else ''}"
            )

    labels = sio.load_slp(str(train))
    node_names = [n.name for n in labels.skeleton.nodes]
    if len(node_names) != args.nodes:
        raise SystemExit(
            f"--nodes {args.nodes} but {train.name} carries {node_names}. "
            f"The dataset and the flag disagree; refusing to guess which is right."
        )
    n_train = len(labels.labeled_frames)

    centroid = CENTROIDS[args.view]
    if not (centroid / "best.ckpt").is_file():
        raise SystemExit(f"round-8 centroid not found: {centroid}")

    paths = load_paths()
    run_name = (f"s07.topdown_{args.nodes}node.round8params"
                f"_{args.pectoral_mode}_n{n_train}")
    ckpt_dir = Path(args.ckpt_dir) / args.view
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = paths.work_dir / "s07_topdown_3node" / args.view
    final_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train_labels_path": [str(train)],
        "val_labels_path": [str(val)],
        "validation_fraction": 0.0,
        "user_instances_only": True,
        "data_pipeline_fw": "torch_dataset_cache_img_disk",
        "cache_img_path": f"{args.cache}_{args.view}_{args.nodes}node",
        "delete_cache_imgs_after_training": False,
        "ensure_rgb": True,
        "max_height": 1200,
        "max_width": 1200,
        "scale": 1.0,
        "crop_size": ROUND8_CI["crop_size"],
        "min_crop_size": ROUND8_CI["min_crop_size"],
        "use_augmentations_train": True,
        # Round 8 ships every intensity probability at 0.0. Held there so the
        # only difference between the arms is the node count.
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
        "backbone_config": {
            "unet": {
                "in_channels": 3,
                "kernel_size": ROUND8_CI["kernel_size"],
                "filters": ROUND8_CI["filters"],
                "filters_rate": ROUND8_CI["filters_rate"],
                "max_stride": ROUND8_CI["max_stride"],
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": ROUND8_CI["convs_per_block"],
                "output_stride": ROUND8_CI["output_stride"],
            }
        },
        "head_configs": {
            "centered_instance": {
                "confmaps": {
                    "part_names": node_names,
                    "anchor_part": ROUND8_CI["anchor_part"],
                    "sigma": ROUND8_CI["sigma"],
                    "output_stride": ROUND8_CI["output_stride"],
                    "loss_weight": 1.0,
                }
            }
        },
        "batch_size": args.batch_size,
        "shuffle_train": True,
        "num_workers": 8,
        "ckpt_save_top_k": 1,
        "ckpt_save_last": True,
        "trainer_accelerator": "auto",
        "enable_progress_bar": False,
        "min_train_steps_per_epoch": ROUND8_CI["min_train_steps_per_epoch"],
        "visualize_preds_during_training": not args.no_viz,
        "keep_viz": not args.no_viz,
        "max_epochs": args.epochs,
        "seed": 0,
        "save_ckpt": True,
        "ckpt_dir": str(ckpt_dir),
        "run_name": run_name,
        "optimizer": "Adam",
        "learning_rate": ROUND8_CI["learning_rate"],
        "lr_scheduler": {
            "reduce_lr_on_plateau": {
                "threshold": 1e-6, "threshold_mode": "abs", "cooldown": 3,
                "patience": 5, "factor": 0.5, "min_lr": 1e-8,
            }
        },
        "early_stopping": True,
        "early_stopping_min_delta": 1e-8,
        "early_stopping_patience": 40,
        "online_mining": True,
        "hard_to_easy_ratio": 2.0,
        "min_hard_keypoints": 2,
        "loss_scale": 5.0,
    }

    print(__doc__.split("=====")[0])
    print(f"view      : {args.view}   nodes: {node_names}")
    print(f"centroid  : {centroid.name}  (round 8, reused unchanged)")
    print(f"train     : {train.name}  ({n_train} frames)")
    print(f"val       : {val.name}")
    print(f"epochs    : {args.epochs}  batch {args.batch_size}  crop "
          f"{ROUND8_CI['crop_size']}")
    print(f"ckpt      : {ckpt_dir / run_name}")

    record = final_dir / f"{run_name}.requested_config.json"
    record.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote     : {record}")
    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    from sleap_nn.train import train as run_training

    run_training(**config)

    source, destination = ckpt_dir / run_name, final_dir / run_name
    print(f"\ncopying {source} -> {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"done. Local copy retained at {source}.")


if __name__ == "__main__":
    main()
