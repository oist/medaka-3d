"""K11 — retrain the centered-instance head with capacity moved to high resolution.

Implements `labs/20260804_ci_high_resolution_capacity`, whose decision rule was
fixed before this ran. **Round 8 is the baseline.**

Three changes against round 8's centered-instance head, and nothing else:

    crop_size     512 -> 384      (192 px half-width; 163.2 px is the longest
                                   labelled span, so 0% of tails are clipped at
                                   scale 1.0 and 2.84% at the scale-1.3 setting)
    max_stride     64 -> 32       (smaller receptive field; the deepest block
                                   was running at 8x8 for a 1.8-cell fish)
    filters        24 -> 64       (2x-2.7x the channels at every stride where a
                                   keypoint is localised)

`filters_rate` stays at **2.0**. Raising it instead reaches more parameters but
puts them in the 12x12 block where the fish spans 3.5 cells; see the lab README.

Data, augmentation, optimiser and schedule are round 8's, unchanged — same
train/val packages, `validation_fraction=0.0`, intensity augmentation at p=0.0.
The **centroid head is not retrained**; inference pairs this with round 8's.

Scored by `tools/k11b_score.py` on the locked benchmark, never by `val/loss`.

Run:
    .venv\\Scripts\\python.exe tools/k11_retrain_ci_highres.py [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from medaka3d import load_catalog, load_paths

ROUND8_TOP = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260722_round8\top")

#: Same recipe, two views. Round 8's side centered-instance head is
#: architecturally identical to its top one — crop 512, filters 24, rate 2.0,
#: max_stride 64, 70,755,866 params — so the three changes are the same and only
#: the label packages differ. Side holds both cameras in one package and round 8
#: trained them together; they must be *scored* separately.
#: Naming follows the project's model-generation convention, not this file's.
#: `k*` prefixes are the **Group-K analysis** series (k01 misdetection taxonomy,
#: k02 renders, ...) and belong on tools. **Models are `round<N>`** — round7,
#: round8, and the round9/round9b side *label* sets of 2026-07-28, which never
#: got a trained model. Round 10 is the next free generation. Run names mirror
#: round 8's shape: `round<N>_<date>_<tag>.<head>_n<frames>`.
VIEWS = {
    "top": {
        "run_name": "round10_260804_hires.centered_instance_n1382",
        "train": "labels.round8.top.train",
        "val": "labels.round8.top.val",
        "n_train": 1382,
    },
    "side": {
        "run_name": "round10_260804_hires.centered_instance_n2764",
        "train": "labels.round8.side.train",
        "val": "labels.round8.side.val",
        "n_train": 2764,
    },
}
#: The top run was trained and scored under its old name; both are accepted when
#: locating the checkpoint so existing write-ups stay findable.
LEGACY_RUN_NAMES = {"top": "k11.centered_instance.crop384_ms32_f64_r2"}
RUN_NAME = VIEWS["top"]["run_name"]

#: Round-8 values, held fixed. Listed rather than defaulted so a diff against
#: round 8's `training_config.yaml` is a diff of intent, not of library defaults.
ROUND8 = {
    "min_crop_size": 100,
    "filters_rate": 2.0,
    "convs_per_block": 2,
    "output_stride": 2,
    "kernel_size": 3,
    "sigma": 2.5,
    "anchor_part": "mouth",
    "batch_size": 16,
    "learning_rate": 1e-4,
    "max_epochs": 200,
    "min_train_steps_per_epoch": 200,
}

#: The three changes.
CHANGES = {"crop_size": 384, "max_stride": 32, "filters": 64}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=ROUND8["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=ROUND8["batch_size"])
    parser.add_argument("--cache", default="C:/sleap_cache/k11_top_ci")
    parser.add_argument("--ckpt-dir", default="C:/sleap_cache/k11_runs",
                        help="local scratch; the finished run is copied to work_dir")
    parser.add_argument("--view", choices=["top", "side"], default="top")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = VIEWS[args.view]
    run_name = spec["run_name"]
    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    train = Path(str(catalog.get(spec["train"])))
    val = Path(str(catalog.get(spec["val"])))
    for path in (train, val):
        if not path.is_file():
            raise SystemExit(f"missing label package: {path}")

    # Checkpoints stay on local disk during training: writing ~500 MB per epoch
    # to the SMB share doubled wall-clock time on an earlier run. The finished
    # run is copied to /bucket so provenance still lands there.
    ckpt_dir = Path(args.ckpt_dir) / args.view
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = paths.work_dir / "k11_retrain" / args.view
    final_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train_labels_path": [str(train)],
        "val_labels_path": [str(val)],
        "validation_fraction": 0.0,
        "user_instances_only": True,
        "data_pipeline_fw": "torch_dataset_cache_img_disk",
        "cache_img_path": f"{args.cache}_{args.view}",
        "delete_cache_imgs_after_training": False,
        "ensure_rgb": True,
        "max_height": 1200,
        "max_width": 1200,
        "scale": 1.0,
        "crop_size": CHANGES["crop_size"],
        "min_crop_size": ROUND8["min_crop_size"],
        "use_augmentations_train": True,
        # Round 8 ships every intensity probability at 0.0. Held there so the
        # only differences are the three architectural ones.
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
                "kernel_size": ROUND8["kernel_size"],
                "filters": CHANGES["filters"],
                "filters_rate": ROUND8["filters_rate"],
                "max_stride": CHANGES["max_stride"],
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": ROUND8["convs_per_block"],
                "output_stride": ROUND8["output_stride"],
            }
        },
        "head_configs": {
            "centered_instance": {
                "confmaps": {
                    "part_names": ["mouth", "tail"],
                    "anchor_part": ROUND8["anchor_part"],
                    "sigma": ROUND8["sigma"],
                    "output_stride": ROUND8["output_stride"],
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
        "min_train_steps_per_epoch": ROUND8["min_train_steps_per_epoch"],
        "visualize_preds_during_training": False,
        "max_epochs": args.epochs,
        "seed": 0,
        "save_ckpt": True,
        "ckpt_dir": str(ckpt_dir),
        "run_name": run_name,
        "optimizer": "Adam",
        "learning_rate": ROUND8["learning_rate"],
        "lr_scheduler": {
            "reduce_lr_on_plateau": {
                "threshold": 1e-6, "threshold_mode": "abs", "cooldown": 3,
                "patience": 5, "factor": 0.5, "min_lr": 1e-8,
            }
        },
        # Stop on a validation plateau rather than always spending 200 epochs.
        # Patience 40 is calibrated on the top run's own curve: its best val loss
        # landed at epoch 79 and never improved again, and the largest gap
        # between successive new bests was 9 epochs — so 40 carries a 4.4x margin
        # over anything the real curve did, while cutting ~40% of the wall clock.
        # `min_delta` is round 8's 1e-8, i.e. any improvement resets the counter,
        # which makes this deliberately reluctant to fire.
        #
        # Caveat, recorded because it is a real one: this stops on `val/loss`,
        # which this project has repeatedly shown is a poor proxy for task
        # quality. Patience 40 should be generous enough that it does not bite,
        # but if task metrics ever peak well after the loss plateaus, this stops
        # blind. `ckpt_save_top_k=1` still keeps the best epoch, not the last.
        "early_stopping": True,
        "early_stopping_min_delta": 1e-8,
        "early_stopping_patience": 40,
        "online_mining": True,
        "hard_to_easy_ratio": 2.0,
        "min_hard_keypoints": 2,
        "loss_scale": 5.0,
    }

    print(__doc__.split("Run:")[0])
    print(f"view  : {args.view}")
    print(f"train : {train.name}  (round-8 split, {spec['n_train']} frames)")
    print(f"val   : {val.name}")
    print(f"change: {CHANGES}   filters_rate held at {ROUND8['filters_rate']}")
    print(f"ckpt  : {ckpt_dir / run_name}")
    print(f"epochs: {args.epochs}  batch {args.batch_size}")
    record = final_dir / f"{run_name}.requested_config.json"
    record.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote : {record}")
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
