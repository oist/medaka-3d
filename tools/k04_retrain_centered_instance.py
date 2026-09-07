"""K04 — retrain the top centered-instance head with the capacity spent differently.

Implements `labs/20260803_centered_instance_capacity`, whose decision rule was
fixed before this ran. Three changes against round 7 and nothing else:
``max_stride`` 64 -> 32, ``filters`` 24 -> 32, intensity augmentation off -> on.

Trained on the **round-7** package so the 113 held-out evaluation frames stay
held out. Scored by `tools/k04b_score_retrain.py`, never by ``val/loss``.

Run:
    .venv\\Scripts\\python.exe tools/k04_retrain_centered_instance.py [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from medaka3d import load_paths

ROUND7_TOP = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260704\top")
RUN_NAME = "k04.centered_instance.ms32_f32_intensity"

#: Round-7 values, held fixed. Listed rather than defaulted so a diff against
#: `training_config.yaml` is a diff of intent, not of library defaults.
ROUND7 = {
    "crop_size": 512,
    "min_crop_size": 100,
    "filters_rate": 2.0,
    "convs_per_block": 2,
    "output_stride": 2,
    "kernel_size": 3,
    "sigma": 2.5,
    "anchor_part": "mouth",
    "batch_size": 16,
    "learning_rate": 1e-4,
    "validation_fraction": 0.1,
    "min_train_steps_per_epoch": 200,
}

#: The three changes.
CHANGES = {"max_stride": 32, "filters": 32, "intensity_aug_p": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=ROUND7["batch_size"])
    parser.add_argument("--cache", default="C:/sleap_cache/k04_top_centered_instance")
    parser.add_argument("--ckpt-dir", default="C:/sleap_cache/k04_runs",
                        help="local scratch for checkpoints; the run is copied to work_dir")
    parser.add_argument("--dry-run", action="store_true", help="build the config, do not train")
    args = parser.parse_args()

    paths = load_paths()
    labels = ROUND7_TOP / "labels.pkg.slp"
    if not labels.is_file():
        raise SystemExit(f"round-7 training package not found: {labels}")

    # Checkpoints go to local disk during training and are copied to work_dir
    # afterwards. Measured on the first attempt: 754 MB (best + last, 31.4 M
    # params) written to the SMB share every epoch that validation improved,
    # which doubled wall-clock time — 165 s/epoch against 82 s of actual compute.
    # The final artefact still lands on /bucket; only the churn stays local.
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = paths.work_dir / "k04_retrain"
    final_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train_labels_path": [str(labels)],
        "validation_fraction": ROUND7["validation_fraction"],
        "user_instances_only": True,
        "data_pipeline_fw": "torch_dataset_cache_img_disk",
        "cache_img_path": args.cache,
        "delete_cache_imgs_after_training": False,
        "ensure_rgb": True,
        "max_height": 1200,
        "max_width": 1200,
        "scale": 1.0,
        "crop_size": ROUND7["crop_size"],
        "min_crop_size": ROUND7["min_crop_size"],
        "use_augmentations_train": True,
        # The one augmentation change. Round 7 configured these and set every
        # probability to 0.0, so the ranges are the experimenter's, not mine.
        "intensity_aug": {
            "uniform_noise_min": 0.0, "uniform_noise_max": 0.04,
            "uniform_noise_p": CHANGES["intensity_aug_p"],
            "gaussian_noise_mean": 0.0, "gaussian_noise_std": 0.02,
            "gaussian_noise_p": CHANGES["intensity_aug_p"],
            "contrast_min": 0.9, "contrast_max": 1.1,
            "contrast_p": CHANGES["intensity_aug_p"],
            "brightness_min": 0.9, "brightness_max": 1.1,
            "brightness_p": CHANGES["intensity_aug_p"],
        },
        "geometry_aug": {
            "rotation_min": -180.0, "rotation_max": 180.0, "rotation_p": 1.0,
            "scale_min": 0.7, "scale_max": 1.3, "scale_p": 1.0,
        },
        "backbone_config": {
            "unet": {
                "in_channels": 3,
                "kernel_size": ROUND7["kernel_size"],
                "filters": CHANGES["filters"],
                "filters_rate": ROUND7["filters_rate"],
                "max_stride": CHANGES["max_stride"],
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": ROUND7["convs_per_block"],
                "output_stride": ROUND7["output_stride"],
            }
        },
        "head_configs": {
            "centered_instance": {
                "confmaps": {
                    "part_names": ["mouth", "tail"],
                    "anchor_part": ROUND7["anchor_part"],
                    "sigma": ROUND7["sigma"],
                    "output_stride": ROUND7["output_stride"],
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
        "min_train_steps_per_epoch": ROUND7["min_train_steps_per_epoch"],
        "visualize_preds_during_training": False,
        "max_epochs": args.epochs,
        # Round 7 left the seed unset, so its run is not reproducible. Fixing one
        # here costs nothing and makes this one repeatable.
        "seed": 0,
        "save_ckpt": True,
        "ckpt_dir": str(ckpt_dir),
        "run_name": RUN_NAME,
        "optimizer": "Adam",
        "learning_rate": ROUND7["learning_rate"],
        "lr_scheduler": {
            "reduce_lr_on_plateau": {
                "threshold": 1e-6, "threshold_mode": "abs", "cooldown": 3,
                "patience": 5, "factor": 0.5, "min_lr": 1e-8,
            }
        },
        "early_stopping": False,
        "online_mining": True,
        "hard_to_easy_ratio": 2.0,
        "min_hard_keypoints": 2,
        "loss_scale": 5.0,
    }

    # The rich progress bar writes braille spinners; a cp1252 stdout cannot
    # encode them and the run dies in the logger rather than in training.
    print(__doc__.split("Run:")[0])
    print(f"labels    : {labels}")
    print(f"ckpt_dir  : {ckpt_dir / RUN_NAME}")
    print(f"changes   : {CHANGES}")
    print(f"epochs    : {args.epochs}  batch {args.batch_size}")
    record = final_dir / f"{RUN_NAME}.requested_config.json"
    record.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote     : {record}")
    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    from sleap_nn.train import train

    train(**config)

    source = ckpt_dir / RUN_NAME
    destination = final_dir / RUN_NAME
    print(f"\ncopying {source} -> {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"done. Local copy retained at {source}.")


if __name__ == "__main__":
    main()
