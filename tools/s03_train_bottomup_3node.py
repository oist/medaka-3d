"""S03 — train a 3-node bottom-up model, at parity with the top-down arms.

Bottom-up is the architecture where a third node can change the *mechanism*
rather than just adding a keypoint: PAFs are how parts get assigned to animals,
and inserting a mid-body node shortens the longest grouping edge from 101 px to
72 px and lets the body chain bend. B0.3 already measured bottom-up emitting
**0 shared-node predictions against top-down's 5 (top) / 3 (side)** at ~92% vs
~73% precision; it lost only on the pre-registered crossing-recall margin.

=============================================================================
SPECIFICATION — fixed before the run
=============================================================================

**Capacity is at parity with the top-down models, deliberately.** Round 7's
bottom-up was `filters=16`, **7.85 M parameters** — against a centered-instance
of 70.8 M. Any comparison against it confounds architecture with capacity. The
backbone here is byte-for-byte the one the top-down arms use:

| arm | filters | max_stride | params | matches |
|---|---|---|---|---|
| `round10` (default) | 64 | 32 | **125,610,183** | round 10 / k11 CI, 125.6 M |
| `round8` | 24 | 64 | 70.8 M | round 8 CI, 70,755,866 |

Measured on this machine (RTX 5070 Ti, 16 GiB): `round10` needs batch 2 and
runs 533 ms/step, ~20.5 h for 200 epochs. `round8` runs batch 4 at 342 ms/step,
~6.6 h. `filters=48/max_stride=32` is *not* offered: it fits nominally at
14.9 GB but with Adam states it spills to shared memory and collapses to
7.3 s/step — 14x slower than a model with more parameters.

The UNet carries **no normalisation layers**, checked rather than assumed, so
batch 2 costs nothing statistically.

**Skeleton: `mouth -> pectoral -> tail`, a chain.** Confirmed with the
experimenter. A redundant `mouth -> tail` edge was considered and rejected:
`sleap_nn.inference.ops.paf.toposort_edges` reduces any skeleton to a **BFS
spanning tree from the topological root**, and `group_instances_sample` iterates
only those edges — so with `m->p, p->t, m->t` the returned order is `(0, 2)` and
**`p->t` is trained but never read at inference**. A chain has no such trap.

**PAF sigma is the least-certain number here.** Round 7 used 16.0 for a single
101 px edge. The edges are now 30 px and 72 px (a 30/70 split, not halves), and
sigma is the half-width of the field around the edge: at 16.0 the short edge
would be wider than it is long, making the field near-isotropic and pulling in
neighbouring animals at exactly the crossings this is meant to fix. Default 8.0,
roughly a body half-width. It is a **swept** parameter and appears in the run
name; `--paf-sigma 5.0` is the side arm.

**Read a side null result carefully — the ceiling there is low, for anatomical
reasons.** From the experimenter: medaka bend **laterally**, and barely at all
dorsoventrally, because of how the body is built. The top camera therefore sees
the curvature, while from the side the mouth and tail project nearly collinear
and the pectoral sits close to that line. Interpolation is consequently *near
optimal* on side and there is little headroom for any model to win.

The data already carries that signature. Perpendicular offset of the pectoral
from the mouth->tail chord, as a fraction of span (p10..p90):

    top_cam    -0.046 .. +0.076     range 0.122
    side1_cam  -0.047 .. +0.048     range 0.094
    side2_cam  -0.057 .. +0.043     range 0.100

Top is ~25% wider and visibly skewed positive; the side views are narrower and
symmetric about zero. So a side arm that fails to beat interpolation on *median
distance* is the predicted outcome, not evidence that the sweep failed. Judge
the side arms on **grouping** instead — catastrophic errors (>50 px), which at
sigma 8.0 were 2 on side and 0 on top, and which a distance percentile hides.

**PAF output stride 2, not round 7's 4.** At stride 4 a 30 px edge spans 7.5
cells. The short edge is the new mechanism; halving the stride is cheap.

**Augmentation matches round 8 and round 10 exactly**, including intensity
probabilities held at 0.0. Enabling them is a separate, cheap experiment; doing
it here would add a variable to a comparison that exists to isolate one.

**What this run cannot tell you.** 90.9% of pectoral positions in the training
split are synthetic (`s02 --pectoral-mode interpolate`), because sleap-nn has no
per-node loss mask and an unlabelled node becomes an explicit "nothing here"
target. So *pectoral localisation accuracy* from this model is partly a
measurement of the interpolation rule. The decision metric — shared-node /
part-mixing rate — is unaffected: it is evaluated on predictions against human
ground truth on the locked HITL-v2 benchmark, which no synthetic label touches.

**n_groups = 1.** Group K only.

Run::

    .venv\\Scripts\\python.exe tools/s03_train_bottomup_3node.py --dry-run
    .venv\\Scripts\\python.exe tools/s03_train_bottomup_3node.py --epochs 2   # smoke
    .venv\\Scripts\\python.exe tools/s03_train_bottomup_3node.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sleap_io as sio

from medaka3d import load_paths

DATA_ROOT = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260804_3node")

# Held fixed from round 8 so only the architecture differs.
ROUND8 = {
    "kernel_size": 3, "filters_rate": 2.0, "convs_per_block": 2, "output_stride": 2,
    "sigma": 2.5, "learning_rate": 1e-4, "max_epochs": 200,
    "min_train_steps_per_epoch": 200, "min_crop_size": 100,
}

ARMS = {
    "round10": {"filters": 64, "max_stride": 32, "batch_size": 2,
                "matches": "round 10 / k11 centered-instance (125.6 M)"},
    "round8": {"filters": 24, "max_stride": 64, "batch_size": 4,
               "matches": "round 8 centered-instance (70.8 M)"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--arm", choices=sorted(ARMS), default="round10")
    parser.add_argument("--view", choices=("top", "side"), default="top")
    parser.add_argument("--pectoral-mode", choices=("interpolate", "annotated-only"),
                        default="interpolate")
    parser.add_argument("--epochs", type=int, default=ROUND8["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--paf-sigma", type=float, default=8.0)
    parser.add_argument("--paf-output-stride", type=int, default=2)
    parser.add_argument("--cache", default="C:/sleap_cache/s03_bu")
    parser.add_argument("--ckpt-dir", default="C:/sleap_cache/s03_runs")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable the per-epoch prediction overlays (on by default).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = ARMS[args.arm]
    batch_size = args.batch_size or spec["batch_size"]
    data_dir = DATA_ROOT / args.view / args.pectoral_mode
    suffix = "annotated" if args.pectoral_mode == "annotated-only" else "interp"
    train = data_dir / f"train_labels.3node_{suffix}.pkg.slp"
    val = data_dir / f"val_labels.3node_{suffix}.pkg.slp"
    for path in (train, val):
        if not path.is_file():
            raise SystemExit(
                f"missing {path}\nBuild it first:\n"
                f"  .venv\\Scripts\\python.exe tools/s02_merge_pectoral.py "
                f"--view {args.view} --pectoral-mode {args.pectoral_mode}"
            )

    # Read the skeleton from the data rather than restating it, so the edges
    # trained are provably the edges in the labels.
    labels = sio.load_slp(str(train))
    skeleton = labels.skeleton
    node_names = [n.name for n in skeleton.nodes]
    edges = [[e.source.name, e.destination.name] for e in skeleton.edges]
    n_train = len(labels.labeled_frames)

    paths = load_paths()
    # PAF sigma is in the run name because it is a swept parameter. Without it a
    # sigma sweep silently overwrites the arm it is meant to be compared against,
    # and the comparison quietly becomes a comparison of a run with itself.
    run_name = (f"s03.bottomup_3node.{args.arm}_f{spec['filters']}_ms{spec['max_stride']}"
                f"_paf{args.paf_sigma:g}_{args.pectoral_mode}_n{n_train}")
    ckpt_dir = Path(args.ckpt_dir) / args.view
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir = paths.work_dir / "s03_bottomup_3node" / args.view
    final_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train_labels_path": [str(train)],
        "val_labels_path": [str(val)],
        "validation_fraction": 0.0,
        "user_instances_only": True,
        "data_pipeline_fw": "torch_dataset_cache_img_disk",
        "cache_img_path": f"{args.cache}_{args.view}_{args.pectoral_mode}",
        "delete_cache_imgs_after_training": False,
        "ensure_rgb": True,
        "max_height": 1200,
        "max_width": 1200,
        "scale": 1.0,
        # Bottom-up is whole-frame: there is no centroid crop.
        "crop_size": None,
        "min_crop_size": ROUND8["min_crop_size"],
        "use_augmentations_train": True,
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
                "filters": spec["filters"],
                "filters_rate": ROUND8["filters_rate"],
                "max_stride": spec["max_stride"],
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": ROUND8["convs_per_block"],
                "output_stride": ROUND8["output_stride"],
            }
        },
        "head_configs": {
            "bottomup": {
                "confmaps": {
                    "part_names": node_names,
                    "sigma": ROUND8["sigma"],
                    "output_stride": ROUND8["output_stride"],
                    "loss_weight": 1.0,
                },
                "pafs": {
                    "edges": edges,
                    "sigma": args.paf_sigma,
                    "output_stride": args.paf_output_stride,
                    "loss_weight": 1.0,
                },
            }
        },
        "batch_size": batch_size,
        "shuffle_train": True,
        "num_workers": 8,
        "ckpt_save_top_k": 1,
        "ckpt_save_last": True,
        "trainer_accelerator": "auto",
        "enable_progress_bar": False,
        "min_train_steps_per_epoch": ROUND8["min_train_steps_per_epoch"],
        # Prediction overlays are written every epoch and kept afterwards, so
        # the PAF-sigma arms can be compared by eye and not only by a median.
        # A grouping failure — parts of two animals joined into one — is obvious
        # in an overlay and nearly invisible in a distance percentile.
        "visualize_preds_during_training": not args.no_viz,
        "keep_viz": not args.no_viz,
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
        # Deliberately reluctant. The val split is 15 frames, and this project
        # has repeatedly shown val/loss is a poor proxy for task quality, so
        # this exists to cap a runaway run rather than to pick a stopping point.
        # ckpt_save_top_k=1 still keeps the best epoch rather than the last.
        "early_stopping": True,
        "early_stopping_min_delta": 1e-8,
        "early_stopping_patience": 40,
        "online_mining": True,
        "hard_to_easy_ratio": 2.0,
        "min_hard_keypoints": 2,
        "loss_scale": 5.0,
    }

    print(__doc__.split("=====")[0])
    print(f"arm       : {args.arm}  ({spec['matches']})")
    print(f"backbone  : filters {spec['filters']}  max_stride {spec['max_stride']}")
    print(f"skeleton  : {' -> '.join(node_names)}   edges {edges}")
    print(f"PAF       : sigma {args.paf_sigma}  output_stride {args.paf_output_stride}")
    print(f"train     : {train.name}  ({n_train} frames)")
    print(f"val       : {val.name}  ({len(sio.load_slp(str(val)).labeled_frames)} frames)")
    print(f"epochs    : {args.epochs}  batch {batch_size}")
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
