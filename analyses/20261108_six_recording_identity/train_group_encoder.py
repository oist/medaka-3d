"""Phase 3: one appearance encoder per trio, trained on three recordings' clean crops.

The pooled training set is the union of the three recordings' prepared clean
crops; a fragment is (recording, lower-risk interval, slot), so positives
never cross recordings. Loss, batch, seed, encoder and the silhouette
stopping rule are Registration 4's (`config_appearance_group_v1.toml`), the
silhouette being computed on a pooled validation sample with three clusters.
The same weights are written into each recording's run directory together
with that recording's embeddings, so `run_appearance_clean.py evaluate`
then names, bridges and fills gaps per recording exactly as for the
per-recording variant. Each run must be at status "prepared" (bundle and
prepare of a prepared_run configuration).

Usage: train_group_encoder.py --group K_1 --config <config_appearance_group_v1.toml>
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
P11_ANALYSIS = HERE.parent / "20261031_p11_identity_evaluation"
sys.path.insert(0, str(P11_ANALYSIS))
import run_appearance_clean as rac  # noqa: E402
import run_appearance_link as r1  # noqa: E402

INTERVAL_OFFSET = 1_000_000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    import torch

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    keys = list(cfg["groups"][args.group])
    tc = cfg["train"]
    runs = {}
    for key in keys:
        _, rc, _, _, out = rac.settings(key, args.config)
        rac.require_status(out, "prepared")
        runs[key] = (rc, out)
    for _, out in runs.values():
        for sub in ("models", "embeddings", "embeddings_warning"):
            (out / sub).mkdir(exist_ok=False)
        rac.status(out, "train running")

    torch.set_num_threads(tc["threads_per_model"])
    lab = r1.load_lab()
    results = {key: [] for key in keys}
    for name, views in tc["models"].items():
        parts = []
        for ri, key in enumerate(keys):
            _, out = runs[key]
            for v, view in enumerate(views):
                with np.load(rac.run_clean_path(out, "clean", view)) as z:
                    idx = np.flatnonzero(z["clean"])
                    parts.append(
                        {
                            "crops": z["crops"][idx],
                            "frame": z["frame"][idx],
                            "slot": z["slot"][idx],
                            "interval": z["interval"][idx].astype(np.int64) + ri * INTERVAL_OFFSET,
                            "camera": np.full(idx.size, v, np.int8),
                            "recording": np.full(idx.size, ri, np.int8),
                            "index": idx,
                        }
                    )
        crops = np.concatenate([p.pop("crops") for p in parts])
        meta = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        fragment = meta["interval"] * 3 + meta["slot"]
        rng = np.random.default_rng(tc["seed"])
        validation = np.sort(
            rng.choice(len(crops), min(tc["validation_crops"], len(crops)), replace=False)
        )
        cross = None
        if len(views) > 1:
            df = pd.DataFrame({k: meta[k] for k in ("frame", "slot", "interval", "camera")})
            df["i"] = np.arange(len(crops))
            pairs = (
                df[df.camera == 0]
                .merge(df[df.camera == 1], on=["frame", "slot", "interval"], suffixes=("_a", "_b"))[
                    ["i_a", "i_b"]
                ]
                .to_numpy()
            )
            if len(pairs) > tc["cross_pairs"]:
                pairs = pairs[np.sort(rng.choice(len(pairs), tc["cross_pairs"], replace=False))]
            cross = pairs
        monitor = lab.SilhouetteMonitor(
            validation,
            cross,
            patience=tc["patience"],
            target=tc["target"],
            target_patience=tc["target_patience"],
            seed=tc["seed"],
        )
        print(
            f"{args.group} {name}: {len(crops)} crops, {np.unique(fragment).size} fragments",
            flush=True,
        )
        model, stats, log = lab.train_contrastive(
            crops,
            fragment,
            meta["interval"],
            np.ones(len(crops), bool),
            steps=tc["max_steps"],
            pairs=tc["pairs"],
            seed=tc["seed"],
            log_every=tc["eval_every"],
            monitor=monitor,
            encoder=tc["encoder"],
        )
        device = str(lab.model_device(model))
        per_recording = {keys[ri]: int((meta["recording"] == ri).sum()) for ri in range(len(keys))}
        del crops
        best_score, best_step, _ = monitor.best
        ev = monitor.evaluations
        payload = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "mean": stats[0],
            "std": stats[1],
            "best_step": best_step,
            "best_silhouette": best_score,
            "stopped": monitor.stopped,
            "views": list(views),
            "encoder": tc["encoder"],
            "out_dim": model.out_dim,
            "device": device,
            "group": args.group,
            "recordings": [runs[k][0]["experiment"] for k in keys],
        }
        history = dict(
            snapshots=np.stack(monitor.snapshots),
            clusters=np.stack(monitor.clusters),
            steps=np.array([e["step"] for e in ev]),
            silhouette=np.array([e["silhouette"] for e in ev]),
            inertia=np.array([e["inertia"] for e in ev]),
            cross_camera_median=np.array([e["cross_camera_median"] for e in ev]),
            validation=validation,
            validation_camera=meta["camera"][validation],
            validation_frame=meta["frame"][validation],
            validation_slot=meta["slot"][validation],
            validation_interval=meta["interval"][validation],
            validation_index=meta["index"][validation],
            validation_recording=meta["recording"][validation],
            views=np.array(views),
            cross_pairs=np.zeros((0, 2), np.int64) if cross is None else cross,
        )
        summary = {
            "model": name,
            "views": list(views),
            "encoder": tc["encoder"],
            "device": device,
            "group": args.group,
            "training_crops": len(fragment),
            "training_crops_per_recording": per_recording,
            "fragments": int(np.unique(fragment).size),
            "validation_crops": int(validation.size),
            "cross_pairs": 0 if cross is None else len(cross),
            "evaluations": len(ev),
            "best_step": best_step,
            "best_silhouette": best_score,
            "stopped": monitor.stopped,
            "last_step": ev[-1]["step"],
            "final_log": log[-1] if log else None,
        }
        for key in keys:
            _, out = runs[key]
            torch.save(payload, out / "models" / f"{name}.pt")
            r1.write_json(out / "models" / f"{name}_log.json", log)
            np.savez_compressed(out / "models" / f"{name}_history.npz", **history)
            for view in views:
                for sub, dest in (("clean", "embeddings"), ("clean_warning", "embeddings_warning")):
                    with np.load(rac.run_clean_path(out, sub, view)) as z:
                        emb = lab.embed(model, z["crops"], stats)
                    np.save(out / dest / f"{view}.npy", emb)
            results[key].append(summary)
        del model

    for key in keys:
        rc, out = runs[key]
        with provenance("appearance_clean_train", out) as rec:
            rec.param("train", tc)
            rec.param(
                "group",
                {
                    "name": args.group,
                    "recordings": keys,
                    "script": Path(__file__).name,
                    "interval_offset": INTERVAL_OFFSET,
                },
            )
            for other in keys:
                inputs = rac.bundle_inputs(runs[other][1])
                for view in rac.VIEWS:
                    entry = inputs.get(f"clean_{view}")
                    if entry is not None:
                        rec.input(f"clean_{other}_{view}", entry["path"], sha256=entry["sha256"])
            for name in tc["models"]:
                for suffix in (".pt", "_log.json", "_history.npz"):
                    path = out / "models" / f"{name}{suffix}"
                    rec.output(f"model_{name}{suffix}", path, sha256=r1.sha256(path))
            for view in rac.VIEWS:
                for sub in ("embeddings", "embeddings_warning"):
                    path = out / sub / f"{view}.npy"
                    rec.output(f"{sub}_{view}", path, sha256=r1.sha256(path))
            rec.param("results", results[key])
            rec.param("device", sorted({r["device"] for r in results[key]}))
            r1.write_json(out / "train_summary.json", results[key])
        rac.status(out, "trained")
        print(f"{key}: trained", flush=True)


if __name__ == "__main__":
    main()
