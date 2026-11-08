"""Registration 2: appearance naming stage (N) and pose-masked crops (B) on P-1-1.

Subcommands, in order: ``mask`` (pose-masked copies of the registration-1
crops), ``train`` (contrastive encoders on usable masked crops) and
``evaluate`` (boundary decisions, the naming stage over all post-preparation
time, and the two registered rules). The method is fixed in
``labs/20261106_contrastive_appearance_link/README.md``, Registration 2. The
registration-1 run is read, never written. Reference labels are read only by
``evaluate``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_appearance_link as r1  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

REPO = r1.REPO
CONFIG = HERE / "config_appearance_naming_v1.toml"
VIEWS = r1.VIEWS
NAMES = ("F", "M1", "M2")


def settings():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["input"]["appearance_config"] != r1.CONFIG.name:
        raise ValueError("Registration-1 configuration name changed")
    r1cfg, _, root, r1out = r1.settings()
    for key in ("steps", "pairs", "seed"):
        if cfg["train"][key] != r1cfg["train"][key]:
            raise ValueError(f"train.{key} differs from registration 1")
    snapshot = r1out / "source" / r1.CONFIG.relative_to(REPO)
    if r1.sha256(snapshot) != r1.sha256(r1.CONFIG):
        raise ValueError("Registration-1 configuration differs from its run snapshot")
    return cfg, r1cfg, root, r1out, root / "jingyi/P-1-1" / cfg["run_id"]


def r1_crops(rec, r1out, view):
    """Registration-1 crops, checked against the hash its training recorded."""
    record = json.loads((r1out / "p11_appearance_train.provenance.json").read_text())
    path = r1out / "crops" / f"{view}.npz"
    return r1.verified(rec, f"r1_crops_{view}", path, record["inputs"][f"crops_{view}"]["sha256"])


def snapshot_sources(rec, out, cfg, folder="source"):
    for path in (
        Path(__file__),
        CONFIG,
        r1.CONFIG,
        Path(r1.__file__),
        REPO / cfg["lab"] / "README.md",
        REPO / cfg["lab"] / "appearance.py",
        REPO / "uv.lock",
    ):
        path = Path(path).resolve()
        dest = out / folder / path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        rec.input(path.relative_to(REPO).as_posix(), path, sha256=r1.sha256(path))


# --------------------------------------------------------------------------- mask


def channel_median(crops: np.ndarray) -> np.ndarray:
    """Exact per-channel median of uint8 pixels, by histogram."""
    flat = crops.reshape(-1, 3)
    out = np.empty(3, np.uint8)
    for c in range(3):
        counts = np.bincount(flat[:, c], minlength=256)
        out[c] = int(np.searchsorted(np.cumsum(counts), (flat.shape[0] + 1) / 2))
    return out


def mask_view(view, crops_path, cache, out_path, fill=None):
    """Pose-mask one view's crops; ``fill`` defaults to these crops' channel median."""
    lab = r1.load_lab()
    with np.load(cache) as c:
        frame_idx, points = c["frame_idx"].astype(np.int64), c["points"]
    with np.load(crops_path) as z:
        arrays = {
            k: z[k]
            for k in ("crops", "frame", "slot", "interval", "overlap", "row", "centre", "heading")
        }
    crops, frame, row = arrays["crops"], arrays["frame"], arrays["row"]
    if not np.array_equal(frame_idx[row], frame):
        raise ValueError(f"{view}: crop rows do not match detection frames")
    fill = channel_median(crops) if fill is None else np.asarray(fill, np.uint8)
    lo = np.searchsorted(frame_idx, frame, side="left")
    hi = np.searchsorted(frame_idx, frame, side="right")
    masked = np.empty_like(crops)
    occluded = np.empty(crops.shape[0], np.float32)
    for i in range(crops.shape[0]):
        masked[i], occluded[i] = lab.mask_crop(
            crops[i],
            points[lo[i] : hi[i]],
            int(row[i] - lo[i]),
            arrays["centre"][i],
            arrays["heading"][i],
            fill,
        )
    usable = occluded <= lab.MAX_OCCLUDED_FRACTION
    np.savez(
        out_path,
        crops=masked,
        occluded=occluded,
        usable=usable,
        fill=fill,
        **{k: arrays[k] for k in ("frame", "slot", "interval", "overlap", "row")},
    )
    overlap = arrays["overlap"]
    return {
        "view": view,
        "crops": int(crops.shape[0]),
        "fill_bgr": [int(v) for v in fill],
        "usable": int(usable.sum()),
        "usable_previously_overlap": int((usable & overlap).sum()),
        "unusable_previously_non_overlap": int((~usable & ~overlap).sum()),
        "occluded_quantiles": {str(q): float(np.quantile(occluded, q)) for q in (0.5, 0.9, 0.99)},
    }


def masked_sheet(path_in: Path, path_out: Path, n=60, seed=0):
    """Usable masked crops that registration 1 excluded as overlap."""
    import cv2

    with np.load(path_in) as z:
        pool = np.flatnonzero(z["usable"] & z["overlap"])
        pick = np.sort(
            np.random.default_rng(seed).choice(pool, size=min(n, pool.size), replace=False)
        )
        tiles = [cv2.resize(z["crops"][i], (96, 96), interpolation=cv2.INTER_NEAREST) for i in pick]
    rows = [np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles) - len(tiles) % 10, 10)]
    cv2.imwrite(str(path_out), np.vstack(rows))


def cmd_mask(cfg, r1cfg, root, r1out, out):
    track = root / r1cfg["input"]["tracking_run"]
    inputs = json.loads((track / "inputs.json").read_text())
    if json.loads((r1out / "status.json").read_text())["status"] != "evaluated":
        raise ValueError("Registration-1 run is not complete")
    out.mkdir(exist_ok=False)
    (out / "masked").mkdir()
    r1.write_json(out / "status.json", {"status": "mask running"})
    with provenance("p11_appearance_naming_mask", out) as rec:
        rec.param("configuration", cfg)
        snapshot_sources(rec, out, cfg)
        jobs = {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for view in VIEWS:
                crops = r1_crops(rec, r1out, view)
                cache = r1.verified(
                    rec,
                    f"detections_{view}",
                    Path(inputs[view]["cache"]),
                    inputs[view]["cache_sha256"],
                )
                jobs[view] = pool.submit(
                    mask_view, view, crops, cache, out / "masked" / f"{view}.npz"
                )
            results = [jobs[view].result() for view in VIEWS]
        for view in VIEWS:
            path = out / "masked" / f"{view}.npz"
            rec.output(f"masked_{view}", path, sha256=r1.sha256(path))
            masked_sheet(path, out / "masked" / f"{view}_recovered_sample.png")
        rec.param("results", results)
        r1.write_json(out / "mask_summary.json", results)
    r1.write_json(out / "status.json", {"status": "masked"})


# --------------------------------------------------------------------------- train


def train_view(view, masked_path, cfg, model_dir, embed_dir):
    import torch

    torch.set_num_threads(cfg["train"]["threads_per_view"])
    lab = r1.load_lab()
    with np.load(masked_path) as z:
        crops, interval, slot, usable = z["crops"], z["interval"], z["slot"], z["usable"]
    fragment = interval.astype(np.int64) * 3 + slot
    model, stats, log = lab.train_contrastive(
        crops,
        fragment,
        interval,
        usable,
        steps=cfg["train"]["steps"],
        pairs=cfg["train"]["pairs"],
        seed=cfg["train"]["seed"],
    )
    torch.save(
        {"state_dict": model.state_dict(), "mean": stats[0], "std": stats[1]},
        model_dir / f"masked__{view}.pt",
    )
    r1.write_json(model_dir / f"masked__{view}_log.json", log)
    np.save(embed_dir / f"masked__{view}.npy", lab.embed(model, crops, stats))
    return {"view": view, "training_crops": int(usable.sum()), "final": log[-1]}


def recorded_output(out, stage, name):
    record = json.loads((out / f"{stage}.provenance.json").read_text())
    return record["outputs"][name]["sha256"]


def cmd_train(cfg, r1cfg, root, r1out, out):
    (out / "models").mkdir(exist_ok=False)
    (out / "embeddings").mkdir(exist_ok=False)
    r1.write_json(out / "status.json", {"status": "train running"})
    with provenance("p11_appearance_naming_train", out) as rec:
        rec.param("train", cfg["train"])
        futures = []
        with ProcessPoolExecutor(max_workers=3) as pool:
            for view in VIEWS:
                path = r1.verified(
                    rec,
                    f"masked_{view}",
                    out / "masked" / f"{view}.npz",
                    recorded_output(out, "p11_appearance_naming_mask", f"masked_{view}"),
                )
                futures.append(
                    pool.submit(train_view, view, path, cfg, out / "models", out / "embeddings")
                )
            results = [f.result() for f in futures]
        for view in VIEWS:
            path = out / "embeddings" / f"masked__{view}.npy"
            rec.output(f"embedding_{view}", path, sha256=r1.sha256(path))
        rec.param("results", results)
        r1.write_json(out / "train_summary.json", results)
    r1.write_json(out / "status.json", {"status": "trained"})


# --------------------------------------------------------------------------- evaluate


def check_r1_embeddings(lab, r1out, view, crops, n, seed):
    """Recompute a random subset of registration-1 A3 embeddings from its model."""
    import torch

    stored = np.load(r1out / "embeddings" / f"full__{view}.npy")
    state = torch.load(r1out / "models" / f"full__{view}.pt", weights_only=False)
    model = lab.build_encoder().to(memory_format=torch.channels_last)
    model.load_state_dict(state["state_dict"])
    model.eval()
    pick = np.sort(np.random.default_rng(seed).choice(crops.shape[0], n, replace=False))
    again = lab.embed(model, crops[pick], (state["mean"], state["std"]))
    error = float(np.abs(again - stored[pick]).max())
    if error > 1e-4:
        raise ValueError(f"{view}: stored A3 embeddings do not match the r1 model ({error})")
    return stored, error


def naming(lab, feats, whole_rows, lower, initial):
    """Registration 2, Part 1 steps 1--2: interval names and how each was reached."""
    ids = lower.segment_id.tolist()
    descs = [[r1.describe(lab, feats, whole_rows, i, s) for s in range(3)] for i in ids]
    labels, rounds = lab.constrained_labels(descs, np.array(initial))
    centroids = lab.label_centroids(descs, labels)
    names, how = {}, {}
    for i, row, assigned in zip(ids, descs, labels, strict=True):
        if assigned.min() >= 0:
            names[i], how[i] = assigned.astype(np.int8), "global"
            continue
        partial = lab.assign_partial(row, centroids)
        if partial is None:
            names[i], how[i] = np.full(3, -1, np.int8), "unknown"
        else:
            kind = "partial_two" if (partial >= 0).all() else "partial_one"
            names[i], how[i] = partial, kind
    return names, how, rounds


def score(name, matched, weights, regions):
    """Correct / wrong / unknown fish-seconds of per-frame slot names, by region.

    Reference animal ``r`` sits in slot ``matched[t, r]``; it is named correctly
    when that slot's name is ``r`` (names are anchor-aligned reference indices).
    """
    held = np.take_along_axis(name, np.maximum(matched, 0).astype(np.int64), axis=1)
    held = np.where(matched >= 0, held, -1)
    animal = np.arange(3)[None]
    assessed = (matched >= 0) & (weights[:, None] > 0)
    masks = {
        "correct": assessed & (held == animal),
        "wrong": assessed & (held >= 0) & (held != animal),
        "unknown": assessed & (held < 0),
    }
    out = {}
    for region, keep in regions.items():
        w = (weights * keep)[:, None]
        res = {k: float((m * w).sum()) for k, m in masks.items()}
        res["named"] = res["correct"] + res["wrong"]
        res["assessed"] = res["named"] + res["unknown"]
        res["wrong_fraction_of_named"] = res["wrong"] / res["named"] if res["named"] else np.nan
        res["named_fraction_of_assessed"] = (
            res["named"] / res["assessed"] if res["assessed"] else np.nan
        )
        res["wrong_by_animal"] = {
            NAMES[r]: float((masks["wrong"][:, r] * w[:, 0]).sum()) for r in range(3)
        }
        out[region] = res
    return out


def physical_naming(colour_raw, length, meta, names, lower):
    """Per identity and camera: median red chromaticity and mouth-tail length."""
    rows = []
    for view in VIEWS:
        m = meta[view]
        keep = ~m["overlap"]
        ident = np.full(m["frame"].size, -1)
        for seg in lower.segment_id:
            sel = m["interval"] == seg
            ident[sel] = names[seg][m["slot"][sel]]
        for k in range(3):
            pick = keep & (ident == k)
            rows.append(
                {
                    "view": view,
                    "identity": k,
                    "anchor_label": NAMES[k],
                    "crops": int(pick.sum()),
                    "median_r_fraction": float(np.median(colour_raw[view][pick, 0])),
                    "median_saturation": float(np.median(colour_raw[view][pick, 4])),
                    "median_length_native_px": float(np.nanmedian(length[view][pick])),
                }
            )
    return pd.DataFrame(rows)


def identity_sheet(crops, m, names, lower, path, per_identity=24, seed=0):
    """One row per named identity: one non-overlap crop from each of many intervals."""
    import cv2

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(3):
        tiles = []
        segments = [s for s in lower.segment_id if (names[s] == k).any()]
        for seg in sorted(
            rng.choice(segments, size=min(per_identity, len(segments)), replace=False)
        ):
            slot = int(np.flatnonzero(names[seg] == k)[0])
            pool = np.flatnonzero((m["interval"] == seg) & (m["slot"] == slot) & ~m["overlap"])
            if pool.size:
                tiles.append(
                    cv2.resize(crops[rng.choice(pool)], (96, 96), interpolation=cv2.INTER_NEAREST)
                )
        tiles += [np.zeros((96, 96, 3), np.uint8)] * (per_identity - len(tiles))
        label = np.zeros((96, 60, 3), np.uint8)
        cv2.putText(label, NAMES[k], (4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(np.hstack([label, *tiles[:per_identity]]))
    cv2.imwrite(str(path), np.vstack(rows))


def rule_part1(result, ev):
    a = result["all"]
    if a["wrong_fraction_of_named"] > ev["max_wrong_fraction_of_named"]:
        return "reject"
    if a["named_fraction_of_assessed"] >= ev["min_named_fraction_of_assessed"]:
        return "carry_forward"
    return "inconclusive"


def rule_part2(n_a3, n_b, wrong_a3, wrong_b, ev):
    a3, b = n_a3["all"], n_b["all"]
    if (
        b["named"] < a3["named"]
        or b["wrong_fraction_of_named"] > ev["max_wrong_fraction_of_named"]
        or wrong_b > wrong_a3
    ):
        return "reject"
    if b["named"] > a3["named"]:
        return "adopt"
    return "inconclusive"


def cmd_evaluate(cfg, r1cfg, root, r1out, out):
    lab = r1.load_lab()
    ev = cfg["evaluate"]
    if ev["end_window_s"] != r1cfg["evaluate"]["end_window_s"]:
        raise ValueError("End window differs from registration 1")
    track, warn = root / r1cfg["input"]["tracking_run"], root / r1cfg["input"]["warning_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    warn_manifest = json.loads((warn / "artifact_manifest.json").read_text())
    dest = out / "evaluation"
    dest.mkdir(exist_ok=False)
    r1.write_json(out / "status.json", {"status": "evaluate running"})
    with provenance("p11_appearance_naming_evaluate", out) as rec:
        snapshot_sources(rec, out, cfg, folder="source_evaluate")
        scoring = r1.verified(
            rec,
            "scoring",
            track / r1cfg["input"]["scoring"],
            manifest[r1cfg["input"]["scoring"]]["sha256"],
        )
        chain = r1.verified(
            rec,
            "c34_label",
            track / r1cfg["input"]["chain"],
            manifest[r1cfg["input"]["chain"]]["sha256"],
        )
        lower_path = r1.verified(
            rec,
            "lower_risk_intervals",
            warn / r1cfg["input"]["lower_risk"],
            warn_manifest[r1cfg["input"]["lower_risk"]]["sha256"],
        )
        mask_path = r1.verified(
            rec,
            "warning_mask",
            warn / r1cfg["input"]["warning_mask"],
            warn_manifest[r1cfg["input"]["warning_mask"]]["sha256"],
        )
        lower = pd.read_csv(lower_path)
        with np.load(scoring) as z:
            times, weights = z["recording_time_s"], z["duration_s"]
            matched, anchor = z["matched_slot"], z["anchor_reference_to_slot"]
            labels = tuple(str(x) for x in z["reference_labels"])
        if labels != NAMES:
            raise ValueError(f"Unexpected reference labels {labels}")
        with np.load(chain) as z:
            xyz = z["xyz"]
        with np.load(mask_path) as z:
            warning = z["warning"]
        n = weights.size
        if xyz.shape[:2] != (n, 3):
            raise ValueError("Chain xyz does not match the scoring frames")
        in_lower = np.zeros(n, bool)
        for row in lower.itertuples():
            in_lower[row.start_frame : row.stop_frame_exclusive] = True
        post = weights > 0
        if np.any(in_lower & warning) or np.any(post & ~in_lower & ~warning):
            raise ValueError("Lower-risk and warning frames must partition scored time")
        regions = {"all": post, "lower_risk": post & in_lower, "warning": post & warning}

        meta_a3, meta_b, colour_raw, length, feats = {}, {}, {}, {}, {"A3": {}, "B": {}}
        checks, crops = {}, {}
        for v, view in enumerate(VIEWS):
            with np.load(r1_crops(rec, r1out, view)) as z:
                crops[view] = z["crops"]
                meta_a3[view] = {k: z[k] for k in ("frame", "slot", "interval", "overlap")}
                colour_raw[view], length[view] = z["color"].astype(np.float64), z["length"]
            feats["A3"][view], checks[view] = check_r1_embeddings(
                lab, r1out, view, crops[view], ev["embedding_check_crops"], seed=v
            )
            masked = r1.verified(
                rec,
                f"masked_{view}",
                out / "masked" / f"{view}.npz",
                recorded_output(out, "p11_appearance_naming_mask", f"masked_{view}"),
            )
            with np.load(masked) as z:
                for key in ("frame", "slot", "interval"):
                    if not np.array_equal(z[key], meta_a3[view][key]):
                        raise ValueError(f"{view}: masked crops are not aligned with r1 crops")
                meta_b[view] = {
                    **{k: meta_a3[view][k] for k in ("frame", "slot", "interval")},
                    "overlap": ~z["usable"],
                }
            emb = r1.verified(
                rec,
                f"embedding_{view}",
                out / "embeddings" / f"masked__{view}.npy",
                recorded_output(out, "p11_appearance_naming_train", f"embedding_{view}"),
            )
            feats["B"][view] = np.load(emb)
        rec.param("r1_embedding_check_max_abs_error", checks)

        ref = r1.reference_slots(lower, list(labels), ev["min_reference_coverage"])
        initial = [int(np.flatnonzero(anchor == s)[0]) for s in range(3)]
        rows = {
            "A3": r1.fragment_rows(meta_a3, times, lower, ev["end_window_s"]),
            "B": r1.fragment_rows(meta_b, times, lower, ev["end_window_s"]),
        }

        tables = [
            r1.boundary_table(lab, feats["A3"], *rows["A3"][:2], lower, ref, "A0_chain", "end")
        ]
        for method in ("A3", "B"):
            pre, post_rows, whole = rows[method]
            tables.append(
                r1.boundary_table(lab, feats[method], pre, post_rows, lower, ref, method, "end")
            )
            tables.append(
                r1.boundary_table(lab, feats[method], whole, whole, lower, ref, method, "whole")
            )
        boundaries = pd.concat(tables, ignore_index=True)
        boundaries.to_csv(dest / "boundaries.csv", index=False)
        summary = {"boundary_outcomes": {}, "naming": {}, "intervals": {}}
        for (method, window), table in boundaries.groupby(["method", "window"]):
            summary["boundary_outcomes"][f"{method}|{window}"] = r1.outcome_counts(table)
        r1_summary = json.loads((r1out / "evaluation" / "summary.json").read_text())
        for window in ("end", "whole"):
            if (
                summary["boundary_outcomes"][f"A3|{window}"]
                != r1_summary["boundary_outcomes"][f"A3_contrastive|{window}"]
            ):
                raise AssertionError(f"A3 {window} boundaries do not reproduce registration 1")
        lo_c, hi_c = ev["cluster_window_s"]
        cluster = boundaries[
            (boundaries.boundary_s >= lo_c)
            & (boundaries.boundary_s <= hi_c)
            & (boundaries.chain_correct == False)  # noqa: E712
        ]
        summary["cluster_chain_wrong_boundaries"] = cluster[
            ["method", "window", "boundary_s", "outcome", "margin"]
        ].to_dict("records")

        bounds = [(r.segment_id, r.start_frame, r.stop_frame_exclusive) for r in lower.itertuples()]
        chain_names = {i: np.array(initial, np.int8) for i in lower.segment_id}
        a0 = score(lab.frame_names(n, bounds, chain_names), matched, weights, regions)
        reference_a0 = r1.identity_time(chain_names, lower, matched, anchor, weights)
        if not np.isclose(a0["lower_risk"]["wrong"], lower.wrong_seconds.sum(), atol=1e-6):
            raise AssertionError("Chain naming does not reproduce the frozen lower-risk score")
        if not np.isclose(a0["lower_risk"]["wrong"], reference_a0["wrong_seconds"], atol=1e-6):
            raise AssertionError("Frame scorer disagrees with paired_identity_time")
        from medaka3d.eval.flip_risk import paired_identity_time

        whole_chain = paired_identity_time(matched, matched, anchor, np.where(post, weights, 0))
        if not np.isclose(a0["all"]["wrong"], whole_chain["wrong_seconds"], atol=1e-6):
            raise AssertionError("Frame scorer disagrees with the whole-recording chain score")
        summary["naming"]["A0_chain"] = a0

        products = {}
        for method in ("A3", "B"):
            names, how, rounds = naming(lab, feats[method], rows[method][2], lower, initial)
            if method == "A3":
                old = pd.read_csv(r1out / "evaluation" / "global_names_A3_contrastive.csv")
                for row in old.itertuples():
                    if how[row.segment_id] == "global" and list(names[row.segment_id]) != [
                        row.slot0,
                        row.slot1,
                        row.slot2,
                    ]:
                        raise AssertionError("A3 global names do not reproduce registration 1")
            summary["intervals"][method] = {
                "rounds": rounds,
                **{
                    k: sum(h == k for h in how.values())
                    for k in ("global", "partial_two", "partial_one", "unknown")
                },
            }
            pd.DataFrame(
                [
                    {
                        "segment_id": i,
                        "how": how[i],
                        **{f"slot{s}": int(names[i][s]) for s in range(3)},
                    }
                    for i in lower.segment_id
                ]
            ).to_csv(dest / f"interval_names_{method}.csv", index=False)
            for carry in (True, False):
                label = f"N_{method}|{'carry' if carry else 'strict'}"
                frame = lab.frame_names(n, bounds, names, carry_warnings=carry)
                summary["naming"][label] = score(frame, matched, weights, regions)
                if carry:
                    products[method] = (frame, names)
            frame = products[method][0]
            by_identity = np.full((n, 3, xyz.shape[2]), np.nan, np.float32)
            for k in range(3):
                for s in range(3):
                    sel = frame[:, s] == k
                    by_identity[sel, k] = xyz[sel, s]
            np.savez_compressed(
                dest / f"identity_tracks_N_{method}.npz",
                name=frame,
                xyz_by_identity=by_identity,
                identity_labels=np.array(NAMES),
                recording_time_s=times,
            )
            rec.output(f"identity_tracks_N_{method}", dest / f"identity_tracks_N_{method}.npz")

        physical = physical_naming(colour_raw, length, meta_a3, products["A3"][1], lower)
        physical.to_csv(dest / "physical_naming.csv", index=False)
        for view in VIEWS:
            identity_sheet(
                crops[view],
                meta_a3[view],
                products["A3"][1],
                lower,
                dest / f"identity_sheet_{view}.png",
            )
        males = physical[physical.anchor_label != "F"]
        summary["physical_naming"] = {
            view: str(g.sort_values("median_r_fraction").anchor_label.iloc[-1])
            for view, g in males.groupby("view")
        }

        p1 = rule_part1(summary["naming"]["N_A3|carry"], ev)
        wrong = {m: summary["boundary_outcomes"][f"{m}|end"]["wrong"] for m in ("A3", "B")}
        p2 = rule_part2(
            summary["naming"]["N_A3|carry"],
            summary["naming"]["N_B|carry"],
            wrong["A3"],
            wrong["B"],
            ev,
        )
        summary["decision_rules"] = {
            "part1_naming_stage_with_A3": p1,
            "part2_masked_crops": p2,
            "part1_inputs": {
                k: summary["naming"]["N_A3|carry"]["all"][k]
                for k in ("wrong_fraction_of_named", "named_fraction_of_assessed")
            },
            "part2_inputs": {
                "named_A3": summary["naming"]["N_A3|carry"]["all"]["named"],
                "named_B": summary["naming"]["N_B|carry"]["all"]["named"],
                "wrong_fraction_B": summary["naming"]["N_B|carry"]["all"][
                    "wrong_fraction_of_named"
                ],
                "wrong_boundaries_A3": wrong["A3"],
                "wrong_boundaries_B": wrong["B"],
            },
        }
        r1.write_json(dest / "summary.json", summary)
        rec.output("summary", dest / "summary.json")
        rec.param("reference_use", "evaluation only; fixed 298-300 s anchor names interval 0")
    r1.write_json(out / "status.json", {"status": "evaluated"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mask", "train", "evaluate"))
    args = parser.parse_args(argv)
    cfg, r1cfg, root, r1out, out = settings()
    if args.stage != "mask" and not out.is_dir():
        raise SystemExit(f"Run directory does not exist: {out}")
    {"mask": cmd_mask, "train": cmd_train, "evaluate": cmd_evaluate}[args.stage](
        cfg, r1cfg, root, r1out, out
    )


if __name__ == "__main__":
    main()
