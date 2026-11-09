"""Step 1: label-free contrastive appearance for linking P-1-1 geometry fragments.

Subcommands run the stages of one pre-registered experiment, in order:
``extract`` (crops and colour features), ``train`` (contrastive encoders) and
``evaluate`` (boundary decisions and whole-recording labelling). The method is
fixed in ``labs/20261106_contrastive_appearance_link/README.md``. Reference
labels are read only by ``evaluate``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d.paths import load_paths
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "config_appearance_link_v1.toml"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def settings():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(cfg["input"]["views"]) != VIEWS:
        raise ValueError("View order must match the C34 source columns")
    paths = load_paths()
    root = paths.raw_data_dir.parent
    return cfg, paths, root, root / "jingyi/P-1-1" / cfg["run_id"]


def load_lab():
    path = REPO / "labs/20261106_contrastive_appearance_link/appearance.py"
    spec = importlib.util.spec_from_file_location("p11_contrastive_appearance", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def verified(rec, name, path, expected):
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"Frozen input changed: {path}")
    rec.input(name, path, sha256=digest)
    return path


def video_hashes(provenance_path: Path) -> dict:
    """Source-video records written by the checksum-verified review pack."""
    found = {}

    def walk(value):
        if isinstance(value, dict):
            if str(value.get("path", "")).endswith(".mp4") and "sha256" in value:
                found[Path(value["path"]).parent.name] = value
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json.loads(provenance_path.read_text(encoding="utf-8")))
    return found


def lower_risk_frames(lower: pd.DataFrame, n: int, step: int):
    interval = np.full(n, -1, np.int32)
    for row in lower.itertuples():
        interval[row.start_frame : row.stop_frame_exclusive] = row.segment_id
    frames = np.flatnonzero((interval >= 0) & (np.arange(n) % step == 0))
    return interval, frames


# --------------------------------------------------------------------------- extract


def extract_view(view, video, cache, source, frames, interval, out_path, out_px=None):
    """Decode one view and build every sampled crop; ``out_px`` defaults to the lab's 64."""
    import cv2

    lab = load_lab()
    out_px = out_px or lab.OUT_PX
    with np.load(cache) as c:
        frame_idx, points = c["frame_idx"].astype(np.int64), c["points"]
    if np.any(np.diff(frame_idx) < 0):
        raise ValueError("Detection rows are not frame-sorted")
    start = np.searchsorted(frame_idx, np.arange(source.shape[0] + 1))
    want = np.zeros(source.shape[0], bool)
    want[frames] = True
    keep = {
        k: []
        for k in (
            "crops",
            "frame",
            "slot",
            "interval",
            "overlap",
            "centre",
            "heading",
            "length",
            "color",
            "row",
        )
    }
    counts = {"no_source": 0, "no_pose": 0, "crops": 0}
    cap = cv2.VideoCapture(str(video))
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source.shape[0]:
        raise ValueError(f"{view}: video frame count differs from tracking")
    for t in range(int(frames[-1]) + 1):
        if not cap.grab():
            raise RuntimeError(f"{view}: decode stopped at frame {t}")
        if not want[t]:
            continue
        ok, image = cap.retrieve()
        if not ok:
            raise RuntimeError(f"{view}: retrieve failed at frame {t}")
        lo, hi = start[t], start[t + 1]
        for slot in range(3):
            k = int(source[t, slot])
            if k < 0:
                counts["no_source"] += 1
                continue
            row = lo + k
            if row >= hi or frame_idx[row] != t:
                raise ValueError(f"{view}: source row {k} absent at frame {t}")
            pose = lab.crop_pose(points[row])
            if pose is None:
                counts["no_pose"] += 1
                continue
            centre, heading, length = pose
            crop = lab.extract_crop(image, centre, heading, out_px)
            keep["crops"].append(crop)
            keep["frame"].append(t)
            keep["slot"].append(slot)
            keep["interval"].append(interval[t])
            keep["overlap"].append(lab.overlaps(points[lo:hi], k, centre))
            keep["centre"].append(centre)
            keep["heading"].append(heading)
            keep["length"].append(length)
            keep["color"].append(lab.color_features(crop, length))
            keep["row"].append(row)
    cap.release()
    arrays = {k: np.asarray(v) for k, v in keep.items()}
    counts["crops"] = int(arrays["frame"].size)
    counts["overlap"] = int(arrays["overlap"].sum())

    # Seek-decoded frames must reproduce the sequentially decoded crops.
    checks = []
    cap = cv2.VideoCapture(str(video))
    for i in np.linspace(0, arrays["frame"].size - 1, 5).astype(int):
        t = int(arrays["frame"][i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, image = cap.read()
        again = lab.extract_crop(image, arrays["centre"][i], arrays["heading"][i], out_px)
        checks.append({"frame": t, "identical": bool(np.array_equal(again, arrays["crops"][i]))})
    cap.release()
    np.savez(out_path, **arrays)
    return {"view": view, "counts": counts, "seek_checks": checks, "video_sha256": sha256(video)}


def contact_sheet(path_in: Path, path_out: Path, n=60, seed=0):
    import cv2

    with np.load(path_in) as z:
        usable = np.flatnonzero(~z["overlap"])
        pick = np.sort(
            np.random.default_rng(seed).choice(usable, size=min(n, usable.size), replace=False)
        )
        tiles = [cv2.resize(z["crops"][i], (96, 96), interpolation=cv2.INTER_NEAREST) for i in pick]
    rows = [np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles) - len(tiles) % 10, 10)]
    cv2.imwrite(str(path_out), np.vstack(rows))


def cmd_extract(cfg, root, out):
    track, warn = root / cfg["input"]["tracking_run"], root / cfg["input"]["warning_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    warn_manifest = json.loads((warn / "artifact_manifest.json").read_text())
    inputs = json.loads((track / "inputs.json").read_text())
    videos = video_hashes(track / cfg["input"]["video_provenance"])
    out.mkdir(exist_ok=False)
    (out / "crops").mkdir()
    write_json(out / "status.json", {"status": "extract running"})
    with provenance("p11_appearance_extract", out) as rec:
        rec.param("configuration", cfg)
        snapshot = out / "source"
        for path in (
            Path(__file__),
            CONFIG,
            REPO / cfg["lab"] / "README.md",
            REPO / cfg["lab"] / "appearance.py",
            REPO / "uv.lock",
        ):
            dest = snapshot / path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            rec.input(path.relative_to(REPO).as_posix(), path, sha256=sha256(path))
        chain = verified(
            rec,
            "c34_label",
            track / cfg["input"]["chain"],
            manifest[cfg["input"]["chain"]]["sha256"],
        )
        verified(
            rec,
            "warning_mask",
            warn / cfg["input"]["warning_mask"],
            warn_manifest[cfg["input"]["warning_mask"]]["sha256"],
        )
        lower_path = verified(
            rec,
            "lower_risk_intervals",
            warn / cfg["input"]["lower_risk"],
            warn_manifest[cfg["input"]["lower_risk"]]["sha256"],
        )
        lower = pd.read_csv(lower_path)
        with np.load(chain) as label:
            source = label["source"]
        with np.load(warn / cfg["input"]["warning_mask"]) as w:
            warning = w["warning"]
        interval, frames = lower_risk_frames(lower, source.shape[0], cfg["crops"]["frame_step"])
        if np.any(warning[interval >= 0]):
            raise ValueError("Lower-risk intervals overlap the warning mask")
        jobs = {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for v, view in enumerate(VIEWS):
                cache = verified(
                    rec,
                    f"detections_{view}",
                    Path(inputs[view]["cache"]),
                    inputs[view]["cache_sha256"],
                )
                video = Path(videos[view]["path"])
                if video.stat().st_size != videos[view]["size"]:
                    raise ValueError(f"{view}: source video size changed")
                jobs[view] = pool.submit(
                    extract_view,
                    view,
                    video,
                    cache,
                    source[:, :, v],
                    frames,
                    interval,
                    out / "crops" / f"{view}.npz",
                )
            results = [jobs[view].result() for view in VIEWS]
        for result in results:
            view = result["view"]
            if result["video_sha256"] != videos[view]["sha256"]:
                raise ValueError(f"{view}: source video checksum changed")
            rec.input(f"video_{view}", videos[view]["path"], sha256=result["video_sha256"])
            rec.output(f"crops_{view}", out / "crops" / f"{view}.npz")
            contact_sheet(out / "crops" / f"{view}.npz", out / "crops" / f"{view}_sample.png")
        rec.param("extraction", results)
        write_json(out / "extract_summary.json", results)
    write_json(out / "status.json", {"status": "extracted"})


# --------------------------------------------------------------------------- train


def train_view(view, variant, crops_path, lower_path, cfg, model_dir, embed_dir):
    import torch

    torch.set_num_threads(cfg["train"]["threads_per_view"])
    lab = load_lab()
    lower = pd.read_csv(lower_path)
    with np.load(crops_path) as z:
        crops, interval, slot, overlap = z["crops"], z["interval"], z["slot"], z["overlap"]
    usable = ~overlap
    if variant == "early":
        early = set(lower.segment_id[lower.end_s < cfg["train"]["early_train_end_s"]])
        usable &= np.isin(interval, list(early))
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
        model_dir / f"{variant}__{view}.pt",
    )
    write_json(model_dir / f"{variant}__{view}_log.json", log)
    np.save(embed_dir / f"{variant}__{view}.npy", lab.embed(model, crops, stats))
    return {"view": view, "variant": variant, "training_crops": int(usable.sum()), "final": log[-1]}


def cmd_train(cfg, root, out):
    warn = root / cfg["input"]["warning_run"]
    lower_path = warn / cfg["input"]["lower_risk"]
    (out / "models").mkdir(exist_ok=False)
    (out / "embeddings").mkdir(exist_ok=False)
    write_json(out / "status.json", {"status": "train running"})
    with provenance("p11_appearance_train", out) as rec:
        rec.param("train", cfg["train"])
        for view in VIEWS:
            crops = out / "crops" / f"{view}.npz"
            rec.input(f"crops_{view}", crops, sha256=sha256(crops))
        futures = []
        with ProcessPoolExecutor(max_workers=3) as pool:
            for variant in ("full", "early"):
                for view in VIEWS:
                    crops = out / "crops" / f"{view}.npz"
                    futures.append(
                        pool.submit(
                            train_view,
                            view,
                            variant,
                            crops,
                            lower_path,
                            cfg,
                            out / "models",
                            out / "embeddings",
                        )
                    )
            results = [f.result() for f in futures]
        rec.param("results", results)
        write_json(out / "train_summary.json", results)
    write_json(out / "status.json", {"status": "trained"})


# --------------------------------------------------------------------------- evaluate


def fragment_rows(meta, times, lower, window_s):
    """Per (interval, slot, view): usable crop rows for pre-end, post-start and whole."""
    pre, post, whole = {}, {}, {}
    starts = lower.set_index("segment_id").start_frame
    stops = lower.set_index("segment_id").stop_frame_exclusive
    for view, m in meta.items():
        t = times[m["frame"]]
        for (i, s), rows in pd.Series(np.arange(m["frame"].size)).groupby(
            [m["interval"], m["slot"]]
        ):
            rows = rows.to_numpy()
            rows = rows[~m["overlap"][rows]]
            whole[i, s, view] = rows
            pre[i, s, view] = rows[t[rows] >= times[stops[i] - 1] - window_s]
            post[i, s, view] = rows[t[rows] <= times[starts[i]] + window_s]
    return pre, post, whole


def describe(lab, features, rows_by_key, i, s):
    out = {}
    for view, feats in features.items():
        rows = rows_by_key.get((i, s, view))
        if rows is not None:
            d = lab.descriptor(feats, rows)
            if d is not None:
                out[view] = d
    return out or None


def reference_slots(lower, labels, min_coverage):
    """Per interval: reference identity index per slot, or None if not fully known."""
    out = {}
    for row in lower.itertuples():
        idx = []
        for s in range(3):
            label = getattr(row, f"slot{s}_majority_label")
            purity = getattr(row, f"slot{s}_reference_purity")
            coverage = getattr(row, f"slot{s}_reference_coverage")
            ok = isinstance(label, str) and purity == 1.0 and coverage >= min_coverage
            idx.append(labels.index(label) if ok else -1)
        out[row.segment_id] = idx if min(idx) >= 0 and len(set(idx)) == 3 else None
    return out


def boundary_table(lab, features, pre_rows, post_rows, lower, ref, method, window):
    rows = []
    ids = lower.segment_id.to_numpy()
    for a, b in pairwise(ids):
        dpre = [describe(lab, features, pre_rows, a, s) for s in range(3)]
        dpost = [describe(lab, features, post_rows, b, s) for s in range(3)]
        distance = np.full((3, 3), np.nan)
        if all(dpre) and all(dpost):
            for s in range(3):
                for t in range(3):
                    distance[s, t] = lab.pair_distance(dpre[s], dpost[t])
        perm, margin = lab.decide(distance) if method != "A0_chain" else (np.arange(3), np.nan)
        reference = None
        if ref[a] is not None and ref[b] is not None:
            reference = [ref[b].index(r) for r in ref[a]]
        outcome = (
            "undecidable"
            if perm is None
            else "reference_unknown"
            if reference is None
            else "correct"
            if list(perm) == reference
            else "wrong"
        )
        record = {
            "method": method,
            "window": window,
            "pre_interval": a,
            "post_interval": b,
            "boundary_s": float(lower.set_index("segment_id").end_s[a]),
            "permutation": None if perm is None else [int(x) for x in perm],
            "reference_permutation": reference,
            "margin": margin,
            "outcome": outcome,
            "chain_correct": None if reference is None else reference == [0, 1, 2],
        }
        for name_index, name in enumerate(("F", "M1", "M2")):
            if reference is None or perm is None:
                record[f"edge_{name}"] = None
            else:
                s = ref[a].index(name_index)
                record[f"edge_{name}"] = bool(perm[s] == reference[s])
        rows.append(record)
    return pd.DataFrame(rows)


def names_sequential(table, lower, initial):
    names = {int(lower.segment_id.iloc[0]): np.array(initial)}
    for row in table.itertuples():
        perm = row.permutation if isinstance(row.permutation, list) else [0, 1, 2]
        nxt = np.empty(3, int)
        for s in range(3):
            nxt[perm[s]] = names[row.pre_interval][s]
        names[row.post_interval] = nxt
    return names


def names_global(lab, features, whole_rows, lower, initial):
    ids = lower.segment_id.tolist()
    descs = [[describe(lab, features, whole_rows, i, s) for s in range(3)] for i in ids]
    labels, rounds = lab.constrained_labels(descs, np.array(initial))
    names = {}
    for i, row in zip(ids, labels, strict=True):
        names[i] = row if row.min() >= 0 else np.array(initial)  # fall back to A0 naming
    return names, rounds, int(sum(row.min() < 0 for row in labels))


def identity_time(names, lower, matched, anchor, weights):
    from medaka3d.eval.flip_risk import paired_identity_time

    after = np.full_like(matched, -1)
    mask = np.zeros(len(weights), bool)
    for row in lower.itertuples():
        lo, hi = row.start_frame, row.stop_frame_exclusive
        mask[lo:hi] = True
        m = matched[lo:hi]
        mapped = np.where(m >= 0, np.asarray(anchor)[names[row.segment_id][np.maximum(m, 0)]], -1)
        after[lo:hi] = mapped
    w = np.where(mask, weights, 0.0)
    return paired_identity_time(matched, after, anchor, w)


def illumination(features, meta, whole_rows, ref, lower, xyz, bins):
    """Same- vs different-fish descriptor distance by 3D separation (diagnostic)."""
    rows = []
    for view, feats in features.items():
        frag = []
        for i in lower.segment_id:
            if ref[i] is None:
                continue
            for s in range(3):
                r = whole_rows.get((i, s, view))
                if r is None or r.size < 5:
                    continue
                pos = np.nanmean(xyz[meta[view]["frame"][r], s], axis=0)
                frag.append((i, ref[i][s], feats[r].mean(0), pos))
        if len(frag) < 2:
            continue
        a_i, a_r, a_f, a_p = zip(*frag, strict=True)
        a_i, a_r, a_f, a_p = map(np.asarray, (a_i, a_r, a_f, a_p))
        iu = np.triu_indices(len(frag), 1)
        other = a_i[iu[0]] != a_i[iu[1]]
        fd = np.linalg.norm(a_f[iu[0]] - a_f[iu[1]], axis=1)[other]
        sep = np.linalg.norm(a_p[iu[0]] - a_p[iu[1]], axis=1)[other]
        same = (a_r[iu[0]] == a_r[iu[1]])[other]
        males = (a_r[iu[0]] > 0)[other] & (a_r[iu[1]] > 0)[other]
        for lo, hi in pairwise(bins):
            inb = (sep >= lo) & (sep < hi)
            for subset, keep in (("all", inb), ("males_only", inb & males)):
                s_d, d_d = fd[keep & same], fd[keep & ~same]
                auc = np.mean(d_d[None, :] > s_d[:, None]) if s_d.size and d_d.size else np.nan
                rows.append(
                    {
                        "view": view,
                        "subset": subset,
                        "separation_lo_mm": lo,
                        "separation_hi_mm": hi,
                        "same_pairs": int(s_d.size),
                        "different_pairs": int(d_d.size),
                        "same_median": float(np.median(s_d)) if s_d.size else np.nan,
                        "different_median": float(np.median(d_d)) if d_d.size else np.nan,
                        "auc_different_farther": float(auc),
                    }
                )
    return pd.DataFrame(rows)


def outcome_counts(table):
    known = table[
        table.outcome.isin(["correct", "wrong", "undecidable"])
        & table.reference_permutation.notna()
    ]
    chain_wrong = known[known.chain_correct == False]  # noqa: E712
    return {
        "boundaries": len(table),
        "reference_known": len(known),
        "correct": int((known.outcome == "correct").sum()),
        "wrong": int((known.outcome == "wrong").sum()),
        "undecidable": int((known.outcome == "undecidable").sum()),
        "chain_wrong_boundaries": len(chain_wrong),
        "correct_on_chain_wrong": int((chain_wrong.outcome == "correct").sum()),
        "newly_wrong_on_chain_correct": int(
            ((known.chain_correct == True) & (known.outcome == "wrong")).sum()  # noqa: E712
        ),
        "male_edges_correct": int(
            sum(
                (known[f"edge_{m}"] == True).sum()  # noqa: E712
                for m in ("M1", "M2")
            )
        ),
        "male_edges_known": int(sum(known[f"edge_{m}"].notna().sum() for m in ("M1", "M2"))),
    }


def cmd_evaluate(cfg, root, out):
    lab = load_lab()
    track, warn = root / cfg["input"]["tracking_run"], root / cfg["input"]["warning_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    ev = cfg["evaluate"]
    dest = out / "evaluation"
    dest.mkdir(exist_ok=False)
    write_json(out / "status.json", {"status": "evaluate running"})
    with provenance("p11_appearance_evaluate", out) as rec:
        scoring = verified(
            rec,
            "scoring",
            track / cfg["input"]["scoring"],
            manifest[cfg["input"]["scoring"]]["sha256"],
        )
        lower = pd.read_csv(warn / cfg["input"]["lower_risk"])
        with np.load(scoring) as z:
            times, weights = z["recording_time_s"], z["duration_s"]
            matched, anchor = z["matched_slot"], z["anchor_reference_to_slot"]
            labels = [str(x) for x in z["reference_labels"]]
        with np.load(track / cfg["input"]["chain"]) as label:
            xyz = label["xyz"]
        meta, colour = {}, {}
        for view in VIEWS:
            with np.load(out / "crops" / f"{view}.npz") as z:
                meta[view] = {k: z[k] for k in ("frame", "slot", "interval", "overlap")}
                c = z["color"].astype(np.float64)
            colour[view] = (c - c.mean(0)) / c.std(0)
        features = {
            "A1_colour": colour,
            "A3_contrastive": {v: np.load(out / "embeddings" / f"full__{v}.npy") for v in VIEWS},
            "A3_early": {v: np.load(out / "embeddings" / f"early__{v}.npy") for v in VIEWS},
        }
        pre, post, whole = fragment_rows(meta, times, lower, ev["end_window_s"])
        ref = reference_slots(lower, labels, ev["min_reference_coverage"])
        initial = [int(np.flatnonzero(anchor == s)[0]) for s in range(3)]

        tables = [
            boundary_table(lab, features["A1_colour"], pre, post, lower, ref, "A0_chain", "end")
        ]
        for method, feats in features.items():
            tables.append(boundary_table(lab, feats, pre, post, lower, ref, method, "end"))
            tables.append(boundary_table(lab, feats, whole, whole, lower, ref, method, "whole"))
        boundaries = pd.concat(tables, ignore_index=True)
        boundaries.to_csv(dest / "boundaries.csv", index=False)

        summary = {"boundary_outcomes": {}, "identity_time": {}, "late_boundaries": {}}
        for (method, window), table in boundaries.groupby(["method", "window"]):
            summary["boundary_outcomes"][f"{method}|{window}"] = outcome_counts(table)
            late = table[table.boundary_s >= ev["late_boundary_start_s"] + ev["end_window_s"]]
            summary["late_boundaries"][f"{method}|{window}"] = outcome_counts(late)

        base = identity_time(
            {i: np.array(initial) for i in lower.segment_id}, lower, matched, anchor, weights
        )
        if not np.isclose(base["wrong_seconds"], lower.wrong_seconds.sum(), atol=1e-6):
            raise AssertionError("Chain naming does not reproduce the frozen lower-risk score")
        summary["identity_time"]["A0_chain"] = base
        for method, feats in features.items():
            table = boundaries[(boundaries.method == method) & (boundaries.window == "end")]
            summary["identity_time"][f"{method}|sequential"] = identity_time(
                names_sequential(table, lower, initial), lower, matched, anchor, weights
            )
            names, rounds, unassigned = names_global(lab, feats, whole, lower, initial)
            result = identity_time(names, lower, matched, anchor, weights)
            result.update({"rounds": rounds, "intervals_unassigned_use_chain": unassigned})
            summary["identity_time"][f"{method}|global"] = result
            pd.DataFrame(
                [
                    {"segment_id": i, **{f"slot{s}": int(n[s]) for s in range(3)}}
                    for i, n in names.items()
                ]
            ).to_csv(dest / f"global_names_{method}.csv", index=False)

        illum = pd.concat(
            [
                illumination(
                    features[m], meta, whole, ref, lower, xyz, ev["separation_bins_mm"]
                ).assign(method=m)
                for m in ("A1_colour", "A3_contrastive")
            ],
            ignore_index=True,
        )
        illum.to_csv(dest / "illumination_by_separation.csv", index=False)

        value_corr = []
        for view in VIEWS:
            m = meta[view]
            keep = ~m["overlap"]
            pos = xyz[m["frame"][keep], m["slot"][keep]]
            value = colour[view][keep, 5]
            for axis, name in enumerate("xyz"):
                ok = np.isfinite(pos[:, axis])
                value_corr.append(
                    {
                        "view": view,
                        "axis": name,
                        "spearman": float(
                            pd.Series(value[ok]).corr(pd.Series(pos[ok, axis]), method="spearman")
                        ),
                    }
                )
        pd.DataFrame(value_corr).to_csv(dest / "value_position_correlation.csv", index=False)

        a3 = summary["boundary_outcomes"]["A3_contrastive|end"]
        a0 = summary["boundary_outcomes"]["A0_chain|end"]
        g3 = summary["identity_time"]["A3_contrastive|global"]
        frac = (
            a3["correct_on_chain_wrong"] / a3["chain_wrong_boundaries"]
            if a3["chain_wrong_boundaries"]
            else np.nan
        )
        reduction = 1 - g3["wrong_seconds"] / base["wrong_seconds"]
        a_ok, a_reject = frac >= 0.5, frac < 0.25
        b_ok = reduction >= 0.5 and a3["wrong"] <= a0["wrong"]
        b_reject = reduction <= 0
        verdict = (
            "carry_forward"
            if a_ok and b_ok
            else "reject"
            if a_reject or b_reject
            else "inconclusive"
        )
        summary["decision_rule"] = {
            "a_fraction_correct_on_chain_wrong": frac,
            "b_global_wrong_reduction": reduction,
            "b_a3_wrong_boundaries": a3["wrong"],
            "b_a0_wrong_boundaries": a0["wrong"],
            "readout": verdict,
        }
        write_json(dest / "summary.json", summary)
        rec.output("summary", dest / "summary.json")
        rec.param("reference_use", "evaluation only; fixed 298-300 s anchor names interval 0")
    write_json(out / "status.json", {"status": "evaluated"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "train", "evaluate"))
    args = parser.parse_args(argv)
    cfg, _, root, out = settings()
    if args.stage != "extract" and not out.is_dir():
        raise SystemExit(f"Run directory does not exist: {out}")
    {"extract": cmd_extract, "train": cmd_train, "evaluate": cmd_evaluate}[args.stage](
        cfg, root, out
    )


if __name__ == "__main__":
    main()
