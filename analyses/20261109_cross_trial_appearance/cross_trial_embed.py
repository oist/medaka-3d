"""Cross-trial appearance test: frozen K_1 and P_1 per-trio encoders applied to K_2 and P_2.

For each source trio, the per-group encoders (top, side), their channel
normalisation and the per-identity reference centroids are read from the fixed
group runs of the six-recording study and never refitted. The reference
centroid of an identity in a camera view is the crop-weighted mean, over the
three source recordings, of the clean lower-risk embeddings that each
recording's N_carry naming gives that identity; the codes are aligned across
days through the study's stored review alignment and that alignment is checked
label-free (the best one-to-one centroid assignment between days must be the
identity). Each target recording's prepared clean crops are then embedded with
the frozen weights and:

1. its lower-risk intervals are named against the frozen reference centroids
   with the lab's own decision rules (``decide`` for three described
   fragments, ``assign_partial`` for one or two), per camera and with all
   cameras;
2. those names are compared with the target's own per-recording encoder names
   (``evaluation/interval_names.csv``) under the one permutation of the
   target's three codes that agrees most, overall, per camera and per time
   bin; disagreeing intervals are listed;
3. the target's label-free appearance clusters (its own codes, and k-means on
   the frozen embeddings) are matched to the reference centroids by best
   one-to-one assignment, with the margin to the second-best assignment, as
   the six-recording study did across days; the displacement of each matched
   cluster from its reference centroid is set against the smallest separation
   between reference identities and against the within-period day-to-day
   displacement of the same identity.

No reference identity exists for any target recording, so every agreement is
a consistency figure, not an accuracy. The reference identities carry the
K_1/P_1 reviewers' names (F, M1, M2); in the targets they are hypotheses
carried by appearance alone.

Usage: cross_trial_embed.py [--config config.toml] [--only K_1|P_1]
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d import load_paths
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P11_ANALYSIS = REPO / "analyses/20261031_p11_identity_evaluation"
sys.path.insert(0, str(P11_ANALYSIS))
import run_appearance_link as r1  # noqa: E402
import run_appearance_naming as rn  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")
MODEL_OF_VIEW = {"top_cam": "top", "side1_cam": "side", "side2_cam": "side"}
PERMS = [np.array(p, np.int8) for p in permutations(range(3))]


# --------------------------------------------------------------------------- helpers


def resolve(spec: str, paths, root: Path) -> Path:
    kind, _, rel = spec.partition(":")
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Path must be relative without '..': {spec!r}")
    if kind == "data":
        return root / rel_path
    if kind == "staging":
        if paths.appearance_staging_dir is None:
            raise ValueError("Set appearance_staging_dir in configs/paths.local.toml")
        return paths.appearance_staging_dir / rel_path
    raise ValueError(f"Unknown path kind in {spec!r}; use data: or staging:")


def best_assignment(a: np.ndarray, b: np.ndarray):
    """Best one-to-one assignment of rows of ``a`` (3, d) to rows of ``b`` (3, d).

    Returns (assignment a_index -> b_index, best cost, margin to the second
    best, matched distances). The cost is the sum of the three matched
    distances, as in the six-recording study's cross-day diagnostic.
    """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    costs = sorted(
        (float(sum(d[i, p[i]] for i in range(3))), tuple(int(x) for x in p)) for p in PERMS
    )
    best_cost, best = costs[0]
    return (
        np.array(best, np.int8),
        best_cost,
        float(costs[1][0] - best_cost),
        [float(d[i, best[i]]) for i in range(3)],
    )


def min_separation(c: np.ndarray) -> float:
    return float(min(np.linalg.norm(c[i] - c[j]) for i, j in combinations(range(3), 2)))


def centroid_cost(desc: dict, centroids: dict, k: int) -> float:
    """Mean over the descriptor's cameras of the distance to identity ``k``'s centroid."""
    terms = [
        np.linalg.norm(vec - centroids[v][k])
        for v, (vec, _) in desc.items()
        if v in centroids and centroids[v][k] is not None
    ]
    return float(np.mean(terms)) if terms else np.nan


def name_intervals(lab, descs, ids, centroids):
    """Frozen-centroid naming of intervals; returns names, how and margins (global only)."""
    names, how, margins = {}, {}, {}
    for i, row in zip(ids, descs, strict=True):
        if row is not None and all(row):
            cost = np.array(
                [[centroid_cost(row[s], centroids, k) for k in range(3)] for s in range(3)]
            )
            perm, margin = lab.decide(cost)
            if perm is not None:
                names[i], how[i], margins[i] = perm.astype(np.int8), "global", margin
                continue
        partial = lab.assign_partial(row, centroids) if row is not None and any(row) else None
        if partial is None:
            names[i], how[i] = np.full(3, -1, np.int8), "unknown"
        else:
            names[i] = partial.astype(np.int8)
            how[i] = "partial_two" if (partial >= 0).all() else "partial_one"
        margins[i] = np.nan
    return names, how, margins


def single_camera(descs, view):
    out = []
    for row in descs:
        if not row:
            out.append(None)
            continue
        out.append(
            [({view: row[s][view]} if row[s] and view in row[s] else None) for s in range(3)]
        )
    return out


def duration_weighted_permutation(frozen, own, duration, ids):
    """Maximise duration of complete three-slot agreement, weighted by three fish.

    An interval contributes only when all three names agree; an unaffected
    fish in a partially disagreeing interval is not counted separately.
    """
    usable = [
        i
        for i in ids
        if frozen.get(i) is not None
        and own.get(i) is not None
        and (frozen[i] >= 0).all()
        and (own[i] >= 0).all()
    ]
    total = float(sum(3 * duration[i] for i in usable))
    scores = []
    for p in PERMS:
        agree = sum(
            3 * duration[i] for i in usable if all(frozen[i][s] == p[own[i][s]] for s in range(3))
        )
        scores.append((float(agree), tuple(int(x) for x in p)))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return np.array(best[1], np.int8), best[0], second[0], total


def read_names_csv(path: Path):
    """interval_names.csv -> ({segment_id: names}, {segment_id: how})."""
    df = pd.read_csv(path)
    names = {
        int(r.segment_id): np.array([r.slot0, r.slot1, r.slot2], np.int8) for r in df.itertuples()
    }
    how = {int(r.segment_id): str(r.how) for r in df.itertuples()}
    return names, how


def sha_of(path: Path) -> str:
    return r1.sha256(path)


# --------------------------------------------------------------------------- reference


def load_models(lab, model_run: Path, device, rec, trio: str):
    import torch

    models = {}
    for name in ("top", "side"):
        path = model_run / "models" / f"{name}.pt"
        digest = sha_of(path)
        rec.input(f"{trio}_model_{name}", path, sha256=digest)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        model = lab.build_encoder(payload["out_dim"], payload["encoder"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        model.to(device)
        stats = (np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32))
        info = {
            "path": str(path),
            "sha256": digest,
            "views": payload["views"],
            "encoder": payload["encoder"],
            "out_dim": payload["out_dim"],
            "best_step": payload["best_step"],
            "best_silhouette": payload["best_silhouette"],
            "group": payload.get("group"),
            "recordings": payload.get("recordings"),
            "mean": [float(x) for x in stats[0]],
            "std": [float(x) for x in stats[1]],
        }
        models[name] = (model, stats, info)
    return models


def sanity_reembed(lab, models, model_run: Path, clean_run: Path, n=2000, seed=0):
    """Re-embed a sample of the model run's own crops; compare with its stored embeddings."""
    out = {}
    for view in VIEWS:
        path = clean_run / "clean" / f"{view}.npz"
        if not path.exists():
            out[view] = {"skipped": f"missing {path}"}
            continue
        stored = np.load(model_run / "embeddings" / f"{view}.npy")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(stored), min(n, len(stored)), replace=False))
        with np.load(path) as z:
            crops = z["crops"][idx]
        model, stats, _ = models[MODEL_OF_VIEW[view]]
        emb = lab.embed(model, crops, stats)
        diff = np.linalg.norm(emb - stored[idx], axis=1)
        out[view] = {
            "crops": int(idx.size),
            "max_distance": float(diff.max()),
            "median_distance": float(np.median(diff)),
            "stored_norm_median": float(np.median(np.linalg.norm(stored[idx], axis=1))),
        }
    return out


def member_centroids(member_run: Path, clean_run: Path, code_to_label: dict, rec, tag: str):
    """Per-view centroids in reference label order, spreads, counts and crop arrays."""
    track = member_run / "evaluation" / "identity_tracks_N_carry.npz"
    rec.input(f"{tag}_N_carry", track, sha256=sha_of(track))
    with np.load(track) as z:
        name = z["name"].astype(np.int8)
    cents, spreads, counts, crops = {}, {}, {}, {}
    for view in VIEWS:
        emb_path = member_run / "embeddings" / f"{view}.npy"
        rec.input(f"{tag}_embeddings_{view}", emb_path, sha256=sha_of(emb_path))
        clean_path = clean_run / "clean" / f"{view}.npz"
        # SHA-256 as recorded by the run that prepared the crops; not re-hashed over the share.
        rec.input(
            f"{tag}_clean_{view}",
            clean_path,
            sha256=rn.recorded_output(clean_run, "appearance_clean_prepare", f"clean_{view}"),
        )
        with np.load(clean_path) as z:
            frame, slot, clean = z["frame"].astype(np.int64), z["slot"].astype(np.int64), z["clean"]
        emb = np.load(emb_path)
        code = name[frame, slot]
        label = np.full(code.shape, -1, np.int8)
        for c, lab_idx in code_to_label.items():
            label[code == int(c)] = int(lab_idx)
        sel = clean & (label >= 0)
        c3 = np.stack([emb[sel & (label == k)].mean(axis=0) for k in range(3)])
        cents[view] = c3
        spreads[view] = [
            float(np.median(np.linalg.norm(emb[sel & (label == k)] - c3[k], axis=1)))
            for k in range(3)
        ]
        counts[view] = [int((sel & (label == k)).sum()) for k in range(3)]
        crops[view] = (emb[sel], label[sel])
    return cents, spreads, counts, crops


def build_reference(lab, cfg, src, paths, root, labels, scores, device, rec, out_dir, trio):
    model_run = resolve(src["model_run"], paths, root)
    models = load_models(lab, model_run, device, rec, trio)
    sanity = {"skipped": "no sanity_clean_run"}
    if "sanity_clean_run" in src:
        sanity = sanity_reembed(
            lab, models, model_run, resolve(src["sanity_clean_run"], paths, root)
        )
    per_day, spreads, counts, crop_arrays, alignment = {}, {}, {}, {}, {}
    for member in src["members"]:
        name = member["recording"]
        code_to_label = scores[name]["candidates"]["N_carry"]["alignment"]["code_to_label"]
        alignment[name] = {str(k): int(v) for k, v in code_to_label.items()}
        per_day[name], spreads[name], counts[name], crop_arrays[name] = member_centroids(
            resolve(member["run"], paths, root),
            resolve(member["clean_run"], paths, root),
            code_to_label,
            rec,
            f"{trio}_{name}",
        )
    days = list(per_day)
    # Label-free check: between any two days the best assignment must be the identity.
    cross_day = {}
    for a, b in combinations(days, 2):
        for view in VIEWS:
            assign, cost, margin, matched = best_assignment(per_day[a][view], per_day[b][view])
            cross_day[f"{a} -> {b} | {view}"] = {
                "assignment_is_identity": bool((assign == np.arange(3)).all()),
                "best_cost": cost,
                "margin": margin,
                "matched_distances": matched,
            }
    if not all(e["assignment_is_identity"] for e in cross_day.values()):
        raise AssertionError(f"{trio}: review alignment and label-free centroid matching disagree")
    pooled, within_disp, separation, within_spread = {}, {}, {}, {}
    for view in VIEWS:
        w = np.array([counts[d][view] for d in days], float)  # (days, 3)
        c = np.stack([per_day[d][view] for d in days])  # (days, 3, d)
        pooled[view] = (c * w[:, :, None]).sum(axis=0) / w.sum(axis=0)[:, None]
        within_disp[view] = [
            float(
                max(
                    np.linalg.norm(per_day[a][view][k] - per_day[b][view][k])
                    for a, b in combinations(days, 2)
                )
            )
            for k in range(3)
        ]
        separation[view] = min_separation(pooled[view])
        within_spread[view] = [
            float(np.median([spreads[d][view][k] for d in days])) for k in range(3)
        ]
    # Within-period per-crop agreement with the pooled centroids: the baseline for the targets.
    within_crop = {}
    for d in days:
        within_crop[d] = {}
        for view in VIEWS:
            emb, label = crop_arrays[d][view]
            nearest = np.linalg.norm(emb[:, None, :] - pooled[view][None], axis=2).argmin(axis=1)
            within_crop[d][view] = {
                "crops": int(label.size),
                "nearest_pooled_centroid_agreement": float((nearest == label).mean()),
                "displacement_from_pooled": [
                    float(np.linalg.norm(per_day[d][view][k] - pooled[view][k])) for k in range(3)
                ],
            }
    ref_dir = out_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    npz = ref_dir / f"{trio}.npz"
    np.savez(
        npz,
        labels=np.array(labels),
        days=np.array(days),
        **{f"pooled_{v}": pooled[v] for v in VIEWS},
        **{f"per_day_{v}": np.stack([per_day[d][v] for d in days]) for v in VIEWS},
        **{f"counts_{v}": np.array([counts[d][v] for d in days]) for v in VIEWS},
    )
    info = {
        "trio": trio,
        "model_run": str(model_run),
        "models": {k: v[2] for k, v in models.items()},
        "sanity_reembed_of_model_run": sanity,
        "members": days,
        "review_alignment_code_to_label": alignment,
        "label_free_cross_day_check": cross_day,
        "crop_counts": counts,
        "within_day_spread_median": spreads,
        "within_period_max_displacement": within_disp,
        "pooled_min_separation": separation,
        "within_period_crops_vs_pooled": within_crop,
    }
    r1.write_json(ref_dir / f"{trio}.json", info)
    for p in (npz, ref_dir / f"{trio}.json"):
        rec.output(f"reference_{trio}_{p.suffix[1:]}", p, sha256=sha_of(p))
    reference = {
        "pooled": pooled,
        "per_day": per_day,
        "days": days,
        "separation": separation,
        "within_disp": within_disp,
        "within_spread": within_spread,
        "within_crop": within_crop,
        "info": info,
    }
    return models, reference


# --------------------------------------------------------------------------- targets


def verify_target_clean(run: Path, shared: Path | None, view: str, rec, tag: str) -> Path:
    path = run / "clean" / f"{view}.npz"
    digest = sha_of(path)
    recorded = rn.recorded_output(run, "appearance_clean_prepare", f"clean_{view}")
    if digest != recorded:
        raise ValueError(f"{path}: SHA-256 differs from the run's prepare provenance")
    if shared is not None:
        manifest = json.loads((shared / "staged_copy_manifest.json").read_text(encoding="utf-8"))
        files = {k.replace("\\", "/"): v for k, v in manifest["files"].items()}
        entry = files.get(f"clean/{view}.npz")
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if not expected:
            raise ValueError(f"{path}: missing SHA-256 in the shared copy's manifest")
        if expected != digest:
            raise ValueError(f"{path}: SHA-256 differs from the shared copy's manifest")
    rec.input(f"{tag}_clean_{view}", path, sha256=digest)
    return path


def per_day_matches(c_own, reference, view, labels):
    out = {}
    for d in reference["days"]:
        assign, cost, margin, matched = best_assignment(c_own, reference["per_day"][d][view])
        out[d] = {
            "assignment": [labels[a] for a in assign],
            "best_cost": cost,
            "margin": margin,
            "matched_distances": matched,
        }
    return out


def process_target(
    lab, cfg, name, tcfg, models, reference, labels, paths, root, device, rec, out_dir
):
    run = resolve(tcfg["run"], paths, root)
    shared = resolve(tcfg["shared"], paths, root) if "shared" in tcfg else None
    tag = f"target_{name}"
    lower_path = run / "bundle" / "lower_risk_intervals.csv"
    timing_path = run / "bundle" / "timing.npz"
    rec.input(f"{tag}_lower_risk", lower_path, sha256=sha_of(lower_path))
    rec.input(f"{tag}_timing", timing_path, sha256=sha_of(timing_path))
    lower = pd.read_csv(lower_path)
    with np.load(timing_path) as z:
        times = z["times"]
    duration = dict(zip(lower.segment_id.astype(int), lower.duration_s.astype(float), strict=True))
    start_s = dict(zip(lower.segment_id.astype(int), lower.start_s.astype(float), strict=True))
    end_s = dict(zip(lower.segment_id.astype(int), lower.end_s.astype(float), strict=True))
    ids = [int(i) for i in lower.segment_id]
    pooled = reference["pooled"]
    centroids = {v: [pooled[v][k] for k in range(3)] for v in VIEWS}

    emb_dir = out_dir / "embeddings" / name
    emb_dir.mkdir(parents=True, exist_ok=True)
    meta, feats, crop_counts = {}, {}, {}
    for view in VIEWS:
        path = verify_target_clean(run, shared, view, rec, tag)
        with np.load(path) as z:
            crops = z["crops"]
            meta[view] = {
                "frame": z["frame"].astype(np.int64),
                "slot": z["slot"].astype(np.int64),
                "interval": z["interval"].astype(np.int64),
                "clean": z["clean"],
            }
        model, stats, _ = models[MODEL_OF_VIEW[view]]
        emb = lab.embed(model, crops, stats)
        del crops
        np.save(emb_dir / f"{view}.npy", emb)
        rec.output(
            f"{tag}_embeddings_{view}",
            emb_dir / f"{view}.npy",
            sha256=sha_of(emb_dir / f"{view}.npy"),
        )
        feats[view] = emb
        meta[view]["overlap"] = ~meta[view]["clean"]
        crop_counts[view] = {"rows": int(emb.shape[0]), "clean": int(meta[view]["clean"].sum())}
        print(f"  {name} {view}: embedded {emb.shape[0]} crops", flush=True)

    # 1. Frozen naming, all cameras and per camera.
    rows = r1.fragment_rows(meta, times, lower, cfg["end_window_s"])
    whole = rows[2]
    descs = [[r1.describe(lab, feats, whole, i, s) for s in range(3)] for i in ids]
    frozen, how, margins = name_intervals(lab, descs, ids, centroids)
    per_view_names = {}
    for view in VIEWS:
        per_view_names[view] = name_intervals(
            lab, single_camera(descs, view), ids, {view: centroids[view]}
        )

    # 2. Own per-recording encoder names.
    own, own_how, own_available = {}, {}, False
    own_path = run / "evaluation" / "interval_names.csv"
    if own_path.exists():
        rec.input(f"{tag}_own_interval_names", own_path, sha256=sha_of(own_path))
        own, own_how = read_names_csv(own_path)
        own_available = sum(h == "global" for h in own_how.values()) >= 2
    pi, agree_s, second_s, both_s = None, np.nan, np.nan, np.nan
    if own_available:
        pi, agree_s, second_s, both_s = duration_weighted_permutation(frozen, own, duration, ids)

    def own_mapped(i):
        o = own.get(i)
        if pi is None or o is None:
            return None
        return np.array([pi[c] if c >= 0 else -1 for c in o], np.int8)

    def lab_or_blank(x):
        return labels[x] if x >= 0 else ""

    bin_w, t0 = cfg["time_bin_s"], cfg["analysis_start_s"]
    records = []
    for i, row in zip(ids, descs, strict=True):
        om = own_mapped(i)
        # Fully named by both procedures (global, or partial_two by elimination).
        both = om is not None and (om >= 0).all() and (frozen[i] >= 0).all()
        entry = {
            "segment_id": i,
            "start_s": start_s[i],
            "end_s": end_s[i],
            "duration_s": duration[i],
            "time_bin": int((start_s[i] - t0) // bin_w),
            "described_slots": sum(1 for s in range(3) if row[s]) if row else 0,
            "frozen_how": how[i],
            "frozen_margin": margins[i],
            **{f"frozen_slot{s}": lab_or_blank(frozen[i][s]) for s in range(3)},
            "own_how": own_how.get(i, ""),
            **{f"own_slot{s}": (lab_or_blank(om[s]) if om is not None else "") for s in range(3)},
            "both_named": bool(both),
            "both_global": bool(both and how[i] == "global" and own_how.get(i) == "global"),
            "agree": bool((frozen[i] == om).all()) if both else "",
        }
        for view in VIEWS:
            n_v, h_v, m_v = per_view_names[view]
            entry[f"{view}_how"] = h_v[i]
            entry[f"{view}_margin"] = m_v[i]
            entry[f"{view}_names"] = "/".join(labels[x] if x >= 0 else "-" for x in n_v[i])
            entry[f"{view}_agrees_all_cameras"] = (
                bool((n_v[i] == frozen[i]).all())
                if h_v[i] == "global" and how[i] == "global"
                else ""
            )
            entry[f"{view}_agrees_own"] = (
                bool((n_v[i] == om).all())
                if h_v[i] == "global" and om is not None and (om >= 0).all()
                else ""
            )
        records.append(entry)
    table = pd.DataFrame(records)
    report_dir = out_dir / "report"
    report_dir.mkdir(exist_ok=True)
    table.to_csv(report_dir / f"{name}_intervals.csv", index=False)
    bins = sorted(int(b) for b in table.time_bin.unique())

    # 3. Per-crop nearest reference centroid.
    crop_level, crop_bins = {}, []
    frozen_ident, own_ident = {}, {}
    for view in VIEWS:
        m = meta[view]
        ident = np.full(m["frame"].size, -1, np.int8)
        o_ident = np.full(m["frame"].size, -1, np.int8)
        for i in ids:
            sel = m["interval"] == i
            if how[i] == "global":
                ident[sel] = frozen[i][m["slot"][sel]]
            om = own_mapped(i)
            if om is not None and (om >= 0).all():
                o_ident[sel] = om[m["slot"][sel]]
        frozen_ident[view], own_ident[view] = ident, o_ident
        pick = m["clean"] & (ident >= 0)
        pick_own = m["clean"] & (o_ident >= 0)
        nearest = np.linalg.norm(feats[view][:, None, :] - pooled[view][None], axis=2).argmin(
            axis=1
        )
        per_identity = {}
        for k, lab_name in enumerate(labels):
            pk = pick & (ident == k)
            disp = (
                float(np.linalg.norm(feats[view][pk].mean(axis=0) - pooled[view][k]))
                if pk.sum() >= lab.MIN_CROPS
                else np.nan
            )
            per_identity[lab_name] = {
                "crops": int(pk.sum()),
                "nearest_pooled_centroid_agreement": (
                    float((nearest[pk] == k).mean()) if pk.any() else np.nan
                ),
                "displacement_from_pooled": disp,
                "displacement_over_separation": disp / reference["separation"][view],
                "within_period_max_displacement": reference["within_disp"][view][k],
                "within_period_spread_median": reference["within_spread"][view][k],
            }
        crop_level[view] = {
            "crops_in_frozen_global_intervals": int(pick.sum()),
            "nearest_pooled_centroid_agreement_with_frozen_name": (
                float((nearest[pick] == ident[pick]).mean()) if pick.any() else np.nan
            ),
            "crops_with_own_name": int(pick_own.sum()),
            "nearest_pooled_centroid_agreement_with_own_name": (
                float((nearest[pick_own] == o_ident[pick_own]).mean()) if pick_own.any() else np.nan
            ),
            "per_identity_frozen_names": per_identity,
        }
        t = times[m["frame"]]
        for b in bins:
            in_bin = (t >= t0 + b * bin_w) & (t < t0 + (b + 1) * bin_w)
            pb, po = pick & in_bin, pick_own & in_bin
            row = {
                "time_bin": b,
                "bin_start_s": t0 + b * bin_w,
                "view": view,
                "crops_frozen": int(pb.sum()),
                "nearest_agreement_frozen": (
                    float((nearest[pb] == ident[pb]).mean()) if pb.any() else np.nan
                ),
                "crops_own": int(po.sum()),
                "nearest_agreement_own": (
                    float((nearest[po] == o_ident[po]).mean()) if po.any() else np.nan
                ),
            }
            for k, lab_name in enumerate(labels):
                pk = pb & (ident == k)
                row[f"crops_{lab_name}"] = int(pk.sum())
                row[f"agree_{lab_name}"] = float((nearest[pk] == k).mean()) if pk.any() else np.nan
                row[f"disp_{lab_name}"] = (
                    float(np.linalg.norm(feats[view][pk].mean(axis=0) - pooled[view][k]))
                    if pk.sum() >= lab.MIN_CROPS
                    else np.nan
                )
            crop_bins.append(row)

    # 4. Cluster-level matching: own codes (when available) and k-means per view.
    own_code_per_view = {}
    own_track = run / "evaluation" / "identity_tracks_N_carry.npz"
    if own_available and own_track.exists():
        rec.input(f"{tag}_own_N_carry", own_track, sha256=sha_of(own_track))
        with np.load(own_track) as z:
            own_name = z["name"].astype(np.int8)
        for view in VIEWS:
            own_code_per_view[view] = own_name[meta[view]["frame"], meta[view]["slot"]]
    rng = np.random.default_rng(cfg["kmeans_seed"])
    matching, silhouettes = {}, {}
    for view in VIEWS:
        m = meta[view]
        entry = {}
        if view in own_code_per_view:
            code = own_code_per_view[view]
            sel = m["clean"] & (code >= 0)
            c_own = np.stack([feats[view][sel & (code == k)].mean(axis=0) for k in range(3)])
            assign, cost, margin, matched = best_assignment(c_own, pooled[view])
            t = times[m["frame"]]
            per_bin = []
            for b in bins:
                in_bin = sel & (t >= t0 + b * bin_w) & (t < t0 + (b + 1) * bin_w)
                if all((in_bin & (code == k)).sum() >= lab.MIN_CROPS for k in range(3)):
                    cb = np.stack(
                        [feats[view][in_bin & (code == k)].mean(axis=0) for k in range(3)]
                    )
                    a_b, cost_b, margin_b, _ = best_assignment(cb, pooled[view])
                    per_bin.append(
                        {
                            "time_bin": b,
                            "crops": int(in_bin.sum()),
                            "assignment": [labels[a] for a in a_b],
                            "equals_global": bool((a_b == assign).all()),
                            "best_cost": cost_b,
                            "margin": margin_b,
                        }
                    )
                else:
                    per_bin.append({"time_bin": b, "crops": int(in_bin.sum()), "assignment": None})
            entry["own_codes"] = {
                "crops": int(sel.sum()),
                "assignment_code_to_label": [labels[a] for a in assign],
                "equals_interval_permutation": (
                    bool((assign == pi).all()) if pi is not None else None
                ),
                "best_cost": cost,
                "margin": margin,
                "matched_distances": matched,
                "within_target_min_separation": min_separation(c_own),
                "within_target_spread_median": [
                    float(
                        np.median(np.linalg.norm(feats[view][sel & (code == k)] - c_own[k], axis=1))
                    )
                    for k in range(3)
                ],
                "per_day": per_day_matches(c_own, reference, view, labels),
                "per_bin": per_bin,
            }
        sel_all = m["clean"]
        km_labels, _ = lab.kmeans(feats[view][sel_all], k=3, seed=cfg["kmeans_seed"])
        c_km = np.stack([feats[view][sel_all][km_labels == k].mean(axis=0) for k in range(3)])
        assign, cost, margin, matched = best_assignment(c_km, pooled[view])
        km_full = np.full(m["frame"].size, -1, np.int8)
        km_full[np.flatnonzero(sel_all)] = km_labels
        t_all = times[m["frame"]]
        km_bins = []
        for b in bins:
            in_bin = sel_all & (t_all >= t0 + b * bin_w) & (t_all < t0 + (b + 1) * bin_w)
            if all((in_bin & (km_full == k)).sum() >= lab.MIN_CROPS for k in range(3)):
                cb = np.stack([feats[view][in_bin & (km_full == k)].mean(axis=0) for k in range(3)])
                a_b, cost_b, margin_b, _ = best_assignment(cb, pooled[view])
                km_bins.append(
                    {
                        "time_bin": b,
                        "crops": int(in_bin.sum()),
                        "assignment": [labels[a] for a in a_b],
                        "equals_global": bool((a_b == assign).all()),
                        "best_cost": cost_b,
                        "margin": margin_b,
                    }
                )
            else:
                km_bins.append({"time_bin": b, "crops": int(in_bin.sum()), "assignment": None})
        entry["kmeans"] = {
            "crops": int(sel_all.sum()),
            "cluster_sizes": [int((km_labels == k).sum()) for k in range(3)],
            "assignment_cluster_to_label": [labels[a] for a in assign],
            "best_cost": cost,
            "margin": margin,
            "matched_distances": matched,
            "within_target_min_separation": min_separation(c_km),
            "per_bin": km_bins,
        }
        idx_all = np.flatnonzero(sel_all)
        sample = np.sort(
            rng.choice(idx_all, min(cfg["silhouette_sample"], idx_all.size), replace=False)
        )
        z_s = feats[view][sample]
        km_s, _ = lab.kmeans(z_s, k=3, seed=cfg["kmeans_seed"])
        sil = {"sample": int(sample.size), "kmeans": lab.silhouette(z_s, km_s)}
        if view in own_code_per_view:
            lab_s = own_code_per_view[view][sample]
            keep = lab_s >= 0
            sil["own_codes"] = lab.silhouette(z_s[keep], lab_s[keep]) if keep.sum() > 3 else np.nan
        f_s = frozen_ident[view][sample]
        keep = f_s >= 0
        sil["frozen_names"] = lab.silhouette(z_s[keep], f_s[keep]) if keep.sum() > 3 else np.nan
        silhouettes[view] = sil
        matching[view] = entry

    # 5. Per-bin interval-level agreement and per-camera agreement.
    def count_flags(col):
        vals = col[col != ""]
        return int((vals == True).sum()), int((vals == False).sum())  # noqa: E712

    bin_rows = []
    for b in bins:
        tb = table[table.time_bin == b]
        both = tb[tb.both_named == True]  # noqa: E712
        agree_rows = both[both.agree == True]  # noqa: E712
        disagree_rows = both[both.agree == False]  # noqa: E712
        has_margin = tb.frozen_margin.notna().any()
        row = {
            "time_bin": b,
            "bin_start_s": t0 + b * bin_w,
            "intervals": len(tb),
            "frozen_global": int((tb.frozen_how == "global").sum()),
            "frozen_unknown": int((tb.frozen_how == "unknown").sum()),
            "frozen_margin_median": float(tb.frozen_margin.median()) if has_margin else np.nan,
            "frozen_margin_min": float(tb.frozen_margin.min()) if has_margin else np.nan,
            "both_named_intervals": len(both),
            "both_named_fish_s": float(3 * both.duration_s.sum()),
            "agree_fish_s": float(3 * agree_rows.duration_s.sum()),
            "disagree_intervals": len(disagree_rows),
            "disagree_fish_s": float(3 * disagree_rows.duration_s.sum()),
        }
        for view in VIEWS:
            row[f"{view}_global"] = int((tb[f"{view}_how"] == "global").sum())
            a, d = count_flags(tb[f"{view}_agrees_all_cameras"])
            row[f"{view}_agrees_all_cameras"], row[f"{view}_disagrees_all_cameras"] = a, d
            a, d = count_flags(tb[f"{view}_agrees_own"])
            row[f"{view}_agrees_own"], row[f"{view}_disagrees_own"] = a, d
        bin_rows.append(row)
    pd.DataFrame(bin_rows).to_csv(report_dir / f"{name}_bins.csv", index=False)
    pd.DataFrame(crop_bins).to_csv(report_dir / f"{name}_crop_bins.csv", index=False)
    for suffix in ("intervals", "bins", "crop_bins"):
        p = report_dir / f"{name}_{suffix}.csv"
        rec.output(f"{tag}_{suffix}", p, sha256=sha_of(p))

    both_tbl = table[table.both_named == True]  # noqa: E712
    disagree_tbl = both_tbl[both_tbl.agree == False]  # noqa: E712
    cam_agree = {}
    for view in VIEWS:
        col = table[f"{view}_agrees_all_cameras"]
        vals = col[col != ""]
        has_margin = table[f"{view}_margin"].notna().any()
        cam_agree[view] = {
            "global_intervals": int((table[f"{view}_how"] == "global").sum()),
            "compared_with_all_cameras": len(vals),
            "agree": int((vals == True).sum()),  # noqa: E712
            "disagree": int((vals == False).sum()),  # noqa: E712
            "disagree_fish_s": float(
                3 * table.loc[vals[vals == False].index, "duration_s"].sum()  # noqa: E712
            ),
            "margin_median": float(table[f"{view}_margin"].median()) if has_margin else np.nan,
        }
    displacement = {}
    for view in VIEWS:
        if "own_codes" not in matching[view]:
            continue
        e = matching[view]["own_codes"]
        disp = {}
        for code, lab_name in enumerate(e["assignment_code_to_label"]):
            k = labels.index(lab_name)
            disp[lab_name] = {
                "cross_period_displacement": e["matched_distances"][code],
                "within_period_max_displacement": reference["within_disp"][view][k],
                "within_period_spread_median": reference["within_spread"][view][k],
                "reference_min_separation": reference["separation"][view],
                "displacement_over_separation": (
                    e["matched_distances"][code] / reference["separation"][view]
                ),
            }
        displacement[view] = disp
    global_margins = [margins[i] for i in ids if how[i] == "global"]
    summary = {
        "recording": name,
        "run": str(run),
        "crops": crop_counts,
        "lower_risk_intervals": len(ids),
        "frozen_naming": {
            k: int(sum(h == k for h in how.values()))
            for k in ("global", "partial_two", "partial_one", "unknown")
        },
        "frozen_named_fish_s": float(
            sum(3 * duration[i] for i in ids if how[i] in ("global", "partial_two"))
        ),
        "frozen_margin_median_global": (
            float(np.nanmedian(global_margins)) if global_margins else np.nan
        ),
        "frozen_margin_min_global": float(np.nanmin(global_margins)) if global_margins else np.nan,
        "own_encoder_available": own_available,
        "own_naming": (
            {
                k: int(sum(h == k for h in own_how.values()))
                for k in ("global", "partial_two", "partial_one", "unknown")
            }
            if own_how
            else None
        ),
        "own_to_reference_permutation": [labels[x] for x in pi] if pi is not None else None,
        "both_named": {
            "intervals": len(both_tbl),
            "global_in_both": int(both_tbl.both_global.sum()),
            "fish_s": both_s,
            "agree_fish_s": agree_s,
            "agree_fraction": (agree_s / both_s) if both_s else np.nan,
            "second_best_permutation_fish_s": second_s,
            "disagree_intervals": len(disagree_tbl),
            "disagree_fish_s": float(3 * disagree_tbl.duration_s.sum()),
            "disagree_global_intervals": int(disagree_tbl.both_global.sum()),
            "disagreeing_intervals": [
                {
                    "segment_id": int(r.segment_id),
                    "start_s": r.start_s,
                    "duration_s": r.duration_s,
                    "frozen_how": r.frozen_how,
                    "own_how": r.own_how,
                    "frozen_margin": r.frozen_margin,
                    "frozen": [r.frozen_slot0, r.frozen_slot1, r.frozen_slot2],
                    "own": [r.own_slot0, r.own_slot1, r.own_slot2],
                }
                for r in disagree_tbl.itertuples()
            ],
        },
        "per_camera_vs_all_cameras": cam_agree,
        "crop_level": crop_level,
        "cluster_matching": matching,
        "displacement": displacement,
        "silhouette": silhouettes,
        "bins": bin_rows,
    }
    return summary


# --------------------------------------------------------------------------- report


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(out_dir, cfg, results):
    lines = [f"# Cross-trial appearance test: {cfg['run_id']}", ""]
    lines.append(
        "Frozen per-trio encoders, normalisation and reference centroids of the source trio "
        "applied to the target recordings. All agreements are consistency figures between two "
        "label-free procedures; no target recording has a reviewed identity."
    )
    for trio, res in results.items():
        ref = res["reference"]["info"]
        lines += ["", f"## {trio} -> targets {', '.join(res['targets'])}", ""]
        top, side = ref["models"]["top"], ref["models"]["side"]
        lines.append(
            f"Model run `{ref['model_run']}`; encoders top step {top['best_step']} "
            f"(silhouette {fmt(top['best_silhouette'])}), side step {side['best_step']} "
            f"(silhouette {fmt(side['best_silhouette'])})."
        )
        san = ref["sanity_reembed_of_model_run"]
        if isinstance(san, dict) and all(
            isinstance(v, dict) and "max_distance" in v for v in san.values()
        ):
            lines.append(
                "Re-embedding check against the model run's stored embeddings: max distance "
                + ", ".join(f"{v} {fmt(e['max_distance'], 5)}" for v, e in san.items())
                + "."
            )
        lines += [
            "",
            "### Reference (within-period, label-free, from the three source days)",
            "",
            "| View | Pooled min separation | Within-period max displacement F / M1 / M2 | "
            "Within-day spread (median) F / M1 / M2 | Cross-day best cost (range) | "
            "Cross-day margin (range) |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
        for view in VIEWS:
            cd = [e for k, e in ref["label_free_cross_day_check"].items() if k.endswith(view)]
            costs = [e["best_cost"] for e in cd]
            margins = [e["margin"] for e in cd]
            spread = [
                np.median([ref["within_day_spread_median"][d][view][k] for d in ref["members"]])
                for k in range(3)
            ]
            lines.append(
                f"| {view} | {fmt(ref['pooled_min_separation'][view])} | "
                + " / ".join(fmt(x) for x in ref["within_period_max_displacement"][view])
                + " | "
                + " / ".join(fmt(float(x)) for x in spread)
                + f" | {fmt(min(costs))}--{fmt(max(costs))}"
                + f" | {fmt(min(margins))}--{fmt(max(margins))} |"
            )
        lines += [
            "",
            "Within-period per-crop agreement with the pooled centroids (clean lower-risk crops "
            "under each day's names):",
            "",
            "| Source day | top | side1 | side2 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for d in ref["members"]:
            wc = ref["within_period_crops_vs_pooled"][d]
            lines.append(
                f"| {d} | "
                + " | ".join(fmt(wc[v]["nearest_pooled_centroid_agreement"]) for v in VIEWS)
                + " |"
            )
        lines += [
            "",
            "### Frozen naming of the targets' lower-risk intervals",
            "",
            "| Target | Intervals | global / partial_two / partial_one / unknown | Named fish-s | "
            "Margin median (min) | Own encoder | Permutation own -> ref | Named by both fish-s | "
            "Agree % | Disagree intervals (fish-s) | of which global in both |",
            "| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
        for name in res["targets"]:
            s = res["summaries"][name]
            fn, bg = s["frozen_naming"], s["both_named"]
            perm = s["own_to_reference_permutation"]
            lines.append(
                f"| {name} | {s['lower_risk_intervals']} | {fn['global']} / {fn['partial_two']} / "
                f"{fn['partial_one']} / {fn['unknown']} | {fmt(s['frozen_named_fish_s'], 1)} | "
                f"{fmt(s['frozen_margin_median_global'])} ({fmt(s['frozen_margin_min_global'])}) | "
                f"{'yes' if s['own_encoder_available'] else 'no'} | "
                f"{'/'.join(perm) if perm else 'n/a'} | {fmt(bg['fish_s'], 1)} | "
                f"{fmt(100 * bg['agree_fraction'], 2) if bg['fish_s'] else 'n/a'} | "
                f"{bg['disagree_intervals']} ({fmt(bg['disagree_fish_s'], 1)}) | "
                f"{bg['disagree_global_intervals']} |"
            )
        lines += [
            "",
            "### Per-identity displacement and per-crop agreement (crops in intervals the frozen "
            "rule names globally)",
            "",
            "| Target | View | F: agree, displacement, disp/sep, within-period max | "
            "M1: agree, displacement, disp/sep, within-period max | "
            "M2: agree, displacement, disp/sep, within-period max |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in res["targets"]:
            s = res["summaries"][name]
            for view in VIEWS:
                pi_ = s["crop_level"][view]["per_identity_frozen_names"]
                cells = [
                    f"{fmt(e['nearest_pooled_centroid_agreement'])}, "
                    f"{fmt(e['displacement_from_pooled'])}, "
                    f"{fmt(e['displacement_over_separation'], 2)}, "
                    f"{fmt(e['within_period_max_displacement'])}"
                    for e in pi_.values()
                ]
                lines.append(f"| {name} | {view} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "### Per-camera naming against the all-camera naming (global intervals in both)",
            "",
            "| Target | top agree / disagree | side1 agree / disagree | side2 agree / disagree | "
            "Disagree fish-s top / side1 / side2 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in res["targets"]:
            ca = res["summaries"][name]["per_camera_vs_all_cameras"]
            lines.append(
                f"| {name} | "
                + " | ".join(f"{ca[v]['agree']} / {ca[v]['disagree']}" for v in VIEWS)
                + " | "
                + " / ".join(fmt(ca[v]["disagree_fish_s"], 1) for v in VIEWS)
                + " |"
            )
        lines += [
            "",
            "### Cluster matching to the reference centroids (label-free clusters of the target "
            "under the frozen model)",
            "",
            "| Target | View | Own-code clusters: assignment, cost, margin, within-target "
            "separation | k-means clusters: assignment, cost, margin, within-target separation | "
            "Displacement / separation per identity (own codes) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in res["targets"]:
            s = res["summaries"][name]
            for view in VIEWS:
                m = s["cluster_matching"][view]
                own_txt = "n/a"
                if "own_codes" in m:
                    o = m["own_codes"]
                    own_txt = (
                        f"{'/'.join(o['assignment_code_to_label'])}, {fmt(o['best_cost'])}, "
                        f"{fmt(o['margin'])}, {fmt(o['within_target_min_separation'])}"
                    )
                k = m["kmeans"]
                km_txt = (
                    f"{'/'.join(k['assignment_cluster_to_label'])}, {fmt(k['best_cost'])}, "
                    f"{fmt(k['margin'])}, {fmt(k['within_target_min_separation'])}"
                )
                disp = s["displacement"].get(view)
                disp_txt = (
                    ", ".join(
                        f"{L} {fmt(d['displacement_over_separation'], 2)}" for L, d in disp.items()
                    )
                    if disp
                    else "n/a"
                )
                lines.append(f"| {name} | {view} | {own_txt} | {km_txt} | {disp_txt} |")
        lines += [
            "",
            "### Per-crop nearest reference centroid and silhouettes under the frozen model",
            "",
            "| Target | View | Crops (frozen global) | Nearest = frozen name | Crops (own name) | "
            "Nearest = own name | Silhouette k-means / own codes / frozen names |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name in res["targets"]:
            s = res["summaries"][name]
            for view in VIEWS:
                c, sil = s["crop_level"][view], s["silhouette"][view]
                lines.append(
                    f"| {name} | {view} | {c['crops_in_frozen_global_intervals']} | "
                    f"{fmt(c['nearest_pooled_centroid_agreement_with_frozen_name'])} | "
                    f"{c['crops_with_own_name']} | "
                    f"{fmt(c['nearest_pooled_centroid_agreement_with_own_name'])} | "
                    f"{fmt(sil['kmeans'])} / {fmt(sil.get('own_codes', np.nan))} / "
                    f"{fmt(sil['frozen_names'])} |"
                )
        lines += [
            "",
            "### Stability over time bins (own-code clusters matched to the reference per bin; "
            "interval agreement per bin)",
            "",
        ]
        for name in res["targets"]:
            s = res["summaries"][name]
            sources = {
                v: ("own_codes" if "own_codes" in s["cluster_matching"][v] else "kmeans")
                for v in VIEWS
            }
            lines += [
                f"**{name}** (per-bin clusters: "
                + ", ".join(f"{v} {sources[v]}" for v in VIEWS)
                + ")",
                "",
                "| Bin start s | Intervals | Frozen global | Margin min | Named by both fish-s | "
                "Agree fish-s | Disagree n (fish-s) | Bin assignment equals global: "
                "top / side1 / side2 | Bin margin top / side1 / side2 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
            per_bin = {
                v: {e["time_bin"]: e for e in s["cluster_matching"][v][sources[v]]["per_bin"]}
                for v in VIEWS
            }
            for b in s["bins"]:
                eq, mg = [], []
                for v in VIEWS:
                    e = per_bin.get(v, {}).get(b["time_bin"])
                    if not e or e.get("assignment") is None:
                        eq.append("n/a")
                        mg.append("n/a")
                    else:
                        eq.append("yes" if e["equals_global"] else "NO")
                        mg.append(fmt(e["margin"], 2))
                lines.append(
                    f"| {fmt(b['bin_start_s'], 0)} | {b['intervals']} | {b['frozen_global']} | "
                    f"{fmt(b['frozen_margin_min'])} | {fmt(b['both_named_fish_s'], 1)} | "
                    f"{fmt(b['agree_fish_s'], 1)} | {b['disagree_intervals']} "
                    f"({fmt(b['disagree_fish_s'], 1)}) | {' / '.join(eq)} | {' / '.join(mg)} |"
                )
            lines.append("")
    (out_dir / "report" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(HERE / "config.toml"))
    parser.add_argument("--only", default=None, help="restrict to one source trio")
    args = parser.parse_args(argv)
    import torch

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = load_paths()
    root = paths.raw_data_dir.parent
    labels = list(cfg["labels"])
    out_dir = resolve(cfg["output"], paths, root) / cfg["run_id"]
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already holds files; choose a new run_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = r1.load_lab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    with provenance("cross_trial_appearance", out_dir) as rec:
        rec.param("config", dict(cfg))
        rec.param("config_path", str(Path(args.config).resolve()))
        rec.param("device", str(device))
        rec.param("only", args.only)
        for p in (
            Path(__file__),
            REPO / cfg["lab"] / "appearance.py",
            Path(r1.__file__),
            Path(rn.__file__),
        ):
            rec.input(p.resolve().relative_to(REPO).as_posix(), p, sha256=sha_of(p))
        scores_path = resolve(cfg["six_study_scores"], paths, root)
        rec.input("six_study_scores", scores_path, sha256=sha_of(scores_path))
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        for trio, src in cfg["sources"].items():
            if args.only and trio != args.only:
                continue
            print(f"== {trio}: reference", flush=True)
            models, reference = build_reference(
                lab, cfg, src, paths, root, labels, scores, device, rec, out_dir, trio
            )
            print(
                json.dumps(reference["info"]["sanity_reembed_of_model_run"], indent=1), flush=True
            )
            summaries = {}
            for name in src["targets"]:
                print(f"== {trio} -> {name}", flush=True)
                summaries[name] = process_target(
                    lab,
                    cfg,
                    name,
                    cfg["targets"][name],
                    models,
                    reference,
                    labels,
                    paths,
                    root,
                    device,
                    rec,
                    out_dir,
                )
                s = summaries[name]
                print(
                    f"  frozen naming {s['frozen_naming']}, own available "
                    f"{s['own_encoder_available']}, perm {s['own_to_reference_permutation']}, "
                    f"agree {s['both_named']['agree_fraction']}, disagree "
                    f"{s['both_named']['disagree_intervals']} intervals",
                    flush=True,
                )
            results[trio] = {
                "reference": reference,
                "targets": list(src["targets"]),
                "summaries": summaries,
            }
            del models
            torch.cuda.empty_cache()
        summary_path = out_dir / "report" / "summary.json"
        r1.write_json(
            summary_path,
            {
                t: {"reference": r["reference"]["info"], "targets": r["summaries"]}
                for t, r in results.items()
            },
        )
        write_markdown(out_dir, cfg, results)
        for p in (summary_path, out_dir / "report" / "summary.md"):
            rec.output(p.name, p, sha256=sha_of(p))
    print(f"written {out_dir}", flush=True)


if __name__ == "__main__":
    main()
