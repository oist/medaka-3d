"""Registration 3: bridging (S) and appearance inside gaps (G) on P-1-1.

Subcommands, in order: ``extract`` (pose-masked crops of warning-interior
frames, same builder as Registration 2), ``embed`` (Registration 2's B models)
and ``evaluate`` (N_B anchors, S, G, and the registered rules). The method is
fixed in ``labs/20261106_contrastive_appearance_link/README.md``,
Registration 3. Registrations 1 and 2 are read, never written. Reference
labels are read only by ``evaluate``.
"""

from __future__ import annotations

import argparse
import json
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
import run_appearance_naming as rn  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

REPO = r1.REPO
CONFIG = HERE / "config_appearance_gap_v1.toml"
VIEWS = r1.VIEWS


def settings():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["input"]["naming_config"] != rn.CONFIG.name:
        raise ValueError("Registration-2 configuration name changed")
    cfg2, r1cfg, root, r1out, out2 = rn.settings()
    snapshot = out2 / "source" / rn.CONFIG.relative_to(REPO)
    if r1.sha256(snapshot) != r1.sha256(rn.CONFIG):
        raise ValueError("Registration-2 configuration differs from its run snapshot")
    if json.loads((out2 / "status.json").read_text())["status"] != "evaluated":
        raise ValueError("Registration-2 run is not complete")
    if cfg["crops"]["frame_step"] != r1cfg["crops"]["frame_step"]:
        raise ValueError("Frame step differs from the lower-risk crops")
    return cfg, cfg2, r1cfg, root, r1out, out2, root / "jingyi/P-1-1" / cfg["run_id"]


def snapshot(rec, out, cfg, folder):
    import shutil

    for path in (
        Path(__file__),
        CONFIG,
        rn.CONFIG,
        Path(rn.__file__),
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


def warning_inputs(rec, root, r1cfg, cfg):
    warn = root / r1cfg["input"]["warning_run"]
    manifest = json.loads((warn / "artifact_manifest.json").read_text())
    mask = r1.verified(
        rec,
        "warning_mask",
        warn / r1cfg["input"]["warning_mask"],
        manifest[r1cfg["input"]["warning_mask"]]["sha256"],
    )
    table = r1.verified(
        rec,
        "warning_intervals",
        warn / cfg["input"]["warning_intervals"],
        manifest[cfg["input"]["warning_intervals"]]["sha256"],
    )
    lower = r1.verified(
        rec,
        "lower_risk_intervals",
        warn / r1cfg["input"]["lower_risk"],
        manifest[r1cfg["input"]["lower_risk"]]["sha256"],
    )
    with np.load(mask) as z:
        warning = z["warning"]
    return warning, pd.read_csv(table), pd.read_csv(lower)


# --------------------------------------------------------------------------- extract


def cmd_extract(cfg, cfg2, r1cfg, root, r1out, out2, out):
    track = root / r1cfg["input"]["tracking_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    inputs = json.loads((track / "inputs.json").read_text())
    videos = r1.video_hashes(track / r1cfg["input"]["video_provenance"])
    out.mkdir(exist_ok=False)
    for sub in ("crops", "masked"):
        (out / sub).mkdir()
    r1.write_json(out / "status.json", {"status": "extract running"})
    with provenance("p11_appearance_gap_extract", out) as rec:
        rec.param("configuration", cfg)
        snapshot(rec, out, cfg, "source")
        chain = r1.verified(
            rec,
            "c34_label",
            track / r1cfg["input"]["chain"],
            manifest[r1cfg["input"]["chain"]]["sha256"],
        )
        warning, table, _ = warning_inputs(rec, root, r1cfg, cfg)
        with np.load(chain) as label:
            source = label["source"]
        if source.shape[0] != warning.size:
            raise ValueError("Warning mask does not match the chain frames")
        warn_id = np.full(warning.size, -1, np.int32)
        for row in table.itertuples():
            warn_id[row.start_frame : row.stop_frame_exclusive] = row.interval_id
        if not np.array_equal(warn_id >= 0, warning):
            raise ValueError("Warning intervals do not reproduce the warning mask")
        step = cfg["crops"]["frame_step"]
        frames = np.flatnonzero(warning & (np.arange(warning.size) % step == 0))
        caches, jobs = {}, {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for v, view in enumerate(VIEWS):
                caches[view] = r1.verified(
                    rec,
                    f"detections_{view}",
                    Path(inputs[view]["cache"]),
                    inputs[view]["cache_sha256"],
                )
                video = Path(videos[view]["path"])
                if video.stat().st_size != videos[view]["size"]:
                    raise ValueError(f"{view}: source video size changed")
                jobs[view] = pool.submit(
                    r1.extract_view,
                    view,
                    video,
                    caches[view],
                    source[:, :, v],
                    frames,
                    warn_id,
                    out / "crops" / f"{view}.npz",
                )
            results = [jobs[view].result() for view in VIEWS]
        for result in results:
            view = result["view"]
            if result["video_sha256"] != videos[view]["sha256"]:
                raise ValueError(f"{view}: source video checksum changed")
            rec.input(f"video_{view}", videos[view]["path"], sha256=result["video_sha256"])
        with ProcessPoolExecutor(max_workers=3) as pool:
            masks = [
                pool.submit(
                    rn.mask_view,
                    view,
                    out / "crops" / f"{view}.npz",
                    caches[view],
                    out / "masked" / f"{view}.npz",
                )
                for view in VIEWS
            ]
            mask_results = [m.result() for m in masks]
        for view in VIEWS:
            path = out / "masked" / f"{view}.npz"
            rec.output(f"masked_{view}", path, sha256=r1.sha256(path))
        summary = {
            "sampled_warning_frames": int(frames.size),
            "extraction": results,
            "masking": mask_results,
        }
        rec.param("results", summary)
        r1.write_json(out / "extract_summary.json", summary)
    r1.write_json(out / "status.json", {"status": "extracted"})


# --------------------------------------------------------------------------- embed


def load_b_model(lab, out2, view):
    import torch

    state = torch.load(out2 / "models" / f"masked__{view}.pt", weights_only=False)
    model = lab.build_encoder().to(memory_format=torch.channels_last)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, (state["mean"], state["std"])


def cmd_embed(cfg, cfg2, r1cfg, root, r1out, out2, out):
    lab = r1.load_lab()
    (out / "embeddings").mkdir(exist_ok=False)
    r1.write_json(out / "status.json", {"status": "embed running"})
    with provenance("p11_appearance_gap_embed", out) as rec:
        checks = {}
        for v, view in enumerate(VIEWS):
            model, stats = load_b_model(lab, out2, view)
            # The B model must reproduce Registration 2's stored embeddings.
            reg2 = r1.verified(
                rec,
                f"reg2_embedding_{view}",
                out2 / "embeddings" / f"masked__{view}.npy",
                rn.recorded_output(out2, "p11_appearance_naming_train", f"embedding_{view}"),
            )
            reg2_masked = r1.verified(
                rec,
                f"reg2_masked_{view}",
                out2 / "masked" / f"{view}.npz",
                rn.recorded_output(out2, "p11_appearance_naming_mask", f"masked_{view}"),
            )
            stored = np.load(reg2)
            with np.load(reg2_masked) as z:
                crops = z["crops"]
            pick = np.sort(
                np.random.default_rng(v).choice(
                    crops.shape[0], cfg["gap"]["model_check_crops"], replace=False
                )
            )
            checks[view] = float(np.abs(lab.embed(model, crops[pick], stats) - stored[pick]).max())
            if checks[view] > 1e-4:
                raise ValueError(f"{view}: B model does not reproduce Registration 2")
            del crops
            masked = r1.verified(
                rec,
                f"masked_{view}",
                out / "masked" / f"{view}.npz",
                rn.recorded_output(out, "p11_appearance_gap_extract", f"masked_{view}"),
            )
            with np.load(masked) as z:
                emb = lab.embed(model, z["crops"], stats)
            path = out / "embeddings" / f"{view}.npy"
            np.save(path, emb)
            rec.output(f"embedding_{view}", path, sha256=r1.sha256(path))
        rec.param("reg2_model_check_max_abs_error", checks)
    r1.write_json(out / "status.json", {"status": "embedded"})


# --------------------------------------------------------------------------- evaluate


def gap_evidence(lab, emb, meta, centroids, tau, skip_intervals):
    """{(frame, slot): (3,) summed log p over cameras} from usable crops."""
    evidence = {}
    for view in VIEWS:
        c = np.stack([centroids[view][k] for k in range(3)])
        for e, frame, slot, usable, interval in zip(
            emb[view],
            meta[view]["frame"],
            meta[view]["slot"],
            meta[view]["usable"],
            meta[view]["interval"],
            strict=True,
        ):
            if not usable or interval in skip_intervals:
                continue
            logp = lab.log_identity_posterior(((e - c) ** 2).sum(axis=1)[None], tau[view])[0]
            key = (int(frame), int(slot))
            evidence[key] = evidence.get(key, 0.0) + logp
    return evidence


def run_g(lab, n, names, anchors, gaps, evidence, times, rate, alpha, threshold):
    out = np.full((n, 3), -1, np.int8)
    for seg, lo, hi in anchors:
        out[lo:hi] = names[seg]
    by_frame = {}
    for (frame, slot), logp in evidence.items():
        by_frame.setdefault(frame, np.zeros((3, 3)))[slot] = logp
    frames = np.array(sorted(by_frame))
    per_gap = []
    for a, b, first, stop in gaps:
        inside = frames[(frames >= first) & (frames < stop)]
        node_frames = [first - 1, *inside.tolist(), stop]
        dt = np.diff(times[node_frames])
        post = lab.gap_posterior(
            names[a], names[b], dt, [by_frame[f] for f in inside], rate, alpha=alpha
        )
        named = lab.node_names(lab.slot_marginals(post), threshold) if inside.size else []
        lab.fill_between_nodes(out, node_frames, [names[a], *named, names[b]])
        per_gap.append({"pre_anchor": a, "post_anchor": b, "nodes": int(inside.size)})
    return out, per_gap


def cmd_evaluate(cfg, cfg2, r1cfg, root, r1out, out2, out):
    lab = r1.load_lab()
    ev, gp = cfg["evaluate"], cfg["gap"]
    track = root / r1cfg["input"]["tracking_run"]
    manifest = json.loads((track / "artifact_manifest.json").read_text())
    dest = out / "evaluation"
    dest.mkdir(exist_ok=False)
    r1.write_json(out / "status.json", {"status": "evaluate running"})
    with provenance("p11_appearance_gap_evaluate", out) as rec:
        snapshot(rec, out, cfg, "source_evaluate")
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
        warning, table, lower = warning_inputs(rec, root, r1cfg, cfg)
        with np.load(scoring) as z:
            times, weights = z["recording_time_s"], z["duration_s"]
            matched = z["matched_slot"]
            anchor = z["anchor_reference_to_slot"]
        with np.load(chain) as z:
            xyz = z["xyz"]
        n = weights.size
        initial = [int(np.flatnonzero(anchor == s)[0]) for s in range(3)]

        # N_B names, recomputed from Registration 2 inputs and checked against its output.
        meta_lr, feats_b = {}, {}
        for view in VIEWS:
            with np.load(rn.r1_crops(rec, r1out, view)) as z:
                meta_lr[view] = {k: z[k] for k in ("frame", "slot", "interval", "overlap")}
            masked = r1.verified(
                rec,
                f"reg2_masked_{view}",
                out2 / "masked" / f"{view}.npz",
                rn.recorded_output(out2, "p11_appearance_naming_mask", f"masked_{view}"),
            )
            with np.load(masked) as z:
                meta_lr[view]["usable"] = z["usable"]
            emb = r1.verified(
                rec,
                f"reg2_embedding_{view}",
                out2 / "embeddings" / f"masked__{view}.npy",
                rn.recorded_output(out2, "p11_appearance_naming_train", f"embedding_{view}"),
            )
            feats_b[view] = np.load(emb)
        meta_b = {
            v: {
                "frame": m["frame"],
                "slot": m["slot"],
                "interval": m["interval"],
                "overlap": ~m["usable"],
            }
            for v, m in meta_lr.items()
        }
        whole = r1.fragment_rows(meta_b, times, lower, cfg2["evaluate"]["end_window_s"])[2]
        names, how, _ = rn.naming(lab, feats_b, whole, lower, initial)
        reg2 = pd.read_csv(out2 / "evaluation" / "interval_names_B.csv")
        for row in reg2.itertuples():
            if list(names[row.segment_id]) != [row.slot0, row.slot1, row.slot2]:
                raise AssertionError("N_B names do not reproduce Registration 2")
        bounds = [(r.segment_id, r.start_frame, r.stop_frame_exclusive) for r in lower.itertuples()]
        anchors, gaps = lab.anchors_and_gaps(bounds, names)

        # Regions: post-preparation time, gap time, the two named encounters.
        post = weights > 0
        in_gap = np.zeros(n, bool)
        for _, _, first, stop in gaps:
            in_gap[first:stop] = True
        regions = {
            "all": post,
            "gap": post & in_gap,
            "lower_risk": post & ~warning,
            "warning": post & warning,
        }
        for wid in ev["encounters"]:
            row = table[table.interval_id == wid].iloc[0]
            keep = np.zeros(n, bool)
            keep[row.start_frame : row.stop_frame_exclusive] = True
            regions[f"encounter_{wid}_{row.start_s:.3f}s"] = post & keep

        baseline = lab.frame_names(n, bounds, names, carry_warnings=True)
        reg2_summary = json.loads((out2 / "evaluation" / "summary.json").read_text())
        base_score = rn.score(baseline, matched, weights, regions)
        if not np.isclose(
            base_score["all"]["wrong"],
            reg2_summary["naming"]["N_B|carry"]["all"]["wrong"],
            atol=1e-9,
        ):
            raise AssertionError("N_B warning carry does not reproduce Registration 2")
        results = {"N_B_carry": base_score}

        bridged = lab.frame_names_bridged(n, bounds, names)
        results["S"] = rn.score(bridged, matched, weights, regions)

        # Label-free calibration and switch rate.
        labels = np.stack([names[i] for i in lower.segment_id])
        descs = [
            [r1.describe(lab, feats_b, whole, i, s) for s in range(3)] for i in lower.segment_id
        ]
        centroids = lab.label_centroids(descs, labels)
        tau, calibration = {}, {}
        for view in VIEWS:
            m = meta_lr[view]
            ident = np.full(m["frame"].size, -1)
            for seg in lower.segment_id:
                sel = m["interval"] == seg
                ident[sel] = names[seg][m["slot"][sel]]
            pick = m["usable"] & m["overlap"] & (ident >= 0)
            c = np.stack([centroids[view][k] for k in range(3)])
            dist2 = ((feats_b[view][pick][:, None, :] - c[None]) ** 2).sum(axis=2)
            tau[view] = lab.fit_temperature(dist2, ident[pick])
            calibration[view] = {"crops": int(pick.sum()), "tau": tau[view]}
        disagree = sum(bool((names[a] != names[b]).any()) for a, b, _, _ in gaps)
        gap_seconds = float(weights[in_gap].sum())
        rate = disagree / gap_seconds
        rec.param("calibration", calibration)
        rec.param(
            "switch_rate",
            {"disagreeing_gaps": disagree, "gap_seconds": gap_seconds, "rate_per_s": rate},
        )

        meta_w, emb_w = {}, {}
        for view in VIEWS:
            masked = r1.verified(
                rec,
                f"masked_{view}",
                out / "masked" / f"{view}.npz",
                rn.recorded_output(out, "p11_appearance_gap_extract", f"masked_{view}"),
            )
            with np.load(masked) as z:
                meta_w[view] = {k: z[k] for k in ("frame", "slot", "interval", "usable")}
            emb = r1.verified(
                rec,
                f"embedding_{view}",
                out / "embeddings" / f"{view}.npy",
                rn.recorded_output(out, "p11_appearance_gap_embed", f"embedding_{view}"),
            )
            emb_w[view] = np.load(emb)
        # Warning crops carry the warning id as interval; they are never anchors.
        warn_evidence = gap_evidence(lab, emb_w, meta_w, centroids, tau, set())
        unnamed = {i for i in lower.segment_id if not (names[i] >= 0).all()}
        lr_meta = {
            v: {
                **{k: meta_lr[v][k] for k in ("frame", "slot", "usable")},
                "interval": meta_lr[v]["interval"],
            }
            for v in VIEWS
        }
        lr_evidence = gap_evidence(
            lab, feats_b, lr_meta, centroids, tau, set(lower.segment_id) - unnamed
        )
        evidence = {**warn_evidence, **lr_evidence}
        if set(warn_evidence) & set(lr_evidence):
            raise AssertionError("Warning and lower-risk evidence overlap")

        variants = {"G": (gp["alpha"], gp["posterior"])}
        for a in gp["alpha_sensitivity"]:
            variants[f"G_alpha{a:g}"] = (a, gp["posterior"])
        for p in gp["posterior_sensitivity"]:
            variants[f"G_posterior{p:g}"] = (gp["alpha"], p)
        frames_g = {}
        for label, (alpha, threshold) in variants.items():
            frames_g[label], per_gap = run_g(
                lab, n, names, anchors, gaps, evidence, times, rate, alpha, threshold
            )
            results[label] = rn.score(frames_g[label], matched, weights, regions)
            if label == "G":
                gap_rows = per_gap

        # Per-gap table for S and G.
        held = {}
        for label, frame in (("S", bridged), ("G", frames_g["G"]), ("N_B_carry", baseline)):
            h = np.take_along_axis(frame, np.maximum(matched, 0).astype(np.int64), axis=1)
            held[label] = np.where(matched >= 0, h, -1)
        assessed = (matched >= 0) & post[:, None]
        for row, (a, b, first, stop) in zip(gap_rows, gaps, strict=True):
            sl = slice(first, stop)
            w = weights[sl, None]
            row.update(
                {
                    "start_s": float(times[first]),
                    "end_s": float(times[stop - 1]),
                    "seconds": float(weights[sl].sum()),
                    "anchors_agree": bool((names[a] == names[b]).all()),
                }
            )
            for label, h in held.items():
                ok = assessed[sl]
                named = (h[sl] >= 0) & ok
                row[f"{label}_named"] = float((named * w).sum())
                row[f"{label}_wrong"] = float(((named & (h[sl] != np.arange(3)[None])) * w).sum())
                row[f"{label}_unknown"] = float((((h[sl] < 0) & ok) * w).sum())
            row["first_frame"], row["stop_frame"] = int(first), int(stop)
        gap_table = pd.DataFrame(gap_rows)
        gap_table.to_csv(dest / "gap_table.csv", index=False)

        b2 = pd.read_csv(out2 / "evaluation" / "boundaries.csv")
        stops = lower.set_index("segment_id").stop_frame_exclusive
        chain_rows = []
        for row in b2[(b2.method == "A0_chain") & (b2.outcome == "wrong")].itertuples():
            frame = int(stops[row.pre_interval])
            hit = gap_table[(gap_table.first_frame <= frame) & (gap_table.stop_frame >= frame)]
            chain_rows.append(
                {
                    "boundary_s": float(row.boundary_s),
                    **(hit.iloc[0].to_dict() if len(hit) else {"gap": None}),
                }
            )

        def rule(candidate, against):
            c, g = results[candidate]["gap"], results[against]["gap"]
            if c["wrong_fraction_of_named"] > ev["max_wrong_fraction_of_named"]:
                return "reject"
            if candidate.startswith("G") and c["named"] < g["named"]:
                return "reject"
            return "adopt" if c["named"] > g["named"] else "inconclusive"

        summary = {
            "results": results,
            "intervals": {
                k: sum(h == k for h in how.values())
                for k in ("global", "partial_two", "partial_one", "unknown")
            },
            "anchors": len(anchors),
            "gaps": len(gaps),
            "calibration": calibration,
            "switch_rate": {
                "disagreeing_gaps": disagree,
                "gap_seconds": gap_seconds,
                "rate_per_s": rate,
            },
            "evidence_nodes": {
                "warning": len({f for f, _ in warn_evidence}),
                "unnamed_lower_risk": len({f for f, _ in lr_evidence}),
            },
            "chain_wrong_boundaries": chain_rows,
            "decision_rules": {"S_vs_N_B_carry": rule("S", "N_B_carry"), "G_vs_S": rule("G", "S")},
        }
        r1.write_json(dest / "summary.json", summary)
        rec.output("summary", dest / "summary.json")
        for label, frame in (("S", bridged), ("G", frames_g["G"])):
            by_identity = np.full((n, 3, xyz.shape[2]), np.nan, np.float32)
            for k in range(3):
                for s in range(3):
                    sel = frame[:, s] == k
                    by_identity[sel, k] = xyz[sel, s]
            path = dest / f"identity_tracks_{label}.npz"
            np.savez_compressed(
                path,
                name=frame,
                xyz_by_identity=by_identity,
                identity_labels=np.array(rn.NAMES),
                recording_time_s=times,
            )
            rec.output(f"identity_tracks_{label}", path, sha256=r1.sha256(path))
        rec.param("reference_use", "evaluation only; fixed 298-300 s anchor names interval 0")
    r1.write_json(out / "status.json", {"status": "evaluated"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "embed", "evaluate"))
    args = parser.parse_args(argv)
    cfg, cfg2, r1cfg, root, r1out, out2, out = settings()
    if args.stage != "extract" and not out.is_dir():
        raise SystemExit(f"Run directory does not exist: {out}")
    {"extract": cmd_extract, "embed": cmd_embed, "evaluate": cmd_evaluate}[args.stage](
        cfg, cfg2, r1cfg, root, r1out, out2, out
    )


if __name__ == "__main__":
    main()
