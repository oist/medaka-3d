"""Phase 1b onward: score identity candidates against the human references.

Candidates: A0 (the chain's slot labels carried from the review's first
segment) and, when an appearance run exists, its N_carry, S and G identity
tracks. Candidate labels are aligned to the review's labels on the first
review segment only; see README.md, "Candidates". Quantities Q1--Q4 of the
README are computed on recording time >= analysis start, over frames where
the review knows the identity and the chain has a position.

Usage: score_candidates.py --variant per_recording [--candidates A0 N_carry S G]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d import load_paths
from medaka3d.eval.review_identity import contiguous_runs
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
TRACKS = ("N_carry", "S", "G")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=float) + "\n", encoding="utf-8")


def align_codes(cand: np.ndarray, ref: np.ndarray, window: np.ndarray):
    """Permutation of candidate codes maximising agreement with the review on ``window``."""
    sel = window[:, None] & (cand >= 0) & (ref >= 0)
    table = np.zeros((3, 3), np.int64)
    for c in range(3):
        for r in range(3):
            table[c, r] = int(((cand == c) & (ref == r) & sel).sum())
    best, best_score = None, -1
    for perm in permutations(range(3)):
        score = sum(table[c, perm[c]] for c in range(3))
        if score > best_score:
            best, best_score = perm, score
    total = int(table.sum())
    if total == 0:
        raise ValueError("Alignment window holds no frame named by both candidate and review")
    aligned = np.where(cand >= 0, np.array(best, np.int8)[np.clip(cand, 0, 2)], -1).astype(np.int8)
    return aligned, {
        "agreement_on_window": float(best_score / total),
        "window_frames_with_both": total,
        "code_to_label": {int(c): int(best[c]) for c in range(3)},
    }


def fish_seconds(mask: np.ndarray, weights: np.ndarray) -> float:
    return float((mask * weights[:, None]).sum())


def q1_identity_time(aligned, ref, known, weights, labels):
    correct = known & (aligned == ref)
    wrong = known & (aligned >= 0) & (aligned != ref)
    withheld = known & (aligned < 0)
    out = {
        "known_fish_seconds": fish_seconds(known, weights),
        "correct_fish_seconds": fish_seconds(correct, weights),
        "wrong_fish_seconds": fish_seconds(wrong, weights),
        "withheld_fish_seconds": fish_seconds(withheld, weights),
    }
    named = out["correct_fish_seconds"] + out["wrong_fish_seconds"]
    known_s = out["known_fish_seconds"]
    out["named_fraction_of_known"] = named / known_s if known_s else None
    out["wrong_fraction_of_named"] = out["wrong_fish_seconds"] / named if named else None
    out["correct_fraction_of_known"] = (
        out["correct_fish_seconds"] / out["known_fish_seconds"]
        if out["known_fish_seconds"]
        else None
    )
    out["per_fish"] = {}
    for k, lab in enumerate(labels):
        sel = known & (ref == k)
        out["per_fish"][lab] = {
            "known": fish_seconds(sel, weights),
            "correct": fish_seconds(sel & correct, weights),
            "wrong": fish_seconds(sel & wrong, weights),
            "withheld": fish_seconds(sel & withheld, weights),
        }
    return out, correct, wrong, withheld


def q2_name_changes(aligned, ref, known):
    """Changes of a slot's candidate name between consecutive review-known frames."""
    changes = with_ref = without_ref = 0
    for s in range(3):
        idx = np.flatnonzero(known[:, s] & (aligned[:, s] >= 0))
        if idx.size < 2:
            continue
        prev, cur = idx[:-1], idx[1:]
        changed = aligned[prev, s] != aligned[cur, s]
        ref_changed = ref[prev, s] != ref[cur, s]
        changes += int(changed.sum())
        with_ref += int((changed & ref_changed).sum())
        without_ref += int((changed & ~ref_changed).sum())
    return {
        "candidate_name_changes": changes,
        "coinciding_with_review_change": with_ref,
        "not_in_review": without_ref,
    }


def slot_of(code: np.ndarray, label: int) -> np.ndarray:
    """Slot index named ``label`` per frame, -1 when absent or ambiguous."""
    hits = code == label
    count = hits.sum(axis=1)
    slot = np.argmax(hits, axis=1)
    return np.where(count == 1, slot, -1)


def q3_pair_identity(aligned, ref, known_all, xyz, weights, labels, equal_mm):
    out = {}
    for a in range(3):
        for b in range(a + 1, 3):
            ra, rb = slot_of(ref, a), slot_of(ref, b)
            ca, cb = slot_of(aligned, a), slot_of(aligned, b)
            frames = known_all & (ra >= 0) & (rb >= 0)
            named = frames & (ca >= 0) & (cb >= 0)
            idx = np.flatnonzero(named)
            d_ref = np.linalg.norm(xyz[idx, ra[idx]] - xyz[idx, rb[idx]], axis=1)
            d_cand = np.linalg.norm(xyz[idx, ca[idx]] - xyz[idx, cb[idx]], axis=1)
            equal = np.zeros(len(xyz), bool)
            equal[idx] = np.abs(d_ref - d_cand) < equal_mm
            out[f"{labels[a]}-{labels[b]}"] = {
                "known_seconds": float(weights[frames].sum()),
                "named_seconds": float(weights[named].sum()),
                "equal_seconds": float(weights[equal].sum()),
                "equal_fraction_of_named": (
                    float(weights[equal].sum() / weights[named].sum())
                    if weights[named].sum()
                    else None
                ),
            }
    return out


def q4_contact_episodes(aligned, ref, known_all, xyz, times, weights, labels, qt):
    f = labels.index("F")
    males = [k for k in range(3) if k != f]
    rf = slot_of(ref, f)
    rm = [slot_of(ref, m) for m in males]
    frames = known_all & (rf >= 0) & (rm[0] >= 0) & (rm[1] >= 0)
    idx = np.flatnonzero(frames)
    d = np.full((len(xyz), 2), np.inf)
    for j in range(2):
        d[idx, j] = np.linalg.norm(xyz[idx, rf[idx]] - xyz[idx, rm[j][idx]], axis=1)
    nearer = np.argmin(d, axis=1)
    contact = frames & (d.min(axis=1) < qt["contact_distance_mm"])
    # Merge gaps shorter than the declared value, then drop short episodes.
    runs = contiguous_runs(contact)
    merged = []
    for lo, hi in runs:
        if merged and times[lo] - times[merged[-1][1] - 1] < qt["contact_merge_gap_s"]:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    episodes = []
    for lo, hi in merged:
        dur = float(weights[lo:hi].sum())
        if dur < qt["contact_min_duration_s"]:
            continue
        ep = np.zeros(len(xyz), bool)
        ep[lo:hi] = contact[lo:hi]
        nearer_slot = np.where(nearer == 0, rm[0], rm[1])
        ok_male = (
            aligned[np.arange(len(xyz)), np.clip(nearer_slot, 0, 2)]
            == ref[np.arange(len(xyz)), np.clip(nearer_slot, 0, 2)]
        )
        ok_female = aligned[np.arange(len(xyz)), np.clip(rf, 0, 2)] == f
        named = (
            ep
            & (aligned[np.arange(len(xyz)), np.clip(nearer_slot, 0, 2)] >= 0)
            & (aligned[np.arange(len(xyz)), np.clip(rf, 0, 2)] >= 0)
        )
        n_frames = int(ep.sum())
        n_named = int(named.sum())
        n_ok = int((ep & named & ok_male & ok_female).sum())
        n_wrong_male = int((ep & named & ~ok_male).sum())
        if n_named < qt["episode_attribution_min_fraction"] * n_frames:
            verdict = "withheld"
        elif n_ok >= qt["episode_attribution_min_fraction"] * n_frames:
            verdict = "correct"
        elif n_wrong_male >= qt["episode_attribution_min_fraction"] * n_frames:
            verdict = "wrong_male"
        else:
            verdict = "mixed"
        episodes.append(
            {
                "start_s": float(times[lo]),
                "end_s": float(times[hi - 1]),
                "duration_s": dur,
                "frames": n_frames,
                "nearer_male": labels[
                    males[int(np.bincount(nearer[lo:hi][contact[lo:hi]], minlength=2).argmax())]
                ],
                "min_distance_mm": float(d[lo:hi].min()),
                "named_frames": n_named,
                "correct_frames": n_ok,
                "wrong_male_frames": n_wrong_male,
                "verdict": verdict,
            }
        )
    counts = {
        v: sum(e["verdict"] == v for e in episodes)
        for v in ("correct", "wrong_male", "mixed", "withheld")
    }
    seconds = float(sum(e["duration_s"] for e in episodes))
    return {"episodes": len(episodes), "episode_seconds": seconds, "verdicts": counts}, episodes


def candidate_codes(run_dir: Path, track: str, n: int):
    path = run_dir / "evaluation" / f"identity_tracks_{track}.npz"
    if not path.exists():
        return None, None
    with np.load(path) as z:
        name = z["name"].astype(np.int8)
    if name.shape != (n, 3):
        raise ValueError(f"{path}: identity tracks have shape {name.shape}, expected {(n, 3)}")
    return name, path


def score_recording(name, rc, cfg, root, refs_dir, variant, candidates, rec, labels):
    qt = cfg["quantities"]
    ref_path = refs_dir / f"{name}_reference.npz"
    rec.input(f"{name}:reference", ref_path, sha256=sha256(ref_path))
    with np.load(ref_path) as z:
        ref = z["name_code"].astype(np.int8)
        a0 = z["a0_code"].astype(np.int8)
        chain_valid = z["chain_valid"]
        times, weights = z["recording_time_s"], z["duration_s"]
        covered = z["review_covered"]
        jingyi = z["jingyi_code"].astype(np.int8) if "jingyi_code" in z.files else None
    n = ref.shape[0]
    chain_dir = root / rc.get("chain_dir", cfg["chain"]["dir_template"].format(rec=name))
    label_path = chain_dir / cfg["chain"]["label_template"].format(rec=name, lam=cfg["lambda_mm"])
    rec.input(f"{name}:chain_label", label_path, sha256=sha256(label_path))
    with np.load(label_path) as z:
        xyz = z["xyz"]
    post = weights > 0
    known = (ref >= 0) & chain_valid & post[:, None]
    known_all = known.all(axis=1)
    # Alignment window: the first review segment that has frames at or after
    # the analysis start, restricted to those frames. Candidates name nothing
    # before the start, and a review's first segment can end before it.
    first_window = np.zeros(n, bool)
    first_lo, first_hi = 0, 0
    for lo, hi in contiguous_runs(covered):
        if post[lo:hi].any():
            first_lo, first_hi = int(lo), int(hi)
            first_window[lo:hi] = post[lo:hi]
            break
    if not first_window.any():
        raise ValueError(f"{name}: no review segment reaches the analysis start")

    codes = {}
    if "A0" in candidates:
        codes["A0"] = (np.broadcast_to(a0, (n, 3)).copy(), None)
    run_key = "appearance_run" if variant == "per_recording" else "appearance_run_group"
    run_dir = root / rc[run_key] if run_key in rc else None
    for track in TRACKS:
        if track in candidates and run_dir is not None:
            code, path = candidate_codes(run_dir, track, n)
            if code is not None:
                rec.input(f"{name}:{variant}:{track}", path, sha256=sha256(path))
                codes[track] = (code, str(path))
    results, episode_rows = {}, []
    for cand, (code, source) in codes.items():
        aligned, alignment = align_codes(code, ref, first_window)
        q1, *_ = q1_identity_time(aligned, ref, known, weights, labels)
        q2 = q2_name_changes(aligned, ref, known)
        q3 = q3_pair_identity(
            aligned, ref, known_all, xyz, weights, labels, qt["pair_distance_equal_mm"]
        )
        q4, episodes = q4_contact_episodes(aligned, ref, known_all, xyz, times, weights, labels, qt)
        for e in episodes:
            episode_rows.append({"recording": name, "variant": variant, "candidate": cand, **e})
        passes = (
            q1["wrong_fraction_of_named"] is not None
            and q1["wrong_fraction_of_named"] <= qt["max_wrong_fraction_of_named"]
            and q1["named_fraction_of_known"] >= qt["min_named_fraction"]
        )
        entry = {
            "source": source,
            "alignment": alignment,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
            "delivery_thresholds_met": bool(passes),
        }
        if jingyi is not None:
            known_j = (jingyi >= 0) & chain_valid & post[:, None]
            qj, *_ = q1_identity_time(aligned, jingyi, known_j, weights, labels)
            disagree = (jingyi >= 0) & (ref >= 0) & (jingyi != ref)
            qs, *_ = q1_identity_time(aligned, ref, known & ~disagree, weights, labels)
            entry["Q1_against_jingyi"] = qj
            entry["Q1_review_minus_disagreements"] = qs
        results[cand] = entry
    status = {"correct_seconds_per_slot": None}
    del status
    return {
        "frames": int(n),
        "known_fish_seconds": fish_seconds(known, weights),
        "first_segment_window_s": [float(times[first_lo]), float(times[first_hi - 1])],
        "candidates": results,
    }, episode_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=("per_recording", "per_group"), required=True)
    parser.add_argument("--candidates", nargs="+", default=["A0", *TRACKS])
    parser.add_argument("--tag", default=None, help="Output folder name under scores/")
    args = parser.parse_args(argv)
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    labels = list(cfg["labels"])
    paths = load_paths()
    root = paths.raw_data_dir.parent
    run = root / cfg["output"] / cfg["run_id"]
    refs_dir = run / "references"
    tag = args.tag or f"{args.variant}_{'-'.join(args.candidates)}"
    out = run / "scores" / tag
    out.mkdir(parents=True, exist_ok=False)
    report, episodes = {}, []
    with provenance("six_identity_scores", out) as rec:
        rec.param("configuration", cfg)
        rec.param("variant", args.variant)
        rec.param("candidates", args.candidates)
        for name, rc in cfg["recordings"].items():
            print(name, flush=True)
            report[name], rows = score_recording(
                name, rc, cfg, root, refs_dir, args.variant, args.candidates, rec, labels
            )
            episodes.extend(rows)
        write_json(out / "scores.json", report)
        rec.output("scores", out / "scores.json")
        pd.DataFrame(episodes).to_csv(out / "contact_episodes.csv", index=False)
        rec.output("contact_episodes", out / "contact_episodes.csv")
        rows = []
        for name, r in report.items():
            for cand, e in r["candidates"].items():
                q1 = e["Q1"]
                rows.append(
                    {
                        "recording": name,
                        "variant": args.variant,
                        "candidate": cand,
                        "known_fish_s": q1["known_fish_seconds"],
                        "correct_fish_s": q1["correct_fish_seconds"],
                        "wrong_fish_s": q1["wrong_fish_seconds"],
                        "withheld_fish_s": q1["withheld_fish_seconds"],
                        "named_fraction": q1["named_fraction_of_known"],
                        "wrong_of_named": q1["wrong_fraction_of_named"],
                        "name_changes": e["Q2"]["candidate_name_changes"],
                        "changes_not_in_review": e["Q2"]["not_in_review"],
                        "pair_F-M1_equal": e["Q3"]["F-M1"]["equal_fraction_of_named"],
                        "pair_F-M2_equal": e["Q3"]["F-M2"]["equal_fraction_of_named"],
                        "pair_M1-M2_equal": e["Q3"]["M1-M2"]["equal_fraction_of_named"],
                        "episodes": e["Q4"]["episodes"],
                        **{f"episodes_{k}": v for k, v in e["Q4"]["verdicts"].items()},
                        "thresholds_met": e["delivery_thresholds_met"],
                    }
                )
        table = pd.DataFrame(rows)
        table.to_csv(out / "summary_table.csv", index=False)
        rec.output("summary_table", out / "summary_table.csv")
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
