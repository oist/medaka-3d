"""Phase 1a: human identity references on the chain's slots for the six recordings.

For each recording the ID-pass review is read onto the chain output that the
recording's appearance run consumes: every finite position of the review
package (the production chain's pectoral anchor, ID_k = slot k) is matched
to that chain's slot at the same frame within 1 mm, mutually unique, and the
slot takes the name of the review segment containing the frame; frames
outside every segment are unknown. The build stops if the review's swap
events replayed from its initial labels do not reproduce its segment table,
or if fewer than 99.5% of the package's positions find a slot. P_1_1 also
gets the Jingyi review matched to slots by position, K_1_2 the female clicks
as a check. Nothing under the data parent is modified; outputs go to the
study's run directory. See README.md, "References".
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d import load_paths
from medaka3d.eval.review_identity import contiguous_runs, interval_weights, match_positions
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
P11_ANALYSIS = HERE.parent / "20261031_p11_identity_evaluation"
VIEWS = ["top_cam", "side1_cam", "side2_cam"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=float) + "\n", encoding="utf-8")


def chain_label_path(cfg, rc, root, name):
    chain_dir = root / rc.get("chain_dir", cfg["chain"]["dir_template"].format(rec=name))
    return chain_dir / cfg["chain"]["label_template"].format(rec=name, lam=cfg["lambda_mm"])


def parse_names(text: str, labels: list[str]) -> dict[int, int]:
    """'M1=ID_1 F=ID_2 M2=ID_3' -> {package id index: label code}."""
    out = {}
    for item in text.split():
        name, fish = item.split("=")
        pid = int(fish.split("_")[1]) - 1
        if pid in out or name not in labels:
            raise ValueError(f"Bad segment names {text!r}")
        out[pid] = labels.index(name)
    if sorted(out) != [0, 1, 2] or sorted(out.values()) != [0, 1, 2]:
        raise ValueError(f"Segment names are not a permutation: {text!r}")
    return out


def load_package_positions(csv: Path, n: int):
    df = pd.read_csv(csv)
    piv = df.pivot(index="frame", columns="fish_id", values=["X", "Y", "Z", "is_interpolated"])
    piv = piv.reindex(range(n))
    ids = ["ID_1", "ID_2", "ID_3"]
    pos = np.stack([piv[[("X", i), ("Y", i), ("Z", i)]].to_numpy(float) for i in ids], axis=1)
    interp = np.stack(
        [piv[("is_interpolated", i)].fillna(False).to_numpy(bool) for i in ids], axis=1
    )
    return pos, interp


def package_to_slots(pos: np.ndarray, xyz: np.ndarray, chain_valid: np.ndarray, max_mm: float):
    """Slot of each package id per frame by mutual nearest position within ``max_mm``."""
    n = pos.shape[0]
    finite = np.isfinite(pos).all(axis=2)
    d = np.linalg.norm(pos[:, :, None, :] - xyz[:, None, :, :], axis=3)  # (n, id, slot)
    d = np.where(finite[:, :, None] & chain_valid[:, None, :] & np.isfinite(d), d, np.inf)
    best = np.argmin(d, axis=2)
    best_d = np.take_along_axis(d, best[:, :, None], axis=2)[:, :, 0]
    slot = np.where(best_d <= max_mm, best, -1)
    # Mutual uniqueness: a slot claimed by two ids is given to neither.
    for f in np.flatnonzero((slot >= 0).sum(axis=1) >= 2):
        s = slot[f]
        vals, counts = np.unique(s[s >= 0], return_counts=True)
        for v in vals[counts > 1]:
            slot[f, s == v] = -1
    matched = slot >= 0
    return slot, {
        "finite_positions": int(finite.sum()),
        "matched_positions": int(matched.sum()),
        "matched_fraction": float(matched.sum() / max(finite.sum(), 1)),
        "same_index_fraction": float(
            np.mean(slot[matched] == np.broadcast_to(np.arange(3), (n, 3))[matched])
        )
        if matched.any()
        else None,
        "median_mm": float(np.median(best_d[matched])) if matched.any() else None,
    }


def replay_swaps(events: pd.DataFrame, n: int, labels: list[str], analysis_start: int):
    """Initial labels, then swaps in frame order -> per-frame package id code (n, 3)."""
    code = np.full((n, 3), -1, np.int8)
    mapping = {}
    for row in events[events.kind == "label"].sort_values("created").itertuples():
        pid = int(str(row.fish).split("_")[1]) - 1
        mapping[pid] = labels.index(str(row.value))
    if sorted(mapping) != [0, 1, 2] or sorted(mapping.values()) != [0, 1, 2]:
        raise ValueError(f"Initial labels are not a permutation: {mapping}")
    swaps = events[events.kind == "swap"].sort_values(["frame", "created"])
    cursor = int(analysis_start)
    for row in swaps.itertuples():
        f = int(row.frame)
        for pid, lab in mapping.items():
            code[cursor:f, pid] = lab
        a, b = (int(x.split("_")[1]) - 1 for x in str(row.fish).split("|"))
        mapping[a], mapping[b] = mapping[b], mapping[a]
        cursor = f
    for pid, lab in mapping.items():
        code[cursor:, pid] = lab
    return code, len(swaps)


def id_pass_codes(idp: Path, n: int, labels: list[str]):
    """Per-frame label code of each package id, review coverage, flags and facts."""
    summary = json.loads((idp / "id_pass_summary.json").read_text())
    segments = pd.read_csv(idp / "id_segments.csv")
    events = pd.read_csv(idp / "human_events.csv")
    id_code = np.full((n, 3), -1, np.int8)
    covered = np.zeros(n, bool)
    for row in segments.itertuples():
        mapping = parse_names(str(row.names), labels)
        lo, hi = int(row.start), int(row.end) + 1
        if covered[lo:hi].any():
            raise ValueError(f"Overlapping segments at {lo}-{hi}")
        covered[lo:hi] = True
        for pid, lab in mapping.items():
            id_code[lo:hi, pid] = lab
    replayed, n_swaps = replay_swaps(events, n, labels, summary["analysis_start"])
    mismatch = covered & (replayed != id_code).any(axis=1)
    if mismatch.any():
        where = np.flatnonzero(mismatch)
        raise ValueError(
            f"Swap replay differs from id_segments on {where.size} frames, "
            f"first {where[:5].tolist()}"
        )
    danger = np.zeros(n, bool)
    for row in events[events.kind == "danger"].itertuples():
        danger[int(row.frame) : int(row.frame_end) + 1] = True
    pose_bad_id = np.zeros((n, 3), bool)
    for row in events[events.kind == "pose_bad"].itertuples():
        if str(row.fish) == "*":
            pose_bad_id[int(row.frame) : int(row.frame_end) + 1, :] = True
        else:
            pid = int(str(row.fish).split("_")[1]) - 1
            pose_bad_id[int(row.frame) : int(row.frame_end) + 1, pid] = True
    facts = {
        "analysis_start_frame": int(summary["analysis_start"]),
        "segments": len(segments),
        "swap_events": int(n_swaps),
        "danger_events": int((events.kind == "danger").sum()),
        "pose_bad_events": int((events.kind == "pose_bad").sum()),
        "label_events": int((events.kind == "label").sum()),
        "summary_identity_changes": summary["identity_changes"],
        "first_segment_names": str(segments.iloc[0].names),
        "swap_replay_matches_segments": True,
    }
    return id_code, covered, danger, pose_bad_id, facts


def to_slots(per_id: np.ndarray, slot_of_id: np.ndarray, fill):
    """Re-index a per-package-id array (n, 3) onto chain slots using slot_of_id."""
    n = per_id.shape[0]
    out = np.full((n, 3), fill, per_id.dtype)
    for pid in range(3):
        sel = slot_of_id[:, pid] >= 0
        out[np.flatnonzero(sel), slot_of_id[sel, pid]] = per_id[sel, pid]
    return out


def jingyi_reference(
    csv: Path,
    xyz: np.ndarray,
    pred_valid: np.ndarray,
    labels: list[str],
    max_mm: float,
    margin_mm: float,
):
    cols = [
        "frame",
        "label",
        "watched",
        "analysed",
        "uncertain",
        "bad3d",
        "pectoral_X",
        "pectoral_Y",
        "pectoral_Z",
    ]
    df = pd.read_csv(csv, usecols=cols)
    n = xyz.shape[0]
    idx = pd.MultiIndex.from_product([range(n), labels], names=["frame", "label"])
    df = df.set_index(["frame", "label"]).reindex(idx)
    ref = df[["pectoral_X", "pectoral_Y", "pectoral_Z"]].to_numpy(float).reshape(n, 3, 3)
    flags = {
        c: df[c].fillna(False).to_numpy(bool).reshape(n, 3)
        for c in ("watched", "analysed", "uncertain", "bad3d")
    }
    valid = (
        flags["watched"]
        & flags["analysed"]
        & ~flags["uncertain"]
        & ~flags["bad3d"]
        & np.isfinite(ref).all(axis=2)
    )
    matches = match_positions(
        ref, xyz, valid, pred_valid, max_distance_mm=max_mm, min_margin_mm=margin_mm
    )
    code = np.full((n, 3), -1, np.int8)
    slots = matches.slots
    for k in range(3):
        sel = slots[:, k] >= 0
        if (code[np.flatnonzero(sel), slots[sel, k]] >= 0).any():
            raise ValueError("Two reference fish matched one slot")
        code[np.flatnonzero(sel), slots[sel, k]] = k
    return code, valid


def main():
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    labels = list(cfg["labels"])
    paths = load_paths()
    root = paths.raw_data_dir.parent
    out = root / cfg["output"] / cfg["run_id"] / "references"
    out.mkdir(parents=True, exist_ok=False)
    mt = cfg["matching"]
    report = {}
    with provenance("six_identity_references", out) as rec:
        rec.param("configuration", cfg)
        for name, rc in cfg["recordings"].items():
            print(name, flush=True)
            label_path = chain_label_path(cfg, rc, root, name)
            rec.input(f"{name}:chain_label", label_path, sha256=sha256(label_path))
            with np.load(label_path) as z:
                xyz, slot_track, source = z["xyz"], z["slot_track"], z["source"]
            n = xyz.shape[0]
            chain_valid = np.isfinite(xyz).all(axis=2) & (slot_track >= 0)
            frametimes = paths.raw_data_dir / name / "top_cam" / "frametimes.npy"
            rec.input(f"{name}:frametimes", frametimes, sha256=sha256(frametimes))
            times = np.load(frametimes)[1].astype(np.float64)
            if times.size != n:
                raise ValueError(f"{name}: {n} chain frames, {times.size} acquisition times")
            weights, expected = interval_weights(times, cfg["analysis_start_s"])
            post = weights > 0

            pkg = rc["package"]
            csv = root / cfg["package"]["csv_template"].format(rec=name, pkg=pkg)
            manifest = root / cfg["package"]["manifest_template"].format(rec=name, pkg=pkg)
            csv_hash = sha256(csv)
            rec.input(f"{name}:package_csv", csv, sha256=csv_hash)
            man = json.loads(manifest.read_text())
            if man["sources"]["csv_3d"]["sha256"] != csv_hash:
                raise ValueError(f"{name}: package CSV differs from the one the GUI displayed")
            pos, interpolated_id = load_package_positions(csv, n)
            slot_of_id, match_facts = package_to_slots(
                pos, xyz, chain_valid, mt["same_slot_max_mm"]
            )
            if match_facts["matched_fraction"] < 0.995:
                raise ValueError(
                    f"{name}: only {match_facts['matched_fraction']:.4f} of the "
                    "package positions find a chain slot"
                )

            idp = root / cfg["package"]["id_pass_template"].format(rec=name, pkg=pkg)
            for f in ("id_segments.csv", "human_events.csv", "id_pass_summary.json"):
                rec.input(f"{name}:id_pass:{f}", idp / f, sha256=sha256(idp / f))
            id_code, covered, danger, pose_bad_id, facts = id_pass_codes(idp, n, labels)
            name_code = to_slots(id_code, slot_of_id, np.int8(-1))
            pose_bad = to_slots(pose_bad_id, slot_of_id, False)
            interpolated = to_slots(interpolated_id, slot_of_id, False)
            # Review-known package positions that found no slot in this chain.
            unmapped = (id_code >= 0) & (slot_of_id < 0) & post[:, None]

            # A0: each slot's majority review label over the first review segment
            # that reaches the analysis start; the review names the package ids,
            # the slots of this chain are read through the position match.
            a0 = np.full(3, -1, np.int8)
            a0_facts = {}
            for lo, hi in contiguous_runs(covered):
                lo, hi = int(lo), int(hi)
                if not post[lo:hi].any():
                    continue
                win = np.zeros(n, bool)
                win[lo:hi] = post[lo:hi]
                for s in range(3):
                    vals = name_code[win, s]
                    vals = vals[vals >= 0]
                    counts = np.bincount(vals, minlength=3) if vals.size else np.zeros(3, int)
                    a0[s] = int(np.argmax(counts)) if vals.size else -1
                    a0_facts[f"slot{s}"] = {
                        "majority_label": labels[a0[s]] if a0[s] >= 0 else None,
                        "majority_fraction": float(counts.max() / vals.size) if vals.size else None,
                    }
                a0_facts["window_s"] = [float(times[lo]), float(times[hi - 1])]
                break
            if sorted(a0.tolist()) != [0, 1, 2]:
                raise ValueError(f"{name}: A0 labels are not a permutation: {a0.tolist()}")

            def fish_seconds(mask, weights=weights):
                return float((mask * weights[:, None]).sum())

            known = (name_code >= 0) & chain_valid & post[:, None]
            summary = {
                "chain_label": str(label_path),
                "frames": int(n),
                "expected_seconds_after_start": float(expected),
                "expected_fish_seconds": float(3 * expected),
                "known_fish_seconds": fish_seconds(known),
                "unknown_review_fish_seconds": fish_seconds((name_code < 0) & post[:, None]),
                "review_known_but_unmapped_fish_seconds": fish_seconds(unmapped),
                "chain_missing_fish_seconds": fish_seconds(~chain_valid & post[:, None]),
                "danger_seconds_after_start": float(weights[danger & post].sum()),
                "pose_bad_fish_seconds": fish_seconds(pose_bad & post[:, None]),
                "package_interpolated_fish_seconds": fish_seconds(interpolated & post[:, None]),
                "package_match": match_facts,
                "a0": a0_facts,
                "review": facts,
            }
            arrays = {
                "name_code": name_code,
                "a0_code": a0,
                "danger": danger,
                "pose_bad": pose_bad,
                "package_interpolated": interpolated,
                "chain_valid": chain_valid,
                "review_covered": covered,
                "slot_of_package_id": slot_of_id,
                "recording_time_s": times,
                "duration_s": weights,
                "labels": np.array(labels),
                "analysis_start_s": np.float64(cfg["analysis_start_s"]),
            }

            if "jingyi_review" in rc:
                jcsv = root / rc["jingyi_review"]
                jhash = sha256(jcsv)
                if jhash != rc["jingyi_review_sha256"]:
                    raise ValueError("Jingyi review checksum differs from the registered one")
                rec.input(f"{name}:jingyi_review", jcsv, sha256=jhash)
                jcode, jvalid = jingyi_reference(
                    jcsv,
                    xyz,
                    chain_valid,
                    labels,
                    mt["jingyi_max_distance_mm"],
                    mt["jingyi_min_margin_mm"],
                )
                both_known = (jcode >= 0) & (name_code >= 0) & post[:, None]
                agree = both_known & (jcode == name_code)
                per_label = {}
                for k, lab in enumerate(labels):
                    sel_j = both_known & (jcode == k)
                    per_label[lab] = {
                        "fish_seconds_both_known": fish_seconds(sel_j),
                        "agreement": float(
                            fish_seconds(sel_j & agree) / max(fish_seconds(sel_j), 1e-9)
                        ),
                    }
                summary["jingyi"] = {
                    "valid_reference_fish_seconds": fish_seconds(jvalid & post[:, None]),
                    "matched_to_slot_fish_seconds": fish_seconds((jcode >= 0) & post[:, None]),
                    "both_known_fish_seconds": fish_seconds(both_known),
                    "agreement_fish_seconds": fish_seconds(agree),
                    "disagreement_fish_seconds": fish_seconds(both_known & ~agree),
                    "per_label": per_label,
                    "jingyi_known_review_unknown_fish_seconds": fish_seconds(
                        (jcode >= 0) & (name_code < 0) & post[:, None]
                    ),
                    "review_known_jingyi_unknown_fish_seconds": fish_seconds(
                        (jcode < 0) & (name_code >= 0) & post[:, None]
                    ),
                }
                arrays["jingyi_code"] = jcode
                arrays["jingyi_valid"] = jvalid

            if "female_reference" in rc:
                sys.path.insert(0, str(P11_ANALYSIS))
                import run_appearance_clean as rac

                gt_path = Path(paths.asset_root) / rc["female_reference"]
                rec.input(f"{name}:female_reference", gt_path, sha256=sha256(gt_path))
                gt = pd.read_csv(gt_path, low_memory=False)
                caches = {}
                for view in VIEWS:
                    cache = root / rc["cache_dir"] / f"{name}__{view}.npz"
                    rec.input(f"{name}:cache_{view}", cache, sha256=sha256(cache))
                    caches[view] = cache
                anchors, counts = rac.female_anchors(gt, caches, source, mt["female_match_px"])
                frames_a = anchors.frame.to_numpy(int)
                slots_a = anchors.slot.to_numpy(int)
                review_at = name_code[frames_a, slots_a]
                scored = post[frames_a]
                f_code = labels.index("F")
                summary["female_clicks"] = {
                    "matching": counts,
                    "anchors_after_start": int(scored.sum()),
                    "review_names_F": int(((review_at == f_code) & scored).sum()),
                    "review_names_male": int(
                        ((review_at >= 0) & (review_at != f_code) & scored).sum()
                    ),
                    "review_unknown": int(((review_at < 0) & scored).sum()),
                }
                anchors.assign(review_code=review_at, after_start=scored).to_csv(
                    out / f"{name}_female_anchors.csv", index=False
                )

            path = out / f"{name}_reference.npz"
            np.savez_compressed(path, **arrays)
            rec.output(f"{name}:reference", path, sha256=sha256(path))
            report[name] = summary
        write_json(out / "summary.json", report)
        rec.output("summary", out / "summary.json")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
