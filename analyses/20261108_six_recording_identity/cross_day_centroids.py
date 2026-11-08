"""Label-free cross-day diagnostic of the per-group encoders (Phase 3).

For each trio and camera view, the three per-recording appearance centroids
(mean embedding of the clean lower-risk crops that the recording's N_carry
names with one code) are matched across each pair of days by embedding
distance. Reported per pair: the cost of the best one-to-one assignment, the
margin to the second-best assignment, and the smallest within-day centroid
separation, so that a reader can judge whether the three individuals are
separable across days at all. No cross-day identity is asserted. With
``--scores-tag`` the review's labels (through the alignment recorded by
score_candidates.py) are read to report whether the best assignment maps F
to F and each male to the same review name; the review's male names across
days are the reviewer's, not an independent fact.

Usage: cross_day_centroids.py [--scores-tag phase3_per_group]
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from medaka3d import load_paths

HERE = Path(__file__).resolve().parent
P11_ANALYSIS = HERE.parent / "20261031_p11_identity_evaluation"
sys.path.insert(0, str(P11_ANALYSIS))
import run_appearance_clean as rac  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")


def centroids_of(run: Path):
    """{view: (3, d) centroid array, within-day spread per code} from N_carry names."""
    with np.load(run / "evaluation" / "identity_tracks_N_carry.npz") as z:
        name = z["name"].astype(np.int8)
    out, spread = {}, {}
    for view in VIEWS:
        with np.load(rac.run_clean_path(run, "clean", view)) as z:
            frame, slot, clean = z["frame"].astype(np.int64), z["slot"].astype(np.int64), z["clean"]
        emb = np.load(run / "embeddings" / f"{view}.npy")
        code = name[frame, slot]
        sel = clean & (code >= 0)
        cents = np.stack([emb[sel & (code == k)].mean(axis=0) for k in range(3)])
        out[view] = cents
        spread[view] = [
            float(np.median(np.linalg.norm(emb[sel & (code == k)] - cents[k], axis=1)))
            for k in range(3)
        ]
    return out, spread


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores-tag", default=None)
    args = parser.parse_args(argv)
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    labels = list(cfg["labels"])
    root = load_paths().raw_data_dir.parent
    study = root / cfg["output"] / cfg["run_id"]
    alignment = None
    if args.scores_tag:
        scores = json.loads((study / "scores" / args.scores_tag / "scores.json").read_text())
        alignment = {
            name: r["candidates"]["N_carry"]["alignment"]["code_to_label"]
            for name, r in scores.items()
            if "N_carry" in r["candidates"]
        }
    report = {}
    for group, members in cfg["groups"].items():
        cents, spreads = {}, {}
        for name in members:
            run = root / cfg["recordings"][name]["appearance_run_group"]
            if not (run / "evaluation" / "identity_tracks_N_carry.npz").exists():
                continue
            cents[name], spreads[name] = centroids_of(run)
        pairs = {}
        for a, b in combinations(cents, 2):
            per_view = {}
            for view in VIEWS:
                d = np.linalg.norm(cents[a][view][:, None, :] - cents[b][view][None, :, :], axis=2)
                costs = sorted(
                    (sum(d[i, p[i]] for i in range(3)), p) for p in permutations(range(3))
                )
                best_cost, best = costs[0]
                second_cost = costs[1][0]
                within_a = min(
                    np.linalg.norm(cents[a][view][i] - cents[a][view][j])
                    for i, j in combinations(range(3), 2)
                )
                within_b = min(
                    np.linalg.norm(cents[b][view][i] - cents[b][view][j])
                    for i, j in combinations(range(3), 2)
                )
                entry = {
                    "best_assignment_a_to_b": list(map(int, best)),
                    "best_cost": float(best_cost),
                    "second_cost": float(second_cost),
                    "margin": float(second_cost - best_cost),
                    "matched_centroid_distances": [float(d[i, best[i]]) for i in range(3)],
                    "min_within_day_separation": float(min(within_a, within_b)),
                    "within_day_spread_a": spreads[a][view],
                    "within_day_spread_b": spreads[b][view],
                }
                if alignment and a in alignment and b in alignment:
                    la = {int(k): v for k, v in alignment[a].items()}
                    lb = {int(k): v for k, v in alignment[b].items()}
                    entry["review_labels_a_to_b"] = {
                        labels[la[i]]: labels[lb[best[i]]] for i in range(3)
                    }
                    entry["female_to_female"] = any(
                        labels[la[i]] == "F" and labels[lb[best[i]]] == "F" for i in range(3)
                    )
                    entry["all_names_preserved"] = all(la[i] == lb[best[i]] for i in range(3))
                per_view[view] = entry
            pairs[f"{a} -> {b}"] = per_view
        report[group] = {"recordings": list(cents), "pairs": pairs}
    out = study / "report"
    out.mkdir(exist_ok=True)
    (out / "cross_day_centroids.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for group, g in report.items():
        print(f"== {group}: {g['recordings']}")
        for pair, per_view in g["pairs"].items():
            for view, e in per_view.items():
                extra = ""
                if "review_labels_a_to_b" in e:
                    extra = (
                        f" | review map {e['review_labels_a_to_b']}"
                        f" F->F {e['female_to_female']} all {e['all_names_preserved']}"
                    )
                print(
                    f"  {pair} {view}: cost {e['best_cost']:.3f} margin {e['margin']:.3f} "
                    f"within-day sep {e['min_within_day_separation']:.3f} "
                    f"matched {np.round(e['matched_centroid_distances'], 3).tolist()}{extra}"
                )


if __name__ == "__main__":
    main()
