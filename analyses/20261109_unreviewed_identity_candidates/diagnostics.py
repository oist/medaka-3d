"""Label-free diagnostics of the fifteen-recording S/G run (registered quantities only)."""

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d import load_paths  # noqa: E402

cfg = tomllib.loads(
    (
        REPO / "analyses/20261031_p11_identity_evaluation/config_appearance_fifteen_v1.toml"
    ).read_text(encoding="utf-8")
)
paths = load_paths()
ROOT = paths.raw_data_dir.parent
STAGING = paths.appearance_staging_dir
KEYS = [
    "k21",
    "k22",
    "k23",
    "p21",
    "p22",
    "p23",
    "e11",
    "e12",
    "e13",
    "e21",
    "e22",
    "e23",
    "g11",
    "g12",
    "g13",
]
rows = []
for key in KEYS:
    rc = cfg[key]
    run = STAGING / rc["output"] / cfg["run_ids"][key]
    row = {"key": key, "recording": rc["experiment"]}
    iv = pd.read_csv(run / "bundle/lower_risk_intervals.csv")
    row["interval0_s"] = float(iv.duration_s[0])
    row["lower_risk_intervals"] = len(iv)
    row["interval_median_s"] = float(iv.duration_s.median())
    bs = json.loads((run / "bundle/summary.json").read_text(encoding="utf-8"))
    for k in (
        "lower_risk_seconds",
        "warning_seconds",
        "warning_intervals",
        "invalid_position_seconds",
    ):
        row[k] = bs[k]
    with np.load(run / "bundle/timing.npz") as z:
        weights = z["weights"]
    chain = ROOT / rc["chain_dir"] / f"C34-label__{rc['experiment']}__lam0.767.npz"
    with np.load(chain) as z:
        xyz = z["xyz"]
    valid = np.isfinite(xyz).all(axis=(1, 2))
    row["post300_s"] = float(weights.sum())
    row["chain_valid_s"] = float(weights[valid].sum())
    row["chain_invalid_s"] = float(weights[~valid].sum())
    for m in json.loads((run / "train_summary.json").read_text(encoding="utf-8")):
        row[f"train_{m['model']}_last_step"] = m["last_step"]
        row[f"train_{m['model']}_best_sil"] = round(m["best_silhouette"], 4)
        row[f"train_{m['model']}_stopped"] = m["stopped"]
    sp = run / "evaluation/summary.json"
    if not sp.exists():
        row["status"] = "evaluate failed"
        rows.append(row)
        continue
    s = json.loads(sp.read_text(encoding="utf-8"))
    row["status"] = "ok"
    row.update({f"intervals_{k}": v for k, v in s["intervals"].items()})
    row["anchors"] = s["anchors"]
    row["gaps"] = s["gaps"]
    row["disagreeing_gaps"] = s["switch_rate"]["disagreeing_gaps"]
    row["gap_seconds"] = round(s["switch_rate"]["gap_seconds"], 1)
    row["switch_rate_per_s"] = s["switch_rate"]["rate_per_s"]
    en = s.get("evidence_nodes") or {}
    row["evidence_nodes_warning"] = en.get("warning")
    row["evidence_nodes_unnamed"] = en.get("unnamed_lower_risk")
    row["centroid_agreement_min"] = min(
        v["nearest_centroid_agreement"] for v in s["calibration"].values()
    )
    names = {}
    for cand in ("N_carry", "S", "G"):
        p = run / f"evaluation/identity_tracks_{cand}.npz"
        row[f"{cand}_exists"] = p.exists()
        if not p.exists():
            continue
        with np.load(p) as z:
            name = z["name"]
        names[cand] = name
        n_named = (name >= 0).sum(axis=1)
        row[f"{cand}_named_fish_s"] = float((weights[valid] * n_named[valid]).sum())
        row[f"{cand}_withheld_fish_s"] = float((weights[valid] * (3 - n_named[valid])).sum())
        row[f"{cand}_named_frac_of_chain_valid"] = row[f"{cand}_named_fish_s"] / (
            3 * row["chain_valid_s"]
        )
        row[f"{cand}_named_on_chain_invalid_fish_s"] = float(
            (weights[~valid] * n_named[~valid]).sum()
        )
        row[f"{cand}_frames_partially_named"] = int(
            ((n_named > 0) & (n_named < 3) & valid & (weights > 0)).sum()
        )
    if "S" in names and "G" in names:
        both = (names["S"] >= 0) & (names["G"] >= 0)
        conflict = both & (names["S"] != names["G"])
        row["S_G_conflict_fish_s"] = float((weights[:, None] * conflict)[valid].sum())
        row["G_minus_S_named_fish_s"] = row["G_named_fish_s"] - row["S_named_fish_s"]
    rows.append(row)
df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "diagnostics.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)
print(df.set_index("key").T.to_string())
