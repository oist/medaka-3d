"""Phase 4: tables and identity timelines from the study's score runs.

Reads the references and one or more score folders (``scores/<tag>/``) of the
study run and writes, under ``report/``: ``summary_table.md`` (one row per
recording, variant and candidate), ``trio_summary.md`` (pooled over the
three recordings of each trio) and one timeline figure per recording in
which each row is a candidate and each frame is coloured by whether all
three review-known slots are named correctly, any is wrong, any is withheld,
or the review does not know. Review swap times are marked.

Usage: report_study.py --tags phase2_per_recording phase3_per_group
"""

from __future__ import annotations

import argparse
import json
import tomllib
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medaka3d import load_paths

HERE = Path(__file__).resolve().parent
# Validated categorical palette of the dataviz reference instance, fixed order.
CORRECT, WRONG, WITHHELD = "#1baf7a", "#eb6834", "#2a78d6"
UNKNOWN, INK = "#d9d7d2", "#52514e"
TRACKS = ("A0", "N_carry", "S", "G")


def load_scores(run: Path, tag: str):
    folder = run / "scores" / tag
    scores = json.loads((folder / "scores.json").read_text())
    table = pd.read_csv(folder / "summary_table.csv")
    table["tag"] = tag
    return scores, table


def status_rows(ref_path: Path, candidates: dict[str, tuple], labels):
    """Per-frame status code per candidate: 0 unknown, 1 correct, 2 wrong, 3 withheld.

    ``candidates`` maps a display name to (identity-track path or None, code->label
    mapping recorded by score_candidates.py).
    """
    with np.load(ref_path) as z:
        ref = z["name_code"].astype(np.int8)
        a0 = z["a0_code"].astype(np.int8)
        valid = z["chain_valid"]
        times, weights = z["recording_time_s"], z["duration_s"]
        covered = z["review_covered"]
    post = weights > 0
    known = (ref >= 0) & valid & post[:, None]
    rows = {}
    for cand, (path, mapping) in candidates.items():
        if cand == "A0":
            code = np.broadcast_to(a0, ref.shape)
        elif path is None or not Path(path).exists():
            continue
        else:
            with np.load(path) as z:
                code = z["name"].astype(np.int8)
            perm = np.array([mapping[str(c)] for c in range(3)], np.int8)
            code = np.where(code >= 0, perm[np.clip(code, 0, 2)], -1)
        wrong = known & (code >= 0) & (code != ref)
        withheld = known & (code < 0)
        status = np.zeros(ref.shape[0], np.int8)
        status[known.any(axis=1)] = 1
        status[withheld.any(axis=1)] = 3
        status[wrong.any(axis=1)] = 2
        rows[cand] = status
    swaps = np.flatnonzero(np.diff(covered.astype(int)) != 0)
    return rows, times, post, swaps


def timeline_figure(name, rows, times, post, swaps, out: Path):
    colors = {0: UNKNOWN, 1: CORRECT, 2: WRONG, 3: WITHHELD}
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.2))
    t0 = float(times[post][0]) if post.any() else 0.0
    for i, status in enumerate(rows.values()):
        s = status.copy()
        s[~post] = 0
        edges = np.flatnonzero(np.diff(np.r_[-1, s, -1]) != 0)
        for lo, hi in pairwise(edges):
            code = int(s[lo])
            if code == 0 and not post[lo:hi].any():
                continue
            ax.barh(
                i,
                times[min(hi, len(times) - 1)] - times[lo],
                left=times[lo],
                height=0.7,
                color=colors[code],
                linewidth=0,
            )
    for f in swaps:
        ax.axvline(times[f], color=INK, linewidth=0.5, alpha=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([c.replace("_", " ") for c in rows])
    ax.set_xlim(t0, float(times[-1]))
    ax.set_xlabel("recording time (s)")
    ax.set_title(
        f"{name}: identity against the review"
        + chr(10)
        + "green: all three correct; orange: any wrong; blue: any withheld; "
        + "grey: review unknown; lines: review segment edges",
        fontsize=9,
        color=INK,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns, headers, floatfmt):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in df.itertuples(index=False):
        cells = []
        for col in columns:
            v = getattr(row, col)
            if isinstance(v, float):
                cells.append(floatfmt.get(col, "{:.3f}").format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args(argv)
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    labels = list(cfg["labels"])
    paths = load_paths()
    root = paths.raw_data_dir.parent
    run = root / cfg["output"] / cfg["run_id"]
    out = run / "report"
    out.mkdir(exist_ok=True)
    tables, candidate_paths = [], {name: {} for name in cfg["recordings"]}
    for tag in args.tags:
        scores, table = load_scores(run, tag)
        tables.append(table)
        for name, r in scores.items():
            variant = table.variant.iloc[0]
            for cand, e in r["candidates"].items():
                key = cand if cand == "A0" else f"{cand} ({variant.replace('_', ' ')})"
                candidate_paths[name][key] = (e["source"], e["alignment"]["code_to_label"])
    table = pd.concat(tables, ignore_index=True)
    table["short"] = table.recording.str.extract(r"_(K_\d_\d|P_\d_\d)_")[0]
    pair_cols = ["pair_F-M1_equal", "pair_F-M2_equal", "pair_M1-M2_equal"]
    table["pair_mean"] = table[pair_cols].mean(axis=1)
    cols = [
        "short",
        "variant",
        "candidate",
        "correct_fish_s",
        "wrong_fish_s",
        "withheld_fish_s",
        "named_fraction",
        "wrong_of_named",
        "pair_mean",
        "episodes",
        "episodes_correct",
        "episodes_wrong_male",
        "episodes_withheld",
        "thresholds_met",
    ]
    heads = [
        "Recording",
        "Variant",
        "Candidate",
        "Correct fish-s",
        "Wrong fish-s",
        "Withheld fish-s",
        "Named",
        "Wrong / named",
        "Pair identity (mean)",
        "Episodes",
        "Correct",
        "Wrong male",
        "Withheld",
        "Thresholds",
    ]
    fmt = {
        "correct_fish_s": "{:.0f}",
        "wrong_fish_s": "{:.1f}",
        "withheld_fish_s": "{:.1f}",
        "named_fraction": "{:.1%}",
        "wrong_of_named": "{:.2%}",
        "pair_mean": "{:.3f}",
    }
    (out / "summary_table.md").write_text(markdown_table(table, cols, heads, fmt), encoding="utf-8")
    table.to_csv(out / "summary_table.csv", index=False)

    trio = []
    for group, members in cfg["groups"].items():
        sub = table[table.recording.isin(members)]
        for (variant, cand), g in sub.groupby(["variant", "candidate"]):
            if len(g) != len(members):
                continue
            named = g.correct_fish_s.sum() + g.wrong_fish_s.sum()
            trio.append(
                {
                    "group": group,
                    "variant": variant,
                    "candidate": cand,
                    "recordings": len(g),
                    "correct_fish_s": g.correct_fish_s.sum(),
                    "wrong_fish_s": g.wrong_fish_s.sum(),
                    "withheld_fish_s": g.withheld_fish_s.sum(),
                    "named_fraction": named / g.known_fish_s.sum(),
                    "wrong_of_named": g.wrong_fish_s.sum() / named if named else float("nan"),
                    "episodes": int(g.episodes.sum()),
                    "episodes_correct": int(g.episodes_correct.sum()),
                    "episodes_wrong_male": int(g.episodes_wrong_male.sum()),
                    "episodes_withheld": int(g.episodes_withheld.sum()),
                    "recordings_meeting_thresholds": int(g.thresholds_met.sum()),
                }
            )
    trio_df = pd.DataFrame(trio)
    tcols = [
        "group",
        "variant",
        "candidate",
        "recordings",
        "correct_fish_s",
        "wrong_fish_s",
        "withheld_fish_s",
        "named_fraction",
        "wrong_of_named",
        "episodes",
        "episodes_correct",
        "episodes_wrong_male",
        "episodes_withheld",
        "recordings_meeting_thresholds",
    ]
    theads = [
        "Trio",
        "Variant",
        "Candidate",
        "Recordings",
        "Correct fish-s",
        "Wrong fish-s",
        "Withheld fish-s",
        "Named",
        "Wrong / named",
        "Episodes",
        "Correct",
        "Wrong male",
        "Withheld",
        "Recordings meeting thresholds",
    ]
    (out / "trio_summary.md").write_text(
        markdown_table(trio_df, tcols, theads, fmt), encoding="utf-8"
    )
    trio_df.to_csv(out / "trio_summary.csv", index=False)

    for name in cfg["recordings"]:
        rows, times, post, swaps = status_rows(
            run / "references" / f"{name}_reference.npz", candidate_paths[name], labels
        )
        if rows:
            timeline_figure(name, rows, times, post, swaps, out / f"{name}_timeline.png")
    print((out / "summary_table.md").read_text(encoding="utf-8"))
    print((out / "trio_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
