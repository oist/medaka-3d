"""Aggregate B0.2 census output into corpus totals and a per-group breakdown.

``b02_crossing_eval_set.py`` prints one block per experiment. This parses one or
more of those transcripts and reports the census the way the project requires it:
**by group**, because the group is the independent statistical unit. An
experiment-level total is a count of repeated measures and is reported only as a
workload figure, never as an n.

Usage:
    .venv\\Scripts\\python.exe tools/b02_aggregate_census.py TRANSCRIPT [TRANSCRIPT ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from medaka3d.cohort import CohortError, ExperimentId

BLOCK = re.compile(
    r"^(?P<exp>\d{8}_[A-Z]+_\d+_\d+_\w+)\s*\n=+\s*\n"
    r"frames: (?P<frames>\d+), scale: (?P<scale>[\d.]+) mm/px[^\n]*\n"
    r"detections/frame: mean (?P<mean_det>[\d.]+); 0 in (?P<zero>\d+), "
    r"<3 in (?P<lt3>\d+), >3 in (?P<gt3>\d+) frames\s*\n"
    r"\s*linked \(3 slots, all pairs\):\s*(?P<linked>\d+) encounters,\s*"
    r"(?P<linked_min>[\d.]+) annotation minutes\s*\n"
    r"\s*link-free \(closest pair\):\s*(?P<free>\d+) encounters,\s*"
    r"(?P<free_min>[\d.]+) annotation minutes",
    re.MULTILINE,
)


def parse(paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in BLOCK.finditer(text):
            exp = m.group("exp")
            rows[exp] = {  # last transcript wins; re-runs are idempotent
                "frames": int(m.group("frames")),
                "scale": float(m.group("scale")),
                "mean_det": float(m.group("mean_det")),
                "zero": int(m.group("zero")),
                "lt3": int(m.group("lt3")),
                "gt3": int(m.group("gt3")),
                "linked": int(m.group("linked")),
                "linked_min": float(m.group("linked_min")),
                "free": int(m.group("free")),
                "free_min": float(m.group("free_min")),
            }
    return rows


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    rows = parse([p for p in paths if p.is_file()])
    if not rows:
        raise SystemExit("no experiment blocks parsed")

    groups: dict[str, list[str]] = {}
    unparsed = []
    for exp in rows:
        try:
            groups.setdefault(ExperimentId.parse(exp).group, []).append(exp)
        except CohortError:
            unparsed.append(exp)

    print(f"experiments parsed: {len(rows)}   groups: {len(groups)}")
    if unparsed:
        print(f"UNPARSEABLE (reported, not dropped): {unparsed}")

    print(f"\n{'group':6s} {'exps':>5s} {'link-free':>10s} {'linked':>8s} "
          f"{'free hours':>11s} {'scale mm/px':>22s}")
    total_free = total_linked = total_free_min = 0.0
    for group in sorted(groups):
        exps = groups[group]
        free = sum(rows[e]["free"] for e in exps)
        linked = sum(rows[e]["linked"] for e in exps)
        free_min = sum(rows[e]["free_min"] for e in exps)
        scales = [rows[e]["scale"] for e in exps]
        total_free += free
        total_linked += linked
        total_free_min += free_min
        print(
            f"{group:6s} {len(exps):5d} {free:10d} {linked:8d} {free_min / 60:11.1f} "
            f"{min(scales):.4f}-{max(scales):.4f}".rjust(0)
        )

    print(
        f"\nCORPUS TOTAL over {len(rows)} experiments in {len(groups)} groups:\n"
        f"  link-free : {int(total_free):,} encounters = {total_free_min / 60:,.1f} h\n"
        f"  linked    : {int(total_linked):,} encounters = "
        f"{sum(rows[e]['linked_min'] for e in rows) / 60:,.1f} h"
    )
    print(
        f"\n  n = {len(groups)} GROUPS. The experiment count is a workload figure,\n"
        f"  not a sample size — days and trials within a group are repeated measures."
    )

    scales = np.array([r["scale"] for r in rows.values()])
    print(
        f"\nscale mm/px across experiments: min {scales.min():.4f}, "
        f"median {np.median(scales):.4f}, max {scales.max():.4f} "
        f"(spread {100 * (scales.max() / scales.min() - 1):.1f}%)"
    )
    print(
        "  This is an operational screening proxy from the extent the animals cover,\n"
        "  NOT physical calibration. The spread is why a per-experiment estimate\n"
        "  cannot be projected from a subset."
    )

    det = np.array([[r["lt3"], r["gt3"], r["zero"], r["frames"]] for r in rows.values()])
    lt3 = 100 * det[:, 0] / det[:, 3]
    gt3 = 100 * det[:, 1] / det[:, 3]
    print(
        f"\ndetection failures per experiment (% of frames):\n"
        f"  <3 instances: {lt3.min():.1f}-{lt3.max():.1f}% (median {np.median(lt3):.1f}%)\n"
        f"  >3 instances: {gt3.min():.1f}-{gt3.max():.1f}% (median {np.median(gt3):.1f}%)"
    )


if __name__ == "__main__":
    main()
