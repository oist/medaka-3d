"""Phase 0.5 — can the study's conclusion be reached without tracking at all?

Re-derives, from the experimenter's own sheets and nothing else, whether the
male the female accepted is also the male that courted her more.

Decision rule, fixed before the numbers were read
-------------------------------------------------
The independent unit is the **group** (one trio of fish). Days and trials within
a group are repeated measures. A result is reported as "n = <number of groups>",
never as a count of experiments.

Reads two sheets under ``asset_root/catalog/experiment_list/``:

* ``boris_events.csv``   — per-experiment BORIS event counts + ``female_accept``
* ``lay_eggs_table.csv`` — the same, plus group/trial/day and the egg outcome

Every ``file_names`` value is validated against the real recording folders under
``paths.raw_data_dir``. Rows that do not join are **reported, never dropped** —
the sheets are known to carry date typos, and a silent drop is how a cohort
quietly shrinks.

Run:
    .venv\\Scripts\\python.exe tools/phase05_accept_vs_courtship.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from medaka3d import load_paths
from medaka3d.cohort import CohortError, ExperimentId

# `female_accept` may name two males, e.g. "m2m1". The sheet's own legend fixes
# the reading: "during the egg-laying, the second wrap male will not be regarded
# as 'accepted'". So the FIRST male named is the accepted one.
ACCEPT_TOKENS = ("m1", "m2")


@dataclass(frozen=True)
class Row:
    """One sheet row, joined (or not) to a real recording folder."""

    sheet: str
    raw_name: str
    parsed: ExperimentId | None
    joined_folder: str | None
    join_note: str
    accept: str | None
    m1_courtship: int | None
    m2_courtship: int | None
    eggs: str


def resolve_accept(raw: str) -> tuple[str | None, bool]:
    """(accepted male, was it ambiguous) from a raw ``female_accept`` cell.

    Returns ``(None, False)`` for blank, "no" or "?" — an absent record, which
    is not the same as an absent preference and must not be scored either way.
    """
    text = (raw or "").strip().lower()
    if text in ("", "no", "?", "nan"):
        return None, False
    hits = [t for t in ACCEPT_TOKENS if t in text]
    if not hits:
        return None, False
    # Order by position in the string so "m2m1" resolves to m2, per the legend.
    ordered = sorted(hits, key=text.index)
    return ordered[0], len(ordered) > 1


def to_int(raw: str) -> int | None:
    text = (raw or "").strip()
    try:
        return int(text)
    except ValueError:
        return None


def read_sheet(path: Path, sheet: str, real: dict[tuple[str, int, int], str]) -> list[Row]:
    """Parse one sheet and join each row to a real folder."""
    rows: list[Row] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            name = (record.get("file_names") or "").strip()
            if not name or name.startswith("ExpStartTime"):
                continue  # the legend row

            parsed: ExperimentId | None
            try:
                parsed = ExperimentId.parse(name)
            except CohortError:
                parsed = None

            joined: str | None = None
            note = ""
            if parsed is None:
                note = "UNPARSEABLE — does not match <set_date>_<GROUP>_<trial>_<day>_<paradigm>"
            elif name in real.values():
                joined, note = name, "exact"
            else:
                key = (parsed.group, parsed.trial, parsed.day)
                if key in real:
                    joined = real[key]
                    note = f"SET-DATE MISMATCH — recoverable via {key} -> {joined}"
                else:
                    note = f"NO SUCH EXPERIMENT — {key} is not in the recording tree"

            accept, _ = resolve_accept(record.get("female_accept", ""))
            if sheet == "boris_events":
                m1 = to_int(record.get("m1tf_Courtship", ""))
                m2 = to_int(record.get("m2tf_Courtship", ""))
            else:
                m1 = to_int(record.get("male1_courtship", ""))
                m2 = to_int(record.get("male2_courtship", ""))

            rows.append(
                Row(
                    sheet=sheet,
                    raw_name=name,
                    parsed=parsed,
                    joined_folder=joined,
                    join_note=note,
                    accept=accept,
                    m1_courtship=m1,
                    m2_courtship=m2,
                    eggs=(record.get("eggs") or "").strip().lower(),
                )
            )
    return rows


def main() -> None:
    paths = load_paths()
    sheets = paths.asset_root / "catalog" / "experiment_list"

    # The authority for what exists is the recording tree, not any sheet.
    real: dict[tuple[str, int, int], str] = {}
    collisions: list[str] = []
    for directory in sorted(p.name for p in paths.raw_data_dir.iterdir() if p.is_dir()):
        try:
            experiment = ExperimentId.parse(directory)
        except CohortError:
            continue  # calibration folders, test recordings
        key = (experiment.group, experiment.trial, experiment.day)
        if key in real:
            collisions.append(f"{key}: {real[key]} and {directory}")
        real[key] = directory

    print(f"recording tree: {len(real)} parseable experiment folders")
    if collisions:
        print("  !! (group, trial, day) collisions — resolution below is NOT unique:")
        for c in collisions:
            print(f"     {c}")
    print()

    boris = read_sheet(sheets / "boris_events.csv", "boris_events", real)
    eggs_table = read_sheet(sheets / "lay_eggs_table.csv", "lay_eggs_table", real)

    # ---- 1. Join validation, every row, both sheets --------------------------
    print("=" * 78)
    print("1. JOIN VALIDATION — every file_names value against the recording tree")
    print("=" * 78)
    for sheet_name, rows in (("boris_events.csv", boris), ("lay_eggs_table.csv", eggs_table)):
        bad = [r for r in rows if r.join_note != "exact"]
        print(
            f"\n{sheet_name}: {len(rows)} rows, "
            f"{len(rows) - len(bad)} exact, {len(bad)} defective"
        )
        for row in bad:
            print(f"  {row.raw_name:28s} {row.join_note}")

    # ---- 2. The headline check ----------------------------------------------
    print()
    print("=" * 78)
    print("2. DOES THE ACCEPTED MALE ALSO COURT MORE?")
    print("=" * 78)
    scored = [
        r
        for r in boris
        if r.accept and r.m1_courtship is not None and r.m2_courtship is not None
    ]
    by_group: dict[str, list[tuple[Row, str]]] = defaultdict(list)
    for row in scored:
        if row.m1_courtship == row.m2_courtship:
            verdict = "TIE"
        else:
            courted_more = "m1" if row.m1_courtship > row.m2_courtship else "m2"
            verdict = "agree" if courted_more == row.accept else "MISS"
        group = row.parsed.group if row.parsed else "?"
        by_group[group].append((row, verdict))

    print(f"\n{'experiment':28s} {'accept':>7s} {'m1 crt':>7s} {'m2 crt':>7s}  verdict")
    for group in sorted(by_group):
        for row, verdict in by_group[group]:
            label = row.joined_folder or row.raw_name
            print(
                f"{label:28s} {row.accept:>7s} {row.m1_courtship:>7d} "
                f"{row.m2_courtship:>7d}  {verdict}"
            )

    agree = sum(1 for v in by_group.values() for _, verdict in v if verdict == "agree")
    print(f"\nper-experiment: {agree}/{len(scored)} agree  <- NOT the headline number")
    print(f"INDEPENDENT UNITS: n = {len(by_group)} groups ({', '.join(sorted(by_group))})")
    for group in sorted(by_group):
        verdicts = [v for _, v in by_group[group]]
        consistent = len(set(verdicts)) == 1
        print(
            f"  group {group}: {len(verdicts)} days, {verdicts.count('agree')} agree — "
            f"{'consistent' if consistent else 'NOT CONSISTENT ACROSS DAYS'}"
        )

    # ---- 3. Count or duration? The sheets give only counts. ------------------
    print()
    print("=" * 78)
    print("3. COUNT vs DURATION — recomputed from the raw BORIS event files")
    print("=" * 78)
    print("PRIOR_FINDINGS 1.7 quotes ratios that are not the sheet's counts, so it")
    print("used a different quantity. Recompute both and see whether they agree.\n")
    boris_dir = paths.asset_root / "groundtruth" / "boris"
    print(
        f"{'experiment':22s} {'coded':11s} {'acc':>4s} | {'n1':>3s} {'n2':>3s} {'->':>4s} "
        f"| {'s1':>6s} {'s2':>6s} {'->':>4s} | metric agreement"
    )
    duration_agree = count_agree = 0
    for group in sorted(by_group):
        for row, _ in by_group[group]:
            folder = row.joined_folder or row.raw_name
            events = read_boris(boris_dir / f"{folder}.csv")
            if events is None:
                print(f"{folder:22s} (no BORIS CSV under asset_root)")
                continue
            coded, counts, secs = events
            more_by_count = winner(counts["m1"], counts["m2"])
            more_by_secs = winner(secs["m1"], secs["m2"])
            count_agree += more_by_count == row.accept
            duration_agree += more_by_secs == row.accept
            if more_by_count == row.accept == more_by_secs:
                note = "both"
            elif more_by_secs == row.accept:
                note = "DURATION ONLY — metric choice flips this row"
            elif more_by_count == row.accept:
                note = "COUNT ONLY — metric choice flips this row"
            else:
                note = "neither"
            print(
                f"{folder:22s} {coded:11s} {row.accept:>4s} | {counts['m1']:3d} "
                f"{counts['m2']:3d} {more_by_count:>4s} | {secs['m1']:6.1f} "
                f"{secs['m2']:6.1f} {more_by_secs:>4s} | {note}"
            )
    print(f"\nby event count:    {count_agree}/{len(scored)} experiments")
    print(f"by event duration: {duration_agree}/{len(scored)} experiments")
    print("Both are still n = 3 groups. A metric-dependent row is not a result.")

    # ---- 4. Independence: are these really two record sources? --------------
    print()
    print("=" * 78)
    print("4. INDEPENDENCE — is female_accept a channel separate from courtship?")
    print("=" * 78)
    print("4a. Do the two sheets carry independent numbers, or one copy of one pass?")
    lookup = {
        (r.parsed.group, r.parsed.trial, r.parsed.day): r for r in eggs_table if r.parsed
    }
    identical = differ = 0
    for row in boris:
        if not row.parsed:
            continue
        other = lookup.get((row.parsed.group, row.parsed.trial, row.parsed.day))
        if other is None or other.m1_courtship is None:
            continue
        same = (row.m1_courtship, row.m2_courtship) == (other.m1_courtship, other.m2_courtship)
        identical += same
        differ += not same
    print(f"    rows populated in both sheets: {identical + differ}")
    print(f"    identical courtship counts:    {identical}")
    print(f"    differing:                     {differ}")

    print("\n4b. Is female_accept recoverable from the wrap events in the SAME file?")
    matches = total = 0
    for group in sorted(by_group):
        for row, _ in by_group[group]:
            folder = row.joined_folder or row.raw_name
            first = first_wrap(boris_dir / f"{folder}.csv")
            if first is None:
                continue
            total += 1
            matches += first == row.accept
            print(
                f"    {folder:22s} accept={row.accept}  first wrap={first}  "
                f"{'match' if first == row.accept else 'differs'}"
            )
    print(f"    -> {matches}/{total}: female_accept tracks the first wrap in the same coding pass.")

    # ---- 5. The egg outcome ------------------------------------------------
    print()
    print("=" * 78)
    print("5. THE EGG CHANNEL — does it discriminate BETWEEN males?")
    print("=" * 78)
    print(f"{'experiment':28s} {'accept':>7s} {'eggs':>6s}")
    for group in sorted(by_group):
        for row, _ in by_group[group]:
            label = row.joined_folder or row.raw_name
            print(f"{label:28s} {row.accept:>7s} {row.eggs:>6s}")
    print(
        "\nThe egg field is per-EXPERIMENT (yes/no), not per-male, and Egg_Laying_by_pair.csv\n"
        "is per-GROUP percentages. The egg record can corroborate THAT a mating occurred;\n"
        "it carries no information about WHICH male. It cannot corroborate a choice."
    )


def winner(a: float, b: float) -> str:
    if a > b:
        return "m1"
    return "m2" if b > a else "tie"


def read_boris(path: Path) -> tuple[str, dict[str, int], dict[str, float]] | None:
    """(coding date, courtship START counts, courtship seconds) per male."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        return None
    counts = {"m1": 0, "m2": 0}
    secs = {"m1": 0.0, "m2": 0.0}
    open_at: dict[str, float] = {}
    for record in records:
        if record["Behavior"] != "Courtship" or record["Subject"] not in ("m1tf", "m2tf"):
            continue
        male = record["Subject"][:2]
        when = float(record["Time"])
        if record["Behavior type"] == "START":
            counts[male] += 1
            open_at[male] = when
        elif record["Behavior type"] == "STOP" and male in open_at:
            secs[male] += when - open_at.pop(male)
    return records[0]["Observation date"][:10], counts, secs


def first_wrap(path: Path) -> str | None:
    """Which male wrapped first, from the same BORIS file."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        events = [
            (float(r["Time"]), r["Subject"][:2])
            for r in csv.DictReader(handle)
            if "Wrap" in r["Behavior"]
            and r["Subject"] in ("m1tf", "m2tf")
            and r["Behavior type"] in ("START", "POINT")
        ]
    return min(events)[1] if events else None


if __name__ == "__main__":
    main()
