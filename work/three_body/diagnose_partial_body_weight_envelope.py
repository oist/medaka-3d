"""Exact affine envelopes of fixed saved candidate costs; no scoring or tracking.

Read only the eleven reranker output archives and their sealed JSON descriptors.
Binary64 values become exact Fractions. The intercept is the exact mathematical
sum of parent and base costs; evaluating a live floating expression near a
breakpoint may round differently. This is a reachability diagnostic, not tuning.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK / "partial_body_saved_bank_comparison_v1"
SEAL_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
MODES = ("row_only", "row_plus_mask")
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
CASES = [(a, f) for f in (82720, 82727, 82728, 82750) for a in ARMS] + [
    (ARMS[0], f) for f in (90141, 90258, 90259)
]
FOCUS = {
    (ARMS[0], 82727): (16, "Root visually preferred prior body candidate; not ID truth"),
    (ARMS[1], 82727): (7, "Root visually mixed prior body candidate; not ID truth"),
    (ARMS[1], 82728): (15, "saved full-horizon backtrace candidate, not a local minimum"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fraction(value):
    return Fraction.from_float(float(value))


def descriptor(value):
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal_approx": float(value),
    }


def winners(lines, weight):
    values = [(idx, intercept + weight * slope) for idx, intercept, slope in lines]
    best = min(cost for _, cost in values)
    return [idx for idx, cost in values if cost == best]


def lower_envelope(lines):
    """All finite w >= 0 optima, including point-only ties and identical lines.

    The full pair-intersection arrangement partitions the nonnegative axis.
    A midpoint oracle is exact within each open cell. Remove a boundary only
    when both neighboring cells and the boundary itself have identical winners.
    """
    if not lines or len({x[0] for x in lines}) != len(lines):
        raise ValueError("nonempty uniquely indexed finite lines required")
    if any(
        not isinstance(a, Fraction) or not isinstance(b, Fraction) or b < 0 for _, a, b in lines
    ):
        raise ValueError("exact rational intercepts and nonnegative slopes required")
    cuts = set()
    for (_, a, b), (_, c, d) in itertools.combinations(lines, 2):
        if b != d:
            crossing = (c - a) / (b - d)
            if crossing > 0:
                cuts.add(crossing)
    ordered = [Fraction(0), *sorted(cuts)]
    cell_winners = []
    for pos, left in enumerate(ordered):
        right = ordered[pos + 1] if pos + 1 < len(ordered) else None
        probe = (left + right) / 2 if right is not None else left + 1
        cell_winners.append(winners(lines, probe))
    retained = []
    for pos, weight in enumerate(ordered[1:], start=1):
        on_point = winners(lines, weight)
        if on_point != cell_winners[pos - 1] or on_point != cell_winners[pos]:
            retained.append((weight, on_point))
    bounds = [Fraction(0), *(w for w, _ in retained)]
    intervals = []
    for pos, left in enumerate(bounds):
        right = bounds[pos + 1] if pos + 1 < len(bounds) else None
        probe = (left + right) / 2 if right is not None else left + 1
        intervals.append(
            {
                "lower": descriptor(left),
                "upper": descriptor(right) if right is not None else None,
                "minimum_indices": winners(lines, probe),
            }
        )
    zero = winners(lines, Fraction(0))
    positive = set().union(*(set(x["minimum_indices"]) for x in intervals))
    positive.update(idx for _, ids in retained for idx in ids)
    return {
        "zero_weight_minimum_indices": zero,
        "positive_breakpoints": [
            {"weight": descriptor(w), "minimum_indices": ids} for w, ids in retained
        ],
        "intervals": intervals,
        "ever_optimal_indices": sorted(positive | set(zero)),
        "positive_weight_optimal_indices": sorted(positive),
        "zero_weight_only_indices": sorted(set(zero) - positive),
        "pairwise_positive_intersections_examined": len(cuts),
        "interval_endpoints_excluded": True,
        "upper_null_means_unbounded": True,
    }


def synthetic_checks():
    f = Fraction
    # Point-only line 5 touches w=1; identical 1/4 stay tied on an interval.
    lines = [
        (0, f(0), f(3)),
        (1, f(2), f(1)),
        (2, f(6), f(0)),
        (3, f(4), f(2)),
        (4, f(2), f(1)),
        (5, f(1), f(2)),
    ]
    e = lower_envelope(lines)
    assert [x["weight"]["decimal_approx"] for x in e["positive_breakpoints"]] == [1.0, 4.0]
    assert [x["minimum_indices"] for x in e["intervals"]] == [[0], [1, 4], [2]]
    assert e["positive_breakpoints"][0]["minimum_indices"] == [0, 1, 4, 5]
    assert e["positive_breakpoints"][1]["minimum_indices"] == [1, 2, 4]
    assert e["ever_optimal_indices"] == [0, 1, 2, 4, 5]
    zero = lower_envelope([(0, f(1), f(3)), (1, f(1), f(2))])
    assert zero["zero_weight_only_indices"] == [0]
    assert zero["intervals"][0]["minimum_indices"] == [1]
    small = f(1, 10**30)
    tiny = lower_envelope([(0, f(0), f(1)), (1, small, f(0))])
    assert tiny["positive_breakpoints"][0]["weight"] == descriptor(small)
    return 3


def signature(array):
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha(np.ascontiguousarray(array).tobytes()),
    }


def load_costs(seal, arm, frame, captures):
    stem = f"cases/{arm}_{frame}"
    saved = {}
    for ext in ("json", "npz"):
        name = stem + "." + ext
        record = seal["outputs"][name]
        path = (ROOT / name).resolve()
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        assert path.is_relative_to(ROOT) and str(path) == record["path"]
        assert len(data) == record["size_bytes"] and sha(data) == record["sha256"]
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        captures[name] = {**record, "mtime_ns_at_capture": after.st_mtime_ns}
        saved[ext] = data
    doc = json.loads(saved["json"])
    assert doc["archive"] == seal["outputs"][stem + ".npz"]
    tree = doc["tree"]
    assert tree["arm"] == arm and tree["frame_idx"] == frame
    assert tree["parent_history_fixed"] and not tree["new_temporal_backtrace"]
    assert not tree["reoptimized_fit"] and not tree["physical_ID_truth"]
    descriptors = {}

    def collect(value):
        if isinstance(value, dict) and "__array__" in value:
            assert set(value) == {"__array__", "signature"}
            key = value["__array__"]
            assert key not in descriptors
            descriptors[key] = value["signature"]
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(tree)
    decoded = {}
    with np.load(io.BytesIO(saved["npz"]), allow_pickle=False) as archive:
        assert len(archive.files) == len(set(archive.files))
        assert set(archive.files) == set(descriptors)

        def get(value, label):
            assert set(value) == {"__array__", "signature"}
            array = archive[value["__array__"]]
            assert not array.dtype.hasobject and signature(array) == value["signature"]
            decoded[label] = value["signature"]
            return array

        bank = tree["baseline_bank"]
        costs = {
            name: get(bank[name], "baseline_bank/" + name)
            for name in (
                "candidate_index",
                "candidate_parent",
                "base_local_cost",
                "parent_path_cost",
                "sixth_instance_term",
                "adjusted_local_cost",
                "adjusted_path_cost",
            )
        }
        costs["candidate_kind"] = bank["candidate_kind"]
        costs["modes"] = {}
        for mode in MODES:
            m = tree["modes"][mode]
            assert m["normalizer"] == 15.0 and m["weight_per_view"] == 1.0
            costs["modes"][mode] = {
                "new_term": get(m["sixth_term"], mode + "/sixth_term"),
                "saved_w1_path": get(m["path_cost"], mode + "/path_cost"),
            }
        final = tree["final_selected_match"]
        costs["final_pose_indices"] = (
            get(final["exact_pose_candidate_indices"], "final_pose_indices").tolist()
            if final["available"]
            else None
        )
    costs["decoded_signatures"] = decoded
    return costs


def analyze(costs, arm, frame):
    indices = costs["candidate_index"]
    count = len(indices)
    assert np.array_equal(indices, np.arange(count))
    for key in (
        "base_local_cost",
        "parent_path_cost",
        "sixth_instance_term",
        "adjusted_local_cost",
        "adjusted_path_cost",
    ):
        assert costs[key].shape == (count,) and costs[key].dtype == np.float64
        assert np.isfinite(costs[key]).all()
    assert np.array_equal(
        costs["base_local_cost"] + costs["sixth_instance_term"], costs["adjusted_local_cost"]
    )
    assert np.array_equal(
        costs["parent_path_cost"] + costs["adjusted_local_cost"], costs["adjusted_path_cost"]
    )
    candidates = [
        {
            "candidate_index": i,
            "candidate_kind": costs["candidate_kind"][i],
            "candidate_parent": int(costs["candidate_parent"][i]),
            "base_local_cost": float(costs["base_local_cost"][i]),
            "parent_path_cost": float(costs["parent_path_cost"][i]),
            "original_sixth_term": float(costs["sixth_instance_term"][i]),
            "original_adjusted_local_cost": float(costs["adjusted_local_cost"][i]),
            "original_adjusted_path_cost": float(costs["adjusted_path_cost"][i]),
            **{
                mode + "_term": (
                    float(costs["modes"][mode]["new_term"][i])
                    if np.isfinite(costs["modes"][mode]["new_term"][i])
                    else None
                )
                for mode in MODES
            },
        }
        for i in range(count)
    ]
    modes = {}
    for mode in MODES:
        slope = costs["modes"][mode]["new_term"]
        assert slope.shape == (count,) and not np.isnan(slope).any() and np.all(slope >= 0)
        finite = np.flatnonzero(np.isfinite(slope)).tolist()
        assert finite
        lines = [
            (
                i,
                fraction(costs["base_local_cost"][i]) + fraction(costs["parent_path_cost"][i]),
                fraction(slope[i]),
            )
            for i in finite
        ]
        envelope = lower_envelope(lines)
        minimum = min(slope[finite])
        factor_winners = [i for i in finite if slope[i] == minimum]
        saved = costs["modes"][mode]["saved_w1_path"]
        expected = costs["parent_path_cost"] + (costs["base_local_cost"] + slope)
        assert np.array_equal(saved, expected)
        saved_winners = np.flatnonzero(saved == saved.min()).tolist()
        exact_winners = winners(lines, Fraction(1))
        analysis = {
            "finite_candidate_indices": finite,
            "excluded_nonfinite_candidate_indices": [i for i in range(count) if i not in finite],
            "factor_only_minimum_value": float(minimum),
            "factor_only_minimum_indices": factor_winners,
            "factor_only_minimum_candidates": [candidates[i] for i in factor_winners],
            "weight_one_exact_minimum_indices": exact_winners,
            "weight_one_saved_path_minimum_indices": saved_winners,
            "weight_one_exact_and_saved_minimum_sets_equal": exact_winners == saved_winners,
            "envelope": envelope,
        }
        if (arm, frame) in FOCUS:
            idx, label = FOCUS[arm, frame]
            assert 0 <= idx < count
            analysis["focus"] = {
                "candidate_index": idx,
                "selection_context": label,
                "candidate_metadata": candidates[idx],
                "factor_rank_competition_1_based": (
                    1 + int(np.count_nonzero(slope[finite] < slope[idx])) if idx in finite else None
                ),
                "factor_equal_indices": [i for i in finite if slope[i] == slope[idx]],
                "ever_optimal_nonnegative_weight": idx in envelope["ever_optimal_indices"],
                "ever_optimal_positive_weight": idx in envelope["positive_weight_optimal_indices"],
                "listed_in_saved_final_pose_matches": (
                    idx in costs["final_pose_indices"]
                    if costs["final_pose_indices"] is not None
                    else None
                ),
            }
        modes[mode] = analysis
    return {
        "arm": arm,
        "frame_idx": frame,
        "candidate_count": count,
        "baseline_path_minimum_indices": np.flatnonzero(
            costs["adjusted_path_cost"] == costs["adjusted_path_cost"].min()
        ).tolist(),
        "saved_full_horizon_final_pose_candidate_indices": costs["final_pose_indices"],
        "decoded_output_array_signatures": costs["decoded_signatures"],
        "candidates": candidates,
        "modes": modes,
    }


def write_markdown(report, path):
    rows = [
        "# Partial-body factor: weight reachability of saved candidates",
        "",
        "This diagnoses which candidates can win when only the new factor weight changes. "
        "It does not select a weight, refit a body, or reconstruct a new trajectory. "
        "Each candidate keeps its own original parent history.",
        "",
        "The cost is `parent + base + w * new_term`, with the old sixth term removed. "
        "`new_term` is the saved sum over three views, using the same divisor 15 for both "
        "row-only and row+mask. A common w therefore scales the whole new factor; it is "
        "not a per-frame or per-camera choice. Absolute costs across arms are incomparable.",
        "",
        "Exact binary64 coefficients are converted to rational numbers. Parent and base "
        "are added exactly; all pair intersections and ties are evaluated exactly. "
        "This describes the mathematical affine model, not floating-rounding behavior "
        "at a boundary. Positive intervals are open; breakpoint ties and w=0 are separate.",
        "",
        "| Arm / frame | Mode | Factor-only minimum | Winners on open positive intervals | "
        "Positive breakpoints (approx.) |",
        "|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        for mode, result in case["modes"].items():
            env = result["envelope"]
            intervals = " → ".join(str(x["minimum_indices"]) for x in env["intervals"])
            cuts = (
                ", ".join(
                    f"{x['weight']['decimal_approx']:.8g}:{x['minimum_indices']}"
                    for x in env["positive_breakpoints"]
                )
                or "none"
            )
            arm = "Sparse" if case["arm"] == ARMS[0] else "Depth"
            rows.append(
                f"| {arm} {case['frame_idx']} | {mode} | "
                f"{result['factor_only_minimum_indices']} | {intervals} | {cuts} |"
            )
    rows.extend(["", "Focused candidates (rank 1 means a minimum of the new factor alone):", ""])
    for case in report["cases"]:
        for mode, result in case["modes"].items():
            if "focus" not in result:
                continue
            focus = result["focus"]
            rows.append(
                f"- {case['arm']} {case['frame_idx']} candidate "
                f"{focus['candidate_index']}, {mode}: factor rank "
                f"{focus['factor_rank_competition_1_based']}; reachable at a positive "
                f"weight = {focus['ever_optimal_positive_weight']}. "
                f"Context: {focus['selection_context']}."
            )
    mismatches = [
        (c["arm"], c["frame_idx"], m)
        for c in report["cases"]
        for m, r in c["modes"].items()
        if not r["weight_one_exact_and_saved_minimum_sets_equal"]
    ]
    rows.extend(
        [
            "",
            f"At w=1, exact-versus-saved winner-set differences: {mismatches}.",
            "",
            "The JSON retains exact breakpoint fractions, all point ties, all candidate "
            "costs/kinds/parents, factor-only minima and full-horizon final-pose references. "
            "A final backtrace candidate need not be a local/fixed-parent-path winner. "
            "Point-only ties are reachable but do not provide a robust weight interval.",
            "",
            "The earlier visual regression still rejects the tested initial setting. "
            "Envelope reachability alone does not establish body quality or physical IDs. "
            "The three negative frames use Sparse banks with OldFast observation pictures; "
            "they are not the old OldFast candidate pool.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows))


def main():
    output = WORK / "PARTIAL_BODY_WEIGHT_ENVELOPE_V1.json"
    markdown = WORK / "PARTIAL_BODY_WEIGHT_ENVELOPE_V1.md"
    assert not output.exists() and not markdown.exists()
    checks = synthetic_checks()
    seal_path = ROOT / "comparison.provenance.json"
    seal_bytes = seal_path.read_bytes()
    assert sha(seal_bytes) == SEAL_SHA
    seal = json.loads(seal_bytes)
    assert seal["status"] == "ok" and seal["stage"] == ROOT.name
    captures = {}
    results = [analyze(load_costs(seal, arm, frame, captures), arm, frame) for arm, frame in CASES]
    for record in captures.values():
        stat = Path(record["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (
            record["size_bytes"],
            record["mtime_ns_at_capture"],
        )
    assert seal_path.read_bytes() == seal_bytes
    report = {
        "schema": "partial_body_weight_envelope_v1",
        "status": "ok",
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha(Path(__file__).read_bytes()),
        },
        "producer_seal": {
            "path": str(seal_path),
            "sha256": SEAL_SHA,
            "size_bytes": len(seal_bytes),
        },
        "synthetic_exact_envelope_checks": checks,
        "coefficient_convention": "exact Fraction(base binary64) + Fraction(parent binary64); "
        "slope = Fraction(saved new sixth-term binary64)",
        "weight_domain": "one common real w >= 0; report separately by each fixed case/mode",
        "finite_candidate_only": True,
        "infinite_slope_excluded_even_at_zero": True,
        "old_sixth_term_replaced_not_added": True,
        "common_saved_normalizer": 15.0,
        "original_parent_history_fixed": True,
        "weight_selected_or_adopted": False,
        "new_fit_or_trajectory_or_ID_created": False,
        "physical_ID_truth": False,
        "captured_output_members": captures,
        "case_count": len(results),
        "cases": results,
        "read_scope": "new reranker success seal, eleven JSON trees and eleven NPZ output "
        "archives; decode cost/index fields only; no poses or distance arrays, "
        "no original inputs or source closure reopened",
        "limits": [
            "No common weight is adopted or chosen from these development frames.",
            "Local affine costs across different arm histories are not comparable.",
            "Exact arithmetic models lines; live float evaluations may tie differently "
            "near breakpoints.",
            "Factor-only minima can remain unreachable if another line has the same "
            "slope and a lower intercept.",
            "Full-horizon final-pose references are inherited from the sealed caller; "
            "they are not independently reopened or treated as local minima.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    write_markdown(report, markdown)
    for path in (output, markdown):
        print(path.name, path.stat().st_size, sha(path.read_bytes()))
    for case in results:
        for mode, result in case["modes"].items():
            print(
                case["arm"],
                case["frame_idx"],
                mode,
                "factor",
                result["factor_only_minimum_indices"],
                "reachable",
                result["envelope"]["ever_optimal_indices"],
                "focus",
                result.get("focus", {}).get("factor_rank_competition_1_based"),
            )


if __name__ == "__main__":
    main()
