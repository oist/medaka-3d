"""Join successful minute comparisons without rerunning inference or assigning truth.

The older report is reused only after its sealed files are checked. Its persistent
geometry method must agree exactly with the new report's unchanged-geometry method.
All identity paths are reported as model assignments, never biological swap events.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import run_experiment as base  # noqa: E402
from identity_output import build_identity_output  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FRAMES = np.arange(76620, 86580, dtype=np.int64)
METRIC_KEYS = (
    "direct_motion", "accepted_motion", "direct_episodes", "accepted_episodes",
    "availability", "female_raw", "female_accepted", "review_fish_frames",
    "all_three_identity_verdict",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
    if not path.is_relative_to(REPO):
        raise ValueError("only preserved workspace evidence is accepted")
    return path


def bind(hashes, path, digest):
    path = local_path(path)
    if not isinstance(digest, str) or len(digest) != 64 or base.sha256(path) != digest:
        raise ValueError(f"sealed SHA mismatch: {path}")
    if str(path) in hashes and hashes[str(path)] != digest:
        raise ValueError("conflicting source SHA records")
    hashes[str(path)] = digest


def verify_record(root, filename, stage, required_outputs, hashes):
    """Fail before opening result values when the upstream stage is not complete."""
    root = local_path(root)
    seal = root / filename
    record = read_json(seal)
    if record.get("status") != "ok" or record.get("stage") != stage:
        raise ValueError("upstream stage has not completed successfully")
    bind(hashes, seal, base.sha256(seal))
    if local_path(record["params"]["output"]) != root:
        raise ValueError("stage output directory is not bound to its seal")
    for role, name in required_outputs.items():
        entry = record["outputs"].get(role)
        if entry is None or local_path(entry["path"]) != root / name:
            raise ValueError(f"missing required sealed output: {role}")
    for entry in record["outputs"].values():
        path = local_path(entry["path"])
        if not path.is_relative_to(root):
            raise ValueError("output escaped its sealed directory")
        bind(hashes, path, entry["sha256"])
    for entry in record["inputs"].values():
        bind(hashes, entry["path"], entry["sha256"])
    return record


def exact_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def check_metric_bridge(old, new):
    if old["frames"] != [76620, 86580] or new["frames"] != old["frames"]:
        raise ValueError("comparisons do not share the registered native minute")
    if old["fps"] != 166 or new["native_fps"] != 166 or new["native_frames"] != 9960:
        raise ValueError("native cadence differs")
    original = old["methods"]["current_variant"]
    frozen = new["methods"]["frozen_geometry_ids"]
    for key in METRIC_KEYS:
        if not exact_json(original[key], frozen[key]):
            raise ValueError(f"shared geometry common metric differs: {key}")
    return True


def compact_method(method, scope):
    row = {key: method[key] for key in ("female_raw", "female_accepted")}
    for mode, key in (("direct", "direct_motion"), ("accepted", "accepted_motion")):
        values = method[key]
        denominator = values["total_animal_frames"]
        numerator = values["observed_animal_frames"]
        if denominator != 29880 or not 0 <= numerator <= denominator:
            raise ValueError("availability denominator is not all native fish-frames")
        if values["observed_fraction"] != numerator / denominator:
            raise ValueError("availability fraction differs from recorded numerator")
        row[mode] = dict(
            fish_frames=numerator, total_fish_frames=denominator,
            fraction=numerator / denominator,
            all_three_frames=values["all_three_observed_frames"],
        )
    for key in ("female_raw", "female_accepted"):
        value = row[key]
        count = value["interval_counts"]
        if set(count) != {"preserved", "swap", "unscored"}:
            raise ValueError("female outcomes must include unscored intervals")
        if sum(count.values()) != value["requested_intervals"]:
            raise ValueError("female interval counts omit requested intervals")
        row[key] = {
            **value,
            "display_interpretation": (
                "P=equal endpoint labels; S=different endpoint labels; U=unscored"
            ),
            "swap_events_inferred": False,
        }
    row["review_fish_frames"] = method["review_fish_frames"]
    row["review_scope"] = scope
    return row


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def exact_array(actual, expected, label):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes(order="C") != expected.tobytes(order="C")):
        raise ValueError(f"array is not byte-exact: {label}")


def check_export(geometry, exported):
    expected = build_identity_output(geometry, exported["identity_to_raw_slot"])
    if set(exported) != set(expected):
        raise ValueError("identity export schema differs")
    for key in expected:
        exact_array(exported[key], expected[key], key)
    raw_flags = geometry["identity_review"]
    flags = exported["prior_review_for_proposed_identity"]
    exact_array(flags.sum(1), raw_flags.sum(1), "per-frame old review conservation")
    return dict(
        raw_review_fish_frames=int(raw_flags.sum()),
        proposed_review_fish_frames=int(flags.sum()),
        raw_review_by_slot=raw_flags.sum(0).tolist(),
        reordered_review_by_identity=flags.sum(0).tolist(),
        old_flags_cleared=0,
        original_arrays_and_export_byte_exact=True,
    )


def path_differences(frames, color, motion):
    for mapping in (color, motion):
        if mapping.shape != (len(frames), 3) or not np.all(np.sort(mapping, axis=1) == range(3)):
            raise ValueError("identity path rows must be complete permutations")
    different = color != motion
    split = np.r_[0, np.flatnonzero(
        np.any(color[1:] != color[:-1], axis=1)
        | np.any(motion[1:] != motion[:-1], axis=1)
    ) + 1, len(frames)]
    intervals = []
    for start, stop in itertools.pairwise(split):
        if different[start].any():
            intervals.append(dict(
                start=int(frames[start]), stop=int(frames[stop - 1] + 1),
                frames=int(stop - start), changed_ids=np.flatnonzero(different[start]).tolist(),
                color_identity_to_raw_slot=color[start].tolist(),
                motion_identity_to_raw_slot=motion[start].tolist(),
            ))
    transitions = {}
    for label, mapping in (("color", color), ("motion_only", motion)):
        rows = np.flatnonzero(np.any(mapping[1:] != mapping[:-1], axis=1)) + 1
        transitions[label] = [dict(
            frame=int(frames[t]), before=mapping[t - 1].tolist(), after=mapping[t].tolist(),
            changed_ids=np.flatnonzero(mapping[t] != mapping[t - 1]).tolist(),
        ) for t in rows]
    return dict(
        different_frames=int(different.any(1).sum()),
        different_identity_frames=int(different.sum()),
        different_frames_by_identity=different.sum(0).tolist(),
        intervals=intervals, transitions=transitions,
        mapping_convention="mapping[identity]=raw geometry slot; stop is exclusive",
        biological_swap_event_count=None,
    ), different


def write_report(path, report):
    lines = [
        "# 封印済み 1 分間の比較", "",
        "P/S/U はメスの注釈端点での ID 維持/相違/未評価です。連続的な入替回数ではありません。",
        "", "| 方式 | 直接支持率 | 受理率 | メス raw P/S/U | メス受理 P/S/U |",
        "|---|---:|---:|---|---|",
    ]
    for name, values in report["methods"].items():
        counts = ["/".join(str(values[k]["interval_counts"][outcome])
                           for outcome in ("preserved", "swap", "unscored"))
                  for k in ("female_raw", "female_accepted")]
        lines.append(
            f"| {name} | {values['direct']['fraction']:.2%} | "
            f"{values['accepted']['fraction']:.2%} | {counts[0]} | {counts[1]} |"
        )
    delta = report["appearance_path_difference"]
    lines += ["", f"色あり/なしの対応差は {delta['different_frames']} フレーム、"
              f"ID ごとには {delta['different_frames_by_identity']} フレームです。",
              "候補の対応変更は生物の ID スワップを直接示しません。", ""]
    lines += ["- " + value for value in report["limitations"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--previous-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists():
        raise FileExistsError("a new workspace output directory is required")
    output.mkdir(parents=True)
    with provenance("sealed_memory_minute_comparison_join", output, name="comparison") as record:
        record.params_from(args)
        code_paths = {Path(__file__).resolve(), LAB / "identity_output.py",
                      LAB / "run_experiment.py", REPO / "src/medaka3d/provenance.py"}
        hashes = {str(p): base.sha256(p) for p in code_paths}
        run, newroot, oldroot = map(
            local_path, (args.run, args.comparison, args.previous_comparison)
        )
        newseal = verify_record(
            newroot, "evaluation.provenance.json",
            "three_body_identity_memory_post_inference_evaluation",
            {"comparison": "comparison.json", "motion_only_identity": "motion_only_identity.npz"},
            hashes,
        )
        if (local_path(newseal["params"]["run"]) != run
                or newseal["params"].get("both_memory_objectives_solved_before_gt") is not True
                or newseal["params"].get("final_inputs_sources_snapshots_outputs_unchanged")
                is not True):
            raise ValueError("memory evaluation lacks objective and evidence completion guards")
        producer = verify_record(
            run, "experiment.provenance.json", "three_body_fixed_geometry_identity_memory_dp",
            {"summary": "summary.json", "identity": "identity.npz"}, hashes,
        )
        oldseal = verify_record(
            oldroot, "evaluation.provenance.json", "three_body_variant_minute_comparison",
            {"comparison": "comparison.json"}, hashes,
        )
        old, new = read_json(oldroot / "comparison.json"), read_json(newroot / "comparison.json")
        check_metric_bridge(old, new)
        geometry_root = local_path(producer["params"]["geometry_run"])
        if (local_path(oldseal["params"]["run"]) != geometry_root
                or local_path(new["geometry_run"]) != geometry_root
                or local_path(old["current_run_path"]) != geometry_root
                or local_path(new["identity_run"]) != run):
            raise ValueError("comparison geometry is not the same sealed run")
        geometry_path = geometry_root / "candidate.npz"
        digest = hashes[str(geometry_path)]
        if old["current_candidate_sha256"] != digest or new["geometry_npz_sha256"] != digest:
            raise ValueError("geometry bridge candidate SHA differs")
        geo_summary = read_json(geometry_root / "summary.json")
        geo_record = read_json(geometry_root / "experiment.provenance.json")
        input_root = local_path(geo_record["params"]["input_root"])
        if input_root != local_path(oldseal["params"]["input_root"]):
            raise ValueError("comparison input roots differ")
        initial, seed, _ = base.registered_seed(76620)
        if not exact_json(geo_summary["seed"], seed):
            raise ValueError("shared geometry seed differs from the registered native first frame")
        shared_evidence = {}
        for entry in oldseal["inputs"].values():
            path = local_path(entry["path"])
            if path.is_relative_to(input_root) or path == local_path(seed["registration_path"]):
                if hashes[str(path)] != entry["sha256"]:
                    raise ValueError("shared calibration/SLEAP/registration SHA differs")
                shared_evidence[str(path)] = entry["sha256"]
        # These paths must be present in the NEW evaluation, not just our joined hash dictionary.
        newinputs = {str(local_path(x["path"])): x["sha256"] for x in newseal["inputs"].values()}
        sleap_paths = {
            str(input_root / "work/c01_s16_predictions/bottomup" / f"{base.EXPERIMENT}__{view}.npz")
            for view in base.VIEWS
        }
        required = {p: h for p, h in shared_evidence.items()
                    if "calibration" in Path(p).parts or p in sleap_paths
                    or p == str(local_path(seed["registration_path"]))}
        if not any("calibration" in Path(p).parts for p in required):
            raise ValueError("no common calibration evidence is bound")
        if not sleap_paths.issubset(required):
            raise ValueError("no common SLEAP evidence is bound")
        if any(newinputs.get(p) != h for p, h in required.items()):
            raise ValueError("new comparison lacks the same calibration/SLEAP/seed evidence")
        geometry = npz(geometry_path)
        exact_array(geometry["frame_idx"], FRAMES, "all native frames")
        exact_array(np.asarray(geo_summary["seed"]["pose_xyz"]), initial, "registered seed pose")
        color, motion = npz(run / "identity.npz"), npz(newroot / "motion_only_identity.npz")
        preservation = {name: check_export(geometry, value)
                        for name, value in (("color", color), ("motion_only", motion))}
        delta, different = path_differences(
            FRAMES, color["identity_to_raw_slot"], motion["identity_to_raw_slot"]
        )
        methods = {name: compact_method(value, value["review_scope"])
                   for name, value in old["methods"].items()}
        for name in ("memory_color_ids", "memory_motion_only_ids"):
            methods[name] = compact_method(
                new["methods"][name],
                "same prior identity review, reordered; DP margins do not clear it",
            )
        report = dict(
            status="sealed_reports_joined_with_geometry_bridge", frames=[76620, 86580],
            native_fps=166, native_frames=len(FRAMES), methods=methods,
            geometry_run=str(geometry_root), geometry_npz_sha256=digest,
            common_metric_bridge_exact=True, input_root=str(input_root),
            shared_calibration_sleap_seed_sha256=required,
            prior_review_preservation=preservation, appearance_path_difference=delta,
            raw_female_endpoint_regression_gate=new["raw_female_endpoint_regression_gate"],
            new_solution_recomputed_in_this_join=False, new_images_rendered_in_this_join=False,
            old_inputs_scope=old["old_method_input_provenance_scope"],
            limitations=[
                "3 匹すべての連続 ID 正解は未確認。P は端点の一致で、"
                "途中の交換後の復帰を検出しない。",
                "未評価 U はゼロエラーではない。S は異なる端点 ID であり、交換イベント数ではない。",
                "受理条件は方式間で異なる。旧 generative は形状と ID の combined review、"
                "C34 は同等の flag がない。",
                "メモリ DP は既存の形状・支持・review を並べ替えるだけで、"
                "欠測・重複体・誤形状を修復しない。",
                "色あり/なしの対応差はモデル依存の候補変更。"
                "色の有効性や生物 ID 正解の証明ではない。",
                "旧方式は既存の封印比較を引用する。元リモート動画の再検証や旧推論の再実行はしない。",
            ],
        )
        arrays_path = output / "path_differences.npz"
        np.savez_compressed(arrays_path, frame_idx=FRAMES,
                            color_mapping=color["identity_to_raw_slot"],
                            motion_mapping=motion["identity_to_raw_slot"], different=different)
        record.output("path_differences", arrays_path, sha256=base.sha256(arrays_path))
        csv_path = output / "path_differences.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "identity", "color_raw_slot", "motion_raw_slot"])
            for t, fish in zip(*np.nonzero(different), strict=True):
                writer.writerow([
                    int(FRAMES[t]), int(fish), int(color["identity_to_raw_slot"][t, fish]),
                    int(motion["identity_to_raw_slot"][t, fish]),
                ])
        record.output("path_csv", csv_path, sha256=base.sha256(csv_path))
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", summary_path, sha256=base.sha256(summary_path))
        report_path = output / "RESULT.md"
        write_report(report_path, report)
        record.output("report", report_path, sha256=base.sha256(report_path))
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in code_paths:
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output("code:" + str(path.relative_to(REPO)), target, sha256=hashes[str(path)])
        for path, digest in {
            **hashes, **{x["path"]: x["sha256"] for x in record.outputs.values()}
        }.items():
            if base.sha256(path) != digest:
                raise RuntimeError("comparison join input/source/output changed")
        record.param("final_inputs_sources_outputs_unchanged", True)
    print(
        json.dumps(dict(output=str(output), different_frames=delta["different_frames"])), flush=True
    )
    return report


if __name__ == "__main__":
    main()
