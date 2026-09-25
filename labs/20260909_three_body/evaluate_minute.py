"""Compare a sealed native minute with seven C34 arms and the previous method.

Only completed tracking outputs enter this CPU-only evaluation. Support and
female endpoint scores remain separate; no male or dense identity truth is
created. The older method's combined review mask is preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from diagnostics import diagnose
from observations import CachedNodes
from review_events import extract_review_events
from run_experiment import (
    BASELINE_DIR,
    BASELINE_LAMBDAS,
    CALIBRATION_DIR,
    EXPERIMENT,
    LAB,
    REPO,
    registered_seed,
    safe_relative,
    sha256,
    skeleton_from_sources,
    staging_catalog,
    validate_result,
    verified_input,
)

from medaka3d.calib import load_rig
from medaka3d.eval.generative import (
    match_mouth_anchors,
    observed_motion_summary,
    score_female_identity,
)
from medaka3d.eval.reacquisition import assess_availability, availability_episodes
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

FRAMES = np.arange(76620, 86580, dtype=np.int64)


def verify_run_evidence(root, summary, *, previous=False):
    """Bind evaluation to the run's actual local inputs and saved source snapshot."""
    root = Path(root).resolve(strict=True)
    record = json.loads(
        (
            root / ("benchmark.provenance.json" if previous else "experiment.provenance.json")
        ).read_text(encoding="utf-8")
    )
    hashes = {}
    for item in record["inputs"].values():
        path = Path(item["path"])
        path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
        if not path.is_relative_to(REPO.resolve()):
            raise ValueError("registered evaluation requires preserved local run inputs")
        digest = sha256(path)
        if item.get("sha256") != digest:
            raise ValueError("run input digest mismatch")
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError("conflicting recorded input digests")
        hashes[str(path)] = digest
    if previous:
        if not {"staged_manifest", "registration"}.issubset(record["inputs"]):
            raise ValueError("previous method lacks preserved input catalog/seed registration")
        _initial, seed, _evaluation = registered_seed(int(FRAMES[0]))
        window = record["params"]["registered_window"]
        if (
            window["seed_pose_xyz"] != seed["pose_xyz"]
            or window["start"] != int(FRAMES[0])
            or window["stop"] != int(FRAMES[-1] + 1)
        ):
            raise ValueError("previous method used a different first pose or native window")
        code_hashes = record["params"]["code_hashes"]
        code_hashes = {Path(path).relative_to(REPO): digest for path, digest in code_hashes.items()}
    else:
        if summary["input_sha256_before"] != summary["input_sha256_after"]:
            raise ValueError("run inputs changed during inference")
        if summary["code_sha256_before"] != summary["code_sha256_after"]:
            raise ValueError("run code changed during inference")
        for value, digest in summary["input_sha256_before"].items():
            path = Path(value)
            path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
            if hashes.get(str(path)) != digest:
                raise ValueError("summary inference input is not bound to its provenance")
        for category in ("raw", "masks", "backgrounds"):
            if any(f"{category}_{view}" not in record["inputs"] for view in VIEWS):
                raise ValueError("run is missing one of its three-view input records")
        code_hashes = summary["code_sha256_before"]
    if not code_hashes:
        raise ValueError("run has no preserved source snapshot hashes")
    for relative, digest in code_hashes.items():
        path = root / "code_snapshot" / safe_relative(relative)
        if sha256(path) != digest:
            raise ValueError("run code snapshot digest mismatch")
        hashes[str(path)] = digest
    return hashes


def require_binding(path, digest, run_hashes):
    if run_hashes.get(str(Path(path).resolve(strict=True))) != digest:
        raise ValueError("evaluation input is not the input used by the completed run")


def load_completed(root, *, previous=False):
    """Require a complete sealed run, not merely a candidate that was written."""
    root = Path(root).resolve(strict=True)
    summary_name = "comparison" if previous else "summary"
    record_path = root / ("benchmark.provenance.json" if previous else "experiment.provenance.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise ValueError("run provenance is not complete and successful")
    hashes = {str(record_path): sha256(record_path)}
    for name, path in (
        (summary_name, root / f"{summary_name}.json"),
        ("candidate", root / "candidate.npz"),
    ):
        digest = sha256(path)
        if record["outputs"][name].get("sha256") != digest:
            raise ValueError(f"sealed {name} digest mismatch")
        hashes[str(path)] = digest
    summary = json.loads((root / f"{summary_name}.json").read_text(encoding="utf-8"))
    if (
        summary.get("technical_completion") is not True
        or summary.get("experiment") != EXPERIMENT
        or summary.get("initialization_count") != 1
        or summary.get("requested_frames") != len(FRAMES)
        or summary.get("returned_frames") != len(FRAMES)
        or summary.get("frames") != [int(FRAMES[0]), int(FRAMES[-1] + 1)]
        or summary.get("code_unchanged_during_run") is not True
    ):
        raise ValueError("expected the complete single-initialization registered minute")
    if not previous and (
        summary.get("native_fps") != 166
        or summary.get("temporal_subsampling") is not False
        or summary.get("development_prefix_smoke") is not False
        or summary.get("track_invocation_count") != 1
        or summary.get("reset_count") != 0
        or summary.get("inputs_unchanged_during_run") is not True
    ):
        raise ValueError("new minute lacks native-frame/no-reset completion evidence")
    with np.load(root / "candidate.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if arrays["frame_idx"].dtype.kind not in "iu" or not np.array_equal(
        arrays["frame_idx"], FRAMES
    ):
        raise ValueError("candidate is not every registered native frame")
    return arrays, summary, hashes


def normalized_points(arrays, *, previous=False):
    """Validate actual saved support; never manufacture a mask bridge for old code."""
    frames = np.asarray(arrays["frame_idx"])
    count = len(frames)
    source = arrays["node_source" if previous else "sources"]
    review = arrays["review" if previous else "identity_review"]
    observed = arrays["observed"]
    pose = arrays["pose_xyz"]
    if (
        np.asarray(source).shape != (count, 3, 3, 3)
        or np.asarray(source).dtype.kind not in "iu"
        or np.any(source < -1)
        or np.asarray(review).shape != (count, 3)
        or review.dtype.kind != "b"
        or np.asarray(observed).shape != (count, 3, 3)
        or observed.dtype.kind != "b"
        or np.asarray(pose).shape != (count, 3, 3, 3)
    ):
        raise ValueError("invalid saved source, review, observation or pose schema")
    if np.any(observed & (~np.isfinite(pose).all(-1) | ((source >= 0).sum(2) < 2))):
        raise ValueError("observed node lacks finite coordinates or two source views")
    xyz, accepted = arrays["xyz"], arrays["accepted_xyz"]
    for value in (xyz, accepted):
        if (
            np.asarray(value).shape != (count, 3, 3)
            or np.isinf(value).any()
            or np.any(np.isfinite(value).any(-1) != np.isfinite(value).all(-1))
        ):
            raise ValueError("saved coordinates must contain whole finite or NaN points")
    np.testing.assert_array_equal(xyz, np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan))
    np.testing.assert_array_equal(accepted, np.where(review[..., None], np.nan, xyz))
    return source, review, observed, xyz, accepted


def common_metrics(frames, source, review, observed, xyz, accepted, anchors):
    return {
        "direct_motion": observed_motion_summary(xyz, frames),
        "accepted_motion": observed_motion_summary(accepted, frames),
        "direct_episodes": availability_episodes(xyz, frames),
        "accepted_episodes": availability_episodes(accepted, frames),
        "availability": assess_availability(accepted, frames),
        "female_raw": score_female_identity(frames, source[:, :, :, 0], observed[:, :, 0], anchors),
        "female_accepted": score_female_identity(
            frames, source[:, :, :, 0], observed[:, :, 0], anchors, review=review
        ),
        "review_fish_frames": int(review.sum()),
        "all_three_identity_verdict": "unverified; independent all-three truth unavailable",
    }


def validate_sources(source, frames, nodes):
    for t, frame in enumerate(frames):
        for vi, view in enumerate(VIEWS):
            rows = source[t, :, vi]
            if np.any(rows >= len(nodes.at(view, int(frame)))):
                raise ValueError("saved source row exceeds the original detector rows")
            for node in range(3):
                present = rows[:, node][rows[:, node] >= 0]
                if len(np.unique(present)) != len(present):
                    raise ValueError("one node observation was assigned to multiple fish")


def markdown_report(report):
    lines = [
        "# 再初期化なし・連続1分の比較",
        "",
        "全9960フレームを比較した技術検証です。3匹のIDの正解は未確認です。",
        "直接支持は胸ヒレの観測支持、受理は各手法が保留しなかった座標を表します。",
        "C34の受理基準はslot_trackであり、新旧手法のreview基準と同一ではありません。",
        "",
        "| 手法 | 直接支持 | 受理 | 最大欠測 fr (受理) | メス区間 維持/変化/未採点 | 追跡秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["methods"].items():
        direct = row["direct_motion"]["observed_fraction"] * 100
        accepted = row["accepted_motion"]["observed_fraction"] * 100
        maximum = max(item["maximum_missing_frames"] for item in row["accepted_episodes"]["slots"])
        counts = row["female_accepted"]["interval_counts"]
        elapsed = row.get("tracking_seconds")
        timing = "不明" if elapsed is None else f"{elapsed:.1f}"
        lines.append(
            f"| {name} | {direct:.2f}% | {accepted:.2f}% | {maximum} | "
            f"{counts['preserved']}/{counts['swap']}/{counts['unscored']} | {timing} |"
        )
    lines.extend(
        [
            "",
            "各メス注釈と全隣接区間の結果、個体別の欠測区間は comparison.json に保存しました。",
            "端点の一致では区間内の往復スワップやオス2匹の対応は検証できません。",
            "輪郭支持、滑らかさ、既存手法との一致もIDの正解として採点していません。",
            "新方式の繰り返す欠測・競合候補は review_events.json、幾何・観測の診断は",
            "diagnostics.json に保存しています。確認場面の数は人手で修正できる数ではありません。",
            "",
            "計算時間は実行時に測定された追跡時間です。実装・画像処理・同時実行条件が異なり、",
            "速度比の対照実験ではありません。新方式の独立Detectron2準備はこの追跡秒に含みません。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--previous",
        type=Path,
        default=REPO / "work/generative_tracking/recovery_v2_floor6_continuous_one_minute",
    )
    parser.add_argument("--input-root", type=Path, default=REPO / "work/generative_tracking_inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    code = [
        Path(__file__),
        LAB / "diagnostics.py",
        LAB / "review_events.py",
        LAB / "run_experiment.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/eval/generative.py",
        REPO / "src/medaka3d/eval/reacquisition.py",
    ]
    code_hashes = {str(path): sha256(path) for path in code}
    current, summary, hashes = load_completed(args.run)
    older, old_summary, old_hashes = load_completed(args.previous, previous=True)
    hashes.update(old_hashes)
    new_evidence = verify_run_evidence(args.run, summary)
    old_evidence = verify_run_evidence(args.previous, old_summary, previous=True)
    hashes.update(new_evidence)
    hashes.update(old_evidence)
    catalog, catalog_path = staging_catalog(args.input_root)
    hashes[str(catalog_path)] = sha256(catalog_path)
    require_binding(catalog_path, hashes[str(catalog_path)], new_evidence)
    require_binding(catalog_path, hashes[str(catalog_path)], old_evidence)
    initial, seed, evaluation = registered_seed(int(FRAMES[0]))
    if summary["seed"]["pose_xyz"] != seed["pose_xyz"]:
        raise ValueError("new minute uses a different registered first seed")
    for key in catalog:
        if key.startswith(CALIBRATION_DIR) or key.startswith("work/c01_s16_predictions/"):
            path = verified_input(args.input_root, key, catalog)
            hashes[str(path)] = sha256(path)
            require_binding(path, hashes[str(path)], new_evidence)
    gt_path = verified_input(args.input_root, evaluation["female_annotations_relative"], catalog)
    hashes[str(gt_path)] = sha256(gt_path)
    require_binding(gt_path, hashes[str(gt_path)], new_evidence)
    if hashes[str(gt_path)] != evaluation["female_annotations_sha256"]:
        raise ValueError("female annotations differ from the registered evaluation data")
    nodes = CachedNodes(args.input_root, EXPERIMENT, FRAMES)
    rig = load_rig(args.input_root / CALIBRATION_DIR)
    with gt_path.open(encoding="utf-8-sig", newline="") as stream:
        anchors = match_mouth_anchors(
            csv.DictReader(stream),
            lambda view, frame: nodes.at(view, frame)[:, 0],
            int(FRAMES[0]),
            int(FRAMES[-1] + 1),
        )
    methods, positions = {}, {}
    for name, arrays, saved, previous in (
        ("three_body_v1", current, summary, False),
        ("previous_generative", older, old_summary, True),
    ):
        source, review, observed, xyz, accepted = normalized_points(arrays, previous=previous)
        validate_sources(source, FRAMES, nodes)
        methods[name] = common_metrics(FRAMES, source, review, observed, xyz, accepted, anchors)
        methods[name]["tracking_seconds"] = saved["tracking_seconds"]
        methods[name]["review_scope"] = (
            "combined geometry/identity review" if previous else "identity review"
        )
        positions[name] = {"frame_idx": FRAMES, "xyz": xyz}
    validate_result(current, FRAMES, nodes)
    for value in BASELINE_LAMBDAS:
        relative = f"{BASELINE_DIR}/C34-label__{EXPERIMENT}__lam{value}.npz"
        path = verified_input(args.input_root, relative, catalog)
        hashes[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as arrays:
            source_rows = arrays["source"][FRAMES]
            accepted_mask = arrays["slot_track"][FRAMES] >= 0
            xyz = np.where(accepted_mask[..., None], arrays["xyz"][FRAMES], np.nan)
        _pose, observed = skeleton_from_sources(rig, nodes, FRAMES, source_rows)
        observed &= accepted_mask[..., None]
        source = np.repeat(source_rows[..., None], 3, axis=-1)
        # A baseline's original whole-instance source is used for all three nodes.
        validate_sources(source, FRAMES, nodes)
        review = np.zeros(accepted_mask.shape, dtype=bool)
        name = f"C34_lambda_{value}"
        methods[name] = common_metrics(FRAMES, source, review, observed, xyz, xyz, anchors)
        methods[name]["review_scope"] = "C34 slot_track availability; no equivalent review flag"
        methods[name]["tracking_seconds"] = None
        positions[name] = {"frame_idx": FRAMES, "xyz": xyz}
    del positions["three_body_v1"]
    diagnostics = diagnose(current, initial_pose_xyz=initial, baseline_positions=positions)
    events = extract_review_events(
        current, review_gap_frames=summary["tracker_config"]["review_gap_frames"]
    )
    report = {
        "status": "technical_comparison_only",
        "frames": [76620, 86580],
        "fps": 166,
        "methods": methods,
        "new_method_review_counts": events["counts"],
        "new_method_staging_seconds": summary.get("precomputed_sequence_staging_seconds"),
        "independent_all_three_truth_loaded": False,
        "inference_performed": False,
        "old_method_input_provenance_scope": (
            "historical sealed record; original remote videos not rehashed here"
        ),
    }
    if any(sha256(path) != digest for path, digest in {**hashes, **code_hashes}.items()):
        raise ValueError("evaluation input changed during comparison")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_minute_comparison", args.output, name="evaluation") as record:
        for path, digest in {**hashes, **code_hashes}.items():
            record.input(path, path, sha256=digest)
        for path in code:
            target = args.output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name, data in (
            ("comparison", report),
            ("diagnostics", diagnostics),
            ("review_events", events),
        ):
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=sha256(path))
        path = args.output / "RESULT.md"
        path.write_text(markdown_report(report), encoding="utf-8")
        record.output("result", path, sha256=sha256(path))
        if any(sha256(path) != digest for path, digest in {**hashes, **code_hashes}.items()):
            raise ValueError("comparison input or code changed while writing results")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "identity_verdict": "unverified",
                "events": events["counts"],
            }
        )
    )
    return report


if __name__ == "__main__":
    main()
