"""Read only sealed 200-frame appearance/persistent outputs; no inference or ID repair."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import evaluate_appearance_minute as appearance  # noqa: E402
import evaluate_minute as common  # noqa: E402
import evaluate_variant_minute as persistent  # noqa: E402
import run_experiment as base  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FRAMES = np.arange(333519, 333719, dtype=np.int64)


def load_sealed(root):
    root = appearance.local_path(root)
    record = appearance.load_json(root / "experiment.provenance.json")
    if record.get("status") != "ok":
        raise ValueError("inference must finish successfully before candidate/summary are read")
    hashes = {
        str(root / "experiment.provenance.json"): base.sha256(root / "experiment.provenance.json")
    }
    for key, name in (("summary", "summary.json"), ("candidate", "candidate.npz")):
        path = root / name
        item = record["outputs"][key]
        if appearance.local_path(item["path"]) != path or base.sha256(path) != item["sha256"]:
            raise ValueError("sealed run output path/SHA differs")
        hashes[str(path)] = item["sha256"]
    summary = appearance.load_json(root / "summary.json")
    with np.load(root / "candidate.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if not np.array_equal(arrays["frame_idx"], FRAMES):
        raise ValueError("only the fixed complete 200-frame development window is allowed")
    return arrays, summary, hashes


def array_comparison(a, b, frames):
    same_schema = a.shape == b.shape and a.dtype == b.dtype
    if not same_schema:
        return dict(
            shape_equal=a.shape == b.shape,
            dtype_equal=a.dtype == b.dtype,
            bitwise_equal=False,
            first_different_frame=None,
        )
    a_bytes = np.ascontiguousarray(a).view(np.uint8).reshape(len(frames), -1)
    b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(len(frames), -1)
    different = np.any(a_bytes != b_bytes, axis=1)
    changed = np.flatnonzero(different)
    result = dict(
        shape_equal=True,
        dtype_equal=True,
        bitwise_equal=not bool(len(changed)),
        differing_frames=len(changed),
        first_different_frame=int(frames[changed[0]]) if len(changed) else None,
        differing_enrollment_prefix_frames=int(different[:166].sum()),
        differing_post_enrollment_frames=int(different[166:].sum()),
    )
    if a.dtype.kind in "fiu":
        valid = np.isfinite(a) & np.isfinite(b)
        result["maximum_absolute_finite_difference"] = (
            float(np.max(np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))))
            if np.any(valid)
            else None
        )
        result["finite_mask_equal"] = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
    return result


def period_stats(arrays, first, stop):
    observed = arrays["observed"][first:stop]
    accepted = np.isfinite(arrays["accepted_xyz"][first:stop]).all(axis=-1)
    result = dict(
        native_frames=int(stop - first),
        frame_start=int(FRAMES[first]),
        frame_stop=int(FRAMES[stop - 1] + 1),
        fish_frames=int(3 * (stop - first)),
        direct_pectoral_fish_frames=int(observed[:, :, 1].sum()),
        accepted_pectoral_fish_frames=int(accepted.sum()),
        direct_pectoral_by_slot=observed[:, :, 1].sum(axis=0).tolist(),
        direct_mouth_by_slot=observed[:, :, 0].sum(axis=0).tolist(),
        accepted_pectoral_by_slot=accepted.sum(axis=0).tolist(),
        identity_review_fish_frames=int(arrays["identity_review"][first:stop].sum()),
    )
    for name in ("costs", "geometric_cost", "appearance_cost"):
        if name in arrays:
            value = arrays[name][first:stop]
            result[name] = dict(
                sum=float(value.sum()),
                mean=float(value.mean()),
                minimum=float(value.min()),
                maximum=float(value.max()),
            )
    if "appearance_enabled" in arrays:
        result["appearance_scored_frames"] = int(
            arrays["appearance_enabled"][first:stop].any(axis=1).sum()
        )
        result["appearance_scored_frames_by_view"] = (
            arrays["appearance_enabled"][first:stop].sum(axis=0).tolist()
        )
        result["sampled_pixels_by_view"] = (
            arrays["appearance_pixels"][first:stop].sum(axis=0).tolist()
        )
    return result


def code_paths():
    return [
        Path(__file__),
        *(
            LAB / name
            for name in (
                "evaluate_appearance_minute.py",
                "evaluate_variant_minute.py",
                "evaluate_minute.py",
                "run_appearance_tracking.py",
                "run_persistent_node_recovery.py",
                "run_node_recovery.py",
                "run_experiment.py",
                "appearance_diagnostic.py",
                "observations.py",
                "prepare_sequence.py",
                "diagnostics.py",
                "review_events.py",
            )
        ),
        *(
            REPO / "src/medaka3d" / name
            for name in (
                "calib.py",
                "types.py",
                "provenance.py",
                "eval/generative.py",
                "eval/reacquisition.py",
            )
        ),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    codes = {str(path): base.sha256(path) for path in code_paths()}
    current, summary, hashes = load_sealed(args.run)
    reference, reference_summary, reference_hashes = load_sealed(args.persistent)
    appearance.validate_registered_arm(summary)
    bound = appearance.verify_appearance_evidence(args.run, summary, expected_frames=FRAMES)
    ref_bound = persistent.verify_variant_evidence(
        args.persistent, reference_summary, expected_frames=FRAMES
    )
    if reference_summary["algorithm_arm"] != appearance.persistent_runner.ALGORITHM_ARM:
        raise ValueError("reference must be the persistent-mouth arm")
    parity = appearance.same_v1_inputs(
        args.run, summary, args.persistent, reference_summary, bound, ref_bound
    )
    for name in ("node_recovery_config", "persistence_config"):
        if not appearance.exact_json(summary[name], reference_summary[name]):
            raise ValueError("recovery/persistence settings differ")
    current = appearance.validate_appearance_arrays(current, FRAMES, summary)
    reference = persistent.validate_variant_arrays(reference, FRAMES, reference_summary)
    record = appearance.load_json(args.run / "experiment.provenance.json")
    input_root = appearance.local_path(record["params"]["input_root"])
    nodes = common.CachedNodes(input_root, base.EXPERIMENT, FRAMES)
    for result in (current, reference):
        common.validate_sources(result["sources"], FRAMES, nodes)
        base.validate_result(result, FRAMES, nodes)
    enrollment, enrollment_hashes = appearance.verify_enrollment(
        args.run, summary, current, FRAMES, nodes=nodes
    )
    hashes.update({**reference_hashes, **bound, **ref_bound, **enrollment_hashes})
    if set(reference) != appearance.APPEARANCE_ARRAYS - {
        "appearance_cost",
        "geometric_cost",
        "appearance_enabled",
        "appearance_pixels",
    }:
        raise ValueError("reference does not contain precisely the original 21 arrays")
    comparisons = {
        name: array_comparison(current[name], reference[name], FRAMES) for name in sorted(reference)
    }
    periods = {
        label: {
            method: period_stats(arrays, start, stop)
            for method, arrays in (("appearance", current), ("persistent", reference))
        }
        for label, start, stop in (
            ("all", 0, 200),
            ("enrollment_prefix", 0, 166),
            ("post_enrollment", 166, 200),
        )
    }
    all_stats = periods["all"]["appearance"]
    gate = dict(
        minimum_direct_pectoral_fish_frames=586,
        minimum_accepted_pectoral_fish_frames=417,
        direct_pectoral_fish_frames=all_stats["direct_pectoral_fish_frames"],
        accepted_pectoral_fish_frames=all_stats["accepted_pectoral_fish_frames"],
    )
    gate["passes_fixed_200_development_gate"] = bool(
        gate["direct_pectoral_fish_frames"] >= 586 and gate["accepted_pectoral_fish_frames"] >= 417
    )
    scored = np.flatnonzero(current["appearance_enabled"].any(axis=1))
    report = dict(
        status="sealed_200_development_analysis",
        frames=[333519, 333719],
        fps=166,
        current_run=str(args.run.resolve()),
        persistent_run=str(args.persistent.resolve()),
        candidate_sha256=base.sha256(args.run / "candidate.npz"),
        persistent_candidate_sha256=base.sha256(args.persistent / "candidate.npz"),
        appearance_persistent_input_parity=parity,
        source_record_and_input_bindings_verified=True,
        current_array_count=len(current),
        reference_array_count=len(reference),
        enrollment_record_audit=enrollment,
        first_scored_frame=int(FRAMES[scored[0]]) if len(scored) else None,
        fixed_200_gate=gate,
        periods=periods,
        all_21_array_comparisons=comparisons,
        all_21_arrays_bitwise_equal=all(item["bitwise_equal"] for item in comparisons.values()),
        inference_performed=False,
        gpu_used=False,
        independent_identity_truth_used=False,
        online_registration_compared_to_final_traceback=False,
        raw_image_enrollment_recomputation_performed=False,
        limitations=[
            "The opened development gate measures observation availability, "
            "not correct identities.",
            "Final prefix differences can result from future beam traceback; "
            "they do not alone prove noncausal enrollment.",
            "Online registration records are validated separately; "
            "raw pixel replay is a separate audit.",
            "Direct node source bounds/exclusivity were validated, "
            "but this script does not refit or relabel any fish.",
        ],
    )
    frozen = {**hashes, **codes}
    if any(base.sha256(path) != digest for path, digest in frozen.items()):
        raise ValueError("input/source changed during read-only analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("appearance_transient_200_analysis", args.output, name="analysis") as rec:
        rec.params_from(args)
        for path, digest in frozen.items():
            rec.input(path, path, sha256=digest)
        for source, digest in codes.items():
            target = args.output / "code_snapshot" / Path(source).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if base.sha256(target) != digest:
                raise ValueError("analysis code snapshot changed during copy")
            rec.output(str(target.relative_to(args.output)), target, sha256=digest)
        output = args.output / "summary.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        rec.output("summary", output, sha256=base.sha256(output))
        outcome = "通過" if gate["passes_fixed_200_development_gate"] else "不通過"
        text = (
            f"# 外見を使った200フレームの保存結果の検証\n\n固定の開発基準は{outcome}。"
            f"直接胸部支持 {gate['direct_pectoral_fish_frames']}/600、"
            f"受理 {gate['accepted_pectoral_fish_frames']}/600 でした。\n\n"
            f"色の採点は {len(scored)} フレーム、最初は {report['first_scored_frame']}。"
            f"登録件数 (view,slot) は {enrollment['counts']} です。\n\n"
            "25配列・オンライン登録12配列と保存来歴を検証しました。元persistentの21配列との"
            "違いは summary.json に全て保存しています。最終backtraceの初期166フレームは、"
            "登録時に用いたonline状態と区別しています。原画像からの登録再計算は別の監査です。\n\n"
            "観測率や色の使用は3匹のIDの正しさを示しません。推論、GPU実行、ID修正はしていません。\n"
        )
        output = args.output / "RESULT.md"
        output.write_text(text, encoding="utf-8")
        rec.output("result", output, sha256=base.sha256(output))
        if any(base.sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError("input/source changed while saving analysis")
    print(json.dumps({"output": str(args.output), "gate": gate, "scored_frames": len(scored)}))
    return report


if __name__ == "__main__":
    main()
