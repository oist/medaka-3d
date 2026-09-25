"""Post-inference comparison of both forward methods and all seven fixed C34s.

Requires a successful completed directional minute comparison before opening
candidate/block payloads. Reuses six sealed strict-support blocks per classical
method and original, plus the new candidate's completed support audit. No fit,
tracking, raw decode, ray reclassification or GPU work occurs. C34 is a
whole-recording offline result with different initialization, not a causal
same-state control. Sparse female P/S/U is not three-animal identity accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import compare_directional_one_minute as minute
import compare_long_support_160s as classical
import numpy as np
import score_long_female_160s as female
from run_long_remaining_supervisor import final_outputs, save_json, snapshot

from medaka3d.eval.generative import observed_motion_summary, score_female_identity
from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
START, STOP, CHUNK = 66400, 76360, 1660
METHODS = ("directional", "original", *("c34_lambda_" + v for v in female.LAMBDAS))
SUPPORT_ROOT = REPO / "work/three_body/long_support_comparison_160s_v1"
FEMALE_ROOT = REPO / "work/three_body/long_female_endpoint_160s_v1"
BRANCH = REPO / "work/three_body/directional_one_minute_66400-76360_v1"
ORIGINAL = female.NEW_CANDIDATE.parent
SUPPORT_SHA = "30e289f0ed77f8a5bff37b11c3fbaea059ad1caf13107830b4cf25e26b82a16b"
FEMALE_SHA = "e5894da372b2f1b577e8c0ac2d6930140231a6674db90594501dda58d1fba5b6"
STAGE = "directional_minute_all7_classical_comparison_v1"
PINS = {
    Path(
        minute.__file__
    ).resolve(): "066f702d0c64ddf048fe5d9c65e6add64f92daf40025e90b50a698355e6707c3",
    Path(
        classical.__file__
    ).resolve(): "3df8f3e3418ab4f5de8e7698510a82c95c17dd4a578510143e6900edd9ec32c7",
    Path(
        female.__file__
    ).resolve(): "f0b8d960ef421e0cca5e01c487230e46c2c0f23065b3169ea1fa4aaca033e7cc",
    Path(
        classical.baseline.__file__
    ).resolve(): "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4",
}
KEYS = (
    "frame_idx",
    "pose_xyz",
    "sources",
    "observed",
    "original_xyz",
    "original_label_available",
    "original_identity_review",
)
require = female.require


def input_binding(record, path, digest):
    path = Path(path).resolve()
    matching = [v for v in record["inputs"].values() if Path(v["path"]).resolve() == path]
    require(
        matching and all(v["sha256"] == digest for v in matching),
        "input binding differs: " + str(path),
    )


def output_binding(record, root, relative, evidence):
    root = Path(root).resolve()
    path = (root / female.relative(relative)).resolve()
    require(path.is_relative_to(root), "output escaped artifact")
    matches = [v for v in record["outputs"].values() if Path(v["path"]).resolve() == path]
    require(
        matches and all(v["sha256"] == matches[0]["sha256"] for v in matches), "unsealed output"
    )
    evidence.bind(path, matches[0]["sha256"])
    require(all(path.stat().st_size == v["size"] for v in matches), "output size differs")
    return path


def sealed(root, filename, stage, evidence, expected=None):
    root = Path(root).resolve()
    record = evidence.read(root / filename, expected)
    require(
        record.get("status") == "ok" and record.get("stage") == stage,
        "successful completed stage required: " + stage,
    )
    for item in record["outputs"].values():
        require(Path(item["path"]).resolve().is_relative_to(root), "historical output escaped root")
    return record


def completed_minute(root, evidence):
    """Success gate first. No candidate, block or annotation array is opened here."""
    root = Path(root).resolve()
    record = sealed(
        root, "comparison.provenance.json", "directional_one_minute_comparison", evidence
    )
    require(record["params"].get("all_bound_files_unchanged") is True, "minute final guard missing")
    input_binding(record, Path(minute.__file__), PINS[Path(minute.__file__).resolve()])
    for filename in ("summary.json", "checkpoint_audit.json", "frames.csv"):
        output_binding(record, root, filename, evidence)
    summary = evidence.read(root / "summary.json")
    require(
        summary.get("schema") == "directional_one_minute_comparison_v1"
        and summary.get("status") == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["native_frames"] == STOP - START
        and set(summary["methods"]) == {"original", "directional"},
        "incomplete minute summary",
    )
    for key in (
        "same_parent_and_horizon",
        "all21_original_and_directional_arrays_exact_backtrace",
        "original_history_and_all_six_checkpoint_prefixes_exact",
        "checkpoint_snapshots_and_earlier_manifest_digests_verified",
    ):
        require(summary.get(key) is True, "required minute audit missing: " + key)
    require(
        summary["logical_initialization_count"] == 1
        and summary["restores_during_one_minute"] == summary["external_pose_resets"] == 0,
        "minute lifecycle differs",
    )
    for name in ("original", "directional"):
        check = summary["methods"][name]["all_native_source_geometry"]
        require(
            check["native_frames"] == STOP - START
            and check["frame_range"] == [START, STOP]
            and check["blocks"] == 6
            and check["block_frames"] == CHUNK
            and check["ray_gate_mm"] == 2.0
            and check["native_pixel_gate"] == 12.0
            and check["all_sources_and_observed_flags_pass"] is True
            and check["temporal_subsampling"] is False,
            "full9960 strict support audit missing",
        )
    return record, summary


def selected_blocks(summary):
    """Keep the final six exact blocks for every registered method; no tuning."""
    methods = female.validate_methods(summary)
    selected = []
    for method in methods:
        blocks = [b for b in method["blocks"] if START <= b["frame_range"][0] < STOP]
        require(
            [b["frame_range"] for b in blocks]
            == [[f, f + CHUNK] for f in range(START, STOP, CHUNK)],
            "six native blocks required",
        )
        for b in blocks:
            require(
                b["start_index"] == b["frame_range"][0] - female.FIRST
                and b["stop_index"] == b["frame_range"][1] - female.FIRST
                and b["ray_gate_mm"] == 2.0
                and b["native_pixel_gate"] == 12.0
                and b["all_sources_and_observed_flags_pass"] is True,
                "block native index or strict gate differs",
            )
            a, z = b["frame_range"]
            require(
                female.relative(b["path"]) == f"{method['name']}/support_f{a:06d}-{z:06d}.npz",
                "method/native block namespace differs",
            )
        selected.append((method, blocks))
    return selected


def validate_support_block(data, start, stop, *, method):
    is_forward = method in ("original", "directional", "new")
    female.validate_block(data, start, stop, method="new" if is_forward else method)
    n = stop - start
    pose, xyz, observed = data["pose_xyz"], data["original_xyz"], data["observed"]
    require(
        pose.shape == (n, 3, 3, 3) and pose.dtype.kind == "f" and not np.isinf(pose).any(),
        "invalid saved/derived pose",
    )
    require(
        xyz.shape == (n, 3, 3) and xyz.dtype.kind == "f" and not np.isinf(xyz).any(),
        "invalid original pectoral coordinates",
    )
    require(
        pose[:, :, 1].tobytes(order="C") == xyz.astype(pose.dtype).tobytes(order="C"),
        "stored pectorals changed",
    )
    require(not np.any(observed & ~np.isfinite(pose).all(-1)), "supported node is not finite")
    if is_forward:
        require(np.isfinite(pose).all(), "forward latent pose must stay finite")
    else:
        require(
            data["slot_track"].shape == (n, 3)
            and data["slot_track"].dtype == np.int32
            and np.all((data["slot_track"] == -1) | (data["slot_track"] == np.arange(3)))
            and np.array_equal(data["original_label_available"], data["slot_track"] >= 0),
            "C34 original label availability changed",
        )
    if "valid_source" in data:
        held = data["sources"] >= 0
        require(np.array_equal(data["valid_source"], held), "saved source gate flags differ")
        for key, limit in (("ray_distance_mm", 2.0), ("native_pixel_error", 12.0)):
            value = data[key]
            require(
                value.shape == held.shape
                and np.isfinite(value[held]).all()
                and np.all((value[held] >= 0) & (value[held] <= limit)),
                "saved supported residual exceeds gate: " + key,
            )
    result = {key: data[key].copy() for key in KEYS}
    # The historical block writer promoted source rows to int64. Use that same
    # lossless representation for both forward candidates; keep their file SHA.
    result["sources"] = result["sources"].astype(np.int64)
    return result


def concatenate_blocks(blocks, *, method):
    require(len(blocks) == 6, "all six native chunks required")
    values = [
        validate_support_block(b, START + i * CHUNK, START + (i + 1) * CHUNK, method=method)
        for i, b in enumerate(blocks)
    ]
    for key in KEYS:
        require(all(b[key].dtype == values[0][key].dtype for b in values), "block dtype drift")
    return {key: np.concatenate([b[key] for b in values]) for key in KEYS}


def measure(data, anchors, *, name):
    frames, xyz = data["frame_idx"], data["original_xyz"]
    observed, labels = data["observed"], data["original_label_available"]
    direct = observed[:, :, 1]
    forward = name in ("original", "directional")
    native = labels & (direct if forward else np.isfinite(xyz).all(-1))
    strict_native = labels & direct
    strict_xyz = np.where(direct[..., None], xyz, np.nan)
    female_values = (frames, data["sources"][..., 0], observed[..., 0], anchors)
    require(all(START <= a.frame < STOP for a in anchors), "anchor outside common minute")
    return dict(
        name=name,
        native_frames=len(frames),
        fish_frame_denominator=int(direct.size),
        direct_chest_fish_frames=int(direct.sum()),
        direct_chest_fraction=float(direct.mean()),
        direct_chest_by_slot=direct.sum(0).tolist(),
        observed_nodes_by_slot=observed.sum(0).tolist(),
        observed_nodes_total=observed.sum((0, 1)).tolist(),
        original_label_available_fish_frames=int(labels.sum()),
        original_method_available_fish_frames=int(native.sum()),
        original_method_available_fraction=float(native.mean()),
        strict_and_original_available_fish_frames=int(strict_native.sum()),
        strict_and_original_available_fraction=float(strict_native.mean()),
        original_review_fish_frames=int(data["original_identity_review"].sum())
        if forward
        else None,
        direct_adjacent_movement=observed_motion_summary(strict_xyz, frames),
        movement_scope="Both endpoints inside common interval, adjacent native frames and directly "
        "supported; no latent or missing-gap displacement enters these steps.",
        female=dict(
            strict_raw=score_female_identity(*female_values),
            strict_with_original_acceptance=score_female_identity(*female_values, review=~labels),
        ),
        node_scope="saved three-node forward geometry"
        if forward
        else (
            "saved C34 chest unchanged; mouth and tail are source-triangulated derived diagnostics"
        ),
        original_acceptance_scope="original observed chest AND not original identity_review"
        if forward
        else (
            "original C34 slot_track>=0 AND finite saved chest; "
            "no comparable identity_review information"
        ),
    )


def load_evidence(comparison_root, support_root, female_root, evidence):
    minute_seal, comparison = completed_minute(comparison_root, evidence)
    support_seal = sealed(
        support_root, "comparison.provenance.json", female.COMPARE_STAGE, evidence, SUPPORT_SHA
    )
    require(
        support_seal["params"]["all_used_input_source_snapshot_output_sha_unchanged"] is True,
        "classical comparison final guard missing",
    )
    summary = evidence.read(output_binding(support_seal, support_root, "summary.json", evidence))
    require(
        summary["status"] == "ok"
        and summary["candidate_sha256"] == female.NEW_CANDIDATE_SHA
        and summary["original_baseline_catalog_sha256"] == female.CATALOG_SHA256,
        "old comparison candidate/catalog differs",
    )
    methods = selected_blocks(summary)
    shared = summary["shared_SL_and_calibration_sha256"]
    require(len(shared) == 7, "same SL3/calibration4 required")
    female_seal = sealed(
        female_root,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        evidence,
        FEMALE_SHA,
    )
    require(
        female_seal["params"]["all_input_source_output_sha_unchanged"] is True,
        "female evaluation guard missing",
    )
    input_binding(minute_seal, Path(female_root) / "evaluation.provenance.json", FEMALE_SHA)
    annotation = output_binding(female_seal, female_root, "original_female_sex_gt.csv", evidence)
    require(evidence.hashes[str(annotation)] == female.GT_SHA, "female CSV version differs")
    matches_path = output_binding(female_seal, female_root, "anchor_matches.json", evidence)
    female_summary = evidence.read(
        output_binding(female_seal, female_root, "summary.json", evidence)
    )
    require(
        female_summary["comparison_seal_sha256"] == SUPPORT_SHA, "female/support artifact differs"
    )
    for path, digest in shared.items():
        for record in (support_seal, female_seal, minute_seal):
            input_binding(record, path, digest)
        evidence.bind(path, digest)
    nodes_paths = {
        v: next(Path(p) for p in shared if Path(p).name == f"{female.EXPERIMENT}__{v}.npz")
        for v in female.VIEWS
    }
    require(
        len(set(nodes_paths.values())) == 3
        and len([p for p in shared if Path(p).suffix == ".json"]) == 4,
        "SL/calibration membership differs",
    )
    nodes = female.RegisteredNodes(nodes_paths, np.arange(START, STOP, dtype=np.int64))
    with annotation.open(encoding="utf-8-sig", newline="") as stream:
        anchors = female.match_mouth_anchors(
            csv.DictReader(stream), lambda v, f: nodes.at(v, f)[:, 0], START, STOP, match_px=5.0
        )
    saved = [r for r in evidence.read(matches_path) if START <= r["frame"] < STOP]
    require(
        json.loads(json.dumps([asdict(a) for a in anchors])) == saved,
        "fresh native5px female matches differ; unscored anchors may not be dropped",
    )
    require(len(anchors) >= 2, "at least two requested female anchors required")
    # Candidate bytes are bound to the successfully audited exact-horizon comparison.
    candidates = {}
    for name, root, seal_name, stage in (
        ("original", ORIGINAL, "continuation.provenance.json", "three_body_long_continuation"),
        ("directional", BRANCH, "tracking.provenance.json", "directional_one_minute_continuous"),
    ):
        producer_path = root / seal_name
        expected = minute_seal["inputs"][str(producer_path)]["sha256"]
        producer = sealed(root, seal_name, stage, evidence, expected)
        candidate_path = output_binding(producer, root, "candidate.npz", evidence)
        input_binding(minute_seal, candidate_path, evidence.hashes[str(candidate_path)])
        if name == "original":
            require(
                evidence.hashes[str(candidate_path)] == summary["candidate_sha256"],
                "original trace differs between comparisons",
            )
        with np.load(candidate_path, allow_pickle=False) as archive:
            expected_keys = (set(minute.RECORD_FIELDS) - {"parent"}) | {
                "xyz",
                "accepted_xyz",
                "body_xyz",
            }
            require(
                len(archive.files) == 21 and set(archive.files) == expected_keys,
                "audited forward21 schema differs",
            )
            candidate = {k: archive[k].copy() for k in archive.files}
        require(
            candidate["frame_idx"].dtype == np.int64
            and np.array_equal(candidate["frame_idx"], np.arange(female.FIRST, STOP)),
            "forward full-horizon native frames differ",
        )
        require(
            minute.short.stats(candidate, START, STOP)
            == comparison["methods"][name]["full_minute"],
            "audited minute counts/cost differ",
        )
        part = {
            k: candidate[k][START - female.FIRST :].copy()
            for k in ("frame_idx", "pose_xyz", "sources", "observed")
        }
        part.update(
            original_xyz=part["pose_xyz"][:, :, 1].copy(),
            original_label_available=~candidate["identity_review"][START - female.FIRST :],
            original_identity_review=candidate["identity_review"][START - female.FIRST :].copy(),
        )
        check = comparison["methods"][name]["all_native_source_geometry"]
        require(
            check["source_rows_checked"] == int((part["sources"] >= 0).sum())
            and check["observed_nodes_checked"] == int(part["observed"].sum()),
            "full source audit is not bound to loaded candidate",
        )
        candidates[name] = validate_support_block(part, START, STOP, method=name)
    block_records = {}
    for method, blocks in methods:
        name = "original" if method["name"] == "new" else method["name"]
        raw_blocks = []
        for b in blocks:
            path = output_binding(support_seal, support_root, b["path"], evidence)
            require(evidence.hashes[str(path)] == b["sha256"], "method block metadata SHA differs")
            with np.load(path, allow_pickle=False) as archive:
                require(len(archive.files) == len(set(archive.files)), "duplicate block arrays")
                raw_blocks.append({k: archive[k].copy() for k in archive.files})
        values = concatenate_blocks(raw_blocks, method=name)
        if name == "original":
            require(
                all(minute.same(values[k], candidates[name][k]) for k in KEYS),
                "original six sealed blocks differ from audited candidate",
            )
        else:
            require(
                method["lambda_value"] == name.removeprefix("c34_lambda_"), "lambda label differs"
            )
            for path, digest in shared.items():
                require(
                    method["input_sha256"].get(path) == digest, "C34 used different SL/calibration"
                )
            registered = method["input_sha256"]
            catalogs = [p for p in registered if Path(p).name == "staging_manifest.json"]
            labels = [
                p
                for p in registered
                if Path(p).name
                == f"C34-label__{female.EXPERIMENT}__lam{method['lambda_value']}.npz"
            ]
            require(
                len(catalogs) == len(labels) == 1
                and len(registered) == 9
                and registered[catalogs[0]] == female.CATALOG_SHA256,
                "C34 catalog/label/shared input membership differs",
            )
            for path, digest in registered.items():
                input_binding(support_seal, path, digest)
            candidates[name] = values
        block_records[name] = blocks
    require(
        set(candidates) == set(METHODS),
        "all seven classical methods and both forward methods required",
    )
    return (
        candidates,
        anchors,
        comparison,
        dict(
            comparison_sha256=evidence.hashes[
                str(Path(comparison_root).resolve() / "comparison.provenance.json")
            ],
            support_comparison_sha256=SUPPORT_SHA,
            female_evaluation_sha256=FEMALE_SHA,
            shared_SL_calibration_sha256=shared,
            reused_native_blocks=block_records,
            original_candidate_sha256=evidence.hashes[str(ORIGINAL / "candidate.npz")],
            directional_candidate_sha256=evidence.hashes[str(BRANCH / "candidate.npz")],
            prior_physical_support_audits_reused=True,
            support_rays_recomputed_here=False,
            historical_baseline_catalog_and_prediction_inputs_rehashed_here=False,
            source_row_storage_normalization=(
                "lossless signed int32/int64 to int64; coordinate dtypes unchanged"
            ),
        ),
    )


def markdown(result):
    lines = [
        "# 同じ1分区間・全7 C34条件との比較",
        "",
        "[66400,76360)、9960 native frames。胸部の分母は全29,880魚フレームで、欠測も残しています。",
        "",
        "|方式|胸部直接支持|元方式利用可|直接支持∩元利用可|口/胸/尾 支持|",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["methods"]:
        lines.append(
            f"|{row['name']}|{row['direct_chest_fish_frames']}|"
            f"{row['original_method_available_fish_frames']}|"
            f"{row['strict_and_original_available_fish_frames']}|{row['observed_nodes_total']}|"
        )
    lines += ["", "|方式|女性端点 raw P/S/U|元方式利用条件付き P/S/U|", "|---|---|---|"]
    for row in result["methods"]:
        counts = [
            row["female"][key]["interval_counts"]
            for key in ("strict_raw", "strict_with_original_acceptance")
        ]
        text = ["/".join(str(c[k]) for k in ("preserved", "swap", "unscored")) for c in counts]
        lines.append(f"|{row['name']}|{text[0]}|{text[1]}|")
    lines += [
        "",
        "P/S/U は隣接する指定女性端点の preserved/swap/unscored。"
        "未採点端点を飛ばして接続せず、Pは途中の戻りswapを否定しません。",
        "",
        "C34の口・尾は元sourceから再三角測量した派生診断です。保存胸部は変更しません。"
        "C34は全動画の将来情報を使う後処理で初期化も異なり、同一状態の因果的対照ではありません。"
        "元C34 label可用性とforwardのreview除外は異なる利用条件です。",
        "",
        "同じSL・校正・native区間での比較であり、各方式のslot番号を生物IDとして対応付けていません。"
        "全7 lambdaを掲載し、最良条件だけの選択はしません。"
        "支持・低移動量は全3個体ID正解の証明ではありません。",
        "既存の封印済み支持監査を再利用。再fit・再追跡・GPU・原画decode・ray再計算は行っていません。",
        "",
    ]
    return "\n".join(lines)


def run(comparison_root, output, *, support_root=SUPPORT_ROOT, female_root=FEMALE_ROOT):
    comparison_root, support_root, female_root, output = map(
        lambda p: Path(p).resolve(), (comparison_root, support_root, female_root, output)
    )
    require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new work output required"
    )
    evidence = female.Evidence()
    codes = {str(p): female.sha(p) for p in classical.code_paths()}
    for path, digest in {**codes, **{str(p): d for p, d in PINS.items()}}.items():
        evidence.bind(path, digest)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="comparison") as record:
        try:
            snapshot(codes, output, record)
            values, anchors, comparison, proof = load_evidence(
                comparison_root, support_root, female_root, evidence
            )
            methods = [measure(values[name], anchors, name=name) for name in METHODS]
            for name in ("original", "directional"):
                result = next(m for m in methods if m["name"] == name)["female"]
                prior = comparison["methods"][name]["female"]
                require(
                    result["strict_raw"] == prior["raw"]
                    and result["strict_with_original_acceptance"] == prior["with_original_review"],
                    "same female endpoints differ from completed minute comparison",
                )
            summary = dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=9960,
                methods=methods,
                evidence=proof,
                requested_female_anchors=len(anchors),
                requested_female_intervals=len(anchors) - 1,
                all_seven_lambdas_retained=True,
                method_slot_biological_alignment=False,
                all_three_ID_accuracy_measured=False,
                same_causal_state_control_for_C34=False,
                same_SL_calibration_and_native_interval=True,
                full_thirty_minute_comparison=False,
                adopted=False,
                GPU_or_new_tracking=False,
            )
            save_json(output / "summary.json", summary, record, "summary")
            save_json(
                output / "anchor_matches.json",
                [asdict(a) for a in anchors],
                record,
                "anchor_matches",
            )
            report = output / "RESULT.md"
            report.write_text(markdown(summary), encoding="utf-8")
            record.output("RESULT", report, sha256=female.sha(report))
        finally:
            for path, digest in evidence.hashes.items():
                record.input(path, path, sha256=digest)
            require(
                {str(p): female.sha(p) for p in classical.code_paths()} == codes,
                "loaded source set changed",
            )
            evidence.guard()
            final_outputs(record)
        record.param("all_opened_input_source_snapshot_output_sha_unchanged", True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional-comparison", type=Path, required=True)
    parser.add_argument("--support-comparison", type=Path, default=SUPPORT_ROOT)
    parser.add_argument("--female-evaluation", type=Path, default=FEMALE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(
        args.directional_comparison,
        args.output,
        support_root=args.support_comparison,
        female_root=args.female_evaluation,
    )


if __name__ == "__main__":
    main()
