"""Read-only CPU audit of the first sealed 1660-frame long-tracking prefix.

No tracking, renderer, detector, optimization, GT, or post-seed baseline is run.
The independent support audit is a separately sealed, candidate-bound prerequisite.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import audit_continuation_comparison as common  # noqa: E402
import long_tracking_inputs as verified  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from source_sequence import SourceSequenceReader  # noqa: E402
from tracker_checkpoint import read_checkpoint  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

BASE, VIEWS = common.BASE, common.VIEWS
FIRST, STOP, FULL_STOP = 49800, 51460, 348600
PROTOCOL = LAB / "long_tracking_protocol.json"
PROTOCOL_SHA = "fda72ac8304be4e46032853a13f895ddb9bf25f2436096b65e27e2418a619c85"
HELPER_SHA = "c7355c194cc9daba79492bd9bbf00a338c24ee34e5a4c3c50d3439e2260e73f4"
RUNNER_SHA = "76b1ba9b06b90338bda27318665f6606ca00565b9809780e635558096303b61d"
LOADER_SHA = "bea59f4e65235df8bce49bed0f1db2e92815f998b3bd7e018f456748f9e173bb"


def history_check(arrays, metadata, initial):
    offsets = arrays["history_offsets"]
    count = STOP - FIRST
    if (
        offsets.dtype != np.int64
        or offsets.shape != (count + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 1)
        or np.any(np.diff(offsets) > 6)
        or metadata["record_fields"] != list(common.HISTORY_FIELDS)
        or metadata["state_fields"] != list(common.STATE_FIELDS)
    ):
        raise ValueError("checkpoint ordered history schema differs")
    if any(len(arrays["history_" + key]) != offsets[-1] for key in common.HISTORY_FIELDS):
        raise ValueError("checkpoint contains unaccounted history rows")
    previous = [
        {
            "pose": initial.copy(),
            "velocity": np.zeros((3, 3)),
            "missing": np.zeros(3, int),
            "review": np.zeros(3, bool),
            "cost": 0.0,
            "node_missing": np.zeros((3, 3), int),
            "direct_pectoral_age": np.zeros(3, int),
        }
    ]
    eligible_calls, nonzero_parents = 0, 0
    for time in range(count):
        eligible_calls += bool(
            np.any(
                (previous[0]["node_missing"][:, 0] >= 4) & (previous[0]["direct_pectoral_age"] >= 4)
            )
        )
        current = []
        for row in range(int(offsets[time]), int(offsets[time + 1])):
            record = {key: arrays["history_" + key][row] for key in common.HISTORY_FIELDS}
            parent_index = int(record["parent"])
            if not 0 <= parent_index < len(previous) or int(record["frame_idx"]) != FIRST + time:
                raise ValueError("checkpoint temporal parent/native frame differs")
            nonzero_parents += parent_index != 0
            parent = previous[parent_index]
            sources, held = record["sources"], record["observed"]
            if (
                not np.isfinite(record["pose_xyz"]).all()
                or np.any(record["pose_xyz"] < [0, 0, -6])
                or np.any(record["pose_xyz"] > [192, 192, 186])
                or np.any(sources < -1)
            ):
                raise ValueError("history pose/source fails the registered finite box contract")
            if not np.array_equal(held, (sources >= 0).sum(axis=1) >= 2):
                raise ValueError("checkpoint observed/source counts differ")
            for view in range(3):
                for node in range(3):
                    used = sources[:, view, node]
                    used = used[used >= 0]
                    if len(used) != len(np.unique(used)):
                        raise ValueError("checkpoint reuses a source across fish")
            body = held[:, 1] | (
                record["mask_supported"] & (sources[:, :, :2] >= 0).any(axis=(1, 2))
            )
            missing = np.where(body, 0, parent["missing"] + 1)
            node_age = np.where(held, 0, parent["node_missing"] + 1)
            pectoral_age = np.where(held[:, 1], parent["direct_pectoral_age"] + 1, 0)
            for key, value in (
                ("body_supported", body),
                ("missing", missing),
                ("node_missing", node_age),
                ("direct_pectoral_age", pectoral_age),
            ):
                common.same_array(record[key], value, "parent recurrence " + key)
            if np.any(parent["review"] & ~record["identity_review"]) or np.any(
                (missing >= 4) & ~record["identity_review"]
            ):
                raise ValueError("checkpoint lost sticky review")
            cost = parent["cost"] + record["costs"]
            common.same_array(np.asarray(cost), np.asarray(record["path_costs"]), "parent cost")
            velocity = 0.5 * parent["velocity"] + 0.5 * (
                record["pose_xyz"][:, 1] - parent["pose"][:, 1]
            )
            norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1.0, 2.0 / np.maximum(norm, 1e-12))
            velocity = np.where(body[:, None], velocity, 0.8 * parent["velocity"])
            current.append(
                dict(
                    pose=record["pose_xyz"],
                    velocity=velocity,
                    missing=missing,
                    review=record["identity_review"],
                    cost=float(cost),
                    node_missing=node_age,
                    direct_pectoral_age=pectoral_age,
                )
            )
        previous = current
    for key in common.STATE_FIELDS:
        common.same_array(arrays["state_" + key], np.asarray([row[key] for row in previous]), key)
    if eligible_calls != metadata["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]:
        raise ValueError("checkpoint first-parent eligible counter differs")
    return dict(
        retained_history_rows=int(offsets[-1]),
        nonzero_parent_rows=nonzero_parents,
        active_beam_count=len(previous),
        independent_age_eligible_calls=eligible_calls,
        parent_cost_velocity_missing_ages_support_review_validated=True,
    )


def last_frames_check(inputs, arrays, metadata):
    reader = SourceSequenceReader(inputs.chunk_roots[-1], inputs.input_root)
    if metadata["previous_frame"] != STOP - 1:
        raise ValueError("checkpoint previous native frame differs")
    reports = []
    for native in range(STOP - 3, STOP):
        index = native - int(reader.frames[0])
        per_view = {}
        for view in VIEWS:
            capture = cv2.VideoCapture(str(reader.videos[view]))
            try:
                if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, native):
                    raise ValueError("cannot seek the native source frame")
                ok, image = capture.read()
                if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != native + 1:
                    raise ValueError("source decode native index differs")
            finally:
                capture.release()
            pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_sha != reader.pixel_hashes[view][index]:
                raise ValueError("native source pixel SHA differs from sealed chunk")
            mask = (
                decode_frame_masks(reader.masks[view], index)
                if reader.masks[view]["view_available"][index]
                else None
            )
            fields = {
                "images_bgr": image,
                "masks": mask,
                "detections": reader.nodes.at(view, native),
                "backgrounds_bgr": reader.backgrounds[view],
            }
            per_view[view] = {
                "raw_bgr_sha256": pixel_sha,
                "mask_count": None if mask is None else len(mask),
                "mask_decoded_sha256": None
                if mask is None
                else hashlib.sha256(mask.tobytes()).hexdigest(),
            }
            if native == STOP - 1:
                for field, value in fields.items():
                    key = "previous_" + field + "_" + view
                    if metadata["previous_observation"][key] != (value is not None):
                        raise ValueError("previous observation availability differs")
                    if value is not None:
                        per_view[view][field] = common.same_array(arrays[key], value, key)
                    elif key in arrays:
                        raise ValueError("missing previous mask became an invented array")
        reports.append({"native_frame": native, "views": per_view})
    return {
        "decoded_native_frames": reports,
        "last_frame_checkpoint_bgr_masks_detections_backgrounds_byte_exact": True,
        "scope": "Three terminal frames per view decoded independently; only the final "
        "frame is the previous-observation payload stored by the checkpoint.",
    }


def verify_support_audit(root, run, candidate_sha, hashes):
    """Bind a separately sealed all-source CPU geometry audit, before any count claims."""
    seal_path = root / "audit.provenance.json"
    seal = common.read_json(seal_path)
    if (
        seal.get("stage") != "long_continuation_direct_support_independent_audit"
        or seal.get("status") != "ok"
        or seal.get("params", {}).get("all_input_source_snapshot_output_sha_unchanged") is not True
    ):
        raise ValueError("direct geometry audit is not successfully sealed")
    common.bind(seal_path, BASE.sha256(seal_path), hashes)
    for info in seal["inputs"].values():
        path = Path(info["path"]).resolve(strict=True)
        common.bind(path, info["sha256"], hashes)
    for info in seal["outputs"].values():
        path = Path(info["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or info.get("size") != path.stat().st_size:
            raise ValueError("direct geometry audit output escaped/changed")
        common.bind(path, info["sha256"], hashes)
    if not any(
        Path(info["path"]).resolve() == run / "candidate.npz" and info["sha256"] == candidate_sha
        for info in seal["inputs"].values()
    ):
        raise ValueError("support audit did not seal this candidate as input")
    output_paths = {Path(info["path"]).resolve() for info in seal["outputs"].values()}
    if not {root / "summary.json", root / "native_support.npz"} <= output_paths:
        raise ValueError("support audit did not seal required outputs")
    summary = common.read_json(root / "summary.json")
    if (
        summary.get("schema") != "long_continuation_direct_support_audit_v1"
        or Path(summary.get("run_path", "")).resolve() != run
        or summary.get("candidate_sha256") != candidate_sha
        or summary.get("producer_seal_sha256") != hashes[str(run / "continuation.provenance.json")]
        or summary.get("frame_range") != [FIRST, STOP]
        or summary.get("native_frames") != STOP - FIRST
        or summary.get("source_exclusivity_all_views_nodes") is not True
        or summary.get("full_sequence_complete") is not False
        or summary.get("independent_ID_truth") is not False
    ):
        raise ValueError("support audit candidate/native range/scope differs")
    geometry = summary["all_node_source_geometry"]
    if (
        geometry.get("ray_gate_mm") != 2.0
        or geometry.get("native_pixel_gate") != 12.0
        or geometry.get("all_sources_and_observed_flags_pass") is not True
    ):
        raise ValueError("support audit did not validate unchanged direct geometry thresholds")
    with np.load(root / "native_support.npz", allow_pickle=False) as archive:
        support = {key: archive[key] for key in archive.files}
    expected_keys = {"frame_idx", "ray_distance_mm", "native_pixel_error", "valid_source"}
    if set(support) != expected_keys:
        raise ValueError("support audit exported an unknown array schema")
    common.same_array(support["frame_idx"], np.arange(FIRST, STOP), "support native frames")
    valid = support["valid_source"]
    if valid.shape != (STOP - FIRST, 3, 3, 3) or valid.dtype != np.bool_:
        raise ValueError("support audit validity shape/dtype differs")
    for key, threshold in (("ray_distance_mm", 2.0), ("native_pixel_error", 12.0)):
        values = support[key]
        if (
            values.shape != valid.shape
            or values.dtype.kind != "f"
            or np.any(~np.isfinite(values[valid]))
            or np.any(values[valid] < 0)
            or np.any(values[valid] > threshold)
        ):
            raise ValueError("support audit numeric array fails its direct geometry threshold")
    return {
        "summary": summary,
        "root": str(root),
        "seal_sha256": hashes[str(seal_path)],
        "all_outputs_and_inputs_SHA_verified": True,
    }, support


def first_episode(mask):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None
    start = int(indices[0])
    end = start + 1
    while end < len(mask) and mask[end]:
        end += 1
    return [FIRST + start, FIRST + end]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO):
        raise ValueError("audit output must be a new workspace path")
    sources = sorted({*common.code_paths(), Path(__file__).resolve()})
    hashes = {str(path): BASE.sha256(path) for path in sources}
    common.bind(Path(common.__file__).resolve(), HELPER_SHA, hashes)
    common.bind(PROTOCOL, PROTOCOL_SHA, hashes)
    common.bind(LAB / "run_long_continuation.py", RUNNER_SHA, hashes)
    common.bind(LAB / "long_tracking_inputs.py", LOADER_SHA, hashes)
    seal_path = root / "continuation.provenance.json"
    seal = common.read_json(seal_path)
    if seal.get("stage") != "three_body_long_continuation" or seal.get("status") != "ok":
        raise ValueError("tracking phase has not completed successfully")
    common.bind(seal_path, BASE.sha256(seal_path), hashes)
    params, binding = seal["params"], seal["params"]["bindings"]
    protocol = common.read_json(PROTOCOL)
    if (
        params["through_frame"] != STOP
        or params["resume_phase"] is not None
        or params.get("all_input_source_snapshot_output_sha_unchanged") is not True
        or binding["tracking_protocol_sha256"] != PROTOCOL_SHA
        or binding["package_versions"] != protocol["package_versions"]
        or binding["torch_cpu_threads"] != protocol["torch_cpu_threads"]
    ):
        raise ValueError("first long phase runtime/endpoint/lifecycle differs")
    input_binding = binding["inputs"]
    seed_root = Path(input_binding["seed"]["registration_path"]).parent
    inputs = verified.load_long_inputs(
        seed_root,
        Path(input_binding["input_stage_root"]),
        Path(input_binding["seed"]["input_root"]),
        through_frame=STOP,
    )
    if (
        common.canonical(inputs.fixed_bindings) != common.canonical(input_binding)
        or tuple(inputs.full_frame_range) != (FIRST, FULL_STOP)
        or not np.array_equal(inputs.frame_idx, np.arange(FIRST, STOP))
    ):
        raise ValueError("current verified prefix differs from recorded input binding")
    expected_inputs = {**inputs.verified_hashes, **binding["code_sha256"]}
    if set(seal["inputs"]) != set(expected_inputs):
        raise ValueError("phase input membership differs from full verified proof")
    for name, expected in expected_inputs.items():
        info = seal["inputs"][name]
        if Path(info["path"]).resolve() != Path(name) or info["sha256"] != expected:
            raise ValueError("phase input path/SHA binding differs")
        common.bind(name, expected, hashes)
    for source, expected in binding["code_sha256"].items():
        snapshot = root / "code_snapshot" / Path(source).relative_to(REPO)
        info = seal["outputs"].get(str(snapshot))
        if info is None or Path(info["path"]).resolve() != snapshot or info["sha256"] != expected:
            raise ValueError("executing source snapshot is not sealed")
        common.bind(snapshot, expected, hashes)
    for info in seal["outputs"].values():
        path = Path(info["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or info.get("size") != path.stat().st_size:
            raise ValueError("foreign/changed phase output")
        common.bind(path, info["sha256"], hashes)
    for role, name in (
        ("candidate", "candidate.npz"),
        ("summary", "summary.json"),
        ("progress", "progress.jsonl"),
        ("consumed_chunks", "consumed_chunks.json"),
        ("checkpoint_manifest", "checkpoint/manifest.json"),
    ):
        if Path(seal["outputs"][role]["path"]).resolve() != root / name:
            raise ValueError("named output path binding differs")
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        if str(root / filename) not in seal["outputs"]:
            raise ValueError("checkpoint payload is not bound to the successful phase")
    consumed = common.read_json(root / "consumed_chunks.json")
    verified.verify_prefix_records(consumed, inputs.chunk_records)
    if common.canonical(consumed) != common.canonical(inputs.chunk_records) or len(consumed) != 1:
        raise ValueError("consumed first chunk evidence differs")
    arrays, metadata, manifest = read_checkpoint(root / "checkpoint", expected_bindings=binding)
    summary = common.read_json(root / "summary.json")
    expected_summary = dict(
        schema="long_continuation_phase_v1",
        technical_completion=True,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        registered_interval=[FIRST, FULL_STOP],
        processed_interval=[FIRST, STOP],
        processed_total=STOP - FIRST,
        phase_frames=[FIRST, STOP],
        phase_processed=STOP - FIRST,
        consumed_chunk_count=1,
        logical_initialization_count=1,
        checkpoint_restore_count=0,
        external_pose_reset_count=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
    )
    if any(
        common.canonical(summary.get(key)) != common.canonical(value)
        for key, value in expected_summary.items()
    ):
        raise ValueError("long phase completion/conditional prefix summary differs")
    if (
        metadata["processed_frames"] != STOP - FIRST
        or metadata["previous_frame"] != STOP - 1
        or metadata["initialize_count"] != 1
        or metadata["restore_count"] != 0
        or metadata["node_recovery_diagnostics"] != summary["node_recovery_diagnostics"]
        or summary["checkpoint_payload_bytes"] != manifest["payload"]["bytes"]
    ):
        raise ValueError("checkpoint lifecycle/counters differ")
    progress = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    processed = [item["processed"] for item in progress]
    if (
        not processed
        or processed != sorted(set(processed))
        or processed[-1] != STOP - FIRST
        or any(
            item["source_frame"] != FIRST + item["processed"] - 1
            or item["phase_processed"] != item["processed"]
            for item in progress
        )
    ):
        raise ValueError("progress counts/native frames differ")
    context = metadata["context"]
    if (
        context["initial_pose"] != common.digest_array(inputs.initial_pose_xyz)
        or context["initial_lengths"]
        != common.digest_array(np.linalg.norm(np.diff(inputs.initial_pose_xyz, axis=1), axis=-1))
        or context["lower"] != common.digest_array(np.array([0.0, 0.0, -6.0]))
        or context["upper"] != common.digest_array(np.array([192.0, 192.0, 186.0]))
        or context["views"] != list(inputs.rig.views)
        or context["world_units"] != inputs.rig.world_units
        or context["cameras"]
        != common.normalize([dataclasses.asdict(c) for c in inputs.rig.cameras])
        or context["tracker_config"] != protocol["tracker_config"]
        or context["recovery_config"] != protocol["node_recovery_config"]
        or context["persistence_config"] != protocol["persistence_config"]
    ):
        raise ValueError("checkpoint first seed/calibration/tracking configuration differs")
    candidate_sha = BASE.sha256(root / "candidate.npz")
    support, support_arrays = verify_support_audit(
        args.support_audit.resolve(strict=True), root, candidate_sha, hashes
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_long_prefix_independent_audit", output, name="audit") as record:
        for path, expected in hashes.items():
            record.input(path, path, sha256=expected)
        for key in (
            "gpu_used",
            "new_tracking_run",
            "human_or_GT_inputs",
            "cross_window_quality_comparison",
        ):
            record.param(key, False)
        outputs = {}
        for source in sources:
            target = output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            outputs[str(target)] = hashes[str(source)]
        history = history_check(arrays, metadata, inputs.initial_pose_xyz)
        if (
            history["retained_history_rows"] != summary["retained_history_rows"]
            or history["active_beam_count"] != summary["active_beam_count"]
        ):
            raise ValueError("summary beam/history counts differ")
        with np.load(root / "candidate.npz", allow_pickle=False) as archive:
            candidate = {key: archive[key] for key in archive.files}
        traced = common.traced_candidate(arrays)
        if set(traced) != set(candidate) or len(candidate) != 21 or len(arrays) != 39:
            raise ValueError("final 21/provisional or checkpoint39 array schema differs")
        comparisons = {
            key: common.same_array(candidate[key], traced[key], key) for key in candidate
        }
        common.same_array(
            support_arrays["valid_source"],
            candidate["sources"] >= 0,
            "independent valid source flags",
        )
        common.same_array(
            support_arrays["valid_source"].sum(axis=2) >= 2,
            candidate["observed"],
            "independent observed flags",
        )
        last = last_frames_check(inputs, arrays, metadata)
        direct = candidate["observed"][:, :, 1]
        accepted = np.isfinite(candidate["accepted_xyz"]).all(axis=-1)
        review = candidate["identity_review"]
        count = len(candidate["frame_idx"])
        if not np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)):
            raise ValueError("candidate native frames differ")
        if np.any(accepted & (~direct | review)):
            raise ValueError("accepted output exposes unsupported/reviewed geometry")
        totals = dict(
            direct_pectoral_count=int(direct.sum()),
            accepted_pectoral_count=int(accepted.sum()),
            identity_review_count=int(review.sum()),
            per_slot_direct_pectoral=direct.sum(0).tolist(),
            per_slot_accepted_pectoral=accepted.sum(0).tolist(),
        )
        if any(summary[key] != value for key, value in totals.items()):
            raise ValueError("producer availability summaries differ")
        statistics = {
            **totals,
            "denominator_fish_frames": count * 3,
            "direct_fraction": float(direct.mean()),
            "conditional_accepted_fraction": float(accepted.mean()),
            "first_review_episode_by_initial_slot": [
                first_episode(review[:, fish]) for fish in range(3)
            ],
            "review_is_biological_identity_truth": False,
        }
        report = dict(
            schema="long_first1660_independent_audit_v1",
            status="ok",
            run=str(root),
            frames=[FIRST, STOP],
            full_registered_range=[FIRST, FULL_STOP],
            logical_initializations=1,
            checkpoint_restores=0,
            external_pose_resets=0,
            candidate_sha256=candidate_sha,
            all_21_arrays_match_full_history_backtrace=comparisons,
            checkpoint_arrays=39,
            checkpoint_logical_array_bytes=sum(value.nbytes for value in arrays.values()),
            history=history,
            previous_and_terminal_frames=last,
            strict_direct_geometry=support,
            availability=statistics,
            prefix_backtrace_provisional=True,
            full30_completed=False,
            seed_binding=inputs.fixed_bindings["seed"],
            consumed_chunk_proofs=consumed,
            tracking_seconds=summary["tracking_seconds"],
            preparation_seconds=summary["preparation_seconds"],
            reader_setup_seconds=summary["reader_setup_seconds"],
            checkpoint_seconds=summary["checkpoint_seconds"],
            checkpoint_payload_bytes=summary["checkpoint_payload_bytes"],
            limitations=[
                "Initial slots are conditional anonymous correspondences, not new ID truth.",
                "The optimistic first-pose registration originates in the whole-recording "
                "C34 output. Neither this audit nor the long tracker reads later C34 coordinates.",
                "Different seed/input/background: no quality equivalence to earlier "
                "windows is claimed.",
                "Only the first 1660 native frames of 298800 were processed; "
                "the final prefix may change.",
                "Direct support is geometric consistency, not biological identity correctness.",
            ],
        )
        rows = [
            "# 最初の1660フレームの独立監査",
            "",
            "成功seal・入力/ソース/snapshot/出力SHA・処理済みchunk証拠を確認しました。"
            "checkpointの全beam履歴から21配列を独立に逆追跡し、保存結果とbytes完全一致しました。",
            "",
            f"区間は [{FIRST},{STOP})、初期化1回、復元0回、外部姿勢リセット0回です。"
            "出力は長区間の暫定prefixであり、30分間の完了結果ではありません。",
            "",
            f"胸部の直接観測は {totals['direct_pectoral_count']}/{count * 3} "
            f"({direct.mean():.2%})、初期ID対応に条件付けた受理は "
            f"{totals['accepted_pectoral_count']}/{count * 3} ({accepted.mean():.2%}) です。",
            "",
            "暫定winning path全sourceの2mm ray/12px reprojection条件は別のCPU監査とcandidate SHAで"
            "結びました。末尾3native framesを3視点で再decodeし、最後の1frameのBGR/masks/SL/背景は"
            "checkpointの前frame payloadと完全一致しています。",
            "",
            "初期slotごとの最初のreview区間(終端を含まない): "
            + json.dumps(statistics["first_review_episode_by_initial_slot"]),
            "",
            "この初期対応は匿名slotの条件付き対応です。全3個体の人手真値はなく、"
            "観測の幾何学的一致や受理率から生物学的IDの正しさを断定しません。"
            "以前の区間とはseed・入力・背景が異なるため、同条件の品質比較も行っていません。",
            "",
            "初期9点は全記録を用いた既存C34結果に由来する楽観的な登録値です。"
            "今回の追跡・監査はその初期値だけを使い、後続のC34座標を読みません。"
            "この10秒では永続的な口欠測による追加候補も実行されていません。",
            "",
        ]
        for filename, data in (
            ("summary.json", json.dumps(report, indent=2, allow_nan=False)),
            ("RESULT.md", "\n".join(rows)),
        ):
            path = output / filename
            path.write_text(data, encoding="utf-8")
            outputs[str(path)] = BASE.sha256(path)
        for path, expected in outputs.items():
            record.output(path, path, sha256=expected)
        common.unchanged({**hashes, **outputs})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps({"output": str(output), "status": "ok"}))


if __name__ == "__main__":
    main()
