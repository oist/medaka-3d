"""CPU interim comparison of committed Tail and audited Bounded through68060.

Only the first immutable checkpoint payload is deserialized. The running Tail
producer needs no final seal, and later winners are never read. Prior Bounded
support is reused from a pinned successful audit; Tail support is recomputed.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import audit_bounded_first_chunk as prior
import numpy as np
import tail_recovery_tracker as tail
from continuous_tracker import RECORD_FIELDS

comparison, checks, REPO = prior.comparison, prior.checks, prior.REPO
bind, input_member, array_signature = prior.bind, prior.input_member, prior.array_signature
START, STOP, FIRST = prior.START, prior.STOP, prior.FIRST
STAGE = "tail_first_chunk_interim_audit_v1"
OLD = REPO / "work/three_body/bounded_mouth_one_minute_66400-76360_v1"
OLD_AUDIT = REPO / "work/three_body/bounded_mouth_first_chunk_interim_audit_v1"
OLD_AUDIT_SHA = "08a7440cb5c851c4a97d98d1d67f665f6762a89c524c2ba87bfa7c0245656f04"
OLD_MANIFEST_SHA = "603490402bbaa6ea99a1694ae3e81ade77ba7b4dfa2f1ff7dec5b16605ba9048"
PRIOR_SOURCE_SHA = "7be951a77f39e8f2dea5e6cf2aeae1a10663036ede0e131a1462beb67c45759e"
RUNNER = REPO / "work/three_body/run_tail_one_minute.py"
RUNNER_SHA = "3c11909ab244cfdb1bae3589baa987a9102827111372db00c853d4666ca18ca8"
CLASS_SHA = "b5a98085c6b1b03bd17e2f0b627ffc3eebdd569cf5add545146d9322761868a2"


def validate_commit_row(branch, row, expected_sha):
    checks.require(
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and all(c in "0123456789abcdef" for c in expected_sha),
        "expected manifest SHA required",
    )
    root = branch / "checkpoints/through_068060"
    checks.require(
        isinstance(row, dict)
        and Path(row["path"]).resolve() == root
        and row["manifest_sha256"] == expected_sha,
        "first checkpoint path/digest differs",
    )
    return root


def committed(branch, expected_sha, hashes):
    rows = checks.read(branch / "checkpoint_progress.json")
    checks.require(isinstance(rows, list) and len(rows) >= 1, "first checkpoint not committed")
    row = copy.deepcopy(rows[0])
    root = validate_commit_row(branch, row, expected_sha)
    bind(root / "manifest.json", expected_sha, hashes)
    # The frozen wrapper requires all three paths registered before codec read.
    # The codec then independently verifies payload SHA against this manifest.
    for name in ("metadata.json", "arrays.npz"):
        path = root / name
        bind(path, checks.sha(path), hashes)
    fixed = checks.read(root / "manifest.json")["bindings"]
    checks.require(
        fixed["schema"] == "tail_one_minute_checkpoint_binding_v1"
        and fixed["native_interval"] == [START, comparison.STOP],
        "Tail binding scope differs",
    )
    codes = fixed["experimental_code_sha256"]
    checks.require(
        codes.get(str(RUNNER)) == RUNNER_SHA
        and codes.get(str(Path(tail.__file__).resolve())) == CLASS_SHA,
        "unqualified Tail producer/class",
    )
    for path, digest in codes.items():
        source = bind(path, digest, hashes)
        checks.require(source.is_relative_to(REPO), "source escaped repository")
        bind(branch / "code_snapshot" / source.relative_to(REPO), digest, hashes)
    migration = bind(branch / "state_import.json", fixed["state_import_sha256"], hashes)
    arrays, meta, _ = comparison.load_bound_checkpoint(root, fixed, hashes)
    comparison.validate_lifecycle(meta, STOP, 4)
    checks.require(
        row["arrays"] == 39
        and all(
            row[k] == meta[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ),
        "commit header differs from checkpoint",
    )
    return arrays, meta, fixed, checks.read(migration), row


def verify_tail_import(migration, parent, parent_meta, new_meta, old_meta):
    checks.require(
        migration["schema"] == "tail_recovery_original_state_import_v1"
        and migration["status"] == "ok",
        "successful explicit Tail migration required",
    )
    bounded_meta = {"context": migration["bounded_context"]}
    prior.verify_bounded_import(migration["bounded_validation"], parent, parent_meta, bounded_meta)
    checks.require(
        migration["arrays"] == array_signature(parent)
        and migration["original_metadata"] == parent_meta
        and migration["old_context"] == parent_meta["context"]
        and migration["new_context"] == new_meta["context"]
        and migration["bounded_context"] == old_meta["context"]
        and migration["bounded_context"] == migration["bounded_validation"]["new_context"],
        "Tail migration original/Bounded context differs",
    )
    expected = copy.deepcopy(parent_meta)
    expected["restore_count"] += 1
    expected["context"] = migration["new_context"]
    checks.require(migration["imported_metadata"] == expected, "migration changed other metadata")
    context = copy.deepcopy(migration["new_context"])
    policy = context.pop("tail_recovery_policy")
    checks.require(
        policy["schema"] == "tail_recovery_continuous_policy_v1"
        and policy["enabled"] is True
        and policy["target_node"] == 2
        and policy["allow_bounded"] is False
        and policy["clears_identity_review"] is False,
        "Tail policy disabled, changed node/method, or clears review",
    )
    context["implementation"] = migration["bounded_context"]["implementation"]
    checks.require(context == migration["bounded_context"], "undeclared Tail context change")
    checks.require(
        migration["all_arrays_byte_preserved"] is True
        and migration["original_metadata_unchanged"] is True
        and migration["logical_initialize_count"] == 1
        and migration["new_begin_calls"] == 0
        and migration["added_restore_count"] == 1
        and migration["original_implementation_not_claimed_for_new_tracking"] is True
        and migration["review_cleared_by_migration"] is False,
        "Tail migration preservation flags differ",
    )


def output_member(seal, root, name, hashes):
    path = (root / name).resolve(strict=True)
    item = seal["outputs"].get(name)
    checks.require(item is not None and Path(item["path"]).resolve() == path, "output role/path")
    return bind(path, item["sha256"], hashes)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify_seal(root, filename, stage, hashes):
    before = dict(hashes)
    seal = checks.verify(root, filename, stage, hashes)
    checks.require(
        all(hashes.get(path) == digest for path, digest in before.items()),
        "seal verification replaced an already bound SHA",
    )
    return seal


def interval(trace):
    checks.require(
        len(trace) == 21 and np.array_equal(trace["frame_idx"], np.arange(FIRST, STOP)),
        "all21 same68060 backtrace required",
    )
    return {key: value[START - FIRST : STOP - FIRST].copy() for key, value in trace.items()}


def verify_prior_support(part, support, totals):
    """Bind an already geometrically audited declaration; no new optical claim."""
    checks.require(
        set(support) == {"frame_idx", "ray_distance_mm", "native_pixel_error", "valid_source"}
        and checks.equal_bytes(support["frame_idx"], part["frame_idx"]),
        "prior support frame/schema differs",
    )
    valid = support["valid_source"]
    checks.require(
        valid.dtype == np.dtype(bool)
        and valid.shape == part["sources"].shape
        and np.array_equal(valid, part["sources"] >= 0)
        and np.array_equal(valid.sum(axis=2) >= 2, part["observed"]),
        "prior strict source/observation differs",
    )
    for key, limit in (("ray_distance_mm", 2.0), ("native_pixel_error", 12.0)):
        values = support[key]
        checks.require(
            values.shape == valid.shape
            and np.isfinite(values[valid]).all()
            and np.all(values[valid] >= 0)
            and np.all(values[valid] <= limit),
            "prior strict geometry differs",
        )
    checks.require(
        totals["native_frames"] == STOP - START
        and totals["frame_range"] == [START, STOP]
        and totals["ray_gate_mm"] == 2.0
        and totals["native_pixel_gate"] == 12.0
        and totals["all_sources_and_observed_flags_pass"] is True
        and totals["source_rows_checked"] == int(valid.sum())
        and totals["observed_nodes_checked"] == int(part["observed"].sum()),
        "prior strict totals differ",
    )


def all_history(arrays):
    """Owned full ordered history, including ancestors; beam rank is not an ID."""
    return {
        name: arrays[name].copy()
        for name in ("history_offsets", *("history_" + key for key in RECORD_FIELDS))
    }


def load_inputs(branch, expected_sha, hashes):
    new, new_meta, fixed, migration, row = committed(branch, expected_sha, hashes)
    bind(Path(prior.__file__), PRIOR_SOURCE_SHA, hashes)
    bind(OLD_AUDIT / "audit.provenance.json", OLD_AUDIT_SHA, hashes)
    seal = verify_seal(OLD_AUDIT, "audit.provenance.json", prior.STAGE, hashes)
    checks.require(
        seal.get("finished_at")
        and seal["params"].get("all_source_input_snapshot_output_SHA_unchanged") is True,
        "successful guarded prior interim audit required",
    )
    summary = checks.read(output_member(seal, OLD_AUDIT, "summary.json", hashes))
    checks.require(
        summary["schema"] == prior.STAGE
        and summary["status"] == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["horizon_stop"] == STOP
        and summary["expected_manifest_sha256"] == OLD_MANIFEST_SHA
        and summary["all39_arrays_semantically_decoded"] is True
        and summary["all21_arrays_backtraced"] is True
        and summary["initial_parent_history_prefix_preserved"] is True
        and summary["shared_input_binding"] == fixed["original_binding"],
        "prior same-horizon proof differs",
    )
    old_root = OLD / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        input_member(seal, old_root / name, hashes)
    checks.require(hashes[str(old_root / "manifest.json")] == OLD_MANIFEST_SHA, "old manifest")
    old_fixed = checks.read(old_root / "manifest.json")["bindings"]
    checks.require(
        old_fixed["schema"] == "bounded_mouth_one_minute_checkpoint_binding_v1"
        and old_fixed["native_interval"] == [START, comparison.STOP]
        and old_fixed["original_binding"] == fixed["original_binding"],
        "different original parent/input configuration",
    )
    for path, digest in old_fixed["experimental_code_sha256"].items():
        source = input_member(seal, path, hashes)
        checks.require(hashes[str(source)] == digest, "old source binding")
        input_member(seal, OLD / "code_snapshot" / source.relative_to(REPO), hashes)
    old, old_meta, _ = comparison.load_bound_checkpoint(old_root, old_fixed, hashes)
    comparison.validate_lifecycle(old_meta, STOP, 4)
    binding = fixed["original_binding"]
    input_member(seal, prior.PARENT / "continuation.provenance.json", hashes)
    parent_seal = verify_seal(
        prior.PARENT, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(parent_seal["params"]["bindings"] == binding, "parent binding mismatch")
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        prior.PARENT / "checkpoint", binding, hashes
    )
    comparison.validate_lifecycle(parent_meta, START, 3)
    verify_tail_import(migration, parent, parent_meta, new_meta, old_meta)
    old_import = input_member(seal, OLD / "state_import.json", hashes)
    checks.require(hashes[str(old_import)] == old_fixed["state_import_sha256"], "old migration SHA")
    prior.verify_bounded_import(checks.read(old_import), parent, parent_meta, old_meta)
    for data in (old, new):
        comparison.verify_history_prefix(parent, data)
    for path, digest in binding["code_sha256"].items():
        bind(path, digest, hashes)
    seed_info = binding["inputs"]["seed"]
    seed_path = bind(seed_info["registration_path"], seed_info["registration_sha256"], hashes)
    seed = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    shared = seed_info["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "SL3/calibration4 required")
    for path, digest in shared.items():
        source = input_member(seal, path, hashes)
        checks.require(hashes[str(source)] == digest, "shared geometry input differs")
    paths = {
        view: next(
            Path(p) for p in shared if Path(p).name == prior.EXPERIMENT + "__" + view + ".npz"
        )
        for view in prior.VIEWS
    }
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig = prior.load_rig(calibration)
    nodes = prior.RegisteredNodes(paths, np.arange(START, STOP))
    decoded = {
        "bounded": prior.semantic_decode(
            prior.bounded.BoundedMouthContinuousTracker, old, old_meta, rig, seed
        ),
        "tail": prior.semantic_decode(tail.TailRecoveryContinuousTracker, new, new_meta, rig, seed),
    }
    traces = {
        name: comparison.short.backtrace(data) for name, data in (("bounded", old), ("tail", new))
    }
    parts = {name: interval(trace) for name, trace in traces.items()}
    prior_part = read_npz(output_member(seal, OLD_AUDIT, "bounded_same68060_candidate.npz", hashes))
    checks.require(
        set(prior_part) == set(parts["bounded"])
        and all(
            checks.equal_bytes(value, parts["bounded"][key]) for key, value in prior_part.items()
        ),
        "old all21 trace differs from pinned interim audit",
    )
    support = read_npz(output_member(seal, OLD_AUDIT, "bounded_strict_support.npz", hashes))
    totals = summary["methods"]["bounded"]["all_native_source_geometry"]
    verify_prior_support(parts["bounded"], support, totals)
    return dict(
        traces=traces,
        parts=parts,
        decoded=decoded,
        nodes=nodes,
        rig=rig,
        row=row,
        fixed=fixed,
        histories={"bounded": all_history(old), "tail": all_history(new)},
        metadata={"bounded": old_meta, "tail": new_meta},
        prior_support=support,
        prior_totals=totals,
        prior_method=summary["methods"]["bounded"],
    )


def compare_support(old, new):
    totals, changes = prior.compare_support(old, new)
    for change in changes:
        for field in ("review", "sources"):
            old_value = change.pop("directional_" + field)
            new_value = change.pop("bounded_" + field)
            change["bounded_" + field] = old_value
            change["tail_" + field] = new_value
    return totals, changes


def main(args):
    branch, output = args.branch.resolve(strict=True), args.output.resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with prior.provenance(STAGE, output, name="audit") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(target) == digest, "source snapshot changed")
            emit(str(target.relative_to(output)))
        data = load_inputs(branch, args.expected_manifest_sha256, hashes)
        methods = {}
        for name, part in data["parts"].items():
            np.savez_compressed(output / f"{name}_same68060_candidate.npz", **part)
            emit(f"{name}_same68060_candidate.npz")
            np.savez_compressed(output / f"{name}_all_history.npz", **data["histories"][name])
            emit(f"{name}_all_history.npz")
            emit(f"{name}_checkpoint_metadata.json", data["metadata"][name])

            def save_support(_meta, arrays, label=name):
                np.savez_compressed(output / f"{label}_strict_support.npz", **arrays)
                emit(f"{label}_strict_support.npz")

            if name == "bounded":
                support = data["prior_totals"]
                save_support(None, data["prior_support"])
            else:
                support = comparison.support_audit.verify_support_partitioned(
                    part, data["nodes"], data["rig"], on_block=save_support
                )
            accepted = part["observed"] & ~part["identity_review"][:, :, None]
            methods[name] = dict(
                counts=comparison.short.stats(data["traces"][name], START, STOP),
                direct_by_slot_node=part["observed"].sum(0).tolist(),
                accepted_by_slot_node=accepted.sum(0).tolist(),
                accepted_by_node=accepted.sum((0, 1)).tolist(),
                selected_kind_counts=dict(Counter(part["proposal_kind"].tolist())),
                all_native_source_geometry=support,
                strict_support_scope="pinned successful audit reused"
                if name == "bounded"
                else "all1660 independently recomputed",
                review_scope="original inherited sticky per-slot review; no human correction",
            )
            emit(f"{name}_reacquisitions.json", prior.node_reacquisitions(data["traces"][name]))
        for key in (
            "counts",
            "direct_by_slot_node",
            "accepted_by_slot_node",
            "accepted_by_node",
            "selected_kind_counts",
        ):
            checks.require(
                methods["bounded"][key] == data["prior_method"][key], "prior method totals differ"
            )
        delta, changes = compare_support(data["parts"]["bounded"], data["parts"]["tail"])
        emit("gain_loss.json", changes)
        emit("committed_checkpoint_record.json", data["row"])
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=1660,
            horizon_stop=STOP,
            expected_manifest_sha256=args.expected_manifest_sha256,
            prior_audit_sha256=OLD_AUDIT_SHA,
            prior_manifest_sha256=OLD_MANIFEST_SHA,
            complete_minute=False,
            parent_run_may_still_be_running=True,
            same_parent_and_horizon=True,
            all39_arrays_semantically_decoded=True,
            all21_arrays_backtraced=True,
            initial_parent_history_prefix_preserved=True,
            logical_initialization_count=1,
            checkpoint_restore_count=4,
            restores_during_interval=0,
            external_pose_resets=0,
            all6_ordered_history_saved=True,
            history_rank_is_not_identity=True,
            methods=methods,
            support_delta=delta,
            checkpoint_validation=data["decoded"],
            shared_input_binding=data["fixed"]["original_binding"],
            CPU_validation_scope="copied context changes device only; no restore/fit/advance",
            identity_review_scope=(
                "frozen semantic decoder checks saved sticky review/ages; "
                "full unretained pool margins are not reconstructed"
            ),
            independent_ID_truth=False,
            adopted=False,
            interim_input_limit=(
                "first commit, parent/input/seed/calibration/code bindings verified; "
                "no new full-run seal, raw/D2 replay, future winner or full-minute claim"
            ),
            old_support_scope=(
                "pinned 08a744 prior audit, all21 byte equality and saved strict arrays/totals; "
                "no old optical recomputation"
            ),
        )
        emit("summary.json", result)
        a, b = methods["bounded"]["counts"], methods["tail"]["counts"]
        (output / "RESULT.md").write_text(
            "# Tail追加・最初10秒の同終端比較\n\n"
            f"[66400,68060) の1660時刻。口/胸部/尾の直接支持: Bounded {a['direct_node_support']} "
            f"→ Tail {b['direct_node_support']}。"
            f"元review下の胸部利用数: {a['accepted_chest_fish_frames']} "
            f"→ {b['accepted_chest_fish_frames']}。\n\n"
            "共通66400親、三段の明示移行、39配列の全履歴意味検証、同68060からの21配列逆追跡を照合。"
            "旧Boundedの厳密支持は封印済み監査を再利用し、新Tailの1660時刻を同2mm/12pxで再計算した。"
            "両方式の全6順位履歴19項目とoffsetsを保存した。順位は仮説の個体対応を意味しない。\n\n"
            "全1分の完了・生物ID正解・採用判断は主張しない。将来の終端から逆追跡すると途中経路は変わり得る。"
            "支持の増減は算術診断であり、未保存の全生成poolやfit途中の原因を確定しない。"
            "reviewは元の保留を継承し、再観測には欠測期間を付す。原画/D2の再読込は行っていない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(
            checks.read(branch / "checkpoint_progress.json")[0] == data["row"],
            "first commit changed",
        )
        checks.require(code == comparison.source_registry.local_code(), "loaded source set changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "final bound input/source/output changed")
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(main(parser.parse_args()), indent=2))
