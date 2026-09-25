"""Compare the recovered five-minute Nodewise path with the fixed Sparse baseline once."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
from pathlib import Path

import compare_nodewise_body_row_five_minutes as metric

proof, common, require, WORK = metric.proof, metric.common, metric.require, metric.WORK
STAGE = "resumed_nodewise_five_minute_comparison_v1"
NODEWISE_STAGE = "nodewise_body_row_five_minute_resume__v1"
NODEWISE_ROOT = WORK / "nodewise_body_row_five_minute_resume_099600_v1"
PARENT_ROOT = WORK / "nodewise_body_row_five_minute_continuation_v1"
PARENT_PLAN_SHA = "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
PARENT_CHECKPOINT_SHA = "60170ebec150caa385f911224a25fee77f615269077b47413a926f8fe6fc5cf1"


def exact_candidate(root, seal, summary, evidence):
    item = dict(root=root, seal=seal, summary=summary)
    return metric.load_candidate(item, evidence)


def load_nodewise(root, seal_sha, evidence):
    root = Path(root).resolve()
    require(root.is_relative_to(WORK), "Nodewise root must be inside workspace")
    evidence.add(root / "tracking.provenance.json", seal_sha)
    seal = proof.read_json(root / "tracking.provenance.json")
    common.exact_fields(
        seal,
        dict(stage="nodewise_body_row_five_minute_resume_099600_v1", status="ok"),
        "successful recovered producer required",
    )
    item = dict(root=root, seal=seal)
    summary = metric.document(item, "summary.json", evidence)
    plan = metric.document(item, "plan.json", evidence)
    index = metric.document(item, "input_index.json", evidence)
    expected = dict(
        process_interval=[99600, 116200],
        scientific_interval=[66400, 116200],
        retained_interval=[49800, 116200],
        initialize_count=1,
        restore_count=5,
        new_begin_calls=0,
        external_pose_resets=0,
    )
    common.exact_fields(summary, dict(status="ok", finish_calls=1, **expected), "final run")
    common.exact_fields(plan, expected, "recovery plan")
    common.exact_fields(
        summary,
        dict(
            identity_or_pose_reinitialization=False,
            all39_roundtrip=True,
            finish_matches_checkpoint_backtrace=True,
            full_five_minute_path_recovered=True,
            physical_ID_truth=False,
        ),
        "recovered state/backtrace contract",
    )
    require(
        summary["candidate"] == seal["outputs"]["candidate.npz"]
        and summary["candidate_arrays"].keys() == metric.ALL21
        and summary["tracker_context"] == plan["tracker_context"]
        and plan["process_recovery_from_committed_frame"] == 99600
        and plan["checkpoint_binding"]["interrupted_parent_checkpoint_sha256"]
        == PARENT_CHECKPOINT_SHA,
        "recovered candidate or parent checkpoint differs",
    )
    require(
        set(index) == {"original_binding", "prefix_chunks"}
        and len(index["prefix_chunks"]) == 40
        and index["original_binding"] == plan["checkpoint_binding"]["original_binding"],
        "complete common source index required",
    )
    parent_plan = PARENT_ROOT / "plan.json"
    evidence.add(parent_plan, PARENT_PLAN_SHA)
    require(
        proof.read_json(parent_plan)["tracker_context"] == plan["tracker_context"],
        "recovery changed the original Nodewise policy context",
    )
    source_map = summary["source_sha256"]
    require(source_map == plan["source_sha256"], "source closure differs")
    for source, digest in source_map.items():
        evidence.add(source, digest)
    item.update(summary=summary, plan=plan, input_index=index, sha256=seal_sha)
    return item


def write_result(output, summary):
    methods = summary["methods"]

    def total_missing(row):
        return sum(sum(slot) for slot in row["missing_by_slot_node"])

    def maximum_missing(row):
        return max(max(slot) for slot in row["longest_missing_run_by_slot_node"])

    def direct_jumps(row):
        return len(row["direct_adjacent_steps_over_10mm"])

    sparse, nodewise = methods["sparse"], methods["nodewise"]
    lines = [
        "# Nodewise連続5分：既存Sparse方式との比較",
        "",
        "録画6:40–11:40の49,800フレームを同じ5:00状態から比較した。"
        "Nodewiseは99,600の全39状態からプロセス復元したが、ID・姿勢の再初期化は0回。",
        "",
        "| 指標 | 既存Sparse | Nodewise |",
        "|---|---:|---:|",
        "| 観測支持なしの部位・フレーム | "
        f"{total_missing(sparse):,} | {total_missing(nodewise):,} |",
        f"| 全3匹の胸が同時に直接支持 | {sparse['all_three_chests_direct_frames']:,} | "
        f"{nodewise['all_three_chests_direct_frames']:,} |",
        "| 最長の連続欠測（フレーム） | "
        f"{maximum_missing(sparse):,} | {maximum_missing(nodewise):,} |",
        f"| 支持付き10 mm超の隣接変位 | {direct_jumps(sparse):,} | {direct_jumps(nodewise):,} |",
        f"| 身体内で観測行が混在 | {sparse['body_frames_with_any_mixed_source_rows']:,} | "
        f"{nodewise['body_frames_with_any_mixed_source_rows']:,} |",
        "",
        "保存されたdirectフラグを同じ事前定義済み指標で集計した。支持は人手正解ではなく、"
        "スロット番号は雌雄・物理IDを表さない。全3匹のスワップ0はこの比較から認定しない。",
        "",
        f"比較データ: `comparison.json`。Nodewise seal `{summary['nodewise_tracking_sha256']}`。",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(nodewise_root, nodewise_sha, output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh output required")
    output.mkdir(parents=True)
    evidence, started = proof.Evidence(), proof.utc()
    try:
        evidence.add(Path(__file__))
        for name, digest in metric.PINS.items():
            evidence.add(WORK / name, digest)
        sparse = metric.read_producer(
            WORK / metric.SPARSE_STAGE,
            metric.SPARSE_SHA,
            metric.SPARSE_STAGE,
            evidence,
        )
        metric.validate_completion(sparse, "sparse", metric.SPARSE_RUNNER_SHA, evidence)
        nodewise = load_nodewise(nodewise_root, nodewise_sha, evidence)
        require(sparse["input_index"] == nodewise["input_index"], "different source inputs")
        candidates = {
            "sparse": metric.load_candidate(sparse, evidence),
            "nodewise": exact_candidate(
                nodewise["root"], nodewise["seal"], nodewise["summary"], evidence
            ),
        }
        methods, pairwise, _, _ = metric.compare_arrays(candidates)
        result = dict(
            schema=STAGE,
            status="ok",
            scientific_interval=[66400, 116200],
            native_frames=49800,
            arms=["sparse", "nodewise"],
            methods=methods,
            pairwise=pairwise,
            sparse_tracking_sha256=metric.SPARSE_SHA,
            nodewise_tracking_sha256=nodewise_sha,
            predefined_metric_source=str(Path(metric.__file__).resolve()),
            candidate_modified=False,
            support_recomputed=False,
            physical_ID_truth=False,
            temporal_ID_mapping_computed=False,
        )
        comparison = output / "comparison.json"
        comparison.write_bytes(proof.json_bytes(result))
        write_result(output, result)
        evidence.final_guard()
        outputs = {
            p.name: dict(path=str(p), sha256=proof.sha(p), size_bytes=p.stat().st_size)
            for p in output.iterdir()
            if p.is_file()
        }
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=outputs,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            error=repr(error),
        )
        raise
    finally:
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodewise-root", type=Path, default=NODEWISE_ROOT)
    parser.add_argument("--nodewise-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.nodewise_root, args.nodewise_sha, args.output)
