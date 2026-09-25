"""Anonymous, fixed-condition diagnosis of a sealed 160s support comparison."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import compare_long_support_160s as comparison
import numpy as np

from medaka3d.provenance import provenance

REPO = comparison.REPO
INPUT = REPO / "work/three_body/long_support_comparison_160s_v1"
SEAL_SHA = "30e289f0ed77f8a5bff37b11c3fbaea059ad1caf13107830b4cf25e26b82a16b"
SOURCE_SHA = "3df8f3e3418ab4f5de8e7698510a82c95c17dd4a578510143e6900edd9ec32c7"
STAGE = "long_160s_all7_common_chest_deficit_diagnosis"
DETAIL_KEYS = (
    "proposal_kind",
    "node_missing",
    "direct_pectoral_age",
    "missing",
    "body_supported",
    "mask_supported",
    "hypotheses",
    "score_margin",
    "proposal_flow_nodes",
    "proposal_flow_parent",
)
require = comparison.require


def load_inputs(evidence):
    record = comparison.sealed_outputs(
        INPUT / "comparison.provenance.json", SEAL_SHA, comparison.STAGE, evidence
    )
    summary = comparison.common.json_output(record, "summary.json", INPUT / "summary.json")
    comparison.check_method_set(summary["methods"])
    require(
        summary["candidate_sha256"] == comparison.CANDIDATE_SHA, "different completed160 candidate"
    )
    methods = {}
    keys = (
        "frame_idx",
        "pose_xyz",
        "sources",
        "observed",
        "original_identity_review",
        "original_label_available",
    )
    for method in summary["methods"]:
        parts = {key: [] for key in keys}
        for block in method["blocks"]:
            path = (INPUT / block["path"]).resolve()
            require(path.is_relative_to(INPUT), "block escaped completed comparison")
            comparison.common.output_binding(record, path, block["sha256"])
            with np.load(path, allow_pickle=False) as data:
                for key in keys:
                    parts[key].append(data[key])
        values = {key: np.concatenate(rows) for key, rows in parts.items()}
        require(
            np.array_equal(values["frame_idx"], np.arange(49800, 76360)),
            "native comparison coverage differs",
        )
        methods[method["name"]] = values
    candidate = comparison.RUN / "candidate.npz"
    comparison.common.input_binding(record, candidate, comparison.CANDIDATE_SHA)
    evidence.add(candidate, comparison.CANDIDATE_SHA)
    with np.load(candidate, allow_pickle=False) as data:
        details = {key: data[key] for key in DETAIL_KEYS}
        for key in ("frame_idx", "pose_xyz", "sources", "observed"):
            require(
                np.array_equal(data[key], methods["new"][key], equal_nan=True),
                "new comparison block differs from bound candidate: " + key,
            )
        require(
            np.array_equal(data["identity_review"], methods["new"]["original_identity_review"]),
            "review flags differ from bound candidate",
        )
    return methods, details


def episodes(frames, selected):
    """Every maximal selected native-contiguous interval, with no gap bridging."""
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(frames[indices]) != 1)) + 1
    return [
        dict(
            start_index=int(part[0]),
            stop_index=int(part[-1] + 1),
            frame_range=[int(frames[part[0]]), int(frames[part[-1]]) + 1],
            frames=len(part),
        )
        for part in np.split(indices, breaks)
    ]


def within_frame_comparison(new_pose, new_sources, old_pose, old_sources):
    """Full matrices; positive row overlap and nearest position are diagnostics only.

    There is no one-to-one permutation optimization and no cross-frame naming.
    Multiple/nonexistent source matches remain -1; no mapping by elimination.
    """
    a, b = new_sources[..., 1], old_sources[..., 1]
    overlap = ((a[:, :, None, :] >= 0) & (a[:, :, None, :] == b[:, None, :, :])).sum(-1)
    distances = np.linalg.norm(new_pose[:, :, None, 1] - old_pose[:, None, :, 1], axis=-1)
    positive = overlap > 0
    source_match = np.where(positive.sum(-1) == 1, positive.argmax(-1), -1)
    finite = np.isfinite(distances)
    safe = np.where(finite, distances, np.inf)
    minima = safe.min(-1, keepdims=True)
    unique = finite & (safe == minima)
    nearest = np.where(unique.sum(-1) == 1, unique.argmax(-1), -1)
    return overlap, distances, source_match, nearest


def histogram(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(key): int(count) for key, count in zip(keys, counts, strict=True)}


def analyze(methods, details):
    new = methods["new"]
    frames, observed = new["frame_idx"], new["observed"]
    old_names = comparison.METHODS[1:]
    new_count = observed[:, :, 1].sum(1)
    old_count = np.stack([methods[name]["observed"][:, :, 1].sum(1) for name in old_names], 1)
    all7 = (old_count == 3).all(1)
    selected = (new_count < 3) & all7
    missing = ~observed[:, :, 1]
    qualifying_missing = missing & selected[:, None]
    spans = episodes(frames, selected)
    for row in spans:
        lo, hi = row["start_index"], row["stop_index"]
        absent = missing[lo:hi]
        row.update(
            missing_chest_fish_frames=int(absent.sum()),
            new_support_count_histogram=histogram(new_count[lo:hi]),
            selected_scene_proposal_kind=histogram(details["proposal_kind"][lo:hi]),
            missing_chest_with_direct_mouth=int((absent & observed[lo:hi, :, 0]).sum()),
            missing_chest_with_body_support=int((absent & details["body_supported"][lo:hi]).sum()),
            missing_chest_review=int((absent & new["original_identity_review"][lo:hi]).sum()),
        )
    largest = sorted(spans, key=lambda row: (-row["frames"], row["frame_range"][0]))[:5]
    arrays = dict(
        frame_idx=frames,
        new_chest_count=new_count,
        c34_chest_count=old_count,
        every_c34_has_three=all7,
        qualifying_frame=selected,
    )
    matching = {"frame_idx": frames[selected]}
    source_summary = {}
    for name in old_names:
        old = methods[name]
        overlap, distance, by_source, nearest = within_frame_comparison(
            new["pose_xyz"][selected],
            new["sources"][selected],
            old["pose_xyz"][selected],
            old["sources"][selected],
        )
        for suffix, value in (
            ("shared_pectoral_source_view_count", overlap),
            ("pectoral_distance_mm", distance),
            ("unique_source_matched_slot", by_source),
            ("unique_spatial_nearest_slot", nearest),
        ):
            matching[name + "__" + suffix] = value
        lost = missing[selected]
        linked = lost & (by_source >= 0)
        rows, fish = np.where(linked)
        matched_distance = distance[rows, fish, by_source[linked]]
        source_summary[name] = dict(
            missing_chest_fish_frames=int(lost.sum()),
            unique_shared_source_match_count=int(linked.sum()),
            no_shared_source_count=int((lost & (overlap.sum(-1) == 0)).sum()),
            multiple_positive_source_match_count=int((lost & ((overlap > 0).sum(-1) > 1)).sum()),
            source_matched_distance_median_mm=(
                float(np.median(matched_distance)) if len(matched_distance) else None
            ),
            source_matched_distance_p99_mm=(
                float(np.percentile(matched_distance, 99)) if len(matched_distance) else None
            ),
        )
    report = dict(
        schema=STAGE + "_v1",
        status="ok",
        frame_range=[int(frames[0]), int(frames[-1]) + 1],
        native_frames=len(frames),
        selection="new strict chest count <3 AND all seven C34 strict chest counts ==3",
        total_new_missing_chest_fish_frames=int(missing.sum()),
        new_less_than_three_frames=int((new_count < 3).sum()),
        every_c34_three_frames=int(all7.sum()),
        qualifying_frames=int(selected.sum()),
        qualifying_missing_chest_fish_frames=int(qualifying_missing.sum()),
        total_qualifying_episodes=len(spans),
        all_episodes=spans,
        largest_five_episodes=largest,
        missing_chest_direct_view_count=histogram(
            (new["sources"][:, :, :, 1] >= 0).sum(2)[qualifying_missing]
        ),
        missing_chest_direct_mouth=int((qualifying_missing & observed[:, :, 0]).sum()),
        missing_chest_direct_tail=int((qualifying_missing & observed[:, :, 2]).sum()),
        missing_chest_body_supported=int((qualifying_missing & details["body_supported"]).sum()),
        missing_chest_mask_supported=int((qualifying_missing & details["mask_supported"]).sum()),
        missing_chest_review=int((qualifying_missing & new["original_identity_review"]).sum()),
        new_chest_supported_but_mouth_missing_fish_frames=int(
            (~missing & ~observed[:, :, 0]).sum()
        ),
        selected_scene_proposal_kind=histogram(details["proposal_kind"][selected]),
        source_correspondence_diagnostics=source_summary,
        no_qualifying_frames=not bool(selected.any()),
        limitations=[
            "C34 is not truth; seven variants share observations, not seven independent sensors.",
            "All per-slot mappings exist only within one frame; no permanent ID is transferred.",
            "proposal_kind is the selected whole-scene proposal, not a per-fish proposal target.",
            "Saved sources are supported rows after fitting; rejected raw SL plans are absent.",
            "No alternative proposal pool or fit is evaluated. Missing-generation and rejected-fit "
            "causes cannot be separated here.",
            "Body mask support is not strict chest support. Review exclusion changes accepted "
            "availability, not the raw strict-support deficit.",
            "Current mouth-recovery generation requires directly supported chest history/current "
            "chest; its name cannot establish recovery of an unsupported chest.",
        ],
    )
    return report, arrays, matching


def write_frame_details(output, methods, details, selected):
    new = methods["new"]
    rows = []
    with (output / "qualifying_frames.jsonl").open("w", encoding="utf-8") as stream:
        for t in np.flatnonzero(selected):
            row = dict(
                frame=int(new["frame_idx"][t]),
                new_chest_count=int(new["observed"][t, :, 1].sum()),
                observed=new["observed"][t].tolist(),
                sources=new["sources"][t].tolist(),
                identity_review=new["original_identity_review"][t].tolist(),
                c34_chest_counts={
                    name: int(methods[name]["observed"][t, :, 1].sum())
                    for name in comparison.METHODS[1:]
                },
            )
            for key in DETAIL_KEYS:
                value = np.asarray(details[key][t])
                row[key] = (
                    None
                    if value.ndim == 0 and value.dtype.kind == "f" and not np.isfinite(value)
                    else value.tolist()
                )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            for fish in np.flatnonzero(~new["observed"][t, :, 1]):
                rows.append(
                    dict(
                        frame=row["frame"],
                        new_raw_slot=int(fish),
                        chest_direct_views=int((new["sources"][t, fish, :, 1] >= 0).sum()),
                        direct_mouth=bool(new["observed"][t, fish, 0]),
                        direct_tail=bool(new["observed"][t, fish, 2]),
                        body_supported=bool(details["body_supported"][t, fish]),
                        mask_supported=bool(details["mask_supported"][t, fish]),
                        review=bool(new["original_identity_review"][t, fish]),
                        mouth_missing_age=int(details["node_missing"][t, fish, 0]),
                        chest_missing_age=int(details["node_missing"][t, fish, 1]),
                        tail_missing_age=int(details["node_missing"][t, fish, 2]),
                        body_missing_age=int(details["missing"][t, fish]),
                        selected_scene_proposal_kind=str(details["proposal_kind"][t]),
                    )
                )
    fields = (
        list(rows[0])
        if rows
        else [
            "frame",
            "new_raw_slot",
            "chest_direct_views",
            "direct_mouth",
            "direct_tail",
            "body_supported",
            "mask_supported",
            "review",
            "mouth_missing_age",
            "chest_missing_age",
            "tail_missing_age",
            "body_missing_age",
            "selected_scene_proposal_kind",
        ]
    )
    with (output / "missing_chest_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(report):
    lines = [
        "# 160秒・全7 C34共通支持に対する胸部欠測診断",
        "",
        f"新方式の胸部欠測 {report['total_new_missing_chest_fish_frames']} 魚frameのうち、"
        f"全7 C34が3匹を支持する条件に該当するのは {report['qualifying_frames']} frame / "
        f"{report['qualifying_missing_chest_fish_frames']} 魚frameです。"
        f"連続区間は {report['total_qualifying_episodes']} 本。"
        "抽出条件は結果に応じて変えていません。",
        "",
        "|大きい順のnative区間|frame数|胸部欠測魚frame|同時に口直接支持|body支持|review|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["largest_five_episodes"]:
        lines.append(
            f"|{row['frame_range']}|{row['frames']}|{row['missing_chest_fish_frames']}|"
            f"{row['missing_chest_with_direct_mouth']}|{row['missing_chest_with_body_support']}|"
            f"{row['missing_chest_review']}|"
        )
    lines += [
        "",
        f"該当胸部の直接支持view数: {report['missing_chest_direct_view_count']}。"
        f"同時に口直接支持 {report['missing_chest_direct_mouth']}、"
        f"尾直接支持 {report['missing_chest_direct_tail']}、"
        f"mask支持 {report['missing_chest_mask_supported']}、"
        f"body支持 {report['missing_chest_body_supported']}、"
        f"review {report['missing_chest_review']}。",
        "胸部支持は2視点以上の2mm/12px条件で、mask/body支持による代用はしていません。"
        "口のみの欠測は胸部支持率の差を直接説明しません。",
        "既存mouth_recoveryは胸部の直接支持を前提にした口候補です。"
        "今回の選択済みproposal_kindはscene全体の記録で、どの魚を回復しようとしたかは確定できません。",
        "全7条件との同frame source共有数・胸部距離行列を保存しました。"
        "source共有が一意な場合だけその局所対応を記録し、空対応や競合を消去法で埋めていません。"
        "距離最近傍も診断値のみで、全体のID対応や時系列の名称移植には使っていません。",
        "C34を真値とはしていません。全7条件も同じ観測に依存します。"
        "元の未採用SL plan・全候補pool・最適化過程を再評価していないため、"
        "生成不足とfit後の棄却の因果分離はこの保存結果だけでは不可能です。",
        "成功160秒比較と同candidateの診断配列のみ使用。GT・原画・GPU・新fit・閾値変更はありません。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    output.mkdir(parents=True, exist_ok=False)
    evidence = comparison.common.Digests()
    with provenance(STAGE, output, name="diagnosis") as record:
        for path in comparison.code_paths():
            evidence.add(path)
            snapshot = output / "code_snapshot" / path.relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            evidence.add(snapshot, evidence.paths[str(path)])
        evidence.add(Path(comparison.__file__), SOURCE_SHA)
        for path, digest in {**comparison.common.PINS, **comparison.PINS}.items():
            evidence.add(path, digest)
        methods, details = load_inputs(evidence)
        report, selected, matching = analyze(methods, details)
        report["comparison_seal_sha256"] = SEAL_SHA
        report["candidate_sha256"] = comparison.CANDIDATE_SHA
        np.savez_compressed(output / "selection.npz", **selected)
        np.savez_compressed(output / "within_frame_correspondence.npz", **matching)
        write_frame_details(output, methods, details, selected["qualifying_frame"])
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "RESULT.md").write_text(markdown(report), encoding="utf-8")
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                digest = comparison.baseline.sha256(path)
                outputs[str(path)] = digest
                record.output(path.relative_to(output).as_posix(), path, sha256=digest)
        evidence.final_guard()
        comparison.common.COMMON.unchanged(outputs)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(
        json.dumps(
            dict(status="ok", qualifying_frames=report["qualifying_frames"], output=str(output))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
