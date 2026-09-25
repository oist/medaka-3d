"""Saved-array interpretation of the fixed 33-frame mouth-loss episode.

Post-hoc arithmetic only; no geometry is refit, projected, accepted or relabeled.
"""

import json
from collections import Counter
from pathlib import Path

import diagnose_directional_mouth_first_chunk as diagnosis
import numpy as np

from medaka3d.provenance import provenance

checks = diagnosis.checks
ROOT = diagnosis.REPO / "work/three_body/directional_mouth_first_chunk_diagnosis_v1"
SEAL_SHA = "e87d4c2c137ca09ced694efd8a96381b8e7b1d95555b539546575cd9dd3ee609"
FIRST, STOP, SLOT = 67663, 67696, 0


def numbers(values):
    return dict(
        minimum=float(values.min()),
        median=float(np.median(values)),
        mean=float(values.mean()),
        maximum=float(values.max()),
    )


def source_structure(rows):
    """Only two or more present nodes can reveal agreement/disagreement."""
    count = np.sum(rows >= 0, axis=1)
    unique = np.array([len(set(row[row >= 0])) for row in rows])
    tuples = Counter(tuple(map(int, row)) for row in rows)
    return dict(
        frames=len(rows),
        no_or_one_present_node=int((count < 2).sum()),
        two_present_same_row=int(((count == 2) & (unique == 1)).sum()),
        two_present_mixed_rows=int(((count == 2) & (unique > 1)).sum()),
        all_three_same_row=int(((count == 3) & (unique == 1)).sum()),
        all_three_mixed_rows=int(((count == 3) & (unique > 1)).sum()),
        mouth_chest_tail_row_patterns=[
            dict(rows=list(key), frames=value) for key, value in sorted(tuples.items())
        ],
    )


def main():
    output = ROOT / "interpretation_v1"
    checks.require(not output.exists(), "new interpretation output required")
    hashes = {
        str(Path(__file__).resolve()): checks.sha(__file__),
        str(Path(diagnosis.__file__).resolve()): checks.sha(diagnosis.__file__),
        str(Path(checks.__file__).resolve()): checks.sha(checks.__file__),
    }
    checks.require(checks.sha(ROOT / "diagnosis.provenance.json") == SEAL_SHA, "wrong diagnosis")
    checks.verify(ROOT, "diagnosis.provenance.json", diagnosis.STAGE, hashes)
    data = {}
    for name in ("original", "directional"):
        with np.load(ROOT / f"{name}_same_horizon.npz", allow_pickle=False) as stored:
            data[name] = {
                key: stored[key].copy()
                for key in ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")
            }
    keep = (data["original"]["frame_idx"] >= FIRST) & (data["original"]["frame_idx"] < STOP)
    old, new = data["original"], data["directional"]
    checks.require(
        keep.sum() == 33
        and old["observed"][keep, SLOT, 0].all()
        and not new["observed"][keep, SLOT, 0].any(),
        "fixed lost episode differs",
    )
    delta = new["pose_xyz"][keep, SLOT] - old["pose_xyz"][keep, SLOT]
    structure = {
        name: {
            view: source_structure(value["sources"][keep, SLOT, vi])
            for vi, view in enumerate(diagnosis.VIEWS)
        }
        for name, value in data.items()
    }
    with np.load(ROOT / "old_mouth_rows_at_new_pose.npz", allow_pickle=False) as stored:
        ray, pixel = (
            stored["ray_distance_mm"][keep, SLOT, :, 0],
            stored["native_pixel_error"][keep, SLOT, :, 0],
        )
    optical = {}
    for vi, view in enumerate(diagnosis.VIEWS):
        finite = np.isfinite(ray[:, vi]) & np.isfinite(pixel[:, vi])
        optical[view] = dict(
            assigned_old_rows=int(finite.sum()),
            ray_mm=numbers(ray[finite, vi]),
            pixels=numbers(pixel[finite, vi]),
        )
    related_runs = {
        name: [
            row
            for row in diagnosis.runs(~value["observed"][..., 0])
            if row["slot"] == SLOT and row["start"] < STOP and row["stop_exclusive"] > 67647
        ]
        for name, value in data.items()
    }
    stats = dict(
        schema="directional_mouth_first_chunk_interpretation_v1",
        native_interval=[FIRST, STOP],
        slot=SLOT,
        selected_after_saved_diagnosis=True,
        same_horizon_stop=diagnosis.STOP,
        node_displacement_mm={
            node: numbers(np.linalg.norm(delta[:, ni], axis=1))
            for ni, node in enumerate(("mouth", "chest", "tail"))
        },
        node_delta_xyz_mean_mm=delta.mean(axis=0).tolist(),
        source_structure=structure,
        old_mouth_rows_at_new_pose=optical,
        related_mouth_missing_runs=related_runs,
        node_supported_frames={
            name: value["observed"][keep, SLOT].sum(axis=0).tolist() for name, value in data.items()
        },
        reviewed_frames={
            name: int(value["identity_review"][keep, SLOT].sum()) for name, value in data.items()
        },
        source_rows_are_per_view_local=True,
        missing_nodes_are_not_row_agreement=True,
        direct_causal_link_to_67647_not_established=True,
        anatomical_or_biological_identity_truth=False,
        dense_identity_modified=False,
    )
    mouth, chest = stats["node_displacement_mm"]["mouth"], stats["node_displacement_mm"]["chest"]
    lines = [
        "# 口支持減少の保存配列による追補",
        "",
        "口支持は4569→4536 / 4980。lost34件のうち33件が S0 [67663,67696) に集中する。"
        "残るlostはS2の67076、gainはS0の67635。",
        "",
        f"33フレームの新旧口3D距離は {mouth['minimum']:.3f}-{mouth['maximum']:.3f} mm "
        f"(中央値 {mouth['median']:.3f})、"
        f"胸部距離は {chest['minimum']:.3f}-{chest['maximum']:.3f} mm "
        f"(中央値 {chest['median']:.3f})。",
        "この区間の口/胸/尾支持数は旧33/32/28、新0/33/33。各方式とも33フレームすべて元reviewが残る。",
        "",
        "旧口sourceを新口位置に当てるとtopは33件すべて0.008-0.249mm、0.044-1.444pxだが、"
        "side1は33件すべて7.158-8.091mm、29.342-33.193px。"
        "旧side2 sourceがある15件も7.283-8.199mm、37.462-42.104pxで支持外となる。"
        "新保存口sourceはtopの1視点だけである。単一top視点に整合する口位置でも側面支持が失われる事例であり、"
        "旧位置が生物学的正解だとは判定していない。",
        "",
        "|方式|view|0-1 node|2 node同row|2 node混成|3 node同row|3 node混成|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, views in structure.items():
        for view, row in views.items():
            cells = [
                row[key]
                for key in (
                    "no_or_one_present_node",
                    "two_present_same_row",
                    "two_present_mixed_rows",
                    "all_three_same_row",
                    "all_three_mixed_rows",
                )
            ]
            lines.append(f"|{name}|{view}|" + "|".join(map(str, cells)) + "|")
    lines += [
        "",
        "このrow番号は各view内の元SLEAP instance番号であり、view間で同番号の魚とはみなさない。"
        "全組合せ・各フレームのsource行はJSON/NPZへ保存。2node一致は欠測した3番目まで一致した意味ではない。"
        "混成は元SLEAP groupingと保存assignmentの不一致を示すだけで、真の魚取り違えを確定しない。",
        "",
        "33フレームは新旧の支持差の区間であり、新口欠測全体の長さではない。新S0の口欠測は"
        "[67636,67779)の143フレーム。旧は[67635,67663)の28フレーム欠測後にこの33フレームで支持を回復する。",
        "",
        "S2の67076では、新口に旧side1 row0を当てても1.712mm/9.989pxで支持可能で、"
        "他slotとのsource衝突もない。一方、新保存sourcesはside1を持たない。"
        "保存配列はfit後支持であり、fit前plan欠落とfit後除外・枝選択をこの情報だけでは区別できない。",
        "",
        "67647の胸部再取得から16フレーム後に差が始まるという時系列だけで直接原因とは断定しない。"
        "新旧の候補履歴・姿勢が分岐した後の局所診断である。位置fit、閾値変更、原画再decode、"
        "GPU、GT採点、ID採用はしていない。1分全体の完走・最終backtraceの結論も含まない。",
        "",
    ]
    output.mkdir()
    with provenance(
        "directional_mouth_first_chunk_interpretation", output, name="report"
    ) as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        source = output / Path(__file__).name
        source.write_bytes(Path(__file__).read_bytes())
        (output / "summary.json").write_text(
            json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
        np.savez_compressed(
            output / "episode.npz",
            frame_idx=old["frame_idx"][keep],
            old_pose=old["pose_xyz"][keep, SLOT],
            new_pose=new["pose_xyz"][keep, SLOT],
            old_sources=old["sources"][keep, SLOT],
            new_sources=new["sources"][keep, SLOT],
            old_observed=old["observed"][keep, SLOT],
            new_observed=new["observed"][keep, SLOT],
            old_mouth_at_new_pose_ray_mm=ray,
            old_mouth_at_new_pose_pixel_error=pixel,
        )
        for path in output.iterdir():
            record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "interpretation input changed")
        for item in record.outputs.values():
            checks.require(
                checks.sha(item["path"]) == item["sha256"], "interpretation output changed"
            )
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
