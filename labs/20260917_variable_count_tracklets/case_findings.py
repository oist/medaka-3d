"""Inspectable event-local diagnostics; never turn scene notes into identity GT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/variable_count_tracklets_v1"
METHODS = ("C27", "S1", "S2", "P0", "P1", "P1aux")


def local_metrics(data, start, stop):
    frames, points, previous = (data[k] for k in ("frame_idx", "points", "previous_row"))
    mask = (frames >= start) & (frames < stop)
    eligible = mask & (frames > start)
    has_previous = previous >= 0
    gap = np.zeros(len(frames), dtype=int)
    gap[has_previous] = frames[has_previous] - frames[previous[has_previous]]
    linked = eligible & (gap == 1)
    step = np.linalg.norm(points[linked, 1] - points[previous[linked], 1], axis=-1)
    n, e, kept = int(mask.sum()), int(eligible.sum()), int(linked.sum())
    return dict(
        rows=n,
        eligible_observations=e,
        adjacent_connections=kept,
        contiguous_runs=n - kept,
        unconnected=e - kept,
        connection_rate=kept / e if e else None,
        abstention_rate=(e - kept) / e if e else None,
        maximum_p_step_px=float(np.nanmax(step)) if len(step) else None,
        confirmed_correct_rate=None,
        confirmed_error_rate=None,
        identity_truth_denominator=0,
    )


def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    human_path = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
    human = json.loads(human_path.read_text(encoding="utf-8"))
    annotations = {r["frame"]: r for r in human["events"]}
    result = []
    for case in protocol["cases"]:
        for center in case.get("centers", []):
            for view in ("top_cam", "side1_cam", "side2_cam"):
                data = {}
                for method in METHODS:
                    with np.load(OUT / case["name"] / f"tracking_{method}_{view}.npz") as z:
                        data[method] = {k: z[k] for k in z.files}
                start, stop = center - 8, center + 9
                frames = data["P1"]["frame_idx"]
                current = (frames > start) & (frames < stop)
                base, new = data["P0"]["previous_row"], data["P1"]["previous_row"]
                disagreement = dict(
                    removed_p0_links=int((current & (base >= 0) & (new < 0)).sum()),
                    added_p1_links=int((current & (base < 0) & (new >= 0)).sum()),
                    changed_predecessors=int(
                        (current & (base >= 0) & (new >= 0) & (base != new)).sum()
                    ),
                )
                metrics = {
                    method: local_metrics(value, start, stop) for method, value in data.items()
                }
                n0, n1 = metrics["P0"]["contiguous_runs"], metrics["P1"]["contiguous_runs"]
                statement = (
                    f"局所17フレームの連続区間数はP0 {n0}→P1 {n1}。"
                    f"P0からの追加接続{disagreement['added_p1_links']}、"
                    f"切断{disagreement['removed_p0_links']}、"
                    f"前任観測の変更{disagreement['changed_predecessors']}。"
                    "個体対応の正誤はこの集計だけでは確定しない。"
                )
                remaining = (
                    "2Dが連続していてもビュー間割当の失敗は残り得る。3D再接続は工程Dで別途評価。"
                )
                if center == 30735 and view == "side2_cam":
                    remaining = (
                        "旧レビューはside2の3D slotの一時ジャンプ。単一検出の2D連続性と"
                        "3Dへの割当変更を区別し、この2D出力のみで修正済みとしない。"
                    )
                if center in (76791, 323819, 77415):
                    remaining = (
                        "人手のno-swap対照。カメラと表示パネルの指定は一部未確定で、"
                        "各2D行の正解IDには変換できない。追加接続・切断を誤修正扱いしないための確認対象。"
                    )
                key = f"{view}__f{center}"
                result.append(
                    dict(
                        case=case["name"],
                        view=view,
                        center=center,
                        local_start=start,
                        local_stop=stop,
                        local_frame_count=17,
                        metrics=metrics,
                        p0_to_p1=disagreement,
                        finding=statement,
                        human_review=annotations.get(center),
                        historical_scene_note=case.get("note"),
                        annotation_scope="event anchor only; not per-row 2D identity ground truth",
                        identity_verdict="unresolved",
                        reflection_verdict="unclassified",
                        remaining_3d=remaining,
                        review_question=(
                            "前後の全検出画像と低速動画で、同じ魚のP観測が継続しているか。"
                            "変更がある場合はnative frameとglobal_rowの組で、"
                            "正しい継続・誤接続・必要な保留・不要な切断を判定してください。"
                        ),
                        video=f"review/{key}.mp4",
                        images=[
                            f"review/{key}__{part}.png" for part in ("before", "center", "after")
                        ],
                    )
                )
    with provenance("variable_count_case_findings", OUT) as rec:
        rec.input("review", human_path, sha256=sha256(human_path))
        rec.input("protocol", OUT / "protocol.json", sha256=sha256(OUT / "protocol.json"))
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        target = OUT / "case_findings.json"
        target.write_text(
            json.dumps(
                dict(records=result, new_human_annotations_created=False),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        rec.output("findings", target, sha256=sha256(target))
    print(f"Saved {len(result)} case/view review packets")


if __name__ == "__main__":
    main()
