"""Attribute finite-pool support losses after an exact, completed diagnostic replay.

Read-only arithmetic. Original future backtrace is used only to identify the
already saved reference winner, never to generate proposals or certify identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
from tracker_checkpoint import read_checkpoint  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FIRST, START, STOP = 49800, 66656, 66706


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root, filename, stage, hashes):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(REPO / "work"), "artifact escaped workspace work")
    path = root / filename
    seal = read(path)
    require(seal["status"] == "ok" and seal["stage"] == stage, "successful stage required")
    hashes[str(path)] = sha(path)
    for item in seal["outputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        require(
            path.is_relative_to(root)
            and path.stat().st_size == item["size"]
            and sha(path) == item["sha256"],
            "sealed output changed",
        )
        hashes[str(path)] = item["sha256"]
    return seal


def equal_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def summarize(root, output):
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    run = verify(root, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes)
    require(
        all(str(root / name) in hashes for name in ("summary.json", "history_comparison.json")),
        "unsealed replay summary or comparisons",
    )
    status = read(root / "summary.json")
    require(
        status["replay_complete"]
        and status["status"] == "original_all_beam_replay_exact"
        and status["exact_frames"] == 306
        and status["diagnostic_frames"] == STOP - START,
        "all 306 native frames and 50 diagnostic frames must be exact",
    )
    comparisons = read(root / "history_comparison.json")
    require(
        len(comparisons["rows"]) == 306 and all(r["exact"] for r in comparisons["rows"]),
        "not all original ordered beams were reproduced",
    )
    reference = Path(comparisons["reference"]).resolve(strict=True)
    reference_seal_path = reference / "continuation.provenance.json"
    require(
        run["inputs"][str(reference_seal_path)]["sha256"] == sha(reference_seal_path),
        "reference does not match the diagnostic replay input",
    )
    original = verify(
        reference, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    for path, digest in original["params"]["bindings"]["code_sha256"].items():
        require(sha(path) == digest, "original checkpoint decoder dependency changed")
        hashes[path] = digest
    history, metadata, _ = read_checkpoint(
        reference / "checkpoint", expected_bindings=original["params"]["bindings"]
    )
    count = metadata["processed_frames"]
    require(count == 76360 - FIRST, "unexpected reference endpoint")
    offsets = history["history_offsets"]
    chosen = np.empty(count, np.int64)
    branch = 0
    for index in range(count - 1, -1, -1):
        require(0 <= branch < offsets[index + 1] - offsets[index], "invalid saved backtrace")
        chosen[index] = branch
        branch = int(history["history_parent"][offsets[index] + branch])
    rows, fish_rows, categories = [], [], Counter()
    for native in range(START, STOP):
        folder = root / "diagnostics" / f"frame_{native:06d}"
        require(str(folder / "arrays.npz") in hashes, "unsealed diagnostic candidate arrays")
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        selected = a["selected_indices"]
        beam = int(chosen[native - FIRST])
        candidate = int(selected[beam])
        original_row = int(offsets[native - FIRST] + beam)
        for field, key in (
            ("pose_xyz", "fitted_pose"),
            ("sources", "fitted_geometric_sources"),
            ("observed", "fitted_geometric_observed"),
        ):
            require(
                equal_bytes(history["history_" + field][original_row], a[key][candidate]),
                "reference winner does not match diagnostic selected candidate",
            )
        generated, fitted = (
            a["generated_strict_observed"][:, :, 1],
            a["fitted_strict_observed"][:, :, 1],
        )
        require(not a["generated_plan_source_collision"].any(), "non-exclusive original plan")
        gen_any = bool(generated.all(1).any())
        fit_any = bool(fitted.all(1).any())
        selected_any = bool(fitted[selected].all(1).any())
        winner_all = bool(fitted[candidate].all())
        category = (
            "reference_winner_has_all_three"
            if winner_all
            else "supported_selected_alternative_exists"
            if selected_any
            else "fitted_support_exists_but_no_selected_scene"
            if fit_any
            else "generated_scene_support_lost_in_fit"
            if gen_any
            else "no_generated_or_fitted_all_three_scene"
        )
        categories[category] += 1
        rows.append(
            dict(
                frame_idx=native,
                candidates=len(fitted),
                reference_winner_beam=beam,
                reference_winner_candidate=candidate,
                generated_all_three=gen_any,
                fitted_all_three=fit_any,
                selected_all_three=selected_any,
                reference_winner_all_three=winner_all,
                category=category,
            )
        )
        group = (a["proposal_parent"] == a["proposal_parent"][candidate]) & (
            a["proposal_permutation"] == a["proposal_permutation"][candidate]
        ).all(1)
        selected_group = group & a["selected_mask"]
        for fish in np.flatnonzero(~fitted[candidate]):
            fish_rows.append(
                dict(
                    frame_idx=native,
                    slot=int(fish),
                    winner_parent=int(a["proposal_parent"][candidate]),
                    winner_candidate=candidate,
                    same_parent_and_permutation_candidates=int(group.sum()),
                    generated_support_same_group=bool(generated[group, fish].any()),
                    fitted_support_same_group=bool(fitted[group, fish].any()),
                    selected_support_same_group=bool(fitted[selected_group, fish].any()),
                    generated_support_any_group=bool(generated[:, fish].any()),
                    fitted_support_any_group=bool(fitted[:, fish].any()),
                    winner_local_cost=float(a["local_cost"][candidate]),
                    winner_path_cost=float(a["path_cost"][candidate]),
                )
            )
    summary = dict(
        schema="chest_candidate_replay_support_attribution_v1",
        status="ok",
        native_interval=[START, STOP],
        frames=len(rows),
        reference_winner_missing_chest_fish_frames=len(fish_rows),
        all_three_scene_categories=dict(categories),
        all_three_scene_frames={
            key: sum(row[key] for row in rows)
            for key in (
                "generated_all_three",
                "fitted_all_three",
                "selected_all_three",
                "reference_winner_all_three",
            )
        },
        same_group_missing_chest_facts={
            key: sum(row[key] for row in fish_rows)
            for key in (
                "generated_support_same_group",
                "fitted_support_same_group",
                "selected_support_same_group",
            )
        },
        independent_biological_ID_truth=False,
        threshold_changed=False,
        additional_fit=False,
        scope=("Finite returned candidates. Same parent/permutation is a computational group, "
               "not ID truth."),
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "chest_candidate_replay_support_attribution", output, name="analysis"
    ) as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name, values in (("frames.csv", rows), ("missing_chests.csv", fish_rows)):
            with (output / name).open("w", encoding="utf-8", newline="") as stream:
                if values:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
        lines = [
            "# 胸部支持が失われる段階",
            "",
            "元checkpointからの306フレームが全候補履歴で再現した後、対象50フレームを集計。",
            "保存された160秒時点の勝者は、既存履歴から評価目的にだけ参照した。",
            "",
            "|3匹すべての胸部支持を持つ場面候補|フレーム数|",
            "|---|---:|",
        ]
        for label, key in (
            ("生成時の候補に存在", "generated_all_three"),
            ("fit後の候補に存在", "fitted_all_three"),
            ("保持された6候補に存在", "selected_all_three"),
            ("保存済み160秒の勝者", "reference_winner_all_three"),
        ):
            lines.append(f"|{label}|{summary['all_three_scene_frames'][key]}/50|")
        lines += [
            "",
            "これは観測支持と候補選択の診断であり、生物IDの正解判定ではない。",
            "候補生成が返す有限集合を調べたもので、棄却前の全交点や最適化の全途中状態は含まない。",
            "同じ親・同じpermutationに絞った欠測魚の内訳は missing_chests.csv に保持した。",
        ]
        (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha(path))
        for path, digest in hashes.items():
            require(sha(path) == digest, "evidence changed during summary")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    require(
        target.is_relative_to(REPO / "work") and not target.exists(),
        "new workspace output required",
    )
    summarize(args.replay.resolve(), target)
