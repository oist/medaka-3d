"""CPU-only fixed-shape mixture-vs-squared color scores on all twelve saved top probes."""

import csv
import dataclasses
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from pixel_appearance_loss import PixelAppearanceConfig, pixel_appearance_loss  # noqa: E402
from pixel_template_assignment import assignment_costs  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

SOURCE = ROOT / "work/three_body/pixel_template_assignment_fixed_probes_v3"
OUT = ROOT / "work/three_body/pixel_mixture_fixed_probes_v1"
FRAMES = np.array(
    [78331, 78332, 78446, 79061, 79108, 79109, 79112, 79236, 79316, 79372, 79405, 79547],
    dtype=np.int64,
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def direct_numpy(features, templates, alpha, weights, permutations, cfg):
    """Independent direct mixture arithmetic; no call to torch or the loss implementation."""
    q0 = np.exp(-cfg.background_cost)
    union = alpha.sum(axis=-1)
    values = []
    for permutation in permutations:
        delta = features[:, None] - templates[permutation][None]
        quality = np.exp(-np.einsum("nfd,nfd->nf", delta, delta) / (2 * cfg.sigma**2))
        probability = cfg.outlier_fraction * q0 + (1 - cfg.outlier_fraction) * (
            np.maximum(1 - union, 0) * q0 + np.einsum("nf,nf->n", alpha, quality)
        )
        assert np.isfinite(probability).all() and (probability > 0).all()
        loss = -np.log(np.minimum(probability, 1))
        values.append(float(np.dot(weights, loss) / weights.sum()) if weights.sum() else 0.0)
    return np.asarray(values)


def main():
    OUT.mkdir(exist_ok=False)
    seal_path = SOURCE / "diagnostic.provenance.json"
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    bindings = {
        str(Path(item["path"]).resolve()): item["sha256"] for item in seal["outputs"].values()
    }
    paths = [
        SOURCE / "summary.json",
        SOURCE / "scores.npz",
        *(SOURCE / f"pixels_f{frame}.npz" for frame in FRAMES),
    ]
    for path in paths:
        assert sha(path) == bindings[str(path.resolve())]
    source_summary = json.loads((SOURCE / "summary.json").read_text())
    assert source_summary["fixed_frames"] == FRAMES.tolist()
    assert source_summary["all_eroded_union_pixels_retained"] is True
    assert source_summary["renderer_device"] == "cpu"
    with np.load(SOURCE / "scores.npz", allow_pickle=False) as arrays:
        old = {key: arrays[key] for key in arrays.files}
    assert np.array_equal(old["frame_idx"], FRAMES)
    sources = [
        Path(__file__),
        LAB / "pixel_appearance_loss.py",
        LAB / "pixel_template_assignment.py",
        ROOT / "src/medaka3d/provenance.py",
    ]
    paths.extend([seal_path, *sources])
    hashes = {str(path): sha(path) for path in paths}
    config = PixelAppearanceConfig()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with provenance("fixed_top_probe_pixel_mixture_scores", OUT, name="diagnostic") as record:
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            record.param("config", dataclasses.asdict(config))
            record.param("fixed_frames", FRAMES.tolist())
            record.param("device", "cpu")
            record.param("dtype", "float64")
            record.param("view", "top_cam only, matching the sealed source diagnostic")
            record.param("template_use", "historical diagnostic only; not online166 enrollment")
            record.param("no_new_render_sampling_or_tracking", True)
            cases, saved, csv_rows = [], {}, []
            for index, frame in enumerate(FRAMES):
                started = time.perf_counter()
                path = SOURCE / f"pixels_f{frame}.npz"
                with np.load(path, allow_pickle=False) as arrays:
                    data = {key: arrays[key] for key in arrays.files}
                assert int(data["frame_idx"]) == frame
                assert np.array_equal(data["templates"], old["templates"])
                assert np.array_equal(data["pose_xyz"], old["pose_xyz"][index])
                assert np.array_equal(data["permutations"], old["permutations"][index])
                assert data["features"].shape == (len(data["pixels_xy"]), 6)
                original_row = source_summary["frames"][index]
                assert original_row["frame"] == frame
                assert len(data["pixels_xy"]) == original_row["eroded_union_pixels"]
                alpha = data["visible_alpha"]
                valid = data["ray_valid"]
                assert valid.dtype == bool and not alpha[~valid].any()
                np.testing.assert_allclose(alpha.sum(-1), data["union_alpha"], atol=2e-15, rtol=0)
                old_again = assignment_costs(data["features"], data["templates"], alpha)
                np.testing.assert_array_equal(old_again["scores"], old["scores"][index])
                np.testing.assert_array_equal(
                    old_again["fish_effective_mass"], old["fish_effective_mass"][index]
                )
                available = bool(original_row["view_available"])
                # All saved core pixels stay in the artifact. Invalid rays are unknown,
                # not evidence of a background-colored animal, so they carry zero weight.
                weights = valid.astype(np.float64) * available
                tensor_alpha = torch.from_numpy(alpha[None, None])
                tensor_features = torch.from_numpy(data["features"][None])
                tensor_weights = torch.from_numpy(weights[None])
                enabled = torch.tensor([available], dtype=torch.bool)
                mixture_scores = []
                with torch.no_grad():
                    for permutation in data["permutations"]:
                        template = torch.from_numpy(data["templates"][permutation][None])
                        result = pixel_appearance_loss(
                            tensor_alpha, tensor_features, template, tensor_weights, enabled, config
                        )
                        assert result.loss.device.type == "cpu"
                        mixture_scores.append(result.loss.item())
                    hidden = pixel_appearance_loss(
                        torch.zeros_like(tensor_alpha),
                        tensor_features,
                        torch.from_numpy(data["templates"][None]),
                        tensor_weights,
                        enabled,
                        config,
                    ).loss.item()
                mixture_scores = np.asarray(mixture_scores)
                independent = direct_numpy(
                    data["features"],
                    data["templates"],
                    alpha,
                    weights,
                    data["permutations"],
                    config,
                )
                np.testing.assert_allclose(mixture_scores, independent, atol=2e-12, rtol=2e-12)
                assert abs(hidden - (config.background_cost if weights.sum() else 0)) < 2e-12
                old_order = old_again["rank_order"]
                new_order = np.argsort(mixture_scores, kind="stable")
                old_rank = np.argsort(old_order)
                new_rank = np.argsort(new_order)
                mass = alpha.sum(axis=0)
                core = len(alpha)
                fraction = mass / core if core else np.zeros(3)
                union = alpha.sum(axis=1)
                quality_distance = ((data["features"][:, None] - data["templates"][None]) ** 2).sum(
                    -1
                )
                quality = np.exp(-quality_distance / (2 * config.sigma**2))
                case = dict(
                    frame=int(frame),
                    view="top_cam",
                    source_pixel_npz_sha256=hashes[str(path)],
                    source_native_bgr_sha256=original_row["bgr_sha256"],
                    source_all_mask_sha256=original_row["all_masks_sha256"],
                    source_core_sha256=original_row["core_sha256"],
                    view_available=available,
                    all_core_pixels=core,
                    valid_rays=int(valid.sum()),
                    foreground_denominator=float(weights.sum()),
                    permutations=data["permutations"],
                    old_squared_scores=old_again["scores"],
                    new_mixture_scores=mixture_scores,
                    independent_numpy_scores=independent,
                    independent_max_abs_error=float(np.max(np.abs(mixture_scores - independent))),
                    old_rank_order=old_order,
                    new_rank_order=new_order,
                    old_rank_by_permutation=old_rank,
                    new_rank_by_permutation=new_rank,
                    full_order_same=np.array_equal(old_order, new_order),
                    lowest_index_same=int(old_order[0]) == int(new_order[0]),
                    new_best_second_gap=float(
                        mixture_scores[new_order[1]] - mixture_scores[new_order[0]]
                    ),
                    new_full_score_spread=float(np.ptp(mixture_scores)),
                    fish_visible_mass_pixel_equivalents=mass,
                    fish_mean_visible_fraction_of_all_core=fraction,
                    fish_exact_zero_mass=mass == 0,
                    fish_at_most_one_pixel_equivalent=mass <= 1,
                    fish_at_most_one_thousandth_of_core=fraction <= 0.001,
                    union_mass=float(union.sum()),
                    completely_uncovered_pixels=int((union == 0).sum()),
                    mean_uncovered_fraction=float(np.mean(np.maximum(1 - union, 0)))
                    if core
                    else None,
                    best_minus_all_hidden=float(mixture_scores[new_order[0]] - hidden),
                    hidden_scene_score=hidden,
                    pixel_quality_qf_quantiles=np.quantile(
                        quality, [0, 0.25, 0.5, 0.75, 1], axis=0
                    ),
                    seconds_cpu_scoring=time.perf_counter() - started,
                )
                cases.append(case)
                for permutation_index, permutation in enumerate(data["permutations"]):
                    csv_rows.append(
                        dict(
                            frame=int(frame),
                            permutation_index=permutation_index,
                            shape_slot_to_template="".join(map(str, permutation)),
                            old_score=old_again["scores"][permutation_index],
                            new_score=mixture_scores[permutation_index],
                            old_rank=int(old_rank[permutation_index]) + 1,
                            new_rank=int(new_rank[permutation_index]) + 1,
                            visible_mass_S0=mass[0],
                            visible_mass_S1=mass[1],
                            visible_mass_S2=mass[2],
                            core_pixels=core,
                            fixed_valid_foreground_denominator=weights.sum(),
                        )
                    )
                # Preserve reused pixels/features/alpha/templates and scores exactly.
                copied = OUT / path.name
                shutil.copyfile(path, copied)
                assert sha(copied) == hashes[str(path)]
                record.output(path.name, copied, sha256=sha(copied))
                saved[f"f{frame}_fgweights"] = weights
                print(
                    json.dumps(
                        dict(
                            frame=int(frame),
                            old_order=old_order.tolist(),
                            new_order=new_order.tolist(),
                            min_mass=float(mass.min()),
                            best_mixture=float(mixture_scores.min()),
                        )
                    ),
                    flush=True,
                )
            arrays_path = OUT / "mixture_scores.npz"
            np.savez_compressed(
                arrays_path,
                frame_idx=FRAMES,
                templates=old["templates"],
                permutations=old["permutations"],
                old_squared_scores=old["scores"],
                new_mixture_scores=np.array([c["new_mixture_scores"] for c in cases]),
                old_rank_order=old["rank_order"],
                new_rank_order=np.array([c["new_rank_order"] for c in cases]),
                fish_visible_mass=old["fish_effective_mass"],
                fixed_foreground_denominator=np.array([c["foreground_denominator"] for c in cases]),
                **saved,
            )
            csv_path = OUT / "all_72_assignment_scores.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
            summary = dict(
                status="completed",
                fixed_frames=FRAMES,
                config=dataclasses.asdict(config),
                input_sha256=hashes,
                source_and_input_unchanged=True,
                runtime=dict(
                    device="cpu",
                    dtype="float64",
                    torch=str(torch.__version__),
                    torch_threads=torch.get_num_threads(),
                    numpy=np.__version__,
                    python=sys.version,
                    platform=platform.platform(),
                ),
                cases=cases,
                comparison=dict(
                    number_of_frames=len(cases),
                    number_of_assignments=len(csv_rows),
                    same_full_rank_order=sum(c["full_order_same"] for c in cases),
                    same_lowest_score_index=sum(c["lowest_index_same"] for c in cases),
                    max_independent_formula_error=max(
                        c["independent_max_abs_error"] for c in cases
                    ),
                    best_mixture_above_hidden_count=sum(
                        c["best_minus_all_hidden"] > 0 for c in cases
                    ),
                ),
                limitations=[
                    "Top view only, on the exact prior all-mask union core pixels.",
                    "Saved latent shapes and historical templates are diagnostic inputs only.",
                    "No pose optimization, tracking, GPU, label assignment, or review change.",
                    "All six permutations are retained; lowest rank is not an ID decision.",
                    "Old denominator is visible mass; new denominator is fixed valid core pixels.",
                    "Scores use different units and their absolute magnitudes are not comparable.",
                    "Tiny visible mass limits ID evidence even when a lexical rank is returned.",
                    "Mass flags <=1 pixel or <=0.001 core fraction are descriptive only.",
                    "sigma=.1, c0=2, rho=.1 are uncalibrated fixed diagnostic settings.",
                    "Future enrollment uses online166 observations, not these old templates.",
                    "Native BGR/core SHA bindings are inherited through the sealed pixel NPZ.",
                    "No source video is reread or rendered; this is score reuse of sealed data.",
                ],
            )
            summary_path = OUT / "summary.json"
            summary_path.write_text(
                json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding="utf-8"
            )
            note = [
                "# 固定12時点の画素色mixture診断",
                "",
                "同じ保存3D形状・上面union-core画素・旧テンプレートに対するスコア比較。"
                "再追跡やID判定は行っていない。sigma=.1、c0=2、rho=.1を固定した。",
                "",
                f"全6順位が同じ時点は{summary['comparison']['same_full_rank_order']}/12、"
                f"最低スコアの割当indexが同じ時点は{summary['comparison']['same_lowest_score_index']}/12。"
                "最低順位の比較は目的関数の違いを示すだけで、IDの正しさを評価しない。",
                "",
                "| frame | 旧最低順列 | 新最低順列 | 新1位-2位差 | S0 mass | S1 mass | S2 mass |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
            for case in cases:
                old_perm = case["permutations"][case["old_rank_order"][0]]
                new_perm = case["permutations"][case["new_rank_order"][0]]
                a, b, c = case["fish_visible_mass_pixel_equivalents"]
                note.append(
                    f"| {case['frame']} | {old_perm.tolist()} | {new_perm.tolist()} | "
                    f"{case['new_best_second_gap']:.5g} | {a:.5g} | {b:.5g} | {c:.5g} |"
                )
            note += [
                "",
                "順列は保存shape slot→template index。massの単位は完全に見える画素の等価数。"
                "小さいmassを十分なID証拠とはみなせない。全72組はCSV/NPZに残した。",
                "",
                "新lossは固定した有効core画素数を分母にし、未被覆のforegroundにもq0が残る。"
                "全hidden時のスコアは2で、空の観測のみ0となる。旧lossはvisible massで割るため、"
                "両者の値の大きさは直接比較できない。",
                "",
                f"新最低lossでも全hiddenの2より大きい時点は"
                f"{summary['comparison']['best_mixture_above_hidden_count']}/12。"
                "その場合は、この色項単独では現在の色が合わない形状を隠す方向が有利になり得る。"
                "この挙動もそのまま保存し、閾値・重みの調整やtrackingへの採用はしていない。",
                "",
                f"独立NumPy直接式との最大差は"
                f"{summary['comparison']['max_independent_formula_error']:.5g}。"
                "旧二乗距離スコアは元配列と完全一致。旧pixel NPZのsealを検査して複製し、"
                "入力/コードの実行前後SHA一致を確認。元rawのnative BGR/core SHAは旧sealから継承。",
                "",
                "これはtop_camのみの条件付き診断で、3視点統合や独立ID ground truthではない。"
                "次のtracking用の初期166frameオンライン登録とはテンプレートの由来が異なる。",
            ]
            report_path = OUT / "RESULT.md"
            report_path.write_text("\n".join(note) + "\n", encoding="utf-8")
            for path in (arrays_path, csv_path, summary_path, report_path):
                record.output(path.name, path, sha256=sha(path))
            for source in sources:
                target = OUT / "source_snapshot" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                assert sha(target) == hashes[str(source)]
                record.output(str(target.relative_to(OUT)), target, sha256=sha(target))
            assert all(sha(path) == digest for path, digest in hashes.items())
            print(json.dumps(jsonable(summary["comparison"])), flush=True)
    finally:
        torch.set_num_threads(original_threads)


if __name__ == "__main__":
    main()
