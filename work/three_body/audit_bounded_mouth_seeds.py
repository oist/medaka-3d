"""Audit historical code bindings and saved fixed202 proposals; no seed solve or fit."""

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import explain_directional_mouth_intersections as prior
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

checks = prior.checks
REPO = prior.prior.REPO
ROOT = REPO / "work/three_body/bounded_mouth_seed_fixed202_v1"
STAGE = "bounded_mouth_seed_fixed202_independent_audit_v1"
EVAL_SHA = "e316da571ff479e384ad9bfd8d69c31da46ad509b3d19bd2d9c52683c6442c96"
SOLVER_SHA = "f8a173486394fffb96f9fb8726a981b37101f3957d9ee00ec88752893f860728"


def sealed(root, name, stage, digest, hashes):
    checks.require(checks.sha(root / name) == digest, "pinned seal changed")
    result = checks.verify(root, name, stage, hashes)
    checks.require(hashes[str(root / name)] == digest, "seal changed during verification")
    return result


def required(path, hashes):
    path = Path(path).resolve()
    checks.require(str(path) in hashes, "unsealed required input: " + str(path))
    return path


def same(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    checks.require(actual.shape == expected.shape and np.array_equal(actual, expected), name)


def near(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected, float)
    checks.require(
        actual.shape == expected.shape
        and np.allclose(actual, expected, atol=1e-10, rtol=0, equal_nan=True),
        name,
    )


def support(candidate, plan, data, rig):
    """Independent node/view loops with the frozen float32 ray cache and 2mm/12px."""
    sources = np.full((3, 3, 3), -1, np.int32)
    distance = np.empty((3, 3, 3))
    pixels = np.full((3, 3, 3), np.nan)
    for view, name in enumerate(prior.prior.VIEWS):
        projected = rig.camera(name).project(candidate)
        for node in range(3):
            used = plan[:, view, node]
            used = used[used >= 0]
            checks.require(len(used) == len(set(used.tolist())), "plan is not exclusive")
            for fish in range(3):
                row = plan[fish, view, node]
                origin, direction = np.zeros(3, np.float32), np.zeros(3, np.float32)
                if row >= 0:
                    origin[:] = data[name + "_ray_origins"][row, node]
                    direction[:] = data[name + "_ray_directions"][row, node]
                delta = candidate[fish, node] - origin
                perpendicular = delta - np.sum(delta * direction) * direction
                distance[fish, view, node] = np.linalg.norm(perpendicular)
                if row >= 0 and distance[fish, view, node] <= 2:
                    error = np.linalg.norm(
                        projected[fish, node] - data[name + "_points"][row, node]
                    )
                    pixels[fish, view, node] = error
                    if np.isfinite(error) and error <= 12:
                        sources[fish, view, node] = row
    return sources, (sources >= 0).sum(axis=1) >= 2, distance, pixels


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = prior.generated.comparison.source_registry.local_code()
    hashes = dict(code)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="audit") as record:
        evaluation = sealed(
            ROOT,
            "evaluation.provenance.json",
            "bounded_mouth_seed_fixed202_evaluation",
            "e2e3e4b3212c360bcc2fccc3f35ba67067e3f625dde4327a897e5b6f281c04ce",
            hashes,
        )
        intersections_root = REPO / "work/three_body/directional_mouth_202_intersections_v1"
        intersections = sealed(
            intersections_root,
            "intersections.provenance.json",
            prior.STAGE,
            "3976a27574edfc6e8ac5d15c078a35856caa7cee4009beac6c7625c9a5416c98",
            hashes,
        )
        generation = sealed(
            prior.ROOT,
            "generation.provenance.json",
            prior.generated.STAGE,
            prior.SHA,
            hashes,
        )
        bindings = []
        for label, historical in (("generation", generation), ("intersections", intersections)):
            for key, item in historical["inputs"].items():
                path = Path(key).resolve()
                if (
                    path.suffix != ".py"
                    or not path.is_relative_to(REPO)
                    or "code_snapshot" in path.parts
                    or str(path) not in evaluation["inputs"]
                ):
                    continue
                current = evaluation["inputs"][str(path)]
                checks.require(
                    Path(item["path"]).resolve() == path == Path(current["path"]).resolve()
                    and item["sha256"] == current["sha256"] == checks.sha(path),
                    "historical/evaluation/live source mismatch: " + str(path),
                )
                snapshot = required(ROOT / "code_snapshot" / path.relative_to(REPO), hashes)
                checks.require(
                    checks.sha(snapshot) == item["sha256"], "historical snapshot mismatch"
                )
                hashes[str(path)] = item["sha256"]
                bindings.append(dict(history=label, path=str(path), sha256=item["sha256"]))
        checked = {row["path"] for row in bindings}
        for path in (
            REPO / "labs/20260909_three_body/node_geometry.py",
            REPO / "labs/20260909_three_body/node_recovery.py",
            REPO / "src/medaka3d/calib.py",
            Path(prior.__file__).resolve(),
        ):
            checks.require(str(path) in checked, "critical historical code binding absent")
        for name, digest in (
            ("bounded_ray_seed.py", SOLVER_SHA),
            ("evaluate_bounded_mouth_seeds.py", EVAL_SHA),
        ):
            path = REPO / "work/three_body" / name
            item = evaluation["inputs"][str(path)]
            snapshot = required(ROOT / "code_snapshot" / path.relative_to(REPO), hashes)
            checks.require(
                Path(item["path"]).resolve() == path
                and item["sha256"] == digest == checks.sha(path) == checks.sha(snapshot),
                "new source/evaluation/snapshot binding mismatch",
            )
            hashes[str(path)] = digest
        summary = checks.read(required(ROOT / "summary.json", hashes))
        frame_rows = checks.read(required(ROOT / "frames.json", hashes))
        checks.require(
            summary["schema"] == "bounded_mouth_seed_fixed202_evaluation"
            and summary["status"] == "ok"
            and summary["native_interval"] == [67663, 67696]
            and summary["combinations"] == 202
            and summary["frame_count"] == len(frame_rows) == 33,
            "evaluation scope differs",
        )
        rows = [
            json.loads(line)
            for line in required(ROOT / "all202.jsonl", hashes).read_text().splitlines()
        ]
        previous = [
            json.loads(line)
            for line in required(intersections_root / "all202.jsonl", hashes)
            .read_text()
            .splitlines()
        ]
        same(
            [(r["frame_idx"], r["index_in_frame"]) for r in rows],
            [(r["frame_idx"], r["index_in_frame"]) for r in previous],
            "202 row alignment",
        )
        checks.require(len(rows) == 202, "fixed202 count")
        checks.require(
            dict(Counter(row["first_gate"] for row in previous)) == summary["original_gate_counts"],
            "original aggregate gates differ",
        )
        checkpoint = prior.prior.BRANCH / "checkpoints/through_068060"
        for name in ("manifest.json", "metadata.json"):
            prior.prior.audit_input(generation, checkpoint / name, hashes)
        context = checks.read(checkpoint / "metadata.json")["context"]
        cfg = context["recovery_config"]
        checks.require(summary["original_recovery_config"] == cfg, "recovery config differs")
        binding = checks.read(checkpoint / "manifest.json")["bindings"]["original_binding"]
        calibration = None
        for path, digest in binding["inputs"]["seed"]["SL_and_calibration_sha256"].items():
            if Path(path).suffix == ".json":
                prior.prior.bind(path, digest, hashes)
                if Path(path).name == "calibration_manifest.json":
                    calibration = Path(path).parent
        checks.require(calibration is not None, "calibration missing")
        rig = load_rig(calibration)
        normalize = prior.generated.comparison.support_audit.COMMON.normalize
        checks.require(
            normalize([dataclasses.asdict(c) for c in rig.cameras]) == context["cameras"],
            "camera context mismatch",
        )
        gates, audited_frames, max_ray_error = Counter(), [], 0.0
        all_sources, all_observed, all_distances, all_pixels = [], [], [], []
        for frame in range(67663, 67696):
            with np.load(
                required(prior.ROOT / f"f{frame}_generation.npz", hashes), allow_pickle=False
            ) as ar:
                data = {key: ar[key].copy() for key in ar.files}
            with np.load(
                required(ROOT / f"f{frame}_bounded.npz", hashes), allow_pickle=False
            ) as ar:
                saved = {key: ar[key].copy() for key in ar.files}
            scene, plan = data["predicted_pose"], data["predicted_plan"]
            for key in ("predicted_pose", "predicted_plan", "initial_lengths"):
                checks.require(checks.equal_bytes(data[key], saved[key]), "unchanged context bytes")
            admitted = []
            for row, old in zip(rows, previous, strict=True):
                if row["frame_idx"] != frame:
                    continue
                same(row["source_rows"], old["source_rows"], "original source row differs")
                checks.require(row["original_first_gate"] == old["first_gate"], "old gate differs")
                source = np.array(row["source_rows"], np.int32)
                point = np.asarray(row["xyz"], float)
                same(point, row["initializer"]["point_xyz"], "saved solver point differs")
                views = np.flatnonzero(source >= 0)
                origin = np.array(
                    [data[prior.prior.VIEWS[v] + "_ray_origins"][source[v], 0] for v in views]
                )
                direction = np.array(
                    [data[prior.prior.VIEWS[v] + "_ray_directions"][source[v], 0] for v in views]
                )
                delta = point - origin
                depth = np.sum(delta * direction, axis=-1)
                perpendicular = delta - depth[:, None] * direction
                rms = float(np.sqrt(np.sum(perpendicular**2) / len(views)))
                length = float(np.linalg.norm(point - scene[0, 1]))
                ratio = length / data["initial_lengths"][0, 0]
                near(rms, row["ray_rms_mm"], "ray RMS differs")
                near(depth, row["forward_distance_along_each_ray_mm"], "forward depth differs")
                near(length, row["mouth_chest_length_mm"], "length differs")
                near(ratio, row["length_ratio"], "ratio differs")
                lower, upper = np.asarray(cfg["lower_mm"]), np.asarray(cfg["upper_mm"])
                near(np.maximum(lower - point, 0), row["box_lower_excess_xyz_mm"], "lower excess")
                near(np.maximum(point - upper, 0), row["box_upper_excess_xyz_mm"], "upper excess")
                candidate, proposed = scene.copy(), plan.copy()
                candidate[0, 0], proposed[0, :, 0] = point, source
                strict, observed, distance, pixels = support(candidate, proposed, data, rig)
                same(strict, row["strict_sources"], "strict source mismatch")
                same(observed, row["observed"], "observed mismatch")
                near(distance, row["strict_float32_ray_distance_mm"], "ray cache distance mismatch")
                max_ray_error = max(
                    max_ray_error,
                    float(
                        np.max(
                            np.abs(
                                distance - np.asarray(row["strict_float32_ray_distance_mm"], float)
                            )
                        )
                    ),
                )
                direct = bool(observed[0, 0] and observed[0, 1])
                mouth_views = int((strict[0, :, 0] >= 0).sum())
                checks.require(
                    direct == row["strict_mouth_and_chest"]
                    and mouth_views == row["strict_mouth_views"],
                    "direct flags mismatch",
                )
                gate = "accepted"
                if not np.isfinite(point).all() or not np.isfinite(rms) or np.any(depth < 0):
                    gate = "nonfinite_or_behind"
                elif rms > cfg["ray_rms_mm"]:
                    gate = "ray_rms"
                elif np.any(point < lower) or np.any(point > upper):
                    gate = "outside_box"
                elif not cfg["minimum_length_ratio"] <= ratio <= cfg["maximum_length_ratio"]:
                    gate = "anatomy"
                elif not direct:
                    gate = "strict_support"
                checks.require(
                    gate == row["first_gate"] and (gate == "accepted") == row["admitted"],
                    "independent gate differs",
                )
                gates[gate] += 1
                all_sources.append(strict)
                all_observed.append(observed)
                all_distances.append(distance)
                all_pixels.append(pixels)
                if gate == "accepted":
                    rank = (
                        -mouth_views,
                        rms,
                        float(np.linalg.norm(point - scene[0, 0])),
                        tuple(source),
                    )
                    admitted.append((rank, row["index_in_frame"], candidate, proposed))
            admitted.sort(key=lambda item: item[0])
            selected = admitted[: cfg["proposals_per_fish"]]
            ids = [item[1] for item in admitted]
            chosen = [item[1] for item in selected]
            expected = dict(
                frame_idx=frame,
                combinations=sum(r["frame_idx"] == frame for r in rows),
                admitted=len(admitted),
                returned=len(selected),
                selected_indices=chosen,
            )
            checks.require(
                frame_rows[frame - 67663] == expected, "per-frame ranked summary differs"
            )
            for row in rows:
                if row["frame_idx"] == frame:
                    checks.require(
                        row["selected_for_fitting"] == (row["index_in_frame"] in chosen),
                        "selected row flag differs",
                    )
            same(saved["admitted_combination_indices"], ids, "NPZ admitted ordering")
            same(saved["admitted_pose"], np.stack([a[2] for a in admitted]), "NPZ admitted pose")
            same(saved["admitted_plan"], np.stack([a[3] for a in admitted]), "NPZ admitted plan")
            same(
                saved["selected_for_fitting"],
                np.arange(len(admitted)) < len(selected),
                "NPZ budget",
            )
            audited_frames.append(expected)
        checks.require(dict(gates) == summary["bounded_gate_counts"], "global gates differ")
        for name, value in (
            ("admitted", sum(f["admitted"] for f in audited_frames)),
            ("returned", sum(f["returned"] for f in audited_frames)),
            ("frames_with_candidates", sum(f["returned"] > 0 for f in audited_frames)),
        ):
            checks.require(summary[name] == value, "global total differs: " + name)
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[67663, 67696],
            evaluation_seal_sha256=hashes[str(ROOT / "evaluation.provenance.json")],
            evaluation_outputs_verified=len(evaluation["outputs"]),
            historical_source_checks=len(bindings),
            unique_historical_sources=len(checked),
            new_solver_sha256=SOLVER_SHA,
            evaluator_sha256=EVAL_SHA,
            gate_counts=dict(gates),
            rows=202,
            frames=33,
            admitted=summary["admitted"],
            returned=summary["returned"],
            every_rank_and_candidate_array_matches=True,
            all202_actual_calibration_support_reprojected=True,
            max_float32_ray_distance_difference_mm=max_ray_error,
            seed_solver_called=False,
            fit_performed=False,
            GPU_used=False,
            biological_identity_or_held_out_performance_claim=False,
        )
        for name, value in (
            ("summary.json", result),
            ("historical_code_bindings.json", bindings),
            ("frames.json", audited_frames),
        ):
            (output / name).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
        np.savez_compressed(
            output / "support_recheck.npz",
            frame_idx=np.array([r["frame_idx"] for r in rows], np.int64),
            combination_index=np.array([r["index_in_frame"] for r in rows], np.int64),
            sources=np.stack(all_sources),
            observed=np.stack(all_observed),
            ray_distance_mm=np.stack(all_distances),
            native_pixel_error=np.stack(all_pixels),
        )
        (output / "RESULT.md").write_text(
            "# 制約付き口候補202組の独立監査\n\n"
            f"元評価の全{len(evaluation['outputs'])}出力SHAと、過去診断・今回評価・live・snapshotの"
            f"共有旧コード{len(checked)}ファイル(照合{len(bindings)}件)の一致を確認した。"
            "新solver/evaluatorも登録SHA・評価input・live・snapshotの一致を検証した。\n\n"
            f"保存202点を元のray/box/口胸長条件で再採点し、全3個体の実校正2mm/12px支持を"
            f"別ループで再投影した。棄却/受理は{dict(gates)}、85受理・62返却・33/33フレーム。"
            "各フレームの元順位、2候補上限、33個の候補NPZとのpose/plan/組合せindex対応が一致した。\n\n"
            "solver再実行・候補再生成・3D fit・追跡・GPU・原画読込は行っていない。"
            "これは同じ事後選択33フレームの提案評価の監査であり、fit後の保持、"
            "未評価区間への汎化、3個体のID正解を示さない。\n",
            encoding="utf-8",
        )
        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(target) == digest, "audit snapshot changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
            checks.require(checks.sha(path) == digest, "audit input changed")
        for path in output.rglob("*"):
            if path.is_file() and not path.name.endswith("provenance.json"):
                record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
        for item in record.outputs.values():
            checks.require(checks.sha(item["path"]) == item["sha256"], "audit output changed")
        record.param("all_bound_source_input_snapshot_output_sha_unchanged", True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
