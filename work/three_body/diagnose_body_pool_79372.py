"""CPU-only, post-completion attribution of every saved body start at 79372.

No fit or candidate selection is performed. Region-wise minima are descriptive
oracle diagnostics within an already opened development frame, not fish truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

FRAME = 79372
REGIONS = (("top_cam", 0), ("side1_cam", 2), ("top_cam", 2))
PARTS = ("node", "shape", "pectoral_motion", "repulsion", "silhouette")
WEIGHTED = ("node", "shape", "pectoral_motion", "orientation_motion", "repulsion", "silhouette")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def assert_hashes(hashes):
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError("diagnostic evidence/source/output changed: " + str(path))


def read_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def verify_complete(root):
    """Open numerical outputs only after the producer has a success seal."""
    root = Path(root).resolve()
    seal_path = root / "experiment.provenance.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "ok" or seal.get("stage") != "three_body_local_volume_start_probe":
        raise ValueError("a completed, success-sealed A experiment is required")
    hashes = {str(seal_path): sha(seal_path)}
    for item in seal["outputs"].values():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(root) or not item.get("sha256"):
            raise ValueError("producer output is not bound inside the completed run")
        hashes[str(path)] = item["sha256"]
    assert_hashes(hashes)
    required = {
        "summary": root / "summary.json",
        f"candidates/{FRAME}": root / f"frame_{FRAME}/candidates.npz",
        f"observations/{FRAME}": root / f"frame_{FRAME}/observations.npz",
        f"frame_summary/{FRAME}": root / f"frame_{FRAME}/summary.json",
    }
    for role, path in required.items():
        if Path(seal["outputs"][role]["path"]).resolve() != path:
            raise ValueError("unexpected producer role path: " + role)
    summary = json.loads(required["summary"].read_text())
    if (
        summary.get("status") != "completed_registered_local_body_volume_probe"
        or FRAME not in summary["native_frame_probes"]
        or len(summary["frames"]) != 13
    ):
        raise ValueError("the registered thirteen-frame experiment is incomplete")
    calibration = Path(summary["original_geometry_run"]) / "calibration_snapshot"
    bound = {str(Path(v["path"]).resolve()): v["sha256"] for v in seal["inputs"].values()}
    for path in calibration.iterdir():
        if path.is_file():
            if str(path.resolve()) not in bound:
                raise ValueError("unbound calibration file: " + str(path))
            hashes[str(path.resolve())] = bound[str(path.resolve())]
    assert_hashes(hashes)
    return seal, summary, calibration, hashes


def collect_poses(arrays):
    """Keep every nominal start, including rejected starts, and every admitted fit."""
    original = arrays["original_pose_xyz"]
    poses, metadata = (
        [original.copy()],
        [
            dict(
                stage="start",
                arm="shared",
                trial=0,
                fit_index=0,
                admissible=True,
                changed_raw_slot=-1,
            )
        ],
    )
    before_to_after = []
    for arm in ("mask", "random"):
        start = arrays[arm + "_pose_xyz"]
        mapping = arrays[arm + "_fit_index"]
        admission = arrays[arm + "_admissible"]
        np.testing.assert_array_equal(start[0], original)
        if mapping[0] != 0 or not admission[0] or not np.array_equal(mapping >= 0, admission):
            raise ValueError("invalid admission/index mapping")
        for trial in range(1, len(start)):
            fit = int(mapping[trial])
            slot = int(
                arrays["mask_component_slot_sign_roll"][trial, 1]
                if arm == "mask"
                else arrays["random_raw_slot"][trial]
            )
            if fit >= 0:
                np.testing.assert_array_equal(start[trial], arrays["fit_input_pose_xyz"][fit])
            poses.append(start[trial].copy())
            metadata.append(
                dict(
                    stage="start",
                    arm=arm,
                    trial=trial,
                    fit_index=fit,
                    admissible=bool(admission[trial]),
                    changed_raw_slot=slot,
                )
            )
    fit_offset = len(poses)
    reverse = {0: ("shared", 0, -1)}
    for i, row in enumerate(metadata):
        fit = row["fit_index"]
        if fit >= 0:
            if fit in reverse and fit != 0:
                raise ValueError("one fitted candidate belongs to multiple non-base starts")
            reverse[fit] = (row["arm"], row["trial"], row["changed_raw_slot"])
            before_to_after.append((i, fit_offset + fit))
    for fit, pose in enumerate(arrays["fit_pose_xyz"]):
        arm, trial, slot = reverse[fit]
        poses.append(pose.copy())
        metadata.append(
            dict(
                stage="after_14_step_fit",
                arm=arm,
                trial=trial,
                fit_index=fit,
                admissible=True,
                changed_raw_slot=slot,
            )
        )
    return np.stack(poses), metadata, np.asarray(before_to_after, np.int64), fit_offset


def score_components(arrays, poses, metadata, config):
    predicted = arrays["predicted_pose_xyz"]
    relative = poses - poses[:, :, 1:2]
    previous_relative = predicted - predicted[:, 1:2]
    orientation = ((relative - previous_relative) ** 2).sum(-1).mean((1, 2)) / 25
    parts = np.full((len(poses), 5), np.nan)
    total = np.full(len(poses), np.nan)
    observed = np.zeros((len(poses), 3, 3), bool)
    plans = np.full((len(poses), 3, 3, 3), -1, np.int32)
    for i, row in enumerate(metadata):
        fit = row["fit_index"]
        if fit < 0:
            continue
        before = row["stage"] == "start"
        parts[i] = arrays["fit_initial_loss_parts" if before else "fit_loss_parts"][fit]
        total[i] = arrays["fit_initial_scores" if before else "fit_scores"][fit]
        observed[i] = arrays["fit_initial_observed" if before else "fit_observed"][fit]
        plans[i] = arrays["fit_plans"][fit]
    weighted = np.column_stack(
        (
            parts[:, 0],
            config["shape_weight"] * parts[:, 1],
            config["motion_weight"] * parts[:, 2],
            config["motion_weight"] * 0.05 * orientation,
            config["repulsion_weight"] * parts[:, 3],
            config["mask_weight"] * parts[:, 4],
        )
    )
    finite = np.isfinite(total)
    np.testing.assert_allclose(weighted[finite].sum(1), total[finite], rtol=1e-12, atol=1e-12)
    return dict(
        score=total,
        loss_parts=parts,
        weighted_loss_parts=weighted,
        orientation_term=orientation,
        direct_observed=observed,
        fixed_plans=plans,
        score_available=finite,
        prediction_pectoral_displacement_mm=np.linalg.norm(
            poses[:, :, 1] - predicted[:, 1], axis=-1
        ),
        prediction_node_rms_mm=np.sqrt(((poses - predicted) ** 2).sum(-1).mean(-1)),
        original_pectoral_displacement_mm=np.linalg.norm(
            poses[:, :, 1] - arrays["original_pose_xyz"][:, 1], axis=-1
        ),
    )


def region_pixels(observations, views):
    masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for view, row in REGIONS:
        if not bool(observations["available_" + view]):
            raise ValueError("registered diagnostic region has an unavailable view")
        mask = observations["masks_" + view][row]
        if mask.dtype != bool or mask.shape != (1200, 1200) or not mask.any():
            raise ValueError("registered region must be a nonempty native boolean mask")
        masks[(view, row)] = (mask, cv2.erode(mask.astype(np.uint8), kernel).astype(bool))
    pixels = {}
    for view in views:
        selected = [m[0] for key, m in masks.items() if key[0] == view]
        union = np.logical_or.reduce(selected) if selected else np.zeros((1200, 1200), bool)
        y, x = np.nonzero(union)
        pixels[view] = np.column_stack((x, y)).astype(float)
    return pixels, masks


def render_regions(rig, poses, observations, width, batch_size=4):
    pixels, masks = region_pixels(observations, rig.views)
    renderer = RefractiveFishRenderer(
        rig, pixels, RendererConfig(width_ratio=width), device="cpu", dtype=torch.float64
    )
    valid = renderer.ray_valid.numpy()
    members, info = [], []
    for view, row in REGIONS:
        vi = list(rig.views).index(view)
        xy = pixels[view].astype(int)
        for kind, mask in zip(("full", "core"), masks[(view, row)], strict=True):
            member = np.zeros(renderer.n_pixels, bool)
            member[: len(xy)] = mask[xy[:, 1], xy[:, 0]]
            member &= valid[vi]
            members.append((vi, member))
            info.append(
                dict(
                    view=view,
                    instance_row=row,
                    region=kind,
                    mask_pixels=int(mask.sum()),
                    valid_pixels=int(member.sum()),
                    mask_array_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                )
            )
    shape = (len(poses), len(info))
    values = dict(
        unexplained_fraction=np.full(shape, np.nan),
        per_body_alpha_fraction=np.full((*shape, 3), np.nan),
        visible_fraction=np.full((*shape, 3), np.nan),
        unique_union_fraction=np.full((*shape, 3), np.nan),
        shared_pair_fraction=np.full((*shape, 3), np.nan),
    )
    with torch.no_grad():
        for start in range(0, len(poses), batch_size):
            end = min(start + batch_size, len(poses))
            render = renderer.render(torch.as_tensor(poses[start:end], dtype=torch.float64))
            alpha, union, visible = (
                render.per_fish_alpha.numpy(),
                render.union_alpha.numpy(),
                render.visible_alpha.numpy(),
            )
            np.testing.assert_allclose(visible.sum(-1), union, rtol=1e-12, atol=1e-12)
            for region, (vi, member) in enumerate(members):
                if not member.any():
                    continue
                a = alpha[:, vi, member]
                values["unexplained_fraction"][start:end, region] = (1 - union[:, vi, member]).mean(
                    1
                )
                values["per_body_alpha_fraction"][start:end, region] = a.mean(1)
                values["visible_fraction"][start:end, region] = visible[:, vi, member].mean(1)
                values["unique_union_fraction"][start:end, region] = np.stack(
                    [
                        (
                            a[..., fish]
                            * np.prod(1 - a[..., [j for j in range(3) if j != fish]], -1)
                        ).mean(1)
                        for fish in range(3)
                    ],
                    -1,
                )
                values["shared_pair_fraction"][start:end, region] = np.stack(
                    [(a[..., i] * a[..., j]).mean(1) for i, j in ((0, 1), (0, 2), (1, 2))], -1
                )
            print(json.dumps(dict(rendered=end, total=len(poses))), flush=True)
    return values, info, {"pixels_" + view: value for view, value in pixels.items()}


def summarize_regions(values, metadata, scores, offset):
    unexplained = values["unexplained_fraction"]
    selected = offset  # Producer selected common refit base in both arms; asserted by main.
    groups = {}
    for arm in ("mask", "random"):
        for stage in ("start", "after_14_step_fit"):
            for admission in (True, False):
                ids = [
                    i
                    for i, row in enumerate(metadata)
                    if row["stage"] == stage
                    and row["arm"] in (arm, "shared")
                    and row["admissible"] == admission
                ]
                if not ids:
                    continue
                best = [
                    int(ids[np.argmin(unexplained[ids, region])])
                    for region in range(unexplained.shape[1])
                ]
                # Equal importance to these two full-instance fractions is only descriptive.
                joint = int(ids[np.argmin(unexplained[ids][:, [0, 2]].mean(1))])

                def describe(index):
                    return dict(
                        index=index,
                        **metadata[index],
                        unexplained_fraction=unexplained[index],
                        score=scores["score"][index],
                        weighted_loss_delta_vs_selected=(
                            scores["weighted_loss_parts"][index]
                            - scores["weighted_loss_parts"][selected]
                        ),
                        prediction_pectoral_displacement_mm=(
                            scores["prediction_pectoral_displacement_mm"][index]
                        ),
                    )

                groups[f"{arm}/{stage}/admissible={admission}"] = dict(
                    count=len(ids),
                    region_minima=[describe(i) for i in best],
                    mean_two_unexplained_regions_minimum=describe(joint),
                    count_both_full_regions_better_than_original=int(
                        np.sum(np.all(unexplained[ids][:, [0, 2]] < unexplained[0, [0, 2]], axis=1))
                    ),
                    note="region minima are retrospective diagnostics, not replacement selections",
                )
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO):
        raise FileExistsError("new workspace output required")
    output.mkdir(parents=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        with provenance("body_pool_region_diagnostic_79372", output, name="diagnostic") as record:
            sources = {Path(__file__).resolve()}
            for module in tuple(sys.modules.values()):
                path = getattr(module, "__file__", None)
                if path and Path(path).suffix == ".py":
                    path = Path(path).resolve()
                    if path.is_relative_to(LAB) or path.is_relative_to(REPO / "src"):
                        sources.add(path)
            code = {str(path): sha(path) for path in sources}
            seal, summary, calibration, hashes = verify_complete(args.run)
            for path, digest in code.items():
                relative = str(Path(path).relative_to(REPO))
                producer = seal["outputs"].get("code:" + relative)
                if producer is not None and producer["sha256"] != digest:
                    raise ValueError("reused frozen source differs from producer: " + relative)
                target = output / "code_snapshot" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if sha(target) != digest:
                    raise ValueError("source changed during snapshot")
                record.output("code:" + relative, target, sha256=digest)
            hashes.update(code)
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            root = args.run.resolve() / f"frame_{FRAME}"
            arrays = read_npz(root / "candidates.npz")
            observations = read_npz(root / "observations.npz")
            frame_summary = json.loads((root / "summary.json").read_text())
            if frame_summary["selected"]["fit_indices"] != {"mask": 0, "random": 0}:
                raise ValueError("this diagnostic's common-selected assumption differs")
            poses, metadata, connections, offset = collect_poses(arrays)
            scores = score_components(arrays, poses, metadata, summary["tracker_config"])
            values, regions, pixels = render_regions(
                load_rig(calibration), poses, observations, summary["tracker_config"]["width_ratio"]
            )
            # Independent dense rendering must reproduce the producer's four selected scenes.
            for region, info in enumerate(regions):
                if info["region"] != "full":
                    continue
                saved = next(
                    row
                    for row in frame_summary["selected_instance_mass"]
                    if row["view"] == info["view"] and row["instance_row"] == info["instance_row"]
                )
                mapping = np.array([0, offset, offset, offset])
                np.testing.assert_allclose(
                    values["unexplained_fraction"][mapping, region],
                    np.asarray(saved["unexplained_mass"]) / saved["valid_pixels"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    values["shared_pair_fraction"][mapping, region],
                    np.asarray(saved["pair_shared_mass"]) / saved["valid_pixels"],
                    rtol=1e-12,
                    atol=1e-12,
                )
            data = output / "all_candidates.npz"
            np.savez_compressed(
                data,
                pose_xyz=poses,
                start_to_fit_indices=connections,
                original_review=arrays["original_review"],
                original_observed=arrays["original_observed"],
                predicted_pose_xyz=arrays["predicted_pose_xyz"],
                parent_missing=arrays["parent_missing"],
                **scores,
                **values,
                **pixels,
            )
            record.output("all_candidates", data, sha256=sha(data))
            result = dict(
                frame=FRAME,
                candidate_rows=metadata,
                region_definitions=regions,
                original_index=0,
                selected_refit_index=offset,
                fit_part_names=PARTS,
                weighted_part_names=WEIGHTED,
                shared_pair_order=((0, 1), (0, 2), (1, 2)),
                original_unexplained_fraction=values["unexplained_fraction"][0],
                selected_unexplained_fraction=values["unexplained_fraction"][offset],
                groups=summarize_regions(values, metadata, scores, offset),
                runtime_seconds=time.perf_counter() - started,
                all_nominal_starts_including_bounds_rejected_rendered=True,
                producer_selected_dense_mass_replay_passed=True,
                objective_component_reconstruction_passed=True,
                no_fitting_or_reselection_or_gt=True,
                evidence_scope="sealed A outputs and bound calibration only; raw videos not reread",
                caveat="D2 regions can contain several animals or reflections; no ID truth",
            )
            path = output / "summary.json"
            path.write_text(json.dumps(clean(result), indent=2, allow_nan=False), encoding="utf-8")
            record.output("summary", path, sha256=sha(path))
            table = output / "all_candidates.csv"
            with table.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "index",
                        "stage",
                        "arm",
                        "trial",
                        "admissible",
                        "fit_index",
                        "score",
                        *WEIGHTED,
                        *[
                            f"{r['view']}_m{r['instance_row']}_{r['region']}_unexplained"
                            for r in regions
                        ],
                        "top_m2_shared_S0_S1",
                        "direct_nodes",
                        "maximum_pectoral_prediction_mm",
                    ]
                )
                for i, row in enumerate(metadata):
                    writer.writerow(
                        [
                            i,
                            row["stage"],
                            row["arm"],
                            row["trial"],
                            row["admissible"],
                            row["fit_index"],
                            scores["score"][i],
                            *scores["weighted_loss_parts"][i],
                            *values["unexplained_fraction"][i],
                            values["shared_pair_fraction"][i, 4, 0],
                            int(scores["direct_observed"][i].sum()) if row["admissible"] else "",
                            scores["prediction_pectoral_displacement_mm"][i].max(),
                        ]
                    )
            record.output("table", table, sha256=sha(table))
            assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in record.outputs.values()}})
            record.param("all_input_code_snapshot_output_sha_unchanged", True)
            record.param("device", "cpu")
            record.param("torch_threads", 1)
    finally:
        torch.set_num_threads(old_threads)
    return result


if __name__ == "__main__":
    main()
