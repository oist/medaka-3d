"""Read three saved post-fit banks; diagnose candidate availability, never tracking truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import compare_temporal_point64_v2 as proof
import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
from node_geometry import RayObservations, supported_nodes  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402

STAGE = "instance_mouth_saved_candidate_diagnosis_v1"
FRAMES = (77709, 77723, 77880)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
TRACKING = WORK / "instance_rescore_ten_second_continuation_v1"
TRACKING_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
REVIEW = WORK / "instance_rescore_ten_second_raw_review_v1"
REVIEW_SHA = "598a8ee9354e1a4f12633553187df1c343f20f6e75d19eabb727703d80371467"
COMPARISON = WORK / "instance_rescore_ten_second_comparison_v1"
COMPARISON_SHA = "ac09310ab01e39ece4ac8bbd0e172ff4e6af4078524dc2f7362f5d1899df656d"
require = proof.require


def read(path, evidence, digest):
    evidence.add(path, digest)
    return json.loads(Path(path).read_text("utf8"))


def member(root, seal, name, evidence):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe member")
    path = (root / name).resolve(strict=True)
    row = seal["outputs"][name]
    require(path.is_relative_to(root) and path == Path(row["path"]).resolve(), "member path")
    require(path.stat().st_size == row["size_bytes"], "member length")
    evidence.add(path, row["sha256"])
    return path


def restore_companion(document, archive):
    """Decode the existing JSON descriptors, rejecting missing/extra/repeated arrays."""
    used = set()

    def restore(value):
        if isinstance(value, dict) and "__array__" in value:
            key = value["__array__"]
            require(set(value) == {"__array__", "signature"}, "descriptor fields")
            require(key in archive and key not in used, "missing/repeated array")
            require(proof.signature(archive[key]) == value["signature"], "array signature")
            used.add(key)
            return archive[key]
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value

    result = restore(document["tree"])
    require(used == set(archive), "extra array")
    return result


def stable_rank(values):
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def retained_candidates(poses, parents, permutations, path_costs, beam_size=6):
    """Frozen one-frame selection rule, not a counterfactual continuous rollout."""
    selected = []
    for i in np.argsort(path_costs, kind="stable"):
        if any(
            parents[i] == parents[j]
            and np.array_equal(permutations[i], permutations[j])
            and np.max(np.linalg.norm(poses[i] - poses[j], axis=-1)) < 0.5
            for j in selected
        ):
            continue
        selected.append(int(i))
        if len(selected) == beam_size:
            break
    result = np.zeros(len(poses), dtype=bool)
    result[selected] = True
    return result


def identify_selected(projected, observed, sources, reference):
    expected = np.stack(
        [
            np.asarray([r["projected_uv"] for r in reference["views"][v]]).reshape(3, 3, 2)
            for v in VIEWS
        ]
    )
    matches = np.all(projected == expected[None], axis=(1, 2, 3, 4))
    matches &= np.all(observed == np.asarray(reference["observed"])[None], axis=(1, 2))
    matches &= np.all(sources == np.asarray(reference["sources"])[None], axis=(1, 2, 3))
    indices = np.flatnonzero(matches)
    require(
        len(indices) == 1, "selected candidate is missing or non-unique; no arbitrary tie choice"
    )
    return int(indices[0])


def analyze_bank(bank, projected, sources, observed, selected, points):
    """Compare against the selected raw slot's frame-local mask/SL group, not another fish."""
    poses, plans = bank["candidate_pose_xyz"], bank["candidate_plan"]
    count = len(poses)
    require(projected.shape == (count, 3, 3, 3, 2), "projection axes")
    error = np.full((count, 3, 3, 3), np.nan)
    mask_cost = np.full((count, 3, 3), np.nan)
    reference_masks = np.full((3, 3), -1, dtype=np.int64)
    reference_rows = np.full((3, 3), -1, dtype=np.int64)
    reference_available = np.zeros(3, bool)
    view_rows = []
    for vi, view in enumerate(VIEWS):
        saved = bank["views"][view]
        reason = saved["reason"]
        if saved["eligible"]:
            unique = int(np.sum(saved["optimal_assignment_mask"][selected])) == 1
            if unique:
                assignment = saved["best_assignment"][selected]
                require(sorted(assignment.tolist()) == [0, 1, 2], "reference mask bijection")
                groups = saved["observation_evidence"]["qualifying_SL_rows"]
                require(len(groups) == 3 and all(len(g) == 1 for g in groups), "unique SL groups")
                row = np.asarray([groups[k][0] for k in assignment], dtype=np.int64)
                require(np.all((row >= 0) & (row < len(points[view]))), "SL group row bounds")
                target = points[view][row]
                require(np.isfinite(target).all(), "complete reference SL group")
                reference_available[vi] = True
                reference_masks[vi], reference_rows[vi] = assignment, row
                distances = np.linalg.norm(projected[:, vi] - target[None], axis=-1)
                error[:, vi] = np.where(np.isfinite(distances), distances, np.inf)
                mask_cost[:, vi] = saved["body_instance_costs"][:, np.arange(3), assignment]
            else:
                reason = "selected_mask_assignment_tied; reference comparison unavailable"
        view_rows.append(
            dict(
                view=view,
                original_eligible=saved["eligible"],
                reason=reason,
                reference_available=bool(reference_available[vi]),
            )
        )
    require(reference_available.any(), "no unambiguous frame-local mask/SL reference")
    mouth_error = np.max(error[:, reference_available][..., 0], axis=1)
    require(mouth_error.shape == (count, 3), "candidate/slot mouth metric axes")
    require(np.isfinite(mouth_error[selected]).all(), "finite selected reference required")
    no_direct_losses = np.all(observed | ~observed[selected], axis=(1, 2))
    no_mask_increase = np.all(
        mask_cost[:, reference_available] <= mask_cost[selected, reference_available][None],
        axis=(1, 2),
    )
    conservative = no_direct_losses & no_mask_increase
    ranks = {
        key: stable_rank(bank[field])
        for key, field in (
            ("base_local_rank", "base_local_cost"),
            ("adjusted_local_rank", "adjusted_local_cost"),
            ("path_rank", "adjusted_path_cost"),
        )
    }
    retained = retained_candidates(
        poses, bank["candidate_parent"], bank["candidate_permutation"], bank["adjusted_path_cost"]
    )
    segments = np.diff(poses, axis=2)
    lengths = np.linalg.norm(segments, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.sum(segments[:, :, 0] * segments[:, :, 1], axis=-1) / np.prod(lengths, axis=-1)
        fold = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    arrays = dict(
        candidate_index=np.arange(count, dtype=np.int64),
        candidate_pose_xyz=poses.copy(),
        candidate_plan=plans.copy(),
        candidate_parent=bank["candidate_parent"].copy(),
        candidate_permutation=bank["candidate_permutation"].copy(),
        candidate_kind=np.asarray(bank["candidate_kind"]),
        base_local_cost=bank["base_local_cost"].copy(),
        adjusted_local_cost=bank["adjusted_local_cost"].copy(),
        adjusted_path_cost=bank["adjusted_path_cost"].copy(),
        projected_nodes=projected.copy(),
        direct_sources=sources.copy(),
        direct_observed=observed.copy(),
        group_node_error_px=error,
        fixed_body_mask_cost=mask_cost,
        reference_mask_assignment=reference_masks,
        reference_SL_rows=reference_rows,
        reference_available=reference_available,
        worst_reference_view_mouth_error_px=mouth_error,
        no_selected_direct_losses=no_direct_losses,
        no_fixed_mask_cost_increase=no_mask_increase,
        conservative_candidate=conservative,
        retained_in_original_beam=retained,
        segment_length_mm=lengths,
        fold_angle_deg=fold,
        **ranks,
    )

    def candidate_row(index, slot):
        loss = observed[selected] & ~observed[index]
        gain = observed[index] & ~observed[selected]
        return dict(
            index=int(index),
            selected=index == selected,
            kind=bank["candidate_kind"][index],
            parent=int(bank["candidate_parent"][index]),
            mouth_error_px=float(mouth_error[index, slot]),
            mouth_error_finite=bool(np.isfinite(mouth_error[index, slot])),
            mouth_error_by_reference_view={
                v: float(error[index, vi, slot, 0])
                for vi, v in enumerate(VIEWS)
                if reference_available[vi]
            },
            mouth_plan_by_view=plans[index, slot, :, 0].tolist(),
            mouth_plan_present_views=int(np.sum(plans[index, slot, :, 0] >= 0)),
            mouth_plan_matches_reference_group={
                v: bool(plans[index, slot, vi, 0] == reference_rows[vi, slot])
                for vi, v in enumerate(VIEWS)
                if reference_available[vi]
            },
            mouth_direct=bool(observed[index, slot, 0]),
            direct_losses=np.argwhere(loss).tolist(),
            direct_gains=np.argwhere(gain).tolist(),
            no_fixed_mask_cost_increase=bool(no_mask_increase[index]),
            nonmouth_group_max_increase_px=float(
                np.max(
                    error[index, reference_available, :, 1:]
                    - error[selected, reference_available, :, 1:]
                )
            ),
            base_local_cost=float(bank["base_local_cost"][index]),
            adjusted_local_cost=float(bank["adjusted_local_cost"][index]),
            path_cost=float(bank["adjusted_path_cost"][index]),
            retained_in_original_beam=bool(retained[index]),
            **{k: int(v[index]) for k, v in ranks.items()},
        )

    comparisons = []
    for slot in range(3):
        best = int(np.argmin(mouth_error[:, slot]))
        eligible = np.flatnonzero(conservative)
        require(len(eligible) > 0, "selected candidate must pass self-comparison")
        guarded = int(eligible[np.argmin(mouth_error[eligible, slot])])
        near = mouth_error[:, slot] <= 12.0
        comparisons.append(
            dict(
                slot=slot,
                selected=candidate_row(selected, slot),
                best_mouth=candidate_row(best, slot),
                best_no_mask_increase_or_direct_loss=candidate_row(guarded, slot),
                any_candidate_within_existing_12px_reference_gate=bool(near.any()),
                any_conservative_candidate_within_existing_12px_reference_gate=bool(
                    (near & conservative).any()
                ),
            )
        )
    return dict(
        candidate_count=count,
        selected_candidate_index=selected,
        views=view_rows,
        comparisons=comparisons,
    ), arrays


def execute(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output")
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        tracking = read(TRACKING / "tracking.provenance.json", evidence, TRACKING_SHA)
        review = read(REVIEW / "render.provenance.json", evidence, REVIEW_SHA)
        comparison = read(COMPARISON / "comparison.provenance.json", evidence, COMPARISON_SHA)
        require(
            tracking["status"] == review["status"] == comparison["status"] == "ok",
            "successful sources",
        )
        require(
            tracking["stage"] == TRACKING.name
            and review["stage"] == REVIEW.name
            and comparison["stage"] == COMPARISON.name,
            "source stages",
        )
        data = json.loads(
            member(REVIEW, review, "source_and_frame_evidence.json", evidence).read_text("utf8")
        )
        require(
            data["frames"] == list(FRAMES)
            and data["horizon_stop"] == 78020
            and data["comparison_sha256"] == COMPARISON_SHA,
            "same-horizon review",
        )
        require(
            comparison["inputs"][str(TRACKING / "tracking.provenance.json")] == TRACKING_SHA,
            "producer comparison edge",
        )
        paths = [Path(p) for p in review["inputs"] if Path(p).name == "calibration_manifest.json"]
        require(len(paths) == 1, "unique calibration")
        calibration = read(paths[0], evidence, review["inputs"][str(paths[0])])
        for row in calibration["model_files"]:
            p = (paths[0].parent / row["path"]).resolve(strict=True)
            require(p.is_relative_to(REPO), "calibration path")
            evidence.add(p, review["inputs"][str(p)])
        rig = load_rig(paths[0].parent)
        require(tuple(rig.views) == VIEWS and rig.world_units == "mm", "rig order/units")
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            p = Path(filename).resolve()
            if (
                p.suffix == ".py"
                and p.is_relative_to(REPO)
                and ".venv" not in p.parts
                and p.is_file()
            ):
                digest = (
                    proof.sha(p)
                    if p == Path(__file__).resolve()
                    else tracking["inputs"].get(str(p), review["inputs"].get(str(p)))
                )
                require(digest is not None, "unbound dependency: " + str(p))
                evidence.add(p, digest)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                source_tracking_sha256=TRACKING_SHA,
                source_review_sha256=REVIEW_SHA,
                selected_horizon=78020,
                raw_video_reads=0,
                mask_decodes=0,
                checkpoint39_reads=0,
                GPU_used=False,
                mouth_metric=(
                    "maximum error over eligible views to the selected raw slot "
                    "mask-linked SL mouth"
                ),
                conservative_filter=(
                    "no increase in any fixed-mask body cost and no loss of selected direct nodes"
                ),
                existing_pixel_gate_px=12.0,
                scoring_weights_fitted=False,
                counterfactual_continuous_rollout=False,
                physical_ID_truth=False,
            ),
        )
        records, selected_rows = [], []
        for t, frame in enumerate(FRAMES):
            document = json.loads(
                member(TRACKING, tracking, f"diagnostics/f{frame}.json", evidence).read_text("utf8")
            )
            p = member(TRACKING, tracking, f"diagnostics/f{frame}.npz", evidence)
            require(
                document["archive"] == tracking["outputs"][f"diagnostics/f{frame}.npz"],
                "companion archive edge",
            )
            with np.load(p, allow_pickle=False) as z:
                require(len(z.files) == len(set(z.files)), "duplicate archive member")
                bank = restore_companion(document, {k: z[k] for k in z.files})
            require(bank["frame_idx"] == frame, "candidate frame")
            original = data["records"][t]
            require(original["frame"] == frame, "review frame")
            points = {
                v: np.asarray(original["observations"][v]["SLEAP_points"], dtype=np.float32)
                for v in VIEWS
            }
            require(
                all(
                    np.where(np.isfinite(a), a.astype(np.float64), None).tolist()
                    == original["observations"][v]["SLEAP_points"]
                    for v, a in points.items()
                ),
                "SL float32 roundtrip",
            )
            rays = RayObservations.from_points(rig, points)
            poses, plans = bank["candidate_pose_xyz"], bank["candidate_plan"]
            projected = np.stack([rig.camera(v).project(poses) for v in VIEWS], axis=1)
            results = [
                supported_nodes(pose, plan, rays, rig)
                for pose, plan in zip(poses, plans, strict=True)
            ]
            sources = np.stack([r[0] for r in results])
            observed = np.stack([r[1] for r in results])
            selected = identify_selected(
                projected, observed, sources, original["methods"]["instance_rescore"]
            )
            selected_rows.append(
                dict(
                    frame_idx=np.int64(frame),
                    pose_xyz=poses[selected],
                    observed=observed[selected],
                    sources=sources[selected],
                )
            )
            before = {k: proof.signature(a) for k, a in bank.items() if isinstance(a, np.ndarray)}
            record, arrays = analyze_bank(bank, projected, sources, observed, selected, points)
            require(
                before
                == {k: proof.signature(a) for k, a in bank.items() if isinstance(a, np.ndarray)},
                "bank mutation",
            )
            record["frame_idx"] = frame
            record["array_signatures"] = {k: proof.signature(a) for k, a in arrays.items()}
            out.write(f"f{frame}.npz", arrays)
            with np.load(output / f"f{frame}.npz", allow_pickle=False) as z:
                require(
                    {k: proof.signature(z[k]) for k in z.files} == record["array_signatures"],
                    "saved array readback",
                )
            out.write(f"f{frame}.json", record)
            records.append(record)
        for key, expected in data["selected_candidate_arrays"]["instance_rescore"].items():
            require(
                proof.signature(np.asarray([r[key] for r in selected_rows])) == expected,
                "selected three-row array signature: " + key,
            )
        lines = [
            "# Saved candidate mouth diagnosis",
            "",
            "Fixed frames and raw numeric slots; the SL groups are observations, "
            "not physical ID truth.",
            "Mouth error is the maximum over available reference views. All ranks are one-based.",
            "",
            "| Frame | Slot | Selected index / error px | Best mouth index / error px "
            "| Best without mask-cost increase or direct loss | Best local / path rank "
            "| Best retained |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in records:
            for c in r["comparisons"]:
                a, b, g = c["selected"], c["best_mouth"], c["best_no_mask_increase_or_direct_loss"]
                lines.append(
                    f"| {r['frame_idx']} | {c['slot']} "
                    f"| {a['index']} / {a['mouth_error_px']:.3f} "
                    f"| {b['index']} / {b['mouth_error_px']:.3f} "
                    f"| {g['index']} / {g['mouth_error_px']:.3f} "
                    f"| {b['adjusted_local_rank']} / {b['path_rank']} "
                    f"| {b['retained_in_original_beam']} |"
                )
        lines += [
            "",
            "The conservative comparison is a model diagnostic, not a correctness criterion.",
            "No missing candidate is invented. Saved banks omit initial proposals "
            "and intermediate optimizer poses.",
            "Different ordering changes future proposals, so these tables do not simulate "
            "another continuous rollout.",
            "See per-frame JSON for mouth plans, lost/gained direct nodes and other-node "
            "error changes; NPZ retains all candidates.",
        ]
        out.write("report.md", "\n".join(lines).encode("utf8"))
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                all_selected_three_row_signatures_match=True,
                GPU_used=False,
                checkpoint39_reads=0,
                mask_decodes=0,
                raw_video_reads=0,
                scoring_weights_fitted=False,
                physical_ID_truth=False,
                counterfactual_continuous_rollout=False,
                tracking_modified=False,
                automatic_adoption=False,
            ),
        )
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "diagnosis.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "diagnosis.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute, "explicit execution required")
    execute(args.output)
