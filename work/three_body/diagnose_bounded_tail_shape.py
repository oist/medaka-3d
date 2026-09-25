"""Fixed all1660/all3 shape description from two sealed same-68060 backtraces.

Angles are descriptive, not an acceptance gate. No image decoding, fitting,
rendering, future backtrace, source reassignment or biological ID inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diagnose_bounded_tail_first_chunk as previous
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medaka3d.provenance import provenance

checks, comparison, REPO = previous.checks, previous.comparison, previous.REPO
START, STOP = previous.START, previous.STOP
ROOT = REPO / "work/three_body/bounded_mouth_first_chunk_tail_diagnosis_v1"
PARENT_SHA = "6a331bdde8f67fc46fd4e4284ff22196ea2000bd2965db93b2bcdaaadf1c0d32"
PREVIOUS_SOURCE_SHA = "6b9e43056debe0025d7bbf4287342dd5d79cdf4d2631434b93a247eff367b450"
STAGE = "bounded_mouth_tail_shape_diagnosis_v1"
METHODS = ("directional", "bounded")
CUTS = (45.0, 90.0, 135.0)
MARKERS = (
    (66524, "First bounded retained/selected"),
    (66628, "New S0 tail gap starts"),
    (66703, "Old S0 tail reacquired"),
    (66908, "New S0 tail reacquired"),
)
NODE_NAMES = ("mouth", "chest", "tail")


def vector_angle(a, b):
    checks.require(a.shape == b.shape and a.shape[-1] == 3, "matching XYZ vectors required")
    la, lb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & (la > 1e-12) & (lb > 1e-12)
    cosine = np.full(la.shape, np.nan)
    np.divide(np.sum(a * b, axis=-1), la * lb, out=cosine, where=valid)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def body_shape(pose):
    checks.require(pose.ndim >= 3 and pose.shape[-2:] == (3, 3), "mouth/chest/tail XYZ required")
    head, tail = pose[..., 0, :] - pose[..., 1, :], pose[..., 2, :] - pose[..., 1, :]
    segments = np.stack((np.linalg.norm(head, axis=-1), np.linalg.norm(tail, axis=-1)), axis=-1)
    return dict(
        angle_deg=vector_angle(head, tail),
        segment_lengths_mm=segments,
        mouth_tail_distance_mm=np.linalg.norm(pose[..., 0, :] - pose[..., 2, :], axis=-1),
        polyline_length_mm=segments.sum(-1),
        tail_vector=tail,
    )


def angle_bins(angles):
    """0:[0,45), 1:[45,90), 2:[90,135), 3:[135,180], -1:undefined."""
    result = np.full(angles.shape, -1, np.int8)
    valid = np.isfinite(angles)
    checks.require(np.all((angles[valid] >= 0) & (angles[valid] <= 180)), "angle outside [0,180]")
    result[valid] = np.searchsorted(CUTS, angles[valid], side="right")
    return result


def distribution(a):
    a = np.asarray(a)
    valid = a[np.isfinite(a)]
    return dict(
        count=int(a.size),
        finite_count=int(valid.size),
        undefined_count=int(a.size - valid.size),
        median=None if not len(valid) else float(np.median(valid)),
        p10_p50_p90=None if not len(valid) else np.percentile(valid, [10, 50, 90]).tolist(),
    )


def threshold_timeline(angles, direct, frames):
    """Angles T,6,3; selected path is represented using a singleton rank dimension."""
    checks.require(
        angles.shape == direct.shape and angles.shape[0] == len(frames) and angles.shape[-1] == 3,
        "angle/support timeline shape mismatch",
    )
    result = []
    for slot in range(3):
        for cut in CUTS:
            below = np.isfinite(angles[:, :, slot]) & (angles[:, :, slot] < cut)
            result.append(
                dict(
                    slot=slot,
                    descriptive_angle_less_than=cut,
                    first_any_rank=previous.first_native(below.any(1), frames),
                    first_every_rank=previous.first_native(below.all(1), frames),
                    first_any_direct_tail_rank=previous.first_native(
                        (below & direct[:, :, slot]).any(1), frames
                    ),
                    below_rank_frames=int(below.sum()),
                    any_below_native_frames=int(below.any(1).sum()),
                    every_below_native_frames=int(below.all(1).sum()),
                )
            )
    return result


def length_loss(segment, lengths):
    checks.require(
        segment.shape[-2:] == (3, 2) and lengths.shape == (3, 2), "shape-loss dimensions"
    )
    return (((segment - lengths) / np.maximum(0.2 * lengths, 1.0)) ** 2).mean(axis=(-2, -1))


def require_input(seal, path, hashes):
    return previous.interim.input_member(seal, path, hashes)


def load_inputs(hashes):
    previous.interim.bind(ROOT / "diagnosis.provenance.json", PARENT_SHA, hashes)
    parent = checks.verify(ROOT, "diagnosis.provenance.json", previous.STAGE, hashes)
    previous.interim.bind(previous.__file__, PREVIOUS_SOURCE_SHA, hashes)
    summary = checks.read(ROOT / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["horizon_stop"] == STOP
        and summary["complete_minute"] is False,
        "successful fixed interim diagnosis required",
    )
    require_input(parent, previous.AUDIT / "audit.provenance.json", hashes)
    checks.require(
        hashes[str(previous.AUDIT / "audit.provenance.json")] == previous.AUDIT_SHA,
        "interim seal differs",
    )
    audit = checks.verify(previous.AUDIT, "audit.provenance.json", previous.interim.STAGE, hashes)
    for path, digest in list(hashes.items()):
        for source in (parent, audit):
            if path in source["inputs"]:
                checks.require(source["inputs"][path]["sha256"] == digest, "audited source differs")
    with np.load(ROOT / "all6_history_comparison.npz", allow_pickle=False) as z:
        history = {k: z[k].copy() for k in z.files}
    frames = history["frame_idx"]
    checks.require(np.array_equal(frames, np.arange(START, STOP)), "history native range differs")
    traces, contexts = {}, []
    for name, root in (("directional", previous.interim.OLD), ("bounded", previous.BRANCH)):
        path = previous.AUDIT / f"{name}_same68060_candidate.npz"
        require_input(parent, path, hashes)
        with np.load(path, allow_pickle=False) as z:
            traces[name] = {k: z[k].copy() for k in z.files}
        checks.require(
            len(traces[name]) == 21 and np.array_equal(traces[name]["frame_idx"], frames),
            "same horizon full21 trace required",
        )
        meta_path = require_input(parent, root / "checkpoints/through_068060/metadata.json", hashes)
        contexts.append(checks.read(meta_path)["context"])
    interim_summary = checks.read(previous.AUDIT / "summary.json")
    seed = interim_summary["shared_input_binding"]["inputs"]["seed"]
    seed_path = require_input(audit, seed["registration_path"], hashes)
    checks.require(
        hashes[str(seed_path)] == seed["registration_sha256"], "initial shape seed changed"
    )
    seed_pose = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    lengths = np.linalg.norm(np.diff(seed_pose, axis=1), axis=-1)
    for context in contexts:
        checks.require(
            previous.interim.array_signature({"p": seed_pose})["p"] == context["initial_pose"]
            and previous.interim.array_signature({"l": lengths})["l"] == context["initial_lengths"],
            "initial shape signature differs",
        )
    checks.require(
        contexts[0]["tracker_config"] == contexts[1]["tracker_config"], "objective weights differ"
    )
    producer_path = require_input(audit, previous.interim.OLD / "tracking.provenance.json", hashes)
    producer = checks.read(producer_path)
    raw_references = [
        dict(path=item["path"], sha256=item["sha256"], path_exists=Path(item["path"]).is_file())
        for item in producer["inputs"].values()
        if Path(item["path"]).suffix.lower() == ".mp4"
    ]
    return traces, history, lengths, contexts[0]["tracker_config"], raw_references, summary


def make_figures(output, frames, selected, retained, traces, delta):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(directional="#2674ac", bounded="#dc7334")

    def markers(ax):
        for frame, _label in MARKERS:
            ax.axvline(frame, color="0.4", linewidth=0.7, linestyle=":", alpha=0.8)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for slot, ax in enumerate(axes):
        for name in METHODS:
            ax.plot(
                frames, selected[name]["angle_deg"][:, slot], color=colors[name], lw=1, label=name
            )
        for cut in CUTS:
            ax.axhline(cut, color="0.8", lw=0.6, linestyle="--")
        markers(ax)
        ax.set(ylim=(0, 185), ylabel=f"S{slot} angle (deg)")
    axes[0].legend(loc="lower right", ncol=2)
    axes[-1].set_xlabel("Native frame; 180 degrees = straight, 0 = folded toward mouth")
    fig.suptitle("Selected paths at the same 68060 horizon — all 1660 frames, three slots")
    fig.savefig(output / "selected_angles_all_slots.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for name in METHODS:
        axes[0].plot(frames, selected[name]["angle_deg"][:, 0], color=colors[name], label=name)
        axes[1].plot(
            frames, selected[name]["segment_lengths_mm"][:, 0, 1], color=colors[name], label=name
        )
        axes[3].step(
            frames,
            traces[name]["observed"][:, 0, 2].astype(int),
            color=colors[name],
            where="mid",
            label=name,
        )
    for cut in CUTS:
        axes[0].axhline(cut, color="0.75", lw=0.7, linestyle="--")
    for node, color in enumerate(("#32884c", "#9463a4", "#dc7334")):
        axes[2].plot(
            frames, delta["node_displacement_mm"][:, 0, node], color=color, label=NODE_NAMES[node]
        )
    for ax in axes:
        markers(ax)
        ax.set_xlim(66480, 66940)
    axes[0].set(ylim=(0, 185), ylabel="S0 angle (deg)")
    axes[1].set_ylabel("Chest-tail length (mm)")
    axes[2].set_ylabel("New vs old node distance (mm)")
    axes[3].set(ylabel="Tail direct support", ylim=(-0.15, 1.2), yticks=[0, 1])
    axes[3].set_xlabel("Native frame")
    axes[0].legend(loc="lower left", ncol=2)
    axes[2].legend(loc="upper right", ncol=3)
    axes[0].set_title(
        "66524: first bounded | 66628: new gap | "
        "66703: old reacquisition | 66908: new reacquisition"
    )
    fig.savefig(output / "S0_shape_gap_detail.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    for name, ax in zip(METHODS, axes, strict=True):
        im = ax.imshow(
            retained[name]["angle_deg"][:, :, 0].T,
            origin="lower",
            aspect="auto",
            extent=(frames[0] - 0.5, frames[-1] + 0.5, -0.5, 5.5),
            cmap="RdYlBu",
            vmin=0,
            vmax=180,
        )
        markers(ax)
        ax.set(ylabel=f"{name} beam rank", yticks=range(6), xlim=(66480, 66940))
    axes[-1].set_xlabel("Native frame; rank is not hypothesis identity; latent poses included")
    fig.colorbar(im, ax=axes, label="S0 mouth-chest-tail angle (deg)")
    fig.suptitle("All six retained ranks, S0 — saved angles, not a new acceptance threshold")
    fig.savefig(output / "S0_all6_angle_history.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    lost = traces["directional"]["observed"][:, 0, 2] & ~traces["bounded"]["observed"][:, 0, 2]
    for name in METHODS:
        axes[0].scatter(
            selected[name]["segment_lengths_mm"][lost, 0, 1],
            selected[name]["angle_deg"][lost, 0],
            s=12,
            alpha=0.6,
            color=colors[name],
            label=name,
        )
    axes[0].set(
        xlabel="Chest-tail length (mm)",
        ylabel="Mouth-chest-tail angle (deg)",
        title="S0 tail-loss subset (descriptive conditioning)",
    )
    axes[0].legend()
    scatter = axes[1].scatter(
        delta["node_displacement_mm"][lost, 0, 1],
        delta["node_displacement_mm"][lost, 0, 2],
        c=delta["tail_direction_difference_deg"][lost, 0],
        cmap="viridis",
        s=14,
    )
    axes[1].set(
        xlabel="New vs old chest distance (mm)",
        ylabel="New vs old tail distance (mm)",
        title="Same saved slot, no biological ID claim",
    )
    fig.colorbar(scatter, ax=axes[1], label="Tail direction difference (deg)")
    fig.savefig(output / "S0_lost_tail_geometry.png", dpi=160)
    plt.close(fig)


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="diagnosis") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(
                    json.dumps(previous.finite_json(value), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for path, digest in code.items():
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(dest) == digest, "snapshot changed")
            emit(str(dest.relative_to(output)))
        emit(
            "plan.json",
            dict(
                native_interval=[START, STOP],
                slots=[0, 1, 2],
                same_horizon=STOP,
                descriptive_bins_deg=[0, *CUTS, 180],
                angle_definition="angle(M-C,T-C)",
                markers=MARKERS,
                no_new_acceptance_threshold=True,
                all_selected_and_retained=True,
                raw_decode=False,
                fit=False,
                GPU=False,
            ),
        )
        traces, history, lengths, config, raw, parent_summary = load_inputs(hashes)
        frames = history["frame_idx"]
        selected = {name: body_shape(traces[name]["pose_xyz"]) for name in METHODS}
        retained = {name: body_shape(history[name + "_pose_xyz"]) for name in METHODS}
        delta = dict(
            node_displacement_mm=np.linalg.norm(
                traces["bounded"]["pose_xyz"] - traces["directional"]["pose_xyz"], axis=-1
            ),
            tail_direction_difference_deg=vector_angle(
                selected["directional"]["tail_vector"], selected["bounded"]["tail_vector"]
            ),
            angle_difference_deg=selected["bounded"]["angle_deg"]
            - selected["directional"]["angle_deg"],
            segment_length_difference_mm=selected["bounded"]["segment_lengths_mm"]
            - selected["directional"]["segment_lengths_mm"],
        )
        result_arrays = dict(frame_idx=frames, initial_lengths_mm=lengths, **delta)
        timelines, shape_errors, per_slot = {}, {}, {}
        for name in METHODS:
            s, h = selected[name], retained[name]
            s["angle_bin"] = angle_bins(s["angle_deg"])
            h["angle_bin"] = angle_bins(h["angle_deg"])
            result_arrays.update({name + "_selected_" + k: v for k, v in s.items()})
            result_arrays.update({name + "_retained_" + k: v for k, v in h.items()})
            for key in (
                "observed",
                "identity_review",
                "proposal_kind",
                "loss_parts",
                "costs",
                "path_costs",
            ):
                result_arrays[name + "_selected_" + key] = traces[name][key].copy()
                result_arrays[name + "_retained_" + key] = history[name + "_" + key].copy()
            timelines[name] = dict(
                selected=threshold_timeline(
                    s["angle_deg"][:, None], traces[name]["observed"][:, None, :, 2], frames
                ),
                retained=threshold_timeline(
                    h["angle_deg"], history[name + "_observed"][..., 2], frames
                ),
            )
            shape_errors[name] = {}
            for scope, values, actual in (
                ("selected", s, traces[name]["loss_parts"][:, 1]),
                ("retained", h, history[name + "_loss_parts"][..., 1]),
            ):
                replay = length_loss(values["segment_lengths_mm"], lengths)
                checks.require(
                    np.allclose(replay, actual, rtol=0, atol=1e-12),
                    "stored length-only shape loss differs",
                )
                shape_errors[name][scope] = float(np.max(np.abs(replay - actual)))
                result_arrays[name + "_" + scope + "length_shape_loss_recomputed"] = replay
            per_slot[name] = []
            for slot in range(3):
                per_slot[name].append(
                    dict(
                        slot=slot,
                        angles=distribution(s["angle_deg"][:, slot]),
                        mouth_chest_length=distribution(s["segment_lengths_mm"][:, slot, 0]),
                        chest_tail_length=distribution(s["segment_lengths_mm"][:, slot, 1]),
                        direct_node_counts=traces[name]["observed"][:, slot].sum(0).tolist(),
                        angle_bin_counts=np.bincount(
                            s["angle_bin"][:, slot] + 1, minlength=5
                        ).tolist(),
                        angle_bin_order=["undefined", "[0,45)", "[45,90)", "[90,135)", "[135,180]"],
                    )
                )
        np.savez_compressed(output / "shape_arrays.npz", **result_arrays)
        emit("shape_arrays.npz")
        emit("descriptive_threshold_timelines.json", timelines)
        rows = []
        for t, native in enumerate(frames):
            for slot in range(3):
                row = dict(frame_idx=int(native), slot=slot)
                for name in METHODS:
                    s = selected[name]
                    row.update(
                        {
                            name + "_angle_deg": s["angle_deg"][t, slot],
                            name + "_mouth_chest_mm": s["segment_lengths_mm"][t, slot, 0],
                            name + "_chest_tail_mm": s["segment_lengths_mm"][t, slot, 1],
                            name + "_angle_bin": int(s["angle_bin"][t, slot]),
                            name + "_identity_review": bool(
                                traces[name]["identity_review"][t, slot]
                            ),
                        }
                    )
                    for ni, node in enumerate(NODE_NAMES):
                        row[name + "_" + node + "_direct"] = bool(
                            traces[name]["observed"][t, slot, ni]
                        )
                        row[node + "_new_old_distance_mm"] = delta["node_displacement_mm"][
                            t, slot, ni
                        ]
                row["tail_direction_new_old_deg"] = delta["tail_direction_difference_deg"][t, slot]
                rows.append(previous.finite_json(row))
        previous.write_csv(output / "all1660_all3_selected_shape.csv", rows)
        emit("all1660_all3_selected_shape.csv")
        lost = traces["directional"]["observed"][..., 2] & ~traces["bounded"]["observed"][..., 2]
        lost_summary = []
        for slot in range(3):
            mask = lost[:, slot]
            item = dict(
                slot=slot,
                lost_tail_frames=int(mask.sum()),
                node_displacement_mm={
                    node: distribution(delta["node_displacement_mm"][mask, slot, n])
                    for n, node in enumerate(NODE_NAMES)
                },
                tail_direction_difference_deg=distribution(
                    delta["tail_direction_difference_deg"][mask, slot]
                ),
            )
            for name in METHODS:
                item[name] = dict(
                    angle_deg=distribution(selected[name]["angle_deg"][mask, slot]),
                    chest_tail_length_mm=distribution(
                        selected[name]["segment_lengths_mm"][mask, slot, 1]
                    ),
                )
            lost_summary.append(item)
        raw_info = dict(
            images_needed_for_saved_angle_or_length_math=False,
            independent_anatomical_confirmation_requires_raw_review=True,
            raw_MP4_references_from_successful_producer=raw,
            raw_MP4_bytes_read_or_SHA_verified_here=False,
            relevant_registered_chunk="chunks/f066400-068060/attempt_0001",
            targeted_cached_image_search=(
                "rg --files work/three_body; filenames containing "
                "66524/66628/66703/66908 and png/jpg/jpeg"
            ),
            matching_cached_images_found=[],
            image_search_limit=(
                "filename search only, not an inventory of embedded crops/video frames"
            ),
            known_existing_saved_diagnostic=(
                "directional_one_minute_66400-76360_v1/diagnostics/"
                "frame_066703/arrays.npz; no raw pixels claimed"
            ),
        )
        emit("raw_review_availability.json", raw_info)
        source_notes = dict(
            length_term=(
                "graph_fit.py:50-51: mean(((two segment lengths-initial lengths)"
                "/max(.2*initial lengths,1))**2)"
            ),
            length_weight=config["shape_weight"],
            temporal_orientation=(
                "graph_fit.py:57-59: mean(||relative nodes-predicted relative nodes||^2)/25"
            ),
            temporal_orientation_multiplier=config["motion_weight"] * 0.05,
            no_independent_instantaneous_bend_angle_curvature_or_mouth_tail_distance_term=True,
            unchanged_directional_model=(
                "directional_device_fitter.py:117-140 changes only chest-center motion quadratic"
            ),
            orientation_not_separately_saved_in_five_loss_parts=True,
            no_full_objective_replay_claim=True,
            initial_lengths_mm=lengths,
            saved_length_shape_loss_max_error=shape_errors,
            implication=(
                "Same segment lengths can represent both straight and folded poses. "
                "Temporal orientation is still penalized against prediction."
            ),
        )
        emit("objective_source_notes.json", source_notes)
        make_figures(output, frames, selected, retained, traces, delta)
        for name in (
            "selected_angles_all_slots.png",
            "S0_shape_gap_detail.png",
            "S0_all6_angle_history.png",
            "S0_lost_tail_geometry.png",
        ):
            emit(name)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=1660,
            same_horizon=STOP,
            per_slot=per_slot,
            lost_tail_conditioned=lost_summary,
            descriptive_threshold_timelines=timelines,
            objective=source_notes,
            retained_undefined_angle_counts={
                name: int((~np.isfinite(retained[name]["angle_deg"])).sum()) for name in METHODS
            },
            markers=MARKERS,
            parent_chronology=parent_summary["chronology"],
            preserved_sealed_support_not_recomputed_here=True,
            no_acceptance_threshold_change=True,
            independent_ID_truth=False,
            full_minute_complete=False,
            raw_decode=0,
            fit_calls=0,
            GPU=False,
        )
        emit("summary.json", summary)
        s0 = lost_summary[0]
        body = (
            "# 同68060終端・尾形状の記述診断\n\n"
            "全1660時刻・3slotsと全6保持履歴を対象。"
            f"S0尾の旧のみ支持 {s0['lost_tail_frames']}時刻では、"
            f"新旧node距離中央値は口 {s0['node_displacement_mm']['mouth']['median']:.5f} mm、"
            f"胸 {s0['node_displacement_mm']['chest']['median']:.5f} mm、"
            f"尾 {s0['node_displacement_mm']['tail']['median']:.5f} mm。\n\n"
            f"口胸尾角度中央値は旧 {s0['directional']['angle_deg']['median']:.4f}°→"
            f"新 {s0['bounded']['angle_deg']['median']:.4f}°。尾方向差中央値 "
            f"{s0['tail_direction_difference_deg']['median']:.4f}°。胸尾長中央値は旧 "
            f"{s0['directional']['chest_tail_length_mm']['median']:.4f}→"
            f"新 {s0['bounded']['chest_tail_length_mm']['median']:.4f} mm。\n\n"
            "この保存形状は『頭胸部の近い位置と尾長を保ちつつ、尾が口側へ折れる』仮説に整合する。"
            "旧source/S0は独立正解でなく、身体の実際の折れやID対応を画像なしで確定しない。\n\n"
            "![全slot時系列](selected_angles_all_slots.png)\n\n"
            "![S0詳細](S0_shape_gap_detail.png)\n\n"
            "![全6履歴](S0_all6_angle_history.png)\n\n"
            "![長さと角度](S0_lost_tail_geometry.png)\n\n"
            "angle=angle(M-C,T-C)、180°が直線、0°が口と尾の同方向。45/90/135°は記述binsで採用閾値ではない。"
            "全6はbeam順位であり仮説identityを対応させていない。初発時刻はdescriptive_threshold_timelines.jsonに"
            "各slot・各bin・選択経路/保持全6・direct支持の有無を分けて保存した。\n\n"
            "graph_fit.py:50-51のshape項は2体節長のみ。:57-59には予測からの相対配置差の時間項があり、"
            ":75-77でmotion_weight * 0.05を掛ける。独立した瞬間の曲がり角・曲率・口尾距離項はない。"
            "Directional変更は胸部centerのみ。このため長さ一致自体は折れ込みを除外しないが、"
            "未保存のprefit plan/全候補なしに生成・fit・選択の原因は断定できない。\n\n"
            "数値/図には原画不要。対象4nativeの保存PNG/JPGは限定filename検索で見つからず、"
            "元MP4と66400:68060chunkは成功producerの参照へ束縛した(原動画bytesを再読/検証していない)。"
            "生物学的解釈には別の原画確認が必要。新fit/GPU/live変更はなく、1分後のbacktraceは変わり得る。\n"
        )
        (output / "RESULT.md").write_text(body, encoding="utf-8")
        emit("RESULT.md")
        checks.require(code == comparison.source_registry.local_code(), "loaded code changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
            checks.require(checks.sha(path) == digest, "input/source changed")
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "output/snapshot changed")
        record.param("all_used_input_source_snapshot_output_SHA_unchanged", True)
    print(
        json.dumps(
            previous.finite_json(dict(lost_tail=lost_summary, timelines=timelines)), indent=2
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
