"""Fixed six-frame five-column raw review, reusing sealed native PNG/observations.

No raw MP4, detector, fitter, GPU or future winner is read. Frame selection was
posthoc in the earlier shape diagnosis, and all inherited crops remain fixed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import audit_tail_first_chunk as audit
import cv2
import matplotlib
import numpy as np
from continuous_tracker import _jsonable
from prepare_sequence import array_sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D

checks, REPO = audit.checks, audit.REPO
WORK = REPO / "work/three_body"
STAGE = "tail_shape_raw_review_v1"
FRAMES = (66524, 66600, 66628, 66703, 66800, 66908)
VIEWS = tuple(audit.prior.VIEWS)
METHODS = ("directional", "bounded", "tail")
RAW = WORK / "bounded_shape_raw_review_v1"
RAW_SHA = "d5fa256848a0298fc0d5f5cbca2e7b7a0d55867b5874f9159d5859c56ddcbd70"
NEW = WORK / "tail_first_chunk_interim_audit_v1"
NEW_SHA = "7f7fe6254e3881fa62ace428f39242cddd850386f24165a4de51c21fae7ff547"
COLORS = ("#00e5ff", "#ffdf00", "#ff50ed")
LABELS = ("M", "C", "T")
MARKERS = ("o", "s", "^")


def sealed(root, filename, stage, digest, hashes):
    audit.bind(root / filename, digest, hashes)
    value = audit.verify_seal(root, filename, stage, hashes)
    checks.require(value.get("finished_at"), "completed evidence required")
    return value


def output(root, seal, name, hashes):
    return audit.output_member(seal, root, name, hashes)


def check_native(image, observation, evidence):
    checks.require(
        image is not None
        and image.dtype == np.uint8
        and image.shape == (1200, 1200, 3)
        and array_sha256(image) == evidence["bgr_sha256"],
        "saved PNG native BGR differs",
    )
    masks, points = observation["masks"], observation["detections"]
    available = observation["mask_available"]
    checks.require(available.dtype == np.dtype(bool) and available.shape == (), "mask flag schema")
    checks.require(
        masks.dtype == np.dtype(bool) and masks.ndim == 3 and masks.shape[1:] == (1200, 1200),
        "mask schema",
    )
    checks.require(
        (
            bool(available)
            and len(masks) == evidence["mask_count"]
            and array_sha256(masks) == evidence["mask_sha256"]
        )
        or (
            not bool(available)
            and len(masks) == 0
            and evidence["mask_count"] is None
            and evidence["mask_sha256"] is None
        ),
        "saved native mask evidence differs",
    )
    checks.require(
        points.ndim == 3
        and points.shape[1:] == (3, 2)
        and audit.array_signature({"points": points})["points"] == evidence["detection_signature"],
        "saved SLEAP row bytes differ",
    )


def crop_status(projected, crop):
    crop = np.asarray(crop)
    checks.require(
        crop.shape == (4,)
        and crop.dtype.kind in "iu"
        and np.all(crop[:2] >= 0)
        and np.all(crop[2:] <= 1200)
        and np.all(crop[2:] > crop[:2]),
        "fixed crop schema",
    )
    checks.require(projected.shape == (3, 3, 2), "all three slots/nodes projection required")
    finite = np.isfinite(projected).all(-1)
    inside = finite & (projected >= crop[:2]).all(-1) & (projected <= crop[2:]).all(-1)
    return finite, inside


def load_inputs(hashes):
    audit.bind(
        Path(audit.__file__),
        "864c0151e079f618746bfdc93eee725307317743a35c4ce42a3b9e83319c5a11",
        hashes,
    )
    raw_source = WORK / "render_bounded_shape_raw_review.py"
    audit.bind(
        raw_source, "be7e20b5edd0dddf9e955bd654d3c5a16a282a7000bd41e5010b6b16d06d4edd", hashes
    )
    raw_seal = sealed(RAW, "review.provenance.json", "bounded_shape_raw_review_v1", RAW_SHA, hashes)
    old_seal = sealed(
        audit.OLD_AUDIT, "audit.provenance.json", audit.prior.STAGE, audit.OLD_AUDIT_SHA, hashes
    )
    new_seal = sealed(NEW, "audit.provenance.json", audit.STAGE, NEW_SHA, hashes)
    audit.input_member(raw_seal, audit.OLD_AUDIT / "audit.provenance.json", hashes)
    audit.input_member(new_seal, audit.OLD_AUDIT / "audit.provenance.json", hashes)
    audit.input_member(raw_seal, raw_source, hashes)
    raw_plan = checks.read(output(RAW, raw_seal, "plan.json", hashes))
    raw_summary = checks.read(output(RAW, raw_seal, "summary.json", hashes))
    checks.require(
        tuple(raw_plan["frames"]) == FRAMES
        and tuple(raw_plan["views"]) == VIEWS
        and raw_plan["raw_native_size"] == [1200, 1200]
        and tuple(raw_summary["frames"]) == FRAMES
        and raw_summary["status"] == "ok"
        and raw_summary["native_images"] == 18
        and raw_summary["horizon_stop"] == 68060,
        "fixed raw review scope differs",
    )
    summaries = [
        checks.read(output(root, seal, "summary.json", hashes))
        for root, seal in ((audit.OLD_AUDIT, old_seal), (NEW, new_seal))
    ]
    for s in summaries:
        checks.require(
            s["status"] == "ok"
            and s["native_interval"] == [66400, 68060]
            and s["all21_arrays_backtraced"] is True
            and s["all39_arrays_semantically_decoded"] is True,
            "same68060 audited candidate required",
        )
    checks.require(
        summaries[0]["shared_input_binding"] == summaries[1]["shared_input_binding"],
        "shared original context differs",
    )
    shared = summaries[0]["shared_input_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        audit.bind(path, digest, hashes)
        for seal in (raw_seal, old_seal, new_seal):
            audit.input_member(seal, path, hashes)
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig = audit.prior.load_rig(calibration)
    actual_context = _jsonable(
        dict(
            cameras=[asdict(cam) for cam in rig.cameras],
            views=list(rig.views),
            world_units=rig.world_units,
        )
    )
    metas = {}
    for name in ("bounded", "tail"):
        metas[name] = checks.read(output(NEW, new_seal, name + "_checkpoint_metadata.json", hashes))
    directional_meta = audit.prior.OLD / "checkpoints/through_068060/metadata.json"
    audit.input_member(old_seal, directional_meta, hashes)
    metas["directional"] = checks.read(directional_meta)
    for name, meta in metas.items():
        checks.require(meta["previous_frame"] == 68059, "context endpoint differs")
        checks.require(
            all(meta["context"][k] == v for k, v in actual_context.items()),
            "actual refractive rig differs from saved context: " + name,
        )
    traces = {}
    for name in METHODS:
        root, seal = (NEW, new_seal) if name == "tail" else (audit.OLD_AUDIT, old_seal)
        path = output(root, seal, name + "_same68060_candidate.npz", hashes)
        if name != "tail":
            audit.input_member(raw_seal, path, hashes)
        values = audit.read_npz(path)
        checks.require(
            len(values) == 21 and np.array_equal(values["frame_idx"], np.arange(66400, 68060)),
            "all21 native candidate schema",
        )
        traces[name] = values
    bounded_new = audit.read_npz(output(NEW, new_seal, "bounded_same68060_candidate.npz", hashes))
    checks.require(
        set(bounded_new) == set(traces["bounded"])
        and all(checks.equal_bytes(v, traces["bounded"][k]) for k, v in bounded_new.items()),
        "Bounded all21 differs across audits",
    )
    rows = checks.read(output(RAW, raw_seal, "native_observation_evidence.json", hashes))
    checks.require(
        len(rows) == 18
        and [(r["frame_idx"], r["view"]) for r in rows] == [(n, v) for n in FRAMES for v in VIEWS],
        "native18 evidence order differs",
    )
    evidence = {(r["frame_idx"], r["view"]): r for r in rows}
    return traces, rig, raw_seal, evidence, shared, metas, raw_plan


def draw_pose(ax, projected, observed):
    for slot, nodes in enumerate(projected):
        color = COLORS[slot]
        for left, right in ((0, 1), (1, 2)):
            if np.isfinite(nodes[[left, right]]).all():
                ax.plot(
                    nodes[[left, right], 0],
                    nodes[[left, right], 1],
                    color=color,
                    lw=1.4,
                    ls="-" if observed[slot, [left, right]].all() else "--",
                )
        for node, (xy, label, marker) in enumerate(zip(nodes, LABELS, MARKERS, strict=True)):
            if np.isfinite(xy).all():
                ax.plot(
                    *xy,
                    marker=marker,
                    color=color,
                    markerfacecolor=color if observed[slot, node] else "none",
                    ms=5,
                    mew=1.2,
                )
                ax.annotate(
                    f"S{slot}{label}",
                    xy,
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    clip_on=True,
                    path_effects=[patheffects.withStroke(linewidth=1.2, foreground="#222222")],
                )


def draw_observation(ax, observation):
    for mask in observation["masks"]:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            xy = contour[:, 0]
            ax.plot(xy[:, 0], xy[:, 1], color="#eeeeee", lw=0.8)
    for row, points in enumerate(observation["detections"]):
        color = plt.get_cmap("tab10")(row % 10)
        ax.plot(points[:, 0], points[:, 1], ".:", color=color, lw=0.8, ms=3)
        if np.isfinite(points[1]).all():
            ax.text(*points[1], "R" + str(row), color=color, fontsize=7, clip_on=True)


def scene_metrics(scene):
    a, b = scene[:, 0] - scene[:, 1], scene[:, 2] - scene[:, 1]
    lengths = np.stack((np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)), -1)
    denom = lengths.prod(-1)
    cosine = np.divide((a * b).sum(-1), denom, out=np.full(3, np.nan), where=denom > 0)
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    return dict(
        segment_lengths_mm=lengths.tolist(),
        mouth_chest_tail_angle_degrees=[float(x) if np.isfinite(x) else None for x in angles],
    )


def run(destination):
    destination = Path(destination).resolve()
    checks.require(destination.is_relative_to(WORK) and not destination.exists(), "new work output")
    code = audit.comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    destination.mkdir(parents=True)
    with audit.prior.provenance(STAGE, destination, name="review") as record:

        def emit(name, value=None):
            path = destination / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        emit(
            "plan.json",
            dict(
                frames=FRAMES,
                views=VIEWS,
                columns=["Raw", "D2 + SLEAP", *METHODS],
                horizon_stop=68060,
                prior_raw_seal_sha256=RAW_SHA,
                old_audit_sha256=audit.OLD_AUDIT_SHA,
                new_audit_sha256=NEW_SHA,
                frame_selection=(
                    "original posthoc shape diagnosis; fixed before this new plotting call"
                ),
                crop_rule="exact prior crop_xyxy; never expand for new Tail",
                raw_video_decodes=0,
                marker_scope=(
                    "filled=direct >=2 views; open=unsupported latent; "
                    "dashed segment if either endpoint latent"
                ),
                biological_ID_truth=False,
                all_minute_quality=False,
            ),
        )
        for path, digest in code.items():
            snap = destination / "code_snapshot" / Path(path).relative_to(REPO)
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(snap) == digest, "source snapshot changed")
            emit(str(snap.relative_to(destination)))
        traces, rig, raw_seal, evidence, shared, metas, old_plan = load_inputs(hashes)
        rows, metrics = [], []
        for native in FRAMES:
            index = native - 66400
            scenes = np.stack([traces[name]["pose_xyz"][index] for name in METHODS])
            observed = np.stack([traces[name]["observed"][index] for name in METHODS])
            sources = np.stack([traces[name]["sources"][index] for name in METHODS])
            projected = np.stack(
                [[rig.camera(view).project(scene) for view in VIEWS] for scene in scenes]
            )
            finite, inside, crops = [], [], []
            fig, axes = plt.subplots(3, 5, figsize=(23, 13), constrained_layout=True)
            for vi, view in enumerate(VIEWS):
                png = output(RAW, raw_seal, f"raw_{native}_{view}.png", hashes)
                saved = audit.read_npz(
                    output(RAW, raw_seal, f"observation_{native}_{view}.npz", hashes)
                )
                image = cv2.imread(str(png), cv2.IMREAD_COLOR)
                check_native(image, saved, evidence[native, view])
                for mi, name in enumerate(METHODS[:2]):
                    checks.require(
                        checks.equal_bytes(projected[mi, vi], saved[name + "_projection"]),
                        "old projection bytes differ",
                    )
                crop = saved["crop_xyxy"].copy()
                flags = [crop_status(projected[mi, vi], crop) for mi in range(3)]
                finite.append(np.stack([pair[0] for pair in flags]))
                inside.append(np.stack([pair[1] for pair in flags]))
                crops.append(crop)
                row = dict(
                    frame_idx=native,
                    view=view,
                    crop_xyxy=crop.tolist(),
                    bgr_sha256=array_sha256(image),
                    original_observation_evidence=evidence[native, view],
                    methods={},
                )
                for col, title in enumerate(
                    ("Raw", "D2 + SLEAP rows", "Directional", "Bounded", "Tail")
                ):
                    ax = axes[vi, col]
                    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax.set_xlim(crop[0], crop[2])
                    ax.set_ylim(crop[3], crop[1])
                    ax.set_aspect("equal")
                    ax.set_title(view + " | " + title, fontsize=10)
                    ax.tick_params(labelsize=7)
                    if col == 1:
                        draw_observation(ax, saved)
                    elif col >= 2:
                        mi, name = col - 2, METHODS[col - 2]
                        draw_pose(ax, projected[mi, vi], observed[mi])
                        finite_flags, inside_flags = flags[mi]
                        clipped = [
                            f"S{f}{LABELS[n]}" for f, n in np.argwhere(finite_flags & ~inside_flags)
                        ]
                        missing = [f"S{f}{LABELS[n]}" for f, n in np.argwhere(~finite_flags)]
                        row["methods"][name] = dict(
                            outside_fixed_crop=clipped, nonfinite_projection=missing
                        )
                        text = (
                            "Outside crop: " + ", ".join(clipped)
                            if clipped
                            else "All finite nodes inside crop"
                        )
                        if missing:
                            text += "\nNonfinite: " + ", ".join(missing)
                        ax.text(
                            0.01,
                            0.01,
                            text,
                            transform=ax.transAxes,
                            fontsize=6.5,
                            color="white",
                            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
                            va="bottom",
                        )
                rows.append(row)
            legend = [Line2D([], [], color=c, lw=2, label=f"S{i}") for i, c in enumerate(COLORS)]
            legend += [
                Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    ls="-",
                    label="Filled / solid: directly supported nodes",
                ),
                Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    markerfacecolor="none",
                    ls="--",
                    label="Open / dashed: latent node or endpoint",
                ),
            ]
            fig.legend(handles=legend, loc="outside lower center", ncol=5, fontsize=9)
            fig.suptitle(
                f"Native {native} | Same 68060 horizon | Fixed inherited crops | "
                "M mouth / C chest / T tail\n"
                "Anonymous slots; row/mask correspondence and biological ID are not established",
                fontsize=13,
            )
            name = f"comparison_{native}.png"
            fig.savefig(destination / name, dpi=145)
            plt.close(fig)
            emit(name)
            np.savez_compressed(
                destination / f"scene_{native}.npz",
                frame_idx=np.asarray(native, np.int64),
                methods=np.asarray(METHODS),
                views=np.asarray(VIEWS),
                pose_xyz=scenes,
                projection_native_uv=projected,
                observed=observed,
                sources=sources,
                identity_review=np.stack([traces[n]["identity_review"][index] for n in METHODS]),
                crop_xyxy=np.stack(crops),
                projection_finite=np.stack(finite, axis=1),
                inside_fixed_crop=np.stack(inside, axis=1),
            )
            emit(f"scene_{native}.npz")
            metrics.append(
                dict(
                    frame_idx=native,
                    methods={
                        name: {
                            **scene_metrics(scenes[mi]),
                            "direct_by_slot_node": observed[mi].tolist(),
                        }
                        for mi, name in enumerate(METHODS)
                    },
                )
            )
        emit("native_projection_evidence.json", rows)
        emit("shape_metrics.json", metrics)
        emit(
            "contexts.json",
            dict(
                actual_shared_SL_calibration_sha256=shared,
                contexts={k: v["context"] for k, v in metas.items()},
                original_raw_plan=old_plan,
            ),
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=FRAMES,
            horizon_stop=68060,
            native_images_reused=18,
            observation_npz_reused=18,
            figures=6,
            all18_decoded_BGR_and_mask_SHA_exact=True,
            old36_projections_byte_exact=True,
            old_Bounded21_byte_exact=True,
            actual_refractive_context_exact=True,
            fixed_crops_preserved=True,
            outside_crop_node_projection_counts={
                name: sum(len(r["methods"][name]["outside_fixed_crop"]) for r in rows)
                for name in METHODS
            },
            raw_video_decodes=0,
            detector_calls=0,
            fit_calls=0,
            GPU=False,
            independent_ID_truth=False,
            full_minute_quality=False,
            frame_selection_posthoc=True,
            source_audits=[audit.OLD_AUDIT_SHA, NEW_SHA],
        )
        emit("summary.json", summary)
        (destination / "RESULT.md").write_text(
            "# 固定原画によるTail形状比較\n\n"
            "固定6時刻、同68060終端。左から原画、D2輪郭とSLEAP行、Directional、Bounded、Tail。"
            "3方式とも同じ原画・元のcropを用い、範囲外の投影はpanel内とJSONへ明記した。"
            "塗りつぶしは2視点以上で直接支持された点、白抜きは未支持の推定姿勢。"
            "片端が未支持の線分は破線。S0/S1/S2は匿名slot、Rは2D行で、個体対応を意味しない。\n\n"
            + "\n\n".join(f"![native {n}](comparison_{n}.png)" for n in FRAMES)
            + "\n\n原画PNG18枚の復号BGR、D2マスク、SLEAP行の保存SHAを照合。"
            "旧2方式の36投影を実屈折rigで再計算しbyte一致した。新Tailも同rigで全3slotを投影した。"
            "保存21配列・共有SL3/校正4・camera contextは封印済み証拠へ結合される。\n\n"
            "時刻は以前の不具合診断から事後選択されたもの。図や直接支持だけで解剖学的正解・生物ID・"
            "1分全体の精度を認定しない。新たな動画decode、fit、追跡、検出器実行は行っていない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(code == audit.comparison.source_registry.local_code(), "loaded code changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "source/input/output changed")
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
