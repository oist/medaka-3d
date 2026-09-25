"""Two saved Depth bank winner changes on native photos and independent masks.

Open M/C/T points show fixed post-fit candidates, not direct support or a new
trajectory. A non-split classification can include missing/ambiguous evidence.
Only saved selection JSON, six native PNGs and six saved mask keys are decoded.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import render_partial_body_bank_comparison as drawing

proof, require, prior = drawing.proof, drawing.require, drawing.prior
WORK, REPO, VIEWS = drawing.WORK, drawing.REPO, drawing.VIEWS
STAGE = "saved_endpoint_mask_priority_changed_visual_v1"
SCORE_ROOT = WORK / "saved_endpoint_mask_priority_v1"
SCORE_SHA = "0178d7d8b141f3d01d02ceb61e0b98c5c39b5fd646de18b8bec6678e934c0cb5"
SCORE_SOURCE_SHA = "0f1175ca074661fc495312455c4f2b9f56c2ff99374143b0497dc9784a899015"
PHOTO_ROOT = drawing.PRIMARY_ROOT
PHOTO_SHA, PHOTO_EVIDENCE_SHA = drawing.PRIMARY_SEAL_SHA, drawing.PRIMARY_EVIDENCE_SHA
FRAMES, MODES = (82727, 82728), ("original_path", "nonsplit_priority")
ARM = "expanded_depth_instance_rescore"
EXPECTED_WINNERS = {82727: (7, 14), 82728: (6, 0)}
PINS = {
    **drawing.PINS,
    "render_partial_body_bank_comparison.py": (
        "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
    ),
    "score_saved_endpoint_mask_priority.py": SCORE_SOURCE_SHA,
}
CLASS_LABELS = {
    "same_instance": "same",
    "split_instances": "SPLIT",
    "uncovered_endpoint": "uncovered",
    "ambiguous_endpoint": "ambiguous",
    "unavailable": "unavailable",
    "invalid_geometry": "invalid XYZ",
    "invalid_projection": "invalid UV",
}


def source_code(evidence):
    for name, digest in PINS.items():
        evidence.add(WORK / name, digest)
    sources = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (
                path.suffix == ".py"
                and path.is_relative_to(REPO)
                and ".venv" not in path.parts
                and path.is_file()
            ):
                sources[str(path)] = evidence.add(path)
    return sources


def selected_case(case, summary):
    frame = case["frame_idx"]
    require(frame in FRAMES and case["arm"] == summary["arm"] == ARM, "wrong selected case")
    require(summary["frame_idx"] == frame and summary["best_changed"] is True, "no saved change")
    require(set(case["selections"]) == set(MODES), "two saved winner modes required")
    expected = EXPECTED_WINNERS[frame]
    require((summary["old_best"], summary["priority_best"]) == expected, "fixed winners differ")
    projections = {}
    for mode, index in zip(MODES, expected, strict=True):
        row = case["selections"][mode]
        require(row["candidate_index"] == index and index < summary["candidates"], "winner differs")
        pose, plan = np.asarray(row["pose_xyz"], np.float64), np.asarray(row["candidate_plan"])
        require(pose.shape == (3, 3, 3) and np.isfinite(pose).all(), "finite fitted pose required")
        require(
            plan.shape == (3, 3, 3) and plan.dtype.kind in "iu" and (plan >= -1).all(), "plan shape"
        )
        costs = np.array([row[k] for k in ("path_cost", "parent_cost", "local_cost")])
        require(np.isfinite(costs).all() and costs[0] == costs[1] + costs[2], "cost ledger differs")
        require(set(row["projections"]) == set(VIEWS), "all camera projections required")
        projections[mode] = {}
        split = np.zeros(3, bool)
        for view in VIEWS:
            labels = row["classifications"][view]
            require(
                len(labels) == 3 and all(v in CLASS_LABELS for v in labels), "body classifications"
            )
            split |= np.asarray(labels) == "split_instances"
            uv = np.asarray(row["projections"][view], np.float64)
            require(
                uv.shape == (3, 3, 2) and np.isfinite(uv).all(), "saved finite projections required"
            )
            uv.flags.writeable = False
            projections[mode][view] = uv
        require(np.array_equal(split, row["split_by_body"]), "split aggregation differs")
        require(
            split.any() if mode == MODES[0] else not split.any(), "expected split removal differs"
        )
    rows = case["selections"]
    require(
        rows[MODES[1]]["path_cost"] - rows[MODES[0]]["path_cost"]
        == summary["selected_priority_path_cost_difference"],
        "cost difference differs",
    )
    return dict(frame_idx=frame, arm=ARM, selections=rows, projections=projections)


def load_context(evidence):
    seal = drawing.read(SCORE_ROOT / "scoring.provenance.json", SCORE_SHA, evidence)
    require(
        seal["stage"] == SCORE_ROOT.name and seal["status"] == "ok", "successful scoring required"
    )
    drawing.edge(seal, WORK / "score_saved_endpoint_mask_priority.py", SCORE_SOURCE_SHA)
    plan, summary, payload = [
        drawing.document(SCORE_ROOT, seal, k + ".json", evidence)
        for k in ("plan", "summary", "render_payload")
    ]
    require(
        plan["schema"] == summary["schema"] == SCORE_ROOT.name and summary["status"] == "ok",
        "saved schema differs",
    )
    require(
        payload["schema"] == SCORE_ROOT.name + "_render_payload"
        and payload["physical_ID_truth"] is False
        and payload["final_trajectory_selection"] is False
        and payload["strict_support_recomputed"] is False,
        "saved selection scope differs",
    )
    require(
        plan["endpoint_tolerance_px"] == 6.0
        and summary["numeric_costs_unchanged"] is True
        and summary["priority_only_not_beam_execution"] is True,
        "saved scientific conditions differ",
    )
    cases = []
    for frame in FRAMES:
        selected = [r for r in payload["cases"] if r["arm"] == ARM and r["frame_idx"] == frame]
        metrics = [r for r in summary["cases"] if r["arm"] == ARM and r["frame_idx"] == frame]
        require(len(selected) == len(metrics) == 1, "one saved case per fixed frame required")
        cases.append(selected_case(selected[0], metrics[0]))
    photo_seal = drawing.read(PHOTO_ROOT / "render.provenance.json", PHOTO_SHA, evidence)
    require(
        photo_seal["status"] == "ok" and photo_seal["stage"] == PHOTO_ROOT.name,
        "native photo preparation failed",
    )
    require(
        photo_seal["outputs"]["source_and_frame_evidence.json"]["sha256"] == PHOTO_EVIDENCE_SHA,
        "photo evidence pin differs",
    )
    doc = drawing.document(PHOTO_ROOT, photo_seal, "source_and_frame_evidence.json", evidence)
    records = {}
    for frame in FRAMES:
        rows = [r for r in doc["records"] if r["frame"] == frame]
        require(len(rows) == 1, "one registered native image record required")
        records[frame] = rows[0]
        for view in VIEWS:
            obs = rows[0]["observations"][view]
            ref = plan["mask_signatures"][f"f{frame}_{view}"]
            require(
                ref["frame_idx"] == frame
                and ref["view"] == view
                and ref["available"] is True
                and obs["mask_available"] is True
                and ref["signature"]["sha256"] == ref["original_mask_sha256"] == obs["mask_sha256"],
                "photo/mask native evidence differs",
            )
    require(len(plan["rig_refs"]) == 4, "registered rig reference count differs")
    for path, digest in plan["rig_refs"].items():
        drawing.edge(seal, path, digest)
        drawing.edge(photo_seal, path, digest)
    mask_ref = plan["mask_archive"]
    drawing.edge(seal, mask_ref["path"], mask_ref["sha256"])
    return dict(
        cases=cases,
        photo_seal=photo_seal,
        records=records,
        mask_archive=mask_ref,
        mask_signatures=plan["mask_signatures"],
        payload_reference=seal["outputs"]["render_payload.json"],
        registered_rig_references_inherited=plan["rig_refs"],
    )


def load_saved_masks(context, evidence):
    ref = context["mask_archive"]
    path = Path(ref["path"]).resolve()
    require(
        path.is_relative_to(WORK) and path.stat().st_size == ref["size_bytes"],
        "saved mask path/size differs",
    )
    evidence.add(path, ref["sha256"])
    masks = {}
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        require(
            len(archive.files) == 21 and set(archive.files) == set(context["mask_signatures"]),
            "saved mask membership differs",
        )
        for frame in FRAMES:
            masks[frame] = {}
            for view in VIEWS:
                value = archive[f"f{frame}_{view}"]
                ref = context["mask_signatures"][f"f{frame}_{view}"]
                obs = context["records"][frame]["observations"][view]
                require(
                    value.dtype == bool
                    and value.shape == (obs["mask_count"], 1200, 1200)
                    and proof.signature(value) == ref["signature"],
                    "saved frame mask differs",
                )
                value.flags.writeable = False
                masks[frame][view] = value
    return masks


def photograph(context, frame, view, evidence):
    obs = context["records"][frame]["observations"][view]
    name = f"native/frame_{frame}_{view}.png"
    require(
        obs["native_png"] == name and obs["canonical_video_frame"] == frame, "photo native differs"
    )
    path = drawing.previous.member(PHOTO_ROOT, context["photo_seal"], name, evidence)
    image = prior.cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), prior.cv2.IMREAD_COLOR)
    require(
        image is not None
        and image.shape == (1200, 1200, 3)
        and image.dtype == np.uint8
        and proof.signature(image)["sha256"] == obs["bgr_sha256"],
        "lossless BGR differs",
    )
    image.flags.writeable = False
    return image, context["photo_seal"]["outputs"][name]


def render(case, images, masks):
    empty = np.empty((0, 3, 2))
    fig, axes = prior.plt.subplots(2, 3, figsize=(18, 12), dpi=160)
    crops = {}
    try:
        fig.suptitle(
            f"Native {case['frame_idx']} | Depth -12 mm fitted bank | original / priority\n"
            "S0 cyan / S1 yellow / S2 magenta are raw slots, not aligned physical IDs. "
            "M circle / C square / T triangle: all open, dashed proposal bodies.\n"
            "D2 contours: original mask numbers, no fill. Fixed 6 px endpoint evidence; "
            "non-split can still be uncovered or ambiguous.\n"
            "Same fitted candidates and original parent histories; no new fit, direct support, "
            "continuous trajectory, sex or identity certification.",
            fontsize=10,
        )
        for vi, view in enumerate(VIEWS):
            projections = [case["projections"][mode][view] for mode in MODES]
            lo, hi = drawing.visual.display_crop(images[view], masks[view], empty, projections)
            crops[view] = dict(lower_xy=lo.tolist(), upper_xy=hi.tolist())
            for ri, mode in enumerate(MODES):
                ax, row = axes[ri, vi], case["selections"][mode]
                drawing.draw_panel(ax, images[view], empty, projections[ri])
                for mask_index, mask in enumerate(masks[view]):
                    y, x = np.nonzero(mask)
                    if len(x):
                        ax.contour(
                            mask, levels=[0.5], colors=["#83ff73"], linewidths=0.7, alpha=0.8
                        )
                        ax.text(
                            float(x.mean()),
                            float(y.mean()),
                            f"D2[{mask_index}]",
                            color="#83ff73",
                            fontsize=7,
                        )
                labels = ", ".join(
                    f"S{i}:{CLASS_LABELS[v]}" for i, v in enumerate(row["classifications"][view])
                )
                ax.set_title(
                    f"{view} | {mode} candidate {row['candidate_index']} ({row['kind']})\n"
                    f"parent {row['candidate_parent']}: {row['parent_cost']:.3f}"
                    f" + local {row['local_cost']:.4f} = {row['path_cost']:.4f}\n{labels}",
                    fontsize=8,
                )
                ax.set_xlim(float(lo[0]), float(hi[0]))
                ax.set_ylim(float(hi[1]), float(lo[1]))
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue(), crops
    finally:
        prior.plt.close(fig)


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        code = source_code(evidence)
        context = load_context(evidence)
        masks = load_saved_masks(context, evidence)
        records, photos = [], []
        for case in context["cases"]:
            frame, images = case["frame_idx"], {}
            for view in VIEWS:
                images[view], ref = photograph(context, frame, view, evidence)
                photos.append(dict(frame_idx=frame, view=view, reference=ref))
            data, crops = render(case, images, masks[frame])
            record = out.write(f"frame_{frame}.png", data)
            records.append(
                dict(frame_idx=frame, image=record, crops=crops, selections=case["selections"])
            )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                cases=records,
                source_code=code,
                scoring_sha256=SCORE_SHA,
                render_payload=context["payload_reference"],
                photo_seal_sha256=PHOTO_SHA,
                original_frame_evidence_sha256=PHOTO_EVIDENCE_SHA,
                photographs=photos,
                native_PNG_decodes=6,
                saved_mask_key_decodes=6,
                saved_mask_archive=context["mask_archive"],
                mask_signatures={
                    f"f{f}_{v}": context["mask_signatures"][f"f{f}_{v}"]
                    for f in FRAMES
                    for v in VIEWS
                },
                projection_source="saved scoring render_payload; no new rig projection",
                registered_rig_references_inherited=context["registered_rig_references_inherited"],
                original_video_SL_masks_checkpoint_reads=0,
                candidate_NPZ_reads=0,
                support_recomputed=False,
                physical_ID_truth=False,
                final_trajectory_selection=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
        )
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
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
