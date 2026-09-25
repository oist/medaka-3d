"""Two full-window QA representatives, six native frames, and fixed saved poses.

Original MP4 whole-file SHAs are inherited, not recomputed. This execution checks
canonical hardlinks/stat, six absolute native seeks and six decoded BGR SHAs.
All pose/support numbers come from sealed QA JSON; no candidate NPZ is reopened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import render_observation_reserve_continuation as prior
from threadpoolctl import threadpool_limits

prep, cv2, drawing = prior.prep, prior.cv2, prior.drawing
WORK, REPO, require = prior.WORK, prior.REPO, prior.require
STAGE = "observation_reserve_selected_cases_raw_review_v1"
FRAMES, VIEWS, ARMS, SHAPE = (75016, 75052), prior.VIEWS, prior.ARMS, prior.SHAPE
QA = WORK / "observation_reserve_ten_second_output_qa_v1"
QA_SHA = "4d624251b02476412b8a98608a60c268b5ac0d159a4da7d869ea52756f459367"
DIAGNOSTIC_SHA = "492068b6eb98e92d6121c04e6bec9f26d2f63a986e6d828bec22150b6673b72b"
COMPARISON_SHA = "41edf2644ac37ba1cac0bb2672545e76d4e1f5a42cde5597d5ea3e2f4ce88347"
TRACKING_SHA = "6ad6ef2dd90d7d693c38331b01a471b0dd43637d9b880813e004e34bece7d2fe"
QA_SOURCE_SHA = "f0263fdb62c376a3e805eece7783582f4b147fbbacb94d68576f1907d6d86930"
PINS = {
    **prior.PINS,
    "work/three_body/render_observation_reserve_continuation.py": (
        "d8b0cc781dd7b2d149b49f4c0d587c50c283dd9850c8d972038a7e54f8fbb5f4"
    ),
    "work/three_body/qa_observation_reserve_continuation.py": QA_SOURCE_SHA,
}


def source_code(evidence):
    for path, digest in PINS.items():
        evidence.capture(REPO / path, digest)
    return evidence.capture(Path(__file__))


def select_samples(diagnostic):
    require(
        diagnostic["status"] == "ok"
        and diagnostic["schema"] == "observation_reserve_independent_output_qa_v1"
        and diagnostic["comparison_seal_sha256"] == COMPARISON_SHA
        and diagnostic["tracking_seal_sha256"] == TRACKING_SHA
        and diagnostic["source_sha256"] == QA_SOURCE_SHA
        and diagnostic["process_interval"] == [74700, 76360]
        and diagnostic["frames"] == 1660,
        "fixed successful full-window QA required",
    )
    scenes = diagnostic["proposed_visual_scenes"]
    require(
        [(r["category"], r["representative_frame"]) for r in scenes]
        == [("newly_lost", 75016), ("newly_supported", 75052)],
        "fixed QA representative selection differs",
    )
    rows = diagnostic["selected_samples"]
    by_frame = {row["frame_idx"]: row for row in rows}
    require(len(by_frame) == len(rows), "duplicate saved sample native frame")
    samples = {}
    for arm in ARMS:
        poses, observed = [], []
        for frame in FRAMES:
            value = by_frame[frame]["arms"][arm]
            # Python JSON floats from the QA's ndarray.tolist retain float64 values;
            # do not recast through float32 or silently accept null geometry.
            pose = np.asarray(value["pose_xyz"], np.float64)
            support = np.asarray(value["observed"])
            require(
                pose.shape == (3, 3, 3) and np.isfinite(pose).all(),
                "finite whole-body QA pose required",
            )
            require(
                support.shape == (3, 3) and support.dtype == np.bool_,
                "saved global support must remain bool [body,node]",
            )
            poses.append(pose)
            observed.append(support)
        samples[arm] = dict(
            frame_idx=np.asarray(FRAMES, np.int64),
            pose_xyz=np.stack(poses),
            observed=np.stack(observed),
        )
    return samples, scenes


def load_headers(evidence):
    seal = evidence.read(QA / "qa.provenance.json", QA_SHA)
    require(
        seal["status"] == "ok"
        and seal["source_sha256"] == QA_SOURCE_SHA
        and seal["comparison_seal_sha256"] == COMPARISON_SHA,
        "successful selected-case QA seal required",
    )
    reference = prior.reference(QA, seal, "diagnostics.json")
    require(reference["sha256"] == DIAGNOSTIC_SHA, "fixed QA diagnostic digest differs")
    diagnostic = json.loads(prior.capture(reference, evidence))
    samples, scenes = select_samples(diagnostic)
    photos = evidence.read(prior.PHOTO_ROOT / "review.provenance.json", prior.PHOTO_SHA)
    require(
        photos["status"] == "ok" and photos["stage"] == prior.PHOTO_ROOT.name,
        "registered source photo success required",
    )
    plan = prior.document(prior.PHOTO_ROOT, photos, "plan.json", evidence)
    source = plan["inputs"]
    require(
        plan["same_terminal_horizon"] == 76360
        and source["comparison_sha256"] == prior.QUALIFIED_SHA,
        "source photographs belong to different native registration",
    )
    chunk = source["chunk"]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (15, 74700, 76360),
        "exact shared source chunk15 required",
    )
    root = (prep.STAGED / chunk["attempt_path"]).resolve()
    require(root.is_relative_to(prep.STAGED), "registered chunk escaped source root")
    manifest = evidence.read(root / "manifest.json", chunk["manifest_sha256"])
    staging = evidence.read(root / "staging.provenance.json", chunk["provenance_sha256"])
    require(
        staging["stage"] == "prepare_independent_mask_chunk"
        and staging["status"] == "ok"
        and staging["outputs"]["manifest"]["sha256"] == chunk["manifest_sha256"]
        and Path(staging["outputs"]["manifest"]["path"]).resolve() == root / "manifest.json"
        and manifest["frames"] == list(range(74700, 76360))
        and manifest["fps"] == 166
        and manifest["image_shape_hw"] == [1200, 1200]
        and manifest["raw_storage"] == "original_source_video",
        "registered native chunk schema differs",
    )
    videos = {}
    for view in VIEWS:
        previous, raw = source["videos"][view], manifest["raw"][view]
        link = chunk["raw_hardlinks"][view]
        canonical = Path(previous["video_path"]).resolve()
        staged = (root / raw["path"]).resolve()
        digest = previous["inherited_video_sha256"]
        require(
            staged.is_relative_to(root)
            and canonical.is_relative_to(WORK / "source_cache")
            and canonical == (WORK / "source_cache" / raw["source_cache_relative_path"]).resolve()
            and raw["sha256"] == link["sha256"] == digest
            and raw["source_offset_frames"] == 74700
            and raw["source_total_frames"] == 348600
            and raw["storage_link"] == "hardlink"
            and len(raw["frame_bgr_sha256"]) == 1660
            and Path(link["path"]).resolve() == staged
            and Path(link["source_path"]).resolve() == canonical
            and [link[k] for k in ("device", "inode", "bytes", "mtime_ns")]
            == previous["expected_stat"]
            and raw["bytes"] == link["bytes"],
            "native video/hardlink/source offset differs",
        )
        for ref, path in (
            (staging["inputs"]["video:" + view], canonical),
            (staging["outputs"][view + "_raw"], staged),
        ):
            require(
                Path(ref["path"]).resolve() == path
                and ref["sha256"] == digest
                and ref["size"] == raw["bytes"],
                "staged video lineage differs",
            )
        require(
            chunk["output_hashes"][view + "_raw"]["sha256"] == digest,
            "consumed raw output digest differs",
        )
        context = copy.deepcopy(previous)
        require(context["hardlinks"] == [str(staged)], "source stat guard hardlink differs")
        context["frames"] = [
            dict(
                frame_idx=f,
                manifest_relative_row=f - 74700,
                expected_bgr_sha256=raw["frame_bgr_sha256"][f - 74700],
            )
            for f in FRAMES
        ]
        prep.video_guard(context)
        videos[view] = context
    rig_refs = source["rig_references"]
    require(
        len(rig_refs) == 4 and {Path(p).name: h for p, h in rig_refs.items()} == prior.RIG_PINS,
        "registered four-file rig differs",
    )
    for path, digest in rig_refs.items():
        require(photos["inputs"][path]["sha256"] == digest, "photo rig input edge differs")
    return dict(
        videos=videos,
        samples=samples,
        scenes=scenes,
        rig_references=rig_refs,
        rig_directory=source["rig_directory"],
        chunk_manifest_sha256=chunk["manifest_sha256"],
    )


def extract_native(videos, *, capture_factory=None):
    require(
        tuple(videos) == VIEWS
        and all(tuple(r["frame_idx"] for r in videos[v]["frames"]) == FRAMES for v in VIEWS),
        "three views/two exact absolute native frames required",
    )
    factory = cv2.VideoCapture if capture_factory is None else capture_factory
    images = {frame: {} for frame in FRAMES}
    for view in VIEWS:
        context = videos[view]
        prep.video_guard(context)
        cap = factory(context["video_path"])
        try:
            require(
                cap.isOpened()
                and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 348600
                and np.isclose(cap.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
                and (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                == SHAPE[:2],
                "full native video header differs",
            )
            for row in context["frames"]:
                frame = row["frame_idx"]
                require(
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    and np.isclose(cap.get(cv2.CAP_PROP_POS_FRAMES), frame, rtol=0, atol=1e-6),
                    "absolute native seek failed",
                )
                okay, bgr = cap.read()
                require(
                    okay
                    and bgr is not None
                    and bgr.shape == SHAPE
                    and bgr.dtype == np.uint8
                    and np.isclose(cap.get(cv2.CAP_PROP_POS_FRAMES), frame + 1, rtol=0, atol=1e-6),
                    "native decode/position differs",
                )
                require(
                    prep.signature(bgr)["sha256"] == row["expected_bgr_sha256"],
                    "native BGR SHA differs",
                )
                images[frame][view] = bgr.copy()
        finally:
            cap.release()
            prep.video_guard(context)
    return images


def project_and_crop(samples, rig):
    projected = {
        arm: {
            view: np.asarray(rig.camera(view).project(samples[arm]["pose_xyz"]), np.float64)
            for view in VIEWS
        }
        for arm in ARMS
    }
    require(
        all(p.shape == (2, 3, 3, 2) for views in projected.values() for p in views.values()),
        "two native whole-body projection axes required",
    )
    crops = {
        view: drawing.crop_for(np.stack([projected[arm][view] for arm in ARMS])) for view in VIEWS
    }
    return projected, crops


def draw_frame(frame, index, samples, projected, crops, images, scene):
    panels = []
    for arm in (None, *ARMS):
        name = {None: "Raw", "nodewise": "Nodewise", "observation_reserve": "Reserve"}[arm]
        panels.append(
            np.concatenate(
                [
                    drawing.draw_panel(
                        images[view],
                        crops[view],
                        f"{name} | {view} | f{frame}",
                        None if arm is None else projected[arm][view][index],
                        None if arm is None else samples[arm]["observed"][index],
                    )
                    for view in VIEWS
                ],
                axis=1,
            )
        )
    header = np.zeros((92, 1800, 3), np.uint8)
    lines = (
        f"Native {frame}; final horizon 76360; QA group {scene['category']} (not accuracy truth)",
        "M circle / C square / T triangle; filled D=stored GLOBAL direct; open N=not direct",
        "D/N is not per-camera truth. Same crop across both selected frames and both arms.",
        "Full-window support-disagreement representatives; raw slot colors are not physical IDs.",
    )
    for row, text in enumerate(lines):
        cv2.putText(
            header,
            text,
            (10, 19 + 22 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.concatenate([header, *panels], axis=0)


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh output required",
    )
    output.mkdir(parents=True)
    evidence, outputs, started = prep.Evidence(), {}, prep.utc()

    def write(name, value):
        data = (
            value
            if isinstance(value, bytes)
            else (json.dumps(drawing.json_value(value), indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        row = dict(path=str(path), sha256=prep.sha(data), size_bytes=len(data))
        outputs[name] = row
        return row

    try:
        write("source_snapshot.py", source_code(evidence))
        context = load_headers(evidence)
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            rig = prior.load_rig(context, evidence)
            projected, crops = project_and_crop(context["samples"], rig)
        images = extract_native(context["videos"])
        samples = context["samples"]
        plan = dict(
            schema=STAGE,
            frames=list(FRAMES),
            views=list(VIEWS),
            row_order=["raw", *ARMS],
            QA_seal_sha256=QA_SHA,
            diagnostics_sha256=DIAGNOSTIC_SHA,
            comparison_sha256=COMPARISON_SHA,
            tracking_sha256=TRACKING_SHA,
            photo_source_sha256=prior.PHOTO_SHA,
            source_pins=PINS,
            rig_references=context["rig_references"],
            videos=context["videos"],
            crops_native_xywh=crops,
            selected_scenes=context["scenes"],
            original_video_full_rehashes=0,
            inherited_whole_video_sha_not_current_rehash=True,
            original_candidate_SL_mask_checkpoint_GPU_reads=0,
            NPZ_decodes=0,
            samples_from_sealed_QA_JSON=True,
            global_support_recomputed=False,
            physical_ID_truth=False,
            anatomical_truth=False,
            same_terminal_horizon=76360,
        )
        write("plan.json", plan)
        native = []
        selected = []
        for index, frame in enumerate(FRAMES):
            for view in VIEWS:
                image = images[frame][view]
                ref = write(f"native/frame_{frame}_{view}.png", prep.png_bytes(image))
                native.append(
                    dict(
                        frame_idx=frame,
                        view=view,
                        image=ref,
                        bgr_signature=prep.signature(image),
                        absolute_native_seek=frame,
                        manifest_relative_row=frame - 74700,
                    )
                )
            rendered = draw_frame(
                frame, index, samples, projected, crops, images[frame], context["scenes"][index]
            )
            write(f"frame_{frame}.png", prep.png_bytes(rendered))
            preview = cv2.resize(
                rendered,
                (1440, round(rendered.shape[0] * 1440 / rendered.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
            okay, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            require(okay, "JPEG preview encoding failed")
            write(f"frame_{frame}_preview.jpg", jpeg.tobytes())
            selected.append(
                dict(
                    frame_idx=frame,
                    arms={
                        arm: dict(
                            pose_xyz=samples[arm]["pose_xyz"][index],
                            observed=samples[arm]["observed"][index],
                            projections={v: projected[arm][v][index] for v in VIEWS},
                        )
                        for arm in ARMS
                    },
                )
            )
        write(
            "projection_evidence.json",
            dict(
                schema=STAGE + "_projections",
                selected=selected,
                native_photographs=native,
                crops_native_xywh=crops,
            ),
        )
        write(
            "summary.json",
            dict(
                **plan,
                status="ok",
                native_PNG_count=6,
                comparison_PNG_count=2,
                preview_JPEG_count=2,
                video_captures=3,
                absolute_native_seeks=6,
                actual_decoded_BGR_hashes_verified=6,
                preview_is_lossy=True,
            ),
        )
        evidence.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error=repr(error))
        raise
    finally:
        with (output / "render.provenance.json").open("xb") as stream:
            stream.write(
                (
                    json.dumps(
                        dict(
                            stage=STAGE,
                            started_at=started,
                            finished_at=prep.utc(),
                            inputs=evidence.records,
                            outputs=outputs,
                            **status,
                        ),
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args().output)
