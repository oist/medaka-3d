"""Nine native photographs and two same76360 body paths at native75053..75055.

Stored global direct support is not per-camera support or physical ID truth.
Original MP4 whole-file digests are inherited; nine decoded BGR SHAs are checked
now, with one capture and three absolute seeks per camera. No refit or SL/masks.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import prepare_top_background_5to6_minutes as prep
from threadpoolctl import threadpool_limits

REPO, WORK, require = prep.REPO, prep.WORK, prep.require
sys.path.insert(0, str(REPO / "src"))
from medaka3d import calib  # noqa: E402

STAGE = "nodewise_first_minute_peak_raw_review_v1"
FRAMES, VIEWS, ARMS = (
    (75053, 75054, 75055),
    ("top_cam", "side1_cam", "side2_cam"),
    ("tail", "nodewise"),
)
SHAPE, START, STOP = (1200, 1200, 3), 66400, 76360
COMPARISON = WORK / "nodewise_body_row_first_minute_comparison_v1"
COMPARISON_SHA = "7d3a205eecf1a83b19d39418a2a7e7e286df323eb4b82854655f479251217604"
LIVE_PLAN = WORK / "nodewise_body_row_five_minute_continuation_v1/plan.json"
LIVE_PLAN_SHA = "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
PINS = {
    "work/three_body/prepare_top_background_5to6_minutes.py": (
        "3c4bd35a83c192cc61082ef783fe5d3859438943d2ea23f62b4b25b6e8b6175c"
    ),
    "work/three_body/compare_nodewise_body_row_first_minute.py": (
        "705ede6ebeee394b7dfa7fc43697c62d0dd9fa23f4208173737ac4a75f170d98"
    ),
    "src/medaka3d/calib.py": ("60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"),
    "src/medaka3d/types.py": ("71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a"),
}


def member(root, seal, name):
    ref = seal["outputs"][name]
    require(Path(ref["path"]).resolve() == (root / name).resolve(), "sealed output path differs")
    return ref


def document(root, seal, name, evidence):
    ref = member(root, seal, name)
    return evidence.read(ref["path"], ref["sha256"])


def load_context(evidence):
    seal = evidence.read(COMPARISON / "comparison.provenance.json", COMPARISON_SHA)
    require(
        seal["stage"] == COMPARISON.name and seal["status"] == "ok",
        "successful same-horizon comparison required",
    )
    plan = document(COMPARISON, seal, "plan.json", evidence)
    signatures = document(COMPARISON, seal, "science_signatures.json", evidence)
    require(
        plan["native_interval"] == [START, STOP]
        and plan["terminal_horizon"] == STOP
        and plan["same_original_metadata_and39_signatures"] is True
        and plan["common_source_prefix_chunks"] == 16
        and plan["science_signatures"] == signatures,
        "comparison scope/signatures differ",
    )
    require(
        seal["inputs"].get(str(prep.INDEX)) == prep.INDEX_SHA
        and seal["inputs"].get(str(LIVE_PLAN)) == LIVE_PLAN_SHA
        and seal["inputs"].get(str(WORK / "compare_nodewise_body_row_first_minute.py"))
        == PINS["work/three_body/compare_nodewise_body_row_first_minute.py"],
        "consumed comparison source/index differs",
    )
    index = evidence.read(prep.INDEX, prep.INDEX_SHA)
    live = evidence.read(LIVE_PLAN, LIVE_PLAN_SHA)
    require(
        index["original_binding"] == live["original_binding"], "registered original input differs"
    )
    chunk = index["prefix_chunks"][15]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (15, 74700, 76360),
        "exact chunk15 required",
    )
    root = (prep.STAGED / chunk["attempt_path"]).resolve()
    require(root.is_relative_to(prep.STAGED), "chunk path escapes source root")
    manifest = evidence.read(root / "manifest.json", chunk["manifest_sha256"])
    staging = evidence.read(root / "staging.provenance.json", chunk["provenance_sha256"])
    binding = index["original_binding"]["inputs"]
    require(
        staging["stage"] == "prepare_independent_mask_chunk"
        and staging["status"] == "ok"
        and staging["params"]["binding"] == binding["detector_binding"],
        "successful common staging required",
    )
    require(
        manifest["frames"] == list(range(74700, 76360))
        and manifest["fps"] == 166
        and manifest["image_shape_hw"] == [1200, 1200]
        and manifest["raw_storage"] == "original_source_video",
        "native source metadata differs",
    )
    mref = staging["outputs"]["manifest"]
    require(
        Path(mref["path"]).resolve() == root / "manifest.json"
        and mref["sha256"] == chunk["manifest_sha256"],
        "manifest output edge differs",
    )
    videos = {}
    for view in VIEWS:
        raw, link = manifest["raw"][view], chunk["raw_hardlinks"][view]
        staged = (root / raw["path"]).resolve()
        canonical = (WORK / "source_cache" / raw["source_cache_relative_path"]).resolve()
        declared = binding["detector_binding"]["input_sha256"]["video:" + view]
        require(
            staged.is_relative_to(root)
            and canonical.is_relative_to(WORK / "source_cache")
            and raw["sha256"] == link["sha256"] == declared
            and raw["bytes"] == link["bytes"]
            and raw["source_offset_frames"] == 74700
            and raw["source_total_frames"] == 348600
            and raw["storage_link"] == "hardlink"
            and len(raw["frame_bgr_sha256"]) == 1660,
            "full native video descriptor differs",
        )
        require(
            Path(link["path"]).resolve() == staged
            and Path(link["source_path"]).resolve() == canonical,
            "canonical hardlink paths differ",
        )
        for ref, path in (
            (staging["inputs"]["video:" + view], canonical),
            (staging["outputs"][view + "_raw"], staged),
        ):
            require(
                Path(ref["path"]).resolve() == path
                and ref["sha256"] == declared
                and ref["size"] == raw["bytes"],
                "staged video lineage differs",
            )
        registered = chunk["output_hashes"][view + "_raw"]
        require(
            Path(registered["path"]).resolve() == staged and registered["sha256"] == declared,
            "consumed raw edge differs",
        )
        videos[view] = dict(
            video_path=str(canonical),
            hardlinks=[str(staged)],
            expected_stat=[link[k] for k in ("device", "inode", "bytes", "mtime_ns")],
            inherited_video_sha256=declared,
            frames=[
                dict(
                    frame_idx=f,
                    manifest_relative_row=f - 74700,
                    expected_bgr_sha256=raw["frame_bgr_sha256"][f - 74700],
                )
                for f in FRAMES
            ],
        )
    shared = binding["seed"]["SL_and_calibration_sha256"]
    rig_refs = {p: h for p, h in shared.items() if Path(p).suffix == ".json"}
    require(len(rig_refs) == 4, "four registered calibration JSONs required")
    manifests = [Path(p) for p in rig_refs if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "unique rig manifest required")
    rig_docs = {p: evidence.read(p, h) for p, h in rig_refs.items()}
    rig_manifest = rig_docs[str(manifests[0])]
    require(
        rig_manifest["view_order"] == list(VIEWS) and rig_manifest["backend"] == "refractive",
        "refractive rig/order differs",
    )
    for row in rig_manifest["model_files"]:
        require(
            rig_refs.get(str(manifests[0].parent / row["path"])) == row["sha256"],
            "rig bundle digest differs",
        )
    return dict(
        videos=videos,
        chunk=chunk,
        rig_directory=str(manifests[0].parent),
        rig_references=rig_refs,
        science={
            arm: dict(
                archive=member(COMPARISON, seal, f"science/{arm}.npz"), signatures=signatures[arm]
            )
            for arm in ARMS
        },
        terminal_horizon=STOP,
        comparison_sha256=COMPARISON_SHA,
    )


def load_science(reference, evidence):
    ref = reference["archive"]
    data = evidence.capture(ref["path"], ref["sha256"])
    require(len(data) == ref["size_bytes"], "science archive size differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(
            len(archive.files) == 3 and set(archive.files) == {"frame_idx", "pose_xyz", "observed"},
            "exact three science fields required",
        )
        arrays = {key: archive[key] for key in archive.files}
    for key, shape, dtype in (
        ("frame_idx", (STOP - START,), np.int64),
        ("pose_xyz", (STOP - START, 3, 3, 3), np.float64),
        ("observed", (STOP - START, 3, 3), np.bool_),
    ):
        value = arrays[key]
        require(
            value.shape == shape
            and value.dtype == dtype
            and prep.signature(value) == reference["signatures"][key],
            "science shape/dtype/signature differs: " + key,
        )
    require(
        np.array_equal(arrays["frame_idx"], np.arange(START, STOP))
        and np.isfinite(arrays["pose_xyz"]).all(),
        "finite native science sequence required",
    )
    selected = {key: value[np.asarray(FRAMES) - START].copy() for key, value in arrays.items()}
    require(np.array_equal(selected["frame_idx"], FRAMES), "native row selection differs")
    for value in selected.values():
        value.flags.writeable = False
    return selected


def extract_native(videos, *, capture_factory=None):
    require(
        tuple(videos) == VIEWS
        and all(tuple(r["frame_idx"] for r in videos[v]["frames"]) == FRAMES for v in VIEWS),
        "three cameras and three exact absolute frames required",
    )
    factory = cv2.VideoCapture if capture_factory is None else capture_factory
    images = {f: {} for f in FRAMES}
    for view in VIEWS:
        ctx = videos[view]
        prep.video_guard(ctx)
        cap = factory(ctx["video_path"])
        try:
            require(
                cap.isOpened()
                and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 348600
                and np.isclose(cap.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
                and (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                == SHAPE[:2],
                "full native video header differs",
            )
            for row in ctx["frames"]:
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
                    "decoded native BGR SHA differs",
                )
                images[frame][view] = bgr.copy()
        finally:
            cap.release()
            prep.video_guard(ctx)
    return images


def crop_for(points, shape=SHAPE):
    points = np.asarray(points).reshape(-1, 2)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        return [0, 0, shape[1], shape[0]]
    points = np.clip(points, [0, 0], [shape[1] - 1, shape[0] - 1])
    low, high = points.min(0), points.max(0)
    size = min(max(shape[:2]), max(320, int(np.ceil((high - low).max())) + 160))
    # Native source images are square; same square crop for every frame and arm.
    size = min(size, min(shape[:2]))
    center = (low + high) / 2
    xy = np.clip(
        np.floor(center - size / 2).astype(int), [0, 0], [shape[1] - size, shape[0] - size]
    )
    return [int(xy[0]), int(xy[1]), size, size]


def support_label(slot, node, direct):
    return f"S{slot}{'MCT'[node]}:{'D' if direct else 'N'}"


def draw_panel(image, crop, title, projected=None, observed=None):
    x, y, w, h = crop
    panel = cv2.resize(image[y : y + h, x : x + w], (600, 600), interpolation=cv2.INTER_AREA)
    if projected is not None:
        colors = ((255, 180, 0), (0, 230, 80), (240, 60, 240))
        for slot, color in enumerate(colors):
            coords = (projected[slot] - [x, y]) * [600 / w, 600 / h]
            good = np.isfinite(coords).all(1)
            for node in (0, 1):
                if good[node : node + 2].all():
                    a, b = np.rint(coords[node : node + 2]).astype(int)
                    cv2.line(panel, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
            for node in range(3):
                if not good[node]:
                    continue
                u, v = np.rint(coords[node]).astype(int)
                if not (0 <= u < 600 and 0 <= v < 600):
                    continue
                direct = bool(observed[slot, node])
                thickness = -1 if direct else 2
                if node == 0:
                    cv2.circle(panel, (u, v), 5, color, thickness, cv2.LINE_AA)
                elif node == 1:
                    cv2.rectangle(panel, (u - 5, v - 5), (u + 5, v + 5), color, thickness)
                else:
                    triangle = np.array([[u, v - 6], [u - 6, v + 5], [u + 6, v + 5]], np.int32)
                    if direct:
                        cv2.fillPoly(panel, [triangle], color)
                    else:
                        cv2.polylines(panel, [triangle], True, color, 2, cv2.LINE_AA)
                text = support_label(slot, node, direct)
                cv2.putText(
                    panel,
                    text,
                    (max(0, min(u + 7, 520)), max(13, v - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    text,
                    (max(0, min(u + 7, 520)), max(13, v - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    top = np.zeros((38, 600, 3), np.uint8)
    cv2.putText(
        top, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.concatenate([top, panel], axis=0)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, list):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    evidence, outputs, started = prep.Evidence(), {}, prep.utc()

    def write(name, value):
        data = (
            value
            if isinstance(value, bytes)
            else (json.dumps(json_value(value), indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=prep.sha(data), size_bytes=len(data))
        return outputs[name]

    try:
        for path, digest in PINS.items():
            evidence.capture(REPO / path, digest)
        require(
            Path(calib.__file__).resolve() == REPO / "src/medaka3d/calib.py",
            "registered projector required",
        )
        write("source_snapshot.py", evidence.capture(Path(__file__)))
        context = load_context(evidence)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                views=list(VIEWS),
                arms=list(ARMS),
                source_pins=PINS,
                inputs=context,
                original_video_full_rehashes=0,
                current_video_verification=(
                    "hardlink/stat, absolute native positions and nine BGR SHAs"
                ),
                inherited_whole_video_SHA_not_current_rehash=True,
                original_candidate_archives_read=False,
                same_terminal_horizon=STOP,
                stored_observed_is_global_direct_not_per_view=True,
            ),
        )
        science = {arm: load_science(context["science"][arm], evidence) for arm in ARMS}
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            rig = calib.load_rig(context["rig_directory"])
            projections = {
                arm: {
                    view: np.asarray(rig.camera(view).project(science[arm]["pose_xyz"]), np.float64)
                    for view in VIEWS
                }
                for arm in ARMS
            }
        require(
            all(p.shape == (3, 3, 3, 2) for arm in projections.values() for p in arm.values()),
            "projected axes differ",
        )
        crops = {
            view: crop_for(np.stack([projections[arm][view] for arm in ARMS])) for view in VIEWS
        }
        images = extract_native(context["videos"])
        native = []
        selected = []
        for t, frame in enumerate(FRAMES):
            for view in VIEWS:
                bgr = images[frame][view]
                ref = write(f"native/frame_{frame}_{view}.png", prep.png_bytes(bgr))
                native.append(
                    dict(
                        frame_idx=frame,
                        view=view,
                        image=ref,
                        bgr_signature=prep.signature(bgr),
                        native_absolute_seek=frame,
                        manifest_relative_row=frame - 74700,
                        PNG_roundtrip_verified=True,
                    )
                )
            rows = []
            for arm in (None, *ARMS):
                rows.append(
                    np.concatenate(
                        [
                            draw_panel(
                                images[frame][view],
                                crops[view],
                                f"{arm or 'Raw'} | {view} | f{frame}",
                                None if arm is None else projections[arm][view][t],
                                None if arm is None else science[arm]["observed"][t],
                            )
                            for view in VIEWS
                        ],
                        axis=1,
                    )
                )
            header = np.zeros((72, 1800, 3), np.uint8)
            for line, text in enumerate(
                (
                    f"Native {frame} = {frame / 166:.6f}s | same terminal 76360 | "
                    "raw slots, no physical-ID labels",
                    "M circle / C square / T triangle; filled D=stored global direct, "
                    "open N=not direct; no per-camera support claim",
                    "Same crop across all frames/arms; full native photos saved separately. "
                    "3D difference is not an observed fish jump.",
                )
            ):
                cv2.putText(
                    header,
                    text,
                    (12, 22 + line * 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            figure = np.concatenate([header, *rows], axis=0)
            write(f"frame_{frame}_comparison.png", prep.png_bytes(figure))
            preview = cv2.resize(
                figure, (1500, round(figure.shape[0] * 1500 / 1800)), interpolation=cv2.INTER_AREA
            )
            okay, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            require(okay, "preview JPEG encode failed")
            write(f"frame_{frame}_preview.jpg", jpeg.tobytes())
            selected.append(
                dict(
                    frame_idx=frame,
                    arms={
                        arm: dict(
                            pose_xyz=science[arm]["pose_xyz"][t],
                            observed_global_direct=science[arm]["observed"][t],
                            projected_uv={view: projections[arm][view][t] for view in VIEWS},
                        )
                        for arm in ARMS
                    },
                )
            )
        write(
            "projection_evidence.json",
            dict(
                schema=STAGE + "_projections",
                crops_native_xywh=crops,
                selected=selected,
                native_photographs=native,
                stored_support_recomputed=False,
                projection_nonfinite_saved_as_null=True,
                marker_labels="D/global stored direct, N/not direct",
                raw_slots_are_physical_IDs=False,
            ),
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            native_PNG_count=9,
            comparison_PNG_count=3,
            preview_JPEG_count=3,
            video_capture_count=3,
            absolute_native_seek_count=9,
            original_video_full_rehashes=0,
            science_fields_per_arm=3,
            science_archive_decodes=2,
            original_candidate_SL_mask_checkpoint_reads=0,
            GPU_calls=0,
            same_terminal_horizon=STOP,
            full_native_photographs_saved=True,
            crops_native_xywh=crops,
            S2_stored_global_direct={arm: science[arm]["observed"][:, 2] for arm in ARMS},
            physical_fish_jump_claimed=False,
            physical_ID_truth=False,
        )
        write("summary.json", summary)
        evidence.final_guard()
        for ctx in context["videos"].values():
            prep.video_guard(ctx)
        for ref in outputs.values():
            data = Path(ref["path"]).read_bytes()
            require(
                prep.sha(data) == ref["sha256"] and len(data) == ref["size_bytes"], "output changed"
            )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=prep.utc(),
            inputs=evidence.records,
            outputs=dict(outputs),
        )
    except BaseException as error:
        write(
            "review.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=prep.utc(),
                inputs=evidence.records,
                outputs=dict(outputs),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write("review.provenance.json", seal)
    print(json.dumps(json_value(summary), indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
