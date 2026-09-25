"""Three saved development frames: raw / Nodewise / reserve, same76360 horizon.

Only six selected fields from the successful comparison, nine bound native PNGs,
and four registered calibration JSONs are read. No original video, masks, SL,
checkpoint payload, fitting, tracking, or GPU operation is performed.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import render_nodewise_first_minute_peak_review as drawing
from threadpoolctl import threadpool_limits

prep, cv2, calib = drawing.prep, drawing.cv2, drawing.calib
WORK, REPO, require = drawing.WORK, drawing.REPO, drawing.require
STAGE = "observation_reserve_ten_second_visual_comparison_v1"
COMPARISON_STAGE = "observation_reserve_ten_second_comparison_v1"
COMPARISON_SHA_SOURCE = "761c7ecf33f7134d19ae8d1406ae78e6021342039c11b3b0544688f9ea48b8e8"
PHOTO_ROOT = WORK / "nodewise_first_minute_peak_raw_review_v1"
PHOTO_SHA = "e00b8b99f67245d3fe5867df712978b946c863dcccd53d87b474f95c5f6f4840"
MILESTONE_SHA = "bed9dd1f0d7192261f4e985a3021097f444a941e180bd1982bad4dbcb01525de"
CHECKPOINT_SHA = "7c9672b6fde9f96ee5d23efe7dbbd6e7f88d48299d81e86e00a87fcca5c5a076"
QUALIFIED_SHA = "7d3a205eecf1a83b19d39418a2a7e7e286df323eb4b82854655f479251217604"
FRAMES, PROBES = (75053, 75054, 75055), (75052, 75053, 75054, 75055, 75056)
ARMS, FIELDS = ("nodewise", "observation_reserve"), ("frame_idx", "pose_xyz", "observed")
ALL_FIELDS = (*FIELDS, "sources", "identity_review", "proposal_kind")
VIEWS, SHAPE = drawing.VIEWS, drawing.SHAPE
RIG_PINS = {
    "calibration_manifest.json": "9bb7dda927c01c018c411d3b56da58f3e44379f4d4ce1491c5ab67d33e5bf105",
    "cam0_bundle_refractive.json": (
        "f6756885f0b4b2c86033f1818f3681a85084ff56f40dd478ccdc66720063106c"
    ),
    "cam1_bundle_refractive.json": (
        "393cbe50dc8d434d97e2a6ec41005abdb92e08d1691370f42a4de978c31af7f0"
    ),
    "cam2_bundle_refractive.json": (
        "87af873ff80710f51c622c9305b20a5219b3aaede3837cdd92c7964ae0cd06f8"
    ),
}
PINS = {
    "work/three_body/render_nodewise_first_minute_peak_review.py": (
        "cbb844fc131c8a6d354406f15522f5369099e193fda6c325331a9cb6ed403b43"
    ),
    "work/three_body/prepare_top_background_5to6_minutes.py": (
        "3c4bd35a83c192cc61082ef783fe5d3859438943d2ea23f62b4b25b6e8b6175c"
    ),
    "work/three_body/compare_observation_reserve_continuation.py": COMPARISON_SHA_SOURCE,
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
}


def reference(root, seal, name):
    row = seal["outputs"][name]
    path = Path(row["path"]).resolve()
    require(
        path == (root / name).resolve() and path.is_relative_to(root.resolve()),
        "sealed member escaped its root",
    )
    return row


def capture(reference, evidence):
    data = evidence.capture(reference["path"], reference["sha256"])
    require(len(data) == reference["size_bytes"], "sealed member size differs")
    return data


def document(root, seal, name, evidence):
    return json.loads(capture(reference(root, seal, name), evidence))


def source_code(evidence):
    for path, digest in PINS.items():
        evidence.capture(REPO / path, digest)
    require(
        Path(calib.__file__).resolve() == REPO / "src/medaka3d/calib.py",
        "registered calibrated projector required",
    )
    return evidence.capture(Path(__file__))


def load_photo_headers(evidence):
    seal = evidence.read(PHOTO_ROOT / "review.provenance.json", PHOTO_SHA)
    require(
        seal["status"] == "ok" and seal["stage"] == PHOTO_ROOT.name,
        "successful native-photo producer required",
    )
    plan, summary, projections = (
        document(PHOTO_ROOT, seal, name + ".json", evidence)
        for name in ("plan", "summary", "projection_evidence")
    )
    require(
        plan["frames"] == summary["frames"] == list(FRAMES)
        and plan["views"] == list(VIEWS)
        and plan["same_terminal_horizon"] == summary["same_terminal_horizon"] == 76360
        and summary["native_PNG_count"] == 9
        and summary["full_native_photographs_saved"] is True
        and projections["schema"] == PHOTO_ROOT.name + "_projections",
        "photo native frames or horizon differ",
    )
    photographs = {}
    for row in projections["native_photographs"]:
        key = (row["frame_idx"], row["view"])
        require(
            key not in photographs and key[0] in FRAMES and key[1] in VIEWS,
            "duplicate or foreign native photograph",
        )
        require(
            row["image"] == reference(PHOTO_ROOT, seal, f"native/frame_{key[0]}_{key[1]}.png")
            and row["native_absolute_seek"] == key[0]
            and row["manifest_relative_row"] == key[0] - 74700
            and row["PNG_roundtrip_verified"] is True
            and row["bgr_signature"]["dtype"] == "|u1"
            and row["bgr_signature"]["shape"] == list(SHAPE),
            "original native photograph binding differs",
        )
        photographs[key] = row
    require(len(photographs) == 9, "all nine photographs required")
    rig_refs = plan["inputs"]["rig_references"]
    require(
        {Path(p).name: h for p, h in rig_refs.items()} == RIG_PINS and len(rig_refs) == 4,
        "registered rig four-file hashes differ",
    )
    for path, digest in rig_refs.items():
        require(seal["inputs"][path]["sha256"] == digest, "rig photo producer input edge differs")
    directories = {str(Path(path).parent) for path in rig_refs}
    require(len(directories) == 1, "rig must be one registered bundle")
    return dict(
        photographs=photographs,
        rig_references=rig_refs,
        rig_directory=directories.pop(),
        photo_sha256=PHOTO_SHA,
    )


def load_headers(result_root, result_sha, evidence):
    result_root = Path(result_root).resolve()
    require(
        len(result_sha) == 64 and all(c in "0123456789abcdef" for c in result_sha),
        "explicit successful comparison SHA required",
    )
    seal = evidence.read(result_root / "comparison.provenance.json", result_sha)
    require(
        seal["status"] == "ok" and seal["stage"] == COMPARISON_STAGE,
        "successful fixed-window comparison required",
    )
    plan, summary, signatures = (
        document(result_root, seal, name + ".json", evidence)
        for name in ("plan", "summary", "review_sample_signatures")
    )
    expected = dict(
        schema=COMPARISON_STAGE,
        process_interval=[74700, 76360],
        retained_interval=[49800, 76360],
        terminal_horizon=76360,
        native_frames=1660,
        development_probes=list(PROBES),
        baseline_milestone_summary_sha256=MILESTONE_SHA,
        baseline_qualification_sha256=QUALIFIED_SHA,
        same_parent_checkpoint_sha256=CHECKPOINT_SHA,
        same_parent39_signatures_and_metadata=True,
        same_chunk15_and_registered16_prefix=True,
        parent_full_five_minute_completion_claimed=False,
        support_recomputed=False,
        raw_slot_correspondence_is_not_physical_ID=True,
    )
    for doc in (plan, summary):
        require(
            all(doc.get(k) == v for k, v in expected.items()),
            "comparison scientific/native/parent binding differs",
        )
    source = str((WORK / "compare_observation_reserve_continuation.py").resolve())
    require(
        summary["status"] == "ok"
        and seal["inputs"][source]
        == plan["source_sha256"][source]
        == summary["source_sha256"][source]
        == COMPARISON_SHA_SOURCE,
        "comparison source binding differs",
    )
    require(
        set(signatures) == {arm + "_" + field for arm in ARMS for field in ALL_FIELDS},
        "twelve declared comparison sample fields required",
    )
    photo = load_photo_headers(evidence)
    return dict(
        **photo,
        sample_reference=reference(result_root, seal, "review_samples.npz"),
        sample_signatures=signatures,
        comparison_sha256=result_sha,
        comparison_binding=expected,
    )


def load_samples(context, evidence):
    raw = capture(context["sample_reference"], evidence)
    selected, verified = {}, {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        require(
            len(archive.files) == 12 and set(archive.files) == set(context["sample_signatures"]),
            "comparison sample archive membership differs",
        )
        for arm in ARMS:
            fields = {}
            for field in FIELDS:
                key = arm + "_" + field
                array = archive[key]
                require(
                    prep.signature(array) == context["sample_signatures"][key],
                    "selected sample array signature differs",
                )
                verified[key] = prep.signature(array)
                fields[field] = array
            frames, poses, observed = (fields[k] for k in FIELDS)
            require(
                frames.shape == (5,)
                and frames.dtype.kind in "iu"
                and np.array_equal(frames, PROBES),
                "five ordered native samples required",
            )
            require(
                poses.shape == (5, 3, 3, 3)
                and poses.dtype == np.float64
                and np.isfinite(poses).all(),
                "finite fitted float64 whole-body poses required",
            )
            require(
                observed.shape == (5, 3, 3) and observed.dtype == np.bool_,
                "stored global direct support axes differ",
            )
            indices = np.searchsorted(frames, FRAMES)
            require(np.array_equal(frames[indices], FRAMES), "display frame membership differs")
            selected[arm] = {field: array[indices].copy() for field, array in fields.items()}
    return selected, verified


def load_rig(context, evidence):
    docs = {p: evidence.read(p, h) for p, h in context["rig_references"].items()}
    manifest = docs[str(Path(context["rig_directory"]) / "calibration_manifest.json")]
    require(
        manifest["view_order"] == list(VIEWS) and manifest["backend"] == "refractive",
        "registered refractive rig view order differs",
    )
    for row in manifest["model_files"]:
        require(
            context["rig_references"].get(str(Path(context["rig_directory"]) / row["path"]))
            == row["sha256"],
            "rig model edge differs",
        )
    return calib.load_rig(context["rig_directory"])


def photograph(context, frame, view, evidence):
    row = context["photographs"][(frame, view)]
    raw = capture(row["image"], evidence)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    require(
        image is not None
        and image.shape == SHAPE
        and image.dtype == np.uint8
        and prep.signature(image) == row["bgr_signature"],
        "saved native PNG BGR differs",
    )
    return image


def project_and_crop(samples, rig):
    projected = {
        arm: {
            view: np.asarray(rig.camera(view).project(samples[arm]["pose_xyz"]), np.float64)
            for view in VIEWS
        }
        for arm in ARMS
    }
    require(
        all(p.shape == (3, 3, 3, 2) for views in projected.values() for p in views.values()),
        "native projection axes differ",
    )
    crops = {
        view: drawing.crop_for(np.stack([projected[arm][view] for arm in ARMS])) for view in VIEWS
    }
    return projected, crops


def draw_frame(frame, index, samples, projected, crops, images):
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
        f"Native {frame}; final horizon 76360; raw slots are not physical IDs",
        "M circle / C square / T triangle; filled D=stored GLOBAL direct; open N=not direct",
        "D/N is not per-camera truth. Same crop across both arms and all three native frames.",
        "Known development cases only; whole-window regressions need separate review.",
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


def run(result_root, result_sha, output):
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
            else (json.dumps(drawing.json_value(value), indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=prep.sha(data), size_bytes=len(data))

    try:
        write("source_snapshot.py", source_code(evidence))
        context = load_headers(result_root, result_sha, evidence)
        samples, verified = load_samples(context, evidence)
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            rig = load_rig(context, evidence)
            projected, crops = project_and_crop(samples, rig)
        plan = dict(
            schema=STAGE,
            frames=list(FRAMES),
            views=list(VIEWS),
            row_order=["raw", *ARMS],
            comparison_sha256=result_sha,
            photo_sha256=PHOTO_SHA,
            source_pins=PINS,
            comparison_binding=context["comparison_binding"],
            sample_reference=context["sample_reference"],
            decoded_sample_signatures=verified,
            rig_references=context["rig_references"],
            crops_native_xywh=crops,
            stored_observed_is_global_direct_not_per_view=True,
            support_recomputed=False,
            physical_ID_truth=False,
            known_development_cases=True,
            original_video_SL_mask_checkpoint_GPU_reads=0,
        )
        write("plan.json", plan)
        selected, photos = [], []
        for index, frame in enumerate(FRAMES):
            images = {view: photograph(context, frame, view, evidence) for view in VIEWS}
            rendered = draw_frame(frame, index, samples, projected, crops, images)
            write(f"frame_{frame}.png", prep.png_bytes(rendered))
            preview = cv2.resize(
                rendered,
                (1440, round(rendered.shape[0] * 1440 / rendered.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
            okay, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            require(okay, "comparison preview encoding failed")
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
            photos.extend(context["photographs"][(frame, view)] for view in VIEWS)
        write(
            "projection_evidence.json",
            dict(
                schema=STAGE + "_projections",
                selected=selected,
                native_photographs=photos,
                crops_native_xywh=crops,
                support_recomputed=False,
                nonfinite_projection_saved_as_null=True,
                raw_slots_are_physical_IDs=False,
            ),
        )
        write(
            "summary.json",
            dict(
                **plan,
                status="ok",
                PNG_count=3,
                preview_JPEG_count=3,
                preview_width=1440,
                preview_is_lossy_copy_of_rendered_PNG=True,
                native_PNG_decodes=9,
                comparison_sample_archive_decodes=1,
                decoded_sample_fields=6,
                declared_fields=12,
                other_six_sample_fields_numerically_verified=False,
                same_terminal_horizon=76360,
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
    parser.add_argument("--comparison-root", type=Path, default=WORK / COMPARISON_STAGE)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_root, args.comparison_sha, args.output)
