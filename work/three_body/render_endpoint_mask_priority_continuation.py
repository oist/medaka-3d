"""Three completed same91300 paths on nine sealed lossless native photographs.

Decode only fifteen small sample arrays and nine PNGs; project using four
registered calibration JSONs. No original video, mask, SLEAP, full trajectory,
checkpoint or GPU reads. Raw slot colors are not matched physical identities.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import compare_endpoint_mask_priority_continuation as comparison
import numpy as np
import prepare_endpoint_negative_native_photos as photos
import render_partial_body_bank_comparison as drawing

proof, require, WORK, REPO = drawing.proof, drawing.require, drawing.WORK, drawing.REPO
VIEWS, FRAMES, ARMS = comparison.VIEWS, comparison.FRAMES, comparison.ARMS
STAGE = "endpoint_mask_priority_ten_second_visual_comparison_v1"
FIELDS = ("frame_idx", "pose_xyz", "observed", "sources", "identity_review")
COMPARISON_SHA = "d368061646a4f580753faf9f17773688e0d46f87a4fb3fd5747225ebe86e6078"
PHOTO_SOURCE_SHA = "70ea8dc9c99a1cbcce5b737e05fa1dde4dd7fbcbdfe6e7a9ff2c8938cc86f7b0"
PHOTO_SEAL_SHA = "dac90d016b249e45fd63d0030ec0f4b041a1b7f414f46cd93049ba29acf67256"
PINS = {
    **photos.PINS,
    **{"work/three_body/" + p: h for p, h in comparison.PINS.items()},
    "work/three_body/compare_endpoint_mask_priority_continuation.py": COMPARISON_SHA,
    "work/three_body/prepare_endpoint_negative_native_photos.py": PHOTO_SOURCE_SHA,
    "work/three_body/render_partial_body_bank_comparison.py": (
        "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
    ),
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
        evidence.add(REPO / name, digest)
    code = {}
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
                code[str(path)] = evidence.add(path)
    return code


def load_headers(result_root, result_sha, photo_root, photo_sha, evidence):
    result_root, photo_root = Path(result_root).resolve(), Path(photo_root).resolve()
    require(
        result_root.is_relative_to(WORK) and photo_root.is_relative_to(WORK),
        "workspace sealed inputs required",
    )
    require(photo_sha == PHOTO_SEAL_SHA, "fixed completed native photo seal required")
    seal = drawing.read(result_root / "comparison.provenance.json", result_sha, evidence)
    require(
        seal["status"] == "ok" and seal["stage"] == comparison.STAGE,
        "completed three-path comparison required",
    )
    drawing.edge(seal, Path(comparison.__file__), COMPARISON_SHA)
    summary, plan, signs, probes = [
        drawing.document(result_root, seal, name + ".json", evidence)
        for name in (
            "summary",
            "plan",
            "review_samples_signatures",
            "priority_final_probe_classification",
        )
    ]
    for doc in (plan, summary):
        require(
            doc["schema"] == comparison.STAGE
            and doc["process_interval"] == [89640, 91300]
            and doc["terminal_horizon"] == 91300
            and doc["frames"] == list(FRAMES),
            "native interval or final horizon differs",
        )
    require(
        summary["status"] == "ok"
        and summary["arms"] == list(ARMS)
        and plan["original39_metadata_and_signatures_equal"] is True
        and plan["parent_checkpoint_sha256"] == comparison.INITIAL_SHA
        and plan["candidate_references"] == summary["candidate_references"],
        "same-parent completed comparison required",
    )
    require(
        plan["source_sha256"].get(str(Path(comparison.__file__).resolve())) == COMPARISON_SHA,
        "comparison plan source differs",
    )
    photo_seal = drawing.read(photo_root / "preparation.provenance.json", photo_sha, evidence)
    require(
        photo_seal["status"] == "ok" and photo_seal["stage"] == photos.STAGE,
        "completed native photograph preparation required",
    )
    drawing.edge(photo_seal, Path(photos.__file__), PHOTO_SOURCE_SHA)
    photo_plan, photo_summary, doc = [
        drawing.document(photo_root, photo_seal, name + ".json", evidence)
        for name in ("plan", "summary", "frame_evidence")
    ]
    require(
        photo_plan["schema"] == photo_summary["schema"] == photos.STAGE
        and photo_summary["status"] == "ok"
        and photo_summary["native_PNG_count"] == 9
        and photo_plan["frames"] == photo_summary["frames"] == list(FRAMES)
        and photo_plan["views"] == photo_summary["views"] == list(VIEWS)
        and doc["schema"] == photos.STAGE + "_frame_evidence"
        and doc["JPEG_pixels_used"] is False,
        "nine native photo schema differs",
    )
    require(
        photo_plan["producer_sha256"]
        == summary["candidate_references"]["nodewise"]["seal_sha256"]
        == photos.PRODUCER_SHA
        and photo_plan["original_binding"] == plan["original_binding"]
        and photo_plan["source_chunk"]["index"] == 24
        and photo_plan["source_chunk"]["manifest_sha256"] == photos.MANIFEST_SHA,
        "photographs and paths must use the same source observation",
    )
    rows = doc["records"]
    require(
        [(r["frame_idx"], r["view"]) for r in rows] == [(f, v) for f in FRAMES for v in VIEWS],
        "photo coverage or native order differs",
    )
    records = {}
    for r in rows:
        frame, view = r["frame_idx"], r["view"]
        name = f"native/frame_{frame}_{view}.png"
        require(
            r["image"] == photo_seal["outputs"][name]
            and r["native_absolute_seek"] == frame
            and r["consumed_chunk_offset"] == frame - 89640
            and r["camera_sync_offset_applied"] == 0
            and r["lossless_PNG_roundtrip_verified"] is True,
            "photo evidence membership/native mapping differs",
        )
        records[frame, view] = r
    shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    rig_refs = {p: h for p, h in shared.items() if Path(p).suffix == ".json"}
    require(
        len(rig_refs) == 4
        and sum(Path(p).name == "calibration_manifest.json" for p in rig_refs) == 1,
        "registered four-file rig required",
    )
    return dict(
        result_root=result_root,
        result_seal=seal,
        plan=plan,
        signatures=signs,
        probes=probes,
        photo_root=photo_root,
        photo_seal=photo_seal,
        photos=records,
        rig_refs=rig_refs,
    )


def validate_samples(samples):
    require(set(samples) == {a + "_" + k for a in ARMS for k in FIELDS}, "fifteen sample fields")
    shapes = {
        "frame_idx": ((3,), "i"),
        "pose_xyz": ((3, 3, 3, 3), "f"),
        "observed": ((3, 3, 3), "b"),
        "sources": ((3, 3, 3, 3), "i"),
        "identity_review": ((3, 3), "b"),
    }
    for arm in ARMS:
        for field, (shape, kind) in shapes.items():
            value = samples[arm + "_" + field]
            require(value.shape == shape and value.dtype.kind == kind, "sample axes: " + field)
        require(
            samples[arm + "_frame_idx"].dtype == np.int64
            and np.array_equal(samples[arm + "_frame_idx"], FRAMES),
            "fixed native sample order",
        )
        pose = samples[arm + "_pose_xyz"]
        require(pose.dtype == np.float64 and np.isfinite(pose).all(), "finite FP64 fitted poses")
        source, observed = samples[arm + "_sources"], samples[arm + "_observed"]
        require(
            np.all(source >= -1) and not np.any(observed & ((source >= 0).sum(2) < 2)),
            "stored direct support source count differs",
        )


def load_samples(context, evidence):
    path = drawing.previous.member(
        context["result_root"], context["result_seal"], "review_samples.npz", evidence
    )
    expected = {a + "_" + k for a in ARMS for k in comparison.FIELDS}
    signs = context["signatures"]
    require(set(signs) == expected and len(expected) == 27, "complete27 sample signatures")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == 27 and set(archive.files) == expected, "sample27 membership")
        samples = {a + "_" + k: archive[a + "_" + k] for a in ARMS for k in FIELDS}
    require(
        comparison.common.signature(samples) == {k: signs[k] for k in samples},
        "fifteen decoded signatures differ",
    )
    validate_samples(samples)
    require([r["frame_idx"] for r in context["probes"]] == list(FRAMES), "three final probes")
    for i, row in enumerate(context["probes"]):
        labels = row["classification_by_view"]
        require(
            set(labels) == set(VIEWS)
            and all(
                len(labels[v]) == 3 and all(x in CLASS_LABELS for x in labels[v]) for v in VIEWS
            ),
            "priority classification axes",
        )
        split = (np.asarray([labels[v] for v in VIEWS]) == "split_instances").any(0)
        require(
            np.array_equal(row["split_by_raw_body"], split)
            and np.array_equal(row["saved_observed"], samples["endpoint_priority_observed"][i])
            and np.array_equal(
                row["saved_identity_review"], samples["endpoint_priority_identity_review"][i]
            )
            and not np.any(split & ~samples["endpoint_priority_identity_review"][i]),
            "final probe and plotted sample differ",
        )
    for a in samples.values():
        a.flags.writeable = False
    return samples


def load_rig(context, evidence):
    for path, digest in context["rig_refs"].items():
        evidence.add(Path(path), digest)
    folder = next(
        Path(p).parent for p in context["rig_refs"] if Path(p).name == "calibration_manifest.json"
    )
    return photos.frozen.observations.calib.load_rig(folder)


def photograph(context, frame, view, evidence):
    name = f"native/frame_{frame}_{view}.png"
    path = drawing.previous.member(context["photo_root"], context["photo_seal"], name, evidence)
    image = drawing.prior.cv2.imdecode(
        np.frombuffer(path.read_bytes(), np.uint8), drawing.prior.cv2.IMREAD_COLOR
    )
    require(
        image is not None
        and image.dtype == np.uint8
        and image.shape == (1200, 1200, 3)
        and proof.signature(image) == context["photos"][frame, view]["bgr_signature"],
        "saved lossless native pixels differ",
    )
    image.flags.writeable = False
    return image


def draw_frame(samples, index, rig, images, probe):
    projections = {
        a: {v: rig.camera(v).project(samples[a + "_pose_xyz"][index]) for v in VIEWS} for a in ARMS
    }
    require(
        all(
            p.shape == (3, 3, 2) and np.isfinite(p).all()
            for views in projections.values()
            for p in views.values()
        ),
        "finite three-body projections required",
    )
    fig, axes = drawing.prior.plt.subplots(3, 3, figsize=(18, 16), squeeze=False)
    crops = {}
    try:
        for vi, view in enumerate(VIEWS):
            image = images[view]
            empty = np.empty((0, 3, 2))
            lo, hi = drawing.visual.display_crop(
                image, None, empty, np.stack([projections[a][view] for a in ARMS])
            )
            crops[view] = dict(lo=lo.tolist(), hi=hi.tolist())
            for ai, arm in enumerate(ARMS):
                ax = axes[ai, vi]
                drawing.draw_panel(ax, image, empty, projections[arm][view])
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
                support = [
                    "".join(map(str, r)) for r in samples[arm + "_observed"][index].astype(int)
                ]
                review = samples[arm + "_identity_review"][index].astype(int).tolist()
                title = (
                    f"{arm} | {view}\n"
                    f"Stored MCT support S0/1/2={','.join(support)}; review={review}"
                )
                if arm == "endpoint_priority":
                    classes = [CLASS_LABELS[c] for c in probe["classification_by_view"][view]]
                    title += "\nFinal pose D2 endpoints S0/1/2=" + "/".join(classes)
                ax.set_title(title, fontsize=8)
        fig.suptitle(
            f"Native {FRAMES[index]} | All paths end at91300 | Same Sparse state at89640\n"
            "M circle / C square / T triangle: open fitted points; dashed whole raw slots. "
            "Cyan S0 / yellow S1 / magenta S2 are NOT matched physical IDs.\n"
            "Stored support is not a per-camera truth check. Uncovered/ambiguous D2 is unknown. "
            "Lossless native PNG; same crop for all methods; no sex/identity certification.",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=160)
        return stream.getvalue(), projections, crops
    finally:
        drawing.prior.plt.close(fig)


def run(result_root, result_sha, photo_root, photo_sha, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output",
    )
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code(evidence)
        context = load_headers(result_root, result_sha, photo_root, photo_sha, evidence)
        samples = load_samples(context, evidence)
        rig = load_rig(context, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_code=code,
                comparison_sha256=result_sha,
                photo_sha256=photo_sha,
                frames=list(FRAMES),
                arms=list(ARMS),
                terminal_horizon=91300,
                source_interval=[89640, 91300],
                rig_references=context["rig_refs"],
                sample_archive=context["result_seal"]["outputs"]["review_samples.npz"],
                decoded_sample_fields=list(FIELDS),
                archive_membership_fields=list(comparison.FIELDS),
                other_twelve_sample_arrays_numerically_verified=False,
                producer_final_path_binding_inherited=True,
            ),
        )
        cases = []
        for i, frame in enumerate(FRAMES):
            images = {v: photograph(context, frame, v, evidence) for v in VIEWS}
            data, projections, crops = draw_frame(samples, i, rig, images, context["probes"][i])
            image_ref = out.write(f"frame_{frame}.png", data)
            cases.append(
                dict(
                    frame_idx=frame,
                    image=image_ref,
                    crops=crops,
                    photographs={v: context["photos"][frame, v] for v in VIEWS},
                    methods={
                        a: dict(
                            **{
                                k: samples[a + "_" + k][i].tolist()
                                for k in FIELDS
                                if k != "frame_idx"
                            },
                            projections={v: p.tolist() for v, p in projections[a].items()},
                        )
                        for a in ARMS
                    },
                    priority_final_probe=context["probes"][i],
                )
            )
            del images
        out.write("projection_evidence.json", dict(schema=STAGE + "_projections", cases=cases))
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                arms=list(ARMS),
                terminal_horizon=91300,
                comparison_sha256=result_sha,
                photo_sha256=photo_sha,
                rendered_PNG_count=3,
                native_PNG_decodes=9,
                sample_arrays_decoded=15,
                registered_rig_JSON_reads=4,
                original_video_SLEAP_mask_checkpoint_or_full_candidate_reads=0,
                selected_terminal_path_not_current_beam=True,
                raw_slots_preserved=True,
                per_method_physical_ID_alignment=False,
                support_recomputed=False,
                physical_ID_truth=False,
            ),
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
    except BaseException as exc:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(exc),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=WORK / comparison.STAGE)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--photos", type=Path, default=WORK / photos.STAGE)
    parser.add_argument("--photos-sha", default=PHOTO_SEAL_SHA)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison, args.comparison_sha, args.photos, args.photos_sha, args.output)
