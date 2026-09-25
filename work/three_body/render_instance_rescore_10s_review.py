"""Three fixed native frames of a successfully compared same-78020 experiment.

Exactly nine new native BGR decodes, checked against the sealed raw diagnosis.
No whole-video hash, checkpoint/DAG read, model, tracking, fitting or human labels.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import compare_temporal_point64_v2 as proof
import numpy as np
import render_fast_tail_capacity_island as prior

WORK, REPO = prior.WORK, prior.REPO
RAW_ROOT = WORK / "fast_tail_capacity_island_raw_diagnosis_v1"
RAW_SHA = "70ff3c73c2bf104e250ad47cb3e4ae864328145d0a95219253400f6162c6ad20"
STAGE = "instance_rescore_ten_second_raw_review_v1"
COMPARISON_STAGE = "instance_rescore_ten_second_comparison_v1"
FRAMES, ARMS = (77709, 77723, 77880), ("fast_tail_control", "instance_rescore")
VIEWS, KEYS = prior.legacy.VIEWS, prior.KEYS
EXECUTION_READY = True
require = proof.require
# Imported dependencies omitted from the old diagnostic's input closure. These
# are separately frozen current source pins, not claims of old consumption.
ADDITIONAL_SOURCE_PINS = {
    "work/three_body/compare_bounded_one_minute.py": (
        "5e1a4f570afadf204f96f3c6e83d55a9089f6fa3a14cf1b4c54cc84e4fe76cd0"
    ),
    "work/three_body/compare_tail_one_minute.py": (
        "da5f50c1dcc1399464c9ca8edfd65287a9f037f78119eee17cef7c27be388737"
    ),
    "work/three_body/mask_transport_diagnostics.py": (
        "6567710637b4a3b69dd03106cc525a1f7fac1659a7ab54acadef3c98517f3e56"
    ),
    "work/three_body/fast_tail_two_minute_metrics.py": (
        "a3ebd24da07308938899695723090fbde20c4bbe27f5dd8fb9d441bef9ee9ddb"
    ),
    "work/three_body/compare_fast_tail_two_minutes.py": (
        "e07b51f364f5d6f85e8e26d34a0051145412df29e34c91d379d2141aec9b649d"
    ),
    "src/medaka3d/eval/encounter_review.py": (
        "49538a9d76180642fce84b4ad90e7c7675ee759aa8f00f573b2e6a60c714d2f9"
    ),
    "labs/20260909_three_body/review_truth.py": (
        "9dd5fe7b96d4a8d84954604a2e647be6bfa2d9652ad78378c51618c5fa70dbd9"
    ),
    "labs/20260909_three_body/evaluate_review.py": (
        "212ff70e92675f7880fd9ca4f02cf31815282b064a62315c31eaeb6ef0e66190"
    ),
    "labs/20260909_three_body/review_scene.py": (
        "4b266f08e9832f1f420fa29912cb6d873f7a935361f8602d0f11f49431b77f71"
    ),
    "work/three_body/import_long_review.py": (
        "4ee9a2af648231658714c017ff3d48232700f09a701092f7a70a75869421bcc5"
    ),
    "work/three_body/receipted_long_phase_operation.py": (
        "fddad2527c4d423a96cf4ccd84367eb22a16e77f23f899f863bd0ed6f8649d97"
    ),
    "work/three_body/receipted_long_handoff.py": (
        "8d029ae65c85cdfc2696c680a6a5ff4637808af389296ca43730c3fde8f053b4"
    ),
    "work/three_body/run_long_receipted_supervisor.py": (
        "ddf95225f99e908e1b065915111766d069badf31f8b048414073c7d5855a9ee6"
    ),
    "work/three_body/long_final_mixed_chain.py": (
        "b41c30830c61480df7e847d6814756743d115cfaf8cdaead85fd922b01a34215"
    ),
}


def read(path, evidence, digest):
    path = prior.raw.local(path)
    evidence.add(path, digest)
    return json.loads(path.read_text(encoding="utf8"))


def member(root, seal, name, evidence):
    path = prior.native.checked_relative(root, name)
    row = seal["outputs"][name]
    require(
        Path(row["path"]).resolve() == path and path.stat().st_size == row["size_bytes"],
        "sealed member path/size differs",
    )
    evidence.add(path, row["sha256"])
    return path


def shared_edge(path, digest, raw_seal, comparison_seal):
    path = str(Path(path).resolve())
    require(
        raw_seal["inputs"].get(path) == comparison_seal["inputs"].get(path) == digest,
        "raw/comparison observation source binding differs",
    )


def video_stat_guard(videos, chunk):
    """Stat/hardlink identity only. Large raw files are never added to a hash guard."""
    for view, path in videos.items():
        link = chunk["raw_hardlinks"][view]
        canonical = prior.raw.local(link["source_path"])
        stat = path.stat()
        require(
            Path(link["path"]).resolve() == path
            and os.path.samefile(path, canonical)
            and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            == (link["device"], link["inode"], link["bytes"], link["mtime_ns"]),
            "sealed raw hardlink identity/stat differs",
        )


def load(comparison_root, comparison_sha, evidence):
    comp = read(comparison_root / "comparison.provenance.json", evidence, comparison_sha)
    require(
        comp["status"] == "ok" and comp["stage"] == COMPARISON_STAGE,
        "successful same78020 comparison required",
    )
    summary = json.loads(member(comparison_root, comp, "summary.json", evidence).read_text("utf8"))
    require(
        summary["status"] == "ok"
        and summary["native_interval"] == [76360, 78020]
        and summary["native_frames"] == 1660
        and summary["terminal_horizon_by_arm"] == dict.fromkeys(ARMS, 78020)
        and summary["all_three_physical_ID_truth"] is False,
        "fixed comparison horizon/scope differs",
    )
    rows, references = {}, {}
    for arm in ARMS:
        name = f"candidates/{arm}.npz"
        path = member(comparison_root, comp, name, evidence)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in KEYS}
        require(
            all(
                proof.signature(value) == summary["candidate_arrays"][arm][key]
                for key, value in arrays.items()
            ),
            "compared five candidate fields differ",
        )
        require(
            np.array_equal(arrays["frame_idx"], np.arange(49800, 78020)),
            "candidate must end at78020, not86320",
        )
        rows[arm] = prior.legacy.anchor_arrays(arrays, np.array(FRAMES, dtype=np.int64))
        references[arm] = comp["outputs"][name]
    source = read(RAW_ROOT / "diagnosis.provenance.json", evidence, RAW_SHA)
    require(
        source["status"] == "ok" and source["stage"] == prior.STAGE,
        "successful fixed raw diagnosis required",
    )
    data = json.loads(
        member(RAW_ROOT, source, "source_and_frame_evidence.json", evidence).read_text("utf8")
    )
    require(
        data["frames"] == list(FRAMES)
        and data["horizon_stop"] == 86320
        and data["native_BGR_decodes"] == 9
        and data["decoded_mask_frames"] == 9,
        "fixed original raw diagnostic scope differs",
    )
    chunk = data["source_chunk"]
    index_paths = [Path(p) for p in comp["inputs"] if Path(p).name == "input_index.json"]
    require(len(index_paths) == 1, "one comparison-bound producer input index required")
    index_path = index_paths[0]
    index = read(index_path, evidence, comp["inputs"][str(index_path)])
    require(
        len(index["prefix_chunks"]) == 17 and index["prefix_chunks"][16] == chunk,
        "new producer did not consume the same source chunk",
    )
    manifest_path = prior.raw.local(chunk["output_hashes"]["manifest"]["path"])
    require(
        source["inputs"].get(str(manifest_path)) == chunk["manifest_sha256"],
        "sealed raw manifest input edge differs",
    )
    manifest = read(manifest_path, evidence, chunk["manifest_sha256"])
    prior.raw.validate_source(
        manifest, chunk, index["original_binding"]["inputs"]["detector_binding"]
    )
    videos, packed = {}, {}
    for view in VIEWS:
        descriptor = manifest["raw"][view]
        path = prior.native.checked_relative(manifest_path.parent, descriptor["path"])
        link = chunk["raw_hardlinks"][view]
        require(
            source["inputs"].get(str(path)) == descriptor["sha256"] == link["sha256"]
            and Path(chunk["output_hashes"][view + "_raw"]["path"]).resolve() == path,
            "sealed raw payload SHA edge differs",
        )
        videos[view] = path
        descriptor = manifest["masks"][view]
        path = prior.native.checked_relative(manifest_path.parent, descriptor["path"])
        require(
            source["inputs"].get(str(path)) == descriptor["sha256"],
            "sealed mask input edge differs",
        )
        evidence.add(path, descriptor["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            packed[view] = {key: archive[key] for key in archive.files}
        require(
            np.array_equal(packed[view]["frame_idx"], manifest["frames"]),
            "mask native frames differ",
        )
        sl_paths = [
            p for p in source["inputs"] if Path(p).name == "20251225_K_1_2_FCMC__" + view + ".npz"
        ]
        require(len(sl_paths) == 1, "one registered raw SL archive per view required")
        # Frame-local SL bytes are restored below; no full SL archive rehash or decode.
        shared_edge(sl_paths[0], source["inputs"][sl_paths[0]], source, comp)
    video_stat_guard(videos, chunk)
    calibration_paths = [
        Path(p) for p in source["inputs"] if Path(p).name == "calibration_manifest.json"
    ]
    require(len(calibration_paths) == 1, "one calibrated rig required")
    calibration_path = calibration_paths[0]
    shared_edge(calibration_path, source["inputs"][str(calibration_path)], source, comp)
    calibration = read(calibration_path, evidence, source["inputs"][str(calibration_path)])
    for row in calibration["model_files"]:
        path = prior.native.checked_relative(calibration_path.parent, row["path"])
        shared_edge(path, row["sha256"], source, comp)
        evidence.add(path, row["sha256"])
    rig = prior.legacy.load_rig(calibration_path.parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "calibration axes differ")
    additional = {str(REPO / name): digest for name, digest in ADDITIONAL_SOURCE_PINS.items()}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if (
            path.suffix == ".py"
            and path.is_relative_to(REPO)
            and ".venv" not in path.parts
            and path.is_file()
        ):
            digest = (
                proof.sha(path)
                if path == Path(__file__).resolve()
                else source["inputs"].get(str(path), additional.get(str(path)))
            )
            require(digest is not None, "unbound rendering dependency: " + str(path))
            evidence.add(path, digest)
    return dict(
        rows=rows,
        references=references,
        data=data,
        chunk=chunk,
        manifest=manifest,
        videos=videos,
        packed=packed,
        rig=rig,
    )


def render(frame, images, masks, points, projections, rows, t):
    fig, axes = prior.plt.subplots(3, 3, figsize=(18, 17), dpi=160)
    fig.suptitle(
        f"Native {frame} | both candidates end at78020 | model diagnosis, not ID truth\n"
        "Rows: raw observations / FastTail control / instance rescore. Columns: three cameras.\n"
        "White D2 contours; dotted untracked SL rows. "
        "S0 cyan / S1 yellow / S2 magenta are RAW slots.\n"
        "M circle / C square / T triangle: filled = >=2-view direct support; open = latent. "
        "Same crop per camera.",
        fontsize=11,
    )
    for vi, view in enumerate(VIEWS):
        cloud = np.concatenate(
            [points[view].reshape(-1, 2), *(projections[a][view].reshape(-1, 2) for a in ARMS)]
        )
        finite = cloud[np.isfinite(cloud).all(axis=1)]
        lo, hi = np.array([0, 0]), np.array([1200, 1200])
        if len(finite):
            lower, upper = (
                np.clip(finite.min(0) - 60, 0, 1200),
                np.clip(finite.max(0) + 60, 0, 1200),
            )
            if np.all(lower < upper):
                lo, hi = lower, upper
        for ri, arm in enumerate(("raw_observations", *ARMS)):
            ax = axes[ri, vi]
            ax.imshow(prior.cv2.cvtColor(images[view], prior.cv2.COLOR_BGR2RGB))
            prior.drawing.draw_observation(
                ax,
                dict(
                    masks=np.empty((0, 1200, 1200), bool) if masks[view] is None else masks[view],
                    detections=points[view],
                ),
            )
            if ri:
                prior.drawing.draw_pose(ax, projections[arm][view], rows[arm]["observed"][t])
            ax.set_xlim(float(lo[0]), float(hi[0]))
            ax.set_ylim(float(hi[1]), float(lo[1]))
            ax.set_title(f"{view} | {arm}")
            ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    stream = io.BytesIO()
    fig.savefig(stream, format="png")
    prior.plt.close(fig)
    return stream.getvalue()


def run(comparison_root, comparison_sha, output):
    require(EXECUTION_READY, "source review and actual render Go required")
    require(proof.valid_sha(comparison_sha), "successful comparison SHA required")
    comparison_root, output = comparison_root.resolve(strict=True), output.resolve()
    require(
        comparison_root.is_relative_to(WORK)
        and output.is_relative_to(WORK)
        and not output.exists(),
        "workspace comparison and fresh output required",
    )
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        context = load(comparison_root, comparison_sha, evidence)
        rows = context["rows"]
        before = {a: {k: proof.signature(v) for k, v in row.items()} for a, row in rows.items()}
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                arms=list(ARMS),
                horizon_stop=78020,
                source_raw_diagnosis_sha256=RAW_SHA,
                comparison_sha256=comparison_sha,
                native_BGR_decodes=9,
                whole_video_rehashes=0,
                checkpoint_or_DAG_reads=0,
                prior_86320_error_is_not_assumed_to_recur_at78020=True,
                additional_current_source_pins=ADDITIONAL_SOURCE_PINS,
                raw_slot_labels_are_physical_IDs=False,
                human_labels_created=False,
            ),
        )
        records = []
        for t, frame in enumerate(FRAMES):
            iterator = prior.raw.iter_native(
                context["videos"], context["manifest"], frame, frame + 1
            )
            try:
                native, images = next(iterator)
                require(
                    native == frame and next(iterator, None) is None, "single native frame required"
                )
            finally:
                iterator.close()
            source = context["data"]["records"][t]
            require(source["frame"] == frame, "sealed raw frame order differs")
            masks, points, observations = {}, {}, {}
            for view in VIEWS:
                original = source["observations"][view]
                bgr_sha = prior.prepare_sequence.array_sha256(images[view])
                require(bgr_sha == original["bgr_sha256"], "raw BGR differs from sealed diagnosis")
                pos = frame - context["chunk"]["start"]
                packed = context["packed"][view]
                decoded = prior.prepare_sequence.decode_frame_masks(packed, pos)
                digest = prior.prepare_sequence.array_sha256(decoded)
                require(
                    digest
                    == original["mask_sha256"]
                    == bytes(packed["frame_mask_sha256"][pos]).decode()
                    and len(decoded) == original["mask_count"]
                    and bool(packed["view_available"][pos]) == original["mask_available"],
                    "sealed target mask differs",
                )
                masks[view] = decoded if original["mask_available"] else None
                points[view] = np.asarray(original["SLEAP_points"], dtype=np.float32)
                require(
                    prior.finite_json(points[view]) == original["SLEAP_points"],
                    "SL row bytes differ",
                )
                observations[view] = dict(
                    bgr_sha256=bgr_sha,
                    mask_sha256=digest,
                    SLEAP_points=prior.finite_json(points[view]),
                )
            projections = {
                a: {v: context["rig"].camera(v).project(row["pose_xyz"][t]) for v in VIEWS}
                for a, row in rows.items()
            }
            methods = {
                a: prior.metrics(
                    row["pose_xyz"][t],
                    row["observed"][t],
                    row["sources"][t],
                    projections[a],
                    masks,
                    points,
                    context["rig"],
                )
                for a, row in rows.items()
            }
            old = source["methods"]["fast_tail"]
            horizon_deltas = {
                v: np.linalg.norm(
                    projections[ARMS[0]][v]
                    - np.array(
                        [r["projected_uv"] for r in old["views"][v]], dtype=np.float64
                    ).reshape(3, 3, 2),
                    axis=-1,
                )
                for v in VIEWS
            }
            out.write(
                f"frame_{frame}.png", render(frame, images, masks, points, projections, rows, t)
            )
            records.append(
                dict(
                    frame=frame,
                    observations=observations,
                    methods=methods,
                    control78020_vs_prior86320=dict(
                        raw_slot_projected_displacement_px={
                            v: prior.finite_json(d) for v, d in horizon_deltas.items()
                        },
                        observed_equal=rows[ARMS[0]]["observed"][t].tolist() == old["observed"],
                        sources_equal=rows[ARMS[0]]["sources"][t].tolist() == old["sources"],
                        is_same_horizon_method_comparison=False,
                        previous_error_recurrence_must_be_visually_checked=True,
                    ),
                )
            )
            print(
                f"Rendered same78020 native {frame}; native BGR decodes {(t + 1) * 3}/9", flush=True
            )
        require(
            before
            == {a: {k: proof.signature(v) for k, v in row.items()} for a, row in rows.items()},
            "candidate rows changed",
        )
        out.write(
            "source_and_frame_evidence.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                horizon_stop=78020,
                comparison_sha256=comparison_sha,
                source_raw_diagnosis_sha256=RAW_SHA,
                candidate_references=context["references"],
                selected_candidate_arrays=before,
                source_raw_hardlinks=context["chunk"]["raw_hardlinks"],
                records=records,
                native_BGR_decodes=9,
                target_mask_decodes=9,
                all_nine_BGR_SHA_match=True,
                whole_video_rehashes=0,
                checkpoint_or_DAG_reads=0,
                GPU_used=False,
                biological_identity_truth=False,
                automatic_adoption=False,
                scope="Fixed diagnostic frames; no held-out accuracy or dense ID continuity.",
            ),
        )
        video_stat_guard(context["videos"], context["chunk"])
        evidence.final_guard()
        out.final_guard()
        require(not sys.modules["torch"].cuda.is_initialized(), "renderer initialized CUDA")
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
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison_root, args.comparison_sha, args.output)
