"""Raw-only continuous review of native [90040,90400), without model suggestions.

Source/CPU preparation until explicitly enabled. The successful transition
renderer selects the scene; canonical video, staging hardlink and native pixel
SHA bind the raw input. Existing anonymous review UI and encoders are reused.
No candidate, mapping, SLEAP or mask arrays are decoded and no solver is called.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import render_fast_tail_five_minute_id_transition as source

proof, require, raw = source.proof, source.require, source.prior.raw
ui, previous, ids = raw.scene_ui, source.previous, source.ids
WORK, REPO, VIEWS = source.WORK, source.REPO, source.VIEWS
STAGE = "fast_tail_transition_raw_review_pack_v1"
SELECTION_ROOT = WORK / "fast_tail_five_minute_id_transition_raw_review_v2"
SELECTION_SHA = "0063b4be2dd9b37200d279c9042dff4862e38d84f9c01446f6559beb146a1924"
SOURCE_SHA = "169c8a7dcd7e3ab932412a968ddae4e04574d99c4e517ee986cdbe4aa544236e"
START, STOP, ANCHORS = 90040, 90400, (90120, 90220, 90380)
FPS, PLAYBACK_FPS = 166, 20
PINS = {
    "work/three_body/prepare_long_raw_review.py": (
        "c1a799d248f9b0f3b0e90329fb034e73af1a6a2c74fab9270dd8df84d07c66f0"
    ),
    "labs/20260909_three_body/review_scene.py": (
        "4b266f08e9832f1f420fa29912cb6d873f7a935361f8602d0f11f49431b77f71"
    ),
    "labs/20260909_three_body/review_truth.py": (
        "9dd5fe7b96d4a8d84954604a2e647be6bfa2d9652ad78378c51618c5fa70dbd9"
    ),
    "labs/20260909_three_body/review_scene.html": (
        "2841767ccb33c59cdc2c965de1a62d03de6c56efba1250c2d2c054c1dc7e8aa9"
    ),
    "labs/20260909_three_body/review_truth.html": (
        "4b74d6036623cc987f2ecfb0e7849ca0f9d1a2798e7070c075e48247c8ecfdef"
    ),
}
EXECUTION_READY = True


def source_code():
    code = source.source_code()
    require(code.get(str(Path(source.__file__).resolve())) == SOURCE_SHA, "frozen renderer changed")
    for name, digest in PINS.items():
        path = (REPO / name).resolve()
        require(raw.sha256(path) == digest, "frozen review helper changed")
        code[str(path)] = digest
    require(str(Path(__file__).resolve()) in code, "new review caller absent from source closure")
    require(
        ui.PLAYBACK_FPS == PLAYBACK_FPS
        and ui.HTML_ORIGIN["sha256"] == PINS["labs/20260909_three_body/review_truth.html"],
        "frozen UI timebase differs",
    )
    return code


def validate_selection(seal, selection, chunk):
    require(
        seal["status"] == "ok" and seal["stage"] == source.STAGE,
        "successful raw selection required",
    )
    ids.edge(seal, Path(source.__file__), SOURCE_SHA)
    require(
        selection["schema"] == source.STAGE
        and selection["status"] == "ok"
        and selection["frames"] == list(source.FRAMES)
        and selection["terminal_horizon"] == 116200
        and selection["source_chunk"] == chunk
        and selection["native_pixel_SHA_checked"] is True,
        "selection scene/source binding differs",
    )
    require(
        chunk["index"] == 24
        and (chunk["start"], chunk["stop"]) == (89640, 91300)
        and chunk["start"] <= START < STOP <= chunk["stop"],
        "whole review must remain in chunk24",
    )


def load_context(evidence):
    selection_seal = previous.read(
        SELECTION_ROOT / "render.provenance.json", evidence, SELECTION_SHA
    )
    selection = proof.read_json(
        previous.member(SELECTION_ROOT, selection_seal, "source_and_frame_evidence.json", evidence)
    )
    tracking_path = ids.TRACKING_ROOT / "tracking.provenance.json"
    ids.edge(selection_seal, tracking_path, ids.TRACKING_SEAL_SHA)
    tracking = previous.read(tracking_path, evidence, ids.TRACKING_SEAL_SHA)
    require(
        tracking["status"] == "ok" and tracking["stage"] == ids.comparison.runner.STAGE,
        "successful final producer required",
    )
    index = ids.document(ids.TRACKING_ROOT, tracking, "input_index.json", evidence)
    plan = ids.document(ids.TRACKING_ROOT, tracking, "plan.json", evidence)
    require(
        index["original_binding"] == plan["original_binding"],
        "producer prefix configuration differs",
    )
    chunk = source.select_chunk(index)
    validate_selection(selection_seal, selection, chunk)
    binding = index["original_binding"]["inputs"]
    root = source.prior.native.checked_relative(
        raw.local(binding["input_stage_root"]), chunk["attempt_path"]
    )
    manifest_path = raw.local(chunk["output_hashes"]["manifest"]["path"])
    require(manifest_path == root / "manifest.json", "consumed manifest root differs")
    for seal in (tracking, selection_seal):
        ids.edge(seal, manifest_path, chunk["manifest_sha256"])
    manifest = previous.read(manifest_path, evidence, chunk["manifest_sha256"])
    staging_path = root / "staging.provenance.json"
    ids.edge(tracking, staging_path, chunk["provenance_sha256"])
    staging = previous.read(staging_path, evidence, chunk["provenance_sha256"])
    require(
        staging["status"] == "ok" and staging["stage"] == "prepare_independent_mask_chunk",
        "successful source staging required",
    )
    require(
        {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in staging["outputs"].items()}
        == chunk["output_hashes"],
        "consumed staging outputs differ",
    )
    raw.validate_source(manifest, chunk, binding["detector_binding"])
    require(
        manifest["binding"] == staging["params"]["binding"], "detector protocol binding differs"
    )
    videos = {}
    for view in VIEWS:
        descriptor, link = manifest["raw"][view], chunk["raw_hardlinks"][view]
        path = source.prior.native.checked_relative(root, descriptor["path"])
        require(
            chunk["output_hashes"][view + "_raw"]
            == dict(path=str(path), sha256=descriptor["sha256"])
            and descriptor["sha256"] == link["sha256"],
            "staged raw output binding differs",
        )
        source.canonical_video_edge(tracking, staging, binding, link, view)
        videos[view] = path  # Never hash the canonical video or staged hardlink payload.
    previous.video_stat_guard(videos, chunk)
    return SimpleNamespace(
        videos=videos,
        manifest=manifest,
        manifest_path=manifest_path,
        chunk=chunk,
        scene=dict(
            context_start_frame=START,
            context_stop_frame_exclusive=STOP,
            proposed_anchor_frames=list(ANCHORS),
        ),
    )


def review_documents(manifest, manifest_sha):
    template, frames, offset = ui.bind_template(
        manifest["frames"], manifest_sha, START, STOP, list(ANCHORS)
    )
    require(
        frames == list(range(START, STOP)) and len(frames) == 360, "all360 native frames required"
    )
    html = ui.render_html(
        (raw.LAB / "review_scene.html").read_text(encoding="utf8"), template, frames
    )
    # Only the generated time label is extended. Frozen UI interactions remain unchanged.
    label = (
        "source ${frame} | original ${(frame/166).toFixed(3)} s | "
        "clip ${((frame-frames[0])/166).toFixed(3)} s"
    )
    for old, new in (
        ("`source ${frame} loading`", "`" + label + " loading`"),
        ("`source ${frame}`", "`" + label + "`"),
    ):
        require(html.count(old) == 1, "frozen frame-label insertion point differs")
        html = html.replace(old, new)
    marker = '<video controls loop src="raw_three_views_slow.mp4"></video>'
    require(html.count(marker) == 1, "frozen video insertion point differs")
    explanation = (
        f"<p>この区間は原映像のフレーム {START}-{STOP - 1} (360枚、"
        f"{360 / FPS:.6f} 秒) です。20 fps の動画では18秒で再生します。"
        "下の時刻は原映像166 fps基準です。"
        "確認時点は仮の候補で、見やすいフレームへ変更できます。</p>"
    )
    html = html.replace(marker, explanation + marker)
    return template, frames, offset, html


def register_output(out, name, digest):
    path = source.prior.native.checked_relative(out.root, name)
    require(
        name not in out.records and path.is_file() and raw.sha256(path) == digest,
        "encoded output registration differs",
    )
    out.records[name] = dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def build_pack(output):
    require(EXECUTION_READY, "source review and explicit ROOT raw-pack Go required")
    ids.cpu_only()
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code()
        for path, digest in code.items():
            evidence.add(Path(path), digest)
        context = load_context(evidence)
        template, frames, offset, html = review_documents(
            context.manifest, context.chunk["manifest_sha256"]
        )
        executable = shutil.which("ffmpeg")
        require(executable is not None, "ffmpeg required for existing H.264 review encoding")
        ffmpeg = Path(executable).resolve(strict=True)
        ffmpeg_sha = raw.sha256(ffmpeg)
        version = subprocess.run(
            [str(ffmpeg), "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        out.write("source_manifest.json", context.manifest_path.read_bytes())
        require(
            out.records["source_manifest.json"]["sha256"] == context.chunk["manifest_sha256"],
            "original manifest bytes changed",
        )
        (output / "frames").mkdir()
        intermediate, hashes, native = raw.write_images(context, output)
        raw.validate_written_scene(context.manifest, frames, list(ANCHORS), hashes, native)
        for name, digest in hashes.items():
            register_output(out, name, digest)
        video = output / "raw_three_views_slow.mp4"
        ui.transcode(ffmpeg, intermediate, video)
        video_qa, intermediate_qa = [
            ui.verify_video(path, len(frames)) for path in (video, intermediate)
        ]
        for path in (video, intermediate):
            register_output(out, path.name, raw.sha256(path))
        out.write("index.html", html.encode("utf8"))
        out.write("truth_template.json", template)
        out.write(
            "anchor_selection.json",
            dict(
                schema=STAGE + "_anchor_selection",
                proposed_anchor_frames=list(ANCHORS),
                review_clip_range=[START, STOP],
                proposed_not_human_confirmed=True,
                actual_clip_view_suitability_checked=False,
                anchor_adjustments=[],
                adjustment_rule=(
                    "Any later clip inspection or anchor change requires a separate dated record; "
                    "preserve this proposal."
                ),
                anonymous_names=["A", "B", "C"],
                physical_identity_labels_created=False,
            ),
        )
        result = dict(
            schema=STAGE,
            status="ok",
            technical_completion=True,
            selection_render_sha256=SELECTION_SHA,
            final_tracking_sha256=ids.TRACKING_SEAL_SHA,
            source_manifest=str(context.manifest_path),
            source_manifest_sha256=context.chunk["manifest_sha256"],
            source_frame_range=template["source_frame_range"],
            review_clip_range=[START, STOP],
            displayed_frames=frames,
            probe_frames=list(ANCHORS),
            frame_count=len(frames),
            source_fps=FPS,
            playback_fps=PLAYBACK_FPS,
            source_duration_seconds=len(frames) / FPS,
            playback_duration_seconds=len(frames) / PLAYBACK_FPS,
            source_absolute_seconds_range=[START / FPS, STOP / FPS],
            source_clip_time_rule="(native_frame - 90040) / 166; source time is native_frame / 166",
            original_source_video_frame_count=raw.STOP,
            absolute_seek_frame=START,
            first_manifest_sha_offset=offset,
            native_pixel_hashes_verified=3 * len(frames),
            native_bgr_sha256=native,
            review_video=video_qa,
            intermediate_video=intermediate_qa,
            consumed_record=context.chunk,
            raw_video_payload_bytes_copied=0,
            whole_video_rehashes=0,
            original_raw_storage=context.manifest["raw_storage"],
            source_manifest_copy_is_audit_only_not_a_relocated_reader_descriptor=True,
            model_outputs_shown=False,
            candidate_arrays_deserialized=False,
            mapping_arrays_deserialized=False,
            masks_or_sleap_arrays_deserialized=False,
            human_truth_created=False,
            source_selection_is_model_derived=True,
            actual_clip_view_suitability_checked=False,
            ui_operations_verified=False,
            browser_used=False,
            gpu_used=False,
            tracking_or_ID_solver_calls=0,
            code_sha256=code,
            ffmpeg=dict(path=str(ffmpeg), sha256=ffmpeg_sha, version=version),
            limitations=[
                "Anonymous A/B/C belong only to this review clip; no model-ID suggestions.",
                "Templates and proposed anchors are not human truth or confirmed visibility.",
                "JPEG/video are lossy; nine anchor PNGs retain exact native pixels.",
                "Source/native/output checks are completed; "
                "interactive browser operations are unverified.",
            ],
        )
        out.write("summary.json", result)
        previous.video_stat_guard(context.videos, context.chunk)
        require(source_code() == code and raw.sha256(ffmpeg) == ffmpeg_sha, "code/runtime changed")
        evidence.final_guard()
        out.final_guard()
        ids.cpu_only()
        seal = dict(stage=STAGE, status="ok", extra_inputs={str(ffmpeg): ffmpeg_sha})
    except BaseException as error:
        seal = dict(stage=STAGE, status="failed", error=repr(error))
        seal.update(
            started_at=started, finished_at=proof.utc(), inputs=evidence.paths, outputs=out.records
        )
        (output / "review.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    seal.update(
        started_at=started, finished_at=proof.utc(), inputs=evidence.paths, outputs=out.records
    )
    (output / "review.provenance.json").write_bytes(proof.json_bytes(seal))
    print(
        json.dumps(dict(output=str(output), frames=len(frames), native_pixel_SHA_checks=1080)),
        flush=True,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "--execute required; no implicit raw decoding")
    return build_pack(args.output)


if __name__ == "__main__":
    main()
