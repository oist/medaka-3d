"""No actual video or tracker: source offsets, ownership, seals and UI scope."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_long_raw_review as module


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def record_output(path):
    return dict(path=str(path), sha256=module.sha256(path), size=path.stat().st_size)


@pytest.fixture
def pixels():
    bgr = np.zeros((1200, 1200, 3), np.uint8)
    bgr[9, 13] = [1, 7, 21]
    return bgr


@pytest.fixture
def native(pixels):
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    first = module.FIRST
    frames = list(range(first, first + module.CHUNK))
    return dict(
        frames=frames,
        raw={view: {"frame_bgr_sha256": [digest] * module.CHUNK} for view in module.VIEWS},
    )


class Capture:
    def __init__(self, pixels, *, problem=None):
        self.pixels, self.problem = pixels, problem
        self.position, self.reads, self.released, self.seeks = 0, 0, False, []

    def isOpened(self):
        return self.problem != "open"

    def get(self, key):
        cv = module.cv2
        if key == cv.CAP_PROP_FRAME_COUNT:
            return module.CHUNK if self.problem == "clip_count" else module.STOP
        if key == cv.CAP_PROP_FPS:
            return 20 if self.problem == "fps" else 166
        if key in (cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FRAME_WIDTH):
            return 1200
        if key == cv.CAP_PROP_POS_FRAMES:
            return self.position + (1 if self.problem == "position" else 0)
        raise AssertionError(key)

    def set(self, key, value):
        assert key == module.cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        self.position = value
        return self.problem != "seek"

    def read(self):
        self.reads += 1
        self.position += 1
        if self.problem == "early_eof":
            return False, None
        if self.problem == "hash":
            return True, np.zeros_like(self.pixels)
        return True, self.pixels

    def release(self):
        self.released = True


def factories(pixels, *, problem=None, problem_view=1):
    captures = [Capture(pixels, problem=problem if i == problem_view else None) for i in range(3)]
    videos = dict(zip(module.VIEWS, [str(i) for i in range(3)], strict=True))
    return captures, videos, lambda path: captures[int(path)]


def test_absolute_seek_and_manifest_relative_sha_without_extra_decode(native, pixels):
    start, stop = module.FIRST + 7, module.FIRST + 10
    # The preceding chunk hashes differ: using frame or frame-start as the index fails.
    for info in native["raw"].values():
        info["frame_bgr_sha256"][:7] = ["0" * 64] * 7
    caps, videos, factory = factories(pixels)
    rows = list(module.iter_native(videos, native, start, stop, capture_factory=factory))
    assert [f for f, _ in rows] == list(range(start, stop))
    assert all(cap.seeks == [start] and cap.reads == 3 and cap.released for cap in caps)


@pytest.mark.parametrize(
    "problem", ["open", "clip_count", "fps", "seek", "position", "early_eof", "hash"]
)
def test_bad_source_fails_and_releases_every_opened_camera(native, pixels, problem):
    caps, videos, factory = factories(pixels, problem=problem)
    with pytest.raises(ValueError):
        list(
            module.iter_native(
                videos, native, module.FIRST + 2, module.FIRST + 5, capture_factory=factory
            )
        )
    assert caps[0].released and caps[1].released
    if problem in ("early_eof", "hash"):
        assert caps[2].released
    else:
        assert not caps[2].seeks  # Third camera was never opened.


def test_early_close_reads_one_frame_and_releases_three(native, pixels):
    caps, videos, factory = factories(pixels)
    stream = module.iter_native(
        videos, native, module.FIRST, module.FIRST + 3, capture_factory=factory
    )
    next(stream)
    stream.close()
    assert all(cap.reads == 1 and cap.released for cap in caps)


def test_factory_failure_after_one_camera_releases_it(native, pixels):
    cap = Capture(pixels)

    def factory(path):
        if path == "1":
            raise OSError("open creation failed")
        return cap

    with pytest.raises(OSError):
        list(
            module.iter_native(
                dict(zip(module.VIEWS, ["0", "1", "2"], strict=True)),
                native,
                module.FIRST,
                module.FIRST + 3,
                capture_factory=factory,
            )
        )
    assert cap.released and not cap.reads


@pytest.mark.parametrize("lo,hi", [(49799, 49803), (49800, 51461), (49802, 49802)])
def test_outside_or_empty_scene_never_opens_source(native, lo, hi):
    with pytest.raises(ValueError):
        list(
            module.iter_native({}, native, lo, hi, capture_factory=lambda _: pytest.fail("no open"))
        )


@pytest.fixture
def inputs(tmp_path, monkeypatch, native):
    monkeypatch.setattr(module, "REPO", tmp_path)
    run, inventory, stage = tmp_path / "run", tmp_path / "inventory", tmp_path / "stage"
    run.mkdir()
    inventory.mkdir()
    chunk = stage / "chunks/f049800-051460/attempt_0001"
    chunk.mkdir(parents=True)
    first, stop = module.FIRST, module.FIRST + module.CHUNK
    detector = dict(input_sha256={})
    raw = {}
    chunk_outputs = {}
    for view in module.VIEWS:
        path = chunk / "raw" / (view + ".mp4")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("fake pixels " + view).encode())
        digest = module.sha256(path)
        detector["input_sha256"]["video:" + view] = digest
        raw[view] = dict(
            path=path.relative_to(chunk).as_posix(),
            sha256=digest,
            source_offset_frames=first,
            source_total_frames=module.STOP,
            frame_bgr_sha256=native["raw"][view]["frame_bgr_sha256"],
        )
        chunk_outputs[view + "_raw"] = record_output(path)
    manifest = dict(
        experiment="20251225_K_1_2_FCMC",
        raw_storage="original_source_video",
        fps=166,
        image_shape_hw=[1200, 1200],
        tracking_performed=False,
        human_truth_created=False,
        raw=raw,
        frames=native["frames"],
        binding=detector,
        protocol_sha256="1" * 64,
    )
    write(chunk / "manifest.json", manifest)
    chunk_outputs["manifest"] = record_output(chunk / "manifest.json")
    write(
        chunk / "staging.provenance.json",
        dict(
            status="ok",
            stage="prepare_independent_mask_chunk",
            outputs=chunk_outputs,
            params={"binding": detector},
        ),
    )
    consumed = [
        dict(
            index=0,
            start=first,
            stop=stop,
            attempt_path=chunk.relative_to(stage).as_posix(),
            manifest_sha256=module.sha256(chunk / "manifest.json"),
            provenance_sha256=module.sha256(chunk / "staging.provenance.json"),
            output_hashes=chunk_outputs,
        )
    ]
    summary = dict(
        schema="long_continuation_phase_v1",
        technical_completion=True,
        processed_interval=[first, stop],
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
    )
    write(run / "summary.json", summary)
    write(run / "consumed_chunks.json", consumed)
    (run / "candidate.npz").write_bytes(b"intentionally not a NumPy archive; hashing only")
    outputs = {
        role: record_output(run / name)
        for role, name in (
            ("summary", "summary.json"),
            ("candidate", "candidate.npz"),
            ("consumed_chunks", "consumed_chunks.json"),
        )
    }
    producer = dict(
        status="ok",
        stage="three_body_long_continuation",
        outputs=outputs,
        params=dict(
            all_input_source_snapshot_output_sha_unchanged=True,
            through_frame=stop,
            bindings={
                "inputs": dict(
                    input_stage_root=str(stage), detector_binding=detector, protocol_sha256="1" * 64
                )
            },
        ),
    )
    write(run / "continuation.provenance.json", producer)
    scene = dict(
        scene_id=0,
        status="proposed_for_human_review",
        selection_issues=[],
        context_start_frame=first + 7,
        context_stop_frame_exclusive=first + 11,
        context_frames=4,
        proposed_anchor_frames=[first + 7, first + 9, first + 10],
    )
    inventory_summary = dict(
        schema="long_backtrace_review_inventory_v1",
        processed_interval=[first, stop],
        native_frames=module.CHUNK,
        candidate_sha256=outputs["candidate"]["sha256"],
        producer_seal_sha256=module.sha256(run / "continuation.provenance.json"),
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        risk_counts={},
    )
    write(inventory / "summary.json", inventory_summary)
    write(
        inventory / "review_proposal.json",
        dict(schema="three_body_raw_review_proposal_v1", risk_counts={}, scenes=[scene]),
    )
    inventory_record = dict(
        status="ok",
        stage=module.INVENTORY_STAGE,
        params={"all_bound_input_source_output_sha_unchanged": True},
        outputs={
            name: record_output(inventory / (name + ".json"))
            for name in ("summary", "review_proposal")
        },
        inputs={
            str(run / name): record_output(run / name)
            for name in (
                "summary.json",
                "candidate.npz",
                "consumed_chunks.json",
                "continuation.provenance.json",
            )
        },
    )
    write(inventory / "inventory.provenance.json", inventory_record)
    return run, inventory, chunk, manifest


def test_prepare_uses_real_source_manifest_without_opening_candidate_or_video(inputs, monkeypatch):
    run, inventory, chunk, manifest = inputs
    monkeypatch.setattr(
        module.np, "load", lambda *a, **k: pytest.fail("candidate/masks must not open")
    )
    monkeypatch.setattr(
        module.cv2, "VideoCapture", lambda *a: pytest.fail("no decode during prepare")
    )
    context = module.prepare_inputs(run, inventory, 0)
    assert context.manifest == manifest and context.chunk_root == chunk
    item = context.scene
    template, frames, offset = module.scene_ui.bind_template(
        manifest["frames"],
        module.sha256(chunk / "manifest.json"),
        item["context_start_frame"],
        item["context_stop_frame_exclusive"],
        item["proposed_anchor_frames"],
    )
    assert template["source_frame_range"] == [module.FIRST, module.FIRST + module.CHUNK]
    assert template["review_clip_range"] == [module.FIRST + 7, module.FIRST + 11] and offset == 7
    assert template["status"] == "unreviewed_template_not_ground_truth"
    assert len(frames) == 4 and context.producer_summary["prefix_backtrace_provisional"]


@pytest.mark.parametrize(
    "problem",
    ["failed_inventory", "failed_producer", "foreign_inventory", "changed_raw", "missing_probe"],
)
def test_bad_evidence_or_selection_fails_before_decode(inputs, monkeypatch, problem):
    run, inventory, chunk, _ = inputs
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda *a: pytest.fail("no decode"))
    if problem in ("failed_inventory", "foreign_inventory"):
        path = inventory / "inventory.provenance.json"
        data = module.read(path)
        if problem == "failed_inventory":
            data["status"] = "failed"
        else:
            data["inputs"][str(run / "candidate.npz")]["sha256"] = "0" * 64
        write(path, data)
    elif problem == "failed_producer":
        path = run / "continuation.provenance.json"
        data = module.read(path)
        data["status"] = "failed"
        write(path, data)
    elif problem == "changed_raw":
        (chunk / "raw/top_cam.mp4").write_bytes(b"changed")
    else:
        path = inventory / "review_proposal.json"
        data = module.read(path)
        data["scenes"][0]["proposed_anchor_frames"][1] = None
        write(path, data)
        seal_path = inventory / "inventory.provenance.json"
        seal = module.read(seal_path)
        seal["outputs"]["review_proposal"] = record_output(path)
        write(seal_path, seal)
    with pytest.raises((ValueError, TypeError)):
        module.prepare_inputs(run, inventory, 0)


@pytest.mark.parametrize("problem", ["source_offset", "source_count", "storage", "frame_hole"])
def test_manifest_storage_and_frame_contract_refuses_mixing(inputs, problem):
    _, _, _, manifest = inputs
    if problem == "storage":
        manifest["raw_storage"] = "FFV1"
    elif problem == "frame_hole":
        manifest["frames"][1] += 1
    else:
        key = "source_offset_frames" if problem == "source_offset" else "source_total_frames"
        manifest["raw"][module.VIEWS[0]][key] = 0
    with pytest.raises(ValueError):
        module.validate_source(
            manifest,
            dict(start=module.FIRST, stop=module.FIRST + module.CHUNK),
            manifest["binding"],
        )


def test_cross_chunk_scene_is_explicitly_unsupported():
    first = module.FIRST
    chunks = [
        dict(index=i, start=first + i * module.CHUNK, stop=first + (i + 1) * module.CHUNK)
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="inside one sealed chunk"):
        module.select_chunk(
            chunks,
            first,
            first + 2 * module.CHUNK,
            first + module.CHUNK - 1,
            first + module.CHUNK + 2,
        )


def test_cli_without_execute_never_prepares_or_decodes(monkeypatch):
    monkeypatch.setattr(module, "build_pack", lambda *a, **k: pytest.fail("must not build"))
    with pytest.raises(ValueError, match="--execute"):
        module.main(["--run", "unused", "--inventory", "unused", "--output", "unused"])


@pytest.mark.parametrize("problem", ["native_count", "native_sha", "missing_jpeg", "missing_png"])
def test_partial_render_cannot_claim_static_or_native_qa(native, problem):
    first = module.FIRST
    frames, probes = list(range(first + 1, first + 5)), [first + 1, first + 2, first + 4]
    values = {view: native["raw"][view]["frame_bgr_sha256"][1:5].copy() for view in module.VIEWS}
    hashes = {f"frames/f{frame}_{view}.jpg": "unused" for frame in frames for view in module.VIEWS}
    hashes.update({f"f{frame}_{view}.png": "unused" for frame in probes for view in module.VIEWS})
    if problem == "native_count":
        values[module.VIEWS[0]].pop()
    elif problem == "native_sha":
        values[module.VIEWS[0]][0] = "0" * 64
    elif problem == "missing_jpeg":
        del hashes[f"frames/f{frames[0]}_{module.VIEWS[0]}.jpg"]
    else:
        del hashes[f"f{probes[0]}_{module.VIEWS[0]}.png"]
    with pytest.raises(ValueError):
        module.validate_written_scene(native, frames, probes, hashes, values)


@pytest.fixture
def build_environment(inputs, monkeypatch):
    run, inventory, _, _ = inputs
    context = module.prepare_inputs(run, inventory, 0)
    original_lab = module.LAB
    test_lab = run.parent / "lab"
    test_lab.mkdir()
    for name in ("review_scene.py", "review_truth.py", "review_truth.html", "review_scene.html"):
        (test_lab / name).write_bytes((original_lab / name).read_bytes())
    driver = run.parent / "driver.py"
    driver.write_text("fake source")
    provenance_path = run.parent / "src/medaka3d/provenance.py"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text("fake snapshot source")
    executable = run.parent / "ffmpeg.exe"
    executable.write_bytes(b"not executed")
    monkeypatch.setattr(module, "LAB", test_lab)
    monkeypatch.setattr(module, "__file__", str(driver))
    monkeypatch.setattr(module, "prepare_inputs", lambda *a: context)
    monkeypatch.setattr(module.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="fixture ffmpeg", returncode=0),
    )
    return run, inventory, context


def test_failed_image_creation_keeps_failed_seal_and_no_success_summary(
    build_environment, monkeypatch
):
    run, inventory, _ = build_environment

    def fail(*args):
        raise ValueError("fixture native SHA mismatch")

    monkeypatch.setattr(module, "write_images", fail)
    output = run.parent / "new_review"
    with pytest.raises(ValueError, match="native SHA mismatch"):
        module.build_pack(run, inventory, 0, output)
    assert module.read(output / "review.provenance.json")["status"] == "failed"
    assert not (output / "summary.json").exists()
    assert not (output / "truth_template.json").exists()


def test_successful_pack_runs_through_real_provenance_output_path_guard(
    build_environment, monkeypatch
):
    run, inventory, context = build_environment

    def images(context, output):
        item = context.scene
        first, stop = item["context_start_frame"], item["context_stop_frame_exclusive"]
        hashes = {}
        for view in module.VIEWS:
            for frame in range(first, stop):
                path = output / "frames" / f"f{frame}_{view}.jpg"
                path.write_bytes(b"fixture image encoder")
                hashes[path.relative_to(output).as_posix()] = module.sha256(path)
            for frame in item["proposed_anchor_frames"]:
                path = output / f"f{frame}_{view}.png"
                path.write_bytes(b"fixture probe encoder")
                hashes[path.name] = module.sha256(path)
        offset = first - context.manifest["frames"][0]
        native = {
            view: context.manifest["raw"][view]["frame_bgr_sha256"][offset : offset + stop - first]
            for view in module.VIEWS
        }
        path = output / "raw_three_views_mpeg4_intermediate.mp4"
        path.write_bytes(b"fixture video encoder")
        return path, hashes, native

    monkeypatch.setattr(module, "write_images", images)
    monkeypatch.setattr(
        module.scene_ui,
        "transcode",
        lambda ffmpeg, source, target: target.write_bytes(source.read_bytes()),
    )
    monkeypatch.setattr(
        module.scene_ui, "verify_video", lambda path, count: dict(decoded_frame_count=count)
    )
    output = run.parent / "successful_review"
    result = module.build_pack(run, inventory, 0, output)
    seal = module.read(output / "review.provenance.json")
    assert (
        seal["status"] == "ok" and seal["params"]["all_input_source_output_sha_unchanged"] is True
    )
    assert all(
        isinstance(item["path"], str) and module.sha256(Path(item["path"])) == item["sha256"]
        for item in seal["outputs"].values()
    )
    assert result["native_pixel_hashes_verified"] == 12
    assert module.read(output / "source_manifest.json") == context.manifest
    assert (
        module.read(output / "truth_template.json")["status"]
        == "unreviewed_template_not_ground_truth"
    )
