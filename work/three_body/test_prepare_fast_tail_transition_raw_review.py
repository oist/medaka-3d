"""Synthetic raw-review caller tests; no real input, decoder, browser or GPU."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import prepare_fast_tail_transition_raw_review as pack
import pytest


def manifest():
    return dict(
        frames=list(range(89640, 91300)),
        raw_storage="original_source_video",
        raw={v: dict(frame_bgr_sha256=["a" * 64] * 1660) for v in pack.VIEWS},
    )


def test_disabled_and_missing_execute_never_load_or_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(pack, "EXECUTION_READY", False)
    monkeypatch.setattr(pack, "load_context", lambda *a: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="--execute"):
        pack.main(["--output", str(tmp_path / "not_created")])
    with pytest.raises(ValueError, match="ROOT raw-pack Go"):
        pack.main(["--execute", "--output", str(tmp_path / "not_created")])
    assert not (tmp_path / "not_created").exists()


def test_template_keeps_full_chunk_binding_but_displays_all360_frames():
    template, frames, offset, html = pack.review_documents(manifest(), "a" * 64)
    assert template["source_frame_range"] == [89640, 91300]
    assert template["review_clip_range"] == [90040, 90400]
    assert frames == list(range(90040, 90400)) and offset == 400
    assert [a["frame"] for a in template["anchors"]] == [90120, 90220, 90380]
    assert all(a["frame_choice"] == "proposed_not_human_confirmed" for a in template["anchors"])
    assert template["status"] == "unreviewed_template_not_ground_truth"
    assert template["model_outputs_shown"] is False
    assert template["reviewer"] is template["reviewed_utc"] is None
    for name in ("A", "B", "C"):
        assert template["biological_naming"][name] == dict(
            sex="unknown", physical_name=None, evidence=None
        )
        assert template["continuity"][name]["entry_to_after"] == "unreviewed"
        for anchor in template["anchors"]:
            assert anchor["animals"][name] == dict(points={}, visibility="unreviewed")
    assert all(token not in html for token in ("__TRUTH__", "__FRAMES__", "__LAST__", "__VIEWS__"))
    assert 'max="359"' in html


def test_native_time_and_slow_video_are_not_confused():
    _, _, _, html = pack.review_documents(manifest(), "a" * 64)
    assert html.count("${(frame/166).toFixed(3)} s") == 2
    assert html.count("${((frame-frames[0])/166).toFixed(3)} s") == 2
    assert "2.168675 秒" in html and "20 fps の動画では18秒" in html
    assert "元映像 166 fps、約 8.3 倍のスロー" in html
    assert "${(frame/20)" not in html
    assert "currentAnimal().points[view]=" in html  # Existing native mouth-click UI retained.
    assert "['ambiguous','曖昧']" in html and "['not_visible','見えない']" in html


def selection_headers():
    chunk = dict(index=24, start=89640, stop=91300)
    seal = dict(
        status="ok",
        stage=pack.source.STAGE,
        inputs={str(Path(pack.source.__file__).resolve()): pack.SOURCE_SHA},
    )
    selection = dict(
        schema=pack.source.STAGE,
        status="ok",
        frames=list(pack.source.FRAMES),
        terminal_horizon=116200,
        source_chunk=copy.deepcopy(chunk),
        native_pixel_SHA_checked=True,
    )
    return seal, selection, chunk


def test_successful_selection_binding():
    pack.validate_selection(*selection_headers())


@pytest.mark.parametrize("change", ["failed", "stage", "source", "chunk", "horizon", "pixels"])
def test_selection_mixups_rejected(change):
    seal, selected, chunk = selection_headers()
    if change == "failed":
        seal["status"] = "failed"
    elif change == "stage":
        seal["stage"] = pack.raw.STAGE
    elif change == "source":
        seal["inputs"] = {}
    elif change == "chunk":
        selected["source_chunk"]["index"] = 23
    elif change == "horizon":
        selected["terminal_horizon"] = 86320
    else:
        selected["native_pixel_SHA_checked"] = False
    with pytest.raises(ValueError):
        pack.validate_selection(seal, selected, chunk)


def test_encoded_output_registration_rejects_escape_and_changed_bytes(tmp_path):
    out = pack.proof.Output(tmp_path / "new")
    (out.root / "frame.jpg").write_bytes(b"synthetic pixels")
    with pytest.raises(ValueError, match="registration"):
        pack.register_output(out, "frame.jpg", "b" * 64)
    with pytest.raises(ValueError):
        pack.register_output(out, "../outside", "b" * 64)
    digest = pack.raw.sha256(out.root / "frame.jpg")
    pack.register_output(out, "frame.jpg", digest)
    with pytest.raises(ValueError, match="registration"):
        pack.register_output(out, "frame.jpg", digest)
    out.final_guard()


@pytest.fixture
def build_fixture(tmp_path, monkeypatch):
    # Synthetic small bytes only. Real frozen frame/pixel completeness validation
    # is exercised without a codec, a recording or a model payload.
    doc = manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf8")
    digest = pack.raw.sha256(path)
    context = SimpleNamespace(
        manifest=doc,
        manifest_path=path,
        videos={},
        chunk=dict(index=24, start=89640, stop=91300, manifest_sha256=digest),
        scene=dict(
            context_start_frame=pack.START,
            context_stop_frame_exclusive=pack.STOP,
            proposed_anchor_frames=list(pack.ANCHORS),
        ),
    )
    ffmpeg = tmp_path / "synthetic_ffmpeg.exe"
    ffmpeg.write_bytes(b"not executed")
    monkeypatch.setattr(pack, "EXECUTION_READY", True)
    monkeypatch.setattr(pack, "source_code", lambda: {})
    monkeypatch.setattr(pack, "load_context", lambda evidence: context)
    monkeypatch.setattr(pack.shutil, "which", lambda _: str(ffmpeg))
    monkeypatch.setattr(
        pack.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="synthetic encoder\n")
    )
    monkeypatch.setattr(pack.previous, "video_stat_guard", lambda *a: None)
    monkeypatch.setattr(
        pack.ui, "transcode", lambda ffmpeg, source, target: target.write_bytes(b"synthetic H264")
    )
    monkeypatch.setattr(
        pack.ui,
        "verify_video",
        lambda path, count: dict(decoded_frame_count=count, playback_fps=20),
    )

    def images(ctx, output):
        hashes = {}
        for view in pack.VIEWS:
            for native in range(pack.START, pack.STOP):
                p = output / "frames" / f"f{native}_{view}.jpg"
                p.write_bytes(b"synthetic JPEG")
                hashes[p.relative_to(output).as_posix()] = pack.raw.sha256(p)
            for native in pack.ANCHORS:
                p = output / f"f{native}_{view}.png"
                p.write_bytes(b"synthetic PNG")
                hashes[p.name] = pack.raw.sha256(p)
        intermediate = output / "raw_three_views_mpeg4_intermediate.mp4"
        intermediate.write_bytes(b"synthetic MPEG4")
        return intermediate, hashes, {v: ["a" * 64] * 360 for v in pack.VIEWS}

    monkeypatch.setattr(pack.raw, "write_images", images)
    return tmp_path / "review_pack", context


def test_successful_synthetic_pack_is_blank_bound_complete_and_not_truth(build_fixture):
    output, context = build_fixture
    result = pack.build_pack(output)
    seal = json.loads((output / "review.provenance.json").read_text())
    template = json.loads((output / "truth_template.json").read_text())
    anchors = json.loads((output / "anchor_selection.json").read_text())
    assert seal["status"] == "ok" and len(seal["outputs"]) == 1096
    assert result["native_pixel_hashes_verified"] == 1080
    assert result["source_duration_seconds"] == 360 / 166
    assert result["playback_duration_seconds"] == 18
    assert (
        result["candidate_arrays_deserialized"]
        is result["masks_or_sleap_arrays_deserialized"]
        is False
    )
    assert result["model_outputs_shown"] is result["human_truth_created"] is False
    assert template["status"] == "unreviewed_template_not_ground_truth"
    assert (
        anchors["anchor_adjustments"] == []
        and anchors["actual_clip_view_suitability_checked"] is False
    )
    assert (output / "source_manifest.json").read_bytes() == context.manifest_path.read_bytes()
    assert all(pack.raw.sha256(Path(r["path"])) == r["sha256"] for r in seal["outputs"].values())


def test_native_decode_failure_cannot_publish_complete_or_filled_review(build_fixture, monkeypatch):
    output, _ = build_fixture

    def fail(*args):
        raise ValueError("synthetic native SHA failure")

    monkeypatch.setattr(pack.raw, "write_images", fail)
    with pytest.raises(ValueError, match="native SHA failure"):
        pack.build_pack(output)
    seal = json.loads((output / "review.provenance.json").read_text())
    assert seal["status"] == "failed"
    assert not (output / "summary.json").exists() and not (output / "truth_template.json").exists()
    assert not (output / "index.html").exists()
