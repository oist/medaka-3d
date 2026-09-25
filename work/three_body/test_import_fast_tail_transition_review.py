"""Synthetic fixtures only, never an actual human answer or original recording."""

import copy
import hashlib
import json
from pathlib import Path

import import_fast_tail_transition_review as importer
import pytest


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def documents():
    manifest = dict(
        experiment="20251225_K_1_2_FCMC",
        frames=list(range(89640, 91300)),
        fps=166,
        image_shape_hw=[1200, 1200],
        raw_storage="original_source_video",
        tracking_performed=False,
        human_truth_created=False,
        binding=dict(
            input_sha256={"video:" + v: str(i + 1) * 64 for i, v in enumerate(importer.VIEWS)}
        ),
        raw={
            v: dict(
                path=v + ".mp4",
                sha256=str(i + 1) * 64,
                source_offset_frames=89640,
                source_total_frames=348600,
                frame_bgr_sha256=["a" * 64] * 1660,
            )
            for i, v in enumerate(importer.VIEWS)
        },
    )
    digest = hashlib.sha256(encoded(manifest)).hexdigest()
    template, frames, _ = importer.creator.ui.bind_template(
        manifest["frames"], digest, 90040, 90400, list(importer.PROPOSALS)
    )
    video = dict(
        decoded_frame_count=360,
        playback_fps=20.0,
        image_shape_hw=[636, 1800],
        all_selected_native_frames_included=True,
    )
    summary = dict(
        schema=importer.creator.STAGE,
        status="ok",
        technical_completion=True,
        review_clip_range=[90040, 90400],
        source_frame_range=[89640, 91300],
        displayed_frames=frames,
        probe_frames=list(importer.PROPOSALS),
        frame_count=360,
        source_fps=166,
        playback_fps=20,
        source_duration_seconds=360 / 166,
        playback_duration_seconds=18.0,
        source_absolute_seconds_range=[90040 / 166, 90400 / 166],
        source_clip_time_rule="(native_frame - 90040) / 166; source time is native_frame / 166",
        absolute_seek_frame=90040,
        first_manifest_sha_offset=400,
        original_source_video_frame_count=348600,
        native_pixel_hashes_verified=1080,
        original_raw_storage="original_source_video",
        source_manifest_sha256=digest,
        raw_video_payload_bytes_copied=0,
        whole_video_rehashes=0,
        model_outputs_shown=False,
        candidate_arrays_deserialized=False,
        mapping_arrays_deserialized=False,
        masks_or_sleap_arrays_deserialized=False,
        human_truth_created=False,
        source_selection_is_model_derived=True,
        gpu_used=False,
        tracking_or_ID_solver_calls=0,
        selection_render_sha256=importer.creator.SELECTION_SHA,
        final_tracking_sha256=importer.creator.ids.TRACKING_SEAL_SHA,
        review_video=video,
        intermediate_video=video.copy(),
        native_bgr_sha256={v: ["a" * 64] * 360 for v in importer.VIEWS},
        consumed_record=dict(index=24, start=89640, stop=91300, manifest_sha256=digest),
        code_sha256={str(Path(importer.creator.__file__).resolve()): importer.CREATOR_SHA},
    )
    proposal = dict(
        schema=importer.creator.STAGE + "_anchor_selection",
        proposed_anchor_frames=list(importer.PROPOSALS),
        review_clip_range=[90040, 90400],
        proposed_not_human_confirmed=True,
        actual_clip_view_suitability_checked=False,
        anchor_adjustments=[],
        anonymous_names=["A", "B", "C"],
        physical_identity_labels_created=False,
    )
    return summary, manifest, template, proposal, digest


def human_fixture(template):
    answer = copy.deepcopy(template)
    answer.update(
        status="human_draft_requires_adjudication",
        reviewer="SYNTHETIC TEST FIXTURE",
        reviewed_utc="2026-09-11T00:00:00Z",
        elapsed_review_seconds=3.0,
        notes="Synthetic fixture only; not actual human evidence.",
    )
    for anchor in answer["anchors"]:
        anchor["frame_choice"] = "human_selected"
    answer["anchors"][0]["frame"] = 90121
    answer["anchors"][0]["animals"]["A"].update(
        visibility="visible", points={"top_cam": [10.25, 20.5]}
    )
    answer["anchors"][1]["animals"]["A"].update(
        visibility="ambiguous", points={"side1_cam": [30, 40]}
    )
    answer["continuity"]["A"].update(entry_to_after="ambiguous", after_to_exit="not_visible")
    answer["biological_naming"]["A"]["sex"] = "female"
    return answer


def test_parser_is_same_function_preserves_changed_proposals_and_all_unknowns():
    _, manifest, template, proposal, _ = documents()
    data = b"\xef\xbb\xbf" + encoded(human_fixture(template))
    parsed = importer.parse_review(data, encoded(manifest), clip_range=[90040, 90400])
    assert importer.parse_review is importer.frozen.parse_review
    assert importer.read_explicit_review is importer.frozen.read_explicit_review
    assert parsed.review_bytes == data and parsed.anchor_frames.tolist() == [90121, 90220, 90380]
    result = importer.describe_review(parsed, proposal)
    assert len(result["endpoints"]) == 9 and len(result["links"]) == 6
    assert result["endpoints"][3]["visibility"] == "ambiguous"
    assert result["endpoints"][3]["original_points"] == {"side1_cam": [30, 40]}
    assert result["endpoints"][8]["visibility"] == "unreviewed"
    assert [x["human_status"] for x in result["links"]] == ["ambiguous", "not_visible"] + [
        "unreviewed"
    ] * 4
    assert result["biological_naming_preserved_not_applied"]["A"]["sex"] == "female"
    assert result["biological_naming_preserved_not_applied"]["B"]["sex"] == "unknown"
    assert result["proposed_to_returned_anchor_changes"][0]["changed"] is True
    assert result["original_anchor_proposal"] == proposal
    assert result["intermediate_anchor_edit_history_inferred"] is False


def test_blank_template_is_not_human_answer():
    _, manifest, template, _, _ = documents()
    with pytest.raises(ValueError, match=r"template|draft"):
        importer.parse_review(encoded(template), encoded(manifest), clip_range=[90040, 90400])


@pytest.mark.parametrize(
    "damage", ["native_count", "fps", "pixels", "source_range", "video_count", "blank", "proposals"]
)
def test_metadata_timebase_denominator_and_blank_template_rejected(damage):
    summary, manifest, template, proposal, digest = documents()
    if damage == "native_count":
        summary["displayed_frames"].pop()
    elif damage == "fps":
        summary["source_fps"] = 20
    elif damage == "pixels":
        summary["native_bgr_sha256"]["top_cam"][3] = "b" * 64
    elif damage == "source_range":
        summary["source_frame_range"] = [90040, 90400]
    elif damage == "video_count":
        summary["review_video"]["decoded_frame_count"] = 359
    elif damage == "blank":
        template["continuity"]["B"]["entry_to_after"] = "confirmed"
    else:
        proposal["anchor_adjustments"].append(dict(synthetic=True))
    with pytest.raises(ValueError):
        importer.validate_documents(summary, manifest, template, proposal, digest)


def test_valid_metadata_360_times_three_source_native_hashes():
    importer.validate_documents(*documents())
    assert len(importer.expected_outputs()) == 1096


def test_canonical_staging_metadata_chain_without_video_files(tmp_path, monkeypatch):
    summary, manifest, _, _, manifest_sha = documents()
    root = tmp_path / "synthetic_source"
    root.mkdir()
    path = root / "manifest.json"
    staging_path = root / "staging.provenance.json"
    tracking_root = tmp_path / "synthetic_tracking"
    tracking_path = tracking_root / "tracking.provenance.json"
    monkeypatch.setattr(importer.creator.ids, "TRACKING_ROOT", tracking_root)
    summary["source_manifest"] = str(path)
    chunk = summary["consumed_record"]
    chunk.update(provenance_sha256="b" * 64, raw_hardlinks={}, output_hashes={})
    chunk["output_hashes"]["manifest"] = dict(path=str(path), sha256=manifest_sha)
    tracking = dict(
        stage=importer.creator.ids.comparison.runner.STAGE,
        status="ok",
        inputs={
            str(path): manifest_sha,
            str(staging_path): "b" * 64,
        },
    )
    staging = dict(
        stage="prepare_independent_mask_chunk",
        status="ok",
        params=dict(binding=manifest["binding"]),
        outputs={},
        inputs={},
    )
    for view in importer.VIEWS:
        canonical = tmp_path.parent / "SYNTHETIC_EXTERNAL_NOT_READ" / (view + ".mp4")
        staged = root / (view + ".mp4")
        sha = manifest["raw"][view]["sha256"]
        chunk["raw_hardlinks"][view] = dict(
            path=str(staged), source_path=str(canonical), sha256=sha
        )
        chunk["output_hashes"][view + "_raw"] = dict(path=str(staged), sha256=sha)
        staging["inputs"]["video:" + view] = dict(path=str(canonical), sha256=sha, size=123)
        tracking["inputs"][str(canonical)] = sha
    staging["outputs"] = copy.deepcopy(chunk["output_hashes"])
    source_docs = {path: manifest, staging_path: staging, tracking_path: tracking}
    monkeypatch.setattr(importer, "read_small", lambda p, e, sha: source_docs[Path(p)])
    pack_seal = dict(
        inputs={
            str(path): manifest_sha,
            str(staging_path): "b" * 64,
            str(tracking_path): importer.creator.ids.TRACKING_SEAL_SHA,
            str(
                importer.creator.SELECTION_ROOT / "render.provenance.json"
            ): importer.creator.SELECTION_SHA,
        }
    )
    bindings = importer.validate_source_chain(
        pack_seal, summary, manifest, importer.proof.Evidence()
    )
    assert len(bindings) == 3 and not any(
        Path(v["canonical_path"]).exists() for v in bindings.values()
    )
    tracking["inputs"].pop(bindings["top_cam"]["canonical_path"])
    with pytest.raises(ValueError, match="input edge"):
        importer.validate_source_chain(pack_seal, summary, manifest, importer.proof.Evidence())


def test_failed_pack_rejected_before_encoded_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "WORK", tmp_path)
    monkeypatch.setattr(importer, "PACK_ROOT", tmp_path / "synthetic_failed_pack")
    monkeypatch.setattr(
        importer, "read_small", lambda *a: dict(stage=importer.creator.STAGE, status="failed")
    )
    monkeypatch.setattr(importer, "output_files", lambda *a: pytest.fail("encoded output read"))
    with pytest.raises(ValueError, match="successful new raw pack"):
        importer.load_pack("a" * 64, importer.proof.Evidence())


def test_all_encoded_membership_size_sha_and_escape_guard(tmp_path):
    root = tmp_path / "synthetic_pack"
    out = importer.proof.Output(root)
    for name in importer.expected_outputs():
        out.write(name, b"SYNTHETIC ENCODED OUTPUT; NEVER DECODED")
    seal = dict(outputs=out.records)
    (root / "review.provenance.json").write_bytes(encoded(seal))
    assert importer.output_files(root, seal, importer.proof.Evidence()) == []
    probe = root / "frames/f90040_top_cam.jpg"
    original = probe.read_bytes()
    probe.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match=r"SHA|hash"):
        importer.output_files(root, seal, importer.proof.Evidence())
    probe.write_bytes(original)
    (root / "human_answer.json").write_text("SYNTHETIC extra answer; never selected or read")
    evidence = importer.proof.Evidence()
    assert importer.output_files(root, seal, evidence) == ["human_answer.json"]
    assert str((root / "human_answer.json").resolve()) not in evidence.paths
    row = seal["outputs"]["summary.json"]
    original_row = row.copy()
    row["size_bytes"] += 1
    with pytest.raises(ValueError, match="path/size"):
        importer.output_files(root, seal, importer.proof.Evidence())
    row.update(original_row)
    row["path"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="path/size"):
        importer.output_files(root, seal, importer.proof.Evidence())
    row.update(original_row)
    seal["outputs"].pop("summary.json")
    with pytest.raises(ValueError, match="roles"):
        importer.output_files(root, seal, importer.proof.Evidence())


def test_disabled_precedes_pack_or_external_answer(tmp_path, monkeypatch):
    assert importer.EXECUTION_READY is False
    monkeypatch.setattr(importer, "load_pack", lambda *a: pytest.fail("pack read"))
    monkeypatch.setattr(importer, "read_explicit_review", lambda *a: pytest.fail("external read"))
    with pytest.raises(ValueError, match="not enabled"):
        importer.run("a" * 64, tmp_path / "answer.json", tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_explicit_synthetic_external_answer_preserved_without_completion_inference(
    tmp_path, monkeypatch
):
    summary, manifest, template, proposal, _ = documents()
    data = b"\xef\xbb\xbf" + encoded(human_fixture(template))
    external = tmp_path / "synthetic_download.json"
    external.write_bytes(data)
    work = tmp_path / "workspace"
    work.mkdir()
    monkeypatch.setattr(importer, "WORK", work)
    monkeypatch.setattr(importer, "EXECUTION_READY", True)
    monkeypatch.setattr(importer, "source_code", lambda: {})
    monkeypatch.setattr(
        importer,
        "load_pack",
        lambda *a: dict(
            summary=summary,
            manifest_bytes=encoded(manifest),
            anchor_selection=proposal,
            video_bindings={},
            unreferenced_pack_files=["synthetic_answer_not_auto_selected.json"],
        ),
    )
    result = work / "synthetic_import_only"
    importer.run("a" * 64, external, result)
    assert (result / "original_review.json").read_bytes() == external.read_bytes() == data
    record = json.loads((result / "import.provenance.json").read_bytes())
    assert record["explicit_review_inputs"] == {
        str(external.resolve()): hashlib.sha256(data).hexdigest()
    }
    saved = json.loads((result / "summary.json").read_bytes())
    assert saved["review_status_preserved"] == "human_draft_requires_adjudication"
    assert (
        saved["labels_filled"]
        is saved["trajectory_modified"]
        is saved["all_five_minute_identity_truth"]
        is False
    )
    assert len(saved["endpoints"]) == 9 and len(saved["links"]) == 6
    assert saved["review_notes_preserved"].startswith("Synthetic fixture only")
    assert saved["biological_naming_preserved_not_applied"]["B"]["sex"] == "unknown"
    assert saved["unreferenced_pack_files_not_used_as_pack_evidence"] == [
        "synthetic_answer_not_auto_selected.json"
    ]


def test_video_path_normalization_has_no_filesystem_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("original-video filesystem access")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    assert importer.metadata_video_path("raw/top.mp4", tmp_path) == tmp_path / "raw/top.mp4"
    assert (
        importer.metadata_video_path(str(tmp_path / "canonical.mp4")) == tmp_path / "canonical.mp4"
    )
    with pytest.raises(ValueError, match="traversal"):
        importer.metadata_video_path("../raw.mp4", tmp_path)
    with pytest.raises(ValueError, match="relative"):
        importer.metadata_video_path(str(tmp_path / "raw.mp4"), tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        importer.metadata_video_path("relative.mp4")
