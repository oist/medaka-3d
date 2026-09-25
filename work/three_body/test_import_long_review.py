"""Synthetic review exports remain test fixtures, never actual human answers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import import_long_review as module  # noqa: E402
import review_truth  # noqa: E402
import run_long_remaining_supervisor as old  # noqa: E402


def encode(value):
    return json.dumps(value, allow_nan=False).encode()


def base_document():
    manifest = dict(experiment="20251225_K_1_2_FCMC", frames=list(range(64740, 66400)),
                    fps=166, image_shape_hw=[1200, 1200])
    raw = encode(manifest)
    review = review_truth.make_template(manifest["frames"], hashlib.sha256(raw).hexdigest(),
                                        [65931, 66151, 66217])
    review.update(status="human_draft_requires_adjudication", reviewer="synthetic test",
                  reviewed_utc="2026-09-09T12:00:00Z", elapsed_review_seconds=2.5,
                  review_clip_range=[65859, 66223])
    for anchor in review["anchors"]:
        anchor["frame_choice"] = "human_selected"
        for f, mark in enumerate(anchor["animals"].values()):
            mark.update(visibility="visible", points={v: [30. + 30*f, 100.] for v in module.VIEWS})
    for state in review["continuity"].values():
        state.update(entry_to_after="confirmed", after_to_exit="confirmed",
                     basis="direct_visual_continuity")
    return review, manifest


def parsed(review=None, manifest=None):
    if review is None:
        review, manifest = base_document()
    return module.parse_review(encode(review), encode(manifest), clip_range=[65859, 66223])


def test_actual_draft_schema_accepts_changed_native_probes_and_preserves_bytes():
    review, manifest = base_document()
    for anchor, frame in zip(review["anchors"], [65930, 66000, 66222], strict=True):
        anchor["frame"] = frame
    data = b"\xef\xbb\xbf" + encode(review)
    value = module.parse_review(data, encode(manifest), clip_range=[65859, 66223])
    assert value.review_bytes == data and value.anchor_frames.tolist() == [65930, 66000, 66222]
    assert value.review["status"] == "human_draft_requires_adjudication"
    assert value.continuity.tolist() == [["confirmed", "confirmed"]] * 3


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(status="unreviewed_template_not_ground_truth"), "draft export"),
    (lambda r: r.update(source_manifest_sha256="a" * 64), "checksum"),
    (lambda r: r.update(review_clip_range=[65860, 66223]), "different displayed clip"),
    (lambda r: r["anchors"][0].update(frame=65858), "outside displayed clip"),
    (lambda r: r["anchors"][0].update(frame=True), "outside displayed clip"),
    (lambda r: r["anchors"][1].update(frame=65931), "chronological"),
    (lambda r: r["anchors"][0].update(frame_choice="proposed_not_human_confirmed"),
     "human-selected"),
    (lambda r: r["anchors"][0]["animals"].update(D={}), "local A/B/C"),
    (lambda r: r["anchors"][0]["animals"]["A"]["points"].update(top_cam=[True, 10]), "coordinates"),
    (lambda r: r["anchors"][0]["animals"]["A"]["points"].update(top_cam=[1200, 10]), "coordinates"),
    (lambda r: r.update(model_outputs_shown=True), "model-blind"),
    (lambda r: r.update(elapsed_review_seconds=True), "elapsed"),
    (lambda r: r.update(reviewed_utc="2026-09-09T12:00:00"), "timezone"),
    (lambda r: r.update(reviewer="  "), "reviewer"),
])
def test_source_scope_and_malformed_export_rejected(change, match):
    review, manifest = base_document()
    change(review)
    with pytest.raises(ValueError, match=match):
        parsed(review, manifest)


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'])
def test_duplicate_or_nonfinite_json_rejected(data):
    with pytest.raises(ValueError, match=r"duplicate|nonfinite"):
        module.document(data, "fixture")


def test_unreviewed_and_ambiguous_kept_without_marks_or_false_confirmation():
    review, manifest = base_document()
    for anchor in review["anchors"]:
        anchor["frame_choice"] = "proposed_not_human_confirmed"
        for mark in anchor["animals"].values():
            mark.update(visibility="unreviewed", points={})
    for state in review["continuity"].values():
        state.update(entry_to_after="ambiguous", after_to_exit="not_visible", basis="unreviewed")
    value = parsed(review, manifest)
    assert np.isnan(value.human_points).all()
    assert value.continuity.tolist() == [["ambiguous", "not_visible"]] * 3


class Camera:
    def project(self, xyz):
        return np.asarray(xyz)[..., :2].copy()

    def rays(self, uv):
        uv = np.asarray(uv)
        origins = np.concatenate((uv, np.zeros((*uv.shape[:-1], 1))), axis=-1)
        directions = np.zeros_like(origins)
        directions[..., 2] = 1
        return origins, directions


@pytest.fixture
def geometry():
    doc = parsed()
    pose = np.zeros((3, 3, 3, 3), np.float64)
    for fish in range(3):
        pose[:, fish, :, 0] = 30. + 30*fish
        pose[:, fish, :, 1] = 100.
        pose[:, fish, :, 2] = 20.
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], pose.shape).copy().astype(np.int64)
    arrays = dict(frame_idx=doc.anchor_frames.copy(), pose_xyz=pose, sources=sources,
                  observed=np.ones((3, 3, 3), bool), identity_review=np.zeros((3, 3), bool))
    points = {int(frame): {view: pose[t, :, :, :2].copy() for view in module.VIEWS}
              for t, frame in enumerate(doc.anchor_frames)}
    return SimpleNamespace(doc=doc, arrays=arrays, points=points,
                           rig=SimpleNamespace(camera=lambda view: Camera()))


def match(f):
    return module.match_anchor_rows(f.doc, f.arrays, f.rig, f.points,
                                    candidate_sha256="c" * 64,
                                    calibration_sha256={"fixture-calibration": "d" * 64},
                                    projection_origin_sha256="e" * 64)


def test_matching_keeps_all_nine_endpoints_six_links_and_original_review(geometry):
    f = geometry
    f.arrays["identity_review"][1, 0] = True
    saved = {k: v.tobytes() for k, v in f.arrays.items()}
    result = match(f)
    assert len(result["endpoints"]) == 9 and len(result["links"]) == 6
    assert [x["raw_match"]["slot"] for x in result["endpoints"]] == [0, 1, 2] * 3
    assert result["endpoints"][3]["accepted_match"]["slot"] is None
    assert all(x["status"] == "staged_endpoint_link" for x in result["links"])
    assert not result["production_correction_ready"]
    assert not result["final_candidate_files_verified_here"]
    assert {k: v.tobytes() for k, v in f.arrays.items()} == saved
    json.dumps(result, allow_nan=False)


def test_final_slot_shuffle_rematches_current_coordinates_not_old_slots(geometry):
    f = geometry
    for key in ("pose_xyz", "sources", "observed", "identity_review"):
        f.arrays[key] = f.arrays[key][:, [2, 0, 1]].copy()
    result = match(f)
    assert [x["raw_match"]["slot"] for x in result["endpoints"]] == [1, 2, 0] * 3


def test_mutated_parsed_points_cannot_retain_original_review_sha(geometry):
    geometry.doc.human_points[0, 0, 0, 0] += 1.
    with pytest.raises(ValueError, match="original bytes"):
        match(geometry)


def test_unobserved_competitor_stays_in_spatial_matching(geometry):
    f = geometry
    f.arrays["pose_xyz"][:, 1, 0] = f.arrays["pose_xyz"][:, 0, 0]
    f.arrays["sources"][:, 1, :, 0] = -1
    f.arrays["observed"][:, 1, 0] = False
    result = match(f)
    assert result["endpoints"][0]["spatial_match"]["eligible_slots"] == [0, 1]
    assert result["endpoints"][0]["raw_match"]["slot"] is None
    assert result["endpoints"][2]["raw_match"]["slot"] == 2


def test_no_elimination_for_unmarked_third_animal(geometry):
    f = geometry
    review, manifest = base_document()
    for anchor in review["anchors"]:
        anchor["animals"]["C"] = dict(visibility="occluded", points={})
    review["continuity"]["C"].update(entry_to_after="not_visible", after_to_exit="not_visible")
    review["biological_naming"]["C"]["sex"] = "female"
    f.doc = parsed(review, manifest)
    result = match(f)
    assert all(result["endpoints"][i]["raw_match"]["slot"] is None for i in (2, 5, 8))
    assert result["biological_naming_preserved_not_applied"]["C"]["sex"] == "female"


@pytest.mark.parametrize("alter,match_text", [
    (lambda f: f.arrays["sources"].__setitem__((0, 1, 0, 0), 0), "exclusive"),
    (lambda f: f.arrays["sources"].__setitem__((0, 1, 0, 0), 99), "unavailable"),
    (lambda f: f.arrays["observed"].__setitem__((0, 0, 0), False), "saved source/observed"),
    (lambda f: f.arrays["pose_xyz"].__setitem__((0, 0, 0, 0), 32.01), "saved source/observed"),
    (lambda f: f.arrays["frame_idx"].__setitem__(0, 65932), "native rows"),
])
def test_native_source_gate_mismatch_refuses_staged_mapping(geometry, alter, match_text):
    alter(geometry)
    with pytest.raises(ValueError, match=match_text):
        match(geometry)


def test_unique_but_one_view_supported_is_unknown(geometry):
    f = geometry
    f.arrays["sources"][:, 0, 1:, 0] = -1
    f.arrays["observed"][:, 0, 0] = False
    result = match(f)
    endpoint = result["endpoints"][0]
    assert endpoint["spatial_match"]["slot"] == 0
    assert endpoint["raw_match"]["slot"] is None
    assert "lacks declared measured support" in endpoint["raw_match"]["reason"]


@pytest.mark.parametrize("offset,slot", [(5., 0), (5.00001, None)])
def test_five_pixel_match_boundary_is_distinct_from_native_support(geometry, offset, slot):
    f = geometry
    f.arrays["pose_xyz"][:, 0, 0, 0] += offset
    for t, frame in enumerate(f.doc.anchor_frames):
        for view in module.VIEWS:
            f.points[int(frame)][view][0, 0] = f.arrays["pose_xyz"][t, 0, 0, :2]
    result = match(f)
    assert result["endpoints"][0]["raw_match"]["slot"] == slot
    assert result["support"][0]["mouth_direct_view_count"][0] == 3


def test_reprojection_gate_is_rechecked_even_when_ray_distance_is_zero(geometry):
    class InconsistentCamera(Camera):
        def project(self, xyz):
            return super().project(xyz) + np.array([13., 0.])
    geometry.rig = SimpleNamespace(camera=lambda view: InconsistentCamera())
    with pytest.raises(ValueError, match="saved source/observed"):
        match(geometry)


def test_two_human_marks_compete_for_one_slot_without_greedy_assignment(geometry):
    f = geometry
    review, manifest = base_document()
    for anchor in review["anchors"]:
        for view in module.VIEWS:
            anchor["animals"]["B"]["points"][view] = [30., 100.]
    f.doc = parsed(review, manifest)
    result = match(f)
    assert result["endpoints"][0]["raw_match"]["slot"] is None
    assert result["endpoints"][1]["raw_match"]["slot"] is None
    assert result["endpoints"][2]["raw_match"]["slot"] == 2


def file_entry(path):
    return dict(path=str(path), size=path.stat().st_size, sha256=module.sha256(path))


@pytest.fixture
def pack(tmp_path):
    root, chunk = tmp_path / "pack", tmp_path / "chunk"
    root.mkdir()
    chunk.mkdir()
    review, manifest = base_document()
    binding = {"input_sha256": {"video:" + v: str(i) * 64 for i, v in enumerate(module.VIEWS)}}
    manifest.update(raw_storage="original_source_video", tracking_performed=False,
                    human_truth_created=False, binding=binding,
                    raw={v: dict(path=v + ".mp4", sha256=str(i) * 64, source_offset_frames=64740,
                                 source_total_frames=348600, frame_bgr_sha256=["a" * 64] * 1660)
                         for i, v in enumerate(module.VIEWS)})
    data = encode(manifest)
    (chunk / "manifest.json").write_bytes(data)
    (root / "source_manifest.json").write_bytes(data)
    review["source_manifest_sha256"] = hashlib.sha256(data).hexdigest()
    review_path = tmp_path / "actual_export_fixture.json"
    review_path.write_bytes(encode(review))
    (root / "truth_template.json").write_bytes(encode(review))
    (root / "index.html").write_bytes(b"raw-only HTML fixture, no browser")
    outputs = {"manifest": file_entry(chunk / "manifest.json")}
    for view in module.VIEWS:
        outputs[view + "_raw"] = dict(path=str(chunk / (view + ".mp4")), size=100,
                                       sha256=manifest["raw"][view]["sha256"])
    (chunk / "staging.provenance.json").write_bytes(encode(dict(
        status="ok", stage="prepare_independent_mask_chunk", params={"binding": binding},
        outputs=outputs)))
    consumed = dict(start=64740, stop=66400, manifest_sha256=hashlib.sha256(data).hexdigest(),
                    provenance_sha256=module.sha256(chunk / "staging.provenance.json"))
    summary = dict(schema="long_raw_review_pack_v1", technical_completion=True,
                    model_outputs_shown=False, human_truth_created=False,
                    original_raw_storage="original_source_video", review_clip_range=[65859, 66223],
                    displayed_frames=list(range(65859, 66223)), frame_count=364,
                    native_pixel_hashes_verified=1092, source_manifest=str(chunk / "manifest.json"),
                    source_manifest_sha256=consumed["manifest_sha256"], consumed_record=consumed,
                    source_frame_range=[64740, 66400], probe_frames=[65931, 66151, 66217],
                    native_bgr_sha256={v: ["a" * 64] * 364 for v in module.VIEWS})
    (root / "summary.json").write_bytes(encode(summary))
    inputs = {str(chunk / name): file_entry(chunk / name)
              for name in ("manifest.json", "staging.provenance.json")}
    for view in module.VIEWS:
        item = outputs[view + "_raw"]
        inputs[item["path"]] = item  # Actual MP4 deliberately absent: parser never opens it.
    seal = dict(status="ok", stage="long_original_video_raw_review",
                params={"all_input_source_output_sha_unchanged": True}, inputs=inputs,
                outputs={name: file_entry(root / name) for name in
                         ("summary.json", "index.html", "truth_template.json",
                          "source_manifest.json")})
    (root / "review.provenance.json").write_bytes(encode(seal))
    return SimpleNamespace(root=root, chunk=chunk, review=review_path, seal=seal,
                           summary=summary, output=tmp_path / "staged", tmp=tmp_path)


def test_source_bound_document_opens_no_raw_video_or_candidate(pack):
    parsed_value, summary, evidence = module.load_document(pack.root, pack.review)
    assert parsed_value.anchor_frames.tolist() == [65931, 66151, 66217]
    assert summary["frame_count"] == 364
    assert not any(path.suffix in (".mp4", ".npz") for path in evidence.hashes)


def test_explicit_external_review_is_read_only_and_other_inputs_stay_workspace_scoped(pack):
    # A sibling stands for Downloads outside this isolated fixture workspace.
    # It still resides inside the real authorized test-output directory.
    external = pack.tmp.parent / ("downloads_" + pack.tmp.name + ".json")
    original = pack.review.read_bytes()
    external.write_bytes(original)
    class ScopedEvidence(module.Evidence):
        def bind(self, path, expected=None):
            module.require(Path(path).resolve().is_relative_to(pack.tmp), "outside test workspace")
            return super().bind(path, expected)
    with pytest.raises(ValueError, match="outside test workspace"):
        ScopedEvidence().bind(external)
    namespace = {**vars(module), "Evidence": ScopedEvidence}
    loader = FunctionType(module.load_document.__code__, namespace, "load_document")
    value, _, evidence = loader(pack.root, external)
    assert value.review_bytes == original and external.read_bytes() == original
    assert evidence.hashes[external] == hashlib.sha256(original).hexdigest()
    assert all(path == external or path.is_relative_to(pack.tmp) for path in evidence.hashes)
    evidence.final_guard()
    external.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed before seal"):
        evidence.final_guard()


@pytest.mark.parametrize("target,change", [
    ("seal", lambda x: x.update(status="failed")),
    ("seal", lambda x: x["params"].update(all_input_source_output_sha_unchanged=False)),
    ("seal", lambda x: x["outputs"].pop("index.html")),
    ("summary", lambda x: x["native_bgr_sha256"]["top_cam"].__setitem__(0, "b" * 64)),
])
def test_pack_failure_missing_bound_output_and_native_sha_refused(pack, target, change):
    value = getattr(pack, target)
    change(value)
    if target == "summary":
        (pack.root / "summary.json").write_bytes(encode(value))
        pack.seal["outputs"]["summary.json"] = file_entry(pack.root / "summary.json")
    (pack.root / "review.provenance.json").write_bytes(encode(pack.seal))
    with pytest.raises((ValueError, KeyError)):
        module.load_document(pack.root, pack.review)


def test_cli_stage_retains_bytes_and_explicitly_has_no_final_mapping(pack):
    namespace = dict(vars(module))
    code_path = pack.tmp / "fixture_import.py"
    code_path.write_bytes(b"# fixture only\n")
    namespace["code_hashes"] = lambda: {str(code_path): module.sha256(code_path)}
    namespace["snapshot"] = FunctionType(old.snapshot.__code__,
                                          {**vars(old), "REPO": pack.tmp}, "snapshot")
    stage = FunctionType(module.stage_document.__code__, namespace, "stage_document")
    stage(pack.root, pack.review, pack.output)
    summary = json.loads((pack.output / "summary.json").read_text())
    assert summary["candidate_mapping"] is None and summary["final_candidate_verified"] is False
    assert not summary["production_correction_ready"] and summary["requested_links"] == 6
    assert (pack.output / "original_review.json").read_bytes() == pack.review.read_bytes()
    record = json.loads((pack.output / "import.provenance.json").read_text())
    assert record["status"] == "ok" and record["stage"] == module.STAGE
    for item in record["outputs"].values():
        assert module.sha256(item["path"]) == item["sha256"]
