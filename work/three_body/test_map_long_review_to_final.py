"""Synthetic review/export fixtures only; no actual human answer or full run."""

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import map_long_review_to_final as m
import numpy as np
import pytest
from test_import_long_review import Camera, base_document, encode


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else encode(value))
    return path


def item(path):
    return dict(path=str(path), sha256=m.review.sha256(path))


@pytest.fixture
def context(tmp_path):
    stage = tmp_path / "stage"
    attempt = stage / "chunks/a"
    pack = tmp_path / "pack"
    prefix = tmp_path / "prefix"
    video_dir = tmp_path / "videos"
    finalrun = tmp_path / "final"
    frames = list(range(64740, 66400))
    index = (64740 - 49800) // 1660
    black = np.zeros((1200, 1200, 3), np.uint8)
    bgr_sha = hashlib.sha256(black.tobytes()).hexdigest()
    raw = {}
    links = {}
    outputs = {}
    detector_inputs = {}
    for view in m.VIEWS:
        original = write(video_dir / (view + ".mp4"), ("synthetic bytes " + view).encode())
        alias = attempt / "raw" / original.name
        alias.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, alias)
        stat = alias.stat()
        raw[view] = dict(
            path="raw/" + original.name,
            sha256=item(alias)["sha256"],
            source_offset_frames=64740,
            source_total_frames=348600,
            frame_bgr_sha256=[bgr_sha] * 1660,
        )
        links[view] = dict(
            path=str(alias),
            source_path=str(original),
            sha256=item(alias)["sha256"],
            device=stat.st_dev,
            inode=stat.st_ino,
            bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        outputs[view + "_raw"] = item(alias)
        detector_inputs["video:" + view] = item(original)
    detector = dict(input_sha256={k: v["sha256"] for k, v in detector_inputs.items()})
    manifest = dict(
        experiment=m.baseline.EXPERIMENT,
        frames=frames,
        fps=166,
        image_shape_hw=[1200, 1200],
        raw_storage="original_source_video",
        tracking_performed=False,
        human_truth_created=False,
        binding=detector,
        protocol_sha256="p" * 64,
        raw=raw,
    )
    manifestpath = write(attempt / "manifest.json", manifest)
    outputs["manifest"] = item(manifestpath)
    source = dict(
        stage="prepare_independent_mask_chunk",
        status="ok",
        params=dict(binding=detector),
        inputs=detector_inputs,
        outputs=outputs,
    )
    sourceseal = write(attempt / "staging.provenance.json", source)
    consumed = [
        dict(index=i, start=49800 + i * 1660, stop=49800 + (i + 1) * 1660) for i in range(180)
    ]
    row = dict(
        consumed[index],
        attempt_path="chunks/a",
        manifest_sha256=item(manifestpath)["sha256"],
        provenance_sha256=item(sourceseal)["sha256"],
        output_hashes=outputs,
        raw_hardlinks=links,
    )
    consumed[index] = row
    consumedpath = write(finalrun / "consumed_chunks.json", consumed)
    shared = {}
    for name in (
        "calibration_manifest.json",
        "cam0.json",
        "cam1.json",
        "cam2.json",
        *(f"{m.baseline.EXPERIMENT}__{v}.npz" for v in m.VIEWS),
    ):
        p = write(tmp_path / "shared" / name, {"synthetic": name})
        shared[str(p)] = item(p)["sha256"]
    binding = dict(
        inputs=dict(
            input_stage_root=str(stage),
            detector_binding=detector,
            detector_inputs=detector_inputs,
            protocol_sha256="p" * 64,
            seed=dict(SL_and_calibration_sha256=shared),
        )
    )
    prefixrecord = dict(
        stage="three_body_long_continuation",
        status="ok",
        params=dict(bindings=binding),
        outputs=dict(candidate=dict(path=str(prefix / "candidate.npz"), sha256="1" * 64)),
    )
    prefixseal = write(prefix / "continuation.provenance.json", prefixrecord)
    reviewdoc, _ = base_document()
    reviewdoc["source_manifest_sha256"] = item(manifestpath)["sha256"]
    parsed = m.review.parse_review(
        encode(reviewdoc), manifestpath.read_bytes(), clip_range=[65859, 66223]
    )
    packinputs = {str(p): item(p) for p in (manifestpath, sourceseal, prefixseal)}
    for rawitem in outputs.values():
        if rawitem["path"].endswith(".mp4"):
            packinputs[rawitem["path"]] = rawitem.copy()
    packoutputs = {}
    for frame in parsed.anchor_frames:
        for view in m.VIEWS:
            name = f"frames/f{frame}_{view}.jpg"
            packoutputs[name] = item(write(pack / name, b"bound lossy synthetic display"))
    packrecord = dict(inputs=packinputs, outputs=packoutputs)
    write(pack / "review.provenance.json", packrecord)
    summary = dict(
        consumed_record=row,
        source_manifest=str(manifestpath),
        producer_run=str(prefix),
        probe_frames=[65931, 66151, 66217],
    )
    finalrecord = dict(
        params=dict(bindings=binding, through_frame=348600),
        outputs=dict(
            consumed_chunks=item(consumedpath),
            candidate=dict(path=str(finalrun / "candidate.npz"), sha256="2" * 64),
        ),
    )
    chain = dict(final_run=finalrun, final_record=finalrecord, binding=binding)
    return SimpleNamespace(
        root=tmp_path,
        pack=pack,
        parsed=parsed,
        summary=summary,
        chain=chain,
        manifest=manifest,
        manifestpath=manifestpath,
        source=source,
        sourceseal=sourceseal,
        videos={v: Path(outputs[v + "_raw"]["path"]) for v in m.VIEWS},
        black=black,
        prefixseal=prefixseal,
        shared=shared,
        consumedpath=consumedpath,
    )


def test_source_binding_accepts_raw_prefix_without_opening_prefix_coordinates(context, monkeypatch):
    f = context
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("coordinate payload opened"))
    bound = m.verify_source_binding(f.chain, f.parsed, f.summary, f.pack, m.final.phase.Digests())
    assert bound["proof"]["same_final_consumed_chunk"] is True
    assert bound["proof"]["prefix_candidate_sha256"] == "1" * 64
    assert bound["proof"]["final_candidate_sha256"] == "2" * 64
    assert bound["proof"]["prefix_coordinate_arrays_opened"] is False
    assert len(bound["displayed"]) == 9


@pytest.mark.parametrize(
    "fault", ["consumed", "prefix_SL", "manifest_path", "raw_source", "displayed_image"]
)
def test_wrong_final_review_source_connection_rejected(context, fault):
    f = context
    if fault == "consumed":
        f.summary = copy.deepcopy(f.summary)
        f.summary["consumed_record"]["manifest_sha256"] = "0" * 64
    elif fault == "prefix_SL":
        data = json.loads(f.prefixseal.read_text())
        data["params"]["bindings"]["inputs"]["seed"]["SL_and_calibration_sha256"] = {}
        write(f.prefixseal, data)
        p = f.pack / "review.provenance.json"
        data = json.loads(p.read_text())
        data["inputs"][str(f.prefixseal)] = item(f.prefixseal)
        write(p, data)
    elif fault == "manifest_path":
        f.summary["source_manifest"] = str(f.sourceseal)
    elif fault == "raw_source":
        f.chain = copy.deepcopy(f.chain)
        f.chain["binding"]["inputs"]["detector_inputs"]["video:" + m.VIEWS[0]]["sha256"] = "0" * 64
    elif fault == "displayed_image":
        path = f.pack / "review.provenance.json"
        data = json.loads(path.read_text())
        data["outputs"].pop(next(iter(data["outputs"])))
        write(path, data)
    with pytest.raises(ValueError):
        m.verify_source_binding(f.chain, f.parsed, f.summary, f.pack, m.final.phase.Digests())


class Capture:
    def __init__(self, f, path, *, fail=None):
        self.f = f
        self.position = 0
        self.fail = fail
        self.closed = False
        self.reads = 0
        self.seeks = []

    def isOpened(self):
        return True

    def get(self, prop):
        cv = m.raw.cv2
        values = {
            cv.CAP_PROP_FRAME_COUNT: 348600,
            cv.CAP_PROP_FPS: 166,
            cv.CAP_PROP_FRAME_HEIGHT: 1200,
            cv.CAP_PROP_FRAME_WIDTH: 1200,
            cv.CAP_PROP_POS_FRAMES: self.position + (1 if self.fail == "offset" else 0),
        }
        return values[prop]

    def set(self, prop, value):
        self.position = int(value)
        self.seeks.append(int(value))
        return True

    def read(self):
        self.position += 1
        self.reads += 1
        if self.fail == "EOF":
            return False, None
        image = self.f.black if self.fail != "pixels" else self.f.black + 1
        return True, image

    def release(self):
        self.closed = True


def test_exact_nine_native_decodes_and_absolute_seeks(context):
    f = context
    captures = []

    def factory(path):
        c = Capture(f, path)
        captures.append(c)
        return c

    result = m.decode_anchor_images(
        f.videos, f.manifest, f.parsed.anchor_frames, capture_factory=factory
    )
    assert len(result) == 9 and sum(c.reads for c in captures) == 9
    assert all(c.closed for c in captures)
    assert [c.seeks[0] for c in captures] == np.repeat(f.parsed.anchor_frames, 3).tolist()
    assert {r["bgr_sha256"] for r in result} == {hashlib.sha256(f.black.tobytes()).hexdigest()}


@pytest.mark.parametrize("fault", ["offset", "EOF", "pixels"])
def test_raw_offset_eof_or_pixel_mismatch_refused_and_handles_closed(context, fault):
    captures = []

    def factory(path):
        c = Capture(context, path, fail=fault)
        captures.append(c)
        return c

    with pytest.raises(ValueError):
        m.decode_anchor_images(
            context.videos, context.manifest, context.parsed.anchor_frames, capture_factory=factory
        )
    assert all(c.closed for c in captures)


def completed_fixture(f):
    count = m.native.STOP - m.native.FIRST
    pose = np.zeros((3, 3, 3))
    pose[:, :, 0] = np.array([30.0, 60.0, 90.0])[:, None]
    pose[:, :, 1] = 100.0
    pose[:, :, 2] = 20.0
    # Final slots deliberately differ from the old raw-label fixture.
    permutation = [2, 0, 1]
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy().astype(np.int64)
    arrays = dict(
        frame_idx=np.arange(m.native.FIRST, m.native.STOP, dtype=np.int64),
        pose_xyz=np.broadcast_to(pose[permutation], (count, 3, 3, 3)),
        sources=np.broadcast_to(sources[permutation], (count, 3, 3, 3)),
        observed=np.ones((count, 3, 3), bool),
        identity_review=np.zeros((count, 3), bool),
    )
    rows = m.anchor_arrays(arrays, f.parsed.anchor_frames)
    points = {
        int(frame): {v: pose[:, :, :2].copy() for v in m.VIEWS} for frame in f.parsed.anchor_frames
    }
    return (
        dict(
            chain=f.chain,
            candidate=arrays,
            checkpoint={"synthetic": "no actual codec"},
            checkpoint_metadata=dict(
                processed_frames=count, previous_frame=m.native.STOP - 1, context={}
            ),
            exact_backtrace={k: dict(sha256="synthetic fixture only") for k in m.ANCHOR_KEYS},
            final_audit_complete=False,
        ),
        rows,
        points,
    )


def test_final_only_guard_and_owned_anchor_slice_preserve_current_permutation(context):
    full, rows, _ = completed_fixture(context)
    assert m.guard_completed(full) is context.chain
    assert rows["pose_xyz"][0, :, 0, 0].tolist() == [90.0, 30.0, 60.0]
    rows["pose_xyz"][:] = 0
    assert full["candidate"]["pose_xyz"][0].any()
    full["candidate"]["frame_idx"] = full["candidate"]["frame_idx"][:-1]
    with pytest.raises(ValueError, match="full30"):
        m.guard_completed(full)


@pytest.mark.parametrize("fault", [None, "camera_context", "SL_view", "input_hash"])
def test_registered_anchor_geometry_keeps_actual_native_row_order(context, monkeypatch, fault):
    f = context

    @dataclasses.dataclass
    class SyntheticCamera:
        name: str
        offset: float

    cameras = [SyntheticCamera(view, float(i)) for i, view in enumerate(m.VIEWS)]
    rig = SimpleNamespace(cameras=cameras, views=m.VIEWS, world_units="mm")
    monkeypatch.setattr(m, "load_rig", lambda path: rig)
    manifest_path = next(Path(p) for p in f.shared if Path(p).name == "calibration_manifest.json")
    write(manifest_path, dict(model_files=[dict(path=f"cam{i}.json") for i in range(3)]))
    # Deliberately keep two native rows in reverse coordinate order. The loader
    # must preserve row indices, not sort/reassign by coordinate proximity.
    points = np.arange(12, dtype=np.float64).reshape(2, 3, 2)[::-1].copy()
    frames = f.parsed.anchor_frames
    for i, view in enumerate(m.VIEWS):
        path = manifest_path.parent / f"{m.baseline.EXPERIMENT}__{view}.npz"
        np.savez_compressed(
            path,
            experiment=np.array(m.baseline.EXPERIMENT),
            view=np.array("wrong" if fault == "SL_view" and i == 0 else view),
            nodes=np.array(m.baseline.NODES),
            frame_idx=np.repeat(frames, 2),
            points=np.tile(points, (3, 1, 1)),
        )
    shared = {p: m.review.sha256(p) for p in f.shared}
    context_data = dict(
        cameras=[dataclasses.asdict(c) for c in cameras], views=list(m.VIEWS), world_units="mm"
    )
    if fault == "camera_context":
        context_data["cameras"][0]["offset"] += 1
    if fault == "input_hash":
        shared[str(manifest_path)] = "0" * 64
    if fault:
        with pytest.raises(ValueError):
            m.load_anchor_geometry(shared, context_data, frames, m.final.phase.Digests())
    else:
        actual_rig, actual_points, calibration = m.load_anchor_geometry(
            shared, context_data, frames, m.final.phase.Digests()
        )
        assert actual_rig is rig and len(calibration) == 4
        for frame in frames:
            for view in m.VIEWS:
                assert actual_points[int(frame)][view].tobytes() == points.tobytes()


def minimal_main(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(m.final, "PINS", {})
    source = write(tmp_path / "mapping_source.py", b"synthetic test source")
    codes = {str(source): m.review.sha256(source)}
    monkeypatch.setattr(m, "code_hashes", lambda: codes.copy())

    def snapshot(code, root, record):
        p = write(root / "code_snapshot/mapping_source.py", source.read_bytes())
        record.output("snapshot", p, sha256=m.review.sha256(p))

    monkeypatch.setattr(m, "snapshot", snapshot)
    return tmp_path / "work/mapping"


def test_unfinished_full_run_fails_before_review_or_matching(tmp_path, monkeypatch):
    output = minimal_main(monkeypatch, tmp_path)
    calls = []

    def fail(*a):
        calls.append("final")
        raise ValueError("unfinished final")

    monkeypatch.setattr(m.final, "load_completed_final", fail)
    monkeypatch.setattr(
        m.review, "load_document", lambda *a: pytest.fail("review opened before fullrun")
    )
    with pytest.raises(ValueError, match="unfinished final"):
        m.run_mapping(tmp_path / "unfinished", tmp_path / "pack", tmp_path / "review.json", output)
    assert calls == ["final"]
    assert json.loads((output / "mapping.provenance.json").read_text())["status"] == "failed"
    assert not (output / "mapping.json").exists()


def test_synthetic_cli_maps_fresh_final_slots_and_preserves_local_scope(context, monkeypatch):
    f = context
    output = minimal_main(monkeypatch, f.root)
    full, _, points = completed_fixture(f)
    calls = []

    def final_loader(*args):
        calls.append("final")
        return full

    monkeypatch.setattr(m.final, "load_completed_final", final_loader)
    reviewpath = write(f.root / "synthetic_review.json", f.parsed.review_bytes)
    doc_evidence = m.raw.Evidence()
    doc_evidence.bind(reviewpath)

    def load_document(*args):
        calls.append("review")
        return f.parsed, f.summary, doc_evidence

    monkeypatch.setattr(m.review, "load_document", load_document)
    monkeypatch.setattr(
        m,
        "load_anchor_geometry",
        lambda *args: (
            SimpleNamespace(camera=lambda v: Camera()),
            points,
            {"synthetic_calibration": "d" * 64},
        ),
    )
    original_decoder = m.decode_anchor_images
    monkeypatch.setattr(
        m,
        "decode_anchor_images",
        lambda videos, manifest, frames: original_decoder(
            videos, manifest, frames, capture_factory=lambda p: Capture(f, p)
        ),
    )
    m.run_mapping(f.root / "completed_final", f.pack, reviewpath, output)
    assert calls == ["final", "review"]
    seal = json.loads((output / "mapping.provenance.json").read_text())
    assert seal["status"] == "ok"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_audit_complete"] is False and summary["native_raw_images_redecoded"] == 9
    assert not summary["dense_trajectory_modified"] and not summary["raw_review_cleared"]
    assert (output / "original_review.json").read_bytes() == reviewpath.read_bytes()
    result = json.loads((output / "mapping.json").read_text())
    assert [row["raw_match"]["slot"] for row in result["endpoints"]] == [1, 2, 0] * 3
    assert all(not row["interior_identity_certified"] for row in result["links"])
    for value in seal["outputs"].values():
        assert m.review.sha256(value["path"]) == value["sha256"]
