"""Synthetic fixtures only: no original photos, NPZ, video or metadata loaders."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import render_partial_body_bank_comparison as m

LOAD_CONTEXT = m.load_context


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    for obj, name, value in (
        (m, "WORK", work),
        (m, "REPO", tmp_path),
        (m.proof, "REPO", tmp_path),
        (m, "PRIMARY_ROOT", work / "primary"),
        (m, "OLD_ROOT", work / "old"),
        (m, "JPEG_ROOT", work / "jpeg"),
    ):
        monkeypatch.setattr(obj, name, value)
    monkeypatch.setattr(m, "load_context", lambda *a: pytest.fail("real metadata forbidden"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NPZ forbidden"))
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", lambda *a: pytest.fail("video forbidden"))
    return work


def case_fixture(frame=82720, arm=None):
    case = dict(arm=arm or m.ARMS[0], frame_idx=frame, selections={})
    summary = dict(arm=case["arm"], frame_idx=frame, candidates=3, rankings={})
    for i, mode in enumerate(m.MODES):
        pose = np.arange(27, dtype=np.float64).reshape(3, 3, 3) + i * 40
        indices = [i] if i != 0 else [0, 2]
        row = dict(
            candidate_index=i,
            path_minimum_indices=indices,
            local_minimum_indices=[2],
            path_cost=10.0 + i,
            local_cost=3.0 + i,
            parent_path_cost=7.0,
            best_local_cost=1.0,
            best_path_cost=10.0 + i,
            support_not_recomputed=True,
            plan_is_assigned_views_not_direct_support=True,
            pose_xyz=pose.tolist(),
            candidate_plan=np.zeros((3, 3, 3), np.int32).tolist(),
            projections={v: (pose[..., :2] + vi).tolist() for vi, v in enumerate(m.VIEWS)},
        )
        case["selections"][mode] = row
        summary["rankings"][mode] = dict(
            best_local_cost=1.0,
            best_path_cost=10.0 + i,
            path_minimum_indices=indices,
            local_minimum_indices=[2],
            path_order=[i] + [j for j in range(3) if j != i],
        )
    return case, summary


def test_selects_saved_path_winner_not_local_winner_and_preserves_slots():
    case, summary = case_fixture()
    before = copy.deepcopy(case)
    selected = m.selected_case(case, summary)
    assert selected["selections"]["baseline"]["row"]["candidate_index"] == 0
    assert selected["selections"]["baseline"]["row"]["path_minimum_indices"] == [0, 2]
    for mode in m.MODES:
        row = selected["selections"][mode]
        assert np.array_equal(row["pose"], case["selections"][mode]["pose_xyz"])
        assert not row["pose"].flags.writeable and not row["plan"].flags.writeable
        assert all(not p.flags.writeable for p in row["projections"].values())
    assert case == before


@pytest.mark.parametrize("change", ("winner", "tie", "ledger", "pose", "plan", "projection"))
def test_invalid_selected_contract_rejected(change):
    case, summary = case_fixture()
    row = case["selections"]["baseline"]
    if change == "winner":
        row["candidate_index"] = 2
    elif change == "tie":
        row["path_minimum_indices"] = [0]
    elif change == "ledger":
        row["path_cost"] += 1
    elif change == "pose":
        row["pose_xyz"][0][0][0] = None
    elif change == "plan":
        row["candidate_plan"][0][0][0] = 0.5
    else:
        row["projections"][m.VIEWS[0]] = []
    with pytest.raises(ValueError):
        m.selected_case(case, summary)


def photograph_fixture(frame):
    view = m.VIEWS[0]
    image = np.indices((1200, 1200)).sum(axis=0).astype(np.uint8)
    image = np.stack((image, image // 2, image // 3), axis=-1)
    points = np.array([[[10, 20], [np.nan, np.nan], [30, 40]]], np.float32)
    obs = dict(
        SLEAP_points=m.prior.finite_json(points),
        SL_points_signature=m.proof.signature(points),
        bgr_sha256=m.prior.prepare_sequence.array_sha256(image),
        canonical_video_frame=frame,
    )
    lossless = frame in m.PRIMARY
    root = m.PRIMARY_ROOT if lossless else m.JPEG_ROOT
    name = f"native/frame_{frame}_{view}.png" if lossless else f"frames/f{frame}_{view}.jpg"
    if lossless:
        obs["native_png"] = name
    path = root / name
    path.parent.mkdir(parents=True)
    success, encoded = m.prior.cv2.imencode(path.suffix, image)
    assert success
    path.write_bytes(encoded.tobytes())
    seal = dict(
        outputs={
            name: dict(path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size)
        }
    )
    hashes = ["a" * 64] * 360
    if not lossless:
        hashes[frame - 90040] = obs["bgr_sha256"]
    context = dict(
        records={frame: dict(observations={view: obs})},
        primary_seal=seal,
        jpeg_seal=seal,
        jpeg_summary=dict(native_bgr_sha256={view: hashes}),
    )
    return context, view, image


def test_native_png_pixels_checked_and_wrong_native_binding_rejected():
    context, view, original = photograph_fixture(82720)
    image, points, record = m.photograph_and_points(context, 82720, view, m.proof.Evidence())
    assert np.array_equal(image, original) and record["exact_source_pixels"]
    assert not record["lossy_JPEG_context"] and np.isnan(points[0, 1]).all()
    context["records"][82720]["observations"][view]["bgr_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="pixel SHA"):
        m.photograph_and_points(context, 82720, view, m.proof.Evidence())


def test_registered_jpeg_is_lossy_context_not_exact_BGR():
    context, view, original = photograph_fixture(90141)
    image, _, record = m.photograph_and_points(context, 90141, view, m.proof.Evidence())
    assert not np.array_equal(image, original)
    assert record["lossy_JPEG_context"] and not record["exact_source_pixels"]
    context["jpeg_summary"]["native_bgr_sha256"][view][101] = "f" * 64
    with pytest.raises(ValueError, match="lineage"):
        m.photograph_and_points(context, 90141, view, m.proof.Evidence())


def test_saved_SL_signature_and_photo_path_escape_rejected():
    context, view, _ = photograph_fixture(82720)
    context["records"][82720]["observations"][view]["SLEAP_points"][0][0][0] += 1
    with pytest.raises(ValueError, match="SL reconstruction"):
        m.photograph_and_points(context, 82720, view, m.proof.Evidence())
    context["records"][82720]["observations"][view]["SLEAP_points"][0][0][0] -= 1
    name = f"native/frame_82720_{view}.png"
    context["primary_seal"]["outputs"][name]["path"] = str(m.WORK.parent / "outside.png")
    with pytest.raises(ValueError, match="path/size"):
        m.photograph_and_points(context, 82720, view, m.proof.Evidence())


class Axis:
    def __init__(self):
        self.plots, self.labels, self.bounds = [], [], []

    def imshow(self, image):
        self.image = image

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def annotate(self, label, xy, **kwargs):
        self.labels.append(label)

    def set_xlim(self, *xy):
        self.bounds.append(xy)

    set_ylim = set_xlim

    def set_title(self, title, **kwargs):
        self.title = title

    def tick_params(self, **kwargs):
        pass


def test_own_drawer_never_claims_direct_support_for_proposed_plans():
    axis = Axis()
    m.draw_panel(
        axis,
        np.zeros((100, 100, 3), np.uint8),
        np.empty((0, 3, 2)),
        np.arange(18, dtype=float).reshape(3, 3, 2),
    )
    markers = [kw for _, kw in axis.plots if "marker" in kw]
    assert len(markers) == 9 and all(kw["markerfacecolor"] == "none" for kw in markers)
    assert [kw["marker"] for kw in markers] == ["o", "s", "^"] * 3
    assert axis.labels == [f"S{s}{n}" for s in range(3) for n in m.LABELS]
    assert all(kw["ls"] == "--" for _, kw in axis.plots if "marker" not in kw)


def test_three_mode_drawing_shares_crop_and_retains_tie_disclaimer(monkeypatch):
    axes = np.array([[Axis() for _ in range(3)] for _ in range(3)], dtype=object)
    titles, closed = [], []

    class Figure:
        def suptitle(self, title, **kwargs):
            titles.append(title)

        def tight_layout(self, **kwargs):
            pass

        def savefig(self, stream, **kwargs):
            stream.write(b"SYNTHETIC_FIGURE")

    monkeypatch.setattr(m.prior.plt, "subplots", lambda *a, **k: (Figure(), axes))
    monkeypatch.setattr(m.prior.plt, "close", lambda *a: closed.append(True))
    case, summary = case_fixture(frame=90141)
    case = m.selected_case(case, summary)
    images = {v: np.zeros((1200, 1200, 3), np.uint8) for v in m.VIEWS}
    points = {v: np.empty((0, 3, 2)) for v in m.VIEWS}
    data, crops = m.render(case, images, points)
    assert data == b"SYNTHETIC_FIGURE" and closed == [True]
    assert "LOSSY JPEG CONTEXT" in titles[0] and "ALL open markers" in titles[0]
    assert "no new continuous trajectory" in titles[0]
    for vi, view in enumerate(m.VIEWS):
        assert axes[0, vi].bounds == axes[1, vi].bounds == axes[2, vi].bounds
        assert "exact minima=[0, 2]" in axes[0, vi].title
        assert view in crops


def test_outside_or_existing_output_cannot_read_inputs(isolated):
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(isolated / "comparison", "a" * 64, isolated.parent / "outside")
    existing = isolated / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(isolated / "comparison", "a" * 64, existing)


def sealed_fixture(root, stage, documents, inputs=None, seal_name="render.provenance.json"):
    out = m.proof.Output(root)
    for name, doc in documents.items():
        out.write(name, doc)
    seal = dict(stage=stage, status="ok", inputs=inputs or {}, outputs=out.records)
    path = root / seal_name
    path.write_bytes(m.proof.json_bytes(seal))
    return seal, m.proof.sha(path)


def complete_metadata_fixture(monkeypatch, change=None):
    """Synthetic successful producer trust boundary, actual new JSON loading path."""
    original = {}
    for prefix, root, frames in (
        ("PRIMARY", m.PRIMARY_ROOT, m.PRIMARY),
        ("OLD", m.OLD_ROOT, m.NEGATIVE),
    ):
        doc = dict(
            status="ok",
            physical_ID_truth=False,
            records=[dict(frame=f, observations={}) for f in frames],
            source_chunk=dict(index=24, start=89640, stop=91300),
        )
        seal, digest = sealed_fixture(
            root, "synthetic_registered_visual", {"source_and_frame_evidence.json": doc}
        )
        docsha = seal["outputs"]["source_and_frame_evidence.json"]["sha256"]
        monkeypatch.setattr(m, prefix + "_SEAL_SHA", digest)
        monkeypatch.setattr(m, prefix + "_EVIDENCE_SHA", docsha)
        original[str(root / "render.provenance.json")] = digest
        original[str(root / "source_and_frame_evidence.json")] = docsha
    jpeg_summary = dict(
        status="ok",
        review_clip_range=[90040, 90400],
        consumed_record=dict(index=24, start=89640, stop=91300),
        selection_render_sha256=m.OLD_SEAL_SHA,
    )
    _, jpegsha = sealed_fixture(
        m.JPEG_ROOT,
        "fast_tail_transition_raw_review_pack_v1",
        {"summary.json": jpeg_summary},
        {str(m.OLD_ROOT / "render.provenance.json"): m.OLD_SEAL_SHA},
        "review.provenance.json",
    )
    monkeypatch.setattr(m, "JPEG_SEAL_SHA", jpegsha)
    cases, summaries = [], []
    for frame in (*m.PRIMARY, *m.NEGATIVE):
        for arm in m.ARMS if frame in m.PRIMARY else m.ARMS[:1]:
            c, s = case_fixture(frame, arm)
            cases.append(c)
            summaries.append(s)
    payload = dict(
        schema=m.COMPARISON_STAGE + "_render_payload_v1",
        modes=list(m.MODES),
        cases=cases,
        physical_ID_truth=False,
        raw_slot_labels_only=True,
        new_trajectory_created=False,
        original_observation_evidence=[
            dict(root=str(root), seal_sha256=sha, evidence_sha256=docsha, frames=list(frames))
            for root, sha, docsha, frames in (
                (m.PRIMARY_ROOT, m.PRIMARY_SEAL_SHA, m.PRIMARY_EVIDENCE_SHA, m.PRIMARY),
                (m.OLD_ROOT, m.OLD_SEAL_SHA, m.OLD_EVIDENCE_SHA, m.NEGATIVE),
            )
        ],
    )
    if change == "source":
        payload["original_observation_evidence"][0]["frames"][0] += 1
    if change == "missing":
        cases.pop()
    summary = dict(
        schema=m.COMPARISON_STAGE,
        status="ok",
        cases=summaries,
        new_trajectory_created=False,
        baseline_bank_unchanged=True,
    )
    plan = dict(schema=m.COMPARISON_STAGE, fixed_parent_history=True, physical_ID_truth=False)
    root = m.WORK / "comparison"
    seal, digest = sealed_fixture(
        root,
        m.COMPARISON_STAGE,
        {"plan.json": plan, "summary.json": summary, "render_payload.json": payload},
        original,
        "comparison.provenance.json",
    )
    if change == "failed":
        seal["status"] = "failed"
        path = root / "comparison.provenance.json"
        path.write_bytes(m.proof.json_bytes(seal))
        digest = m.proof.sha(path)
    monkeypatch.setattr(m, "source_code", lambda *a: {})
    return root, digest


@pytest.mark.parametrize(
    "change,reason",
    (("source", "references"), ("missing", "render cases"), ("failed", "successful reranking")),
)
def test_complete_metadata_path_rejects_wrong_source_missing_case_or_failed_seal(
    monkeypatch, change, reason
):
    root, digest = complete_metadata_fixture(monkeypatch, change)
    with pytest.raises(ValueError, match=reason):
        LOAD_CONTEXT(root, digest, m.proof.Evidence())


def test_whole_synthetic_eleven_case_orchestration_reads_photos_once(monkeypatch, isolated):
    root, digest = complete_metadata_fixture(monkeypatch)
    monkeypatch.setattr(m, "load_context", LOAD_CONTEXT)
    calls, rendered = [], []
    image = np.zeros((1200, 1200, 3), np.uint8)

    def photograph(context, frame, view, evidence):
        assert frame in (*m.PRIMARY, *m.NEGATIVE) and view in m.VIEWS
        calls.append((frame, view))
        return (
            image,
            np.empty((0, 3, 2), np.float32),
            dict(
                frame=frame,
                view=view,
                lossy_JPEG_context=frame in m.NEGATIVE,
                exact_source_pixels=frame in m.PRIMARY,
            ),
        )

    def render(case, images, points):
        assert set(images) == set(points) == set(m.VIEWS)
        assert set(case["selections"]) == set(m.MODES)
        rendered.append((case["arm"], case["frame"]))
        return b"SYNTHETIC_RENDER_STUB", dict(same_crop=True)

    monkeypatch.setattr(m, "photograph_and_points", photograph)
    monkeypatch.setattr(m, "render", render)
    output = isolated / "rendered"
    m.run(root, digest, output)
    seal = m.proof.read_json(output / "render.provenance.json")
    summary = m.proof.read_json(output / "summary.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 13
    assert calls == [(f, v) for f in (*m.PRIMARY, *m.NEGATIVE) for v in m.VIEWS]
    assert len(rendered) == 11 and len(set(rendered)) == 11
    assert summary["contact_sheets"] == 11 and not summary["support_recomputed"]
    assert not summary["physical_ID_truth"] and not summary["new_trajectory_created"]
