"""Synthetic CPU contract tests; no actual recording, candidate or identity solve."""

import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import render_fast_tail_five_minute_id_transition as render


def index():
    return dict(
        original_binding={"synthetic": True},
        prefix_chunks=[
            dict(index=i, start=49800 + 1660 * i, stop=49800 + 1660 * (i + 1)) for i in range(40)
        ],
    )


def test_execution_gate_and_exact_chunk24(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="ROOT render Go"):
        render.run(tmp_path / "disabled")
    assert not (tmp_path / "disabled").exists()
    assert render.select_chunk(index()) == dict(index=24, start=89640, stop=91300)
    assert render.FRAMES == (90141, 90147, 90258, 90259, 90269, 90290)


@pytest.mark.parametrize("change", ["schema", "missing", "order", "range", "bool_index"])
def test_bad_consumed_index_rejected(change):
    doc = index()
    if change == "schema":
        doc["rationale_references"] = {}
    elif change == "missing":
        doc["prefix_chunks"].pop()
    elif change == "order":
        doc["prefix_chunks"][23], doc["prefix_chunks"][24] = (
            doc["prefix_chunks"][24],
            doc["prefix_chunks"][23],
        )
    elif change == "range":
        doc["prefix_chunks"][24]["start"] = 89100
    else:
        doc["prefix_chunks"][0]["index"] = False
    with pytest.raises(ValueError):
        render.select_chunk(doc)


def identity_headers():
    binding = dict(
        comparison_candidate_reference={"arrays": {"test": "digest"}},
        comparison_seal_sha256=render.ids.COMPARISON_SEAL_SHA,
        candidate_archive_references={"final": {"sha256": "same"}, "milestone": {"sha256": "same"}},
    )
    plan = dict(
        schema=render.ids.STAGE,
        native_interval=[66400, 116200],
        native_frames=49800,
        beam_width=256,
        physical_ID_truth=False,
        automatic_adoption=False,
        gaps_conditional_on_retained_graph=True,
        gaps_are_probabilities=False,
        global_optimality_certified=False,
        **binding,
        raw21_signatures={"test": "digest"},
        initial_P=[0, 1, 2],
        node_indices=list(render.ids.NODES),
        dp_config=copy.deepcopy(render.ids.CONFIG),
        source_sha256={str(Path(render.ids.__file__).resolve()): render.ID_SOURCE_SHA},
    )
    summary = copy.deepcopy(plan)
    summary.update(
        status="ok",
        terminal_horizon=116200,
        solve_calls=1,
        original_raw21_unchanged=True,
        raw_sources_review_unchanged=True,
        review_flags_cleared=False,
        intermediate_ID_resets=0,
        automatic_retry=False,
        cap_increase=False,
    )
    seal = dict(
        stage=render.ids.STAGE,
        status="ok",
        inputs={
            str(Path(render.ids.__file__).resolve()): render.ID_SOURCE_SHA,
            str(
                render.ids.COMPARISON_ROOT / "comparison.provenance.json"
            ): render.ids.COMPARISON_SEAL_SHA,
        },
    )
    return seal, plan, summary, binding


def test_separate_identity_binding_and_same_candidate():
    render.validate_identity(*identity_headers())


@pytest.mark.parametrize(
    "change", ["source", "stage", "candidate", "horizon", "truth", "initial", "retry"]
)
def test_identity_header_mixups_rejected(change):
    seal, plan, summary, binding = identity_headers()
    if change == "source":
        seal["inputs"][str(Path(render.ids.__file__).resolve())] = "other"
    elif change == "stage":
        seal["stage"] = render.visual.STAGE
    elif change == "candidate":
        summary["raw21_signatures"]["test"] = "another candidate"
    elif change == "horizon":
        summary["terminal_horizon"] = 86320
    elif change == "truth":
        summary["physical_ID_truth"] = True
    elif change == "initial":
        plan["initial_P"] = [2, 1, 0]
    else:
        summary["automatic_retry"] = True
    with pytest.raises(ValueError):
        render.validate_identity(seal, plan, summary, binding)


def candidate():
    n = render.STOP - render.FIRST
    a = dict(
        frame_idx=np.arange(render.FIRST, render.STOP),
        pose_xyz=np.zeros((n, 3, 3, 3)),
        sources=np.full((n, 3, 3, 3), -1, np.int64),
        observed=np.zeros((n, 3, 3), bool),
        identity_review=np.ones((n, 3), bool),
    )
    a["pose_xyz"][:, 0, 0, 0] = a["frame_idx"]
    return a, dict(arrays={k: render.proof.signature(v) for k, v in a.items()})


def test_candidate_five_fields_readonly_and_no_fake_comparison_copy():
    a, ref = candidate()
    selected = render.candidate_rows(a, ref)
    assert set(selected) == set(render.KEYS)
    np.testing.assert_array_equal(selected["pose_xyz"][:, 0, 0, 0], render.FRAMES)
    assert selected["identity_review"].all() and (selected["sources"] == -1).all()
    assert not selected["observed"].any()
    assert all(not v.flags.writeable for v in selected.values())
    a["observed"][0, 0, 0] = True
    with pytest.raises(ValueError, match="signatures"):
        render.candidate_rows(a, ref)
    a["observed"][0, 0, 0] = False
    a["frame_idx"][20] += 1
    ref["arrays"]["frame_idx"] = render.proof.signature(a["frame_idx"])
    with pytest.raises(ValueError, match="horizon"):
        render.candidate_rows(a, ref)


def mapping():
    n = render.STOP - render.START
    p = np.tile(np.array([0, 1, 2], np.int64), (n, 1))
    p[90147 - render.START :] = [2, 1, 0]
    a = dict(frame_idx=np.arange(render.START, render.STOP), permutation_by_id=p)
    for prefix, shape in (
        ("conditional_state_alternative", (n,)),
        ("conditional_identity_alternative", (n, 3)),
    ):
        a[prefix + "_status"] = np.full(shape, "not_retained_unknown", dtype="U24")
        a[prefix + "_gap"] = np.full(shape, np.inf)
    return a


def test_mapping_orientation_unknown_infinity_and_readonly():
    a = mapping()
    rows = render.mapping_rows(a)
    np.testing.assert_array_equal(rows["permutation_by_id"][0], [0, 1, 2])
    np.testing.assert_array_equal(rows["permutation_by_id"][1:], np.tile([2, 1, 0], (5, 1)))
    assert np.isinf(rows["conditional_identity_alternative_gap"]).all()
    assert (rows["conditional_identity_alternative_status"] == "not_retained_unknown").all()
    assert all(not v.flags.writeable for v in rows.values())
    for position in (0, 900):
        bad = {k: v.copy() for k, v in a.items()}
        bad["permutation_by_id"][position] = [0, 0, 2]
        with pytest.raises(ValueError, match="permutation"):
            render.mapping_rows(bad)


def image_fixture():
    image = np.zeros((80, 90, 3), np.uint8)
    masks = np.zeros((3, 80, 90), bool)
    points = np.array(
        [
            [[12, 16], [16, 17], [20, 18]],
            [[32, 36], [36, 37], [40, 38]],
            [[52, 56], [56, 57], [60, 58]],
        ],
        np.float32,
    )
    for i in range(3):
        masks[i, 15 + 20 * i : 20 + 20 * i, 10 + 20 * i : 22 + 20 * i] = True
    observed = np.ones((3, 3), bool)
    observed[0, 0] = False
    return image, masks, points, observed


def test_non_involutive_ID_to_slot_reindex_is_geometry_exact():
    _, _, points, observed = image_fixture()
    points[1, 2] = np.nan
    p = np.array([1, 2, 0])  # Not its own inverse: catches the wrong mapping orientation.
    xy, obs = render.remap_display(points, observed, p)
    np.testing.assert_array_equal(xy[0], points[1])
    np.testing.assert_array_equal(obs[2], observed[0])
    assert not xy.flags.writeable and not obs.flags.writeable
    with pytest.raises(ValueError, match="bijective"):
        render.remap_display(points, observed, np.array([0, 0, 2]))


def test_ID_panel_changes_only_labels_colors_and_keeps_latent_nodes():
    image, masks, points, observed = image_fixture()
    before = [a.tobytes() for a in (image, masks, points, observed)]
    fig, axes = render.prior.plt.subplots(1, 2)
    try:
        render.visual.draw_panel(axes[0], image, masks, points, points, observed, "raw")
        render.draw_identity_panel(
            axes[1], image, masks, points, points, observed, np.array([1, 2, 0])
        )
        labels = {t.get_text() for t in axes[1].texts}
        assert {"ID0M(S1)", "ID1C(S2)", "ID2T(S0)", "SL0M", "D2#2"} <= labels
        assert not {f"S{i}{n}" for i in range(3) for n in "MCT"} & labels

        def geometry(ax):
            return sorted(
                (
                    np.asarray(line.get_xdata()).tobytes(),
                    np.asarray(line.get_ydata()).tobytes(),
                    line.get_marker(),
                    line.get_linestyle(),
                )
                for line in ax.lines
            )

        assert geometry(axes[0]) == geometry(axes[1])
        circles = [line for line in axes[1].lines if line.get_marker() == "o"]
        assert circles[2].get_markerfacecolor() == "none"
        assert circles[0].get_markerfacecolor() != "none"
    finally:
        render.prior.plt.close(fig)
    assert before == [a.tobytes() for a in (image, masks, points, observed)]


def test_synthetic_composite_has_nine_axes_and_same_crops(monkeypatch):
    image, masks, points, observed = image_fixture()
    axes_seen = []
    original = render.prior.plt.close

    def close(fig):
        if hasattr(fig, "axes"):
            axes_seen.extend((ax.get_xlim(), ax.get_ylim()) for ax in fig.axes)
        original(fig)

    monkeypatch.setattr(render.prior.plt, "close", close)
    png, crops = render.render(
        90147,
        dict.fromkeys(render.VIEWS, image),
        dict.fromkeys(render.VIEWS, masks),
        dict.fromkeys(render.VIEWS, points),
        dict.fromkeys(render.VIEWS, points),
        observed,
        np.array([2, 1, 0]),
        ["finite_gap"] * 3,
    )
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(crops) == 3 and len(axes_seen) == 9
    for column in range(3):
        assert axes_seen[column] == axes_seen[3 + column] == axes_seen[6 + column]


def packed_fixture(frames):
    d = {
        k: np.empty(0, dtype=dt) for k, dt in render.observations.chunk_masks.SCHEMA_DTYPES.items()
    }
    d.update(
        image_shape_hw=np.array([1200, 1200], np.int32),
        frame_idx=frames,
        view_available=np.zeros(len(frames), bool),
        frame_mask_sha256=np.full(len(frames), b"0" * 64, "S64"),
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, **d)
    return stream.getvalue()


def test_masks_captured_once_sha_bound_readonly_all13_and_no_decode(monkeypatch, tmp_path):
    frames = np.arange(89640, 91300, dtype=np.int64)
    path = tmp_path / "synthetic.npz"
    payload = packed_fixture(frames)
    path.write_bytes(payload)
    reads = []
    original = Path.read_bytes

    def read_bytes(p):
        reads.append(p)
        return original(p)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        render.prior.prepare_sequence,
        "decode_frame_masks",
        lambda *a: pytest.fail("capture must not decode individual frames"),
    )
    arrays, record = render.capture_masks(path, hashlib.sha256(payload).hexdigest(), frames)
    assert reads == [path] and len(arrays) == len(record["arrays"]) == 13
    assert record["payload_reads"] == record["payload_decodes"] == 1
    assert all(not value.flags.writeable for value in arrays.values())
    assert record["file_stat"] == render.observations.file_stat(path)
    with pytest.raises(ValueError, match="payload changed"):
        render.capture_masks(path, "f" * 64, frames)
    with pytest.raises(ValueError, match="native extent"):
        render.capture_masks(path, hashlib.sha256(payload).hexdigest(), frames + 1)


def test_exact_requested_native_calls_close_iterators_without_rehash(monkeypatch):
    calls, closed = [], []

    def reader(videos, manifest, start, stop):
        calls.append((start, stop))
        try:
            yield start, dict.fromkeys(render.VIEWS, np.zeros((2, 2, 3), np.uint8))
        finally:
            closed.append(start)

    monkeypatch.setattr(render.prior.raw, "iter_native", reader)
    context = dict(videos={}, manifest={})
    images = [render.native_images(context, frame) for frame in render.FRAMES]
    assert calls == [(f, f + 1) for f in render.FRAMES] and closed == list(render.FRAMES)
    assert sum(len(item) for item in images) == 18


def test_source_relative_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        render.prior.native.checked_relative(tmp_path, "../unregistered/manifest.json")


def canonical_headers(tmp_path):
    canonical = tmp_path / "synthetic_canonical.mp4"
    canonical.write_bytes(b"synthetic fixture; never decoded")
    canonical = canonical.resolve()
    digest = "a" * 64
    expected = dict(path=str(canonical), sha256=digest)
    tracking = dict(inputs={str(canonical): digest})  # No staged hardlink input edge.
    staging = dict(inputs={"video:top_cam": dict(**expected, size=35, mtime="synthetic")})
    binding = dict(detector_inputs={"video:top_cam": expected})
    link = dict(source_path=str(canonical), sha256=digest)
    return tracking, staging, binding, link


def test_canonical_only_tracking_edge_and_rich_staging_input(tmp_path):
    tracking, staging, binding, link = canonical_headers(tmp_path)
    render.canonical_video_edge(tracking, staging, binding, link, "top_cam")
    assert list(tracking["inputs"]) == [link["source_path"]]


@pytest.mark.parametrize("change", ["canonical_edge", "staging_sha", "detector_path"])
def test_canonical_video_chain_tampering_rejected(tmp_path, change):
    tracking, staging, binding, link = canonical_headers(tmp_path)
    if change == "canonical_edge":
        tracking["inputs"] = {}
    elif change == "staging_sha":
        staging["inputs"]["video:top_cam"]["sha256"] = "b" * 64
    else:
        binding["detector_inputs"]["video:top_cam"]["path"] = "another_video.mp4"
    with pytest.raises(ValueError):
        render.canonical_video_edge(tracking, staging, binding, link, "top_cam")
