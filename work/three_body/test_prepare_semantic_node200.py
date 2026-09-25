"""Finite transport/qualification wiring fixtures; no real media/model/SL input."""

import copy

import cv2
import numpy as np
import prepare_semantic_node200 as m
import pytest


class Capture:
    def __init__(self, images):
        self.images = images
        self.pos = 0
        self.reads = 0
        self.settings = {
            cv2.CAP_PROP_FPS: 166,
            cv2.CAP_PROP_FRAME_COUNT: len(images),
            cv2.CAP_PROP_FRAME_HEIGHT: images[0].shape[0],
            cv2.CAP_PROP_FRAME_WIDTH: images[0].shape[1],
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"FFV1"),
        }

    def isOpened(self):
        return True

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_FRAMES
        self.pos = int(value)
        return True

    def get(self, key):
        return self.pos if key == cv2.CAP_PROP_POS_FRAMES else self.settings[key]

    def read(self):
        image = self.images[self.pos]
        self.pos += 1
        self.reads += 1
        return True, image


@pytest.fixture
def clip(monkeypatch):
    monkeypatch.setattr(m, "FRAMES", tuple(range(m.START, m.START + 3)))
    images = [np.full((2, 3, 3), i, np.uint8) for i in range(3)]
    return Capture(images), [m.signature(x)["sha256"] for x in images]


def test_exact_clip_offsets_no_201st_read(clip):
    capture, hashes = clip
    rows = list(m.read200(capture, hashes, shape=(2, 3)))
    assert [r[0] for r in rows] == list(m.FRAMES)
    assert capture.reads == 3 and capture.pos == 3
    assert all(row[1] is image for row, image in zip(rows, capture.images, strict=True))


@pytest.mark.parametrize(
    "key,value",
    [
        (cv2.CAP_PROP_FRAME_COUNT, 348600),
        (cv2.CAP_PROP_FRAME_COUNT, 3.1),
        (cv2.CAP_PROP_FPS, 165),
        (cv2.CAP_PROP_FRAME_HEIGHT, 3),
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264")),
    ],
)
def test_transport_headers_rejected_before_read(clip, key, value):
    capture, hashes = clip
    capture.settings[key] = value
    with pytest.raises(ValueError):
        list(m.read200(capture, hashes, shape=(2, 3)))
    assert capture.reads == 0


def test_wrong_native_pixel_hash_stops_at_bad_frame(clip):
    capture, hashes = clip
    hashes[1] = "0" * 64
    with pytest.raises(ValueError, match="BGR"):
        list(m.read200(capture, hashes, shape=(2, 3)))
    assert capture.reads == 2


def test_wrong_dtype_does_not_silently_convert(clip):
    capture, hashes = clip
    capture.images[0] = capture.images[0].astype(np.float32)
    with pytest.raises(ValueError, match="dtype"):
        list(m.read200(capture, hashes, shape=(2, 3)))


@pytest.fixture
def qualification():
    sources = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    observed = np.ones((3, 3), bool)
    source = dict(
        frame_idx=np.asarray(m.START, np.int64),
        sources=sources,
        observed=observed,
        ray_distance_mm=np.full((3, 3, 3), 0.5),
        pixel_error_px=np.ones((3, 3, 3)),
    )
    for vi, view in enumerate(m.VIEWS):
        source[view + "_points"] = np.arange(18, dtype=np.float32).reshape(3, 3, 2) + 20 + vi
    # Duplicate pixels are different semantic nodes and must retain separate rows.
    source[m.VIEWS[0] + "_points"][0, 1] = source[m.VIEWS[0] + "_points"][0, 0]
    selected = dict(
        frame_idx=np.arange(m.START, m.START + 64, dtype=np.int64),
        sources=np.broadcast_to(sources, (64, 3, 3, 3)).copy(),
        observed=np.broadcast_to(observed, (64, 3, 3)).copy(),
    )
    selected.update({"fixture_" + str(i): np.zeros(64) for i in range(18)})
    origin = {"selected_array_signatures": {k: m.signature(v) for k, v in selected.items()}}
    rows, sets = [], []
    for b in range(3):
        for n in range(3):
            for view in m.VIEWS:
                xy = source[view + "_points"][b, n]
                rows.append(
                    dict(
                        window="known_gap",
                        native=m.START,
                        body=b,
                        node=n,
                        view=view,
                        source_row=b,
                        observed=True,
                        eligible=True,
                        source_array_key=view + "_points",
                        source_array_indices=[b, n],
                        source_xy_dtype=xy.dtype.str,
                        source_xy_sha256=m.signature(xy)["sha256"],
                        source_xy=xy.tolist(),
                    )
                )
            sets.append(
                dict(
                    window="known_gap",
                    body=b,
                    node=n,
                    qualified_views=list(m.VIEWS),
                    count=3,
                    at_least_two=True,
                )
            )
    return source, selected, origin, rows, sets


def test_initial27_original_pixels_labels_duplicates_and_inputs_unchanged(qualification):
    source, selected, *_ = qualification
    before = {k: m.signature(v) for k, v in source.items()}
    before_selected = {k: m.signature(v) for k, v in selected.items()}
    result, rows, sets = m.initial_queries(*qualification)
    assert len(rows) == len(result["slots"]) == 27
    assert len(sets) == len(result["node_sets"]) == 9
    for view, a in result["views"].items():
        assert len(a) == 7 and len(a["xy"]) == 9
        assert a["xy"].dtype == np.float32
        np.testing.assert_array_equal(a["query_index"], np.arange(9, dtype=np.int32))
        for i in range(9):
            b, n = divmod(i, 3)
            assert a["body_slot"][i] == b and a["node_index"][i] == n
            assert a["source_row"][i] == b
            assert m.signature(a["xy"][i]) == m.signature(source[view + "_points"][b, n])
    assert result["views"][m.VIEWS[0]]["duplicate_pixel"][:2].all()
    assert before == {k: m.signature(v) for k, v in source.items()}
    assert before_selected == {k: m.signature(v) for k, v in selected.items()}


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda q: q[0][m.VIEWS[0] + "_points"].__setitem__((0, 0, 0), 99), "raw SL"),
        (lambda q: q[0]["sources"].__setitem__((0, 0, 0), 1), "source/selected"),
        (lambda q: q[1]["frame_idx"].__setitem__(0, m.START + 1), "arrays differ"),
        (lambda q: q[0]["ray_distance_mm"].__setitem__((0, 0, 0), 2.01), "strict"),
        (lambda q: q[0]["pixel_error_px"].__setitem__((0, 0, 0), np.nan), "strict"),
        (lambda q: q[3][0].__setitem__("eligible", False), "status"),
        (lambda q: q[4][0].__setitem__("count", 2), "node set"),
    ],
)
def test_qualification_connections_reject_drift(qualification, mutation, reason):
    mutation(qualification)
    with pytest.raises(ValueError, match=reason):
        m.initial_queries(*qualification)


def test_no_qualified_source_does_not_become_projected_query(qualification):
    source, selected, origin, rows, sets = qualification
    source["sources"][0, :, 0] = -1
    source["observed"][0, 0] = False
    selected["sources"][:, 0, :, 0] = -1
    selected["observed"][:, 0, 0] = False
    origin["selected_array_signatures"] = {k: m.signature(v) for k, v in selected.items()}
    for row in rows:
        if row["body"] == 0 and row["node"] == 0:
            row.update(source_row=-1, observed=False, eligible=False)
    sets[0].update(qualified_views=[], count=0, at_least_two=False)
    result, _, _ = m.initial_queries(*qualification)
    assert len(result["slots"]) == 27
    assert [len(result["views"][v]["xy"]) for v in m.VIEWS] == [8] * 3
    assert not result["node_sets"][0]["two_view_query_set_available"]


@pytest.mark.parametrize("decoded", [None, np.empty((0, 1200, 1200), bool)])
def test_observation_unavailable_vs_zero_and_raw_sl_bytes(monkeypatch, decoded):
    data = {"a": np.asarray([4], np.int32)}
    packed = {"a": data["a"].copy()}
    calls = []

    def select(data_arg, offset, frame, count):
        calls.append((offset, frame, count))
        assert data_arg is data
        return packed, decoded

    monkeypatch.setattr(m.native, "select_packed_frame", select)
    points = np.array([[[0.0, -0.0], [np.nan, 1.0], [2.0, 3.0]]], np.float32)
    before = m.signature(points)
    result = m.observation(data, m.START + 199, 0, points)
    assert calls == [(199, m.START + 199, 0)]
    assert bool(result["mask_available"]) == (decoded is not None)
    assert result["masks"].shape == (0, 1200, 1200)
    assert m.signature(result["detections"]) == before == m.signature(points)
    assert result["detections"] is not points
    assert result["packed_a"] is packed["a"]


@pytest.fixture
def seed_inputs():
    pose = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    seed = dict(
        frame=m.START,
        initialization_count=1,
        later_baseline_input=False,
        pose_xyz=pose.tolist(),
        source=[[0, 1, 2]] * 3,
        limitation="optimistic offline initial seed",
    )
    ctx = dict(
        initial_pose=m.signature(pose),
        initial_lengths=m.signature(np.linalg.norm(np.diff(pose, axis=1), axis=-1)),
        views=list(m.VIEWS),
        world_units="mm",
        implementation="AcceleratedContinuousPersistentNodeTracker",
    )
    metadata = dict(
        context=ctx,
        initialize_count=1,
        restore_count=0,
        processed_frames=200,
        previous_frame=m.STOP - 1,
    )
    summary = dict(
        original_candidate_21_arrays_byte_exact=True,
        original_candidate_loaded_only_after_full_inference=True,
    )
    origin = dict(
        reference=dict(
            seal=m.REFERENCE_SHA,
            metadata=m.METADATA_SHA,
            start=m.START,
            stop=m.STOP,
            candidate="a" * 64,
        ),
        retrospective_reference=True,
    )
    return seed, metadata, summary, origin


def test_original_seed_contract_keeps_retrospective_labels_distinct(seed_inputs):
    before = copy.deepcopy(seed_inputs)
    result = m.seed_contract(*seed_inputs)
    assert result["retrospective_conditional_body_labels"]
    assert not result["later_reference_poses_used_as_tracker_seeds"]
    assert not result["biological_ID_truth"]
    assert seed_inputs == before


def test_reordered_seed_slots_rejected_by_original_context(seed_inputs):
    seed_inputs[0]["pose_xyz"].reverse()
    with pytest.raises(ValueError, match="seed/context"):
        m.seed_contract(*seed_inputs)


def test_new_output_guard_precedes_any_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "metadata_inputs", lambda _: pytest.fail("unexpected input work"))
    with pytest.raises(ValueError, match="new workspace output"):
        m.run(tmp_path)
