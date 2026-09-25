"""Synthetic flow/checkpoint fields only; no real histories, raw or GPU."""

import copy
import dataclasses

import long_final_boundary_observations as m
import numpy as np
import pytest
from test_long_final_native_spool import sample


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, digest=None):
        actual = m.spool.sha(path)
        assert digest is None or actual == digest
        self.paths[str(path)] = actual
        return actual


class Reader:
    def read_view(self, n, v):
        return sample(n, v, "missing" if v == m.VIEWS[0] else "empty")[1]


def backgrounds():
    return {v: np.zeros((4, 5, 3), np.uint8) for v in m.VIEWS}


def test_fixed33_stops_restore32_reader147():
    stops = m.endpoint_stops()
    assert (
        len(stops) == 33
        and stops[:5] == (51460, 53120, 59760, 66400, 76360)
        and stops[-2:] == (345280, 348600)
    )
    assert set(stops[:-1]) <= set(m.spool.boundaries())
    assert all(n - 1 in m.spool.retained_frames() for n in stops)


def flow_setup(monkeypatch, fault=None):
    first = m.spool.FIRST
    state = dict(
        history_offsets=np.array([0, 6, 12, 18], np.int64),
        history_frame_idx=np.repeat(np.arange(first, first + 3, dtype=np.int64), 6),
        history_pose_xyz=np.arange(18 * 27, dtype=np.float64).reshape(18, 3, 3, 3),
        history_proposal_flow_nodes=np.zeros((18, 3, 3), bool),
        history_proposal_flow_parent=np.zeros(18, np.int64),
    )
    events = []
    threads = [7]
    monkeypatch.setattr(m.cv2, "getNumThreads", lambda: threads[0])

    def set_threads(n):
        threads[0] = n
        events.append(("threads", n))

    monkeypatch.setattr(m.cv2, "setNumThreads", set_threads)

    @dataclasses.dataclass
    class Flow:
        previous_frame_idx: int
        frame_idx: int
        node_valid: np.ndarray
        pose_xyz: np.ndarray

    def factory(a, b, rig, pose, config):
        events.append(("flow", a.frame_idx, b.frame_idx))
        assert threads[0] == 2
        np.testing.assert_array_equal(pose, state["history_pose_xyz"][0])
        assert a.masks[m.VIEWS[0]] is None and a.masks[m.VIEWS[1]].shape == (0, 4, 5)
        if fault == "raises":
            raise RuntimeError("synthetic flow failure")
        if fault == "mutation":
            a.images_bgr[m.VIEWS[0]][0, 0, 0] += 1
        return Flow(a.frame_idx, b.frame_idx, np.zeros((3, 3), bool), np.full((3, 3, 3), np.nan))

    monkeypatch.setattr(m.FlowObservations, "from_frames", factory)
    if fault == "sixth_flags":
        state["history_proposal_flow_nodes"][11, 2, 2] = True
    if fault == "parent":
        state["history_proposal_flow_parent"][10] = 1
    if fault == "offset":
        state["history_offsets"][2] = 11
    if fault == "native":
        state["history_frame_idx"][7] += 1
    return state, events, threads


def test_original_flow_signature_first_rank_not_backtrace_all6_nonmutation_and_threads(monkeypatch):
    state, events, threads = flow_setup(monkeypatch)
    before = m.spool.signatures(state)
    saved = []
    result = m._flow_checks(
        Reader(),
        state,
        object(),
        backgrounds(),
        dataclasses.asdict(m.FlowConfig()),
        2,
        m.spool.FIRST,
        [m.spool.FIRST + 1],
        {m.spool.FIRST + 1},
        lambda r, a: saved.append((r, a)),
    )
    assert m.spool.signatures(state) == before and threads[0] == 7
    assert events == [("threads", 2), ("flow", m.spool.FIRST, m.spool.FIRST + 1), ("threads", 7)]
    assert (
        result[0]["previous_first_history_row"] == 0 and len(result[0]["six_flow_flags_exact"]) == 6
    )
    assert result[0]["boundary_kind"] == "restore" and saved[0][1]["node_valid"].shape == (3, 3)


@pytest.mark.parametrize(
    "fault", ["raises", "mutation", "sixth_flags", "parent", "offset", "native"]
)
def test_flow_failure_restores_threads_and_rejects_any_sixth_history_drift(monkeypatch, fault):
    state, events, threads = flow_setup(monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError)):
        m._flow_checks(
            Reader(),
            state,
            object(),
            backgrounds(),
            dataclasses.asdict(m.FlowConfig()),
            2,
            m.spool.FIRST,
            [m.spool.FIRST + 1],
            set(),
        )
    assert threads[0] == 7 and events[-1] == ("threads", 7)


def checkpoint(root):
    stop = m.spool.FIRST + 3
    frame = m.frame_from_spool(Reader(), stop - 1, backgrounds())
    arrays = {"history_forbidden": np.arange(4)}
    presence = {}
    for c in m.CATEGORIES:
        for v in m.VIEWS:
            key = "previous_" + c + "_" + v
            value = getattr(frame, c)[v]
            presence[key] = value is not None
            if value is not None:
                arrays[key] = value.copy()
    context = {"tracker_config": {"flow_proposals": True}, "fixture": "synthetic"}
    meta = dict(
        previous_frame=stop - 1,
        processed_frames=3,
        initialize_count=1,
        restore_count=0,
        context=context,
        previous_observation=presence,
    )
    binding = {"fixture": "synthetic"}
    manifest = m.codec.write_checkpoint(root, arrays, meta, bindings=binding)
    ref = dict(
        checkpoint_root=str(root),
        stop=stop,
        restore_count=0,
        manifest_sha256=m.spool.sha(root / "manifest.json"),
        bindings=binding,
    )
    return ref, arrays, meta, manifest


def test_previous_loader_reads_only_present11_members_never_history(tmp_path):
    ref, arrays, meta, manifest = checkpoint(tmp_path / "checkpoint")
    opened = []
    inverse = {r["storage_key"]: k for k, r in manifest["arrays"].items()}

    class Archive:
        def __init__(self, path, allow_pickle):
            assert allow_pickle is False
            self.inner = np.load(path, allow_pickle=False)
            self.files = self.inner.files

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.inner.close()

        def __getitem__(self, key):
            assert inverse[key].startswith("previous_")
            opened.append(inverse[key])
            return self.inner[key]

    result, actual, proof = m.load_previous_members(
        ref, Evidence(), meta["context"], archive_factory=Archive
    )
    assert actual == meta and set(result) == set(opened) == {
        k for k in arrays if k.startswith("previous_")
    }
    assert len(opened) == 11 and proof["historical_histories_loaded"] is False
    checks = m.compare_previous(Reader(), ref, result, actual, backgrounds())
    assert checks["checks"]["previous_masks_" + m.VIEWS[0]] == {"present": False}
    for k, v in result.items():
        assert m.spool.signature(v) == m.spool.signature(arrays[k])


@pytest.mark.parametrize("fault", ["binding", "context", "stop", "restore", "member", "presence"])
def test_previous_identity_presence_or_bytes_rejected(tmp_path, fault):
    ref, arrays, meta, _manifest = checkpoint(tmp_path / "checkpoint")
    context = copy.deepcopy(meta["context"])
    if fault == "binding":
        ref["bindings"] = {"fixture": "wrong"}
    elif fault == "context":
        context["fixture"] = "wrong"
    elif fault == "stop":
        ref["stop"] += 1
    elif fault == "restore":
        ref["restore_count"] += 1
    if fault in ("member", "presence"):
        actual = {k: v.copy() for k, v in arrays.items() if k.startswith("previous_")}
        if fault == "member":
            actual["previous_images_bgr_" + m.VIEWS[1]][0, 0, 0] += 1
        else:
            meta["previous_observation"]["previous_masks_" + m.VIEWS[0]] = True
        with pytest.raises(ValueError):
            m.compare_previous(Reader(), ref, actual, meta, backgrounds())
    else:
        with pytest.raises(ValueError):
            m.load_previous_members(ref, Evidence(), context)


def test_background_encoded_vs_decoded_are_separate_and_nonmutating():
    bg = backgrounds()
    before = m.spool.signatures(bg)
    refs = {v: {"path": v + ".png", "sha256": "a" * 64} for v in m.VIEWS}
    binding = m.background_binding(bg, refs)
    assert m.spool.signatures(bg) == before
    assert all(
        binding[v]["encoded_file"] == refs[v]
        and binding[v]["decoded_array"] == m.spool.signature(bg[v])
        for v in m.VIEWS
    )
