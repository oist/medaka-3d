"""Synthetic CPU concurrency parity; no saved images, fit, tracker, or GPU."""

import ast
import dataclasses
import importlib.util
import inspect
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(REPO / "labs/20260909_three_body")]
import parallel_flow_views as parallel  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "_parallel_original_motion_fixture", REPO / "tests/test_three_body_motion.py"
)
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


@pytest.fixture
def scene():
    return fixtures.textured_translation.__wrapped__()


def check_all_fields(actual, expected):
    assert type(actual) is type(expected) is parallel.frozen.FlowObservations
    fields = dataclasses.fields(expected)
    assert len(fields) == 15
    for field in fields:
        a, b = np.asarray(getattr(actual, field.name)), np.asarray(getattr(expected, field.name))
        assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes()), field.name
        if a.ndim:
            assert not a.flags.writeable


@pytest.mark.parametrize(
    "case",
    [
        "gray",
        "BGR",
        "blank",
        "None",
        "empty",
        "current_empty",
        "overlap",
        "inconsistent_view",
        "nonfinite_pose",
    ],
)
def test_frozen_exact_all15_and_readonly_inputs(scene, case, monkeypatch):
    rig, pose, _, previous, current = scene
    views = parallel.VIEWS
    if case == "BGR":
        for frame in (previous, current):
            frame.images_bgr = {
                v: cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) for v, a in frame.images_bgr.items()
            }
    elif case == "blank":
        for frame in (previous, current):
            frame.images_bgr = {v: np.full_like(a, 30) for v, a in frame.images_bgr.items()}
    elif case == "None":
        previous.masks = dict.fromkeys(views)
    elif case == "empty":
        previous.masks = {v: np.zeros((0, 200, 200), bool) for v in views}
    elif case == "current_empty":
        current.masks = {v: np.zeros_like(a) for v, a in current.masks.items()}
    elif case == "overlap":
        pose[1] = pose[0]
    elif case == "inconsistent_view":
        view = views[2]
        transform = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 7.0]])
        current.images_bgr[view] = cv2.warpAffine(
            previous.images_bgr[view], transform, (200, 200), borderValue=15
        )
        current.masks[view] = np.stack(
            [
                cv2.warpAffine(
                    a.astype(np.uint8), transform, (200, 200), flags=cv2.INTER_NEAREST
                ).astype(bool)
                for a in previous.masks[view]
            ]
        )
    elif case == "nonfinite_pose":
        pose[0] = np.nan
    arrays = [pose]
    for frame in (previous, current):
        arrays.extend(frame.images_bgr.values())
        arrays.extend(a for a in frame.masks.values() if a is not None)
    before = [(a.shape, a.dtype, a.tobytes()) for a in arrays]
    for a in arrays:
        a.setflags(write=False)
    threads = cv2.getNumThreads()

    def forbidden(*args, **kwargs):
        raise AssertionError("global OpenCV setting changed")

    monkeypatch.setattr(cv2, "setNumThreads", forbidden)
    monkeypatch.setattr(cv2, "setRNGSeed", forbidden)
    expected = parallel.frozen.FlowObservations.from_frames(previous, current, rig, pose)
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = parallel.from_frames_parallel(previous, current, rig, pose, executor=executor)
        check_all_fields(actual, expected)
        again = parallel.from_frames_parallel(previous, current, rig, pose, executor=executor)
        check_all_fields(again, expected)
        assert executor.submit(lambda: 7).result() == 7
    assert cv2.getNumThreads() == threads
    assert before == [(a.shape, a.dtype, a.tobytes()) for a in arrays]
    for field in dataclasses.fields(actual):
        output = np.asarray(getattr(actual, field.name))
        assert not any(np.shares_memory(output, a) for a in arrays)


def test_parallel_completion_order_does_not_change_view_slots(scene, monkeypatch):
    rig, pose, _, previous, current = scene
    expected = parallel.frozen.FlowObservations.from_frames(previous, current, rig, pose)
    real = parallel._view
    barrier = threading.Barrier(3, timeout=10)
    finished = [threading.Event() for _ in range(3)]
    order = []

    def reversed_completion(*args):
        view = args[3]
        vi = parallel.VIEWS.index(view)
        result = real(*args)
        barrier.wait()
        if vi < 2:
            assert finished[vi + 1].wait(10)
        order.append(vi)
        finished[vi].set()
        return result

    monkeypatch.setattr(parallel, "_view", reversed_completion)
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = parallel.from_frames_parallel(previous, current, rig, pose, executor=executor)
    assert order == [2, 1, 0]
    check_all_fields(actual, expected)


def test_first_view_exception_after_all_other_workers_finish(scene, monkeypatch):
    rig, pose, _, previous, current = scene
    barrier = threading.Barrier(3, timeout=10)
    finished = [threading.Event() for _ in range(3)]

    def failing(*args):
        vi = parallel.VIEWS.index(args[3])
        try:
            barrier.wait()
            if vi == 0:
                assert finished[1].wait(10) and finished[2].wait(10)
                raise ValueError("top failed last")
            if vi == 1:
                raise ValueError("side1 failed first")
            return None
        finally:
            finished[vi].set()

    monkeypatch.setattr(parallel, "_view", failing)
    with ThreadPoolExecutor(max_workers=3) as executor:
        with pytest.raises(ValueError, match="top failed last"):
            parallel.from_frames_parallel(previous, current, rig, pose, executor=executor)
        assert all(e.is_set() for e in finished)
        assert executor.submit(lambda: 9).result() == 9


def test_partial_submission_failure_drains_already_started_workers(scene, monkeypatch):
    rig, pose, _, previous, current = scene
    release = threading.Event()
    finished = []

    def worker(*args):
        assert release.wait(10)
        finished.append(args[3])

    monkeypatch.setattr(parallel, "_view", worker)
    with ThreadPoolExecutor(max_workers=3) as pool:

        class RejectThird:
            calls = 0

            def submit(self, *args):
                self.calls += 1
                if self.calls == 3:
                    release.set()
                    raise RuntimeError("executor rejected third")
                return pool.submit(*args)

        with pytest.raises(RuntimeError, match="executor rejected third"):
            parallel.from_frames_parallel(previous, current, rig, pose, executor=RejectThird())
        assert set(finished) == set(parallel.VIEWS[:2])


@pytest.mark.parametrize("invalid", ["gap", "shape"])
def test_invalid_header_never_submits_work(scene, invalid):
    rig, pose, _, previous, current = scene
    if invalid == "gap":
        current.frame_idx += 1
    else:
        pose = pose[:, :2]

    class ForbiddenExecutor:
        def submit(self, *args):
            raise AssertionError("invalid header submitted a worker")

    with pytest.raises(ValueError):
        parallel.from_frames_parallel(previous, current, rig, pose, executor=ForbiddenExecutor())


def test_failed_nonfinite_lk_points_remain_neutral_in_all15(scene, monkeypatch):
    rig, pose, _, previous, current = scene

    def failed_forward(before, after, points, *args, **kwargs):
        assert np.isfinite(points).all()
        return (
            np.full_like(points, np.nan),
            np.ones((len(points), 1), np.uint8),
            np.zeros((len(points), 1), np.float32),
        )

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", failed_forward)
    expected = parallel.frozen.FlowObservations.from_frames(previous, current, rig, pose)
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = parallel.from_frames_parallel(previous, current, rig, pose, executor=executor)
    check_all_fields(actual, expected)
    assert not actual.node_valid.any()


def test_post_view_fusion_ast_is_the_frozen_serial_computation():
    def tail(function):
        parsed = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
        start = next(
            i
            for i, n in enumerate(parsed)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "valid"
        )
        return [ast.dump(node, include_attributes=False) for node in parsed[start:-1]]

    assert tail(parallel.from_frames_parallel) == tail(parallel.frozen.FlowObservations.from_frames)
