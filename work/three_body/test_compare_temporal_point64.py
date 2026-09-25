"""New orchestration/recording tests only; no real model, inputs or GPU."""

from types import SimpleNamespace

import compare_temporal_point64 as m
import numpy as np
import pytest


class Tensor:
    def __init__(self, a):
        self.a = a
        self.shape, self.ndim = a.shape, a.ndim

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.a.copy())


def raw_output():
    return (
        Tensor(np.zeros((1, 16, 2, 2), np.float32)),
        Tensor(np.full((1, 16, 2), 0.8, np.float32)),
        Tensor(np.full((1, 16, 2), 0.9, np.float32)),
        None,
    )


def test_forward_hook_returns_none_and_preserves_original_objects_and_bytes():
    hook = m.CoreForwardRecorder()
    output = raw_output()
    objects = tuple(id(v) for v in output)
    before = [m.signature(v.a) for v in output[:3]]
    assert hook(None, (), output) is None
    assert objects == tuple(id(v) for v in output)
    assert before == [m.signature(v.a) for v in output[:3]]
    assert all(
        saved is not original for saved, original in zip(hook.records[0], output[:3], strict=True)
    )
    output[1].a[:] = 0
    assert np.all(hook.records[0][1].a == np.float32(0.8))


@pytest.mark.parametrize("bad", [(None,), (*raw_output()[:3], {})])
def test_forward_hook_rejects_changed_core_contract(bad):
    with pytest.raises(ValueError):
        m.CoreForwardRecorder()(None, (), bad)


def public(recorded=False):
    values = raw_output()
    if recorded:
        assert m.CoreForwardRecorder()(None, (), values) is None
    return values[0].a * np.array([1199 / 511, 1199 / 383], np.float32), values[1].a * values[
        2
    ].a > 0.6


def test_fake_official_public_result_identical_with_recording():
    a, b = public(False), public(True)
    for left, right in zip(a, b, strict=True):
        m.same(left, right, "public")


def prefix(hi, n, call=0):
    xy = np.full((hi, n, 2), call, np.float32)
    v, c = np.full((hi, n), 0.8, np.float32), np.full((hi, n), 0.9, np.float32)
    return dict(
        xy=xy,
        model_xy=xy.copy(),
        reported=v * c > 0.6,
        sigmoid_visibility=v,
        sigmoid_confidence=c,
        product=v * c,
    )


class Backend:
    def __init__(self, n, drift=None):
        self.n, self.calls, self.drift, self.initialized = n, [], drift, False

    def initialize(self, frames, queries):
        assert len(frames) == 1 and frames[0] == 0
        self.initialized = True
        self.query_input = queries.copy()
        return None, None

    def predict(self, frames):
        assert self.initialized
        self.calls.append(list(frames))
        i = len(self.calls) - 1
        hi = m.WINDOWS[i][1]
        result = prefix(hi, self.n, i)
        if self.drift:
            self.drift(result, i)
        return result, dict(
            forward_seconds=0.1, input_transfer_seconds=0.0, capture_transfer_seconds=0.0
        )


def test_seven_prefixes_first_final_and_all_queries_are_retained():
    queries = np.array([[0, 1], [2, 3]], np.float32)
    before = queries.copy()
    backend = Backend(2)
    saved, hooks = [], []
    arrays, calls = m.run_online(
        list(range(64)),
        queries,
        backend,
        lambda r, a: saved.append((r, a)),
        before_call=hooks.append,
    )
    assert backend.calls == [list(range(lo, hi)) for lo, hi in m.WINDOWS]
    assert hooks == [-1, *range(7)]
    assert len(saved) == len(calls) == 7
    assert sum(r["output_prefix"] * 2 for r, _ in saved) == 560
    np.testing.assert_array_equal(arrays["first_xy"][:16], 0)
    np.testing.assert_array_equal(arrays["first_xy"][16:24], 1)
    np.testing.assert_array_equal(arrays["final_xy"], 6)
    np.testing.assert_array_equal(arrays["last_value_changed_call"], 6)
    np.testing.assert_array_equal(
        arrays["first_horizon"], np.r_[np.full(16, 15), np.repeat(np.arange(23, 64, 8), 8)]
    )
    m.same(queries, before, "queries")


def test_zero_query_preserves_seven_logical_requests_with_no_backend():
    saved = []
    arrays, calls = m.run_online(
        list(range(64)), np.empty((0, 2), np.float32), None, lambda r, a: saved.append((r, a))
    )
    assert len(calls) == len(saved) == 7
    assert not any(r["forward_called"] for r in calls)
    assert arrays["first_xy"].shape == (64, 0, 2)
    assert arrays["first_horizon"][-1] == 63


@pytest.mark.parametrize(
    "drift",
    [
        lambda r, i: r.__setitem__("xy", r["xy"][:-1]),
        lambda r, i: r.__setitem__("xy", r["xy"].astype(np.float64)),
        lambda r, i: r.__setitem__("reported", ~r["reported"]),
        lambda r, i: r.__setitem__("product", np.zeros_like(r["product"])),
        lambda r, i: r.__setitem__("extra", np.zeros(1)),
    ],
)
def test_online_contract_drift_stops_without_resampling(drift):
    backend, saved = Backend(2, drift), []
    with pytest.raises(ValueError):
        m.run_online(
            list(range(64)), np.zeros((2, 2), np.float32), backend, lambda *r: saved.append(r)
        )
    assert len(backend.calls) == 1 and saved == []


def test_future_return_does_not_overwrite_first_nan_bits_or_false_flags():
    special = np.asarray([0x7FC00123], np.uint32).view(np.float32)[0]

    def drift(r, i):
        if i == 0:
            r["xy"][0, 0, 0] = special
            for key in ("sigmoid_visibility", "sigmoid_confidence", "product"):
                r[key][0, 0] = special
            r["reported"][0, 0] = False

    result, _ = m.run_online(
        list(range(64)), np.zeros((1, 2), np.float32), Backend(1, drift), lambda *a: None
    )
    assert result["first_xy"][0, 0, 0].view(np.uint32) == 0x7FC00123
    assert not result["first_visibility"][0, 0]
    assert result["final_visibility"][0, 0]


def test_gate_precedes_filesystem_and_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    monkeypatch.setattr(m, "load_inputs", lambda *a: pytest.fail("input read before draft gate"))
    monkeypatch.setattr(m, "gpu_memory", lambda: pytest.fail("GPU query before gate"))
    destination = tmp_path / "never"
    with pytest.raises(ValueError, match="draft"):
        m.run(
            SimpleNamespace(
                execute=True, output=destination, device="cuda", preparation=m.PREPARATION
            )
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "sample,startup", [({"free_MiB": 6143}, True), ({"free_MiB": 2047}, False)]
)
def test_memory_rejection_is_a_stop_not_a_smaller_query_retry(sample, startup):
    with pytest.raises(ValueError):
        m.check_memory(sample, startup)


def test_output_guard_keeps_failed_partial_bytes(tmp_path):
    writer = m.Output(tmp_path / "new")
    writer.write("partial.npz", {"x": np.arange(3)})
    writer.write("partial.json", {"flag": False, "count": 0, "missing": None})
    writer.final_guard()
    (writer.root / "partial.json").write_bytes(b"altered")
    with pytest.raises(ValueError):
        writer.final_guard()
    assert (writer.root / "partial.npz").exists()


def test_paired_join_preserves_same_query_missing_and_primary63_denominator():
    flags = np.ones((64, 2), bool)
    a = dict(xy=np.zeros((64, 2, 2), np.float32), finite=flags.copy(), reported=flags.copy())
    b = {k: v.copy() for k, v in a.items()}
    for name in ("initial_mad", "initial_zncc", "adjacent_mad", "adjacent_zncc"):
        a[name], b[name] = np.zeros((64, 2)), np.ones((64, 2))
    b["reported"][10, 1] = False
    values = m.comparison_arrays(a, b)
    assert values["initial_mad__b_minus_a"].shape == (64, 2)
    assert np.isnan(values["initial_mad__b_minus_a"][10, 1])
    assert values["initial_mad__a_only"][10, 1]
    assert m.finite_distribution(values["initial_mad__b_minus_a"][1:])["denominator"] == 126


def test_initial_patch_uses_saved_query_even_when_ct_frame0_is_refined():
    image = np.repeat(np.tile(np.arange(20, dtype=np.uint8), (20, 1))[..., None], 3, axis=2)
    frames = [image] * 64
    masks = [np.ones((1, 20, 20), bool)] * 64
    query = np.array([[8, 8]], np.float32)
    original = np.broadcast_to(query, (64, 1, 2)).copy()
    refined = original.copy()
    refined[0, 0, 0] += 1
    refined[1, 0] = 0
    before = refined.copy()
    arguments = (frames, masks, query, np.zeros(1, np.int64), [{}])
    raw = m.diagnose_method(*arguments, original, np.ones((64, 1), bool), lambda *a: None)
    changed = m.diagnose_method(*arguments, refined, np.ones((64, 1), bool), lambda *a: None)
    assert raw["initial_mad"][0, 0] == 0
    assert changed["initial_mad"][0, 0] == pytest.approx(1 / 255)
    assert np.isnan(changed["initial_mad"][1, 0]) and not changed["patch_inside"][1, 0]
    m.same(refined, before, "model's original coordinates")
