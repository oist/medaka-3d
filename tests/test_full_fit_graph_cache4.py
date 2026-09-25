"""LRU ownership around a fake frozen parent; no model, packet or CUDA execution."""

import hashlib
import sys
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import full_fit_graph_cache4 as m  # noqa: E402


@dataclass(frozen=True)
class Config:
    opt_steps: int = 14
    weight: float = 0.2


class ExpectedFailure(RuntimeError):
    pass


def target(n=7, token=1, failure=None):
    return SimpleNamespace(
        pixels={v: np.full((n, 2), token, np.float32) for v in ("top", "side1", "side2")},
        token=token,
        failure=failure,
    )


@pytest.fixture
def fake(monkeypatch):
    """The single-slot parent protocol is faked, never its numerical equations."""
    log = SimpleNamespace(live=weakref.WeakSet(), events=[], allocated=0, peak=0, calls=[])

    class Graph:
        def __init__(self, inputs):
            log.allocated += 1
            self.ident = log.allocated
            log.live.add(self)
            log.peak = max(log.peak, len(log.live))
            self.token = inputs.token
            self.failure = inputs.failure
            self.moment = self.variance = 987
            log.events.append(("capture", self.ident, self.token))

        def configure_frame(self, inputs):
            log.events.append(("configure", self.ident, inputs.token))
            if inputs.failure == "configure":
                raise ExpectedFailure("configure")
            self.token, self.failure = inputs.token, inputs.failure

        def replay(self):
            # In the frozen parent this reset is inside the full replay graph.
            self.moment = self.variance = 0
            log.events.append(("reset", self.ident, self.token))
            if self.failure == "replay":
                raise ExpectedFailure("replay")
            self.moment, self.variance = self.token, self.token**2
            log.events.append(("replay", self.ident, self.token))

    def frozen_fit(self, proposals, states, rays, targets):
        log.calls.append((proposals, states, rays, targets))
        if self.device.type != "cuda":
            return self.cpu_result
        # Independently compute the renderer's padded N, as the frozen parent does.
        count = max(len(targets.pixels.get(v, ())) for v in self.rig.views)
        key = (len(proposals), count, self.config, self.directional)
        if self.__dict__.get("_full_fit_key") != key:
            self.__dict__.pop("_full_fit_graph", None)
            self._full_fit_key = key
            if targets.failure == "capture":
                raise ExpectedFailure("capture")
            self._full_fit_graph = Graph(targets)
            self._full_fit_capture_count = self.__dict__.get("_full_fit_capture_count", 0) + 1
        else:
            self._full_fit_graph.configure_frame(targets)
        self._full_fit_graph.replay()
        if targets.failure == "final_finite":
            raise ExpectedFailure("final_finite")
        if targets.failure == "reentrant":
            self._fit(proposals, states, rays, targets)
        if targets.failure == "parent_key":
            self._full_fit_key = "unexpected parent key"
        self.last_diagnostics = {"token": targets.token}
        self.last_full_fit_trace = {"token": np.asarray([targets.token], np.int64)}
        self.parent_return = tuple(np.asarray([targets.token + i], np.float64) for i in range(4))
        return self.parent_return

    monkeypatch.setattr(m.FrozenFullFitFitter, "_fit", frozen_fit)
    obj = object.__new__(m.Cache4FullFitDirectionalDeviceFitter)
    obj.device = SimpleNamespace(type="cuda")
    obj.rig = SimpleNamespace(views=("top", "side1", "side2"))
    obj.config, obj.directional = Config(), True
    return obj, log


def call(obj, batch=15, *, n=7, token=1, failure=None):
    return obj._fit([object() for _ in range(batch)], object(), object(), target(n, token, failure))


def batches(obj):
    return [k[0] for k in obj._full_fit_cache4]


def test_fixed_eight_fit_sequence_evicts_before_capture_and_configures_mru_hit(fake):
    obj, log = fake
    for i, batch in enumerate((15, 14, 16, 18, 17, 16, 15, 16)):
        call(obj, batch, token=i + 1)
        assert len(log.live) <= 4
    assert batches(obj) == [18, 17, 15, 16]
    assert obj.full_fit_cache_statistics == dict(
        capacity=4,
        captures=6,
        hits=2,
        misses=6,
        evictions=2,
        failures=0,
        resident=4,
        peak_resident=4,
    )
    assert log.peak == 4  # A fifth capture never overlaps the evicted owned graph.
    assert [(e[0], e[2]) for e in log.events if e[0] == "configure"] == [
        ("configure", 6),
        ("configure", 8),
    ]
    assert len([e for e in log.events if e[0] == "reset"]) == 8
    assert len([e for e in log.events if e[0] == "replay"]) == 8
    assert obj._full_fit_graph is obj._full_fit_cache4[next(reversed(obj._full_fit_cache4))]


def test_every_hit_configures_current_inputs_and_parent_outputs_are_returned_untouched(fake):
    obj, log = fake
    a, b = target(token=3), target(token=99)
    proposals, states, rays = [object()], object(), object()
    obj._fit(proposals, states, rays, a)
    result = obj._fit(proposals, states, rays, b)
    assert log.calls[-1] == (proposals, states, rays, b)
    assert result is obj.parent_return
    assert obj.last_diagnostics == {"token": 99}
    np.testing.assert_array_equal(obj.last_full_fit_trace["token"], [99])
    assert obj._full_fit_graph.moment == 99 and obj._full_fit_graph.variance == 99**2
    assert log.events[-3:] == [("configure", 1, 99), ("reset", 1, 99), ("replay", 1, 99)]


@pytest.mark.parametrize("changed", ["batch", "pixels", "config", "directional"])
def test_each_frozen_key_component_requires_distinct_capture(fake, changed):
    obj, _ = fake
    call(obj)
    batch, n = 15, 7
    if changed == "batch":
        batch += 1
    elif changed == "pixels":
        n += 1
    elif changed == "config":
        obj.config = Config(weight=0.3)
    else:
        obj.directional = False
    call(obj, batch, n=n)
    assert obj.full_fit_cache_statistics["captures"] == 2
    assert obj.full_fit_cache_statistics["resident"] == 2


def test_ragged_views_and_missing_view_use_renderer_max_n_not_first_view(fake):
    obj, _ = fake
    t = target(n=2)
    t.pixels = {"top": np.zeros((2, 2)), "side1": np.zeros((9, 2))}
    assert m.parent_cache_key(obj, [1, 2], t) == (2, 9, obj.config, True)
    obj._fit([1, 2], None, None, t)
    t.pixels = {"side2": np.ones((9, 2))}
    obj._fit([1, 2], None, None, t)
    assert obj.full_fit_cache_statistics["hits"] == 1


@pytest.mark.parametrize("failure", ["configure", "replay", "final_finite", "parent_key"])
def test_failed_hit_is_discarded_healthy_other_graphs_survive_and_retry_recaptures(fake, failure):
    obj, _ = fake
    for batch in range(1, 5):
        call(obj, batch)
    healthy = {k: id(v) for k, v in obj._full_fit_cache4.items() if k[0] != 2}
    error = RuntimeError if failure == "parent_key" else ExpectedFailure
    with pytest.raises(error):
        call(obj, 2, failure=failure)
    assert {k: id(v) for k, v in obj._full_fit_cache4.items()} == healthy
    assert "_full_fit_graph" not in obj.__dict__ and "_full_fit_key" not in obj.__dict__
    assert not obj._full_fit_cache4_inflight
    assert obj.last_full_fit_cache4 == dict(status="failed", hit=True, resident=3, evicted=False)
    call(obj, 2, token=12)
    assert obj.full_fit_cache_statistics["captures"] == 5
    assert obj.full_fit_cache_statistics["failures"] == 1
    assert batches(obj) == [1, 3, 4, 2]


def test_failed_capture_clears_dangling_parent_key_and_does_not_restore_evicted_lru(fake):
    obj, log = fake
    for batch in range(1, 5):
        call(obj, batch)
    with pytest.raises(ExpectedFailure, match="capture"):
        call(obj, 5, failure="capture")
    assert batches(obj) == [2, 3, 4]
    assert len(log.live) == 3
    assert "_full_fit_key" not in obj.__dict__ and "_full_fit_graph" not in obj.__dict__
    assert obj.last_full_fit_cache4 == dict(status="failed", hit=False, resident=3, evicted=True)
    call(obj, 5)
    assert obj.full_fit_cache_statistics == dict(
        capacity=4,
        captures=5,
        hits=0,
        misses=6,
        evictions=1,
        failures=1,
        resident=4,
        peak_resident=4,
    )


def test_reentrant_parent_call_rejects_and_outer_failure_cleans_selected_entry(fake):
    obj, _ = fake
    with pytest.raises(RuntimeError, match="non-reentrant"):
        call(obj, failure="reentrant")
    assert obj.full_fit_cache_statistics["resident"] == 0
    assert not obj._full_fit_cache4_inflight
    call(obj)
    assert obj.full_fit_cache_statistics["captures"] == 2


@pytest.mark.parametrize("bad", [None, "empty", "shape"])
def test_bad_pixel_metadata_leaves_existing_graph_and_counters_unchanged(fake, bad):
    obj, _ = fake
    call(obj)
    stats = obj.full_fit_cache_statistics
    active = obj._full_fit_graph
    t = None if bad is None else target(n=0)
    if bad == "shape":
        t.pixels["top"] = np.zeros((2, 3))
    with pytest.raises(ValueError):
        obj._fit([1], None, None, t)
    assert obj.full_fit_cache_statistics == stats
    assert obj._full_fit_graph is active


def test_cpu_directly_delegates_even_without_cuda_key_metadata(fake):
    obj, log = fake
    obj.device = SimpleNamespace(type="cpu")
    obj.cpu_result = object()
    del obj.rig
    assert obj._fit(None, None, None, None) is obj.cpu_result
    assert len(log.calls) == 1 and not log.events
    assert "_full_fit_cache4" not in obj.__dict__
    assert obj.full_fit_cache_statistics["captures"] == 0


def test_policy_statistics_are_host_copies_and_alias_inherits_parent_constructor(fake):
    obj, _ = fake
    assert m.Cache4FullFitDirectionalDeviceFitter is m.FullFitDirectionalDeviceFitter
    assert type(obj).__init__ is m.FrozenFullFitFitter.__init__
    policy, stats = obj.full_fit_cache_policy, obj.full_fit_cache_statistics
    policy["key_fields"].append("invalid")
    stats["capacity"] = 999
    assert obj.full_fit_cache_statistics["capacity"] == 4
    assert len(obj.full_fit_cache_policy["key_fields"]) == 4
    assert (
        hashlib.sha256((REPO / "work/three_body/full_fit_graph_fitter.py").read_bytes()).hexdigest()
        == m.PARENT_SOURCE_SHA
    )


def test_invalid_owned_cache_is_rejected_before_parent_dispatch(fake):
    obj, log = fake
    obj._full_fit_cache4 = OrderedDict((i, object()) for i in range(5))
    with pytest.raises(RuntimeError, match="invalid full-fit LRU4"):
        call(obj)
    assert not log.calls
