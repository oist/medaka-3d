"""CPU only: the bucket adapter must match direct execution at its padded shape."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / p) for p in ("work/three_body", "labs/20260909_three_body", "src")]
import bucketed_full_fit_graph as m  # noqa: E402
from renderer import RefractiveFishRenderer  # noqa: E402
from test_full_fit_graph_sequence import exact  # noqa: E402
from test_full_fit_graph_sequence import fixture as original_fixture  # noqa: E402
from tracker import TrackerConfig  # noqa: E402


def fixture(directional=True):
    case = original_fixture(directional)
    case.config = TrackerConfig(**vars(case.config))
    case._public_renderer = object.__new__(RefractiveFishRenderer)
    case._public_renderer.__dict__ = case._renderer.__dict__.copy()
    return case


class CPUFullGraph(m.FullFitGraph):
    def __init__(self, *args, directional=True):
        self._initialize(*args, directional)
        self.replays = 0

    def replay(self, pose):
        self._validate_pose(pose)
        with torch.no_grad():
            self._start.copy_(pose)
        self._operation()
        self.replays += 1
        return self


def padded_reference(case):
    """Build the 16-row oracle independently of the adapter's padding helpers.

    Different batch shapes can select different floating-point kernels. Compare
    every score/update byte-for-byte at the same shape; this does not qualify
    unpadded-vs-padded numerical or real-video identity parity.
    """
    batch = len(case._start)
    rows = torch.tensor([*range(batch), *([0] * (16 - batch))])
    inputs = dataclasses.replace(
        case._inputs,
        origins=case._inputs.origins[rows],
        directions=case._inputs.directions[rows],
        held=case._inputs.held[rows],
        predicted=case._inputs.predicted[rows],
        motion_sigma=case._inputs.motion_sigma[rows],
    )
    reference = CPUFullGraph(
        case.config,
        case._start[rows],
        inputs,
        case._public_renderer,
        case._axes[rows],
        case._parallel[rows],
        case._enabled[rows],
        case._lower,
        case._upper,
        directional=case.directional,
    )
    reference._operation()
    return reference


def configure(proxy, case):
    proxy.configure_frame(
        case._inputs,
        case._public_renderer,
        case._axes,
        case._parallel,
        case._enabled,
        case._lower,
        case._upper,
    )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU tests must not construct CUDA graphs")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("directional", [False, True])
def test_all15_objectives_and14_updates_match_direct_padded_reference(directional):
    case = fixture(directional)
    before = case._start.clone()
    case._operation()
    reference = padded_reference(case)
    batch = len(before)
    precision = torch.finfo(before.dtype)
    # Keep the original cross-shape numerical check. Cancellation near zero
    # needs an absolute error budget scaled to the tensor's FP64 magnitude.
    # The adapter-vs-direct-padded comparison below remains byte-exact.
    for key in case.trace:
        scale = max(float(case.trace[key].abs().max()), precision.tiny)
        torch.testing.assert_close(
            reference.trace[key][:, :batch],
            case.trace[key],
            rtol=32 * precision.eps,
            atol=32 * precision.eps * scale,
        )
    exact(reference.best_iteration[:batch], case.best_iteration)
    proxy = m.BucketGraphProxy(case.config, directional, graph_factory=CPUFullGraph)
    configure(proxy, case)
    out = proxy.replay(case._start)
    for key in reference.trace:
        exact(out.trace[key], reference.trace[key][:, :batch])
    for key in (
        "best_pose",
        "best_score",
        "best_iteration",
        "initial_score",
        "initial_parts",
    ):
        exact(getattr(out, key), getattr(reference, key)[:batch])
    for key in ("score_ok", "gradient_ok", "pose_ok"):
        exact(getattr(out, key), getattr(reference, key))
    exact(before, case._start)
    assert out.best_pose.shape == (2, 3, 3, 3)
    assert proxy.statistics["padded_candidates"] == 16
    assert proxy.statistics["real_candidates"] == 2


def test_shared_broadcast_fields_not_padded_when_batch_one():
    case = fixture()
    inputs = dataclasses.replace(
        case._inputs, **{key: getattr(case._inputs, key)[:1] for key in m.BATCH_FIELDS}
    )
    padded = m.pad_inputs(inputs, 1, 16)
    for f in dataclasses.fields(inputs):
        original, actual = getattr(inputs, f.name), getattr(padded, f.name)
        if f.name in m.BATCH_FIELDS:
            assert len(actual) == 16
            for row in actual:
                exact(row, original[0])
        else:
            assert actual is original


@pytest.mark.parametrize("bad", [0, 25, True, 2.0, -1])
def test_invalid_batch_rejected(bad):
    with pytest.raises(ValueError):
        m.bucket_for(bad)


def test_bucket_boundaries_and_no_wasted_fourth_bucket():
    assert [m.bucket_for(b) for b in (1, 14, 16, 17, 20, 21, 24)] == [16, 16, 16, 20, 20, 24, 24]
    with pytest.raises(ValueError, match="smallest bucket"):
        m.pad_rows(torch.zeros(14, 3), 14, 20)


def test_hit_reconfigure_resets_moments_and_owns_static_inputs():
    case = fixture()
    proxy = m.BucketGraphProxy(case.config, True, graph_factory=CPUFullGraph)
    configure(proxy, case)
    proxy.replay(case._start)
    cached = next(iter(proxy.cache.values()))
    cached._moment.fill_(1234)
    cached._variance.fill_(4321)
    changed = dataclasses.replace(case._inputs, predicted=case._inputs.predicted + 0.123)
    case.configure_frame(
        changed, case._renderer, case._axes, case._parallel, case._enabled, case._lower, case._upper
    )
    reference = padded_reference(case)
    configure(proxy, case)
    proxy.replay(case._start)
    for key in reference.trace:
        exact(proxy.trace[key], reference.trace[key][:, : len(case._start)])
    changed.predicted.fill_(999)
    assert not (cached._inputs.predicted == 999).any()
    assert proxy.statistics["captures"] == 1 and proxy.statistics["hits"] == 1
    assert proxy.statistics["resident"] == 1 and cached.replays == 2


def test_invalid_reconfigure_discards_active_graph_and_published_views():
    case = fixture()
    proxy = m.BucketGraphProxy(case.config, True, graph_factory=CPUFullGraph)
    configure(proxy, case)
    proxy.replay(case._start)
    invalid = dataclasses.replace(
        case._inputs, motion_sigma=torch.zeros_like(case._inputs.motion_sigma)
    )
    proxy.configure_frame(
        invalid,
        case._public_renderer,
        case._axes,
        case._parallel,
        case._enabled,
        case._lower,
        case._upper,
    )
    with pytest.raises(ValueError, match="positive scales"):
        proxy.replay(case._start)
    assert proxy.pending is None and not proxy.cache
    assert not hasattr(proxy, "best_pose") and not hasattr(proxy, "trace")
    assert proxy.statistics["failures"] == 1


def test_configure_replay_must_alternate():
    case = fixture()
    proxy = m.BucketGraphProxy(case.config, True, graph_factory=CPUFullGraph)
    with pytest.raises(RuntimeError):
        proxy.replay(case._start)
    configure(proxy, case)
    with pytest.raises(RuntimeError):
        configure(proxy, case)


def test_public_parent_receives_only_actual_prefix(monkeypatch):
    # Exercise the production adapter's parent interface without real CUDA.
    from types import SimpleNamespace

    cfg = SimpleNamespace()
    fitter = object.__new__(m.BucketedFullFitDirectionalDeviceFitter)
    fitter.device, fitter.config, fitter.directional = torch.device("cuda"), cfg, True
    key = (14, 1024, cfg, True)
    monkeypatch.setattr(m, "parent_cache_key", lambda *a: key)
    called = []

    def fake_parent(self, proposals, states, rays, targets):
        assert self._full_fit_key is key
        assert isinstance(self._full_fit_graph, m.BucketGraphProxy)
        called.append((proposals, states, rays, targets))
        return "unchanged parent output"

    monkeypatch.setattr(m.FrozenFitter, "_fit", fake_parent)
    args = ([object()] * 14, object(), object(), object())
    assert fitter._fit(*args) == "unchanged parent output"
    assert called == [args] and not fitter._bucket_inflight
