"""Tiny CPU arithmetic/lifetime and fake-device cache contracts; no CUDA/packets."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
import fp32_pose_proposals as m  # noqa: E402
import full_fit_graph_fitter as frozen  # noqa: E402
from qualify_fixed_body_width_cuda import synthetic_case  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("FP32 synthetic tests cannot initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **kwargs: None)
    yield
    torch.set_num_threads(before)


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.detach().numpy().tobytes() == b.detach().numpy().tobytes()


def cpu_arguments():
    cfg, pose, inputs, fixed = synthetic_case(0, "cpu")
    cfg.opt_steps, cfg.learning_rate_mm = 14, 0.25
    pixels = {view: fixed.pixels[i, :4].copy() for i, view in enumerate(fixed.views)}
    renderer = RefractiveFishRenderer(fixed.rig, pixels, fixed.config, dtype=torch.float32)
    inputs = dataclasses.replace(inputs, target=inputs.target[:, :4], weight=inputs.weight[:, :4])
    inputs = m.FitInputs(
        **{
            field.name: getattr(inputs, field.name).float().clone()
            for field in dataclasses.fields(inputs)
        }
    )
    axes = torch.zeros((2, 3, 3), dtype=torch.float32)
    axes[..., 0] = 1
    parallel = inputs.motion_sigma * 1.5
    enabled = torch.tensor([[True, False, True], [False, True, True]])
    lower = torch.tensor([0.0, 0.0, -6.0])
    upper = torch.tensor([36.0, 192.0, 186.0])
    return [cfg, pose.float(), inputs, renderer, axes, parallel, enabled, lower, upper]


def cpu_graph(args=None, directional=True):
    args = cpu_arguments() if args is None else args
    graph = object.__new__(m.FP32ProposalGraph)
    graph._initialize(*args, directional)
    return graph


@pytest.mark.parametrize("directional", [False, True])
@pytest.mark.parametrize("foreach", [None, True])
def test_actual_default_adam_all14_updates_and15_scores(directional, foreach):
    graph, scratch = cpu_graph(directional=directional), cpu_graph(directional=directional)
    assert m.FP32ProposalGraph._operation is frozen.FullFitGraph._operation
    assert graph.uses_positive_product_bound is False
    graph._operation()
    graph._check_generated()
    pose = graph._start.clone().requires_grad_(True)
    # None is the real public CPU default; True additionally selects the same
    # foreach family the production CUDA default must select before capture.
    optimizer = torch.optim.Adam([pose], lr=graph.config.learning_rate_mm, foreach=foreach)
    best, scores, iterations = (
        pose.detach().clone(),
        torch.full((2,), torch.inf),
        torch.full((2,), -1, dtype=torch.int64),
    )
    for step in range(15):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scratch._pose.copy_(pose)
        frozen.GraphFit._operation(scratch)
        score, parts, gradient = [
            value.detach().clone()
            for value in (scratch._scores, scratch._parts, scratch._pose.grad)
        ]
        with torch.no_grad():
            difference, correction, center = frozen.center_correction(
                pose[:, :, 1] - scratch._inputs.predicted[:, :, 1],
                scratch._inputs.motion_sigma,
                scratch._axes,
                scratch._parallel,
                scratch._enabled,
            )
            if directional:
                score += scratch.config.motion_weight * difference
            for name, value in (
                ("pose_before", pose),
                ("score", score),
                ("parts", parts),
                ("base_gradient", gradient),
            ):
                exact(graph.trace[name][step], value)
            changed = score < scores
            best[changed], iterations[changed] = pose.detach()[changed], step
            scores = torch.minimum(scores, score)
            if step == 0:
                expected_parts = parts.clone()
                if directional:
                    expected_parts[:, 2] = center
                exact(graph.initial_score, score)
                exact(graph.initial_parts, expected_parts)
            if step == 14:
                break
            pose.grad = gradient.clone()
            if directional:
                pose.grad[:, :, 1] += scratch.config.motion_weight * correction
            exact(graph.trace["update_gradient"][step], pose.grad)
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(scratch._lower, scratch._upper)
        exact(graph.trace["moment"][step], optimizer.state[pose]["exp_avg"])
        exact(graph.trace["variance"][step], optimizer.state[pose]["exp_avg_sq"])
        exact(graph.trace["pose_after"][step], pose)
    exact(graph.best_pose, best)
    exact(graph.best_score, scores)
    exact(graph.best_iteration, iterations)
    assert all(bool(flag) for flag in graph.finite_flags.values())


def test_all_moments_and_best_reset_and_original_inputs_owned():
    args = cpu_arguments()
    graph = cpu_graph(args)
    graph._operation()
    first = graph.snapshot()
    graph._working.fill_(-100)
    graph._moment.fill_(800)
    graph._variance.fill_(900)
    graph.best_score.fill_(-torch.inf)
    graph._operation()
    for name, value in graph.snapshot().items():
        exact(value, first[name])
    args[1].fill_(123)
    args[2].predicted.fill_(234)
    args[4].fill_(345)
    args[3].ray_origins.fill_(456)
    args[3]._fractions.fill_(567)
    assert not bool((graph._start == 123).all())
    assert not bool((graph._inputs.predicted == 234).all())
    assert not bool((graph._axes == 345).all())
    assert not bool((graph._renderer.ray_origins == 456).all())
    assert not bool((graph._renderer._fractions == 567).all())


def test_all_same_key_frame_inputs_update_and_match_fresh():
    graph, fresh = cpu_graph(), cpu_graph()
    inputs = dataclasses.replace(
        graph._inputs,
        predicted=graph._inputs.predicted + 0.05,
        target=1 - graph._inputs.target,
        weight=graph._inputs.weight * 0.75,
        origins=graph._inputs.origins + 0.02,
    )
    renderer = cpu_arguments()[3]
    renderer.ray_origins += 0.03
    axes = graph._axes.roll(1, -1)
    enabled = ~graph._enabled
    parallel = graph._parallel * 1.1
    lower, upper = graph._lower + 0.01, graph._upper - 0.1
    for current in (graph, fresh):
        current.configure_frame(inputs, renderer, axes, parallel, enabled, lower, upper)
        current._operation()
    for name, value in graph.snapshot().items():
        exact(value, fresh.snapshot()[name])
    inputs.predicted.fill_(777)
    renderer.ray_origins.fill_(888)
    enabled.fill_(False)
    assert not bool((graph._inputs.predicted == 777).all())
    assert not bool((graph._renderer.ray_origins == 888).all())
    assert bool(graph._enabled.any())


@pytest.mark.parametrize(
    "fault",
    ["f64", "shape", "empty", "pose_nan", "inputs64", "axes_nan", "bool", "renderer64", "rays64"],
)
def test_initializer_rejects_wrong_dtype_shape_and_nonfinite(fault):
    args = cpu_arguments()
    directional = True
    if fault == "f64":
        args[1] = args[1].double()
    elif fault == "shape":
        args[1] = args[1][:, :, :2]
    elif fault == "empty":
        args[1] = args[1][:0]
    elif fault == "pose_nan":
        args[1][0, 0, 0, 0] = torch.nan
    elif fault == "inputs64":
        args[2] = dataclasses.replace(args[2], predicted=args[2].predicted.double())
    elif fault == "axes_nan":
        args[4][0, 0, 0] = torch.nan
    elif fault == "renderer64":
        args[3].dtype = torch.float64
    elif fault == "rays64":
        args[3].ray_origins = args[3].ray_origins.double()
    else:
        directional = 1
    with pytest.raises(ValueError):
        cpu_graph(args, directional)


def test_rejected_configure_does_not_partially_update_static_buffers():
    graph = cpu_graph()
    before = graph._inputs.predicted.clone()
    inputs = dataclasses.replace(graph._inputs, predicted=graph._inputs.predicted + 1)
    with pytest.raises(ValueError):
        graph.configure_frame(
            inputs,
            graph._renderer,
            graph._axes,
            graph._parallel * torch.nan,
            graph._enabled,
            graph._lower,
            graph._upper,
        )
    exact(before, graph._inputs.predicted)


def test_zero_gradient_ties_keep_original_at_every_iteration(monkeypatch):
    def flat_operation(graph):
        graph._scores = torch.zeros(graph.batch_size, dtype=graph.dtype)
        graph._parts = torch.zeros((graph.batch_size, 5), dtype=graph.dtype)
        graph._pose.grad.zero_()

    graph = cpu_graph(directional=False)
    monkeypatch.setattr(frozen.GraphFit, "_operation", flat_operation)
    graph._operation()
    graph._check_generated()
    exact(graph.best_pose, graph._start)
    assert graph.best_iteration.tolist() == [0, 0]
    assert graph.trace["moment"].count_nonzero() == 0
    assert graph.trace["variance"].count_nonzero() == 0


class CapturedCPUReplay:
    """Simulate overwrite of captured output storage while executing real CPU math."""

    def __init__(self, graph):
        self.graph = graph
        graph._operation()

    def replay(self):
        graph = self.graph
        old_trace = graph.trace
        old_fields = {
            name: getattr(graph, name)
            for name in (
                "best_pose",
                "best_score",
                "best_iteration",
                "initial_score",
                "initial_parts",
                "score_ok",
                "gradient_ok",
                "pose_ok",
            )
        }
        graph._operation()
        for name, old in old_trace.items():
            old.copy_(graph.trace[name])
        graph.trace = old_trace
        for name, old in old_fields.items():
            old.copy_(getattr(graph, name))
            setattr(graph, name, old)


def test_borrowed_trace_overwrites_but_snapshot_survives_next_generate():
    graph = cpu_graph()
    graph._graph = CapturedCPUReplay(graph)
    assert graph.generate(graph._start.clone()) is graph
    owned = graph.snapshot()
    borrowed = graph.trace["pose_before"]
    assert all(not value.requires_grad for value in owned.values())
    assert owned["pose_before"].data_ptr() != borrowed.data_ptr()
    next_pose = graph._start + 0.125
    graph.generate(next_pose)
    assert borrowed is graph.trace["pose_before"]
    exact(borrowed[0], next_pose)
    assert not torch.equal(owned["pose_before"][0], next_pose)
    assert graph.replays == 2


@pytest.mark.parametrize("fault", ["score", "gradient", "pose", "trace", "best_index"])
def test_nonfinite_or_invalid_output_never_published_by_generate(fault):
    graph = cpu_graph()
    graph._operation()

    def replay():
        if fault in ("score", "gradient", "pose"):
            getattr(graph, fault + "_ok").fill_(False)
        elif fault == "trace":
            graph.trace["base_gradient"][-1, 0, 0, 0, 0] = torch.inf
        else:
            graph.best_iteration[0] = 15

    graph._graph = SimpleNamespace(replay=replay)
    with pytest.raises(FloatingPointError):
        graph.generate(graph._start.clone())


def test_public_constructor_and_cache_reject_cpu_without_cuda_calls():
    args = cpu_arguments()
    with pytest.raises(ValueError, match="CUDA float32"):
        m.FP32ProposalGraph(*args)
    cache = m.FP32ProposalCache()
    with pytest.raises(ValueError, match="CUDA"):
        cache.generate(*args)
    assert cache.statistics == dict(
        capacity=4,
        captures=0,
        hits=0,
        misses=0,
        evictions=0,
        failures=0,
        resident=0,
        peak_resident=0,
    )


class DeviceDescriptorOnly(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")


def test_lru4_hit_reconfigures_eviction_and_failure_discard_are_independent(monkeypatch):
    calls = []
    failure = [False]

    class FakeGraph:
        def __init__(self, config, pose, inputs, renderer, *extras, **kwargs):
            self.batch = len(pose)
            self.inputs = inputs
            self.replays = 0
            assert kwargs == {"directional": True, "warmup": 3}
            calls.append((self.batch, "capture"))

        def configure_frame(self, inputs, renderer, *extras):
            self.inputs = inputs
            calls.append((self.batch, "configure"))

        def generate(self, pose):
            calls.append((self.batch, "generate", self.inputs))
            if failure[0]:
                raise ValueError("synthetic invalid output")
            self.replays += 1
            return self

    monkeypatch.setattr(m, "FP32ProposalGraph", FakeGraph)
    cache = m.FP32ProposalCache()
    renderer = SimpleNamespace(n_pixels=4, views=("a", "b", "c"), config=RendererConfig())

    def generate(batch, inputs=0):
        # CPU storage with only a device descriptor override; fake graph never
        # performs tensor operations or asks CUDA about this descriptor.
        pose = torch.zeros((batch, 3, 3, 3)).as_subclass(DeviceDescriptorOnly)
        return cache.generate(1, pose, inputs, renderer, None, None, None, None, None)

    for batch in (15, 14, 16, 16):
        generate(batch, inputs=batch)
    assert cache.statistics == dict(
        capacity=4,
        captures=3,
        hits=1,
        misses=3,
        evictions=0,
        failures=0,
        resident=3,
        peak_resident=3,
    )
    assert calls[-2:] == [(16, "configure"), (16, "generate", 16)]
    generate(17)
    generate(15, inputs=999)  # Touch15: 14 is now least recently successful.
    generate(18)
    assert [key[0] for key in cache._cache] == [16, 17, 15, 18]
    assert cache.statistics["evictions"] == 1
    assert cache.statistics["peak_resident"] == 4
    failure[0] = True
    with pytest.raises(ValueError):
        generate(15)
    assert [key[0] for key in cache._cache] == [16, 17, 18]
    assert cache.statistics["failures"] == 1
    failure[0] = False
    generate(15)
    assert cache.statistics["captures"] == 6
    external = cache.statistics
    external["capacity"] = 999
    assert cache.statistics["capacity"] == 4
    assert not hasattr(cache, "_fit_graph_cache")
    assert not hasattr(cache, "_full_fit_graph")
