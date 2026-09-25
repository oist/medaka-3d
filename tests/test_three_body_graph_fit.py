"""The complete captured fitter must preserve the frozen objective and Adam output."""

import ast
import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def modules():
    names = ("prepare_sequence", "observations", "node_geometry", "renderer", "fast_union",
             "tracker", "graph_union", "graph_fit", "accelerated_tracker")
    previous, result = {n: sys.modules.get(n) for n in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"three_body_graph_fit_test_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            result[name] = module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return result


M = modules()
T, G, A = M["tracker"], M["graph_fit"], M["accelerated_tracker"]


def reference_scope(tracker, proposals, states, rays, targets):
    """Execute the actual frozen _fit prefix, stopping before optimizer construction.

    No copied objective formula is used by this oracle. This also preserves the
    old float32 assigned-ray conversion and denominator computation independently.
    """
    tree = ast.parse((LAB / "tracker.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ThreeBodyTracker")
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fit")
    stop = next(i for i, n in enumerate(fit.body)
                if isinstance(n, ast.FunctionDef) and n.name == "objective")
    fit.body = [*fit.body[:stop + 1], ast.Return(ast.Call(ast.Name("locals", ast.Load()), [], []))]
    fit.name = "reference_prefix"
    namespace = dict(T.ThreeBodyTracker._fit.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module([fit], [])), str(LAB / "tracker.py"),
                 "exec"), namespace)
    return namespace[fit.name](tracker, proposals, states, rays, targets)


@pytest.fixture
def case(rig):
    rig = replace(rig, cameras=tuple(
        replace(c, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if c.view == VIEWS[0] else 2,
                distortion=(-.15, .07, .003, -.002, -.01)) for c in rig.cameras))
    centers = np.array([[70., 65., 80.], [73., 65., 90.], [135., 120., 135.]])
    pose = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 2., -1.]])[None]
    return rig, pose


def inputs(case, batch, frame, device, image=True):
    rig, seed = case
    actual = seed + np.array([.15, -.08, .03]) * frame
    points = {v: rig.camera(v).project(actual) for v in VIEWS}
    rays = M["node_geometry"].RayObservations.from_points(rig, points)
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    plan[2, 1:, 2] = -1  # Missing tail has explicit missing costs, not fake observations.
    state = T.State(seed, np.zeros((3, 3)), np.array([0, frame, 4 + frame]),
                    np.zeros(3, bool), 0.)
    rng = np.random.default_rng(83)
    proposals = [T.Proposal(seed + rng.uniform(-.1, .1, seed.shape), actual - .05,
                            plan, 0, "test") for _ in range(batch)]
    targets = None
    if image:
        offsets = np.array([[-2., 0.], [0., -1.], [1., 2.]])
        pixels = {v: (uv.reshape(-1, 2)[:, None] + offsets[None]).reshape(-1, 2)
                  for v, uv in points.items()}
        pixels[VIEWS[1]][-2:] = np.nan  # Invalid padded rays must stay neutral.
        targets = M["observations"].PixelTargets(
            pixels, (rng.random((3, 27)) > .5).astype(float),
            np.full((3, 27), 1 / 27), np.zeros((3, 27), bool))
    cfg = T.TrackerConfig(device=device, opt_steps=3)
    original = T.ThreeBodyTracker(rig, seed, cfg)
    return original, proposals, [state], rays, targets


@pytest.mark.parametrize("image", [False, True])
def test_objective_copy_matches_actual_frozen_source_cpu(case, image):
    args = inputs(case, 3, 1, "cpu", image)
    scope = reference_scope(*args)
    value = scope["pose"]
    frame = G.FitInputs(**{f.name: scope.get(f.name) for f in fields(G.FitInputs)})
    old = scope["objective"](value)
    new = G.objective(value, args[0].config, frame, scope["denominator"], scope["renderer"])
    for a, b in zip(new[:2], old[:2], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga, = torch.autograd.grad(old[0].sum(), value, retain_graph=True)
    gb, = torch.autograd.grad(new[0].sum(), value)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    with pytest.raises(ValueError, match="CUDA"):
        G.GraphFit(args[0].config, value, frame, scope["renderer"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture requires GPU")
@pytest.mark.parametrize("batch,image", [(1, False), (1, True), (6, True), (18, True)])
def test_capture_and_complete_fit_preserve_changed_frames(case, batch, image):
    graph, accelerated = None, None
    for index in (0, 1, 0):
        args = inputs(case, batch, index, "cuda", image)
        original, proposals, states, rays, targets = args
        scope = reference_scope(*args)
        frame = G.FitInputs(**{f.name: scope.get(f.name) for f in fields(G.FitInputs)})
        pose = scope["pose"]
        score, parts, _ = scope["objective"](pose)
        gradient, = torch.autograd.grad(score.sum(), pose)
        if graph is None:
            graph = G.GraphFit(original.config, pose, frame, scope["renderer"], warmup=1)
            accelerated = A.AcceleratedThreeBodyTracker(original.rig, original.initial,
                                                       original.config)
        else:
            graph.configure(frame, scope["renderer"])
        got = graph.evaluate(pose)
        for a, b in zip((got.scores, got.parts, got.pose_gradient),
                        (score, parts, gradient), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        old = original._fit(proposals, states, rays, targets)
        new = accelerated._fit(proposals, states, rays, targets)
        for a, b in zip(new, old, strict=True):
            np.testing.assert_array_equal(a, b)
        frozen_score = got.scores.clone()
        graph.evaluate(pose + .2)
        torch.testing.assert_close(got.scores, frozen_score, rtol=0, atol=0)
        bad_pose = pose.detach().clone()
        bad_pose[0, 0, 0, 0] = torch.nan
        with pytest.raises(ValueError, match="finite"):
            graph.evaluate(bad_pose)
        bad = replace(frame, motion_sigma=frame.motion_sigma * 0)
        with pytest.raises(ValueError, match="positive"):
            graph.configure(bad, scope["renderer"])
        restored = graph.evaluate(pose)
        torch.testing.assert_close(restored.scores, score, rtol=0, atol=0)
    # The subclass must not replace temporal or support methods indirectly.
    assert set(A.AcceleratedThreeBodyTracker.__dict__) - {
        "__module__", "__doc__", "__firstlineno__", "__static_attributes__"
    } == {"_fit"}
