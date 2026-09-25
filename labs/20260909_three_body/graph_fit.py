"""Fixed-shape CUDA capture of the complete frozen three-body fit objective.

Only tensor execution is captured. Input validation, Adam, best-iterate selection,
and the final visibility diagnostic remain outside. Explicit first derivatives
are returned; this is not an ordinary differentiable function or thread-safe API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations

import torch
from fast_union import render_union
from graph_union import _CaptureProducts, _CaptureRenderer, _positive_transmission_bound


@dataclass(frozen=True)
class FitInputs:
    origins: torch.Tensor  # B,F,V,N,3; float32 observation cache converted to float64
    directions: torch.Tensor
    held: torch.Tensor  # B,F,V,N; zero/one in the objective dtype
    predicted: torch.Tensor  # B,F,N,3
    motion_sigma: torch.Tensor  # B,F
    lengths: torch.Tensor  # 1,F,2
    length_sigma: torch.Tensor
    node_weight: torch.Tensor  # 1,1,1,N
    target: torch.Tensor | None = None  # V,pixels
    weight: torch.Tensor | None = None  # original target weights * ray_valid


def objective(value, cfg, inputs, denominator, renderer=None, diagnostics=False, width=None):
    """Operations copied in order from ThreeBodyTracker._fit's nested objective.

    The provenance runner records the frozen source and method hashes. The width
    argument is only for the privately validated capture renderer; the public
    diagnostic uses the original renderer and its original default width.
    """
    origins, directions, held = inputs.origins, inputs.directions, inputs.held
    predicted, motion_sigma = inputs.predicted, inputs.motion_sigma
    lengths, length_sigma, node_weight = inputs.lengths, inputs.length_sigma, inputs.node_weight
    target, weight = inputs.target, inputs.weight
    delta = value[:, :, None] - origins
    depth = torch.clamp((delta * directions).sum(-1, keepdim=True), min=0)
    error_sq = ((delta - depth * directions) ** 2).sum(-1) / cfg.ray_sigma_mm**2
    robust = torch.where(error_sq <= 9, error_sq, 6 * torch.sqrt(error_sq + 1e-12) - 9)
    node = ((held * robust + (1 - held) * cfg.missing_cost) * node_weight).sum(
        (1, 2, 3)
    ) / denominator
    segment = torch.linalg.vector_norm(torch.diff(value, dim=2), dim=-1)
    shape = (((segment - lengths) / length_sigma) ** 2).mean((1, 2))
    center = (
        (((value[:, :, 1] - predicted[:, :, 1]) / motion_sigma[..., None]) ** 2)
        .sum(-1)
        .mean(-1)
    )
    relative = value - value[:, :, 1:2]
    previous_relative = predicted - predicted[:, :, 1:2]
    orientation = ((relative - previous_relative) ** 2).sum(-1).mean((1, 2)) / 25
    pair_sq = torch.stack(
        [((value[:, a, 1] - value[:, b, 1]) ** 2).sum(-1)
         for a, b in combinations(range(3), 2)], dim=-1,
    )
    repulsion = torch.exp(-pair_sq / 8).mean(-1)
    silhouette = torch.zeros_like(node)
    rendering = None
    if renderer is not None:
        if diagnostics:
            rendering = renderer.render(value)
            union = rendering.union_alpha
        else:
            union = render_union(renderer, value, width_ratio=width)
        silhouette = (((union - target) ** 2) * weight).sum((1, 2)) / 3
    total = (
        node + cfg.shape_weight * shape
        + cfg.motion_weight * (center + 0.05 * orientation)
        + cfg.repulsion_weight * repulsion + cfg.mask_weight * silhouette
    )
    parts = torch.stack((node, shape, center, repulsion, silhouette), dim=-1)
    return total, parts, rendering


@dataclass(frozen=True)
class GraphFitResult:
    scores: torch.Tensor
    parts: torch.Tensor
    pose_gradient: torch.Tensor


class GraphFit:
    """One B/N/config capture; configure updates one frame, evaluate updates pose.

    The public inputs already use the fitter's dtype/device and derived constants.
    Validation completes before any static buffer is modified. Output tensors are
    copied by default; borrowed outputs are overwritten by the next evaluate.
    Width is the immutable tracker's configured width and is not differentiated.
    """

    def __init__(self, config, initial_pose, inputs, renderer=None, warmup=3):
        if initial_pose.device.type != "cuda":
            raise ValueError("GraphFit requires CUDA tensors")
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        if initial_pose.ndim != 4 or initial_pose.shape[1:] != (3, 3, 3) or not len(initial_pose):
            raise ValueError("pose must have shape (B,3,3,3)")
        self.config, self.device, self.dtype = config, initial_pose.device, initial_pose.dtype
        self.batch_size = len(initial_pose)
        self.n_pixels = None if renderer is None else renderer.n_pixels
        self._renderer = None
        self._renderer_signature = None if renderer is None else (
            renderer.views, renderer.config, renderer.device, renderer.dtype, renderer.n_pixels)
        self._validate_pose(initial_pose)
        self._validate_inputs(inputs, renderer)
        self._pose = torch.empty_like(initial_pose, memory_format=torch.contiguous_format)
        self._pose.copy_(initial_pose.detach()).requires_grad_(True)
        self._pose.grad = torch.zeros_like(self._pose)
        self._inputs = FitInputs(**{
            f.name: (None if getattr(inputs, f.name) is None
                     else getattr(inputs, f.name).detach().clone())
            for f in fields(inputs)
        })
        self.denominator = 9 * float(inputs.node_weight.sum())
        self._width = None
        self.uses_positive_product_bound = False
        if renderer is not None:
            renderer.gaussians(initial_pose)
            self._renderer = object.__new__(_CaptureRenderer)
            self._renderer.__dict__ = renderer.__dict__.copy()
            for name in ("ray_origins", "ray_directions", "ray_valid"):
                setattr(self._renderer, name, getattr(renderer, name).detach().clone())
            self._width = torch.full((self.batch_size, 3), renderer.config.width_ratio,
                                     dtype=self.dtype, device=self.device)
            bound = _positive_transmission_bound(renderer.config, self.dtype)
            self.uses_positive_product_bound = bound > 4 * torch.finfo(self.dtype).eps
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                for _ in range(warmup):
                    self._operation()
            current.wait_stream(self._stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._operation()
        self._borrowed = GraphFitResult(
            self._scores.detach(), self._parts.detach(), self._pose.grad.detach())

    def _validate_pose(self, pose):
        if (pose.shape != (self.batch_size, 3, 3, 3) or pose.device != self.device
                or pose.dtype != self.dtype):
            raise ValueError("pose must retain captured shape, device and dtype")
        if not bool(torch.isfinite(pose).all()):
            raise ValueError("pose must be finite")

    def _validate_inputs(self, inputs, renderer):
        b = self.batch_size
        shapes = dict(origins=(b, 3, 3, 3, 3), directions=(b, 3, 3, 3, 3),
                      held=(b, 3, 3, 3), predicted=(b, 3, 3, 3), motion_sigma=(b, 3),
                      lengths=(1, 3, 2), length_sigma=(1, 3, 2), node_weight=(1, 1, 1, 3))
        if (renderer is None) != (self.n_pixels is None):
            raise ValueError("image presence must retain the captured configuration")
        if renderer is not None:
            signature = (renderer.views, renderer.config, renderer.device, renderer.dtype,
                         renderer.n_pixels)
            if signature != self._renderer_signature:
                raise ValueError("replacement renderer must retain shape, views, config and dtype")
            if renderer.config.width_ratio != self.config.width_ratio:
                raise ValueError("renderer width must match the tracker configuration")
            shapes.update(target=(3, self.n_pixels), weight=(3, self.n_pixels))
        elif inputs.target is not None or inputs.weight is not None:
            raise ValueError("image tensors require a renderer")
        checks = []
        for name, shape in shapes.items():
            value = getattr(inputs, name)
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != self.device or value.dtype != self.dtype):
                raise ValueError(f"invalid {name} shape, device or dtype")
            checks.append(torch.isfinite(value).all())
        checks += [((inputs.held == 0) | (inputs.held == 1)).all(),
                   (inputs.motion_sigma > 0).all(), (inputs.lengths > 0).all(),
                   (inputs.length_sigma > 0).all(), (inputs.node_weight > 0).all()]
        if renderer is not None:
            origin, direction, valid = (renderer.ray_origins, renderer.ray_directions,
                                        renderer.ray_valid)
            if (origin.shape != (3, self.n_pixels, 3) or direction.shape != origin.shape
                    or valid.shape != (3, self.n_pixels) or valid.dtype != torch.bool):
                raise ValueError("invalid image ray buffers")
            norm = torch.linalg.vector_norm(direction, dim=-1)
            checks += [torch.isfinite(origin).all(), torch.isfinite(direction).all(),
                       torch.isclose(norm, torch.ones_like(norm),
                                     rtol=32 * torch.finfo(self.dtype).eps, atol=0).all(),
                       (inputs.weight >= 0).all()]
        if not bool(torch.stack(checks).all()):
            raise ValueError("frame tensors must be finite, valid and have positive scales")

    def configure(self, inputs, renderer=None):
        self._validate_inputs(inputs, renderer)
        # node_weight is a fixed engineering choice: denominator is captured as a scalar.
        if not torch.equal(inputs.node_weight, self._inputs.node_weight):
            raise ValueError("node weights must retain their captured value")
        with torch.no_grad():
            for field in fields(inputs):
                value = getattr(inputs, field.name)
                if value is not None:
                    getattr(self._inputs, field.name).copy_(value)
            if renderer is not None:
                for name in ("ray_origins", "ray_directions", "ray_valid"):
                    getattr(self._renderer, name).copy_(getattr(renderer, name))

    def _operation(self):
        self._pose.grad.zero_()
        with _CaptureProducts(self.uses_positive_product_bound):
            self._scores, self._parts, _ = objective(
                self._pose, self.config, self._inputs, self.denominator,
                self._renderer, width=self._width)
        self._scores.sum().backward()

    def evaluate(self, pose, copy_outputs=True):
        self._validate_pose(pose)
        with torch.no_grad():
            self._pose.copy_(pose)
            self._graph.replay()
            if copy_outputs:
                return GraphFitResult(*(value.clone() for value in (
                    self._borrowed.scores, self._borrowed.parts, self._borrowed.pose_gradient)))
            return self._borrowed
