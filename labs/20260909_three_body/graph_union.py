"""CUDA Graph prototype for fixed-shape union image loss and its pose gradient.

This does not implement an optimizer or ordinary autograd function. Pose gradients
are returned explicitly. Each batch/pixel shape needs its own capture. Validation
and buffer copies remain outside the graph and are part of public-call overhead.
The frozen renderer and tracking modules are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from fast_union import render_union
from renderer import GaussianBodies, RefractiveFishRenderer
from torch.autograd.function import once_differentiable
from torch.overrides import TorchFunctionMode

ORIGIN = {
    "path": "labs/20260909_three_body/renderer.py",
    "sha256": "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c",
    "adapted": "gaussians operations unchanged; public validation separated from graph capture",
}
PROD_BACKWARD_ORIGIN = (
    "https://github.com/pytorch/pytorch/blob/"
    "cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/autograd/FunctionsManual.cpp#L765-L849"
)


class _CaptureLastDimProd(torch.autograd.Function):
    """Original forward product and both original backward branches, without .item().

    The installed torch commit tests the total zero count on the host. Select
    between its two formulas using a device scalar instead. The zero-safe branch
    is global across the input, as in that implementation, not per product row.
    This preserves its floating operation order, including when a zero elsewhere
    changes which formula is used. Only first derivatives are supported here.
    """

    @staticmethod
    def forward(ctx, value):
        result = value.prod(dim=-1)
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        value, result = ctx.saved_tensors
        gradient = gradient.unsqueeze(-1)
        ordinary = gradient * (result.unsqueeze(-1) / value)
        ones = torch.ones_like(value[..., :1])
        normal = torch.cat((ones, value[..., :-1]), dim=-1).cumprod(dim=-1)
        reverse = torch.cat((ones, value[..., 1:].flip((-1,))), dim=-1)
        reverse = reverse.cumprod(dim=-1).flip((-1,))
        safe = gradient * (normal * reverse)
        return torch.where((value == 0).any(), safe, ordinary)


class _CapturePositiveLastDimProd(torch.autograd.Function):
    """Original no-zero derivative, only after a conservative config-only proof."""

    @staticmethod
    def forward(ctx, value):
        result = value.prod(dim=-1)
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        value, result = ctx.saved_tensors
        return gradient.unsqueeze(-1) * (result.unsqueeze(-1) / value)


def _positive_transmission_bound(config, dtype):
    """Conservative lower bound survives Gaussian product and 1-(1-product).

    Every alpha <= rounded opacity since Mahalanobis distance is nonnegative.
    Round the alpha bound upward, its complement downward, and allow K eps of
    product relative error. Requiring >4 eps also protects the final subtraction
    from unity. Configs without that margin keep the general zero-safe backward.
    """
    scalar = np.float64 if dtype == torch.float64 else np.float32
    eps = np.finfo(scalar).eps
    upper_alpha = np.nextafter(scalar(config.opacity), scalar(np.inf))
    lower_factor = float(np.nextafter(scalar(1) - upper_alpha, scalar(0)))
    if lower_factor <= 0:
        return 0.
    return float(lower_factor**config.gaussians_per_fish * (1 - config.gaussians_per_fish * eps))


class _CaptureProducts(TorchFunctionMode):
    """Intercept only the two union-product calls; never patch Tensor or a module."""

    def __init__(self, proven_positive=False):
        super().__init__()
        self.operation = (_CapturePositiveLastDimProd if proven_positive else _CaptureLastDimProd)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if (func in (torch.Tensor.prod, torch.prod) and dim == -1
                and not kwargs.get("keepdim", False) and kwargs.get("dtype") is None):
            return self.operation.apply(args[0])
        return func(*args, **kwargs)


class _CaptureRenderer(RefractiveFishRenderer):
    """Private fixed-shape geometry; GraphUnion validates every public call first."""

    def gaussians(self, pose, width):
        vectors = pose[..., 1:, :] - pose[..., :-1, :]
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        unit = vectors / lengths.clamp_min(1e-12).unsqueeze(-1)
        fallback = torch.zeros_like(unit)
        fallback[..., 0] = 1
        unit = torch.where((lengths > 1e-12).unsqueeze(-1), unit, fallback)
        centers = pose[..., :-1, :][..., self._segment, :]
        centers = centers + vectors[..., self._segment, :] * self._fractions[:, None]
        axial = (lengths / self._counts * self.config.axial_overlap)[..., self._segment]
        radial = lengths.sum(dim=-1, keepdim=True) * width.unsqueeze(-1) * self._taper
        return GaussianBodies(
            centers=centers,
            tangents=unit[..., self._segment, :],
            axial_sigma_mm=axial.clamp_min(self.config.minimum_sigma_mm),
            radial_sigma_mm=radial.clamp_min(self.config.minimum_sigma_mm),
            segment_lengths_mm=lengths,
            nondegenerate=(lengths > self.config.minimum_sigma_mm).all(dim=-1),
        )


@dataclass(frozen=True)
class GraphUnionResult:
    union_alpha: torch.Tensor  # B,V,N
    loss: torch.Tensor  # B; weighted sum over V,N divided by 3, not by B
    pose_gradient: torch.Tensor  # B,3,3,3; gradient of loss.sum()


class GraphUnion:
    """Capture one batch size, then replay with new poses, samples and water rays.

    ``renderer`` fixes camera order, N, dtype and Gaussian config. The initial
    pose has shape (B,3,3,3). Public evaluate requires the same B and targets of
    shape (V,N). A replacement renderer may change sampled rays/validity but must
    retain that shape, dtype, device, view order and config. Width may be scalar,
    (3,), or (B,3); None reuses the constructor width. Width is not differentiated.

    Validation is deliberately outside capture. Result tensors are independent
    copies by default. ``copy_outputs=False`` explicitly borrows buffers that the
    next call overwrites. This object is not thread-safe; it owns one stream and
    capture. It has no optimizer state, saved best pose, ID or tracking semantics.
    """

    def __init__(self, renderer, initial_pose_xyz, width_ratio=None, warmup=3):
        if renderer.device.type != "cuda":
            raise ValueError("GraphUnion requires a CUDA renderer")
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        self.device, self.dtype = renderer.device, renderer.dtype
        self.views, self.config = renderer.views, renderer.config
        self.n_pixels = renderer.n_pixels
        self.transmission_lower_bound = _positive_transmission_bound(self.config, self.dtype)
        self.uses_positive_product_bound = (
            self.transmission_lower_bound > 4 * torch.finfo(self.dtype).eps)
        pose = torch.as_tensor(initial_pose_xyz, device=self.device, dtype=self.dtype)
        if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or not len(pose):
            raise ValueError("initial pose must have shape (B,3,3,3)")
        self.batch_size = len(pose)
        width = self._width_tensor(self.config.width_ratio if width_ratio is None else width_ratio)
        # Reuse the original public checks before constructing private capture state.
        renderer.gaussians(pose, width)
        self._default_width = width.detach().clone()
        self._pose = torch.empty_like(pose, memory_format=torch.contiguous_format)
        self._pose.copy_(pose.detach()).requires_grad_(True)
        self._pose.grad = torch.zeros_like(self._pose)
        self._width = width.detach().clone()
        self._target = torch.zeros((3, self.n_pixels), device=self.device, dtype=self.dtype)
        self._weights = torch.ones_like(self._target)
        self._renderer = object.__new__(_CaptureRenderer)
        self._renderer.__dict__ = renderer.__dict__.copy()
        self._renderer.ray_origins = renderer.ray_origins.detach().clone()
        self._renderer.ray_directions = renderer.ray_directions.detach().clone()
        self._renderer.ray_valid = renderer.ray_valid.detach().clone()
        self._validate_rays(renderer)
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
        self._borrowed = GraphUnionResult(
            self._union.detach(), self._loss.detach(), self._pose.grad.detach())

    def _width_tensor(self, value):
        width = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if width.ndim == 0 or width.shape == (3,):
            return width.expand(self.batch_size, 3)
        if width.shape != (self.batch_size, 3):
            raise ValueError("width_ratio must be scalar, (3,), or (B,3)")
        return width

    def _validate_rays(self, renderer):
        if (renderer.views != self.views or renderer.n_pixels != self.n_pixels
                or renderer.config != self.config or renderer.dtype != self.dtype
                or renderer.device != self.device):
            raise ValueError(
                "replacement renderer must retain shape, views, config, dtype and device")
        origin, direction, valid = (renderer.ray_origins, renderer.ray_directions,
                                    renderer.ray_valid)
        if (origin.shape != (3, self.n_pixels, 3) or direction.shape != origin.shape
                or valid.shape != (3, self.n_pixels) or valid.dtype != torch.bool):
            raise ValueError("invalid replacement ray buffer shapes or validity dtype")
        norms = torch.linalg.vector_norm(direction, dim=-1)
        checks = torch.stack((torch.isfinite(origin).all(), torch.isfinite(direction).all(),
                              torch.isclose(norms, torch.ones_like(norms),
                                            rtol=32 * torch.finfo(self.dtype).eps, atol=0).all()))
        if not bool(checks.all()):
            raise ValueError("ray buffers must be finite with normalized directions")

    def _operation(self):
        self._pose.grad.zero_()
        with _CaptureProducts(self.uses_positive_product_bound):
            self._union = render_union(self._renderer, self._pose, self._width)
        weight = self._weights * self._renderer.ray_valid
        self._loss = ((self._union - self._target).square() * weight).sum((1, 2)) / 3
        self._loss.sum().backward()

    def evaluate(self, pose_xyz, target, weights, width_ratio=None, renderer=None,
                 copy_outputs=True):
        pose = torch.as_tensor(pose_xyz, dtype=self.dtype, device=self.device)
        target = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        width = self._width_tensor(self._default_width if width_ratio is None else width_ratio)
        if pose.shape != self._pose.shape:
            raise ValueError("pose must retain the captured (B,3,3,3) shape")
        if target.shape != self._target.shape or weights.shape != target.shape:
            raise ValueError("target and weights must retain the captured (3,N) shape")
        checks = torch.stack((torch.isfinite(pose).all(),
                              (torch.isfinite(width) & (width > 0)).all(),
                              torch.isfinite(target).all(),
                              (torch.isfinite(weights) & (weights >= 0)).all()))
        finite_pose, valid_width, finite_target, valid_weights = checks.detach().cpu().tolist()
        if not finite_pose:
            raise ValueError("pose must be finite")
        if not valid_width:
            raise ValueError("width_ratio must be finite and positive")
        if not finite_target or not valid_weights:
            raise ValueError("targets must be finite and weights finite and nonnegative")
        if renderer is not None:
            self._validate_rays(renderer)
        # Host validation above has completed before replay. Copies and replay
        # execute on the caller's current stream; CUDA graph replay uses it.
        with torch.no_grad():
            self._pose.copy_(pose)
            self._width.copy_(width)
            self._target.copy_(target)
            self._weights.copy_(weights)
            if renderer is not None:
                self._renderer.ray_origins.copy_(renderer.ray_origins)
                self._renderer.ray_directions.copy_(renderer.ray_directions)
                self._renderer.ray_valid.copy_(renderer.ray_valid)
            self._graph.replay()
            if copy_outputs:
                return GraphUnionResult(*(value.clone() for value in (
                    self._borrowed.union_alpha, self._borrowed.loss, self._borrowed.pose_gradient)))
            return self._borrowed
