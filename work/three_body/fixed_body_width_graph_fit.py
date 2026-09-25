"""Fixed-radial-width adaptation of frozen GraphFit; CUDA qualification pending.

Only the capture renderer allocation and width-reference validation are new.
The complete frozen objective, CUDA operation, product backward mode, validation
of FitInputs, configure copies, and evaluate/replay are inherited unchanged.
The width tensor passed by the frozen operation is not a learnable parameter;
the capture renderer replaces its radial result with the registered absolute one.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import torch
from fixed_body_width_renderer import FixedWidthRenderer
from graph_fit import FitInputs, GraphFit, GraphFitResult
from graph_union import _CaptureRenderer, _positive_transmission_bound


class _FixedWidthCaptureRenderer(_CaptureRenderer, FixedWidthRenderer):
    """Private no-host-check Gaussian path with the same owned absolute widths."""

    def gaussians(self, pose, width):
        bodies = _CaptureRenderer.gaussians(self, pose, width)
        radial = self._fixed_radial_sigma_mm.unsqueeze(0).expand(pose.shape[0], -1, -1).clone()
        return replace(bodies, radial_sigma_mm=radial)


def make_capture_renderer(renderer):
    """Clone all tensor/array storage; no CUDA APIs or real input reads here."""
    if type(renderer) is not FixedWidthRenderer:
        raise ValueError("fixed-width capture requires a registered FixedWidthRenderer")
    expected = renderer.validate_fixed_width()
    result = object.__new__(_FixedWidthCaptureRenderer)
    result.__dict__ = {
        name: value.detach().clone()
        if isinstance(value, torch.Tensor)
        else value.copy()
        if isinstance(value, np.ndarray)
        else value
        for name, value in renderer.__dict__.items()
    }
    if result.validate_fixed_width() != expected:
        raise ValueError("capture fixed-width signature differs")
    return result


class FixedWidthGraphFit(GraphFit):
    """Original B/N/config capture plus immutable registered radial signature."""

    def __init__(self, config, initial_pose, inputs, renderer=None, warmup=3):
        if initial_pose.device.type != "cuda":
            raise ValueError("GraphFit requires CUDA tensors")
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        if initial_pose.ndim != 4 or initial_pose.shape[1:] != (3, 3, 3) or not len(initial_pose):
            raise ValueError("pose must have shape (B,3,3,3)")
        if renderer is not None and type(renderer) is not FixedWidthRenderer:
            raise ValueError("fixed-width graph requires the registered FixedWidthRenderer")
        self._fixed_width_signature = None if renderer is None else renderer.validate_fixed_width()
        self.config, self.device, self.dtype = config, initial_pose.device, initial_pose.dtype
        self.batch_size = len(initial_pose)
        self.n_pixels = None if renderer is None else renderer.n_pixels
        self._renderer = None
        self._renderer_signature = (
            None
            if renderer is None
            else (
                renderer.views,
                renderer.config,
                renderer.device,
                renderer.dtype,
                renderer.n_pixels,
            )
        )
        self._validate_pose(initial_pose)
        self._validate_inputs(inputs, renderer)
        self._pose = torch.empty_like(initial_pose, memory_format=torch.contiguous_format)
        self._pose.copy_(initial_pose.detach()).requires_grad_(True)
        self._pose.grad = torch.zeros_like(self._pose)
        self._inputs = FitInputs(
            **{
                f.name: (
                    None
                    if getattr(inputs, f.name) is None
                    else getattr(inputs, f.name).detach().clone()
                )
                for f in fields(inputs)
            }
        )
        self.denominator = 9 * float(inputs.node_weight.sum())
        self._width = None
        self.uses_positive_product_bound = False
        if renderer is not None:
            renderer.gaussians(initial_pose)
            self._renderer = make_capture_renderer(renderer)
            self._width = torch.full(
                (self.batch_size, 3),
                renderer.config.width_ratio,
                dtype=self.dtype,
                device=self.device,
            )
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
            self._scores.detach(), self._parts.detach(), self._pose.grad.detach()
        )

    def _validate_inputs(self, inputs, renderer):
        # Validate before inherited configure can mutate any static frame buffer.
        if renderer is not None:
            if type(renderer) is not FixedWidthRenderer:
                raise ValueError("replacement requires registered FixedWidthRenderer")
            if renderer.validate_fixed_width() != self._fixed_width_signature:
                raise ValueError("replacement fixed-width reference/signature differs")
            if self._renderer is not None and (
                type(self._renderer) is not _FixedWidthCaptureRenderer
                or self._renderer.validate_fixed_width() != self._fixed_width_signature
            ):
                raise ValueError("captured fixed-width reference/signature differs")
        super()._validate_inputs(inputs, renderer)
