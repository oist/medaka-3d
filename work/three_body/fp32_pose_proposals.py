"""FP32-only numerical pose proposals; no FP64 scoring or tracking policy.

The frozen FullFitGraph._operation supplies all15 score evaluations, the same
14 default-foreach Adam updates, reset and strict-best selection. Only its host
construction is specialized here. This changes numerical precision deliberately;
it has no FP64 parity, speed or biological-ID qualification.

Production construction/generation requires CUDA float32. _initialize remains
usable with tiny CPU tensors for tests of the identical operation. Returned
graphs/trace tensors are borrowed: the next use of that cached graph overwrites
their storage. Copy before any further cache call, or call graph.snapshot().
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields

import torch
from full_fit_graph_fitter import (
    STEPS,
    FitInputs,
    FullFitGraph,
    GraphFit,
    _CaptureRenderer,
    _default_to_fused_or_foreach,
    _positive_transmission_bound,
    accumulate_finite,
    frozen_adam_schedule,
    require_finite,
)

PARENT_SOURCE_SHA = "3255c53de0e7ed4ca02872229a3bd198160b9651b556bcd0c121caf9c339a376"
CACHE_CAPACITY = 4
WARMUP = 3
TRACE_FIELDS = (
    "pose_before",
    "score",
    "parts",
    "base_gradient",
    "update_gradient",
    "moment",
    "variance",
    "pose_after",
)


class FP32ProposalGraph(FullFitGraph):
    """One fixed B/N/config/device FP32 capture, independent of any FP64 cache.

    trace['pose_before'] is X32(0..14), including the rounded initial pose.
    trace['score'] includes the directional correction when enabled; 'parts'
    keeps the frozen base-parts convention. initial_parts has corrected center.
    This component never returns FP64 rescores, mask support or new beam states.
    """

    def __init__(
        self,
        config,
        initial_pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        lower,
        upper,
        *,
        directional=True,
        warmup=WARMUP,
    ):
        if (
            not isinstance(initial_pose, torch.Tensor)
            or initial_pose.device.type != "cuda"
            or initial_pose.dtype != torch.float32
        ):
            raise ValueError("FP32 proposal capture requires CUDA float32")
        if type(warmup) is not int or warmup != WARMUP:
            raise ValueError("exactly3 proposal warmup fits required")
        if _default_to_fused_or_foreach([initial_pose], False, use_fused=False) != (False, True):
            raise ValueError("installed default Adam does not select unfused foreach")
        self._initialize(
            config,
            initial_pose,
            inputs,
            renderer,
            axes,
            parallel_sigma,
            enabled,
            lower,
            upper,
            directional,
        )
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                for _ in range(WARMUP):
                    self._operation()
            current.wait_stream(self._stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._operation()
        self.replays = 0

    def _initialize(
        self,
        config,
        initial_pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        lower,
        upper,
        directional,
    ):
        """Own input tensors before execution; CPU access is synthetic-test only."""
        if (
            not isinstance(initial_pose, torch.Tensor)
            or initial_pose.dtype != torch.float32
            or initial_pose.ndim != 4
            or initial_pose.shape[1:] != (3, 3, 3)
            or len(initial_pose) == 0
        ):
            raise ValueError("FP32 pose must have nonempty shape (B,3,3,3)")
        if type(directional) is not bool or renderer is None:
            raise ValueError("explicit direction policy and original renderer required")
        self.schedule = frozen_adam_schedule(config.learning_rate_mm, config.opt_steps)
        self.config, self.device, self.dtype = config, initial_pose.device, initial_pose.dtype
        self.batch_size, self.directional = len(initial_pose), directional
        self.n_pixels = renderer.n_pixels
        self._renderer_signature = (
            renderer.views,
            renderer.config,
            renderer.device,
            renderer.dtype,
            renderer.n_pixels,
        )
        self._validate_pose(initial_pose)
        self._validate_inputs(inputs, renderer)
        self._validate_extra(axes, parallel_sigma, enabled, lower, upper)
        self._start = initial_pose.detach().clone()
        self._working = initial_pose.detach().clone()
        self._pose = torch.empty_like(initial_pose, memory_format=torch.contiguous_format)
        self._pose.copy_(initial_pose.detach()).requires_grad_(True)
        self._pose.grad = torch.zeros_like(self._pose)
        self._moment, self._variance = (
            torch.zeros_like(self._working),
            torch.zeros_like(self._working),
        )
        self._inputs = FitInputs(
            **{
                field.name: None
                if getattr(inputs, field.name) is None
                else getattr(inputs, field.name).detach().clone()
                for field in fields(inputs)
            }
        )
        self.denominator = 9 * float(inputs.node_weight.sum())
        renderer.gaussians(initial_pose)
        self._renderer = object.__new__(_CaptureRenderer)
        # Own all tensor constants as well as rays, preserving native32 fractions
        # and taper from the separately constructed caller renderer.
        self._renderer.__dict__ = {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in renderer.__dict__.items()
        }
        self._width = torch.full(
            (self.batch_size, 3), renderer.config.width_ratio, dtype=self.dtype, device=self.device
        )
        bound = _positive_transmission_bound(renderer.config, self.dtype)
        self.uses_positive_product_bound = bound > 4 * torch.finfo(self.dtype).eps
        self._axes, self._parallel, self._enabled = (
            axes.detach().clone(),
            parallel_sigma.detach().clone(),
            enabled.detach().clone(),
        )
        self._lower, self._upper = lower.detach().clone(), upper.detach().clone()
        self.replays = 0

    def _validate_inputs(self, inputs, renderer):
        # The frozen base validates FitInputs dtype, but its public ray check
        # permits other floating dtypes. A proposal graph must never silently
        # consume FP64 renderer constants or cast replacement FP64 rays.
        if renderer is None or renderer.dtype != self.dtype:
            raise ValueError("proposal renderer must use native float32 constants")
        for name in ("ray_origins", "ray_directions", "ray_valid"):
            value = getattr(renderer, name)
            dtype = torch.bool if name == "ray_valid" else self.dtype
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or value.device != self.device
            ):
                raise ValueError("proposal renderer ray dtype/device differs")
        GraphFit._validate_inputs(self, inputs, renderer)

    def _check_generated(self):
        if set(self.trace) != set(TRACE_FIELDS):
            raise RuntimeError("frozen proposal trace fields changed")
        self.trace_ok = accumulate_finite(
            torch.ones((), dtype=torch.bool, device=self.device),
            *self.trace.values(),
            self.initial_score,
            self.initial_parts,
            self.best_pose,
            self.best_score,
        )
        self.trace_ok &= ((self.best_iteration >= 0) & (self.best_iteration <= STEPS)).all()
        require_finite(self.score_ok, self.gradient_ok, self.pose_ok & self.trace_ok)

    def generate(self, pose):
        """Reset/replay once; reject nonfinite data before publishing borrowed self."""
        self.replay(pose)
        self._check_generated()
        return self

    @property
    def finite_flags(self):
        """Borrowed device flags; a successful generate has checked all four."""
        return dict(
            score_ok=self.score_ok,
            gradient_ok=self.gradient_ok,
            pose_ok=self.pose_ok,
            trace_ok=self.trace_ok,
        )

    def snapshot(self):
        """Detached owned tensors, stable across replay, reconfiguration or eviction."""
        self._check_generated()
        values = dict(
            **self.trace,
            best_pose=self.best_pose,
            best_score=self.best_score,
            best_iteration=self.best_iteration,
            initial_score=self.initial_score,
            initial_parts=self.initial_parts,
            **self.finite_flags,
        )
        return {name: value.detach().clone() for name, value in values.items()}


class FP32ProposalCache:
    """Serial, independent LRU4; no alias or entry in a tracker/FP64 cache.

    Returned graph references can prolong residency after eviction. The caller
    must copy outputs and release graph references before subsequent calls; the
    caller also owns the allocator/headroom guard. A failed active graph is
    removed, other healthy entries remain. No retry or silent capacity change.
    """

    def __init__(self):
        self._cache = OrderedDict()
        self._inflight = False
        self._captures = self._hits = self._misses = self._evictions = self._failures = 0
        self._peak = 0

    @property
    def statistics(self):
        return dict(
            capacity=CACHE_CAPACITY,
            captures=self._captures,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            failures=self._failures,
            resident=len(self._cache),
            peak_resident=self._peak,
        )

    def generate(
        self,
        config,
        pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        lower,
        upper,
        *,
        directional=True,
    ):
        if self._inflight:
            raise RuntimeError("FP32 proposal cache requires serial non-reentrant use")
        if (
            not isinstance(pose, torch.Tensor)
            or pose.device.type != "cuda"
            or pose.dtype != torch.float32
            or pose.ndim != 4
            or pose.shape[1:] != (3, 3, 3)
            or not len(pose)
            or type(directional) is not bool
            or renderer is None
        ):
            raise ValueError("FP32 proposal cache requires CUDA (B,3,3,3) and explicit policy")
        key = (
            len(pose),
            renderer.n_pixels,
            config,
            directional,
            pose.dtype,
            pose.device,
            renderer.views,
            renderer.config,
        )
        # Config must be hashable just as in the frozen fitter cache.
        hash(key)
        hit = key in self._cache
        self._inflight = True
        graph = None
        try:
            if hit:
                self._hits += 1
                graph = self._cache.pop(key)
                graph.configure_frame(inputs, renderer, axes, parallel_sigma, enabled, lower, upper)
            else:
                self._misses += 1
                if len(self._cache) == CACHE_CAPACITY:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                graph = FP32ProposalGraph(
                    config,
                    pose,
                    inputs,
                    renderer,
                    axes,
                    parallel_sigma,
                    enabled,
                    lower,
                    upper,
                    directional=directional,
                    warmup=WARMUP,
                )
                self._captures += 1
            self._peak = max(self._peak, len(self._cache) + 1)
            graph.generate(pose)
            self._cache[key] = graph
            return graph
        except BaseException:
            self._failures += 1
            self._cache.pop(key, None)
            raise
        finally:
            self._inflight = False
