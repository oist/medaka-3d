"""Unqualified FP64 graph execution with outer-batch buckets16/20/24.

Duplicate the first complete hypothesis only inside the captured optimizer.
Slice the real prefix before the frozen parent's eager final objective/support.
No fish, source, target, weight, Adam step or beam candidate is added publicly.
Batch-shape changes can change CUDA rounding; actual all77 parity is REQUIRED.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields

import torch
from full_fit_graph_cache4 import parent_cache_key
from full_fit_graph_fitter import FullFitDirectionalDeviceFitter as FrozenFitter
from full_fit_graph_fitter import FullFitGraph
from graph_fit import FitInputs

BUCKETS = (16, 20, 24)
BATCH_FIELDS = frozenset(("origins", "directions", "held", "predicted", "motion_sigma"))
PARENT_SHA = "3255c53de0e7ed4ca02872229a3bd198160b9651b556bcd0c121caf9c339a376"


def bucket_for(batch):
    if type(batch) is not int or not 1 <= batch <= BUCKETS[-1]:
        raise ValueError("outer batch must be an integer in1..24")
    return next(size for size in BUCKETS if batch <= size)


def pad_rows(value, batch, bucket):
    """Keep all real rows first; deterministic valid duplicate, never a zero pose."""
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != batch:
        raise ValueError("explicit leading batch tensor required")
    if bucket != bucket_for(batch):
        raise ValueError("fixed smallest bucket required")
    if bucket == batch:
        return value
    return torch.cat((value, value[:1].expand(bucket - batch, *value.shape[1:])), dim=0)


def pad_inputs(inputs, batch, bucket):
    # Shared lengths (1,F,2), node weights (1,1,1,N), image rays/targets remain
    # shared even when the real batch happens to be1 or3. Never infer from shape.
    if not isinstance(inputs, FitInputs):
        raise ValueError("original FitInputs required")
    return FitInputs(
        **{
            f.name: pad_rows(getattr(inputs, f.name), batch, bucket)
            if f.name in BATCH_FIELDS
            else getattr(inputs, f.name)
            for f in fields(inputs)
        }
    )


class BucketGraphProxy:
    """Serial interface consumed by the frozen parent's configure/replay calls.

    A maximum of three real graphs is owned. Prefix views are borrowed until
    the next configure call; the frozen parent copies diagnostics and returns.
    The test factory may implement CPU _initialize/_operation, never production.
    """

    _published = (
        "best_pose",
        "best_score",
        "best_iteration",
        "initial_score",
        "initial_parts",
        "score_ok",
        "gradient_ok",
        "pose_ok",
        "trace",
    )

    def __init__(self, config, directional, *, graph_factory=FullFitGraph):
        self.config, self.directional, self.graph_factory = config, directional, graph_factory
        self.cache, self.pending, self.active_key = OrderedDict(), None, None
        self.counts = dict(
            captures=0,
            hits=0,
            misses=0,
            evictions=0,
            failures=0,
            peak_resident=0,
            real_candidates=0,
            padded_candidates=0,
        )

    def clear_published(self):
        for name in self._published:
            self.__dict__.pop(name, None)

    def configure_frame(self, inputs, renderer, axes, parallel_sigma, enabled, lower, upper):
        if self.pending is not None:
            raise RuntimeError("configure/replay must alternate serially")
        self.clear_published()
        self.pending = inputs, renderer, axes, parallel_sigma, enabled, lower, upper

    def discard_active(self):
        if self.active_key is not None:
            self.cache.pop(self.active_key, None)
        self.pending, self.active_key = None, None
        self.clear_published()

    def replay(self, pose):
        if self.pending is None:
            raise RuntimeError("configure required before replay")
        inputs, renderer, axes, parallel, enabled, lower, upper = self.pending
        batch, bucket = len(pose), bucket_for(len(pose))
        key = (
            bucket,
            renderer.n_pixels,
            self.config,
            self.directional,
            pose.device,
            pose.dtype,
            tuple(renderer.views),
            renderer.config,
        )
        self.active_key = key
        hit = key in self.cache
        self.counts["hits" if hit else "misses"] += 1
        try:
            padded = pad_rows(pose, batch, bucket)
            padded_inputs = pad_inputs(inputs, batch, bucket)
            extra = (
                pad_rows(axes, batch, bucket),
                pad_rows(parallel, batch, bucket),
                pad_rows(enabled, batch, bucket),
                lower,
                upper,
            )
            if hit:
                graph = self.cache.pop(key)
                graph.configure_frame(padded_inputs, renderer, *extra)
            else:
                if len(self.cache) == len(BUCKETS):
                    self.cache.popitem(last=False)
                    self.counts["evictions"] += 1
                graph = self.graph_factory(
                    self.config,
                    padded,
                    padded_inputs,
                    renderer,
                    *extra,
                    directional=self.directional,
                )
                self.counts["captures"] += 1
            graph.replay(padded)
            self.cache[key] = graph
            self.counts["peak_resident"] = max(self.counts["peak_resident"], len(self.cache))
            for name in (
                "best_pose",
                "best_score",
                "best_iteration",
                "initial_score",
                "initial_parts",
            ):
                setattr(self, name, getattr(graph, name)[:batch])
            for name in ("score_ok", "gradient_ok", "pose_ok"):
                setattr(self, name, getattr(graph, name))
            self.trace = {name: value[:, :batch] for name, value in graph.trace.items()}
            self.counts["real_candidates"] += batch
            self.counts["padded_candidates"] += bucket
        except BaseException:
            self.counts["failures"] += 1
            self.discard_active()
            raise
        finally:
            self.pending = None
        return self

    @property
    def statistics(self):
        return dict(
            **self.counts, resident=len(self.cache), capacity=len(BUCKETS), buckets=list(BUCKETS)
        )


class BucketedFullFitDirectionalDeviceFitter(FrozenFitter):
    """Execution adapter; frozen parent's original-B final rendering is reused."""

    @property
    def bucket_statistics(self):
        proxy = self.__dict__.get("_bucket_proxy")
        return None if proxy is None else proxy.statistics

    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            return super()._fit(proposals, states, rays, targets)
        if self.__dict__.get("_bucket_inflight", False):
            raise RuntimeError("bucket fitter requires serial non-reentrant calls")
        key = parent_cache_key(self, proposals, targets)
        bucket_for(key[0])
        proxy = self.__dict__.get("_bucket_proxy")
        if proxy is None:
            proxy = self._bucket_proxy = BucketGraphProxy(self.config, self.directional)
        if proxy.config != self.config or proxy.directional != self.directional:
            raise ValueError("one immutable fitter config/directional policy required")
        # Give the unchanged parent its configure branch. Its real renderer still
        # checks N; any unexpected construction is rejected after the call.
        self._full_fit_key, self._full_fit_graph = key, proxy
        self._bucket_inflight = True
        try:
            result = super()._fit(proposals, states, rays, targets)
            if self._full_fit_graph is not proxy or self._full_fit_key != key:
                raise RuntimeError("frozen parent proxy interface changed")
            return result
        except BaseException:
            proxy.discard_active()
            self.__dict__.pop("_full_fit_graph", None)
            self.__dict__.pop("_full_fit_key", None)
            raise
        finally:
            self._bucket_inflight = False
