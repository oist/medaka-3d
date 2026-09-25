"""Four-entry LRU ownership around the frozen complete-fit CUDA implementation.

Only graph residency changes. The parent still constructs/configures each graph,
resets every replay, performs all15 evaluations/14 updates, validates finite
outputs, and copies all diagnostics/trace/return arrays. No model math is copied.
CPU delegates directly. This component has no actual GPU qualification yet.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from full_fit_graph_fitter import FullFitDirectionalDeviceFitter as FrozenFullFitFitter

PARENT_SOURCE_SHA = "3255c53de0e7ed4ca02872229a3bd198160b9651b556bcd0c121caf9c339a376"
CACHE_CAPACITY = 4
POLICY_SCHEMA = "full_fit_graph_lru4_v1"


def parent_cache_key(fitter, proposals, targets):
    """Same B/N/config/directional key; N mirrors renderer's ragged-view padding.

    This only reads host pixel metadata. Parent renderer remains responsible for
    all physical ray, input-value, dtype/device and extra-view validation.
    """
    if targets is None:
        raise ValueError("full-fit cache requires original D2 targets")
    counts = []
    for view in fitter.rig.views:
        pixels = np.asarray(targets.pixels.get(view, np.empty((0, 2))), dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixel arrays must have shape (N, 2)")
        counts.append(len(pixels))
    if not counts or max(counts) == 0:
        raise ValueError("at least one sampled image pixel is required")
    return len(proposals), max(counts), fitter.config, fitter.directional


class FullFitDirectionalDeviceFitter(FrozenFullFitFitter):
    """Fixed LRU4 adapter; one serial caller per fitter instance.

    Successful graphs alone enter the cache. The parent active alias refers to
    an existing cached graph, not a fifth graph. On a miss at capacity, eviction
    happens before parent construction. External references/exception tracebacks
    can extend a graph's lifetime; callers retain their existing GPU memory guard.
    No global allocator, stream, capture-pool or configuration settings change.
    """

    @property
    def full_fit_cache_policy(self):
        return dict(
            schema=POLICY_SCHEMA,
            capacity=CACHE_CAPACITY,
            eviction="least recently successful use; evict before capture",
            key_fields=["batch_size", "renderer_n_pixels", "config", "directional"],
            parent_source_sha256=PARENT_SOURCE_SHA,
            numerical_path="frozen parent _fit, configure_frame and replay unchanged",
            failure="discard active graph/key; retain other healthy entries",
            CPU="direct frozen parent dispatch; no LRU operations",
            ownership_scope="at most4 adapter-owned graph objects including parent active alias",
            GPU_qualification="not established by this source-only adapter",
        )

    @property
    def full_fit_cache_statistics(self):
        """Owned-residency counters; capture count remains the parent's counter.

        Misses/hits count selected attempts, including failures. Rejections before
        selecting a key do not count. Values are fresh host scalars, not GPU polls.
        """
        return dict(
            capacity=CACHE_CAPACITY,
            captures=self.__dict__.get("_full_fit_capture_count", 0),
            hits=self.__dict__.get("_full_fit_cache4_hits", 0),
            misses=self.__dict__.get("_full_fit_cache4_misses", 0),
            evictions=self.__dict__.get("_full_fit_cache4_evictions", 0),
            failures=self.__dict__.get("_full_fit_cache4_failures", 0),
            resident=len(self.__dict__.get("_full_fit_cache4", ())),
            peak_resident=self.__dict__.get("_full_fit_cache4_peak_resident", 0),
        )

    def _record_cache4_owned_peak(self, cache):
        # Called before inserting the active graph into cache. A hit has already
        # been removed from cache, so the alias adds exactly one owned graph.
        owned = len(cache) + int("_full_fit_graph" in self.__dict__)
        self._full_fit_cache4_peak_resident = max(
            self.__dict__.get("_full_fit_cache4_peak_resident", 0), owned
        )

    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            return super()._fit(proposals, states, rays, targets)
        if self.__dict__.get("_full_fit_cache4_inflight", False):
            raise RuntimeError("full-fit cache4 requires serial non-reentrant calls")
        # A malformed key fails before any graph is selected or configured.
        key = parent_cache_key(self, proposals, targets)
        cache = self.__dict__.setdefault("_full_fit_cache4", OrderedDict())
        if not isinstance(cache, OrderedDict) or len(cache) > CACHE_CAPACITY:
            raise RuntimeError("invalid full-fit LRU4 cache state")
        self._full_fit_cache4_inflight = True
        hit = key in cache
        evicted = None
        # Remove the parent's most recent alias first, so eviction really releases
        # the adapter's ownership before the next capture is allocated.
        self.__dict__.pop("_full_fit_graph", None)
        self.__dict__.pop("_full_fit_key", None)
        if hit:
            self._full_fit_graph = cache.pop(key)
            self._full_fit_key = key
            self._full_fit_cache4_hits = self.__dict__.get("_full_fit_cache4_hits", 0) + 1
        else:
            self._full_fit_cache4_misses = self.__dict__.get("_full_fit_cache4_misses", 0) + 1
            if len(cache) == CACHE_CAPACITY:
                evicted = next(iter(cache))
                del cache[evicted]
                self._full_fit_cache4_evictions = (
                    self.__dict__.get("_full_fit_cache4_evictions", 0) + 1
                )
        try:
            result = super()._fit(proposals, states, rays, targets)
            self._record_cache4_owned_peak(cache)
            # Parent cache key is also checked after its real renderer determines N.
            if self.__dict__.get("_full_fit_key") != key or "_full_fit_graph" not in self.__dict__:
                raise RuntimeError("frozen parent graph/key contract differs")
            cache[key] = self._full_fit_graph
            if len(cache) > CACHE_CAPACITY:
                raise RuntimeError("full-fit LRU4 exceeded fixed capacity")
        except BaseException:
            if key not in cache:
                self._record_cache4_owned_peak(cache)
            cache.pop(key, None)
            self.__dict__.pop("_full_fit_graph", None)
            self.__dict__.pop("_full_fit_key", None)
            self._full_fit_cache4_failures = self.__dict__.get("_full_fit_cache4_failures", 0) + 1
            self.last_full_fit_cache4 = dict(
                status="failed", hit=hit, resident=len(cache), evicted=evicted is not None
            )
            raise
        finally:
            self._full_fit_cache4_inflight = False
        self.last_full_fit_cache4 = dict(
            status="ok", hit=hit, resident=len(cache), evicted=evicted is not None
        )
        return result


# Explicit import name for qualified callers; it is the same class object.
Cache4FullFitDirectionalDeviceFitter = FullFitDirectionalDeviceFitter
