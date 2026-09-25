"""Anonymous fixed-query diagnostics; no fish IDs, geometry, or model imports.

LK uses full native frames and the frozen flow's numerical LK/corner parameters.
It deliberately retains query identity across frames instead of reselecting
corners or warping anatomical nodes. It is not production FlowObservations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag

import cv2
import numpy as np


@dataclass(frozen=True)
class PointConfig:
    max_corners: int = 64
    corner_quality: float = 0.02
    corner_spacing_px: float = 3.0
    core_kernel_size: int = 5
    window_size: int = 21
    pyramid_levels: int = 2
    min_eigenvalue: float = 1e-4
    max_fb_error_px: float = 1.0
    max_lk_error: float = 35.0
    max_feature_step_px: float = 20.0


CONFIG = PointConfig()


class Failure(IntFlag):
    FORWARD = 1
    BACKWARD = 2
    NONFINITE = 4
    FB = 8
    PHOTOMETRIC = 16
    STEP = 32
    OUTSIDE = 64
    ALREADY_LOST = 128


def require(ok, message):
    if not ok:
        raise ValueError(message)


def grayscale(image):
    image = np.asarray(image)
    require(image.dtype == np.uint8, "uint8 native image required")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    require(image.ndim == 3 and image.shape[2] == 3, "BGR or gray image required")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def select_queries(first_bgr, masks):
    """All anonymous masks in original order, including empty/coreless rows.

    None is an unavailable view; a zero-length stack is available but empty.
    Repeated query pixels from different masks retain separate references.
    No minimum-three-points restriction is imposed on this point diagnostic.
    """
    gray = grayscale(first_bgr)
    points, origins, rows = [], [], []
    if masks is None:
        return {
            "xy": np.empty((0, 2), np.float32),
            "mask_index": np.empty(0, np.int64),
            "rows": [],
            "mask_count": None,
            "available": False,
            "config": asdict(CONFIG),
        }
    masks = np.asarray(masks)
    require(
        masks.dtype == bool and masks.ndim == 3 and masks.shape[1:] == gray.shape,
        "bool N,H,W masks in native coordinates required",
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CONFIG.core_kernel_size, CONFIG.core_kernel_size)
    )
    for i, mask in enumerate(masks):
        core = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        count = int(np.count_nonzero(mask))
        core_count = int(np.count_nonzero(core))
        query = None
        reason = "empty_mask" if count == 0 else "empty_core" if core_count == 0 else None
        if reason is None:
            query = cv2.goodFeaturesToTrack(
                gray,
                CONFIG.max_corners,
                CONFIG.corner_quality,
                CONFIG.corner_spacing_px,
                mask=core,
                blockSize=3,
            )
            if query is None:
                reason = "no_corners"
        xy = np.empty((0, 2), np.float32) if query is None else query[:, 0].copy()
        require(np.isfinite(xy).all(), "nonfinite query")
        start = sum(len(p) for p in points)
        points.append(xy)
        origins.extend([i] * len(xy))
        rows.append(
            dict(
                mask_index=i,
                mask_pixels=count,
                core_pixels=core_count,
                query_start=start,
                query_stop=start + len(xy),
                query_count=len(xy),
                status="ready" if len(xy) else reason,
            )
        )
    return {
        "xy": np.concatenate(points) if points else np.empty((0, 2), np.float32),
        "mask_index": np.asarray(origins, np.int64),
        "rows": rows,
        "mask_count": len(masks),
        "available": True,
        "config": asdict(CONFIG),
    }


def lk_edge(previous_gray, current_gray, points):
    """One edge; preserve attempted points and measured rejection reasons.

    OpenCV can leave outputs undefined for failed status rows. Such coordinates
    and errors are stored as NaN, never passed to backward LK or called evidence.
    """
    before, after = grayscale(previous_gray), grayscale(current_gray)
    require(before.shape == after.shape, "native frame dimensions changed")
    xy = np.asarray(points)
    require(xy.dtype == np.float32 and xy.ndim == 2 and xy.shape[1] == 2, "float32 N,2 query")
    require(np.isfinite(xy).all(), "only finite active queries may reach LK")
    n = len(xy)
    result = dict(
        proposed_xy=np.full((n, 2), np.nan, np.float32),
        returned_xy=np.full((n, 2), np.nan, np.float32),
        forward_status=np.zeros(n, bool),
        backward_status=np.zeros(n, bool),
        lk_error=np.full(n, np.nan, np.float32),
        fb_error_px=np.full(n, np.nan, np.float32),
        step_px=np.full(n, np.nan, np.float32),
        failure_bits=np.zeros(n, np.uint16),
        valid=np.zeros(n, bool),
    )
    if not n:
        return result
    options = dict(
        winSize=(CONFIG.window_size, CONFIG.window_size),
        maxLevel=CONFIG.pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=CONFIG.min_eigenvalue,
    )
    target, status, error = cv2.calcOpticalFlowPyrLK(
        before, after, xy[:, None, :].copy(), None, **options
    )
    bits = result["failure_bits"]
    if target is None or status is None or error is None:
        bits[:] = Failure.FORWARD
        return result
    require(target.shape == (n, 1, 2), "LK output query count changed")
    status = status.reshape(-1).astype(bool)
    require(status.shape == (n,) and error.size == n, "LK status/error count changed")
    result["forward_status"][:] = status
    bits[~status] |= np.uint16(Failure.FORWARD)
    finite = np.isfinite(target[:, 0]).all(axis=1) & np.isfinite(error.reshape(-1))
    bits[status & ~finite] |= np.uint16(Failure.NONFINITE)
    selected = np.flatnonzero(status & finite)
    result["proposed_xy"][selected] = target[selected, 0]
    result["lk_error"][selected] = error.reshape(-1)[selected]
    if not len(selected):
        return result
    returned, back_status, _ = cv2.calcOpticalFlowPyrLK(
        after, before, target[selected].copy(), None, **options
    )
    result["step_px"][selected] = np.linalg.norm(target[selected, 0] - xy[selected], axis=1)
    if returned is None or back_status is None:
        bits[selected] |= np.uint16(Failure.BACKWARD)
    else:
        require(returned.shape == (len(selected), 1, 2), "backward LK query count changed")
        back_status = back_status.reshape(-1).astype(bool)
        require(back_status.shape == (len(selected),), "backward status count changed")
        result["backward_status"][selected] = back_status
        bits[selected[~back_status]] |= np.uint16(Failure.BACKWARD)
        finite_back = np.isfinite(returned[:, 0]).all(axis=1)
        bits[selected[back_status & ~finite_back]] |= np.uint16(Failure.NONFINITE)
        local = np.flatnonzero(back_status & finite_back)
        good = selected[local]
        result["returned_xy"][good] = returned[local, 0]
        result["fb_error_px"][good] = np.linalg.norm(returned[local, 0] - xy[good], axis=1)
    bits[result["fb_error_px"] > CONFIG.max_fb_error_px] |= np.uint16(Failure.FB)
    bits[result["lk_error"] > CONFIG.max_lk_error] |= np.uint16(Failure.PHOTOMETRIC)
    bits[result["step_px"] > CONFIG.max_feature_step_px] |= np.uint16(Failure.STEP)
    p = result["proposed_xy"]
    outside = (
        (p[:, 0] < 0) | (p[:, 0] >= before.shape[1]) | (p[:, 1] < 0) | (p[:, 1] >= before.shape[0])
    )
    bits[outside] |= np.uint16(Failure.OUTSIDE)
    result["valid"][:] = bits == 0
    return result


def lk_chain(frames, initial_xy, *, edge=lk_edge):
    """Run each consecutive frame once. Loss is terminal; never reseed."""
    require(len(frames) >= 1, "at least one frame required")
    xy = np.asarray(initial_xy)
    require(xy.dtype == np.float32 and xy.ndim == 2 and xy.shape[1] == 2, "float32 N,2 query")
    require(np.isfinite(xy).all(), "nonfinite initial query")
    t_count, n = len(frames), len(xy)
    empty = lk_edge(grayscale(frames[0]), grayscale(frames[0]), np.empty((0, 2), np.float32))
    arrays = {
        name: np.full((t_count, *value.shape), np.nan, value.dtype)
        if np.issubdtype(value.dtype, np.floating)
        else np.zeros((t_count, *value.shape), value.dtype)
        for name, value in empty.items()
    }
    # The zero-query edge defines fields and dtypes; allocate all original queries.
    arrays = {
        k: np.full((t_count, n, *v.shape[2:]), np.nan, v.dtype)
        if np.issubdtype(v.dtype, np.floating)
        else np.zeros((t_count, n, *v.shape[2:]), v.dtype)
        for k, v in arrays.items()
    }
    arrays["xy"] = np.full((t_count, n, 2), np.nan, np.float32)
    arrays["xy"][0] = xy
    arrays["valid"][0] = True
    arrays["attempted"] = np.zeros((t_count, n), bool)
    previous = grayscale(frames[0])
    for t in range(1, t_count):
        current = grayscale(frames[t])
        require(current.shape == previous.shape, "native image dimensions changed")
        active = np.flatnonzero(arrays["valid"][t - 1])
        arrays["failure_bits"][t] = Failure.ALREADY_LOST
        if len(active):
            outcome = edge(previous, current, arrays["xy"][t - 1, active].copy())
            require(set(outcome) == set(empty), "LK edge schema changed")
            for key, value in outcome.items():
                require(value.shape == (len(active), *empty[key].shape[1:]), "edge shape changed")
                arrays[key][t, active] = value
            arrays["attempted"][t, active] = True
            accepted = active[outcome["valid"]]
            arrays["xy"][t, accepted] = outcome["proposed_xy"][outcome["valid"]]
        previous = current
    return arrays


class EmissionLedger:
    """First emitted and revised prefix outputs, retaining actual input horizon."""

    def __init__(self, frames, queries):
        require(type(frames) is int and frames > 0, "positive frame count")
        require(type(queries) is int and queries >= 0, "nonnegative query count")
        self.frames, self.queries = frames, queries
        self.first_xy = np.full((frames, queries, 2), np.nan, np.float32)
        self.first_visibility = np.zeros((frames, queries), bool)
        self.first_horizon = np.full(frames, -1, np.int64)
        self.first_release_index = np.full(frames, -1, np.int64)
        self.final_xy = self.first_xy.copy()
        self.final_visibility = self.first_visibility.copy()
        self.final_horizon = self.first_horizon.copy()
        self.events = []

    def add(self, xy, visibility, *, input_start, input_stop):
        require(type(input_start) is int and type(input_stop) is int, "integer input interval")
        require(0 <= input_start < input_stop <= self.frames, "invalid input interval")
        if self.events:
            require(input_stop > self.events[-1]["input_stop"], "nonincreasing input horizon")
        xy, visibility = np.asarray(xy), np.asarray(visibility)
        prefix = len(xy)
        require(0 < prefix <= input_stop, "prediction extends beyond input horizon")
        require(xy.dtype == np.float32 and xy.shape == (prefix, self.queries, 2), "xy schema")
        require(
            visibility.dtype == bool and visibility.shape == (prefix, self.queries),
            "visibility schema",
        )
        if self.events:
            require(prefix >= self.events[-1]["output_prefix"], "output prefix shrank")
        new = np.flatnonzero(self.first_horizon[:prefix] == -1)
        self.first_xy[new] = xy[new]
        self.first_visibility[new] = visibility[new]
        self.first_horizon[new] = input_stop - 1
        self.first_release_index[new] = len(self.events)
        self.final_xy[:prefix] = xy
        self.final_visibility[:prefix] = visibility
        self.final_horizon[:prefix] = input_stop - 1
        self.events.append(
            dict(input_start=input_start, input_stop=input_stop, output_prefix=prefix)
        )

    def arrays(self):
        return {
            name: getattr(self, name).copy()
            for name in (
                "first_xy",
                "first_visibility",
                "first_horizon",
                "first_release_index",
                "final_xy",
                "final_visibility",
                "final_horizon",
            )
        }


def mask_memberships(xy, masks, shape):
    """All-mask membership; available-empty remains distinct from unavailable.

    Coordinates use rint (ties-to-even); continuous and rounded bounds are separate.
    This is independent-mask image consistency, not a point or animal ID label.
    """
    xy = np.asarray(xy)
    require(xy.ndim == 2 and xy.shape[1] == 2, "N,2 coordinates required")
    h, w = shape
    finite = np.isfinite(xy).all(axis=1)
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    pixel = np.full((len(xy), 2), -1, np.int64)
    roundable = finite & (np.abs(xy) < 2**62).all(axis=1)
    pixel[roundable] = np.rint(xy[roundable]).astype(np.int64)
    sampled = roundable & (pixel[:, 0] >= 0) & (pixel[:, 0] < w)
    sampled &= (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    if masks is None:
        return dict(
            available=False,
            finite=finite,
            in_image=inside,
            sampled=sampled,
            pixel_xy=pixel,
            membership=np.zeros((len(xy), 0), bool),
        )
    masks = np.asarray(masks)
    require(masks.dtype == bool and masks.ndim == 3 and masks.shape[1:] == (h, w), "mask schema")
    membership = np.zeros((len(xy), len(masks)), bool)
    membership[sampled] = masks[:, pixel[sampled, 1], pixel[sampled, 0]].T
    return dict(
        available=True,
        finite=finite,
        in_image=inside,
        sampled=sampled,
        pixel_xy=pixel,
        membership=membership,
    )


def sample_patches(gray_image, xy):
    """Exact bilinear 11x11 gray/255 samples, without outside padding."""
    gray = grayscale(gray_image)
    xy = np.asarray(xy, np.float64)
    require(xy.ndim == 2 and xy.shape[1] == 2, "N,2 points required")
    h, w = gray.shape
    finite = np.isfinite(xy).all(axis=1)
    inside = finite & (xy[:, 0] >= 5) & (xy[:, 0] <= w - 6)
    inside &= (xy[:, 1] >= 5) & (xy[:, 1] <= h - 6)
    patches = np.full((len(xy), 121), np.nan, np.float64)
    chosen = np.flatnonzero(inside)
    if len(chosen):
        dx, dy = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
        xx = xy[chosen, 0, None] + dx.ravel()[None]
        yy = xy[chosen, 1, None] + dy.ravel()[None]
        x0, y0 = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        wx, wy = xx - x0, yy - y0
        value = (
            gray[y0, x0] * (1 - wx) * (1 - wy)
            + gray[y0, x1] * wx * (1 - wy)
            + gray[y1, x0] * (1 - wx) * wy
            + gray[y1, x1] * wx * wy
        )
        patches[chosen] = value / 255.0
    return dict(values=patches, finite_center=finite, inside=inside)


def patch_comparison(a, b):
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 121, "patch shapes differ")
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    mad = np.full(len(a), np.nan, np.float64)
    zncc = np.full(len(a), np.nan, np.float64)
    variance_ok = np.zeros(len(a), bool)
    chosen = np.flatnonzero(valid)
    if len(chosen):
        aa, bb = a[chosen], b[chosen]
        mad[chosen] = np.mean(np.abs(aa - bb), axis=1)
        da, db = aa - aa.mean(axis=1, keepdims=True), bb - bb.mean(axis=1, keepdims=True)
        norm_a, norm_b = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        # A constant patch is unscorable for ZNCC, including floating mean roundoff.
        nonconstant = (np.ptp(aa, axis=1) > 0) & (np.ptp(bb, axis=1) > 0)
        nonconstant &= (norm_a > 0) & (norm_b > 0)
        good = chosen[nonconstant]
        variance_ok[good] = True
        zncc[good] = np.sum(da[nonconstant] * db[nonconstant], axis=1)
        zncc[good] /= norm_a[nonconstant] * norm_b[nonconstant]
    return dict(mad=mad, zncc=zncc, pair_available=valid, variance_ok=variance_ok)


def patch_diagnostics(frames, xy):
    """All query/frame samples. First frame is retained but not a tracking score."""
    xy = np.asarray(xy)
    require(xy.ndim == 3 and xy.shape[0] == len(frames) and xy.shape[2] == 2, "T,N,2 tracks")
    require(len(frames) > 0, "empty video")
    t_count, n = xy.shape[:2]
    output = {
        "finite_center": np.zeros((t_count, n), bool),
        "patch_inside": np.zeros((t_count, n), bool),
    }
    for prefix in ("initial", "adjacent"):
        for field in ("mad", "zncc"):
            output[f"{prefix}_{field}"] = np.full((t_count, n), np.nan, np.float64)
        for field in ("pair_available", "variance_ok"):
            output[f"{prefix}_{field}"] = np.zeros((t_count, n), bool)
    initial = previous = None
    for t, frame in enumerate(frames):
        sampled = sample_patches(frame, xy[t])
        current = sampled["values"]
        output["finite_center"][t] = sampled["finite_center"]
        output["patch_inside"][t] = sampled["inside"]
        if initial is None:
            initial = current.copy()
        for prefix, reference in (("initial", initial), ("adjacent", previous)):
            if reference is not None:
                for key, value in patch_comparison(reference, current).items():
                    output[f"{prefix}_{key}"][t] = value
        previous = current
    return output


def paired_metrics(a, b, a_reported, b_reported):
    """Retain full denominators and same-query paired differences per metric.

    Reported flags are algorithm-specific; their intersection is a conditional
    diagnostic subset, not a calibrated shared visibility estimate.
    """
    require(a.keys() == b.keys(), "metric fields differ")
    a_reported, b_reported = np.asarray(a_reported), np.asarray(b_reported)
    require(a_reported.dtype == bool and b_reported.dtype == bool, "bool reported flags")
    require(a_reported.shape == b_reported.shape and a_reported.ndim == 2, "T,N flags")
    output = {}
    for key in a:
        aa, bb = np.asarray(a[key]), np.asarray(b[key])
        if not np.issubdtype(aa.dtype, np.floating):
            continue
        require(aa.shape == bb.shape == a_reported.shape, "metric dimensions differ")
        af, bf = np.isfinite(aa), np.isfinite(bb)
        both = af & bf & a_reported & b_reported
        delta = np.full(aa.shape, np.nan, np.float64)
        delta[both] = bb[both] - aa[both]
        output[key] = dict(
            a_finite=af,
            b_finite=bf,
            common_reported=both,
            a_only=(af & a_reported) & ~(bf & b_reported),
            b_only=(bf & b_reported) & ~(af & a_reported),
            b_minus_a=delta,
        )
    return output
