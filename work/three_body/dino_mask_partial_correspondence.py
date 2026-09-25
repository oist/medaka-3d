"""Pure CPU crop and anonymous patch-correspondence arithmetic, protocol v1.

Native inputs are uint8 RGB and bool masks. Tight half-open bbox -> integer
square (odd spare pixel right/bottom); padding is float32 ImageNet mean. The
background control replaces native mask-exterior pixels BEFORE each condition's
single INTER_LINEAR resize to224. Native foreground values are identical;
resampled boundary values can differ. No model, files, GPU, learned thresholds,
top-N filtering, temporal propagation or identity assignment are used here.

Core is the existing5x5 ellipse zero-border erosion. Every native pixel with
positive-area overlap with an inverse14x14 patch-cell must be in that core.
All256 geometry rows and all token-pair distances/NN ties are retained.
See DINO_PARTIAL_PROTOCOL.md for the fixed coordinate and missing-value rules.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import cv2
import numpy as np

VERSION = "dino_mask_partial_correspondence_v1"
IMAGE_SIZE, PATCH_SIZE, GRID_SIZE = 224, 14, 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRAMES = (66524, 66600, 66628, 66703, 66800, 66908)
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def eroded_core(mask):
    """Exactly the existing ellipse erosion, without its color minimum12 gate."""
    _require(
        isinstance(mask, np.ndarray)
        and mask.dtype == bool
        and mask.ndim == 2
        and min(mask.shape) > 0,
        "nonempty native bool mask dimensions required",
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(
        mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)


def token_geometry(core, square_xyxy):
    """Keep every cell; integer bounds cover all positive-area native overlaps."""
    _require(
        isinstance(core, np.ndarray)
        and core.dtype == bool
        and core.ndim == 2
        and min(core.shape) > 0,
        "native bool core required",
    )
    square = np.asarray(square_xyxy)
    _require(square.shape == (4,) and square.dtype.kind in "iu", "integer square xyxy required")
    sx, sy, ex, ey = map(int, square)
    side = ex - sx
    _require(side > 0 and ey - sy == side, "positive square required")
    rows, cols = np.indices((GRID_SIZE, GRID_SIZE))
    crop_xy = np.stack(
        ((cols.ravel() + 0.5) * PATCH_SIZE - 0.5, (rows.ravel() + 0.5) * PATCH_SIZE - 0.5), axis=1
    )
    native_xy = (crop_xy + 0.5) * (side / IMAGE_SIZE) + np.array([sx, sy]) - 0.5
    bounds, area, counts, eligible = [], [], [], []
    height, width = core.shape
    for row, col in zip(rows.ravel(), cols.ravel(), strict=True):
        x0, y0 = sx + int(col) * side // GRID_SIZE, sy + int(row) * side // GRID_SIZE
        x1 = sx + ((int(col) + 1) * side + GRID_SIZE - 1) // GRID_SIZE
        y1 = sy + ((int(row) + 1) * side + GRID_SIZE - 1) // GRID_SIZE
        cell_area = (x1 - x0) * (y1 - y0)
        inside = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        count = int(core[cy0:cy1, cx0:cx1].sum()) if cx0 < cx1 and cy0 < cy1 else 0
        bounds.append((x0, y0, x1, y1))
        area.append(cell_area)
        counts.append(count)
        eligible.append(inside and count == cell_area)
    return dict(
        token_index=np.arange(GRID_SIZE**2, dtype=np.int64),
        token_center_crop_xy=crop_xy,
        token_center_native_xy=native_xy,
        token_native_xyxy=np.asarray(bounds, np.int64),
        token_footprint_pixels=np.asarray(area, np.int64),
        token_core_pixels=np.asarray(counts, np.int64),
        token_eligible=np.asarray(eligible, bool),
    )


def prepare_mask_crop(image_rgb, mask):
    """Return owned model inputs and full geometry; empty-mask images remain NaN.

    native->crop: u=(x-sx+.5)*224/L-.5. Native mask-exterior is replaced by
    ImageNet mean before either image is resized. Each resize is float32 linear;
    mask/native-region nearest-center maps are for display, not token admission.
    """
    _require(
        isinstance(image_rgb, np.ndarray)
        and image_rgb.dtype == np.uint8
        and image_rgb.ndim == 3
        and image_rgb.shape[-1] == 3
        and min(image_rgb.shape[:2]) > 0,
        "native uint8 RGB image required",
    )
    _require(
        isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == image_rgb.shape[:2],
        "full native bool mask required",
    )
    core = eroded_core(mask)
    mean, std = np.asarray(IMAGENET_MEAN, np.float32), np.asarray(IMAGENET_STD, np.float32)
    metadata = dict(
        schema=VERSION,
        native_shape_hw=list(mask.shape),
        image_rgb_sha256=_array_sha(image_rgb),
        mask_sha256=_array_sha(mask),
        core_sha256=_array_sha(core),
        mask_pixels=int(mask.sum()),
        core_pixels=int(core.sum()),
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        resize="OpenCV INTER_LINEAR float32; native mean-background replacement",
        background_mean_rgb=IMAGENET_MEAN,
        normalization_std_rgb=IMAGENET_STD,
    )
    if not mask.any():
        return dict(
            **metadata,
            crop_status="empty_mask",
            token_status="no_core_tokens",
            bbox_xyxy=np.full(4, -1, np.int64),
            square_xyxy=np.full(4, -1, np.int64),
            source_xyxy=np.full(4, -1, np.int64),
            padding_ltrb=np.zeros(4, np.int64),
            square_side=0,
            native_to_crop=np.full((3, 3), np.nan),
            crop_to_native=np.full((3, 3), np.nan),
            raw_rgb=np.full((IMAGE_SIZE, IMAGE_SIZE, 3), np.nan, np.float32),
            masked_rgb=np.full((IMAGE_SIZE, IMAGE_SIZE, 3), np.nan, np.float32),
            raw_normalized_chw=np.full((3, IMAGE_SIZE, IMAGE_SIZE), np.nan, np.float32),
            masked_normalized_chw=np.full((3, IMAGE_SIZE, IMAGE_SIZE), np.nan, np.float32),
            resized_mask=np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool),
            resized_native_region=np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool),
            token_index=np.arange(GRID_SIZE**2, dtype=np.int64),
            token_center_crop_xy=np.full((GRID_SIZE**2, 2), np.nan),
            token_center_native_xy=np.full((GRID_SIZE**2, 2), np.nan),
            token_native_xyxy=np.full((GRID_SIZE**2, 4), -1, np.int64),
            token_footprint_pixels=np.zeros(GRID_SIZE**2, np.int64),
            token_core_pixels=np.zeros(GRID_SIZE**2, np.int64),
            token_eligible=np.zeros(GRID_SIZE**2, bool),
            eligible_token_count=0,
        )
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    sx, sy = x0 - (side - (x1 - x0)) // 2, y0 - (side - (y1 - y0)) // 2
    ex, ey = sx + side, sy + side
    height, width = mask.shape
    ix0, iy0, ix1, iy1 = max(0, sx), max(0, sy), min(width, ex), min(height, ey)
    square = np.empty((side, side, 3), np.float32)
    square[:] = mean
    square_mask, native_region = np.zeros((side, side), bool), np.zeros((side, side), bool)
    target = np.s_[iy0 - sy : iy1 - sy, ix0 - sx : ix1 - sx]
    source = np.s_[iy0:iy1, ix0:ix1]
    square[target] = image_rgb[source].astype(np.float32) / 255
    square_mask[target], native_region[target] = mask[source], True
    controlled = square.copy()
    controlled[~square_mask] = mean
    raw = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    masked = cv2.resize(controlled, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    sample_index = ((2 * np.arange(IMAGE_SIZE, dtype=np.int64) + 1) * side) // (2 * IMAGE_SIZE)
    scale = IMAGE_SIZE / side
    affine = np.array(
        [[scale, 0, (0.5 - sx) * scale - 0.5], [0, scale, (0.5 - sy) * scale - 0.5], [0, 0, 1]],
        np.float64,
    )
    inverse = np.array(
        [[1 / scale, 0, sx + 0.5 / scale - 0.5], [0, 1 / scale, sy + 0.5 / scale - 0.5], [0, 0, 1]],
        np.float64,
    )
    geometry = token_geometry(core, np.array([sx, sy, ex, ey], np.int64))
    count = int(geometry["token_eligible"].sum())
    return dict(
        **metadata,
        crop_status="ready",
        token_status="available" if count else "no_core_tokens",
        bbox_xyxy=np.array([x0, y0, x1, y1], np.int64),
        square_xyxy=np.array([sx, sy, ex, ey], np.int64),
        square_side=side,
        source_xyxy=np.array([ix0, iy0, ix1, iy1], np.int64),
        padding_ltrb=np.array([ix0 - sx, iy0 - sy, ex - ix1, ey - iy1], np.int64),
        native_to_crop=affine,
        crop_to_native=inverse,
        raw_rgb=raw,
        masked_rgb=masked,
        raw_normalized_chw=np.ascontiguousarray(((raw - mean) / std).transpose(2, 0, 1)),
        masked_normalized_chw=np.ascontiguousarray(((masked - mean) / std).transpose(2, 0, 1)),
        resized_mask=square_mask[np.ix_(sample_index, sample_index)].copy(),
        resized_native_region=native_region[np.ix_(sample_index, sample_index)].copy(),
        eligible_token_count=count,
        **geometry,
    )


def normalize_tokens(tokens):
    """Scale before L2; keep zero/nonfinite rows invalid without epsilon gating."""
    _require(
        isinstance(tokens, np.ndarray)
        and tokens.ndim == 2
        and tokens.dtype.kind == "f"
        and tokens.shape[1] > 0,
        "floating [tokens,features] array required",
    )
    values = tokens.astype(np.float64, copy=True)
    finite = np.isfinite(values).all(axis=1)
    scale = np.full(len(values), np.nan)
    scale[finite] = np.max(np.abs(values[finite]), axis=1)
    valid = finite & (scale > 0)
    normalized, norms = np.full(values.shape, np.nan), np.full(len(values), np.nan)
    norms[finite & ~valid] = 0
    scaled = values[valid] / scale[valid, None]
    scaled_norm = np.sqrt(np.sum(scaled * scaled, axis=1))
    normalized[valid] = scaled / scaled_norm[:, None]
    with np.errstate(over="ignore"):
        norms[valid] = scale[valid] * scaled_norm
    return dict(
        normalized=normalized,
        input_finite=finite,
        nonzero_finite=valid.copy(),
        valid=valid,
        norm=norms,
    )


def _geometry(values, count):
    eligible = np.asarray(values["token_eligible"])
    crop = np.asarray(values["token_center_crop_xy"], dtype=np.float64)
    native = np.asarray(values["token_center_native_xy"], dtype=np.float64)
    _require(
        eligible.dtype == bool
        and eligible.shape == (count,)
        and crop.shape == native.shape == (count, 2),
        "token geometry shapes differ",
    )
    return (
        eligible.copy(),
        crop.copy(),
        native.copy(),
        np.isfinite(crop).all(1) & np.isfinite(native).all(1),
    )


def _distribution(values):
    _require(np.isfinite(values).all(), "distribution requires finite distances")
    if not len(values):
        return dict(
            count=0,
            minimum=np.nan,
            q25=np.nan,
            median=np.nan,
            q75=np.nan,
            maximum=np.nan,
            mean=np.nan,
        )
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
    return dict(
        count=len(values),
        minimum=float(quantiles[0]),
        q25=float(quantiles[1]),
        median=float(quantiles[2]),
        q75=float(quantiles[3]),
        maximum=float(quantiles[4]),
        mean=float(values.mean()),
    )


def token_correspondence(left_tokens, right_tokens, left_geometry, right_geometry):
    """All cosine pairs and directed/mutual exact NN ties, never an ID score.

    Geometry dictionaries may come from prepare_mask_crop or synthetic K-row
    geometry. Coverage's requested denominator includes invalid selected tokens.
    If either side has no usable token, rates are NaN rather than measured zero.
    """
    left, right = normalize_tokens(left_tokens), normalize_tokens(right_tokens)
    _require(left_tokens.shape[1] == right_tokens.shape[1], "token feature dimensions differ")
    le, lc, ln, lf = _geometry(left_geometry, len(left_tokens))
    re, rc, rn, rf = _geometry(right_geometry, len(right_tokens))
    lv, rv = le & lf & left["valid"], re & rf & right["valid"]
    li, ri = np.flatnonzero(lv), np.flatnonzero(rv)
    comparable = bool(len(li) and len(ri))
    distances = np.full((len(lv), len(rv)), np.nan)
    left_min, right_min = np.full(len(lv), np.nan), np.full(len(rv), np.nan)
    if comparable:
        similarity = left["normalized"][li] @ right["normalized"][ri].T
        sub = 1 - np.clip(similarity, -1, 1)
        distances[np.ix_(li, ri)] = sub
        left_min[li], right_min[ri] = sub.min(1), sub.min(0)
    finite = np.isfinite(distances)
    forward = finite & (distances == left_min[:, None])
    reverse = finite & (distances == right_min[None, :])
    mutual = forward & reverse
    pairs = np.argwhere(forward | reverse)
    pi, pj = pairs[:, 0], pairs[:, 1]
    matched_left, matched_right = mutual.any(1), mutual.any(0)

    def side(eligible, valid, coordinate_finite, normalization, matched, nn_count):
        selected, usable, matched_count = int(eligible.sum()), int(valid.sum()), int(matched.sum())
        return dict(
            eligible_count=selected,
            valid_count=usable,
            matched_count=matched_count,
            invalid_eligible_count=int((eligible & ~valid).sum()),
            matched_fraction=matched_count / selected if comparable else np.nan,
            matched_valid_fraction=matched_count / usable if comparable else np.nan,
            tie_fraction=float((nn_count[valid] > 1).mean()) if comparable else np.nan,
            eligible=eligible,
            coordinate_finite=coordinate_finite,
            usable=valid,
            normalization=normalization,
        )

    status = (
        "comparable"
        if comparable
        else "no_eligible_tokens"
        if not le.any() or not re.any()
        else "no_valid_tokens"
    )
    if comparable and ((le & ~lv).any() or (re & ~rv).any()):
        status = "partial_invalid_tokens"
    return dict(
        schema=VERSION,
        status=status,
        comparable=comparable,
        distances=distances,
        finite_pair=finite,
        forward_nn=forward,
        reverse_nn=reverse,
        mutual_nn=mutual,
        left_minimum=left_min,
        right_minimum=right_min,
        left=side(le, lv, lf, left, matched_left, forward.sum(1)),
        right=side(re, rv, rf, right, matched_right, reverse.sum(0)),
        correspondences=dict(
            pairs=pairs,
            distances=distances[pi, pj],
            left_crop_xy=lc[pi],
            right_crop_xy=rc[pj],
            left_native_xy=ln[pi],
            right_native_xy=rn[pj],
            forward=forward[pi, pj],
            reverse=reverse[pi, pj],
            mutual=mutual[pi, pj],
        ),
        finite_distance_distribution=_distribution(distances[finite]),
        mutual_distance_distribution=_distribution(distances[mutual]),
    )


def enumerate_mask_pairs(mask_counts):
    """All45 fixed pair/view requests; None observation and zero masks stay distinct."""
    _require(
        set(mask_counts) == {(frame, view) for frame in FRAMES for view in VIEWS},
        "all18 fixed frame/view mask counts required",
    )
    _require(
        all(value is None or (type(value) is int and value >= 0) for value in mask_counts.values()),
        "nonnegative Python mask counts or None required",
    )
    result = []
    for early, late in combinations(sorted(FRAMES), 2):
        for view in VIEWS:
            left, right = mask_counts[early, view], mask_counts[late, view]
            available = left is not None and right is not None
            pairs = np.asarray(
                [(i, j) for i in range(left or 0) for j in range(right or 0)], dtype=np.int64
            ).reshape(-1, 2)
            result.append(
                dict(
                    earlier_frame=early,
                    later_frame=late,
                    view=view,
                    earlier_mask_count=left,
                    later_mask_count=right,
                    status="missing_observation"
                    if not available
                    else "available"
                    if len(pairs)
                    else "no_mask_pairs",
                    mask_pairs=pairs,
                )
            )
    return result
