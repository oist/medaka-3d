"""Label-free appearance evidence for linking geometry fragments.

Nothing in this module reads reference identity labels. Fragment membership and
coexistence come from geometry only; evaluation lives with the runner and the
shared ``medaka3d.eval`` functions.
"""

from __future__ import annotations

from itertools import pairwise, permutations

import cv2
import numpy as np

from medaka3d.types import Identity

PERMS = np.array(list(permutations(range(3))), dtype=np.int8)
MOUTH, PECTORAL, TAIL = 0, 1, 2
NATIVE_CROP_PX = 160
OUT_PX = 64
TRAIN_PX = 56
OVERLAP_RADIUS_PX = 80.0
MIN_CROPS = 5
COLOR_FEATURES = (
    "median_r_fraction",
    "median_g_fraction",
    "mean_s_cos_h_scaled",
    "mean_s_sin_h_scaled",
    "median_saturation",
    "median_value",
)


# --------------------------------------------------------------------------- crops


def crop_pose(points: np.ndarray):
    """Centre, heading (radians, tail-to-mouth) and mouth-tail length, or None.

    ``points`` is (3 nodes, xy) in native pixels with NaN for missing nodes.
    """
    have = np.isfinite(points).all(axis=1)
    if have[MOUTH] and have[TAIL]:
        centre = (points[MOUTH] + points[TAIL]) / 2
        axis = points[MOUTH] - points[TAIL]
    elif have[PECTORAL] and have[MOUTH]:
        centre, axis = points[PECTORAL], points[MOUTH] - points[PECTORAL]
    elif have[PECTORAL] and have[TAIL]:
        centre, axis = points[PECTORAL], points[PECTORAL] - points[TAIL]
    else:
        return None
    norm = float(np.hypot(*axis))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    length = np.nan
    if have[MOUTH] and have[TAIL]:
        length = float(np.hypot(*(points[MOUTH] - points[TAIL])))
    return centre.astype(float), float(np.arctan2(axis[1], axis[0])), length


def crop_matrix(centre, heading, native=NATIVE_CROP_PX, out=OUT_PX):
    """Affine map from image to crop: heading points to +x, centre to the crop centre."""
    scale = out / native
    c, s = np.cos(-heading) * scale, np.sin(-heading) * scale
    rot = np.array([[c, -s], [s, c]])
    shift = np.array([out / 2, out / 2]) - rot @ np.asarray(centre, float)
    return np.hstack([rot, shift[:, None]])


def extract_crop(image: np.ndarray, centre, heading, out: int = OUT_PX) -> np.ndarray:
    """The one crop builder used for training, features and inference.

    ``out`` is the crop side in pixels (Registration 5 uses 128); the native
    160 px window is the same.
    """
    matrix = crop_matrix(centre, heading, out=out)
    return cv2.warpAffine(
        image, matrix, (out, out), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT
    )


def train_px(out: int = OUT_PX) -> int:
    """Training/inference window for a crop side: 7/8 of it (56 of 64, 112 of 128)."""
    return out * TRAIN_PX // OUT_PX


def overlaps(frame_points: np.ndarray, own_row: int, centre) -> bool:
    """True when another detection row of this camera-frame lies near the crop centre."""
    others = np.delete(frame_points, own_row, axis=0).reshape(-1, 2)
    others = others[np.isfinite(others).all(axis=1)]
    if not others.size:
        return False
    return bool((np.hypot(*(others - centre).T) < OVERLAP_RADIUS_PX).any())


# --------------------------------------------------------------------------- pose masks
# Registration 2. Geometry in units of the row's own mouth-tail length.

MASK_HALF_WIDTH = 0.14
MASK_TAIL_EXTENSION = 0.15
MASK_MOUTH_EXTENSION = 0.05
DEFAULT_LENGTH_NATIVE = 0.7 * NATIVE_CROP_PX
MAX_OCCLUDED_FRACTION = 0.2


def to_crop(points: np.ndarray, centre, heading, out: int = OUT_PX) -> np.ndarray:
    """Native image points (n, 2) to crop pixel coordinates; NaN stays NaN."""
    matrix = crop_matrix(centre, heading, out=out)
    return points @ matrix[:, :2].T + matrix[:, 2]


def band_mask(points_crop: np.ndarray, length_native: float, out: int = OUT_PX) -> np.ndarray:
    """Pixels of a body band along mouth, pectoral and tail points, in crop space.

    The band runs from ``MASK_MOUTH_EXTENSION`` beyond the mouth end to
    ``MASK_TAIL_EXTENSION`` beyond the tail end, with half-width
    ``MASK_HALF_WIDTH``, all as fractions of the row's mouth-tail length. OpenCV
    draws thick lines with round ends, so each end reaches a further half-width.
    Fewer than two points give an empty mask. ``out`` is the crop side.
    """
    mask = np.zeros((out, out), np.uint8)
    have = np.isfinite(points_crop).all(axis=1)
    chain = [points_crop[k] for k in (MOUTH, PECTORAL, TAIL) if have[k]]
    if len(chain) < 2:
        return mask.astype(bool)
    length = length_native if np.isfinite(length_native) else DEFAULT_LENGTH_NATIVE
    length *= out / NATIVE_CROP_PX
    front, back = chain[0], chain[-1]
    axis = front - back
    norm = float(np.hypot(*axis))
    if norm < 1e-6:
        return mask.astype(bool)
    axis /= norm
    front_extension = MASK_MOUTH_EXTENSION if have[MOUTH] else 0.0
    back_extension = MASK_TAIL_EXTENSION if have[TAIL] else 0.0
    chain[0] = front + axis * front_extension * length
    chain[-1] = back - axis * back_extension * length
    # Sub-pixel polyline: 4 fractional bits.
    poly = np.round(np.array(chain) * 16).astype(np.int32)
    thickness = max(1, round(2 * MASK_HALF_WIDTH * length))
    cv2.polylines(mask, [poly], False, 1, thickness, cv2.LINE_8, shift=4)
    return mask.astype(bool)


def row_length(points: np.ndarray) -> float:
    if np.isfinite(points[[MOUTH, TAIL]]).all():
        return float(np.hypot(*(points[MOUTH] - points[TAIL])))
    return np.nan


def mask_crop(
    crop: np.ndarray, frame_points: np.ndarray, own_row: int, centre, heading, fill: np.ndarray
):
    """Fill other detection rows' bands; return (masked crop, occluded own-band fraction)."""
    out = crop.shape[0]
    others = np.zeros((out, out), bool)
    for r in range(frame_points.shape[0]):
        if r != own_row:
            pts = frame_points[r]
            others |= band_mask(to_crop(pts, centre, heading, out), row_length(pts), out)
    own_pts = frame_points[own_row]
    own = band_mask(to_crop(own_pts, centre, heading, out), row_length(own_pts), out)
    occluded = float((own & others).sum() / own.sum()) if own.any() else 1.0
    out = crop.copy()
    out[others] = fill
    return out, occluded


# --------------------------------------------------------------------------- colour


def body_mask(length_out: float, out: int = OUT_PX) -> np.ndarray:
    length = length_out if np.isfinite(length_out) else 0.5 * out
    a = max(2, round(0.35 * length))
    b = max(2, round(0.07 * length))
    mask = np.zeros((out, out), np.uint8)
    cv2.ellipse(mask, (out // 2, out // 2), (a, b), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def color_features(crop: np.ndarray, length_native: float) -> np.ndarray:
    """Six bounded colour values of `labs/20260909_three_body`, on a body-axis ellipse."""
    out = crop.shape[0]
    mask = body_mask(length_native * out / NATIVE_CROP_PX, out)
    bgr = crop[mask].astype(np.float32) / 255
    normalized = bgr / np.maximum(bgr.sum(-1, keepdims=True), 1e-8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angle, saturation, value = np.deg2rad(hsv[:, 0]), hsv[:, 1], hsv[:, 2]
    return np.array(
        [
            np.median(normalized[:, 2]),
            np.median(normalized[:, 1]),
            (1 + np.mean(saturation * np.cos(angle))) / 2,
            (1 + np.mean(saturation * np.sin(angle))) / 2,
            np.median(saturation),
            np.median(value),
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- contrastive


ENCODERS = ("cnn4", "resnet18_8d")


def build_encoder(dim: int = 16, kind: str = "cnn4"):
    """``cnn4``: four conv blocks, 16-d (Registrations 1--4). ``resnet18_8d``:
    torchvision ResNet18 from random initialisation with a bias-free 8-d head,
    idtracker.ai v6's embedding (Registration 5, A). ``model.out_dim`` records
    the output width.
    """
    import torch.nn as nn

    if kind == "resnet18_8d":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8, bias=False)
        model.out_dim = 8
        return model
    if kind != "cnn4":
        raise ValueError(f"Unknown encoder {kind!r}; choose from {ENCODERS}")

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    model = nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        block(128, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, dim),
    )
    model.out_dim = dim
    return model


def channel_stats(crops: np.ndarray, chunk: int = 4096):
    """Per-channel mean and population standard deviation of crops scaled to [0, 1].

    Two passes over chunks of crops, accumulated in float64, so the whole array
    is never converted at once: a 128 px crop set converted in one piece would
    need tens of GiB. Numerically the same statistic as ``x.mean(0)`` and
    ``x.std(0)`` on the full float64 array, up to summation order.
    """
    n = 0
    total = np.zeros(3, np.float64)
    for start in range(0, len(crops), chunk):
        x = crops[start : start + chunk].reshape(-1, 3).astype(np.float64) / 255
        n += len(x)
        total += x.sum(0)
    mean = total / max(n, 1)
    squares = np.zeros(3, np.float64)
    for start in range(0, len(crops), chunk):
        x = crops[start : start + chunk].reshape(-1, 3).astype(np.float64) / 255
        squares += ((x - mean) ** 2).sum(0)
    std = np.sqrt(squares / max(n, 1))
    return mean.astype(np.float32), std.astype(np.float32)


def _to_tensor(crops, idx, stats, offsets=None, device=None):
    import torch

    mean, std = stats
    out = crops.shape[1]
    train = train_px(out)
    lo = (out - train) // 2
    if offsets is None:
        batch = crops[idx, lo : lo + train, lo : lo + train]
    else:
        pairs = zip(idx, offsets, strict=True)
        batch = np.stack([crops[i, y : y + train, x : x + train] for i, (y, x) in pairs])
    batch = (batch.astype(np.float32) / 255 - mean) / std
    # NHWC storage viewed as NCHW is the channels-last layout the encoder uses.
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2)
    return tensor if device is None else tensor.to(device, non_blocking=True)


def model_device(model):
    return next(model.parameters()).device


def pair_sampler(fragment: np.ndarray, interval: np.ndarray, usable: np.ndarray, seed: int):
    """Yield (anchor, positive, negative) crop indices from geometry structure only.

    Positives share a fragment; negatives come from another fragment of the same
    interval (temporally coexisting, hence a different fish). Intervals are drawn
    in proportion to their usable crop count among those with >=2 fragments.
    """
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(usable)
    by_fragment: dict[int, np.ndarray] = {}
    for f in np.unique(fragment[rows]):
        members = rows[fragment[rows] == f]
        if members.size >= 2:
            by_fragment[int(f)] = members
    by_interval: dict[int, list[int]] = {}
    for f, members in by_fragment.items():
        by_interval.setdefault(int(interval[members[0]]), []).append(f)
    keys = [k for k, v in by_interval.items() if len(v) >= 2]
    if not keys:
        raise ValueError("no interval has two usable coexisting fragments")
    weight = np.array([sum(by_fragment[f].size for f in by_interval[k]) for k in keys], float)
    weight /= weight.sum()

    def draw(n):
        anchors, positives, negatives = [], [], []
        for k in rng.choice(keys, size=n, p=weight):
            f, g = rng.choice(by_interval[int(k)], size=2, replace=False)
            a, p = rng.choice(by_fragment[int(f)], size=2, replace=False)
            anchors.append(a)
            positives.append(p)
            negatives.append(rng.choice(by_fragment[int(g)]))
        return np.array(anchors), np.array(positives), np.array(negatives)

    return draw, rng


def train_contrastive(
    crops,
    fragment,
    interval,
    usable,
    *,
    steps=3000,
    pairs=128,
    seed=20261106,
    lr=1e-3,
    weight_decay=1e-4,
    log_every=100,
    monitor=None,
    encoder="cnn4",
    device=None,
):
    """Contrastive training on geometry fragments; returns (model, stats, loss log).

    Without ``monitor`` training runs exactly ``steps`` updates (Registrations
    1--3). With a ``SilhouetteMonitor`` (Registration 4) it is called after
    every ``log_every`` updates, from 0, may stop training, and its best
    parameters are restored at the end; ``steps`` is then the cap. ``device``
    defaults to CUDA when available (Registration 5), else the CPU.
    """
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    stats = channel_stats(crops[usable])
    model = build_encoder(kind=encoder).to(device).to(memory_format=torch.channels_last)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    draw, rng = pair_sampler(fragment, interval, usable, seed)
    out = crops.shape[1]
    span = out - train_px(out) + 1
    log = []
    model.train()
    for step in range(steps):
        if monitor is not None and step % log_every == 0 and monitor(step, model, crops, stats):
            break
        a, p, n = draw(pairs)
        idx = np.concatenate([a, p, a, n])
        offsets = rng.integers(0, span, size=(idx.size, 2))
        z = model(_to_tensor(crops, idx, stats, offsets, device))
        za, zp, za2, zn = z.split(pairs)
        d_pos = (za - zp).norm(dim=1)
        d_neg = (za2 - zn).norm(dim=1)
        loss = (d_pos**2).mean() + (torch.relu(1.0 - d_neg) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % log_every == 0 or step == steps - 1:
            log.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "d_pos_median": d_pos.median().item(),
                    "d_neg_median": d_neg.median().item(),
                }
            )
    else:
        if monitor is not None and steps % log_every == 0:
            monitor(steps, model, crops, stats)
    if monitor is not None and monitor.best is not None:
        model.load_state_dict(monitor.best[2])
    model.eval()
    return model, stats, log


def embed(model, crops, stats, batch=None):
    """Embed crops in inference batches; the result does not depend on the batch size.

    The default batch holds 1024 crops of 64 px and shrinks with the crop area
    (256 at 128 px) so that the first block's activations stay under about
    1 GB on the GPU; three processes embedding 128 px crops at 1024 exhausted
    a 16 GB card and spilled into system memory.
    """
    import torch

    if batch is None:
        side = crops.shape[1] if crops.ndim == 4 else OUT_PX
        batch = max(64, 1024 * OUT_PX * OUT_PX // (side * side))
    device = model_device(model)
    out = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            idx = np.arange(start, min(start + batch, len(crops)))
            out.append(model(_to_tensor(crops, idx, stats, device=device)).cpu().numpy())
    if out:
        return np.concatenate(out)
    return np.zeros((0, getattr(model, "out_dim", 16)), np.float32)


# --------------------------------------------------------------------------- decisions


def descriptor(features: np.ndarray, rows: np.ndarray):
    """(mean feature, crop count) over usable rows, or None below the minimum count."""
    if rows.size < MIN_CROPS:
        return None
    return features[rows].mean(axis=0), int(rows.size)


def pair_distance(pre: dict, post: dict) -> float:
    """Mean over shared cameras of Euclidean distance; NaN without a shared camera.

    ``pre`` and ``post`` map camera name to (vector, crop count).
    """
    shared = [v for v in pre if v in post]
    if not shared:
        return np.nan
    return float(np.mean([np.linalg.norm(pre[v][0] - post[v][0]) for v in shared]))


def decide(distance: np.ndarray):
    """Minimum summed-distance permutation and its margin; None when undecidable.

    ``distance[s, t]`` compares pre slot ``s`` with post slot ``t``; the returned
    permutation maps pre slot ``s`` to post slot ``perm[s]``.
    """
    if not np.isfinite(distance).all():
        return None, np.nan
    costs = np.array([distance[np.arange(3), q].sum() for q in PERMS])
    order = np.argsort(costs, kind="stable")
    return PERMS[order[0]].copy(), float(costs[order[1]] - costs[order[0]])


def constrained_labels(descriptors: list[list[dict | None]], initial: np.ndarray, rounds=50):
    """Assign each interval's three fragments to three identities, one each.

    ``descriptors[i][s]`` maps camera to (vector, crop count), or is None.
    ``initial`` gives the identity of interval 0's slots. An interval with a
    missing or empty fragment keeps -1 for all three slots: a partial assignment
    would not be one-to-one evidence. Returns (labels[i, s], rounds used).
    """
    n = len(descriptors)
    labels = np.full((n, 3), -1, np.int8)
    labels[0] = initial
    cameras = sorted({v for row in descriptors for d in row if d for v in d})
    centroids = {v: [None, None, None] for v in cameras}
    for s, k in enumerate(initial):
        for v, (vec, _) in (descriptors[0][s] or {}).items():
            centroids[v][k] = vec
    used = 0
    while used < rounds:
        used += 1
        new = labels.copy()
        for i, row in enumerate(descriptors):
            if i == 0 or not all(row):
                continue
            cost = np.full((3, 3), np.nan)
            for s in range(3):
                for k in range(3):
                    terms = [
                        np.linalg.norm(vec - centroids[v][k])
                        for v, (vec, _) in row[s].items()
                        if centroids[v][k] is not None
                    ]
                    cost[s, k] = np.mean(terms) if terms else np.nan
            perm, _ = decide(cost)
            if perm is not None:
                new[i] = perm
        for v in cameras:
            for k in range(3):
                vecs, weights = [], []
                for i, row in enumerate(descriptors):
                    for s in range(3):
                        if new[i, s] == k and row[s] and v in row[s]:
                            vecs.append(row[s][v][0])
                            weights.append(row[s][v][1])
                if vecs:
                    centroids[v][k] = np.average(np.stack(vecs), axis=0, weights=weights)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels, used


# --------------------------------------------------------------------------- naming stage
# Registration 2.


def label_centroids(descriptors, labels):
    """Crop-weighted per-camera centroid of each identity under ``labels``."""
    cameras = sorted({v for row in descriptors for d in row if d for v in d})
    centroids = {v: [None, None, None] for v in cameras}
    for v in cameras:
        for k in range(3):
            vecs, weights = [], []
            for row, names in zip(descriptors, labels, strict=True):
                for s in range(3):
                    if names[s] == k and row[s] and v in row[s]:
                        vecs.append(row[s][v][0])
                        weights.append(row[s][v][1])
            if vecs:
                centroids[v][k] = np.average(np.stack(vecs), axis=0, weights=weights)
    return centroids


def _centroid_cost(desc, centroids, k):
    terms = [
        np.linalg.norm(vec - centroids[v][k])
        for v, (vec, _) in desc.items()
        if v in centroids and centroids[v][k] is not None
    ]
    return float(np.mean(terms)) if terms else np.nan


def assign_partial(row, centroids):
    """Names for an interval with one or two described fragments; None otherwise.

    Two described fragments take the minimum-cost one-to-one identities and the
    third fragment the remaining identity by elimination. One described
    fragment takes its nearest centroid; the other two stay -1.
    """
    described = [s for s in range(3) if row[s]]
    if len(described) not in (1, 2):
        return None
    cost = np.array([[_centroid_cost(row[s], centroids, k) for k in range(3)] for s in described])
    if not np.isfinite(cost).all():
        return None
    names = np.full(3, -1, np.int8)
    if len(described) == 1:
        names[described[0]] = int(np.argmin(cost[0]))
        return names
    best = min(
        ((a, b) for a in range(3) for b in range(3) if a != b),
        key=lambda ab: cost[0, ab[0]] + cost[1, ab[1]],
    )
    names[described[0]], names[described[1]] = best
    missing = ({0, 1, 2} - set(described)).pop()
    names[missing] = ({0, 1, 2} - set(best)).pop()
    return names


def frame_names(n_frames, lower, names, carry_warnings=True):
    """Per-frame slot names from interval names; -1 is unknown.

    ``lower`` is a sequence of (segment_id, start_frame, stop_frame_exclusive)
    in time order; frames between consecutive intervals are warning interiors.
    With ``carry_warnings`` a warning slot takes name ``k`` only when both
    flanking intervals name that slot ``k``.
    """
    out = np.full((n_frames, 3), -1, np.int8)
    lower = list(lower)
    for seg, lo, hi in lower:
        out[lo:hi] = names[seg]
    if carry_warnings:
        for (a, _, a_hi), (b, b_lo, _) in pairwise(lower):
            if b_lo > a_hi:
                agree = (names[a] == names[b]) & (names[a] >= 0)
                out[a_hi:b_lo] = np.where(agree, names[a], -1)
    return out


# --------------------------------------------------------------------------- gaps
# Registration 3.

TAU_GRID = np.logspace(-4, 1, 501)
GAP_ALPHA = 0.25
GAP_POSTERIOR = 0.99


def anchors_and_gaps(lower, names):
    """Anchors (intervals with three names) and the gaps between consecutive ones.

    ``lower`` is (segment_id, start_frame, stop_frame_exclusive) in time order.
    Returns (anchors, gaps); a gap is (pre_anchor, post_anchor, first, stop).
    """
    anchors = [(seg, lo, hi) for seg, lo, hi in lower if (names[seg] >= 0).all()]
    gaps = [(a, b, a_hi, b_lo) for (a, _, a_hi), (b, b_lo, _) in pairwise(anchors) if b_lo > a_hi]
    return anchors, gaps


def frame_names_bridged(n_frames, lower, names):
    """S: anchor names on anchors; gap slots named where both anchors agree."""
    out = np.full((n_frames, 3), -1, np.int8)
    anchors, gaps = anchors_and_gaps(lower, names)
    for seg, lo, hi in anchors:
        out[lo:hi] = names[seg]
    for a, b, first, stop in gaps:
        out[first:stop] = np.where(names[a] == names[b], names[a], -1)
    return out


def fit_temperature(dist2: np.ndarray, labels: np.ndarray, grid=TAU_GRID) -> float:
    """Maximum-likelihood tau of softmax(-dist2 / tau) for the given identities."""
    best, best_ll = None, -np.inf
    for tau in grid:
        logits = -dist2 / tau
        logits -= logits.max(axis=1, keepdims=True)
        log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        ll = log_p[np.arange(labels.size), labels].mean()
        if ll > best_ll:
            best, best_ll = float(tau), ll
    return best


def log_identity_posterior(dist2: np.ndarray, tau: float) -> np.ndarray:
    logits = -dist2 / tau
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def perm_index(names) -> int:
    hit = np.flatnonzero((np.asarray(names)[None] == PERMS).all(axis=1))
    if hit.size != 1:
        raise ValueError(f"{names} is not a permutation of three identities")
    return int(hit[0])


def gap_posterior(start_names, end_names, dt, slot_loglik, rate, alpha=GAP_ALPHA):
    """Posterior over the six permutations at each interior node of one gap.

    ``dt`` has one entry per transition (``len(slot_loglik) + 1``).
    ``slot_loglik[j]`` is (3 slots, 3 identities) summed log-likelihood, zero
    where a slot has no usable crop. The first and last nodes are fixed to the
    anchor names. Returns (nodes, 6) posteriors.
    """
    m = len(slot_loglik)
    if len(dt) != m + 1:
        raise ValueError("dt must have one entry per transition")
    if not rate > 0:
        raise ValueError("switch rate must be positive")
    emit = np.array([alpha * ll[np.arange(3)[None], PERMS].sum(axis=1) for ll in slot_loglik])
    emit = emit.reshape(m, 6)

    def log_transition(step):
        stay = np.exp(-rate * step)
        t = np.full((6, 6), np.log(max((1 - stay) / 5, 1e-300)))
        np.fill_diagonal(t, np.log(max(stay, 1e-300)))
        return t

    def logsumexp(x, axis):
        top = x.max(axis=axis, keepdims=True)
        return (top + np.log(np.exp(x - top).sum(axis=axis, keepdims=True))).squeeze(axis)

    start = np.full(6, -np.inf)
    start[perm_index(start_names)] = 0.0
    end = np.full(6, -np.inf)
    end[perm_index(end_names)] = 0.0
    forward = np.empty((m, 6))
    previous = start
    for j in range(m):
        forward[j] = logsumexp(previous[:, None] + log_transition(dt[j]), axis=0) + emit[j]
        previous = forward[j]
    backward = np.empty((m, 6))
    following = end
    for j in range(m - 1, -1, -1):
        backward[j] = logsumexp(log_transition(dt[j + 1]) + following[None, :], axis=1)
        following = backward[j] + emit[j]
    log_post = forward + backward
    log_post -= logsumexp(log_post, axis=1)[:, None]
    return np.exp(log_post)


def slot_marginals(posterior: np.ndarray) -> np.ndarray:
    """(nodes, 3 slots, 3 identities) marginals of permutation posteriors."""
    out = np.zeros((posterior.shape[0], 3, 3))
    for j, perm in enumerate(PERMS):
        out[:, np.arange(3), perm] += posterior[:, j : j + 1]
    return out


def node_names(marginals: np.ndarray, threshold=GAP_POSTERIOR) -> np.ndarray:
    best = marginals.argmax(axis=2)
    return np.where(marginals.max(axis=2) >= threshold, best, -1).astype(np.int8)


def fill_between_nodes(out, node_frames, names_at_nodes):
    """Frames strictly between consecutive nodes keep a slot name both nodes share."""
    for (f0, n0), (f1, n1) in pairwise(zip(node_frames, names_at_nodes, strict=True)):
        out[f0] = n0
        if f1 > f0 + 1:
            out[f0 + 1 : f1] = np.where((n0 == n1) & (n0 >= 0), n0, -1)
    out[node_frames[-1]] = names_at_nodes[-1]


# --------------------------------------------------------------------------- clean crops, stopping
# Registration 4. Label-free: fragment structure, detection geometry and the
# embedding's own cluster quality.

CLEAN_MIN_ROWS = 3
VALIDATION_CROPS = 3000
EVAL_EVERY = 100
STOP_PATIENCE = 30
STOP_TARGET = 0.91
STOP_TARGET_PATIENCE = 2
MAX_STEPS = 6000


def leftward(heading) -> np.ndarray:
    """True where the heading points left in the image (|h| > 90 degrees)."""
    return np.abs(np.asarray(heading, float)) > np.pi / 2


def dorsal_up(crops: np.ndarray, heading) -> np.ndarray:
    """D: side-view crops flipped vertically where the fish swims left.

    The builder rotates the heading onto +x, which turns a leftward side-view
    fish belly-up; the vertical flip restores dorsal-up with the head still on
    the right. The flip is a function of the heading only, so the one builder
    still serves training and inference. Top-view crops are never flipped.
    """
    out = crops.copy()
    flip = leftward(heading)
    out[flip] = crops[flip][:, ::-1]
    return out


def clean_mask(occluded, overlap, rows_in_frame) -> np.ndarray:
    """X: no own-body occlusion, no other keypoint within the overlap radius, >= 3 rows."""
    return (
        (np.asarray(occluded) == 0)
        & ~np.asarray(overlap, bool)
        & (np.asarray(rows_in_frame) >= CLEAN_MIN_ROWS)
    )


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d2, 0.0)


def kmeans(x, k=3, restarts=10, iters=100, seed=0):
    """Lloyd's k-means from random data points; (labels, inertia) of the best restart."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    best_labels, best_inertia = None, np.inf
    for _ in range(restarts):
        centres = x[rng.choice(len(x), k, replace=False)].copy()
        labels = None
        for _ in range(iters):
            d2 = _squared_distances(x, centres)
            new = d2.argmin(axis=1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                members = labels == c
                # An emptied cluster restarts at the point farthest from every centre.
                centres[c] = (
                    x[members].mean(axis=0) if members.any() else x[d2.min(axis=1).argmax()]
                )
        inertia = float(_squared_distances(x, centres)[np.arange(len(x)), labels].sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels.copy(), inertia
    return best_labels, best_inertia


def silhouette(x, labels) -> float:
    """Mean Euclidean silhouette; -1 with fewer than two clusters; singletons score 0."""
    x = np.asarray(x, np.float64)
    labels = np.asarray(labels)
    present = np.unique(labels)
    if present.size < 2:
        return -1.0
    d = np.sqrt(_squared_distances(x, x))
    size = np.array([(labels == c).sum() for c in present], float)
    sums = np.stack([d[:, labels == c].sum(axis=1) for c in present], axis=1)
    own = np.searchsorted(present, labels)
    rows = np.arange(len(x))
    a = sums[rows, own] / np.maximum(size[own] - 1, 1)
    other = sums / size[None, :]
    other[rows, own] = np.inf
    b = other.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[size[own] <= 1] = 0.0
    return float(s.mean())


def pca2(x):
    """Two-component PCA projection and the explained-variance ratio of each."""
    x = np.asarray(x, np.float64)
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = s**2 / max(float((s**2).sum()), 1e-12)
    return centred @ vt[:2].T, explained[:2]


class SilhouetteMonitor:
    """T: silhouette stopping on a fixed validation set, keeping the best parameters.

    ``validation`` indexes the crops embedded at every evaluation; k-means with
    three clusters and the silhouette of that partition are label-free.
    ``cross_pairs`` (m, 2) are same-fragment, same-frame crop pairs from two
    cameras whose median embedding distance is tracked. Stops after
    ``patience`` evaluations without a new best, or ``target_patience``
    consecutive evaluations at or above ``target``.
    """

    def __init__(
        self,
        validation,
        cross_pairs=None,
        *,
        patience=STOP_PATIENCE,
        target=STOP_TARGET,
        target_patience=STOP_TARGET_PATIENCE,
        seed=20261106,
    ):
        self.validation = np.asarray(validation)
        self.cross_pairs = None if cross_pairs is None else np.asarray(cross_pairs).reshape(-1, 2)
        self.patience, self.target, self.target_patience = patience, target, target_patience
        self.seed = seed
        self.evaluations: list[dict] = []
        self.snapshots: list[np.ndarray] = []
        self.clusters: list[np.ndarray] = []
        self.best = None  # (silhouette, step, state_dict)
        self.since_best = 0
        self.at_target = 0
        self.stopped = None

    def __call__(self, step, model, crops, stats) -> bool:
        import copy

        model.eval()
        z = embed(model, crops[self.validation], stats)
        cross = np.nan
        if self.cross_pairs is not None and len(self.cross_pairs):
            za = embed(model, crops[self.cross_pairs[:, 0]], stats)
            zb = embed(model, crops[self.cross_pairs[:, 1]], stats)
            cross = float(np.median(np.linalg.norm(za - zb, axis=1)))
        model.train()
        labels, inertia = kmeans(z, seed=self.seed)
        score = silhouette(z, labels)
        self.snapshots.append(z.astype(np.float32))
        self.clusters.append(labels.astype(np.int8))
        self.evaluations.append(
            {
                "step": int(step),
                "silhouette": score,
                "inertia": inertia,
                "cross_camera_median": cross,
            }
        )
        if self.best is None or score > self.best[0]:
            self.best = (score, int(step), copy.deepcopy(model.state_dict()))
            self.since_best = 0
        else:
            self.since_best += 1
        self.at_target = self.at_target + 1 if score >= self.target else 0
        if self.at_target >= self.target_patience:
            self.stopped = "target"
        elif self.since_best >= self.patience:
            self.stopped = "patience"
        return self.stopped is not None


class PrecomputedIdentifier:
    """Identifier Protocol adapter over per-frame slot identities computed above."""

    name = "contrastive_appearance_fragments"

    def __init__(
        self, slot_identity: np.ndarray, names=(Identity.FEMALE, Identity.MALE_1, Identity.MALE_2)
    ):
        self.slot_identity = slot_identity
        self.names = names

    def identify(self, tracks, frames):
        out = {}
        for frame in frames:
            row = self.slot_identity[frame]
            out[int(frame)] = {
                s: (self.names[k] if k >= 0 else Identity.UNKNOWN) for s, k in enumerate(row)
            }
        return out
