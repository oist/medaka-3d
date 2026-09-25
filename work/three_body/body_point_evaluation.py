"""Seven-arm common-image metrics with an independent NumPy half-ray gap.

Every arm is scored against every shared initial anchor for BOTH LK and CT
evidence. The scorer never selects an anchor, mask, view or pixel using a pose.
These spatial samples share D2 with fitting and are not independent ID truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import body_point_transport as point_transport
import numpy as np
import object_mask_transport as transport

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import torch  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

ARMS = ("original", "lk", "ct_first", "ct_gap", "static_base", "static_gap", "ct_body_points")
EVIDENCE = ("lk", "ct_first")
VIEWS = transport.VIEWS
require = transport.require


def render_scores(union, ownership, targets, foreground, weights, valid, *, rendered_q):
    """M,V,P and M,V,P,body against E,A,V,P,body; retain all axes.

    Per-view values use the original weighted sum, and whole-image losses use
    the fixed three-view denominator. Zero mass is retained, not renormalized.
    combined_* means owner + gap, WITHOUT lambda or union/base objective.
    The NumPy gap expression is independent of the training loss implementation.
    """
    require(
        union.dtype == ownership.dtype == targets.dtype == np.float64,
        "float64 common render and targets required",
    )
    require(union.ndim == 3 and union.shape[:2] == (7, 3), "seven arms and three views required")
    require(ownership.shape == (*union.shape, 3), "three ownership bodies required")
    require(
        targets.ndim == 5
        and targets.shape[0] == 2
        and 1 <= targets.shape[1] <= 6
        and targets.shape[2:] == (*union.shape[1:], 3),
        "all evidence/anchor axes required",
    )
    require(
        foreground.shape == weights.shape == valid.shape == union.shape[1:] and valid.dtype == bool,
        "pixel evidence schema differs",
    )
    require(
        all(np.isfinite(a).all() for a in (union, ownership, targets, foreground, weights)),
        "nonfinite evaluation input",
    )
    require(
        ((foreground == 0) | (foreground == 1)).all()
        and (weights >= 0).all()
        and ((targets >= 0) & (targets <= 1)).all()
        and ((targets > 0).sum(axis=-1) <= 1).all(),
        "invalid target assignment",
    )
    weight = weights.astype(np.float64) * valid
    union_view = (((union - foreground) ** 2) * weight).sum(axis=-1)
    # E,M,A,V,P,B; no min/argmin or renormalization over the anchor axis.
    products = (
        targets[:, None]
        * (1 - ownership[None, :, None]) ** 2
        * (weight * foreground)[None, None, None, :, :, None]
    )
    owner_body_view = products.sum(axis=-2)
    owner_view = owner_body_view.sum(axis=-1)
    mass = (targets * (weight * foreground)[None, None, :, :, None]).sum(axis=-2)
    require(
        isinstance(rendered_q, np.ndarray)
        and rendered_q.dtype == np.float64
        and rendered_q.shape == ownership.shape,
        "float64 seven-arm ray distance required",
    )
    q_valid = np.broadcast_to(valid[None, :, :, None], rendered_q.shape)
    require(
        np.isfinite(rendered_q[q_valid]).all() and (rendered_q[q_valid] >= 0).all(),
        "valid ray distance must be finite and nonnegative",
    )
    # Exclude the renderer's invalid-ray inf BEFORE sqrt/multiplication.
    neutral_q = np.where(q_valid, rendered_q, 0.0)
    gap = np.sqrt(np.maximum(neutral_q, 1.0)) - 1.0
    gap_products = (
        targets[:, None] * gap[None, :, None] * (weight * foreground)[None, None, None, :, :, None]
    )
    gap_body_view = gap_products.sum(axis=-2)
    gap_view = gap_body_view.sum(axis=-1)
    owner_loss = owner_view.sum(axis=-1) / 3
    gap_loss = gap_view.sum(axis=-1) / 3
    return dict(
        union_per_view=union_view,
        union_loss=union_view.sum(axis=-1) / 3,
        owner_per_body_view=owner_body_view,
        owner_per_view=owner_view,
        owner_loss=owner_loss,
        target_mass=mass,
        positive_target_mass=mass > 0,
        gap_per_body_view=gap_body_view,
        gap_per_view=gap_view,
        gap_loss=gap_loss,
        combined_per_body_view=owner_body_view + gap_body_view,
        combined_per_view=owner_view + gap_view,
        combined_loss=owner_loss + gap_loss,
    )


def score_fixed_images(poses, sample, packets, rig):
    require(
        poses.dtype == np.float64 and poses.shape == (7, 3, 3, 3) and np.isfinite(poses).all(),
        "seven fixed arm poses required",
    )
    require(
        len(packets) == 2 and all(type(p) is transport.TargetPacket for p in packets),
        "common LK and CT packets required",
    )
    pixels = sample["unused_pixels"]
    signature = transport.signature(pixels)
    require(all(p.pixels_signature == signature for p in packets), "common unused pixels differ")
    require(
        np.array_equal(packets[0].anchor_ids, packets[1].anchor_ids)
        and packets[0].initial_native == packets[1].initial_native
        and packets[0].native == packets[1].native == int(sample["frame_idx"]),
        "common initial bank or native differs",
    )
    require(
        packets[0].mask_signatures == packets[1].mask_signatures
        and packets[0].query_sha256 == packets[1].query_sha256
        and np.array_equal(packets[0].release_horizons, packets[1].release_horizons),
        "common masks/query bytes/release differ",
    )
    for left, right in zip(packets[0].anchors, packets[1].anchors, strict=True):
        require(
            left.reference_sha256 == right.reference_sha256
            and all(
                transport.signature(a) == transport.signature(b)
                for a, b in zip(left.links, right.links, strict=True)
            ),
            "evidence arms use different initial anchor definitions",
        )
    require(not sample["unused_fit_overlap"].any(), "unused pixels include fit coordinates")
    renderer = RefractiveFishRenderer(
        rig,
        dict(zip(VIEWS, pixels, strict=True)),
        RendererConfig(width_ratio=0.06),
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        result = renderer.render(torch.as_tensor(poses, dtype=torch.float64))
    valid = result.ray_valid.cpu().numpy()
    require(np.array_equal(valid, sample["unused_ray_valid"]), "fixed ray validity changed")
    union = result.union_alpha.cpu().numpy()
    ownership = result.ownership.cpu().numpy()
    rendered_q = result.per_fish_mahalanobis_sq.cpu().numpy()
    scores = render_scores(
        union,
        ownership,
        np.stack([p.target for p in packets]),
        sample["unused_target"],
        sample["unused_weights"],
        valid,
        rendered_q=rendered_q,
    )
    scores.update(
        rendered_union=union,
        rendered_ownership=ownership,
        rendered_q=rendered_q,
        ray_valid=valid.copy(),
        anchor_ids=packets[0].anchor_ids.copy(),
        frame_idx=np.asarray(packets[0].native, np.int64),
    )
    return scores


def missing_runs(observed, frames):
    """Every maximal false run, including initial and terminal intervals."""
    require(
        observed.dtype == bool and observed.ndim == 1 and len(observed) == len(frames),
        "one temporal support flag per native required",
    )
    require(
        len(frames) >= 1 and np.array_equal(np.diff(frames), np.ones(len(frames) - 1, int)),
        "contiguous frames required",
    )
    padded = np.r_[False, ~observed, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [
        dict(start=int(frames[lo]), stop=int(frames[hi - 1] + 1), frames=int(hi - lo))
        for lo, hi in zip(starts, stops, strict=True)
    ]


def summarize_support(candidates):
    require(set(candidates) == set(ARMS), "all seven arms required")
    frames = candidates["original"]["frame_idx"]
    require(
        frames.shape == (200,)
        and frames.dtype == np.int64
        and np.array_equal(frames, np.arange(333519, 333719, dtype=np.int64)),
        "fixed200 native frames required",
    )
    baseline = candidates["original"]["observed"]
    result = {}
    for arm in ARMS:
        candidate = candidates[arm]
        require(np.array_equal(candidate["frame_idx"], frames), "arm native mismatch")
        observed, pose = candidate["observed"], candidate["pose_xyz"]
        require(
            observed.dtype == bool
            and observed.shape == (200, 3, 3)
            and pose.dtype == np.float64
            and pose.shape == (200, 3, 3, 3)
            and np.isfinite(pose).all(),
            "candidate node schema differs",
        )
        gained, lost = observed & ~baseline, ~observed & baseline
        gaps = [[missing_runs(observed[:, b, n], frames) for n in range(3)] for b in range(3)]
        joint = observed[1:] & observed[:-1]
        step = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
        result[arm] = dict(
            all_node_time_denominator=1800,
            per_node_denominator=600,
            direct_count_by_body_node=observed.sum(axis=0).tolist(),
            direct_count_by_node=observed.sum(axis=(0, 1)).tolist(),
            missing_count_by_node=(~observed).sum(axis=(0, 1)).tolist(),
            direct_gain_by_body_node=gained.sum(axis=0).tolist(),
            direct_loss_by_body_node=lost.sum(axis=0).tolist(),
            gain_rows=[
                dict(native=int(frames[t]), body=int(b), node=int(n))
                for t, b, n in np.argwhere(gained)
            ],
            loss_rows=[
                dict(native=int(frames[t]), body=int(b), node=int(n))
                for t, b, n in np.argwhere(lost)
            ],
            missing_runs_by_body_node=gaps,
            longest_missing_by_body_node=[
                [max((r["frames"] for r in runs), default=0) for runs in body] for body in gaps
            ],
            adjacent_direct_pairs=int(joint.sum()),
            adjacent_direct_jump_over10mm=int(np.count_nonzero(joint & (step > 10))),
            adjacent_latent_jump_over10mm=int(np.count_nonzero(~joint & (step > 10))),
            support_source="candidate.observed; caller must verify current SLEAP geometry",
            latent_jump_includes_pairs_with_either_endpoint_unsupported=True,
            independent_ID_truth=False,
        )
    return result


def point_render_scores(ownership, rendered_q, weights, valid):
    """Independent NumPy owner+gap on E,M,V,Q,body and E,A,V,Q,body.

    Weights already contain m/S. All evidence methods and all anchors remain
    separate. Per-view/body sums are unreduced; losses have fixed /3. This uses
    the fit observation domain and is an in-sample diagnostic, not ID truth.
    """
    require(
        isinstance(ownership, np.ndarray)
        and ownership.dtype == np.float64
        and ownership.ndim == 5
        and ownership.shape[:3] == (2, 7, 3)
        and ownership.shape[-1] == 3,
        "two evidence/seven arms/three views/body point ownership required",
    )
    e, _, v, q, bodies = ownership.shape
    require(
        isinstance(rendered_q, np.ndarray)
        and rendered_q.dtype == np.float64
        and rendered_q.shape == ownership.shape
        and isinstance(weights, np.ndarray)
        and weights.dtype == np.float64
        and weights.ndim == 5
        and weights.shape[0] == e
        and 1 <= weights.shape[1] <= 6
        and weights.shape[2:] == (v, q, bodies)
        and isinstance(valid, np.ndarray)
        and valid.dtype == bool
        and valid.shape == (e, v, q),
        "all point evidence/anchor/distance/validity axes required",
    )
    require(
        np.isfinite(weights).all()
        and (weights >= 0).all()
        and ((weights > 0).sum(axis=-1) <= 1).all(),
        "point mass must be finite nonnegative exclusive evidence",
    )
    good = np.broadcast_to(valid[:, None, :, :, None], ownership.shape)
    require(
        np.isfinite(ownership[good]).all()
        and np.isfinite(rendered_q[good]).all()
        and (rendered_q[good] >= 0).all(),
        "valid body point render must be finite and q nonnegative",
    )
    own = np.where(good, ownership, 0.0)
    distance = np.where(good, rendered_q, 0.0)
    mass = weights * valid[:, None, :, :, None]
    owner = (mass[:, None] * (1 - own[:, :, None]) ** 2).sum(axis=-2)
    gap = (mass[:, None] * (np.sqrt(np.maximum(distance[:, :, None], 1.0)) - 1)).sum(axis=-2)
    return dict(
        point_owner_per_body_view=owner,
        point_gap_per_body_view=gap,
        point_combined_per_body_view=owner + gap,
        point_owner_loss=owner.sum(axis=(-1, -2)) / 3,
        point_gap_loss=gap.sum(axis=(-1, -2)) / 3,
        point_combined_loss=owner.sum(axis=(-1, -2)) / 3 + gap.sum(axis=(-1, -2)) / 3,
        point_mass_by_view_body=mass.sum(axis=-2),
        positive_gap_point_count=((mass[:, None] > 0) & (distance[:, :, None] > 1)).sum(axis=-2),
    )


def score_body_points(poses, packets, rig):
    """Score ALL seven poses on BOTH saved point domains and ALL t0 anchors."""
    require(
        isinstance(poses, np.ndarray)
        and poses.dtype == np.float64
        and poses.shape == (7, 3, 3, 3)
        and np.isfinite(poses).all(),
        "seven finite body-point evaluation poses required",
    )
    require(
        len(packets) == 2 and all(type(p) is point_transport.BodyPointPacket for p in packets),
        "canonical LK and CT body-point packets required",
    )
    left, right = packets
    for name in ("pixels", "fit_foreground", "fit_weights", "fit_ray_valid"):
        require(
            transport.signature(getattr(left, name)) == transport.signature(getattr(right, name)),
            "point evidence methods must share original fit samples: " + name,
        )
    require(
        left.native == right.native
        and left.initial_native == right.initial_native
        and left.query_sha256 == right.query_sha256
        and left.pixels_signature == right.pixels_signature
        and left.whole_mask_packet.mask_signatures == right.whole_mask_packet.mask_signatures
        and np.array_equal(left.query_counts, right.query_counts)
        and np.array_equal(left.anchor_ids, right.anchor_ids)
        and np.array_equal(left.release_horizons, right.release_horizons),
        "point evidence methods must share initial bank/native/query/mask/release",
    )
    for a, b in zip(left.whole_mask_packet.anchors, right.whole_mask_packet.anchors, strict=True):
        require(
            a.reference_sha256 == b.reference_sha256
            and all(
                transport.signature(x) == transport.signature(y)
                for x, y in zip(a.links, b.links, strict=True)
            ),
            "different body-point initial anchor definitions",
        )
    owners, distances = [], []
    for packet in packets:
        if packet.point_pixels.shape[1] == 0:
            owners.append(np.empty((7, 3, 0, 3), np.float64))
            distances.append(np.empty((7, 3, 0, 3), np.float64))
            continue
        renderer = RefractiveFishRenderer(
            rig,
            {
                view: packet.point_pixels[v, : packet.query_counts[v]]
                for v, view in enumerate(VIEWS)
            },
            RendererConfig(width_ratio=0.06),
            device="cpu",
            dtype=torch.float64,
        )
        for name in ("ray_origins", "ray_directions", "ray_valid"):
            require(
                transport.signature(getattr(renderer, name).numpy())
                == transport.signature(getattr(packet, name)),
                "evaluation point ray packet mismatch: " + name,
            )
        with torch.no_grad():
            rendered = renderer.render(torch.as_tensor(poses, dtype=torch.float64))
        owners.append(rendered.ownership.numpy())
        distances.append(rendered.per_fish_mahalanobis_sq.numpy())
    owners, distances = np.stack(owners), np.stack(distances)
    valid = np.stack([p.ray_valid for p in packets])
    result = point_render_scores(
        owners, distances, np.stack([p.point_weight for p in packets]), valid
    )
    result.update(
        point_rendered_ownership=owners,
        point_rendered_q=distances,
        point_ray_valid=valid,
        point_anchor_ids=left.anchor_ids.copy(),
        point_frame_idx=np.asarray(left.native, np.int64),
        point_query_counts=left.query_counts.copy(),
    )
    return result
