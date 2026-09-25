"""Common-image and support metrics for the fixed200 mask transport experiment.

Every arm is scored against every shared initial anchor for BOTH LK and CT
evidence. The scorer never selects an anchor, mask, view or pixel using a pose.
These spatial samples share D2 with fitting and are not independent ID truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import object_mask_transport as transport

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
import torch  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

ARMS = ("original", "lk", "ct_first")
EVIDENCE = ("lk", "ct_first")
VIEWS = transport.VIEWS
require = transport.require


def render_scores(union, ownership, targets, foreground, weights, valid):
    """M,V,P and M,V,P,body against E,A,V,P,body; retain all axes.

    Per-view values use the original weighted sum, and whole-image losses use
    the fixed three-view denominator. Zero mass is retained, not renormalized.
    """
    require(union.dtype == ownership.dtype == targets.dtype == np.float64,
            "float64 common render and targets required")
    require(union.ndim == 3 and union.shape[:2] == (3, 3), "three arms/views required")
    require(ownership.shape == (*union.shape, 3), "three ownership bodies required")
    require(targets.ndim == 5 and targets.shape[0] == 2 and 1 <= targets.shape[1] <= 6
            and targets.shape[2:] == (*union.shape[1:], 3), "all evidence/anchor axes required")
    require(foreground.shape == weights.shape == valid.shape == union.shape[1:]
            and valid.dtype == bool, "pixel evidence schema differs")
    require(all(np.isfinite(a).all() for a in (union, ownership, targets, foreground, weights)),
            "nonfinite evaluation input")
    require(((foreground == 0) | (foreground == 1)).all() and (weights >= 0).all()
            and ((targets >= 0) & (targets <= 1)).all()
            and ((targets > 0).sum(axis=-1) <= 1).all(), "invalid target assignment")
    weight = weights.astype(np.float64) * valid
    union_view = (((union - foreground) ** 2) * weight).sum(axis=-1)
    # E,M,A,V,P,B; no min/argmin or renormalization over the anchor axis.
    products = (targets[:, None] * (1 - ownership[None, :, None]) ** 2
                * (weight * foreground)[None, None, None, :, :, None])
    owner_body_view = products.sum(axis=-2)
    owner_view = owner_body_view.sum(axis=-1)
    mass = (targets * (weight * foreground)[None, None, :, :, None]).sum(axis=-2)
    return dict(union_per_view=union_view, union_loss=union_view.sum(axis=-1) / 3,
                owner_per_body_view=owner_body_view, owner_per_view=owner_view,
                owner_loss=owner_view.sum(axis=-1) / 3, target_mass=mass,
                positive_target_mass=mass > 0)


def score_fixed_images(poses, sample, packets, rig):
    require(poses.dtype == np.float64 and poses.shape == (3, 3, 3, 3)
            and np.isfinite(poses).all(), "three fixed arm poses required")
    require(len(packets) == 2 and all(type(p) is transport.TargetPacket for p in packets),
            "common LK and CT packets required")
    pixels = sample["unused_pixels"]
    signature = transport.signature(pixels)
    require(all(p.pixels_signature == signature for p in packets), "common unused pixels differ")
    require(np.array_equal(packets[0].anchor_ids, packets[1].anchor_ids)
            and packets[0].initial_native == packets[1].initial_native
            and packets[0].native == packets[1].native == int(sample["frame_idx"]),
            "common initial bank or native differs")
    require(packets[0].mask_signatures == packets[1].mask_signatures
            and packets[0].query_sha256 == packets[1].query_sha256
            and np.array_equal(packets[0].release_horizons, packets[1].release_horizons),
            "common masks/query bytes/release differ")
    for left, right in zip(packets[0].anchors, packets[1].anchors, strict=True):
        require(left.reference_sha256 == right.reference_sha256
                and all(transport.signature(a) == transport.signature(b)
                        for a, b in zip(left.links, right.links, strict=True)),
                "evidence arms use different initial anchor definitions")
    require(not sample["unused_fit_overlap"].any(), "unused pixels include fit coordinates")
    renderer = RefractiveFishRenderer(
        rig, dict(zip(VIEWS, pixels, strict=True)), RendererConfig(width_ratio=0.06),
        device="cpu", dtype=torch.float64,
    )
    with torch.no_grad():
        result = renderer.render(torch.as_tensor(poses, dtype=torch.float64))
    valid = result.ray_valid.cpu().numpy()
    require(np.array_equal(valid, sample["unused_ray_valid"]), "fixed ray validity changed")
    union = result.union_alpha.cpu().numpy()
    ownership = result.ownership.cpu().numpy()
    scores = render_scores(union, ownership, np.stack([p.target for p in packets]),
                           sample["unused_target"], sample["unused_weights"], valid)
    scores.update(rendered_union=union, rendered_ownership=ownership, ray_valid=valid.copy(),
                  anchor_ids=packets[0].anchor_ids.copy(),
                  frame_idx=np.asarray(packets[0].native, np.int64))
    return scores


def missing_runs(observed, frames):
    """Every maximal false run, including initial and terminal intervals."""
    require(observed.dtype == bool and observed.ndim == 1 and len(observed) == len(frames),
            "one temporal support flag per native required")
    require(len(frames) >= 1 and np.array_equal(np.diff(frames), np.ones(len(frames)-1, int)),
            "contiguous frames required")
    padded = np.r_[False, ~observed, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [dict(start=int(frames[lo]), stop=int(frames[hi-1] + 1), frames=int(hi-lo))
            for lo, hi in zip(starts, stops, strict=True)]


def summarize_support(candidates):
    require(set(candidates) == set(ARMS), "all three arms required")
    frames = candidates["original"]["frame_idx"]
    require(frames.shape == (200,) and frames.dtype == np.int64
            and np.array_equal(frames, np.arange(333519, 333719, dtype=np.int64)),
            "fixed200 native frames required")
    baseline = candidates["original"]["observed"]
    result = {}
    for arm in ARMS:
        candidate = candidates[arm]
        require(np.array_equal(candidate["frame_idx"], frames), "arm native mismatch")
        observed, pose = candidate["observed"], candidate["pose_xyz"]
        require(observed.dtype == bool and observed.shape == (200, 3, 3)
                and pose.dtype == np.float64 and pose.shape == (200, 3, 3, 3)
                and np.isfinite(pose).all(), "candidate node schema differs")
        gained, lost = observed & ~baseline, ~observed & baseline
        gaps = [[missing_runs(observed[:, b, n], frames) for n in range(3)] for b in range(3)]
        joint = observed[1:] & observed[:-1]
        step = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
        result[arm] = dict(
            all_node_time_denominator=1800, per_node_denominator=600,
            direct_count_by_body_node=observed.sum(axis=0).tolist(),
            direct_count_by_node=observed.sum(axis=(0, 1)).tolist(),
            missing_count_by_node=(~observed).sum(axis=(0, 1)).tolist(),
            direct_gain_by_body_node=gained.sum(axis=0).tolist(),
            direct_loss_by_body_node=lost.sum(axis=0).tolist(),
            gain_rows=[dict(native=int(frames[t]), body=int(b), node=int(n))
                       for t, b, n in np.argwhere(gained)],
            loss_rows=[dict(native=int(frames[t]), body=int(b), node=int(n))
                       for t, b, n in np.argwhere(lost)],
            missing_runs_by_body_node=gaps,
            longest_missing_by_body_node=[[max((r["frames"] for r in runs), default=0)
                                           for runs in body] for body in gaps],
            adjacent_direct_pairs=int(joint.sum()),
            adjacent_direct_jump_over10mm=int(np.count_nonzero(joint & (step > 10))),
            adjacent_latent_jump_over10mm=int(np.count_nonzero(~joint & (step > 10))),
            support_source="candidate.observed; caller must verify current SLEAP geometry",
            latent_jump_includes_pairs_with_either_endpoint_unsupported=True,
            independent_ID_truth=False,
        )
    return result
