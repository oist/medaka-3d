"""Score a fixed pose bank with the frozen FP64 final-fit objective.

Bank[0] must be the caller's exact original FP64 initial proposal. Bank[1:16]
must be the fifteen FP32 generator poses promoted to FP64 by the caller. This
pure helper validates the numerical schema; it cannot prove that provenance.
It never optimizes, clamps, relabels, adds a pose, or changes the outer B proposals.

All sixteen rows use the same FP64 diagnostic renderer path as final evaluation.
Each outer proposal selects its first strict minimum independently. The selected
pose is evaluated once more on that same path; a changed score/parts or an initial
score regression is rejected. This does not reproduce the old FP64 search path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from directional_device_fitter import center_correction
from graph_fit import FitInputs, GraphFit, objective
from renderer import RefractiveFishRenderer

SCHEMA = "fp64_pose_bank_rescore_v1"
BANK_SIZE = 16


def _tensor(value, shape, device, name, *, dtype=torch.float64):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != dtype
        or value.device != device
    ):
        raise ValueError(f"{name} must retain required shape, device and dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate(pose_bank, config, inputs, renderer, axes, parallel_sigma, enabled, directional):
    if (
        not isinstance(pose_bank, torch.Tensor)
        or pose_bank.ndim != 5
        or pose_bank.shape[0] != BANK_SIZE
        or pose_bank.shape[1] < 1
        or pose_bank.shape[2:] != (3, 3, 3)
        or pose_bank.dtype != torch.float64
        or pose_bank.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("pose_bank must be FP64 (16,B,3,3,3), B>0 on CPU or CUDA")
    if type(directional) is not bool or config.opt_steps != 14:
        raise ValueError("explicit directional bool and original14-step config required")
    if not bool(torch.isfinite(pose_bank).all()):
        raise ValueError("all sixteen pose-bank rows must be finite")
    if not isinstance(inputs, FitInputs) or type(renderer) is not RefractiveFishRenderer:
        raise ValueError("original FitInputs and original dynamic renderer required")
    device, b, n = pose_bank.device, pose_bank.shape[1], renderer.n_pixels
    if renderer.dtype != torch.float64 or renderer.device != device:
        raise ValueError("original renderer must be FP64 on the pose-bank device")
    # Call only the frozen read-only input validator, never GraphFit.__init__ or
    # any CUDA graph/capture API. Original constants/scales/width remain inputs.
    validation = SimpleNamespace(
        batch_size=b,
        n_pixels=n,
        config=config,
        device=device,
        dtype=torch.float64,
        _renderer_signature=(renderer.views, renderer.config, device, torch.float64, n),
    )
    GraphFit._validate_inputs(validation, inputs, renderer)
    _tensor(renderer.ray_origins, (3, n, 3), device, "image ray origins")
    _tensor(renderer.ray_directions, (3, n, 3), device, "image ray directions")
    _tensor(renderer.ray_valid, (3, n), device, "image ray validity", dtype=torch.bool)
    if bool((inputs.weight[~renderer.ray_valid] != 0).any()):
        raise ValueError("FitInputs.weight must already include original image ray validity")
    _tensor(axes, (b, 3, 3), device, "directional axes")
    _tensor(parallel_sigma, (b, 3), device, "directional parallel sigma")
    _tensor(enabled, (b, 3), device, "directional enabled", dtype=torch.bool)
    if not bool((parallel_sigma > 0).all()):
        raise ValueError("parallel sigma must be positive")


def _evaluate(
    pose, config, inputs, renderer, denominator, axes, parallel_sigma, enabled, directional
):
    scores, parts, rendering = objective(
        pose, config, inputs, denominator, renderer, diagnostics=True
    )
    if directional:
        difference, _, center = center_correction(
            pose[:, :, 1] - inputs.predicted[:, :, 1],
            inputs.motion_sigma,
            axes,
            parallel_sigma,
            enabled,
        )
        scores = scores + config.motion_weight * difference
        parts = torch.cat((parts[:, :2], center[:, None], parts[:, 3:]), dim=-1)
    for name, value in (
        ("score", scores),
        ("parts", parts),
        ("union", rendering.union_alpha),
        ("visible alpha", rendering.visible_alpha),
        ("ownership", rendering.ownership),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite FP64 bank " + name)
    return scores, parts, rendering


@torch.no_grad()
def rescore_pose_bank(
    pose_bank,
    *,
    config,
    inputs,
    renderer,
    axes,
    parallel_sigma,
    enabled,
    directional=True,
):
    """Return detached, owned same-device tensors; no input is modified.

    selected_bank_index: int64[B]; bank_scores: float64[16,B];
    bank_parts: float64[16,B,5], with corrected center when directional;
    initial_score/initial_parts: float64[B]/[B,5];
    fitted_pose_xyz/local_cost/final_loss_parts: float64[B,3,3,3]/[B]/[B,5];
    mask_supported/view_supported: bool[B,3]/[B,3,3] (proposal,view,body);
    positive/negative: float64[B,3,3];
    visible_alpha/ownership: float64[B,3,N,3]; union_alpha: float64[B,3,N];
    ray_valid: bool[3,N]. Metadata: schema, bank_size, objective_evaluations,
    directional and selection_policy. No gradients or fitted identity output.
    """
    _validate(pose_bank, config, inputs, renderer, axes, parallel_sigma, enabled, directional)
    denominator = 9 * float(inputs.node_weight.sum())
    score_rows, part_rows = [], []
    best_pose = pose_bank[0].clone()
    best_score = None
    selected = torch.zeros(pose_bank.shape[1], dtype=torch.int64, device=pose_bank.device)
    for index in range(BANK_SIZE):
        scores, parts, _ = _evaluate(
            pose_bank[index],
            config,
            inputs,
            renderer,
            denominator,
            axes,
            parallel_sigma,
            enabled,
            directional,
        )
        score_rows.append(scores.clone())
        part_rows.append(parts.clone())
        if index == 0:
            best_score = scores.clone()
        else:
            improved = scores < best_score
            best_pose = torch.where(improved[:, None, None, None], pose_bank[index], best_pose)
            best_score = torch.where(improved, scores, best_score)
            selected = torch.where(improved, index, selected)
    bank_scores, bank_parts = torch.stack(score_rows), torch.stack(part_rows)
    final_score, final_parts, rendering = _evaluate(
        best_pose, config, inputs, renderer, denominator, axes, parallel_sigma, enabled, directional
    )
    proposal = torch.arange(pose_bank.shape[1], device=pose_bank.device)
    if not torch.equal(final_score, best_score) or not torch.equal(
        final_parts, bank_parts[selected, proposal]
    ):
        raise FloatingPointError("FP64 selected-pose reevaluation changed score or parts")
    if not bool((final_score <= bank_scores[0]).all()):
        raise FloatingPointError("FP64 selected pose regressed from original initial score")

    # Frozen Full/Fast final mask-support operations, in their original order.
    weight, target = inputs.weight, inputs.target
    contribution = rendering.visible_alpha
    distinct = rendering.ownership >= 0.65
    positive = (contribution * distinct * weight[None, :, :, None] * target[None, :, :, None]).sum(
        2
    )
    negative = (contribution * weight[None, :, :, None] * (1 - target)[None, :, :, None]).sum(2)
    view_support = (positive >= 0.025) & (positive >= 2 * negative)
    support = view_support.sum(1) >= 2
    if not bool(torch.isfinite(positive).all() & torch.isfinite(negative).all()):
        raise FloatingPointError("nonfinite FP64 final support")
    tensors = dict(
        selected_bank_index=selected,
        bank_scores=bank_scores,
        bank_parts=bank_parts,
        initial_score=bank_scores[0],
        initial_parts=bank_parts[0],
        fitted_pose_xyz=best_pose,
        local_cost=final_score,
        final_loss_parts=final_parts,
        mask_supported=support,
        view_supported=view_support,
        positive=positive,
        negative=negative,
        visible_alpha=contribution,
        ownership=rendering.ownership,
        union_alpha=rendering.union_alpha,
        ray_valid=renderer.ray_valid,
    )
    return dict(
        schema=SCHEMA,
        bank_size=BANK_SIZE,
        objective_evaluations=BANK_SIZE + 1,
        directional=directional,
        selection_policy="strict score <; first bank index wins ties; one pose per proposal",
        **{name: value.detach().clone() for name, value in tensors.items()},
    )
