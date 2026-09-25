"""Experimental exact-union scoring for all16 FP64 poses, visibility only at end.

Uses the frozen objective's diagnostics=False branch, whose silhouette uses the
same renderer Gaussian/ray-response and union arithmetic. Full visibility is
computed once for the selected pose. Actual saved-bank equivalence must be
qualified separately; this file does not replace the sealed original rescorer.
"""

from __future__ import annotations

import torch
from directional_device_fitter import center_correction
from fp64_pose_bank_rescore import BANK_SIZE, _evaluate, _validate
from graph_fit import objective

SCHEMA = "fp64_union_pose_bank_rescore_v1"


@torch.no_grad()
def rescore_union_pose_bank(
    pose_bank, *, config, inputs, renderer, axes, parallel_sigma, enabled, directional=True
):
    _validate(pose_bank, config, inputs, renderer, axes, parallel_sigma, enabled, directional)
    denominator = 9 * float(inputs.node_weight.sum())
    score_rows, part_rows = [], []
    best_pose, best_score = pose_bank[0].clone(), None
    selected = torch.zeros(pose_bank.shape[1], dtype=torch.int64, device=pose_bank.device)
    finite = torch.ones((), dtype=torch.bool, device=pose_bank.device)
    for index in range(BANK_SIZE):
        scores, parts, _ = objective(
            pose_bank[index], config, inputs, denominator, renderer, diagnostics=False
        )
        if directional:
            difference, _, center = center_correction(
                pose_bank[index, :, :, 1] - inputs.predicted[:, :, 1],
                inputs.motion_sigma,
                axes,
                parallel_sigma,
                enabled,
            )
            scores = scores + config.motion_weight * difference
            parts = torch.cat((parts[:, :2], center[:, None], parts[:, 3:]), dim=-1)
        finite = finite & torch.isfinite(scores).all() & torch.isfinite(parts).all()
        score_rows.append(scores.clone())
        part_rows.append(parts.clone())
        if index == 0:
            best_score = scores.clone()
        else:
            improved = scores < best_score
            best_pose = torch.where(improved[:, None, None, None], pose_bank[index], best_pose)
            best_score = torch.where(improved, scores, best_score)
            selected = torch.where(improved, index, selected)
    if not bool(finite):
        raise FloatingPointError("nonfinite FP64 union-bank evaluation")
    bank_scores, bank_parts = torch.stack(score_rows), torch.stack(part_rows)
    final_score, final_parts, rendering = _evaluate(
        best_pose, config, inputs, renderer, denominator, axes, parallel_sigma, enabled, directional
    )
    proposal = torch.arange(pose_bank.shape[1], device=pose_bank.device)
    if not torch.equal(final_score, best_score) or not torch.equal(
        final_parts, bank_parts[selected, proposal]
    ):
        raise FloatingPointError("selected full diagnostic differs from union-only score or parts")
    if not bool((final_score <= bank_scores[0]).all()):
        raise FloatingPointError("selected full diagnostic worsened original initial score")
    weight, target = inputs.weight, inputs.target
    contribution = rendering.visible_alpha
    distinct = rendering.ownership >= 0.65
    positive = (contribution * distinct * weight[None, :, :, None] * target[None, :, :, None]).sum(
        2
    )
    negative = (contribution * weight[None, :, :, None] * (1 - target)[None, :, :, None]).sum(2)
    view_support = (positive >= 0.025) & (positive >= 2 * negative)
    if not bool(torch.isfinite(positive).all() & torch.isfinite(negative).all()):
        raise FloatingPointError("nonfinite FP64 union-bank final support")
    tensors = dict(
        selected_bank_index=selected,
        bank_scores=bank_scores,
        bank_parts=bank_parts,
        initial_score=bank_scores[0],
        initial_parts=bank_parts[0],
        fitted_pose_xyz=best_pose,
        local_cost=final_score,
        final_loss_parts=final_parts,
        mask_supported=view_support.sum(1) >= 2,
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
        union_objective_evaluations=BANK_SIZE,
        full_diagnostic_evaluations=1,
        directional=directional,
        selection_policy="strict score <; first bank index wins ties; one pose per proposal",
        **{name: value.detach().clone() for name, value in tensors.items()},
    )
