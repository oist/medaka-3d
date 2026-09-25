"""A latent quadratic tail bend, with the frozen ray/compositing model unchanged.

The three anatomical nodes remain mouth, pectoral and tail. A control offset is
not another observed node. Zero offset recovers the original two straight body
segments. Gaussian widths and axial scales use the old chord formula, optionally
from a fixed reference pose; they are never enlarged to match the curve's arc.
This module adds representation only: no fit, support decision or ID inference.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
from renderer import GaussianBodies, RefractiveFishRenderer


def _pose(value, *, dtype, device):
    pose = torch.as_tensor(value, dtype=dtype, device=device)
    if pose.shape == (3, 3, 3):
        pose = pose.unsqueeze(0)
    if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or not len(pose):
        raise ValueError("pose must have shape (3,3,3) or (B,3,3,3)")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError("pose must be finite")
    return pose


def _offset(value, pose):
    if value is None:
        return torch.zeros_like(pose[:, :, 1])
    offset = torch.as_tensor(value, dtype=pose.dtype, device=pose.device)
    if offset.shape == (3, 3):
        offset = offset.unsqueeze(0).expand(len(pose), 3, 3)
    if offset.shape != (len(pose), 3, 3) or not bool(torch.isfinite(offset).all()):
        raise ValueError("tail control offset must be finite (3,3) or (B,3,3)")
    return offset


def _minimum_speed(chord, offset):
    # B'(t) = a + b*t. This also detects a cusp between sampled Gaussians.
    a, b = chord + 2 * offset, -4 * offset
    parameter = (-(a * b).sum(-1) / b.square().sum(-1).clamp_min(1e-24)).clamp(0, 1)
    speed = torch.linalg.vector_norm(a + parameter[..., None] * b, dim=-1)
    return speed, parameter


@dataclass(frozen=True)
class TailCurveDiagnostics:
    chord_length_mm: torch.Tensor  # B,3
    arc_length_mm: torch.Tensor  # B,3; split 64-point Gauss-Legendre quadrature
    arc_quadrature_difference_mm: torch.Tensor  # |64-point - 32-point|, not a bound
    minimum_speed_mm: torch.Tensor  # min ||dB/dt|| on [0,1]
    minimum_speed_parameter: torch.Tensor
    degenerate: torch.Tensor  # vanishing derivative or collapsed endpoint chord


def tail_curve_diagnostics(pose_xyz, tail_control_offset_mm=None, *,
                           dtype=torch.float64, device=None) -> TailCurveDiagnostics:
    """Report arc/chord lengths without imposing an anatomy constraint.

    Integrals are split at the derivative's minimum to handle collinear cusps.
    The quadrature difference is a numerical diagnostic, not a rigorous bound.
    No arc/chord ratio is invented for a collapsed chord. Degenerate curves have
    finite lengths but cannot supply a regular tangent everywhere.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("diagnostics require float32 or float64")
    pose = _pose(pose_xyz, dtype=dtype, device=device)
    offset = _offset(tail_control_offset_mm, pose)
    chord = pose[:, :, 2] - pose[:, :, 1]
    length = torch.linalg.vector_norm(chord, dim=-1)
    speed, split = _minimum_speed(chord, offset)

    def integrate(count):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        nodes = torch.as_tensor((nodes + 1) / 2, dtype=dtype, device=pose.device)
        weights = torch.as_tensor(weights / 2, dtype=dtype, device=pose.device)
        correction = torch.zeros_like(length)
        for start, width in ((torch.zeros_like(split), split), (split, 1 - split)):
            t = start[..., None] + width[..., None] * nodes
            derivative = chord[..., None, :] + (2 - 4 * t)[..., None] * offset[..., None, :]
            # Baseline-plus-correction makes the zero-offset chord exact.
            values = torch.linalg.vector_norm(derivative, dim=-1) - length[..., None]
            correction = correction + width * (values * weights).sum(-1)
        return length + correction

    arc32, arc64 = integrate(32), integrate(64)
    if not bool(torch.isfinite(torch.stack((length, speed, arc32, arc64))).all()):
        raise ValueError("curve diagnostics overflowed")
    return TailCurveDiagnostics(length, arc64, (arc64 - arc32).abs(), speed, split,
                                (length <= 1e-12) | (speed <= 1e-12))


class CurvedFishRenderer(RefractiveFishRenderer):
    """Original head plus a quadratic pectoral-to-tail Bezier centerline.

    ``control = (pectoral + tail)/2 + tail_control_offset_mm``. Offsets have
    shape (3,3) or (B,3,3), and retain autograd connections. An optional
    ``sigma_reference_pose_xyz`` freezes both Gaussian scales to the original
    chord-based values of that pose (with the supplied width ratio).

    Ray response and visibility are the inherited implementation. Render calls
    bind parameters to a temporary shallow copy: cached rays are read-only and
    neither inputs nor this renderer retain mutable per-call curve state.
    """

    def gaussians(self, pose_xyz, width_ratio=None, *, tail_control_offset_mm=None,
                  sigma_reference_pose_xyz=None):
        bound = getattr(self, "_curve_arguments", None)
        if bound is not None:
            tail_control_offset_mm, sigma_reference_pose_xyz = bound
        pose = _pose(pose_xyz, dtype=self.dtype, device=self.device)
        offset = _offset(tail_control_offset_mm, pose)
        original = super().gaussians(pose, width_ratio)
        scales = original
        if sigma_reference_pose_xyz is not None:
            reference = _pose(sigma_reference_pose_xyz, dtype=self.dtype, device=self.device)
            if len(reference) == 1:
                reference = reference.expand(len(pose), 3, 3, 3)
            if reference.shape != pose.shape:
                raise ValueError("sigma reference batch must match the current pose")
            scales = super().gaussians(reference, width_ratio)
        t = self._fractions[self._head_count:]
        centers_tail = original.centers[:, :, self._head_count:] + (
            2 * (1 - t) * t
        )[None, None, :, None] * offset[:, :, None]
        chord = pose[:, :, 2] - pose[:, :, 1]
        derivative = chord[:, :, None] + (2 - 4 * t)[None, None, :, None] * offset[:, :, None]
        speed = torch.linalg.vector_norm(derivative, dim=-1)
        tail_tangent = derivative / speed.clamp_min(1e-12)[..., None]
        tail_tangent = torch.where(
            (speed > 1e-12)[..., None], tail_tangent,
            original.tangents[:, :, self._head_count:],
        )
        minimum_speed, _ = _minimum_speed(chord, offset)
        if not all(bool(torch.isfinite(value).all()) for value in (
            centers_tail, speed, minimum_speed, scales.axial_sigma_mm, scales.radial_sigma_mm,
        )):
            raise ValueError("curve geometry overflowed")
        return GaussianBodies(
            centers=torch.cat((original.centers[:, :, :self._head_count], centers_tail), dim=2),
            tangents=torch.cat((original.tangents[:, :, :self._head_count], tail_tangent), dim=2),
            axial_sigma_mm=scales.axial_sigma_mm,
            radial_sigma_mm=scales.radial_sigma_mm,
            segment_lengths_mm=original.segment_lengths_mm,
            nondegenerate=original.nondegenerate & (minimum_speed > 1e-12),
        )

    def render(self, pose_xyz, width_ratio=None, *, tail_control_offset_mm=None,
               sigma_reference_pose_xyz=None):
        bound = copy(self)
        bound._curve_arguments = (tail_control_offset_mm, sigma_reference_pose_xyz)
        return RefractiveFishRenderer.render(bound, pose_xyz, width_ratio)
