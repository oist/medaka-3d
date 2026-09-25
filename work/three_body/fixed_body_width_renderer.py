"""Absolute per-body radial widths from the registered seed, never current length.

Only radial sigma changes parameterization. The frozen renderer still computes
current centers, tangents, axial sigma, half-ray responses, and visibility.
The reference must be the registered seed, not a current or retrospective pose.
Body axes remain the registered slots; no permutation-dependent width reassignment.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
from renderer import RefractiveFishRenderer  # noqa: E402


def array_signature(value):
    array = np.ascontiguousarray(value)
    return array.dtype.str, tuple(array.shape), hashlib.sha256(array.tobytes()).hexdigest()


class FixedWidthRenderer(RefractiveFishRenderer):
    """Frozen radial[3,K], with the original current-pose axial parameterization.

    Widths are evaluated once by the original Gaussian formula on the supplied
    seed in this renderer's dtype/device, then cloned as detached owned storage.
    Public array/tensor access returns copies, including returned GaussianBodies.
    validate_fixed_width is called outside capture at graph creation/configure;
    it detects changed private storage without introducing per-iterate CPU reads.
    """

    def __init__(
        self,
        rig,
        pixels_by_view,
        config=None,
        device="cpu",
        dtype=torch.float32,
        *,
        reference_pose_xyz,
    ):
        if (
            not isinstance(reference_pose_xyz, np.ndarray)
            or reference_pose_xyz.dtype != np.float64
            or reference_pose_xyz.shape != (3, 3, 3)
            or not np.isfinite(reference_pose_xyz).all()
            or not (np.linalg.norm(np.diff(reference_pose_xyz, axis=1), axis=-1) > 0).all()
        ):
            raise ValueError(
                "registered reference must be finite float64[3,3,3] with positive segments"
            )
        reference = np.ascontiguousarray(reference_pose_xyz).copy()
        self._reference_signature = array_signature(reference)
        self._reference_bytes = reference.tobytes()
        super().__init__(rig, pixels_by_view, config, device, dtype)
        with torch.no_grad():
            seed_bodies = RefractiveFishRenderer.gaussians(self, reference)
            self._fixed_radial_sigma_mm = seed_bodies.radial_sigma_mm[0].detach().clone()
        self._radial_signature = array_signature(self._fixed_radial_sigma_mm.cpu().numpy())
        self._fixed_config = tuple(sorted(asdict(self.config).items()))
        self._fixed_width_signature = (
            "registered_absolute_radial_sigma_v1",
            self._reference_signature,
            self._fixed_config,
            self._radial_signature,
        )
        self.validate_fixed_width()

    @property
    def reference_signature(self):
        return self._reference_signature

    @property
    def fixed_width_signature(self):
        return self._fixed_width_signature

    @property
    def reference_pose_xyz(self):
        return np.frombuffer(self._reference_bytes, dtype=np.float64).reshape(3, 3, 3).copy()

    @property
    def radial_sigma_mm(self):
        return self._fixed_radial_sigma_mm.detach().clone()

    def validate_fixed_width(self):
        radial = self._fixed_radial_sigma_mm
        if (
            not isinstance(radial, torch.Tensor)
            or radial.shape != (3, self.config.gaussians_per_fish)
            or radial.dtype != self.dtype
            or radial.device != self.device
            or radial.requires_grad
            or radial.grad_fn is not None
            or not bool(torch.isfinite(radial).all())
            or not bool((radial >= self.config.minimum_sigma_mm).all())
            or array_signature(radial.detach().cpu().numpy()) != self._radial_signature
            or tuple(sorted(asdict(self.config).items())) != self._fixed_config
            or array_signature(self.reference_pose_xyz) != self._reference_signature
        ):
            raise ValueError("registered fixed-width storage or reference signature changed")
        return self._fixed_width_signature

    def gaussians(self, pose_xyz, width_ratio=None):
        if width_ratio is not None:
            raise ValueError("fixed absolute width forbids a width_ratio override")
        bodies = RefractiveFishRenderer.gaussians(self, pose_xyz)
        radial = (
            self._fixed_radial_sigma_mm.unsqueeze(0).expand(bodies.centers.shape[0], -1, -1).clone()
        )
        return replace(bodies, radial_sigma_mm=radial)
