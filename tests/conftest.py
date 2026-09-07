"""Shared fixtures.

The synthetic rig here is a **pinhole** rig — every refractive index is 1.0, so
the interfaces are optically invisible. That is deliberate: refraction is pinned
in `test_calib.py`, and a rig with a closed-form forward projection is what lets
`test_reconstruct.py` place three known animals at known 3D positions and check
that the correspondence solver finds them. Testing the correspondence logic
through an iterative projection would confound the two.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from medaka3d.calib import load_rig

#: Camera x/y/z axes expressed in world coordinates, i.e. the rows of R.
_POSES = {
    "cam0": (
        [96.0, 96.0, 500.0],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 186.0],
    ),
    "cam1": (
        [96.0, -400.0, 93.0],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [0.0, 1.0, 0.0],
        [0.0, -4.0, 0.0],
    ),
    "cam2": (
        [500.0, 96.0, 93.0],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [-1.0, 0.0, 0.0],
        [196.0, 0.0, 0.0],
    ),
}


def _rotation_vector(rotation: np.ndarray) -> list[float]:
    """Inverse Rodrigues. The closed form is singular at pi and the real cam0 is
    exactly a pi-rotation, so this uses scipy rather than the textbook formula."""
    from scipy.spatial.transform import Rotation

    return list(Rotation.from_matrix(np.asarray(rotation)).as_rotvec())


@pytest.fixture
def rig(tmp_path):
    """A three-camera pinhole rig in the manifest's own view order."""
    views = ["top_cam", "side1_cam", "side2_cam"]
    files = []
    for name, view in zip(_POSES, views, strict=True):
        center, rows, normal, point = _POSES[name]
        bundle = {
            "name": name,
            "fx": 2000.0,
            "fy": 2000.0,
            "cx": 600.0,
            "cy": 600.0,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
            "k4": 0.0, "k5": 0.0, "k6": 0.0,
            "rotation_vector": _rotation_vector(np.asarray(rows)),
            "camera_center": center,
            "interface_normal": normal,
            "interface_point": point,
            "glass_thickness_mm": 4.0,
            "index_air": 1.0,
            "index_glass": 1.0,
            "index_water": 1.0,
            "layer_count": 1,
        }
        (tmp_path / f"{name}_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        files.append({"camera": name, "view": view, "path": f"{name}_bundle.json"})
    (tmp_path / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "backend": "refractive",
                "world_units": "mm",
                "camera_order": list(_POSES),
                "view_order": views,
                "model_files": files,
            }
        ),
        encoding="utf-8",
    )
    return load_rig(tmp_path)


def project(camera, xyz: np.ndarray) -> np.ndarray:
    """World mm -> pixels, valid only for the undistorted unit-index fixture rig."""
    xyz = np.asarray(xyz, dtype=float)
    cam = (xyz - camera.center) @ camera.rotation.T
    return np.stack(
        [
            camera.fx * cam[..., 0] / cam[..., 2] + camera.cx,
            camera.fy * cam[..., 1] / cam[..., 2] + camera.cy,
        ],
        axis=-1,
    )
