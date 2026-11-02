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
import sys
import uuid
from contextlib import contextmanager, nullcontext
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from medaka3d.calib import load_rig


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Audit tests deliberately reject paths outside the repository. Pytest's
    # default OS temp directory would violate that contract in CI and clones.
    # Use a fresh, short path so no existing directory is recursively replaced
    # and long immutable calibration names still fit ordinary Windows limits.
    if config.option.basetemp is None:
        repository = Path(__file__).resolve().parents[1]
        temporary = repository / "work" / ("pytest_" + uuid.uuid4().hex[:8])
        if not temporary.resolve().is_relative_to(repository) or temporary.exists():
            raise pytest.UsageError("A new workspace-local test temporary directory is required.")
        config.option.basetemp = str(temporary)


def pytest_collection_modifyitems(items):
    """Run original-artifact audits only when their declared inputs are present.

    Absence is optional; malformed or mismatched artifacts still fail the
    original assertions. Synthetic contract tests do not use this marker.
    """
    for item in items:
        for marker in item.iter_markers("requires_artifacts"):
            if not marker.args:
                raise pytest.UsageError("requires_artifacts must declare its original input paths")
            missing = [str(path) for path in marker.args if not Path(path).exists()]
            if missing:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Optional archived run inputs are absent: " + missing[0]
                    )
                )


class LabImportScope:
    """Keep a script-style lab's flat imports out of other experiments' caches.

    Retain one module graph per lab so shared classes and monkeypatch targets do
    not change between its tests. Restore both aliases and search paths after
    collection and after each complete test/fixture protocol.
    """

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.names = {path.stem for path in self.directory.glob("*.py")}
        self.modules = {}

    @contextmanager
    def activate(self):
        previous_path = sys.path[:]
        missing = object()
        previous_modules = {name: sys.modules.get(name, missing) for name in self.names}
        try:
            sys.path.insert(0, str(self.directory))
            for name in self.names:
                sys.modules.pop(name, None)
            sys.modules.update(self.modules)
            yield
        finally:
            self.modules = {
                name: module
                for name in self.names
                if (module := sys.modules.get(name)) is not None
                and getattr(module, "__file__", None)
                and Path(module.__file__).resolve().parent == self.directory
            }
            for name, module in previous_modules.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            sys.path[:] = previous_path


_GENERATIVE_LAB = "20260908_generative_tracking"
_GENERATIVE_IMPORTS = LabImportScope(Path(__file__).resolve().parents[1] / "labs" / _GENERATIVE_LAB)


@cache
def _lab_scope(path: Path):
    # These frozen tests declare the lab path in their source. Cover runtime
    # benchmark imports too, without changing the archived experiment scripts.
    if _GENERATIVE_LAB in path.read_text(encoding="utf-8"):
        return _GENERATIVE_IMPORTS
    return None


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    scope = _lab_scope(collector.path) if isinstance(collector, pytest.Module) else None
    with scope.activate() if scope else nullcontext():
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item):
    scope = _lab_scope(item.path)
    with scope.activate() if scope else nullcontext():
        return (yield)


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
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "k5": 0.0,
            "k6": 0.0,
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
