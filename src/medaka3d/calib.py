"""Refractive three-camera geometry: a pixel becomes a ray in water; rays become a point.

The tank is 192 x 192 mm inside with 186 mm of water, imaged through 4 mm glass by
three cameras. Light bends twice on the way out of the side cameras' glass and once
at the top camera's free surface, so a pinhole model is wrong by millimetres inside
the volume — which is the same order as the distances this project has to resolve
between two fish. The promoted calibration models the refraction explicitly and is
the only backend used here.

**Only back-projection is implemented, deliberately.** Pixel -> ray in water is
closed form: unproject, march to the interface, apply Snell twice. The inverse,
world -> pixel, needs an iterative solve for the refraction point and is not
needed by anything in this module: the cost that ranks a cross-view correspondence
is the point-to-ray residual, in mm, which lives entirely on the ray side. That is
also the unit the calibration itself was validated in
(``heldout_wet10_point_to_ray``, rms 0.237 mm), so a residual computed here is
directly comparable to the number the calibration claims.

``Instance3D.residual`` is documented as being "in the calibration backend's own
unit". For this backend that unit is **mm**, and it is the RMS point-to-ray
distance over the contributing cameras.

The rotation convention was verified against the promoted bundles rather than
assumed: cam0's rotation vector is a pi-rotation about +y, which under
``x_cam = R @ x_world + t`` puts its optical axis along -z, i.e. looking down into
the tank from z = 538 mm. ``tests/test_calib.py`` pins it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medaka3d.types import VIEWS

#: Newton iterations for the distortion inverse. The Brown model is contractive
#: here (|k1| < 0.25 at r ~ 0.3), and the residual is below 1e-12 normalised
#: units — a millionth of a pixel — after 5. Pinned by a test rather than tuned.
UNDISTORT_ITERATIONS = 20


class CalibrationError(RuntimeError):
    """Raised when a calibration bundle is missing, malformed, or inconsistent."""


def rodrigues(rotation_vector: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 matrix, OpenCV's convention (``x_cam = R @ x_world + t``)."""
    r = np.asarray(rotation_vector, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def refract(direction: np.ndarray, normal: np.ndarray, eta: float) -> np.ndarray:
    """Snell's law in vector form. ``normal`` points *along* the direction of travel.

    Args:
        direction: (..., 3) unit incident directions.
        normal: (3,) unit interface normal, oriented into the medium being entered.
        eta: ``n_from / n_to``.

    Returns:
        (..., 3) unit refracted directions. NaN where total internal reflection
        occurs — a real physical outcome that must not be silently clamped, since
        a clamped ray would produce a plausible-looking 3D point from light that
        never reached the camera.
    """
    d = np.asarray(direction, dtype=float)
    n = -np.asarray(normal, dtype=float).reshape(3)  # opposed to travel, as Snell wants
    cos_i = -np.sum(d * n, axis=-1, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    out = eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0.0))) * n
    return np.where(k < 0.0, np.nan, out / np.linalg.norm(out, axis=-1, keepdims=True))


@dataclass(frozen=True, slots=True)
class RefractiveCamera:
    """One camera's intrinsics, pose and water interface, from a promoted bundle."""

    name: str
    view: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]  # k1, k2, p1, p2, k3
    rotation: np.ndarray  # 3x3, world -> camera
    center: np.ndarray  # (3,) camera centre in world mm
    interface_normal: np.ndarray  # (3,) unit, pointing from air toward the water
    interface_point: np.ndarray  # (3,) a point on the first interface
    glass_thickness_mm: float
    index_air: float
    index_glass: float
    index_water: float
    layer_count: int

    @classmethod
    def from_bundle(cls, path: Path, view: str) -> RefractiveCamera:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("k4", "k5", "k6"):
            if float(raw.get(key, 0.0)) != 0.0:
                raise CalibrationError(
                    f"{path}: rational distortion term {key} is non-zero "
                    f"({raw[key]}); this module implements the Brown model only"
                )
        if int(raw["layer_count"]) not in (1, 2):
            raise CalibrationError(f"{path}: layer_count {raw['layer_count']} is not 1 or 2")
        normal = np.asarray(raw["interface_normal"], dtype=float)
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, atol=1e-9):
            raise CalibrationError(f"{path}: interface_normal is not a unit vector (|n| = {norm})")
        return cls(
            name=str(raw["name"]),
            view=view,
            fx=float(raw["fx"]),
            fy=float(raw["fy"]),
            cx=float(raw["cx"]),
            cy=float(raw["cy"]),
            distortion=(
                float(raw["k1"]),
                float(raw["k2"]),
                float(raw["p1"]),
                float(raw["p2"]),
                float(raw["k3"]),
            ),
            rotation=rodrigues(np.asarray(raw["rotation_vector"], dtype=float)),
            center=np.asarray(raw["camera_center"], dtype=float),
            interface_normal=normal,
            interface_point=np.asarray(raw["interface_point"], dtype=float),
            glass_thickness_mm=float(raw["glass_thickness_mm"]),
            index_air=float(raw["index_air"]),
            index_glass=float(raw["index_glass"]),
            index_water=float(raw["index_water"]),
            layer_count=int(raw["layer_count"]),
        )

    # -- pixels ------------------------------------------------------------
    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """(..., 2) pixels -> (..., 2) ideal normalised image coordinates."""
        uv = np.asarray(uv, dtype=float)
        k1, k2, p1, p2, k3 = self.distortion
        distorted = np.stack(
            [(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], axis=-1
        )
        xy = distorted.copy()
        for _ in range(UNDISTORT_ITERATIONS):
            x, y = xy[..., 0], xy[..., 1]
            r2 = x * x + y * y
            radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            xy = (distorted - np.stack([dx, dy], axis=-1)) / radial[..., None]
        return xy

    def rays(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(..., 2) pixels -> (origin, direction) of the corresponding rays *in water*.

        Origins sit on the water-side interface, so a point-to-ray distance is a
        distance to light that actually travelled through the water. NaN pixels
        propagate to NaN rays rather than to a confident wrong point.
        """
        uv = np.asarray(uv, dtype=float)
        xy = self.undistort(uv)
        cam = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
        direction = cam @ self.rotation  # == (R.T @ d_cam) for each row
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

        origin = np.broadcast_to(self.center, direction.shape).copy()
        normal = self.interface_normal

        # Air -> (glass ->) water. layer_count 1 is the top camera's free surface:
        # there is no pane above the water, so the 4 mm of glass must not be
        # applied there even though the bundle still carries its thickness.
        if self.layer_count == 2:
            planes = [
                (self.interface_point, self.index_air, self.index_glass),
                (
                    self.interface_point + normal * self.glass_thickness_mm,
                    self.index_glass,
                    self.index_water,
                ),
            ]
        else:
            planes = [(self.interface_point, self.index_air, self.index_water)]

        for point, n_from, n_to in planes:
            denominator = direction @ normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((point - origin) @ normal) / denominator
            # A ray running parallel to, or away from, the interface never enters
            # the water. Marking it NaN is the honest outcome.
            t = np.where(denominator <= 1e-12, np.nan, t)
            origin = origin + t[..., None] * direction
            direction = refract(direction, normal, n_from / n_to)

        return origin, direction


@dataclass(frozen=True, slots=True)
class Rig:
    """The three cameras, in the manifest's own view order."""

    cameras: tuple[RefractiveCamera, ...]
    manifest: Path
    world_units: str = "mm"

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(camera.view for camera in self.cameras)

    def camera(self, view: str) -> RefractiveCamera:
        for cam in self.cameras:
            if cam.view == view:
                return cam
        raise CalibrationError(f"no camera for view {view!r}; rig has {self.views}")

    def rays(self, uv_by_view: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Stack per-view pixels into (..., n_views, 3) ray origins and directions.

        The stacking order is ``self.views`` — the manifest's order — and every
        view must be supplied. side1 and side2 share every frame index, so a
        silently missing or duplicated view is exactly the class of mistake this
        project has already published once; it is an error, not a default.
        """
        missing = set(self.views) - set(uv_by_view)
        extra = set(uv_by_view) - set(self.views)
        if missing or extra:
            raise CalibrationError(
                f"rays() needs exactly {self.views}; missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}"
            )
        pairs = [self.camera(view).rays(uv_by_view[view]) for view in self.views]
        return (
            np.stack([o for o, _ in pairs], axis=-2),
            np.stack([d for _, d in pairs], axis=-2),
        )


def load_rig(calibration_dir: Path) -> Rig:
    """Load a promoted fixed-refractive calibration directory.

    ``calibration_dir`` holds ``calibration_manifest.json`` and the per-camera
    bundles. The manifest's ``view_order`` is authoritative: it is what ties
    ``cam1`` to ``side1_cam``, and getting that wrong would swap two cameras that
    share every frame index and produce a confident, silent, wrong reconstruction.
    """
    calibration_dir = Path(calibration_dir)
    manifest_path = calibration_dir / "calibration_manifest.json"
    if not manifest_path.is_file():
        raise CalibrationError(f"no calibration_manifest.json in {calibration_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("backend") != "refractive":
        raise CalibrationError(
            f"{manifest_path}: backend is {manifest.get('backend')!r}, not 'refractive'. "
            f"The archived polynomial backend is a historical-reproduction asset and "
            f"never an automatic fallback."
        )
    if manifest.get("world_units") != "mm":
        raise CalibrationError(f"{manifest_path}: world_units is {manifest.get('world_units')!r}")

    view_order = tuple(manifest["view_order"])
    if view_order != VIEWS:
        raise CalibrationError(f"{manifest_path}: view_order {view_order} is not {VIEWS}")

    cameras = []
    for entry, view in zip(manifest["model_files"], view_order, strict=True):
        if entry["view"] != view:
            raise CalibrationError(
                f"{manifest_path}: model_files order disagrees with view_order "
                f"({entry['view']} vs {view})"
            )
        cameras.append(RefractiveCamera.from_bundle(calibration_dir / entry["path"], view))
    return Rig(cameras=tuple(cameras), manifest=manifest_path)


def triangulate(
    origins: np.ndarray, directions: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares point closest to a bundle of rays, and the residual in mm.

    Args:
        origins: (..., n_rays, 3) points on each ray.
        directions: (..., n_rays, 3) unit directions.
        valid: optional (..., n_rays) bool saying which rays exist. Rays marked
            False are **left out of the fit entirely**, and a row with fewer than
            two of them returns NaN. Without it every ray counts and any
            non-finite ray makes the whole row NaN, which is the right answer
            when the caller intended all the rays to be there.

    Returns:
        ``(xyz, rms_mm)`` where ``xyz`` is (..., 3) and ``rms_mm`` is (...), the
        root-mean-square perpendicular distance from ``xyz`` to the rays that
        took part. For a correct correspondence this is the calibration's own
        accuracy (0.24 mm rms on held-out wet checkerboards); for a wrong one it
        is the separation between two animals, which is why it can rank
        correspondences.

    Rows containing any non-finite ray, or whose rays are too near-parallel to
    intersect (a rank-deficient normal matrix), return NaN rather than a
    least-squares answer that would look like a measurement.

    ``valid`` exists because a camera that lost one animal has not lost the
    other two. Dropping the whole row would throw away two good rays to punish a
    missing third; dropping the ray keeps the measurement and says, through
    ``rms_mm``, that it rests on less.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if origins.shape != directions.shape or origins.shape[-1] != 3:
        raise ValueError(f"origins {origins.shape} and directions {directions.shape} must match")
    if origins.shape[-2] < 2:
        raise ValueError("triangulation needs at least two rays")

    eye = np.eye(3)
    projector = eye - directions[..., :, None] * directions[..., None, :]
    if valid is None:
        n_rays = np.full(origins.shape[:-2], origins.shape[-2])
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != origins.shape[:-1]:
            raise ValueError(f"valid {valid.shape} must be {origins.shape[:-1]}")
        # A ray that does not exist contributes nothing to the normal equations.
        # Zeroing the projector is not the same as zeroing the direction: a zero
        # direction leaves the identity behind and would tug the fit.
        projector = np.where(valid[..., None, None], projector, 0.0)
        origins = np.where(valid[..., None], origins, 0.0)
        n_rays = valid.sum(axis=-1)
    a = projector.sum(axis=-3)
    b = (projector @ origins[..., None]).sum(axis=-3)[..., 0]

    finite = np.isfinite(a).all(axis=(-2, -1)) & np.isfinite(b).all(axis=-1) & (n_rays >= 2)
    # A 3x3 with a tiny determinant means the rays share a direction and the
    # "closest point" is anywhere along a line. 1e-6 is scale-free: the projector
    # sum has eigenvalues in [0, n_rays].
    solvable = finite & (np.abs(np.linalg.det(np.where(finite[..., None, None], a, eye))) > 1e-6)

    safe = np.where(solvable[..., None, None], a, eye)
    solution = np.linalg.solve(safe, np.where(solvable[..., None], b, 0.0)[..., None])[..., 0]
    xyz = np.where(solvable[..., None], solution, np.nan)

    offset = xyz[..., None, :] - origins
    perpendicular = offset - np.sum(offset * directions, axis=-1)[..., None] * directions
    squared = np.sum(perpendicular * perpendicular, axis=-1)
    if valid is not None:
        squared = np.where(valid, squared, 0.0)
    rms = np.sqrt(squared.sum(axis=-1) / np.maximum(n_rays, 1))
    return xyz, rms
