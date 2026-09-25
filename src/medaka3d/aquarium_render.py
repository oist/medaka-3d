"""Software 3D aquarium view of measured mouth, pectoral, and tail positions.

Geometry is a schematic fish fitted to each frame's three observed nodes. This
module never interpolates time, repairs identities, or fabricates missing nodes.
Colours identify input slots, not biological identities. All coordinates are mm.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# BGR colours, chosen to remain distinct against the dark blue aquarium.
SLOT_COLOURS = ((83, 190, 255), (234, 216, 83), (186, 132, 247))
_TEXT = (233, 235, 225)
_MUTED = (157, 170, 166)
_RED = (100, 113, 255)
_RIGHT = np.array([0.8, -0.6, 0.0])
_UP = np.array([-0.3, -0.4, np.sqrt(0.75)])
_DEPTH = np.cross(_RIGHT, _UP)
_LIGHT = np.array([-0.3, -0.4, 0.8660254])


def _bounds_array(tank_bounds: np.ndarray | None) -> np.ndarray:
    bounds = np.array([[0.0, 0.0, 0.0], [192.0, 192.0, 186.0]])
    if tank_bounds is not None:
        bounds = np.asarray(tank_bounds, dtype=float)
        if bounds.shape == (3, 2):
            bounds = bounds.T
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("tank_bounds must be finite with shape (2, 3) or (3, 2)")
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError("tank_bounds maximum must exceed minimum on every axis")
    return bounds


def _corners(bounds: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [bounds[i, 0], bounds[j, 1], bounds[k, 2]]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )


def _projection(bounds: np.ndarray, width: int, height: int) -> tuple[float, np.ndarray]:
    corners = _corners(bounds)
    points = np.column_stack((corners @ _RIGHT, -corners @ _UP))
    extent = np.ptp(points, axis=0)
    scale = min(width * 0.57 / extent[0], height * 0.67 / extent[1])
    center = np.array([width * 0.351, height * 0.52])
    offset = center - (points.min(axis=0) + points.max(axis=0)) * 0.5 * scale
    return scale, offset


def project_points(
    xyz: np.ndarray,
    *,
    tank_bounds: np.ndarray | None = None,
    width: int = 1600,
    height: int = 900,
) -> np.ndarray:
    """Project world coordinates to pixels in the main orthographic aquarium."""
    bounds = _bounds_array(tank_bounds)
    scale, offset = _projection(bounds, width, height)
    points = np.asarray(xyz, dtype=float)
    return np.stack((points @ _RIGHT, -points @ _UP), axis=-1) * scale + offset


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(np.clip(point, -100_000, 100_000)).astype(int))


def _poly(image: np.ndarray, points: np.ndarray, colour: tuple) -> None:
    if np.isfinite(points).all():
        cv2.fillConvexPoly(
            image,
            np.rint(np.clip(points, -100_000, 100_000)).astype(np.int32),
            tuple(int(c) for c in colour),
            cv2.LINE_AA,
        )


def _line(
    image: np.ndarray, a: np.ndarray, b: np.ndarray, colour: tuple, thickness: int = 1
) -> None:
    if np.isfinite(a).all() and np.isfinite(b).all():
        cv2.line(image, _pixel(a), _pixel(b), colour, thickness, cv2.LINE_AA)


def _text(
    image: np.ndarray,
    value: str,
    pos: tuple[int, int],
    size: float = 0.5,
    colour: tuple = _TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, pos, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thickness, cv2.LINE_AA)


def _dashed(
    image: np.ndarray, a: np.ndarray, b: np.ndarray, colour: tuple = _RED, thickness: int = 2
) -> None:
    if not np.isfinite([a, b]).all():
        return
    length = np.linalg.norm(b - a)
    if length < 1:
        return
    steps = min(2000, int(length / 12) + 1)
    for index in range(steps):
        t = index * 12 / length
        if t >= 1:
            break
        _line(image, a + (b - a) * t, a + (b - a) * min(1, t + 6 / length), colour, thickness)


def _mini_layout(width: int, height: int, axis: int) -> tuple[int, int, int, int]:
    # axis 1: XY, axis 2: XZ.
    return (
        int(width * 0.736),
        int(height * (0.216 if axis == 1 else 0.565)),
        int(width * 0.213),
        int(height * 0.272),
    )


def _mini_project(
    xyz: np.ndarray, bounds: np.ndarray, width: int, height: int, axis: int
) -> np.ndarray:
    x, y, w, h = _mini_layout(width, height, axis)
    size = bounds[1] - bounds[0]
    scale = min(w / size[0], h / size[axis])
    offset = np.array([x + (w - size[0] * scale) / 2, y + (h + size[axis] * scale) / 2])
    points = (np.asarray(xyz) - bounds[0])[..., [0, axis]]
    return points * np.array([scale, -scale]) + offset


@lru_cache(maxsize=8)
def _background(width: int, height: int, flat_bounds: tuple[float, ...]) -> np.ndarray:
    bounds = np.array(flat_bounds).reshape(2, 3)
    image = np.empty((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(0, 1, height)[:, None, None]
    top = np.array([38, 29, 19])[None, None, :]
    bottom = np.array([24, 22, 17])[None, None, :]
    image[:] = np.asarray(top * (1 - gradient) + bottom * gradient, dtype=np.uint8)
    sx, sy = width / 1600, height / 900

    def project(points: np.ndarray) -> np.ndarray:
        scale, offset = _projection(bounds, width, height)
        points = np.asarray(points)
        return np.column_stack((points @ _RIGHT, -points @ _UP)) * scale + offset

    corners = _corners(bounds)
    pixels = project(corners)
    # Rear walls and floor: subtle tint lets the fish remain the focal point.
    _poly(image, pixels[[0, 2, 3, 1]], (47, 43, 26))
    _poly(image, pixels[[0, 1, 5, 4]], (48, 39, 24))
    _poly(image, pixels[[0, 4, 6, 2]], (48, 47, 30))
    for axis in (0, 1):
        other = 1 - axis
        for coordinate in np.arange(np.ceil(bounds[0, axis] / 20) * 20, bounds[1, axis] + 0.01, 20):
            a, b = bounds[0].copy(), bounds[0].copy()
            a[axis] = b[axis] = coordinate
            b[other] = bounds[1, other]
            _line(image, *project(np.array([a, b])), (64, 65, 43))
    for z in np.arange(np.ceil(bounds[0, 2] / 20) * 20, bounds[1, 2] + 0.01, 20):
        a = bounds[0].copy()
        a[2] = z
        b = a.copy()
        b[0] = bounds[1, 0]
        c = a.copy()
        c[1] = bounds[1, 1]
        _line(image, *project(np.array([a, b])), (55, 51, 33))
        _line(image, *project(np.array([a, c])), (53, 55, 33))
    # All twelve tank edges, with nearest edges drawn more softly.
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i:
                _line(image, pixels[i], pixels[j], (108, 115, 83), max(1, round(sx)))
    _text(image, "ORTHOGRAPHIC  /  XYZ", (int(56 * sx), int(151 * sy)), 0.43 * sx, _MUTED)
    _text(image, "20 mm floor grid", (int(56 * sx), int(177 * sy)), 0.39 * sx, _MUTED)
    # Reference coordinate triad gives orientation without covering the tank.
    origin = np.array([width * 0.083, height * 0.77])
    triad_colours = ((135, 157, 217), (144, 201, 152), (229, 171, 134))
    for axis, (label, colour) in enumerate(zip("XYZ", triad_colours, strict=True)):
        direction = np.array([_RIGHT[axis], -_UP[axis]]) * 46 * sx
        endpoint = origin + direction
        cv2.arrowedLine(
            image,
            _pixel(origin),
            _pixel(endpoint),
            colour,
            max(1, round(sx)),
            cv2.LINE_AA,
            tipLength=0.13,
        )
        _text(image, label, _pixel(endpoint + np.array([6 * sx, 2 * sy])), 0.43 * sx, colour)

    cv2.line(
        image,
        (int(width * 0.697), int(height * 0.145)),
        (int(width * 0.697), int(height * 0.855)),
        (67, 66, 48),
        1,
        cv2.LINE_AA,
    )
    for axis, heading, vertical in ((1, "TOP VIEW", "Y"), (2, "SIDE VIEW", "Z")):
        x, y, _w, _h = _mini_layout(width, height, axis)
        _text(image, heading, (x, y - int(23 * sy)), 0.49 * sx, _TEXT)
        _text(
            image, f"X / {vertical}   mm", (x + int(224 * sx), y - int(23 * sy)), 0.38 * sx, _MUTED
        )
        low, high = _mini_project(bounds, bounds, width, height, axis)
        cv2.rectangle(
            image,
            _pixel([low[0], high[1]]),
            _pixel([high[0], low[1]]),
            (41, 39, 26),
            -1,
            cv2.LINE_AA,
        )
        for d in (0, axis):
            for value in np.arange(np.ceil(bounds[0, d] / 40) * 40, bounds[1, d] + 0.01, 40):
                a, b = bounds[0].copy(), bounds[1].copy()
                a[d] = b[d] = value
                _line(
                    image,
                    *_mini_project(np.array([a, b]), bounds, width, height, axis),
                    (61, 60, 43),
                )
        cv2.rectangle(
            image,
            _pixel([low[0], high[1]]),
            _pixel([high[0], low[1]]),
            (107, 112, 84),
            1,
            cv2.LINE_AA,
        )
        _text(
            image,
            f"{bounds[0, axis]:g}",
            _pixel([low[0] - 23 * sx, low[1] + 5 * sy]),
            0.32 * sx,
            _MUTED,
        )
        _text(
            image,
            f"{bounds[1, axis]:g}",
            _pixel([low[0] - 31 * sx, high[1] + 5 * sy]),
            0.32 * sx,
            _MUTED,
        )
        _text(
            image,
            f"{bounds[1, 0]:g}",
            _pixel([high[0] - 20 * sx, low[1] + 19 * sy]),
            0.32 * sx,
            _MUTED,
        )
    cv2.line(
        image,
        (int(48 * sx), int(801 * sy)),
        (int(1552 * sx), int(801 * sy)),
        (74, 72, 52),
        1,
        cv2.LINE_AA,
    )
    return image


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if np.isfinite(length) and length > 1e-7 else fallback


def _fish_mesh(nodes: np.ndarray, colour: tuple) -> tuple[list, list]:
    """Return shaded world-space faces and eye centres for one schematic fish."""
    mouth, pec, tail = nodes
    front_length = np.linalg.norm(mouth - pec)
    rear_length = np.linalg.norm(pec - tail)
    length = front_length + rear_length
    if not np.isfinite(length) or min(front_length, rear_length) < 0.05:
        return [], []
    forward = _unit(mouth - tail, np.array([1.0, 0.0, 0.0]))
    lateral = _unit(np.cross(np.array([0.0, 0.0, 1.0]), forward), np.array([1.0, 0.0, 0.0]))
    up = _unit(np.cross(forward, lateral), np.array([0.0, 0.0, 1.0]))
    # Dimensionless stations follow the measured centreline; width/depth are
    # schematic proportions, not additional anatomical measurements.
    radius = min(length * 0.081, 3.5)
    stations = (
        (0.0, 0.05),
        (0.07, 0.50),
        (0.17, 0.86),
        (0.32, 1.0),
        (0.47, 0.89),
        (0.62, 0.64),
        (0.77, 0.36),
        (0.88, 0.19),
        (0.92, 0.16),
    )
    pec_t = front_length / length

    def center(t: float) -> np.ndarray:
        if t <= pec_t:
            return mouth + (pec - mouth) * t / pec_t
        return pec + (tail - pec) * (t - pec_t) / (1 - pec_t)

    angles = np.linspace(0, np.pi * 2, 12, endpoint=False)
    rings = []
    for t, r in stations:
        local_forward = _unit(mouth - pec if t < pec_t else pec - tail, forward)
        local_lateral = _unit(np.cross([0.0, 0.0, 1.0], local_forward), lateral)
        local_up = _unit(np.cross(local_forward, local_lateral), up)
        rings.append(
            center(t)
            + radius
            * r
            * (np.cos(angles)[:, None] * local_lateral * 0.70 + np.sin(angles)[:, None] * local_up)
        )
    faces = []
    base = np.array(colour, dtype=float)

    def face(vertices: np.ndarray, tint: np.ndarray, light: bool = True) -> None:
        normal = _unit(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]), up)
        brightness = 0.52 + 0.48 * abs(float(normal @ _LIGHT)) if light else 1
        shaded = np.clip(tint * brightness, 0, 255).astype(int)
        faces.append((float(np.mean(vertices, axis=0) @ _DEPTH), vertices, tuple(shaded)))

    for i in range(len(rings) - 1):
        for j in range(len(angles)):
            k = (j + 1) % len(angles)
            points = np.array([rings[i][j], rings[i + 1][j], rings[i + 1][k], rings[i][k]])
            dorsal = max(0.0, np.sin((angles[j] + angles[k]) / 2))
            tint = base * (1 - 0.20 * dorsal) + np.array([245, 248, 247]) * 0.09
            face(points, tint)
    # Tail endpoint remains exactly the measured tail node; the fan is transverse.
    caudal_root = center(0.88)
    tail_tint = base * 0.70 + np.array([32, 35, 26])
    face(np.array([caudal_root, tail + up * radius * 1.22, tail - up * radius * 1.22]), tail_tint)
    face(
        np.array(
            [
                center(0.44) + up * radius * 0.91,
                center(0.66) + up * radius * 1.75,
                center(0.80) + up * radius * 0.38,
            ]
        ),
        tail_tint,
    )
    face(
        np.array(
            [
                center(0.47) - up * radius * 0.70,
                center(0.70) - up * radius * 1.24,
                center(0.84) - up * radius * 0.20,
            ]
        ),
        tail_tint * 0.9,
    )
    for sign in (-1, 1):
        face(
            np.array(
                [
                    pec + lateral * sign * radius * 0.55,
                    pec
                    - forward * length * 0.14
                    + lateral * sign * radius * 1.55
                    - up * radius * 0.28,
                    pec - forward * length * 0.20 + lateral * sign * radius * 0.60,
                ]
            ),
            tail_tint,
        )
    eye_center = center(0.115) + up * radius * 0.24
    eyes = [
        (eye_center + lateral * sign * radius * 0.40, max(0.20, radius * 0.16))
        for sign in (-1, 1)
        if (lateral * sign) @ _DEPTH < 0
    ]
    return faces, eyes


def _trail(image: np.ndarray, points: np.ndarray, colour: tuple, thickness: int) -> None:
    if len(points) < 2:
        return
    finite = np.isfinite(points).all(axis=1)
    valid_edges = finite[:-1] & finite[1:]
    base = np.array(colour, dtype=float)
    for index in np.flatnonzero(valid_edges):
        strength = 0.19 + 0.48 * (index + 1) / (len(points) - 1)
        faded = tuple(int(c) for c in base * strength + np.array([37, 35, 24]) * (1 - strength))
        _line(image, points[index], points[index + 1], faded, thickness)


def render_frame(
    pose: np.ndarray,
    trails: list[np.ndarray],
    *,
    frame: int,
    source_fps: float,
    elapsed_seconds: float,
    duration_seconds: float,
    alerts: list[dict],
    title: str = "Medaka / tracked motion",
    width: int = 1600,
    height: int = 900,
    tank_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Render a BGR frame without changing any measured position.

    ``pose`` has shape (3 slots, 3 nodes, 3 coordinates), with nodes ordered
    mouth, pectoral, tail. NaNs break trails and mark missing observations.
    Partial poses show only observed nodes and adjacent measured segments.
    ``alerts`` use ``type`` (jump/reappearance), ``slot``, optional
    ``distance_mm``, and optional ``start_xyz``/``end_xyz``. They are diagnostic
    candidates, never statements that an identity swap actually occurred.
    ``tank_bounds`` is min/max XYZ, shape (2, 3); (3, 2) is also accepted.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3, 3, 3):
        raise ValueError("pose must have shape (3 slots, 3 nodes, 3 coordinates)")
    if len(trails) != 3:
        raise ValueError("trails must contain one Nx3 array for each of three slots")
    if width < 640 or height < 360:
        raise ValueError("render size must be at least 640 x 360")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    bounds = _bounds_array(tank_bounds)
    image = _background(width, height, tuple(bounds.ravel())).copy()
    sx, sy = width / 1600, height / 900
    scale, offset = _projection(bounds, width, height)

    def project(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.stack((points @ _RIGHT, -points @ _UP), axis=-1) * scale + offset

    _text(
        image,
        "TRACKING  /  AQUARIUM STUDY",
        (int(48 * sx), int(42 * sy)),
        0.43 * sx,
        (161, 187, 169),
    )
    _text(image, title, (int(46 * sx), int(86 * sy)), 0.94 * sx, _TEXT, max(1, round(2 * sx)))
    _text(
        image,
        "Measured 3D pose  |  schematic fish",
        (int(48 * sx), int(116 * sy)),
        0.46 * sx,
        _MUTED,
    )
    _text(
        image,
        f"{elapsed_seconds:05.2f} / {duration_seconds:05.2f} s",
        (int(1228 * sx), int(77 * sy)),
        0.71 * sx,
        _TEXT,
        max(1, round(2 * sx)),
    )
    _text(image, "CLIP TIME", (int(1230 * sx), int(43 * sy)), 0.39 * sx, _MUTED)

    for slot, trail in enumerate(trails):
        trail = np.asarray(trail, dtype=float)
        if trail.size == 0:
            continue
        if trail.ndim != 2 or trail.shape[1] != 3:
            raise ValueError("each trail must have shape (N, 3)")
        _trail(image, project(trail), SLOT_COLOURS[slot], max(1, round(2 * sx)))
        for axis in (1, 2):
            _trail(
                image,
                _mini_project(trail, bounds, width, height, axis),
                SLOT_COLOURS[slot],
                max(1, round(sx)),
            )

    observed = np.isfinite(pose).all(axis=2)
    faces, eyes = [], []
    for slot in range(3):
        colour = SLOT_COLOURS[slot]
        nodes = pose[slot]
        if observed[slot, 1]:
            floor = nodes[1].copy()
            floor[2] = bounds[0, 2]
            floor_pixel, center_pixel = project(np.array([floor, nodes[1]]))
            _dashed(image, floor_pixel, center_pixel, (81, 85, 59), 1)
            cv2.ellipse(
                image,
                _pixel(floor_pixel),
                (max(3, int(10 * sx)), max(2, int(3 * sy))),
                0,
                0,
                360,
                (30, 32, 22),
                -1,
                cv2.LINE_AA,
            )
        if observed[slot].all():
            fish_faces, fish_eyes = _fish_mesh(nodes, colour)
            faces.extend(fish_faces)
            eyes.extend(fish_eyes)
            if not fish_faces:
                for a, b in zip(project(nodes[:-1]), project(nodes[1:]), strict=True):
                    _line(image, a, b, colour, max(1, round(2 * sx)))
        else:
            projected = project(nodes)
            for node in np.flatnonzero(observed[slot]):
                cv2.circle(
                    image, _pixel(projected[node]), max(2, round(3 * sx)), colour, -1, cv2.LINE_AA
                )
            for index in (0, 1):
                if observed[slot, index : index + 2].all():
                    _line(
                        image, projected[index], projected[index + 1], colour, max(1, round(2 * sx))
                    )
        for axis in (1, 2):
            points = _mini_project(nodes, bounds, width, height, axis)
            for index in (0, 1):
                if observed[slot, index : index + 2].all():
                    _line(image, points[index], points[index + 1], colour, max(1, round(2 * sx)))
            for index in np.flatnonzero(observed[slot]):
                cv2.circle(
                    image,
                    _pixel(points[index]),
                    max(2, round((3 if index == 0 else 2) * sx)),
                    colour,
                    -1,
                    cv2.LINE_AA,
                )
            if observed[slot].any():
                anchor = points[np.flatnonzero(observed[slot])[0]]
                _text(
                    image,
                    str(slot),
                    _pixel(anchor + np.array([6 * sx, -5 * sy])),
                    0.38 * sx,
                    colour,
                )

    # _DEPTH points away from the viewer; paint distant triangles first.
    for _, vertices, colour in sorted(faces, key=lambda item: item[0], reverse=True):
        _poly(image, project(vertices), colour)
    for eye, radius in eyes:
        pixel = _pixel(project(eye))
        cv2.circle(
            image, pixel, max(2, round(radius * scale * 1.4)), (215, 224, 220), -1, cv2.LINE_AA
        )
        cv2.circle(image, pixel, max(1, round(radius * scale)), (19, 25, 24), -1, cv2.LINE_AA)

    # A small label offset per slot keeps labels legible when fish overlap.
    for slot in range(3):
        colour = SLOT_COLOURS[slot]
        if observed[slot].any():
            center = pose[slot, 1] if observed[slot, 1] else pose[slot, observed[slot]][0]
            point = project(center)
            label = point + np.array([22 * sx, (-24 + slot * 20) * sy])
            _line(
                image,
                point + np.array([5 * sx, 0]),
                label + np.array([-4 * sx, -5 * sy]),
                tuple(int(c * 0.60) for c in colour),
            )
            _text(image, f"ID {slot}", _pixel(label), 0.45 * sx, colour)

    active = []
    for alert in alerts:
        slot = int(alert.get("slot", -1))
        if slot not in (0, 1, 2):
            continue
        kind = "JUMP candidate" if alert.get("type") == "jump" else "REAPPEARANCE"
        distance = alert.get("distance_mm")
        suffix = f"  {float(distance):.1f} mm" if distance is not None else ""
        active.append(f"ID {slot}  {kind}{suffix}")
        a = np.asarray(alert.get("start_xyz", np.full(3, np.nan)), dtype=float)
        b = np.asarray(alert.get("end_xyz", pose[slot, 1]), dtype=float)
        if a.shape == (3,) and b.shape == (3,):
            _dashed(image, project(a), project(b), _RED, max(1, round(2 * sx)))
            for axis in (1, 2):
                _dashed(
                    image,
                    _mini_project(a, bounds, width, height, axis),
                    _mini_project(b, bounds, width, height, axis),
                    _RED,
                    max(1, round(sx)),
                )
        if b.shape == (3,) and np.isfinite(b).all():
            radius = int((18 + 5 * (0.5 + 0.5 * np.sin(elapsed_seconds * 15))) * sx)
            cv2.circle(
                image, _pixel(project(b)), max(4, radius), _RED, max(1, round(2 * sx)), cv2.LINE_AA
            )
    if active:
        for index, text in enumerate(active[:3]):
            _text(image, text, (int(285 * sx), int((750 + index * 20) * sy)), 0.42 * sx, _RED)

    for slot in range(3):
        x = int((51 + slot * 250) * sx)
        colour = SLOT_COLOURS[slot]
        cv2.circle(image, (x, int(833 * sy)), max(3, round(5 * sx)), colour, -1, cv2.LINE_AA)
        count = int(observed[slot].sum())
        status = "observed" if count == 3 else f"partial {count}/3" if count else "missing"
        _text(image, f"ID {slot}  {status}", (x + int(15 * sx), int(839 * sy)), 0.46 * sx, colour)
    _text(
        image,
        f"SOURCE FRAME  {frame:,}    |    t = {frame / source_fps:.3f} s",
        (int(1004 * sx), int(838 * sy)),
        0.43 * sx,
        _TEXT,
    )
    _text(
        image,
        "Raw positions / no temporal smoothing / gaps kept / colour = tracking slot",
        (int(49 * sx), int(875 * sy)),
        0.42 * sx,
        _MUTED,
    )
    _text(image, f"{source_fps:g} Hz source", (int(1371 * sx), int(875 * sy)), 0.4 * sx, _MUTED)
    progress = np.clip(elapsed_seconds / max(duration_seconds, 1e-9), 0, 1)
    cv2.line(
        image,
        (0, height - 3),
        (int(progress * (width - 1)), height - 3),
        (172, 198, 163),
        max(2, round(3 * sy)),
        cv2.LINE_AA,
    )
    return image
