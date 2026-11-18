"""Synthetic benchmark objects defined by signed distance functions.

Every object stands on the plane y = 0 (world Y is up), is roughly one unit
across, and exposes ``sdf(points)`` (negative inside) and ``albedo(points)``.
Rotationally symmetric bodies are built from a 2D cross-section polygon that
is converted once into a finely sampled exact distance grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

Field = Callable[[np.ndarray], np.ndarray]


def polygon_sdf(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Exact signed distance from 2D ``points`` to a closed ``polygon``."""
    px, py = points[:, 0], points[:, 1]
    d2 = np.full(len(points), np.inf)
    sign = np.ones(len(points))
    count = len(polygon)
    for i in range(count):
        vi = polygon[i]
        vj = polygon[i - 1]
        ex, ey = vj[0] - vi[0], vj[1] - vi[1]
        wx, wy = px - vi[0], py - vi[1]
        denom = ex * ex + ey * ey
        t = np.clip((wx * ex + wy * ey) / denom, 0.0, 1.0) if denom > 0 else 0.0
        bx, by = wx - ex * t, wy - ey * t
        d2 = np.minimum(d2, bx * bx + by * by)
        c1 = py >= vi[1]
        c2 = py < vj[1]
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        sign[flip] *= -1.0
    return sign * np.sqrt(d2)


class RevolvedSection:
    """Solid of revolution about the Y axis from a right-half cross-section.

    ``half`` lists (r, y) vertices of the region with r >= 0, starting and
    ending on the axis (r = 0). It is mirrored so that the axis itself is
    interior rather than a spurious boundary.
    """

    def __init__(self, half: np.ndarray, cell: float = 0.0015) -> None:
        half = np.asarray(half, dtype=np.float64)
        mirrored = half[::-1].copy()
        mirrored[:, 0] *= -1.0
        polygon = np.vstack((half, mirrored[1:-1]))
        self.r_max = float(half[:, 0].max()) + 0.06
        self.y_min = float(half[:, 1].min()) - 0.06
        self.y_max = float(half[:, 1].max()) + 0.06
        self.cell = cell
        nr = int(np.ceil(self.r_max / cell)) + 1
        ny = int(np.ceil((self.y_max - self.y_min) / cell)) + 1
        rr, yy = np.meshgrid(
            np.arange(nr) * cell, self.y_min + np.arange(ny) * cell, indexing="xy"
        )
        grid = polygon_sdf(np.column_stack((rr.ravel(), yy.ravel())), polygon)
        self.grid = grid.reshape(ny, nr)
        self.half = half

    def __call__(self, points: np.ndarray) -> np.ndarray:
        radial = np.hypot(points[:, 0], points[:, 2])
        y = points[:, 1]
        coords = np.vstack(
            (
                (np.clip(y, self.y_min, self.y_max) - self.y_min) / self.cell,
                np.clip(radial, 0.0, self.r_max) / self.cell,
            )
        )
        inside = map_coordinates(self.grid, coords, order=1, mode="nearest")
        excess = np.hypot(
            np.maximum(radial - self.r_max, 0.0),
            np.maximum(np.maximum(self.y_min - y, y - self.y_max), 0.0),
        )
        return inside + excess


def vessel_section(
    profile: Callable[[np.ndarray], np.ndarray],
    height: float,
    *,
    hollow: bool,
    wall: float = 0.04,
    floor: float = 0.06,
    samples: int = 120,
) -> np.ndarray:
    """Right-half cross-section for an outer profile ``r = profile(y)``."""
    ys = np.linspace(0.0, height, samples)
    rs = profile(ys)
    outer = np.column_stack((rs, ys))
    if not hollow:
        return np.vstack(([[0.0, 0.0]], outer, [[0.0, height]]))
    inner_ys = np.linspace(height, floor, samples)
    inner_rs = np.maximum(profile(inner_ys) - wall, 0.01)
    inner = np.column_stack((inner_rs, inner_ys))
    return np.vstack(([[0.0, 0.0]], outer, inner, [[0.0, floor]]))


def torus_handle(
    center: tuple[float, float],
    major: float,
    minor: float,
    x_cut: float,
    side: float = 1.0,
) -> Field:
    """Half torus in the XY plane, keeping the part beyond ``x = x_cut``."""
    cx, cy = center

    def sdf(points: np.ndarray) -> np.ndarray:
        x = side * points[:, 0]
        ring = np.hypot(np.hypot(x - cx, points[:, 1] - cy) - major, points[:, 2])
        return np.maximum(ring - minor, x_cut - x)

    return sdf


def tapered_capsule(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    ra: float,
    rb: float,
) -> Field:
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    axis = b_arr - a_arr
    length2 = float(axis @ axis)

    def sdf(points: np.ndarray) -> np.ndarray:
        t = np.clip(((points - a_arr) @ axis) / length2, 0.0, 1.0)
        closest = a_arr + t[:, None] * axis
        return np.linalg.norm(points - closest, axis=1) - (ra + (rb - ra) * t)

    return sdf


def rounded_box(
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    radius: float,
) -> Field:
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half, dtype=np.float64) - radius

    def sdf(points: np.ndarray) -> np.ndarray:
        q = np.abs(points - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside - radius

    return sdf


def ellipsoid(
    center: tuple[float, float, float], radii: tuple[float, float, float]
) -> Field:
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)

    def sdf(points: np.ndarray) -> np.ndarray:
        p = points - c
        k0 = np.linalg.norm(p / r, axis=1)
        k1 = np.linalg.norm(p / (r * r), axis=1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    return sdf


def smooth_union(fields: list[Field], k: float) -> Field:
    def sdf(points: np.ndarray) -> np.ndarray:
        d = fields[0](points)
        for f in fields[1:]:
            e = f(points)
            if k <= 0:
                d = np.minimum(d, e)
                continue
            h = np.clip(0.5 + 0.5 * (e - d) / k, 0.0, 1.0)
            d = e * (1.0 - h) + d * h - k * h * (1.0 - h)
        return d

    return sdf


def _speckled(base: tuple[float, float, float], strength: float = 0.12) -> Field:
    base_arr = np.asarray(base, dtype=np.float64)

    def albedo(points: np.ndarray) -> np.ndarray:
        q = np.floor(points * 90.0)
        noise = np.sin(q @ np.array([12.9898, 78.233, 37.719])) * 43758.5453
        noise = noise - np.floor(noise)
        low = np.sin(points @ np.array([7.1, 3.3, 5.7])) * 0.5 + 0.5
        shade = 1.0 + strength * (noise - 0.5) + 0.5 * strength * (low - 0.5)
        return np.clip(base_arr[None, :] * shade[:, None], 0.0, 1.0)

    return albedo


@dataclass(frozen=True)
class BenchObject:
    name: str
    category: str  # "round" or "other"
    sdf: Field
    albedo: Field
    bounds: tuple[np.ndarray, np.ndarray]
    genus: int
    hollow: bool
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def center(self) -> np.ndarray:
        lo, hi = self.bounds
        return 0.5 * (lo + hi)

    @property
    def radius(self) -> float:
        lo, hi = self.bounds
        return float(0.5 * np.linalg.norm(hi - lo))


def _bounds(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _cup() -> BenchObject:
    height = 0.72

    def profile(y):
        t = y / height
        return 0.33 + 0.17 * np.sqrt(np.clip(t, 0.0, 1.0)) - 0.02 * t

    body = RevolvedSection(vessel_section(profile, height, hollow=True, wall=0.04))
    handle = torus_handle((0.56, 0.4), 0.17, 0.045, 0.4)
    return BenchObject(
        "cup",
        "round",
        smooth_union([body, handle], 0.02),
        _speckled((0.45, 0.26, 0.15)),
        _bounds([-0.5, 0.0, -0.5], [0.8, height, 0.5]),
        genus=1,
        hollow=True,
        tags=("handle",),
    )


def _bowl() -> BenchObject:
    height = 0.38

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        return 0.22 + 0.33 * np.sin(0.5 * np.pi * t) ** 0.8

    body = RevolvedSection(
        vessel_section(profile, height, hollow=True, wall=0.035, floor=0.05)
    )
    return BenchObject(
        "bowl",
        "round",
        body,
        _speckled((0.42, 0.55, 0.62), 0.06),
        _bounds([-0.56, 0.0, -0.56], [0.56, height, 0.56]),
        genus=0,
        hollow=True,
    )


def _vase() -> BenchObject:
    height = 1.1

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        bulge = 0.2 + 0.2 * np.sin(np.pi * np.clip(t / 0.75, 0.0, 1.0)) ** 1.2
        neck = 0.11 + 0.07 * np.clip((t - 0.85) / 0.15, 0.0, 1.0) ** 2
        blend = np.clip((t - 0.62) / 0.2, 0.0, 1.0)
        return (1.0 - blend) * bulge + blend * neck

    body = RevolvedSection(vessel_section(profile, height, hollow=True, wall=0.03))
    return BenchObject(
        "vase",
        "round",
        body,
        _speckled((0.25, 0.38, 0.55), 0.08),
        _bounds([-0.42, 0.0, -0.42], [0.42, height, 0.42]),
        genus=0,
        hollow=True,
        tags=("neck",),
    )


def _bottle() -> BenchObject:
    height = 1.15

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        body = 0.24 * np.ones_like(t)
        shoulder = 0.24 - 0.15 * np.sin(0.5 * np.pi * np.clip((t - 0.55) / 0.2, 0, 1))
        neck = np.where(t > 0.75, 0.09, shoulder)
        cap = np.where(t > 0.92, 0.1, neck)
        return np.where(t < 0.55, body, cap)

    body = RevolvedSection(vessel_section(profile, height, hollow=False))
    return BenchObject(
        "bottle",
        "round",
        body,
        _speckled((0.18, 0.45, 0.3), 0.05),
        _bounds([-0.25, 0.0, -0.25], [0.25, height, 0.25]),
        genus=0,
        hollow=False,
        notes="capped",
        tags=("neck",),
    )


def _jar() -> BenchObject:
    height = 0.8

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        body = 0.34 + 0.04 * np.sin(np.pi * np.clip(t / 0.78, 0.0, 1.0))
        lid = np.where(t > 0.78, 0.36 - 0.25 * ((t - 0.78) / 0.14) ** 2, body)
        knob = np.where(t > 0.9, 0.07, lid)
        return np.clip(knob, 0.05, None)

    body = RevolvedSection(vessel_section(profile, height, hollow=False))
    return BenchObject(
        "lidded_jar",
        "round",
        body,
        _speckled((0.62, 0.5, 0.32), 0.07),
        _bounds([-0.39, 0.0, -0.39], [0.39, height, 0.39]),
        genus=0,
        hollow=False,
        tags=("lid",),
    )


def _teapot() -> BenchObject:
    height = 0.62

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        body = 0.22 + 0.2 * np.sin(np.pi * np.clip(t / 0.82, 0.0, 1.0)) ** 0.7
        lid = np.where(t > 0.82, 0.16 - 0.1 * (t - 0.82) / 0.1, body)
        return np.where(t > 0.92, 0.04, lid)

    body = RevolvedSection(vessel_section(profile, height, hollow=False))
    handle = torus_handle((0.42, 0.33), 0.16, 0.04, 0.3)
    spout = tapered_capsule((-0.3, 0.2, 0.0), (-0.62, 0.48, 0.0), 0.075, 0.035)
    return BenchObject(
        "teapot",
        "round",
        smooth_union([body, handle, spout], 0.03),
        _speckled((0.55, 0.32, 0.38), 0.05),
        _bounds([-0.66, 0.0, -0.43], [0.63, height, 0.43]),
        genus=1,
        hollow=False,
        tags=("handle", "spout"),
    )


def _amphora() -> BenchObject:
    height = 1.0

    def profile(y):
        t = np.clip(y / height, 0.0, 1.0)
        bulge = 0.12 + 0.24 * np.sin(np.pi * np.clip(t / 0.8, 0.0, 1.0)) ** 1.1
        neck = 0.12 + 0.05 * np.clip((t - 0.88) / 0.12, 0.0, 1.0)
        blend = np.clip((t - 0.62) / 0.2, 0.0, 1.0)
        return (1.0 - blend) * bulge + blend * neck

    body = RevolvedSection(vessel_section(profile, height, hollow=True, wall=0.03))
    handles = [torus_handle((0.17, 0.7), 0.15, 0.03, 0.1, side=s) for s in (1.0, -1.0)]
    return BenchObject(
        "amphora",
        "round",
        smooth_union([body, *handles], 0.02),
        _speckled((0.66, 0.36, 0.2), 0.08),
        _bounds([-0.42, 0.0, -0.37], [0.42, height, 0.37]),
        genus=2,
        hollow=True,
        tags=("handle", "two_handles", "neck"),
    )


def _box() -> BenchObject:
    return BenchObject(
        "box",
        "other",
        rounded_box((0.0, 0.45, 0.0), (0.32, 0.45, 0.18), 0.02),
        _speckled((0.85, 0.3, 0.25), 0.05),
        _bounds([-0.32, 0.0, -0.18], [0.32, 0.9, 0.18]),
        genus=0,
        hollow=False,
    )


def _duck() -> BenchObject:
    parts = [
        ellipsoid((0.0, 0.3, 0.0), (0.42, 0.28, 0.3)),
        ellipsoid((0.27, 0.66, 0.0), (0.18, 0.18, 0.17)),
        ellipsoid((0.47, 0.64, 0.0), (0.12, 0.04, 0.07)),
        ellipsoid((-0.38, 0.42, 0.0), (0.12, 0.08, 0.1)),
    ]
    return BenchObject(
        "toy_duck",
        "other",
        smooth_union(parts, 0.06),
        _speckled((0.95, 0.8, 0.2), 0.05),
        _bounds([-0.5, 0.0, -0.31], [0.6, 0.85, 0.31]),
        genus=0,
        hollow=False,
    )


def _shoe() -> BenchObject:
    parts = [
        rounded_box((0.0, 0.04, 0.0), (0.5, 0.04, 0.17), 0.035),
        ellipsoid((-0.08, 0.16, 0.0), (0.42, 0.15, 0.16)),
        ellipsoid((-0.3, 0.3, 0.0), (0.17, 0.22, 0.15)),
    ]
    return BenchObject(
        "shoe",
        "other",
        smooth_union(parts, 0.05),
        _speckled((0.3, 0.32, 0.36), 0.08),
        _bounds([-0.5, 0.0, -0.17], [0.5, 0.52, 0.17]),
        genus=0,
        hollow=False,
    )


def _mannequin() -> BenchObject:
    def pair(make):
        return [make(1.0), make(-1.0)]

    parts = [
        ellipsoid((0, 0.885, 0.005), (0.055, 0.07, 0.06)),  # head
        tapered_capsule((0, 0.77, 0), (0, 0.835, 0.004), 0.028, 0.024),  # neck
        ellipsoid((0, 0.765, 0), (0.14, 0.035, 0.06)),  # shoulders
        ellipsoid((0, 0.70, 0.004), (0.115, 0.09, 0.075)),  # chest
        ellipsoid((0, 0.60, 0), (0.085, 0.06, 0.06)),  # waist
        ellipsoid((0, 0.52, 0), (0.105, 0.075, 0.07)),  # hips
    ]
    parts += pair(
        lambda s: tapered_capsule(
            (s * 0.13, 0.755, 0), (s * 0.19, 0.60, -0.01), 0.031, 0.025
        )
    )
    parts += pair(
        lambda s: tapered_capsule(
            (s * 0.19, 0.60, -0.01), (s * 0.225, 0.465, 0.015), 0.024, 0.018
        )
    )
    parts += pair(lambda s: ellipsoid((s * 0.232, 0.43, 0.02), (0.016, 0.038, 0.026)))
    parts += pair(
        lambda s: tapered_capsule(
            (s * 0.05, 0.50, 0), (s * 0.055, 0.27, 0.005), 0.046, 0.033
        )
    )
    parts += pair(
        lambda s: tapered_capsule(
            (s * 0.055, 0.27, 0.005), (s * 0.058, 0.055, -0.005), 0.033, 0.022
        )
    )
    parts += pair(lambda s: ellipsoid((s * 0.06, 0.04, 0.03), (0.026, 0.02, 0.058)))
    body = smooth_union(parts, 0.02)
    base = rounded_box((0, 0.008, 0.01), (0.16, 0.008, 0.13), 0.006)
    skin, stand = (
        _speckled((0.74, 0.56, 0.42), 0.06),
        _speckled((0.22, 0.22, 0.24), 0.04),
    )

    def albedo(points: np.ndarray) -> np.ndarray:
        return np.where((points[:, 1] < 0.02)[:, None], stand(points), skin(points))

    return BenchObject(
        "mannequin",
        "other",
        smooth_union([body, base], 0.0),
        albedo,
        _bounds([-0.27, 0.0, -0.14], [0.27, 0.96, 0.16]),
        genus=1,  # legs + hips + base plate form one loop
        hollow=False,
        tags=("thin_parts",),
    )


def _parts(parts: list[tuple[Field, tuple[float, float, float]]], k: float):
    """Smooth union of parts, coloured by whichever part is nearest."""
    fields = [f for f, _ in parts]
    colours = np.asarray([c for _, c in parts], dtype=np.float64)
    sdf = smooth_union(fields, k)

    def albedo(points: np.ndarray) -> np.ndarray:
        nearest = np.argmin(np.vstack([f(points) for f in fields]), axis=0)
        base = colours[nearest]
        q = np.floor(points * 160.0)
        noise = np.sin(q @ np.array([12.9898, 78.233, 37.719])) * 43758.5453
        noise = noise - np.floor(noise)
        return np.clip(base * (1.0 + 0.08 * (noise - 0.5))[:, None], 0.0, 1.0)

    return sdf, albedo


def _scaled(field: Field, scale: tuple[float, float, float]) -> Field:
    """Anisotropically scaled field (distance is a conservative bound)."""
    s = np.asarray(scale, dtype=np.float64)

    def sdf(points: np.ndarray) -> np.ndarray:
        return field(points / s) * float(s.min())

    return sdf


def _ea_nasir() -> BenchObject:
    """Ea-nasir, Old Babylonian copper merchant of Ur (c. 1750 BC).

    No likeness of him survives: this is an imagined figure in period dress
    (fringed wool robe and shawl, long beard, hands clasped as in Mesopotamian
    statuary) holding a copper bun ingot, on a stone base.
    """
    robe_profile = RevolvedSection(
        vessel_section(lambda y: 0.142 - 0.05 * (y / 0.52) ** 1.3, 0.52, hollow=False)
    )
    robe, skin, hair = (0.24, 0.31, 0.56), (0.62, 0.45, 0.32), (0.11, 0.08, 0.06)
    shawl, copper, stone = (0.66, 0.42, 0.18), (0.78, 0.43, 0.19), (0.46, 0.44, 0.41)

    def pair(make):
        return [make(1.0), make(-1.0)]

    parts = [
        (rounded_box((0.0, 0.015, 0.0), (0.2, 0.015, 0.17), 0.008), stone),
        (
            _scaled(
                lambda p: robe_profile(p - np.array([0.0, 0.03, 0.0])), (1.0, 1.0, 0.82)
            ),
            robe,
        ),
        (ellipsoid((0.0, 0.64, 0.0), (0.105, 0.13, 0.078)), robe),
        (
            _scaled(
                tapered_capsule((0.09, 0.5, 0.074), (-0.11, 0.75, 0.024), 0.03, 0.034),
                (1.0, 1.0, 0.8),
            ),
            shawl,
        ),
        (ellipsoid((0.0, 0.755, 0.0), (0.135, 0.04, 0.07)), shawl),
        (tapered_capsule((0.0, 0.77, 0.0), (0.0, 0.83, 0.005), 0.03, 0.026), skin),
        (ellipsoid((0.0, 0.875, 0.005), (0.056, 0.066, 0.061)), skin),
        (ellipsoid((0.0, 0.905, -0.008), (0.06, 0.042, 0.064)), hair),
        (ellipsoid((0.0, 0.79, 0.04), (0.045, 0.058, 0.032)), hair),
        (ellipsoid((0.0, 0.895, 0.052), (0.012, 0.012, 0.012)), skin),
    ]
    parts += [
        (
            tapered_capsule(
                (s * 0.125, 0.745, 0.0), (s * 0.14, 0.62, 0.02), 0.033, 0.028
            ),
            robe,
        )
        for s in (1.0, -1.0)
    ]
    parts += [
        (
            tapered_capsule(
                (s * 0.14, 0.62, 0.02), (s * 0.035, 0.665, 0.085), 0.027, 0.022
            ),
            skin,
        )
        for s in (1.0, -1.0)
    ]
    parts += [
        (ellipsoid((0.0, 0.667, 0.098), (0.05, 0.03, 0.028)), skin),
        (ellipsoid((0.0, 0.672, 0.128), (0.05, 0.019, 0.036)), copper),
    ]
    parts += [
        (ellipsoid((s * 0.05, 0.04, 0.07), (0.03, 0.018, 0.045)), skin)
        for s in (1.0, -1.0)
    ]
    del pair
    sdf, albedo = _parts(parts, 0.012)
    return BenchObject(
        "ea_nasir",
        "other",
        sdf,
        albedo,
        _bounds([-0.21, 0.0, -0.18], [0.21, 0.97, 0.18]),
        genus=0,
        hollow=False,
        notes=(
            "Ea-nasir, Old Babylonian copper merchant of Ur (c. 1750 BC). No "
            "likeness of him survives: an imagined figure in period dress "
            "(fringed wool robe and shawl, long beard, hands clasped as in "
            "Mesopotamian statuary) holding a copper bun ingot."
        ),
        tags=("figure", "artifact"),
    )


BUILDERS: dict[str, Callable[[], BenchObject]] = {
    "cup": _cup,
    "bowl": _bowl,
    "vase": _vase,
    "bottle": _bottle,
    "lidded_jar": _jar,
    "teapot": _teapot,
    "amphora": _amphora,
    "box": _box,
    "toy_duck": _duck,
    "shoe": _shoe,
    "mannequin": _mannequin,
}


def _cuneiform_depth(width: int, height: int, lines: int, seed: int) -> np.ndarray:
    """Depth map (0..1) of rows of wedge impressions, stylised cuneiform."""
    import cv2

    rng = np.random.default_rng(seed)
    depth = np.zeros((height, width), dtype=np.float32)
    margin_x, margin_y = 0.08 * width, 0.05 * height
    pitch = (height - 2 * margin_y) / lines
    size = 0.55 * pitch

    def wedge(cx: float, cy: float, angle: float, length: float) -> None:
        d = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-d[1], d[0]])
        head = np.array([cx, cy])
        tri = np.array(
            [
                head - 0.25 * size * d + 0.28 * size * n,
                head - 0.25 * size * d - 0.28 * size * n,
                head + 0.35 * size * d,
            ],
            dtype=np.float32,
        )
        cv2.fillConvexPoly(
            depth, np.round(tri * 4).astype(np.int32), 1.0, cv2.LINE_AA, 2
        )
        tail = head + length * size * d
        cv2.line(
            depth,
            tuple(np.round(head * 4).astype(int)),
            tuple(np.round(tail * 4).astype(int)),
            0.55,
            max(1, int(0.12 * size)),
            cv2.LINE_AA,
            2,
        )

    for row in range(lines):
        y = margin_y + (row + 0.5) * pitch
        x = margin_x + rng.uniform(0, 0.4) * size
        while x < width - margin_x - size:
            kind = rng.choice(["h", "v", "hh", "vv", "corner", "hv"])
            if kind in ("h", "hh"):
                for k in range(1 if kind == "h" else 2):
                    wedge(x, y + (k - 0.5 * (kind == "hh")) * 0.35 * size, 0.0, 1.1)
                x += 1.5 * size
            elif kind in ("v", "vv"):
                for k in range(1 if kind == "v" else 2):
                    wedge(x + k * 0.4 * size, y - 0.45 * size, np.pi / 2, 0.9)
                x += (0.9 + 0.4 * (kind == "vv")) * size
            elif kind == "corner":
                wedge(x, y, np.pi / 4, 0.25)
                x += 0.7 * size
            else:
                wedge(x, y, 0.0, 0.8)
                wedge(x + 0.9 * size, y - 0.45 * size, np.pi / 2, 0.9)
                x += 1.6 * size
            x += rng.uniform(0.15, 0.45) * size
    return cv2.GaussianBlur(depth, (0, 0), 1.2)


def _nanni_tablet() -> BenchObject:
    """Nanni's complaint to Ea-nasir (British Museum 131236, Ur, c. 1750 BC).

    A replica in proportion (11.6 x 5.0 x 2.6 cm), standing upright: baked
    clay with slightly pillowed faces, a couple of chipped corners, and rows
    of wedge impressions on both faces. The signs are stylised cuneiform, not
    the tablet's actual text. Nanni complains that Ea-nasir delivered copper
    ingots of poor quality and treated his messenger with contempt.
    """
    hx, hy, hz = 0.2155, 0.5, 0.112
    center = np.array([0.0, hy, 0.0])
    res_w, res_h = 440, 1024
    front = _cuneiform_depth(res_w, res_h, 14, seed=1750)
    back = _cuneiform_depth(res_w, res_h, 13, seed=1751)
    groove = 0.006
    box = rounded_box(tuple(center), (hx, hy, hz), 0.07)
    # Shallow broken corners: carving spheres centred just outside the edge.
    chips = [
        ((hx + 0.035, 2 * hy + 0.02, hz * 0.7), 0.075),
        ((hx + 0.05, 2 * hy - 0.06, hz * 0.9), 0.06),
        ((-hx - 0.04, 0.03, -hz - 0.01), 0.07),
    ]

    def inscription(points: np.ndarray) -> np.ndarray:
        u = (points[:, 0] + hx) / (2 * hx) * (res_w - 1)
        v = (1.0 - points[:, 1] / (2 * hy)) * (res_h - 1)
        coords = np.vstack((v, u))
        f = map_coordinates(front, coords, order=1, mode="constant")
        b = map_coordinates(back, coords, order=1, mode="constant")
        return np.where(points[:, 2] >= 0, f, b)

    def sdf(points: np.ndarray) -> np.ndarray:
        q = (points - center) / np.array([hx, hy, hz])
        pillow = (
            0.012
            * np.clip(1.0 - q[:, 0] ** 2, 0, 1)
            * np.clip(1.0 - q[:, 1] ** 2, 0, 1)
        )
        d = box(points) - pillow + groove * inscription(points)
        for c, r in chips:
            d = np.maximum(d, r - np.linalg.norm(points - np.asarray(c), axis=1))
        return d

    def albedo(points: np.ndarray) -> np.ndarray:
        base = np.array([0.66, 0.5, 0.36])
        q = np.floor(points * 220.0)
        noise = np.sin(q @ np.array([12.9898, 78.233, 37.719])) * 43758.5453
        noise = noise - np.floor(noise)
        mottle = np.sin(points @ np.array([23.0, 17.0, 29.0])) * 0.5 + 0.5
        shade = 1.0 + 0.1 * (noise - 0.5) - 0.12 * mottle - 0.35 * inscription(points)
        return np.clip(base[None, :] * shade[:, None], 0.0, 1.0)

    return BenchObject(
        "nanni_complaint_tablet",
        "other",
        sdf,
        albedo,
        _bounds([-hx, 0.0, -hz - 0.02], [hx, 2 * hy, hz + 0.02]),
        genus=0,
        hollow=False,
        notes=(
            "Replica of Nanni's complaint tablet to Ea-nasir (British Museum "
            "131236, Ur, c. 1750 BC) in proportion (11.6 x 5.0 x 2.6 cm); "
            "stylised cuneiform, not the actual text."
        ),
        tags=("artifact", "tablet"),
    )


# Historical artifacts: not part of the default benchmark run.
ARTIFACTS: dict[str, Callable[[], BenchObject]] = {
    "ea_nasir": _ea_nasir,
    "nanni_complaint_tablet": _nanni_tablet,
}


def get_object(name: str) -> BenchObject:
    builders = {**BUILDERS, **ARTIFACTS}
    try:
        return builders[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark object '{name}'. Choose from: {', '.join(builders)}."
        ) from exc
