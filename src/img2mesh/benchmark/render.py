"""Sphere-traced product-shot renders of benchmark objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..geometry import CameraIntrinsics
from .objects import BenchObject

# Product-style views: (azimuth, elevation) in degrees. Azimuth 0 looks along
# -Z with +X (where handles sit) on the right. The last view is held out.
DEFAULT_VIEWS: tuple[tuple[float, float], ...] = (
    (0.0, 4.0),
    (25.0, 14.0),
    (-30.0, 9.0),
    (55.0, 32.0),
    (-15.0, 52.0),
    (8.0, 86.0),
    (150.0, 22.0),
)

_KEY_LIGHT = np.array([-0.45, 0.8, 0.55]) / np.linalg.norm([-0.45, 0.8, 0.55])
_FILL_LIGHT = np.array([0.6, 0.3, 0.5]) / np.linalg.norm([0.6, 0.3, 0.5])


@dataclass(frozen=True)
class RenderedView:
    image: Image.Image
    mask: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: CameraIntrinsics
    azimuth: float
    elevation: float


def look_at(
    azimuth: float, elevation: float, distance: float, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-to-camera rotation/translation (x right, y down, z forward)."""
    az, el = np.radians(azimuth), np.radians(elevation)
    offset = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    center = target + distance * offset
    forward = -offset
    up = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up) > 0.999:
        up = np.array([0.0, 0.0, -1.0 if forward[1] < 0 else 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return rotation, -rotation @ center, center


def _normals(obj: BenchObject, points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grads = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = eps
        grads.append(obj.sdf(points + step) - obj.sdf(points - step))
    n = np.column_stack(grads)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _soft_shadow(
    obj: BenchObject, points: np.ndarray, light: np.ndarray, k: float = 10.0
) -> np.ndarray:
    shade = np.ones(len(points))
    t = np.full(len(points), 0.02)
    active = np.ones(len(points), dtype=bool)
    for _ in range(40):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        d = obj.sdf(points[idx] + light * t[idx, None])
        shade[idx] = np.minimum(shade[idx], k * np.maximum(d, 0.0) / t[idx])
        t[idx] += np.clip(d, 0.01, 0.15)
        active[idx] = (d > 1e-4) & (t[idx] < 2.5)
    return np.clip(shade, 0.0, 1.0)


def _ambient_occlusion(
    obj: BenchObject, points: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    occ = np.zeros(len(points))
    weight = 1.0
    for i in range(1, 6):
        h = 0.03 * i
        d = obj.sdf(points + normals * h)
        occ += weight * np.maximum(h - d, 0.0)
        weight *= 0.7
    return np.clip(1.0 - 3.0 * occ, 0.0, 1.0)


def render_view(
    obj: BenchObject,
    azimuth: float,
    elevation: float,
    *,
    size: int = 256,
    fill: float = 0.78,
    distance_factor: float = 12.0,
) -> RenderedView:
    """Render ``obj`` on a white canvas from a product-style camera."""
    target = obj.center
    radius = obj.radius
    distance = distance_factor * radius
    focal = fill * 0.5 * size * distance / radius
    intrinsics = CameraIntrinsics.from_focal_length(size, size, focal)
    rotation, translation, eye = look_at(azimuth, elevation, distance, target)

    us, vs = np.meshgrid(np.arange(size) + 0.0, np.arange(size) + 0.0)
    dirs_cam = np.column_stack(
        (
            (us.ravel() - intrinsics.cx) / focal,
            (vs.ravel() - intrinsics.cy) / focal,
            np.ones(size * size),
        )
    )
    dirs = dirs_cam @ rotation
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Bounding-sphere entry/exit.
    oc = eye - target
    b = dirs @ oc
    c = oc @ oc - (1.05 * radius) ** 2
    disc = b * b - c
    hit_sphere = disc > 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = np.maximum(-b - sq, 0.0)
    t_far = -b + sq

    hit = np.zeros(size * size, dtype=bool)
    active = hit_sphere.copy()
    eps = 2e-4 * radius
    for _ in range(220):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        d = obj.sdf(eye + dirs[idx] * t[idx, None])
        done = d < eps
        hit[idx[done]] = True
        t[idx] += 0.9 * np.maximum(d, 0.0)
        active[idx] = ~done & (t[idx] < t_far[idx])

    image = np.full((size * size, 3), 1.0)
    idx = np.flatnonzero(hit)
    if len(idx):
        points = eye + dirs[idx] * t[idx, None]
        normals = _normals(obj, points)
        facing = np.where((normals * dirs[idx]).sum(axis=1)[:, None] > 0, -1.0, 1.0)
        normals = normals * facing
        key = np.clip(normals @ _KEY_LIGHT, 0.0, 1.0)
        key *= _soft_shadow(obj, points + normals * 2e-3, _KEY_LIGHT)
        fill_l = np.clip(normals @ _FILL_LIGHT, 0.0, 1.0)
        ao = _ambient_occlusion(obj, points, normals)
        light = 0.28 * ao + 0.75 * key + 0.22 * fill_l * ao
        image[idx] = np.clip(obj.albedo(points) * light[:, None] * 1.25, 0.0, 1.0)

    rgb = (image.reshape(size, size, 3) * 255.0 + 0.5).astype(np.uint8)
    return RenderedView(
        image=Image.fromarray(rgb),
        mask=hit.reshape(size, size),
        rotation=rotation,
        translation=translation,
        intrinsics=intrinsics,
        azimuth=azimuth,
        elevation=elevation,
    )


def render_views(
    obj: BenchObject,
    views: tuple[tuple[float, float], ...] = DEFAULT_VIEWS,
    *,
    size: int = 256,
) -> list[RenderedView]:
    return [render_view(obj, az, el, size=size) for az, el in views]
