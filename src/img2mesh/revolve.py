"""Shape-prior reconstruction for rotationally symmetric objects.

Cups, mugs, bowls, vases and bottles are surfaces of revolution, optionally
with a single handle. Pure silhouette carving cannot recover their concave
interior and needs accurate camera poses, which casual photo sets (or crops
of a product shot) rarely provide. This module instead:

1. measures the outer profile from near-side-on views,
2. revolves it into a smooth (optionally hollow) body,
3. sweeps the handle from the view where it lies in the image plane, using a
   top view (when available) for its out-of-plane thickness,
4. fits a camera pose per view against the known shape, and
5. projects vertex colours from the photographs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import ReconstructionError
from .geometry import CameraIntrinsics
from .image import largest_component

PROFILE_SAMPLES = 160
SIDE_VIEW_MAX_ELEVATION = 25.0
TOP_VIEW_MIN_ELEVATION = 70.0
_Y_UP_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True, slots=True)
class ViewAnalysis:
    mask: np.ndarray
    body: np.ndarray
    axis_x: float
    handle_side: int
    handle_area: int
    elevation_deg: float
    rim_y: float
    foot_y: float
    diameter_px: float
    free_radius: np.ndarray
    symmetry_error: float
    profile: np.ndarray | None
    height_ratio: float | None


@dataclass(frozen=True, slots=True)
class RevolutionModel:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    rotations: tuple[np.ndarray, ...]
    translations: tuple[np.ndarray, ...]
    profile: np.ndarray
    height_ratio: float
    hollow: bool
    has_handle: bool


def _row_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    any_row = mask.any(axis=1)
    width = mask.shape[1]
    left = np.where(any_row, np.argmax(mask, axis=1), -1).astype(np.float64)
    right = np.where(any_row, width - 1 - np.argmax(mask[:, ::-1], axis=1), -1).astype(
        np.float64
    )
    return left, right


def _body_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    k = max(5, int(round(0.22 * min(bw, bh))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    body = largest_component(opened) > 0
    return body if body.any() else mask.copy()


def analyze_view(mask: np.ndarray) -> ViewAnalysis:
    """Measure axis, handle side, elevation and outer profile of one view."""
    mask = np.asarray(mask) > 0
    height = mask.shape[0]
    body = _body_mask(mask)

    body_rows = np.flatnonzero(body.any(axis=1))
    lo, hi = body_rows[0], body_rows[-1]
    core = body_rows[
        (body_rows >= lo + 0.2 * (hi - lo)) & (body_rows <= hi - 0.2 * (hi - lo))
    ]
    if len(core) == 0:
        core = body_rows
    b_left, b_right = _row_edges(body)
    axis_x = float(np.median(0.5 * (b_left[core] + b_right[core])))

    grown = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    handle = mask & ~grown
    if handle.any():
        handle = largest_component(handle.astype(np.uint8) * 255) > 0
    hy, hx = np.nonzero(handle)
    handle_area = int(len(hx))
    handle_side = 0
    if handle_area > 0.01 * np.count_nonzero(mask):
        handle_side = 1 if float(np.mean(hx)) > axis_x else -1
    else:
        handle_area = 0

    left, right = _row_edges(mask)
    valid = left >= 0
    r_left = np.where(valid, axis_x - left, np.nan)
    r_right = np.where(valid, right - axis_x, np.nan)
    if handle_side > 0:
        free, other = r_left, r_right
    elif handle_side < 0:
        free, other = r_right, r_left
    else:
        free = 0.5 * (r_left + r_right)
        other = free

    handle_rows = np.zeros(height, dtype=bool)
    if handle_area:
        pad = max(3, int(0.03 * height))
        handle_rows[max(0, hy.min() - pad) : min(height, hy.max() + pad + 1)] = True
        if handle_side > 0:
            r_right_full = np.where(valid, right - axis_x, np.nan)
            other = np.where(handle_rows, np.nan, r_right_full)
        else:
            r_left_full = np.where(valid, axis_x - left, np.nan)
            other = np.where(handle_rows, np.nan, r_left_full)

    rows = np.flatnonzero(valid)
    top, bottom = rows[0], rows[-1]
    upper = rows[rows <= top + 0.5 * (bottom - top)]
    r_max = float(np.nanmax(free[upper]))

    # The top outline across the rim is the far half of the rim ellipse:
    # T(x) = rim_y - b * sqrt(1 - ((x - axis) / R)^2), with b = R sin(el).
    # Near the rim ends the outline is steep, so a pixel of axis or radius
    # error biases the fit; search small offsets of both alongside it.
    width = mask.shape[1]
    cols_x = np.arange(width, dtype=np.float64)
    top_edge = np.argmax(mask, axis=0).astype(np.float64)
    any_col = mask.any(axis=0)
    best_fit = (np.inf, float(top), 0.0, r_max)
    for dx in np.arange(-2.0, 2.01, 0.5):
        for dr in np.arange(-2.0, 2.01, 0.5):
            radius = max(r_max + dr, 1.0)
            u = (cols_x - axis_x - dx) / radius
            has = any_col & (np.abs(u) < 0.95)
            if np.count_nonzero(has) < 5:
                continue
            s = np.sqrt(1.0 - u[has] ** 2)
            design = np.column_stack((np.ones_like(s), -s))
            coef, *_ = np.linalg.lstsq(design, top_edge[has], rcond=None)
            resid = float(np.mean((design @ coef - top_edge[has]) ** 2))
            if resid < best_fit[0]:
                best_fit = (resid, float(coef[0]), float(coef[1]), radius)
    _, rim_y, semi_minor, fit_radius = best_fit
    rim_y = float(np.clip(rim_y, top, bottom))
    sin_el = float(np.clip(semi_minor / fit_radius, 0.0, 1.0))
    elevation = float(np.degrees(np.arcsin(sin_el)))
    rim_r = free[int(round(rim_y))]
    diameter = 2.0 * float(rim_r if np.isfinite(rim_r) else r_max)

    ax = int(round(axis_x))
    band = slice(max(0, ax - max(2, int(0.1 * r_max))), ax + max(3, int(0.1 * r_max)))
    col_rows = np.flatnonzero(mask[:, band].any(axis=1))
    bottom_axis = float(col_rows[-1])

    foot_y = bottom_axis
    for _ in range(3):
        probe = int(np.clip(round(foot_y), top, bottom))
        r_foot = float(np.nan_to_num(free[probe], nan=0.0))
        foot_y = bottom_axis - r_foot * sin_el

    both = np.isfinite(free) & np.isfinite(other)
    both[: int(rim_y) + 1] = False
    both[int(foot_y) :] = False
    if np.count_nonzero(both) > 5:
        symmetry_error = float(
            np.median(np.abs(free[both] - other[both])) / max(diameter, 1.0)
        )
    else:
        symmetry_error = 0.0

    profile = None
    height_ratio = None
    if elevation <= SIDE_VIEW_MAX_ELEVATION and foot_y - rim_y > 4:
        ys = np.arange(int(np.ceil(rim_y)), int(np.floor(foot_y)) + 1)
        rs = free[ys]
        ok = np.isfinite(rs)
        if np.count_nonzero(ok) > 4:
            ts = (ys[ok] - rim_y) / (foot_y - rim_y)
            samples = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
            profile = np.interp(samples, ts, rs[ok]) / diameter
            cos_el = max(np.cos(np.radians(elevation)), 0.2)
            height_ratio = float((foot_y - rim_y) / cos_el / diameter)

    return ViewAnalysis(
        mask=mask,
        body=body,
        axis_x=axis_x,
        handle_side=handle_side,
        handle_area=handle_area,
        elevation_deg=elevation,
        rim_y=rim_y,
        foot_y=float(foot_y),
        diameter_px=diameter,
        free_radius=free,
        symmetry_error=symmetry_error,
        profile=profile,
        height_ratio=height_ratio,
    )


def _circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return float(4.0 * np.pi * area / max(perimeter * perimeter, 1e-9))


def is_surface_of_revolution(analyses: Sequence[ViewAnalysis]) -> bool:
    """Heuristic check that the views show one rotationally symmetric body."""
    side = [a for a in analyses if a.profile is not None]
    if not side:
        return False
    if np.median([a.symmetry_error for a in side]) > 0.035:
        return False
    for a in analyses:
        if a.elevation_deg >= TOP_VIEW_MIN_ELEVATION and _circularity(a.body) < 0.8:
            return False
    return True


def _detect_open_top(
    analyses: Sequence[ViewAnalysis], grays: Sequence[np.ndarray]
) -> bool:
    best = None
    for a, gray in zip(analyses, grays, strict=True):
        if a.elevation_deg < 20.0:
            continue
        if best is None or a.elevation_deg > best[0].elevation_deg:
            best = (a, gray)
    if best is None:
        return False
    a, gray = best
    radius = 0.5 * a.diameter_px
    semi_minor = max(radius * np.sin(np.radians(a.elevation_deg)), 2.0)
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    ell = ((xx - a.axis_x) / radius) ** 2 + ((yy - a.rim_y) / semi_minor) ** 2
    inner = (ell < 0.55**2) & a.mask
    rim = (ell > 0.92**2) & (ell < 1.0) & a.mask
    if np.count_nonzero(inner) < 20 or np.count_nonzero(rim) < 20:
        return False
    return float(np.median(gray[inner])) < float(np.median(gray[rim])) - 5.0


def _smooth_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -_smooth_min(-a, -b, k)


class _BodyField:
    """Analytic signed distance of the revolved (optionally hollow) body.

    The outer wall is the profile ``r = R(y)``; distances are first-order
    exact near the surface, which is all the iso-surface extraction needs,
    and avoid the stair-stepping a rasterised distance transform leaves.
    """

    def __init__(
        self,
        profile: np.ndarray,
        height: float,
        *,
        hollow: bool,
        wall: float,
        floor: float,
    ) -> None:
        # profile[0] is the rim (y = height), profile[-1] the foot (y = 0).
        self.ys = np.linspace(height, 0.0, len(profile))[::-1]
        self.rs = np.asarray(profile, dtype=np.float64)[::-1]
        slope = np.gradient(self.rs, self.ys)
        self.norm = np.sqrt(1.0 + slope * slope)
        self.height = height
        self.hollow = hollow
        self.wall = wall
        self.floor = floor
        self.round = 0.35 * wall

    def _outer(self, radial: np.ndarray, y: np.ndarray) -> np.ndarray:
        r_y = np.interp(y, self.ys, self.rs)
        return (radial - r_y) / np.interp(y, self.ys, self.norm)

    def sample_cavity(self, radial: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        if not self.hollow:
            return None
        return _smooth_max(
            self._outer(radial, y) + self.wall, self.floor - y, self.round
        )

    def sample(self, radial: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = self.round
        solid = _smooth_max(self._outer(radial, y), -y, k)
        solid = _smooth_max(solid, y - self.height, 0.5 * k)
        cavity = self.sample_cavity(radial, y)
        if cavity is None:
            return solid
        return _smooth_max(solid, -cavity, 0.5 * k)


def _local_thickness(inside: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Radius of the largest inscribed disc covering each pixel."""
    thickness = np.zeros(inside.shape, dtype=np.float64)
    r_max = int(np.ceil(depth.max()))
    for radius in range(1, r_max + 1):
        centers = (depth >= radius).astype(np.uint8)
        if not centers.any():
            break
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        covered = (cv2.dilate(centers, kernel) > 0) & inside
        thickness[covered] = radius
    thickness[inside & (thickness == 0)] = 1.0
    return thickness


class _HandleField:
    """Swept-tube signed distance built from the in-plane handle silhouette."""

    def __init__(
        self,
        view: ViewAnalysis,
        height: float,
        wall: float,
        depth_scale: float | None,
    ) -> None:
        mask = view.mask
        h, w = mask.shape
        side = view.handle_side
        xx = np.arange(w)[None, :]
        offset = side * (xx - view.axis_x)
        free = np.nan_to_num(view.free_radius, nan=0.0)[:, None]

        handle = mask & (offset > free + 1.0)
        handle = cv2.morphologyEx(
            handle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        handle = largest_component(handle * 255) > 0
        if not handle.any():
            raise ReconstructionError("Handle silhouette could not be isolated.")

        d_px = view.diameter_px
        hy = np.flatnonzero(handle.any(axis=1))
        depth0 = distance_transform_edt(handle)
        pad = int(np.ceil(depth0.max())) + 2
        rows = np.zeros(h, dtype=bool)
        rows[max(0, hy[0] - pad) : min(h, hy[-1] + pad + 1)] = True
        strip = mask & (offset > free - 2.0 * wall * d_px) & rows[:, None]
        support = handle | strip

        depth_in = distance_transform_edt(support)
        thickness = _local_thickness(support, depth_in)
        depth_out, (iy, ix) = distance_transform_edt(~support, return_indices=True)
        rho = np.where(support, thickness, thickness[iy, ix])
        self.ecc = np.where(support, thickness - depth_in, rho + depth_out) / d_px
        self.rho = rho / d_px

        medial = handle & (depth0 >= 0.8 * thickness)
        rho_handle = float(np.median(thickness[medial])) / d_px if medial.any() else 0.0
        if depth_scale is None or rho_handle <= 0:
            self.k = 1.0
        else:
            self.k = float(np.clip(0.5 * depth_scale / rho_handle, 0.6, 2.5))

        self.axis_x = view.axis_x
        self.side = side
        self.d_px = d_px
        self.rim_y = view.rim_y
        self.foot_y = view.foot_y
        self.height = height
        hx = np.nonzero(handle)[1]
        self.reach = float(np.max(side * (hx - view.axis_x))) / d_px
        ys_h = np.nonzero(handle)[0]
        self.y_range = (self._world_y(ys_h.max()), self._world_y(ys_h.min()))
        self.thickness = rho_handle

    def _world_y(self, py: float) -> float:
        return float(
            self.height * (self.foot_y - py) / max(self.foot_y - self.rim_y, 1.0)
        )

    def sample(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        px = self.axis_x + self.side * x * self.d_px
        py = self.foot_y - y / self.height * (self.foot_y - self.rim_y)
        coords = np.vstack((py, px))
        ecc = map_coordinates(self.ecc, coords, order=1, mode="nearest")
        rho = map_coordinates(self.rho, coords, order=1, mode="nearest")
        return np.sqrt(ecc * ecc + (z / self.k) ** 2) - rho


def _top_view_handle_depth(view: ViewAnalysis) -> float | None:
    """Out-of-plane handle width (in rim diameters) seen from above."""
    body = view.body
    ys, xs = np.nonzero(body)
    center = np.array([xs.mean(), ys.mean()])
    radius = float(np.sqrt(len(xs) / np.pi))
    my, mx = np.nonzero(view.mask)
    pts = np.column_stack((mx, my)).astype(np.float64)
    dist = np.linalg.norm(pts - center, axis=1)
    outside = pts[dist > 1.1 * radius]
    if len(outside) < 30:
        return None
    direction = outside.mean(axis=0) - center
    direction /= max(np.linalg.norm(direction), 1e-9)
    perp = np.array([-direction[1], direction[0]])
    along = (outside - center) @ direction
    across = (outside - center) @ perp
    bins = np.round(along).astype(int)
    widths = []
    for b in np.unique(bins):
        sel = across[bins == b]
        if len(sel) > 1:
            widths.append(float(sel.max() - sel.min() + 1.0))
    if not widths:
        return None
    return float(np.median(widths)) / (2.0 * radius)


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _build_mesh(
    body: _BodyField,
    handle: _HandleField | None,
    height: float,
    r_out: float,
    resolution: int,
    wall: float,
) -> tuple[np.ndarray, np.ndarray]:
    from .silhouette import marching_tetrahedra

    margin = 0.04 + 2.0 * wall
    x_hi = max(r_out, handle.reach if handle else r_out) + margin
    lo = np.array([-r_out - margin, -margin, -r_out - margin])
    hi = np.array([x_hi, height + margin, r_out + margin])
    voxel = float(np.max(hi - lo)) / (resolution - 1)
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    axes = [lo[i] + voxel * np.arange(shape[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    x, y, z = gx.ravel(), gy.ravel(), gz.ravel()
    radial = np.hypot(x, z)

    sdf = body.sample(radial, y)
    if handle is not None:
        sdf = _smooth_min(sdf, handle.sample(x, y, z), 0.6 * wall)
        cavity = body.sample_cavity(radial, y)
        if cavity is not None:
            # Cut after blending so the fillet never grows into the cavity.
            sdf = np.maximum(sdf, -cavity)

    field = np.clip(0.5 - sdf / (2.0 * voxel), 0.0, 1.0).reshape(shape)
    return marching_tetrahedra(field, lo, voxel, level=0.5)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def _base_rotation(az: float, el: float, roll: float = 0.0) -> np.ndarray:
    rot = Rotation.from_euler("zxy", [roll, el, az], degrees=True).as_matrix()
    return rot @ _Y_UP_TO_CAMERA


def _splat(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ui = np.round(uv[:, 0]).astype(np.int64)
    vi = np.round(uv[:, 1]).astype(np.int64)
    ok = (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
    img = np.zeros(shape, dtype=np.uint8)
    img[vi[ok], ui[ok]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return img > 0


def _fit_pose(
    points: np.ndarray,
    view: ViewAnalysis,
    intrinsics: CameraIntrinsics,
    *,
    grid: int = 128,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a camera pose so the model silhouette matches ``view.mask``.

    ``points`` must densely sample the model surface: the silhouette is
    rendered by splatting them onto a ``grid``-sized crop around the mask.
    """
    mask = view.mask
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 0.12 * max(x1 - x0, y1 - y0)
    span = max(x1 - x0, y1 - y0) + 2 * pad
    sx0, sy0 = int(round(x0 - pad)), int(round(y0 - pad))
    size = int(np.ceil(span)) + 1
    crop = np.zeros((size, size), dtype=np.uint8)
    src = mask[max(sy0, 0) :, max(sx0, 0) :].astype(np.uint8)
    dy, dx = max(-sy0, 0), max(-sx0, 0)
    hh = min(src.shape[0], size - dy)
    ww = min(src.shape[1], size - dx)
    crop[dy : dy + hh, dx : dx + ww] = src[:hh, :ww]
    gscale = grid / float(size)
    target = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA) > 0
    bw, bh = float(x1 - x0 + 1), float(y1 - y0 + 1)
    bcx, bcy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    # The exhaustive orientation search only needs a coarse silhouette: use a
    # smaller grid and a matching subsample of points (same splat density).
    coarse = max(32, grid * 5 // 8)
    cscale = coarse / float(size)
    coarse_target = cv2.resize(crop, (coarse, coarse), interpolation=cv2.INTER_AREA) > 0
    coarse_pts = points[:: max(1, round((grid / coarse) ** 2))]

    def ortho_fit(rot: np.ndarray, pts: np.ndarray = coarse_pts):
        p = pts @ rot.T
        px, py = p[:, 0], p[:, 1]
        s = 0.5 * (bw / max(np.ptp(px), 1e-9) + bh / max(np.ptp(py), 1e-9))
        a = bcx - s * 0.5 * (px.max() + px.min())
        b = bcy - s * 0.5 * (py.max() + py.min())
        uv = np.column_stack(((s * px + a - sx0) * cscale, (s * py + b - sy0) * cscale))
        return _iou(_splat(uv, (coarse, coarse)), coarse_target), s, a, b

    el_guess = view.elevation_deg
    els = sorted(
        {
            float(np.clip(sign * (el_guess + d), -90.0, 90.0))
            for sign in (1.0, -1.0)
            for d in (-15.0, -7.5, 0.0, 7.5, 15.0)
        }
    )

    def looks_up(rot: np.ndarray) -> bool:
        # Product shots look at the object from the side or above.
        return float(rot[2, 1]) > 0.1

    best_score, best_params = -1.0, (0.0, 0.0, 0.0)
    for az in range(0, 360, 10):
        for el in els:
            rot = _base_rotation(az, el)
            if looks_up(rot):
                continue
            score = ortho_fit(rot)[0]
            if score > best_score:
                best_score, best_params = score, (float(az), el, 0.0)
    az0, el0, _ = best_params
    for az in np.arange(az0 - 8.0, az0 + 9.0, 4.0):
        for el in np.arange(el0 - 6.0, el0 + 7.0, 3.0):
            el_c = float(np.clip(el, -90.0, 90.0))
            for roll in (-4.0, 0.0, 4.0):
                rot = _base_rotation(az, el_c, roll)
                if looks_up(rot):
                    continue
                score = ortho_fit(rot)[0]
                if score > best_score:
                    best_score, best_params = score, (float(az), el_c, roll)

    rot0 = _base_rotation(*best_params)
    _, s, a, b = ortho_fit(rot0, points)
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    tz = fx / s
    t0 = np.array([(a - cx) / s, (b - cy) / s, tz])

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot = Rotation.from_rotvec(params[:3]).as_matrix() @ rot0
        return rot, t0 + params[3:] * np.array([1.0, 1.0, tz])

    def objective(params: np.ndarray) -> float:
        rot, trans = unpack(params)
        if looks_up(rot):
            return 1.0
        cam = points @ rot.T + trans
        z = np.maximum(cam[:, 2], 1e-6)
        uv = np.column_stack(
            (
                (fx * cam[:, 0] / z + cx - sx0) * gscale,
                (fy * cam[:, 1] / z + cy - sy0) * gscale,
            )
        )
        return -_iou(_splat(uv, (grid, grid)), target)

    simplex = np.zeros((7, 6))
    for k, step in enumerate((0.05, 0.05, 0.05, 0.03, 0.03, 0.05)):
        simplex[k + 1, k] = step
    opt = minimize(
        objective,
        np.zeros(6),
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxiter": 400, "fatol": 1e-5},
    )
    rot, trans = unpack(opt.x)
    return rot, trans, float(-opt.fun)


def taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 8,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Volume-preserving Laplacian smoothing (Taubin lambda|mu)."""
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    counts = np.bincount(edges.ravel(), minlength=len(verts)).astype(np.float64)
    counts = np.maximum(counts, 1.0)[:, None]
    out = verts.copy()
    for _ in range(iterations):
        for factor in (lam, mu):
            summed = np.zeros_like(out)
            np.add.at(summed, edges[:, 0], out[edges[:, 1]])
            np.add.at(summed, edges[:, 1], out[edges[:, 0]])
            out = out + factor * (summed / counts - out)
    return out


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(verts)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _depth_buffer(
    cam: np.ndarray,
    faces: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Per-pixel nearest depth from every vertex plus every face centroid.

    Marching-tetrahedra faces project to about a pixel, so vertices and
    centroids cover the surface; a 3x3 min filter closes remaining gaps.
    """
    width, height = intrinsics.image_width, intrinsics.image_height
    pts = np.concatenate((cam, cam[faces].mean(axis=1)), axis=0)
    z = pts[:, 2]
    front = z > 1e-6
    u = np.round(intrinsics.fx * pts[front, 0] / z[front] + intrinsics.cx)
    v = np.round(intrinsics.fy * pts[front, 1] / z[front] + intrinsics.cy)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    zbuf = np.full(height * width, 1e9, dtype=np.float32)
    np.minimum.at(
        zbuf,
        v[ok].astype(np.int64) * width + u[ok].astype(np.int64),
        z[front][ok].astype(np.float32),
    )
    return cv2.erode(zbuf.reshape(height, width), np.ones((3, 3), np.uint8))


def _axisymmetric_prior(
    verts: np.ndarray,
    colors: np.ndarray,
    known: np.ndarray,
    bins: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour of known vertices sharing a (height, radius) ring."""
    radial = np.hypot(verts[:, 0], verts[:, 2])
    y = verts[:, 1]

    def bucket(values: np.ndarray) -> np.ndarray:
        lo, hi = float(values.min()), float(values.max())
        return np.clip(
            ((values - lo) / max(hi - lo, 1e-9) * bins).astype(int), 0, bins - 1
        )

    key = bucket(y) * bins + bucket(radial)
    sums = np.zeros((bins * bins, 3))
    counts = np.zeros(bins * bins)
    np.add.at(sums, key[known], colors[known])
    np.add.at(counts, key[known], 1.0)
    has = counts[key] > 0
    prior = np.zeros_like(colors)
    prior[has] = sums[key[has]] / counts[key[has], None]
    return prior, has


def project_vertex_colors(
    verts: np.ndarray,
    faces: np.ndarray,
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    axisymmetric: bool = False,
) -> np.ndarray:
    """Blend per-vertex colours from every view where the vertex is visible.

    With ``axisymmetric`` (model axis = world Y through the origin), surfaces
    no photo sees take the mean colour of their (height, radius) ring.
    """
    normals = _vertex_normals(verts, faces)
    accum = np.zeros((len(verts), 3))
    weight = np.zeros(len(verts))
    width, height = intrinsics.image_width, intrinsics.image_height
    tol = 0.006 * float(np.max(np.ptp(verts, axis=0)))

    for image, mask, rot, trans in zip(
        images, masks, rotations, translations, strict=True
    ):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        # Keep away from the silhouette edge, where pixels blend background.
        mask = (
            cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        cam = verts @ rot.T + trans
        z = cam[:, 2]
        safe_z = np.maximum(z, 1e-6)
        u = intrinsics.fx * cam[:, 0] / safe_z + intrinsics.cx
        v = intrinsics.fy * cam[:, 1] / safe_z + intrinsics.cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inside = (z > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

        zbuf = _depth_buffer(cam, faces, intrinsics)
        visible = np.zeros(len(verts), dtype=bool)
        idx = np.flatnonzero(inside)
        visible[idx] = (z[idx] <= zbuf[vi[idx], ui[idx]] + tol) & mask[vi[idx], ui[idx]]

        center = -rot.T @ trans
        view_dir = center[None, :] - verts
        view_dir /= np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-12)
        facing = np.clip(np.sum(normals * view_dir, axis=1), 0.0, 1.0)
        w = np.where(visible & (facing > 0.1), facing**4, 0.0)

        sel = w > 0
        coords = np.vstack((v[sel], u[sel]))
        color = np.column_stack(
            [
                map_coordinates(rgb[:, :, c], coords, order=1, mode="nearest")
                for c in range(3)
            ]
        )
        accum[sel] += w[sel, None] * color
        weight[sel] += w[sel]

    known = weight > 1e-6
    if not known.any():
        return np.full((len(verts), 3), 0.6)
    colors = np.zeros((len(verts), 3))
    colors[known] = accum[known] / weight[known, None]

    if axisymmetric:
        prior, has_prior = _axisymmetric_prior(verts, colors, known)
        w0 = 0.02
        blend = has_prior
        colors[blend] = (accum[blend] + w0 * prior[blend]) / (weight[blend, None] + w0)
        known = known | has_prior

    if not known.all():
        tree = cKDTree(verts[known])
        _, nearest = tree.query(verts[~known], k=8)
        colors[~known] = colors[known][nearest].mean(axis=1)

    # Light Laplacian smoothing to hide view seams.
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    counts = np.bincount(edges.ravel(), minlength=len(verts)).astype(np.float64)
    counts = np.maximum(counts, 1.0)[:, None]
    for _ in range(2):
        summed = np.zeros_like(colors)
        np.add.at(summed, edges[:, 0], colors[edges[:, 1]])
        np.add.at(summed, edges[:, 1], colors[edges[:, 0]])
        colors = 0.5 * colors + 0.5 * summed / counts
    return np.clip(colors, 0.0, 1.0)


def reconstruct_revolution(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    analyses: Sequence[ViewAnalysis] | None = None,
    resolution: int = 132,
    hollow: bool | None = None,
    wall_thickness: float = 0.045,
    floor_thickness: float = 0.07,
    colorize: bool = True,
) -> RevolutionModel:
    """Build a watertight surface-of-revolution model (plus handle) from views."""
    if analyses is None:
        analyses = [analyze_view(m) for m in masks]
    side = [a for a in analyses if a.profile is not None]
    if not side:
        raise ReconstructionError(
            "Revolution reconstruction needs at least one near side-on view."
        )

    # The closer to side-on, the less the foot/rim ellipses distort the profile.
    min_el = min(a.elevation_deg for a in side)
    side = [a for a in side if a.elevation_deg <= min_el + 8.0]
    stack = np.vstack([a.profile for a in side])
    if len(stack) >= 3:
        # Drop outlier views, then average: a per-sample median switches
        # between views and leaves slope kinks that show up as rings.
        dev = np.mean(np.abs(stack - np.median(stack, axis=0)), axis=1)
        keep = dev <= max(2.5 * float(np.median(dev)), 0.004)
        stack = stack[keep]
    profile = gaussian_filter1d(stack.mean(axis=0), sigma=4.0, mode="nearest")
    height = float(np.median([a.height_ratio for a in side]))

    grays = [
        cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        for img in images
    ]
    if hollow is None:
        hollow = _detect_open_top(analyses, grays)

    wall = wall_thickness
    body = _BodyField(
        profile,
        height,
        hollow=hollow,
        wall=wall,
        floor=floor_thickness * height,
    )

    handle_field = None
    handle_views = [a for a in side if a.handle_side != 0]
    if handle_views:
        hv = max(handle_views, key=lambda a: a.handle_area / a.diameter_px**2)
        top = [a for a in analyses if a.elevation_deg >= TOP_VIEW_MIN_ELEVATION]
        depth = _top_view_handle_depth(top[0]) if top else None
        try:
            handle_field = _HandleField(hv, height, wall, depth)
        except ReconstructionError:
            handle_field = None

    r_out = float(profile.max())
    verts, faces = _build_mesh(body, handle_field, height, r_out, resolution, wall)
    verts = taubin_smooth(verts, faces)

    stride = max(1, len(verts) // 30000)
    samples = verts[::stride]
    rotations = []
    translations = []
    for a in analyses:
        rot, trans, _ = _fit_pose(samples, a, intrinsics)
        rotations.append(rot)
        translations.append(trans)

    colors = None
    if colorize:
        colors = project_vertex_colors(
            verts,
            faces,
            images,
            [a.mask for a in analyses],
            rotations,
            translations,
            intrinsics,
            axisymmetric=True,
        )

    return RevolutionModel(
        vertices=verts,
        faces=faces,
        vertex_colors=colors,
        rotations=tuple(rotations),
        translations=tuple(translations),
        profile=profile,
        height_ratio=height,
        hollow=bool(hollow),
        has_handle=handle_field is not None,
    )
