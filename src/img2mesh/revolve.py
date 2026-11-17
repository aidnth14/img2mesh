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
from dataclasses import dataclass, replace

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
TOP_VIEW_MIN_ELEVATION = 60.0
_Y_UP_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True, slots=True)
class Appendage:
    """A handle, spout or similar part sticking out of the revolved body."""

    mask: np.ndarray
    side: int  # +1: image right of the axis, -1: left
    area: int
    reach_px: float  # how far it extends beyond the body radius


@dataclass(frozen=True, slots=True)
class ViewAnalysis:
    mask: np.ndarray
    body: np.ndarray
    axis_x: float
    appendages: tuple[Appendage, ...]
    elevation_deg: float
    rim_y: float
    foot_y: float
    diameter_px: float  # topmost circle (rim, lip, cap or knob)
    scale_px: float  # widest body diameter: the unit for profile and height
    radius: np.ndarray  # symmetric body half-width per row (NaN if none)
    mirror_iou: float
    top_aspect: float | None
    profile: np.ndarray | None
    height_ratio: float | None
    rim_radius_px: float = 0.0
    bottom_y: float = 0.0

    @property
    def handle_side(self) -> int:
        return self.appendages[0].side if self.appendages else 0

    @property
    def handle_area(self) -> int:
        return self.appendages[0].area if self.appendages else 0

    @property
    def symmetry_error(self) -> float:
        return 1.0 - self.mirror_iou

    @property
    def free_radius(self) -> np.ndarray:
        return self.radius


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
    appendage_count: int

    @property
    def has_handle(self) -> bool:
        return self.appendage_count > 0


def _body_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    k = max(5, int(round(0.22 * min(bw, bh))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    body = largest_component(opened) > 0
    return body if body.any() else mask.copy()


def _mirror_iou(body: np.ndarray, axis_x: float) -> float:
    ys, xs = np.nonzero(body)
    mirrored = np.round(2.0 * axis_x - xs).astype(np.int64)
    inside = (mirrored >= 0) & (mirrored < body.shape[1])
    inter = np.count_nonzero(body[ys[inside], mirrored[inside]])
    return float(inter / max(2 * len(xs) - inter, 1))


def _fit_axis(body: np.ndarray) -> tuple[float, float]:
    """Vertical mirror axis maximising the body's left/right overlap."""
    rows = np.flatnonzero(body.any(axis=1))
    lo, hi = rows[0], rows[-1]
    core = rows[(rows >= lo + 0.2 * (hi - lo)) & (rows <= hi - 0.2 * (hi - lo))]
    if len(core) == 0:
        core = rows
    sub = body[core]
    left = np.argmax(sub, axis=1)
    right = sub.shape[1] - 1 - np.argmax(sub[:, ::-1], axis=1)
    guess = float(np.median(0.5 * (left + right)))
    width = float(np.median(right - left)) + 1.0
    best = (-1.0, guess)
    for offset in np.arange(-0.05 * width, 0.05 * width + 0.25, 0.5):
        score = _mirror_iou(body, guess + offset)
        if score > best[0]:
            best = (score, guess + offset)
    return best[1], best[0]


def _symmetric_radius(mask: np.ndarray, axis_x: float) -> np.ndarray:
    """Per-row half-width of the solid run through the axis (shorter side).

    Taking the shorter side ignores a handle or spout attached on one side;
    using only the run through the axis ignores parts beyond a gap (a
    handle seen through its hole).
    """
    ax = int(np.clip(round(axis_x), 0, mask.shape[1] - 1))
    on_axis = mask[:, ax]
    left_part = mask[:, : ax + 1][:, ::-1]
    right_part = mask[:, ax:]
    len_l = np.where(
        left_part.all(axis=1), left_part.shape[1], np.argmin(left_part, axis=1)
    )
    len_r = np.where(
        right_part.all(axis=1), right_part.shape[1], np.argmin(right_part, axis=1)
    )
    r_left = axis_x - (ax - len_l + 0.5)
    r_right = (ax + len_r - 0.5) - axis_x
    radius = np.minimum(r_left, r_right).astype(np.float64)
    radius[~on_axis] = np.nan
    return radius


def _find_appendages(
    mask: np.ndarray, axis_x: float, radius: np.ndarray
) -> tuple[Appendage, ...]:
    h, w = mask.shape
    xx = np.arange(w)[None, :]
    reach = np.abs(xx - axis_x)
    r_row = np.where(np.isfinite(radius), radius, -1.0)[:, None]
    outside = mask & (reach > r_row + 1.5)
    outside = cv2.morphologyEx(
        outside.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    min_area = max(30, int(0.004 * np.count_nonzero(mask)))
    found = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        if distance_transform_edt(comp).max() < 2.0:
            continue  # a sliver along the body edge, not a part
        cy, cx = np.nonzero(comp)
        side = 1 if float(np.mean(cx)) > axis_x else -1
        base = np.nanmedian(np.where(np.isfinite(radius[cy]), radius[cy], np.nan))
        base = 0.0 if not np.isfinite(base) else float(base)
        found.append(
            Appendage(
                mask=comp,
                side=side,
                area=area,
                reach_px=float(np.max(np.abs(cx - axis_x))) - base,
            )
        )
    found.sort(key=lambda app: -app.area)
    return tuple(found)


def _lip_ellipse(radius: np.ndarray, top_axis: float) -> tuple[float, float, float]:
    """Topmost circle's image ellipse: (centre row, semi-minor, radius).

    Walking down from the top, the half-width grows along the far half of the
    topmost circle (a rim, lip, cap or knob) until that circle's widest row,
    after which it shrinks (a neck) or stops growing (a wall or knob side).
    A well-resolved ellipse is fitted as ``y = yc - b * sqrt(1 - (r / R)^2)``
    over the rising rows; a thin one (near side-on) is measured directly as
    the drop from the top of the outline to the widest row.
    """
    rows = np.flatnonzero(np.isfinite(radius))
    r = radius[rows]
    peak, stall, peak_row = -np.inf, 0, rows[0]
    for row, value in zip(rows, r, strict=True):
        if value > peak + 0.5:
            stall, peak_row = 0, row
        else:
            stall += 1
        peak = max(peak, value)
        shrinking = value < peak - max(1.5, 0.02 * peak)
        if shrinking or stall >= max(4, int(np.ceil(np.sqrt(peak))) + 1):
            break
    # Only rows up to the last real widening belong to the topmost circle;
    # a slightly wider row further down a near-cylindrical wall does not.
    segment = rows <= peak_row + 2
    rim_r = max(float(r[segment].max()), 1.0)
    first_row = int(rows[segment][np.argmax(r[segment] >= rim_r - 0.5)])
    sag = float(max(first_row - top_axis, 0.0))
    rising = (rows <= first_row) & (r > 0.05 * rim_r) & (r < 0.97 * rim_r)
    if np.count_nonzero(rising) >= 6:
        s = np.sqrt(1.0 - (r[rising] / rim_r) ** 2)
        design = np.column_stack((np.ones_like(s), -s))
        (yc, b), *_ = np.linalg.lstsq(design, rows[rising].astype(float), rcond=None)
        if b > 0:
            return float(yc), float(min(b, rim_r)), rim_r
    return float(first_row), min(sag, rim_r), rim_r


def _top_aspect(body: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(
        body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (_, _), (a, b), _ = cv2.fitEllipse(contour)
    return float(min(a, b) / max(a, b, 1e-9))


def _measure_profile(
    radius: np.ndarray,
    rim_y: float,
    rim_r: float,
    bottom_axis: float,
    elevation: float,
) -> tuple[float, float, np.ndarray | None, float | None]:
    """Foot row, body scale, normalised profile and height for an elevation.

    The outline's lowest point is the near edge of the foot circle, so the
    foot's own row sits ``r_foot * sin(el)`` above it; heights shrink by
    ``cos(el)``. Profiles are only taken from near side-on views.
    """
    sin_el = float(np.sin(np.radians(elevation)))
    rows = np.flatnonzero(np.isfinite(radius))
    top, bottom = rows[0], rows[-1]
    foot_y = bottom_axis
    for _ in range(3):
        probe = int(np.clip(round(foot_y), top, bottom))
        r_foot = float(np.nan_to_num(radius[probe], nan=0.0))
        foot_y = bottom_axis - r_foot * sin_el

    body_rows = np.arange(int(np.ceil(rim_y)), int(np.floor(foot_y)) + 1)
    body_r = radius[body_rows] if len(body_rows) else np.array([])
    widest = float(np.nanmax(body_r)) if np.isfinite(body_r).any() else rim_r
    scale = 2.0 * max(widest, rim_r)

    profile = None
    height_ratio = None
    if elevation <= SIDE_VIEW_MAX_ELEVATION and foot_y - rim_y > 4:
        ok = np.isfinite(body_r)
        if np.count_nonzero(ok) > 4:
            ts = (body_rows[ok] - rim_y) / (foot_y - rim_y)
            samples = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
            profile = np.interp(samples, ts, body_r[ok]) / scale
            cos_el = max(np.cos(np.radians(elevation)), 0.2)
            height_ratio = float((foot_y - rim_y) / cos_el / scale)
    return float(foot_y), scale, profile, height_ratio


def with_elevation(view: ViewAnalysis, elevation: float) -> ViewAnalysis:
    """Re-measure a view's foot, profile and height for a known elevation."""
    foot_y, scale, profile, height_ratio = _measure_profile(
        view.radius, view.rim_y, view.rim_radius_px, view.bottom_y, elevation
    )
    return replace(
        view,
        elevation_deg=float(elevation),
        foot_y=foot_y,
        scale_px=scale,
        profile=profile,
        height_ratio=height_ratio,
    )


def analyze_view(mask: np.ndarray) -> ViewAnalysis:
    """Measure axis, appendages, elevation and outer profile of one view."""
    mask = np.asarray(mask) > 0
    body = _body_mask(mask)
    axis_x, mirror = _fit_axis(body)

    radius = _symmetric_radius(mask, axis_x)
    appendages = _find_appendages(mask, axis_x, radius)
    sides = {app.side for app in appendages}
    if sides == {-1, 1}:
        # Parts attached on both sides of the same rows inflate both halves;
        # only there, fall back to the part-free (opened) body.
        rows_by_side = {
            side: np.any(
                [app.mask.any(axis=1) for app in appendages if app.side == side],
                axis=0,
            )
            for side in (-1, 1)
        }
        both = rows_by_side[-1] & rows_by_side[1]
        body_radius = _symmetric_radius(body, axis_x)
        width = float(np.nanmax(radius)) if np.isfinite(radius).any() else 1.0
        tol = max(3.0, 0.04 * width)
        inflated = both & np.isfinite(body_radius) & (radius > body_radius + tol)
        if inflated.any():
            radius = np.where(inflated, body_radius, radius)
            appendages = _find_appendages(mask, axis_x, radius)

    ax = int(round(axis_x))
    band = slice(max(0, ax - 2), ax + 3)
    col_rows = np.flatnonzero(mask[:, band].any(axis=1))
    top_axis, bottom_axis = float(col_rows[0]), float(col_rows[-1])

    rim_y, semi_minor, rim_r = _lip_ellipse(radius, top_axis)
    sin_el = float(np.clip(semi_minor / rim_r, 0.0, 1.0))
    elevation = float(np.degrees(np.arcsin(sin_el)))
    diameter = 2.0 * rim_r

    foot_y, scale, profile, height_ratio = _measure_profile(
        radius, rim_y, rim_r, bottom_axis, elevation
    )

    return ViewAnalysis(
        mask=mask,
        body=body,
        axis_x=axis_x,
        appendages=appendages,
        elevation_deg=elevation,
        rim_y=rim_y,
        foot_y=float(foot_y),
        diameter_px=diameter,
        scale_px=scale,
        radius=radius,
        mirror_iou=mirror,
        top_aspect=_top_aspect(body) if elevation >= 60.0 else None,
        rim_radius_px=rim_r,
        bottom_y=bottom_axis,
        profile=profile,
        height_ratio=height_ratio,
    )


def is_surface_of_revolution(analyses: Sequence[ViewAnalysis]) -> bool:
    """Heuristic check that the views show one rotationally symmetric body.

    Requires a side-on view; left/right mirror symmetry of the (appendage
    free) body in every view; side views that agree on height/diameter
    (a box's width changes with azimuth, a cup's does not); and a round
    outline from any top-down view.
    """
    side = [a for a in analyses if a.profile is not None]
    if not side:
        return False
    mirrors = [a.mirror_iou for a in analyses]
    if np.median(mirrors) < 0.93 or min(mirrors) < 0.88:
        return False
    heights = [a.height_ratio for a in side]
    if len(heights) >= 2:
        spread = (max(heights) - min(heights)) / max(float(np.median(heights)), 1e-9)
        if spread > 0.15:
            return False
    for a in analyses:
        if a.elevation_deg >= TOP_VIEW_MIN_ELEVATION:
            if a.top_aspect is None or a.top_aspect < 0.85:
                return False
    return True


def _detect_open_top(
    analyses: Sequence[ViewAnalysis], grays: Sequence[np.ndarray]
) -> bool:
    """Is the top open (a cup, bowl or vase) rather than closed (a lid, cap)?

    Seen from about 30 degrees up, an open vessel shows its shaded far inner
    wall and floor inside the rim ellipse, darker than the lit rim band; a
    closed top faces the light and is as bright or brighter. Top-down views
    are poor judges (the band there includes grazing outer wall), so views
    between 12 and 45 degrees are averaged, weighted towards 30 degrees.
    """
    candidates = [
        (a, gray)
        for a, gray in zip(analyses, grays, strict=True)
        if 12.0 <= a.elevation_deg <= 45.0
    ]
    if not candidates:
        elevated = [
            (a, gray)
            for a, gray in zip(analyses, grays, strict=True)
            if a.elevation_deg >= 12.0
        ]
        if not elevated:
            return False
        candidates = [min(elevated, key=lambda c: abs(c[0].elevation_deg - 30.0))]
    margins, weights = [], []
    for a, gray in candidates:
        radius = 0.5 * a.diameter_px
        semi_minor = max(radius * np.sin(np.radians(a.elevation_deg)), 2.0)
        yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
        ell = ((xx - a.axis_x) / radius) ** 2 + ((yy - a.rim_y) / semi_minor) ** 2
        inner = (ell < 0.55**2) & a.mask
        rim = (ell > 0.92**2) & (ell < 1.0) & a.mask
        if np.count_nonzero(inner) < 20 or np.count_nonzero(rim) < 20:
            continue
        margins.append(float(np.median(gray[rim])) - float(np.median(gray[inner])))
        weights.append(1.0 / (1.0 + abs(a.elevation_deg - 30.0) / 10.0))
    if not margins:
        return False
    return float(np.average(margins, weights=weights)) > 5.0


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
    """Swept-tube signed distance built from one appendage's silhouette.

    World +X is image right in the view the appendage was measured in, so
    parts on both sides of the body share one frame.
    """

    def __init__(
        self,
        view: ViewAnalysis,
        appendage: Appendage,
        height: float,
        wall: float,
        depth_scale: float | None,
    ) -> None:
        mask = view.mask
        h, w = mask.shape
        side = appendage.side
        xx = np.arange(w)[None, :]
        offset = side * (xx - view.axis_x)
        free = np.nan_to_num(view.radius, nan=0.0)[:, None]

        part = appendage.mask
        if not part.any():
            raise ReconstructionError("Appendage silhouette is empty.")

        d_px = view.scale_px
        hy = np.flatnonzero(part.any(axis=1))
        depth0 = distance_transform_edt(part)
        pad = int(np.ceil(depth0.max())) + 2
        rows = np.zeros(h, dtype=bool)
        rows[max(0, hy[0] - pad) : min(h, hy[-1] + pad + 1)] = True
        strip = mask & (offset > free - 2.0 * wall * d_px) & rows[:, None]
        strip &= offset > 0
        support = part | strip

        depth_in = distance_transform_edt(support)
        thickness = _local_thickness(support, depth_in)
        depth_out, (iy, ix) = distance_transform_edt(~support, return_indices=True)
        rho = np.where(support, thickness, thickness[iy, ix])
        self.ecc = np.where(support, thickness - depth_in, rho + depth_out) / d_px
        self.rho = rho / d_px

        medial = part & (depth0 >= 0.8 * thickness)
        rho_part = float(np.median(thickness[medial])) / d_px if medial.any() else 0.0
        if depth_scale is None or rho_part <= 0:
            self.k = 1.0
        else:
            self.k = float(np.clip(0.5 * depth_scale / rho_part, 0.6, 2.5))

        self.axis_x = view.axis_x
        self.side = side
        self.d_px = d_px
        self.rim_y = view.rim_y
        self.foot_y = view.foot_y
        self.height = height
        px_x = np.nonzero(part)[1]
        self.extent = (
            float(np.min(px_x - view.axis_x)) / d_px,
            float(np.max(px_x - view.axis_x)) / d_px,
        )
        self.reach = appendage.reach_px / d_px
        self.thickness = rho_part

    def sample(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        px = self.axis_x + x * self.d_px
        py = self.foot_y - y / self.height * (self.foot_y - self.rim_y)
        coords = np.vstack((py, px))
        ecc = map_coordinates(self.ecc, coords, order=1, mode="nearest")
        rho = map_coordinates(self.rho, coords, order=1, mode="nearest")
        return np.sqrt(ecc * ecc + (z / self.k) ** 2) - rho


def _top_view_widths(view: ViewAnalysis) -> list[tuple[float, float]]:
    """(reach, out-of-plane width) per part seen from above, in diameters."""
    body = view.body
    ys, xs = np.nonzero(body)
    center = np.array([xs.mean(), ys.mean()])
    radius = float(np.sqrt(len(xs) / np.pi))
    yy, xx = np.nonzero(view.mask)
    dist = np.hypot(xx - center[0], yy - center[1])
    outside = np.zeros(view.mask.shape, dtype=np.uint8)
    sel = dist > 1.1 * radius
    outside[yy[sel], xx[sel]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    parts = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 30:
            continue
        cy, cx = np.nonzero(labels == label)
        pts = np.column_stack((cx, cy)).astype(np.float64) - center
        direction = pts.mean(axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        perp = np.array([-direction[1], direction[0]])
        along = pts @ direction
        across = pts @ perp
        bins = np.round(along).astype(int)
        widths = [
            float(across[bins == b].max() - across[bins == b].min() + 1.0)
            for b in np.unique(bins)
            if np.count_nonzero(bins == b) > 1
        ]
        if widths:
            reach = (float(along.max()) - radius) / (2.0 * radius)
            parts.append((reach, float(np.median(widths)) / (2.0 * radius)))
    return parts


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


class _FieldGrid:
    """Sampling grid for the model's signed distance, body terms cached."""

    def __init__(
        self,
        body: _BodyField,
        parts: Sequence[_HandleField],
        height: float,
        r_out: float,
        resolution: int,
        wall: float,
    ) -> None:
        margin = 0.04 + 2.0 * wall
        x_lo = min([-r_out] + [p.extent[0] for p in parts]) - margin
        x_hi = max([r_out] + [p.extent[1] for p in parts]) + margin
        self.lo = np.array([x_lo, -margin, -r_out - margin])
        hi = np.array([x_hi, height + margin, r_out + margin])
        self.voxel = float(np.max(hi - self.lo)) / (resolution - 1)
        self.shape = np.ceil((hi - self.lo) / self.voxel).astype(int) + 1
        axes = [self.lo[i] + self.voxel * np.arange(self.shape[i]) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        self.x, self.y, self.z = gx.ravel(), gy.ravel(), gz.ravel()
        radial = np.hypot(self.x, self.z)
        self.body_sdf = body.sample(radial, self.y)
        self.cavity = body.sample_cavity(radial, self.y)
        self.wall = wall

    def mesh(self, parts: Sequence[_HandleField]) -> tuple[np.ndarray, np.ndarray]:
        from .silhouette import marching_tetrahedra

        sdf = self.body_sdf
        for part in parts:
            sdf = _smooth_min(sdf, part.sample(self.x, self.y, self.z), 0.6 * self.wall)
        if parts and self.cavity is not None:
            # Cut after blending so the fillet never grows into the cavity.
            sdf = np.maximum(sdf, -self.cavity)
        field = np.clip(0.5 - sdf / (2.0 * self.voxel), 0.0, 1.0).reshape(self.shape)
        return marching_tetrahedra(field, self.lo, self.voxel, level=0.5)


def _build_mesh(
    body: _BodyField,
    parts: Sequence[_HandleField],
    height: float,
    r_out: float,
    resolution: int,
    wall: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _FieldGrid(body, parts, height, r_out, resolution, wall).mesh(parts)


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


class _ViewTarget:
    """A view's mask cropped to a square ``grid`` for fast overlap scoring."""

    def __init__(self, mask: np.ndarray, grid: int = 128) -> None:
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = 0.12 * max(x1 - x0, y1 - y0)
        span = max(x1 - x0, y1 - y0) + 2 * pad
        self.sx0, self.sy0 = int(round(x0 - pad)), int(round(y0 - pad))
        self.size = int(np.ceil(span)) + 1
        crop = np.zeros((self.size, self.size), dtype=np.uint8)
        src = mask[max(self.sy0, 0) :, max(self.sx0, 0) :].astype(np.uint8)
        dy, dx = max(-self.sy0, 0), max(-self.sx0, 0)
        hh = min(src.shape[0], self.size - dy)
        ww = min(src.shape[1], self.size - dx)
        crop[dy : dy + hh, dx : dx + ww] = src[:hh, :ww]
        self.crop = crop
        self.grid = grid
        self.gscale = grid / float(self.size)
        self.target = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA) > 0
        self.bbox = (float(x0), float(x1), float(y0), float(y1))

    def iou(
        self,
        points: np.ndarray,
        rot: np.ndarray,
        trans: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> float:
        cam = points @ rot.T + trans
        z = np.maximum(cam[:, 2], 1e-6)
        uv = np.column_stack(
            (
                (intrinsics.fx * cam[:, 0] / z + intrinsics.cx - self.sx0)
                * self.gscale,
                (intrinsics.fy * cam[:, 1] / z + intrinsics.cy - self.sy0)
                * self.gscale,
            )
        )
        return _iou(_splat(uv, (self.grid, self.grid)), self.target)


def _looks_up(rot: np.ndarray) -> bool:
    # Product shots look at the object from the side or above.
    return float(rot[2, 1]) > 0.1


def camera_elevation(rot: np.ndarray) -> float:
    """Elevation (degrees above the horizon) of a fitted model-frame camera."""
    forward = rot[2]  # camera z axis in model coordinates
    return float(np.degrees(np.arcsin(np.clip(-forward[1], -1.0, 1.0))))


def _fit_pose(
    points: np.ndarray,
    view: ViewAnalysis,
    intrinsics: CameraIntrinsics,
    *,
    grid: int = 128,
    init: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a camera pose so the model silhouette matches ``view.mask``.

    ``points`` must densely sample the model surface: the silhouette is
    rendered by splatting them onto a ``grid``-sized crop around the mask.
    With ``init`` the orientation search is skipped and only refined.
    """
    vt = _ViewTarget(view.mask, grid)
    sx0, sy0, size, crop = vt.sx0, vt.sy0, vt.size, vt.crop
    gscale, target = vt.gscale, vt.target
    x0, x1, y0, y1 = vt.bbox
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

    looks_up = _looks_up
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

    if init is not None:
        rot0, t0 = init[0], np.asarray(init[1], dtype=np.float64).reshape(3)
        tz = float(t0[2])
    else:
        rot0, t0, tz = None, None, 0.0

    best_score, best_params = -1.0, (0.0, 0.0, 0.0)
    for az in range(0, 360, 10) if init is None else ():
        for el in els:
            rot = _base_rotation(az, el)
            if looks_up(rot):
                continue
            score = ortho_fit(rot)[0]
            if score > best_score:
                best_score, best_params = score, (float(az), el, 0.0)
    az0, el0, _ = best_params
    for az in np.arange(az0 - 8.0, az0 + 9.0, 4.0) if init is None else ():
        for el in np.arange(el0 - 6.0, el0 + 7.0, 3.0):
            el_c = float(np.clip(el, -90.0, 90.0))
            for roll in (-4.0, 0.0, 4.0):
                rot = _base_rotation(az, el_c, roll)
                if looks_up(rot):
                    continue
                score = ortho_fit(rot)[0]
                if score > best_score:
                    best_score, best_params = score, (float(az), el_c, roll)

    if init is None:
        rot0 = _base_rotation(*best_params)
        _, s, a, b = ortho_fit(rot0, points)
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


def _combine_profiles(
    side: Sequence[ViewAnalysis],
) -> tuple[np.ndarray, float]:
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
    return profile, height


def _build_parts(
    analyses: Sequence[ViewAnalysis],
    side: Sequence[ViewAnalysis],
    height: float,
    wall: float,
) -> list[_HandleField]:
    parts: list[_HandleField] = []
    part_views = [a for a in side if a.appendages]
    if not part_views:
        return parts
    pv = max(
        part_views,
        key=lambda a: sum(app.area for app in a.appendages) / a.scale_px**2,
    )
    largest = pv.appendages[0].area
    for app in pv.appendages:
        if app.area < 0.2 * largest:
            continue
        try:
            parts.append(_HandleField(pv, app, height, wall, None))
        except ReconstructionError:
            continue
    return parts


_DEPTH_SCALES = (0.6, 0.8, 1.0, 1.3, 1.7, 2.2)


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
    """Build a watertight surface-of-revolution model (plus parts) from views.

    1. Measure the profile from near side-on views and build a coarse model.
    2. Fit every camera against it and read each view's true elevation from
       its fitted camera; re-measure the profiles with those and rebuild.
    3. Pick each handle/spout's out-of-plane thickness by silhouette overlap
       over all views.
    4. Build the final mesh, refine the cameras and project colours.
    """
    if analyses is None:
        analyses = [analyze_view(m) for m in masks]
    analyses = list(analyses)
    side = [a for a in analyses if a.profile is not None]
    if not side:
        raise ReconstructionError(
            "Revolution reconstruction needs at least one near side-on view."
        )

    grays = [
        cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        for img in images
    ]
    if hollow is None:
        hollow = _detect_open_top(analyses, grays)

    coarse_res = max(48, resolution // 2)

    def assemble(views: Sequence[ViewAnalysis]):
        side_views = [a for a in views if a.profile is not None]
        profile, height = _combine_profiles(side_views)
        # Wall thickness scales with the widest part, not the (maybe narrow) lip.
        wall = wall_thickness * max(1.0, 2.0 * float(profile.max()))
        body = _BodyField(
            profile, height, hollow=hollow, wall=wall, floor=floor_thickness * height
        )
        parts = _build_parts(views, side_views, height, wall)
        return profile, height, wall, body, parts

    def fit_all(points, inits=None):
        poses = []
        for idx, a in enumerate(analyses):
            init = None if inits is None else inits[idx]
            rot, trans, _ = _fit_pose(points, a, intrinsics, init=init)
            poses.append((rot, trans))
        return poses

    # 1-2. Coarse model, cameras, elevation feedback.
    profile, height, wall, body, parts = assemble(analyses)
    r_out = float(profile.max())
    grid = _FieldGrid(body, parts, height, r_out, coarse_res, wall)
    coarse_v, _ = grid.mesh(parts)
    poses = fit_all(coarse_v)

    fitted = [
        with_elevation(a, max(camera_elevation(rot), 0.0))
        for a, (rot, _) in zip(analyses, poses, strict=True)
    ]
    if any(a.profile is not None for a in fitted):
        moved = max(
            abs(f.elevation_deg - a.elevation_deg)
            for f, a in zip(fitted, analyses, strict=True)
            if a.profile is not None or f.profile is not None
        )
        if moved > 2.0:
            analyses_fb = fitted
            profile, height, wall, body, parts = assemble(analyses_fb)
            r_out = float(profile.max())
            grid = _FieldGrid(body, parts, height, r_out, coarse_res, wall)
            coarse_v, _ = grid.mesh(parts)
            poses = fit_all(coarse_v, poses)

    # 3. Out-of-plane thickness of each part, by overlap over all views.
    if parts:
        targets = [_ViewTarget(a.mask) for a in analyses]

        def overlap(points):
            return float(
                np.mean(
                    [
                        t.iou(points, rot, trans, intrinsics)
                        for t, (rot, trans) in zip(targets, poses, strict=True)
                    ]
                )
            )

        for part in parts:
            scores = []
            for k in _DEPTH_SCALES:
                part.k = k
                scores.append(overlap(grid.mesh(parts)[0]))
            part.k = _DEPTH_SCALES[int(np.argmax(scores))]

    # 4. Final mesh, refined cameras, colours.
    verts, faces = _build_mesh(body, parts, height, r_out, resolution, wall)
    verts = taubin_smooth(verts, faces)
    stride = max(1, len(verts) // 30000)
    poses = fit_all(verts[::stride], poses)
    rotations = [rot for rot, _ in poses]
    translations = [trans for _, trans in poses]

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
        appendage_count=len(parts),
    )
