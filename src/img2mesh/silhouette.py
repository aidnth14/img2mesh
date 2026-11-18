from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .config import ConfigurationError, InputError, ReconstructionError
from .geometry import CameraIntrinsics
from .image import clean_mask, largest_component
from .mesh import MeshResult

_MT_TETS = np.array(
    [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ],
    dtype=np.int64,
)

_MT_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True, slots=True)
class VisualHullResult:
    mesh: MeshResult
    camera_rotations: np.ndarray
    camera_translations: np.ndarray
    intrinsics: CameraIntrinsics
    masks: tuple[np.ndarray, ...]
    per_view_iou: tuple[float, ...]
    per_view_dice: tuple[float, ...]
    method: str = "visual_hull"
    # Filled by shape_prior="best": held-out validation and a verdict.
    holdout_iou: float | None = None
    quality: str | None = None
    advice: tuple[str, ...] = ()
    candidates: tuple[tuple[str, float], ...] = ()

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.per_view_iou))

    @property
    def min_iou(self) -> float:
        return float(np.min(self.per_view_iou))

    @property
    def mean_dice(self) -> float:
        return float(np.mean(self.per_view_dice))


def extract_foreground_mask(
    image: Image.Image,
    *,
    color_distance_threshold: float = 24.0,
    min_hole_area: int = 150,
) -> np.ndarray:
    """Segment the foreground subject using border background estimation."""
    if not isinstance(image, Image.Image):
        raise InputError("extract_foreground_mask expects a PIL image.")

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    border = np.vstack([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg_median = np.median(border, axis=0)
    color_dist = np.linalg.norm(lab - bg_median[None, None, :], axis=2)

    is_fg = (color_dist > color_distance_threshold) & (
        (hsv[:, :, 1] > 26) | (hsv[:, :, 2] < 205)
    )
    cleaned = clean_mask(
        is_fg.astype(np.uint8) * 255,
        kernel_size=3,
        opening_iterations=1,
        closing_iterations=1,
    )
    mask = largest_component(cleaned)

    if min_hole_area > 0 and np.any(mask):
        inverse = 255 - mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverse, connectivity=4
        )
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < min_hole_area:
                mask[labels == label] = 255

    return mask


def _horizontal_thickness_map(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    thickness = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        row = mask[y]
        if not np.any(row):
            continue
        padded = np.concatenate(([False], row, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end in zip(starts, ends, strict=True):
            radius = 0.5 * (end - start)
            center = 0.5 * (start + end - 1)
            xs = np.arange(start, end, dtype=np.float32)
            thickness[y, start:end] = np.sqrt(
                np.maximum(radius * radius - (xs - center) ** 2, 0.0)
            )
    return thickness


def _inflate_anchor_thickness(
    mask: np.ndarray,
    bbox_w: float,
    bbox_h: float,
) -> np.ndarray:
    kernel_size = max(5, int(round(0.22 * min(bbox_w, bbox_h))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    body = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0
    body_thickness = _horizontal_thickness_map(body)
    appendage_thickness = 1.2 * distance_transform_edt(mask)
    return np.where(body, body_thickness, appendage_thickness)


def _pose_from_euler(
    az_deg: float,
    el_deg: float,
    roll_deg: float,
    tx: float,
    ty: float,
    tz: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_euler(
        "zxy", [roll_deg, el_deg, az_deg], degrees=True
    ).as_matrix()
    translation = np.array([tx, ty, tz], dtype=np.float64)
    return rotation, translation


def _make_voxel_grid(resolution: int, bound: float) -> np.ndarray:
    lin = np.linspace(-bound, bound, resolution, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))


def _carve_voxels(
    points: np.ndarray,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    occupied = np.ones(len(points), dtype=bool)
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    width, height = intrinsics.image_width, intrinsics.image_height

    for (rotation, translation), mask in zip(poses, masks, strict=True):
        if not np.any(occupied):
            break
        cam = points[occupied] @ rotation.T + translation.reshape(1, 3)
        z = cam[:, 2]
        valid = z > 0.1
        safe_z = np.maximum(z, 1e-4)
        u = np.round(fx * (cam[:, 0] / safe_z) + cx).astype(np.int32)
        v = np.round(fy * (cam[:, 1] / safe_z) + cy).astype(np.int32)
        inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        keep = np.zeros(len(u), dtype=bool)
        keep[inside] = mask[v[inside], u[inside]]
        occ_indices = np.flatnonzero(occupied)
        occupied[occ_indices[~keep]] = False

    return occupied


def _evaluate_pose_silhouette(
    hull_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    mask: np.ndarray,
    dist_outside: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    downsample: int = 128,
    base_tz: float | None = None,
    roll_deg: float = 0.0,
) -> float:
    width, height = intrinsics.image_width, intrinsics.image_height
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

    cam = hull_points @ rotation.T + translation.reshape(1, 3)
    z = np.maximum(cam[:, 2], 1e-4)
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy

    ds_w = downsample
    ds_h = max(16, round(downsample * height / width))
    sx = ds_w / float(width)
    sy = ds_h / float(height)
    ui = np.round(u * sx).astype(np.int32)
    vi = np.round(v * sy).astype(np.int32)
    valid = (cam[:, 2] > 0.1) & (ui >= 0) & (ui < ds_w) & (vi >= 0) & (vi < ds_h)

    proj = np.zeros((ds_h, ds_w), dtype=np.uint8)
    proj[vi[valid], ui[valid]] = 255
    proj_bool = cv2.dilate(proj, np.ones((3, 3), np.uint8)) > 0

    mask_ds = (
        cv2.resize(
            mask.astype(np.uint8),
            (ds_w, ds_h),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    inter = np.count_nonzero(proj_bool & mask_ds)
    union = np.count_nonzero(proj_bool | mask_ds)
    iou = inter / max(union, 1)

    u_clip = np.clip(np.round(u).astype(np.int32), 0, width - 1)
    v_clip = np.clip(np.round(v).astype(np.int32), 0, height - 1)
    outside_penalty = float(np.mean(dist_outside[v_clip, u_clip]))
    scale_penalty = (
        0.0 if base_tz is None else 0.15 * ((translation[2] - base_tz) / base_tz) ** 2
    )
    roll_penalty = 0.0002 * (roll_deg**2)
    return -iou + 0.03 * outside_penalty + scale_penalty + roll_penalty


def _refine_pose_bounded(
    hull_points: np.ndarray,
    initial_params: Sequence[float],
    mask: np.ndarray,
    dist_outside: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    base_tz: float,
    angle_bound: float = 10.0,
    trans_bound: float = 0.25,
    maxiter: int = 120,
) -> np.ndarray:
    x0 = np.asarray(initial_params, dtype=np.float64)
    simplex = np.tile(x0, (7, 1))
    for k, step in enumerate((3.0, 3.0, 2.5, 0.06, 0.06, 0.06)):
        simplex[k + 1, k] += step

    def objective(params: np.ndarray) -> float:
        if np.any(np.abs(params[:3] - x0[:3]) > angle_bound) or np.any(
            np.abs(params[3:] - x0[3:]) > trans_bound
        ):
            return 10.0
        rotation, translation = _pose_from_euler(*params)
        return _evaluate_pose_silhouette(
            hull_points,
            rotation,
            translation,
            mask,
            dist_outside,
            intrinsics,
            downsample=128,
            base_tz=base_tz,
            roll_deg=float(params[2]),
        )

    opt = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "maxiter": maxiter,
            "xatol": 0.15,
            "fatol": 1e-4,
        },
    )
    return np.asarray(opt.x, dtype=np.float64)


def marching_tetrahedra(
    field: np.ndarray,
    origin: Sequence[float],
    voxel_size: float,
    *,
    level: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight triangle surface mesh from a 3D scalar field."""
    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    origin_arr = np.asarray(origin, dtype=np.float64) - voxel_size
    px, py, pz = padded.shape

    vals = np.stack(
        [
            padded[dx : px - 1 + dx, dy : py - 1 + dy, dz : pz - 1 + dz]
            for dx, dy, dz in _MT_CORNERS
        ],
        axis=-1,
    )
    inside = vals >= level
    active = np.any(inside, axis=-1) & (~np.all(inside, axis=-1))
    ci, cj, ck = np.nonzero(active)
    if len(ci) == 0:
        raise ReconstructionError("Isosurface extraction produced no surface cells.")

    cell_coords = np.column_stack((ci, cj, ck))
    tet_vals = vals[ci, cj, ck][:, _MT_TETS]
    tet_in = tet_vals >= level
    counts = tet_in.sum(axis=-1)
    m_idx, t_idx = np.nonzero((counts > 0) & (counts < 4))

    t_cells = cell_coords[m_idx]
    t_corners = _MT_TETS[t_idx]
    t_vals = tet_vals[m_idx, t_idx]
    t_in = tet_in[m_idx, t_idx]

    stride_y = pz
    stride_x = py * pz
    c_off = t_cells[:, None, :] + _MT_CORNERS[t_corners]
    flat_c = c_off[..., 0] * stride_x + c_off[..., 1] * stride_y + c_off[..., 2]

    # Corner order per tet: inside corners first, then outside, each ascending
    # (a stable sort), so position 0 is the first inside corner and position
    # ``n_in`` the first outside corner.
    perm = np.argsort(~t_in, axis=1, kind="stable")
    n_in = t_in.sum(axis=1)
    rows = np.arange(len(perm))[:, None]
    pf = flat_c[rows, perm]
    pv = t_vals[rows, perm]

    def corner_xyz(flat_ids: np.ndarray) -> np.ndarray:
        x, rem = np.divmod(flat_ids, stride_x)
        y, z = np.divmod(rem, stride_y)
        return origin_arr + voxel_size * np.stack((x, y, z), axis=-1)

    single = (n_in == 1) | (n_in == 3)
    double = n_in == 2
    s_rows = np.flatnonzero(single)
    d_rows = np.flatnonzero(double)

    # One triangle: apex a is the lone in (or out) corner; b, c, d the others.
    s_one_in = n_in[s_rows] == 1
    s_apex = np.where(s_one_in, 0, 3)
    s_others = np.where(s_one_in[:, None], [1, 2, 3], [0, 1, 2])
    s_a = np.repeat(s_apex[:, None], 3, axis=1)
    # Two triangles: in corners (0, 1) to out corners (2, 3).
    d_a = np.array([[0, 0, 1, 1]])
    d_b = np.array([[2, 3, 2, 3]])

    edge_rows = np.concatenate((np.repeat(s_rows, 3), np.repeat(d_rows, 4)))
    edge_a = np.concatenate(
        (s_a.ravel(), np.broadcast_to(d_a, (len(d_rows), 4)).ravel())
    )
    edge_b = np.concatenate(
        (s_others.ravel(), np.broadcast_to(d_b, (len(d_rows), 4)).ravel())
    )
    f0 = pf[edge_rows, edge_a]
    f1 = pf[edge_rows, edge_b]
    v0 = pv[edge_rows, edge_a]
    v1 = pv[edge_rows, edge_b]

    total = int(px) * int(py) * int(pz)
    keys = np.minimum(f0, f1) * total + np.maximum(f0, f1)
    unique_keys, first, edge_vertex = np.unique(
        keys, return_index=True, return_inverse=True
    )
    denom = v1[first] - v0[first]
    safe = np.abs(denom) > 1e-12
    alpha = np.full(len(unique_keys), 0.5)
    alpha[safe] = (level - v0[first][safe]) / denom[safe]
    # Keep vertices a hair off grid corners: corner values within ~1e-7 of
    # the level otherwise give near-zero-area triangles whose removal below
    # would punch holes in the otherwise closed surface.
    alpha = np.clip(alpha, 1e-4, 1.0 - 1e-4)
    p0 = corner_xyz(f0[first])
    p1 = corner_xyz(f1[first])
    vertices = p0 + alpha[:, None] * (p1 - p0)

    n_single = 3 * len(s_rows)
    s_tris = edge_vertex[:n_single].reshape(-1, 3)
    d_quads = edge_vertex[n_single:].reshape(-1, 4)  # v00, v01, v10, v11
    d_tris = np.concatenate((d_quads[:, [0, 1, 3]], d_quads[:, [0, 3, 2]]), axis=0)
    faces = np.concatenate((s_tris, d_tris), axis=0)
    tri_rows = np.concatenate((s_rows, d_rows, d_rows))

    p_in = corner_xyz(pf[tri_rows, 0])
    p_out = corner_xyz(pf[tri_rows, n_in[tri_rows]])
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", normal, p_out - p_in) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    if len(faces) == 0:
        raise ReconstructionError("Marching tetrahedra produced no faces.")

    triangles = vertices[faces]
    cross_norms = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = float(np.max(np.ptp(vertices, axis=0)))
    threshold = np.finfo(np.float64).eps * max(scale * scale, 1e-300)
    valid_faces = faces[cross_norms > threshold]
    if len(valid_faces) < 4:
        raise ReconstructionError("Extracted surface contains insufficient triangles.")

    used = np.unique(valid_faces.reshape(-1))
    if len(used) != len(vertices):
        remap = np.empty(len(vertices), dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        vertices = vertices[used]
        valid_faces = remap[valid_faces]

    return vertices, valid_faces


def drop_small_components(
    vertices: np.ndarray, faces: np.ndarray, fraction: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Remove disconnected fragments smaller than ``fraction`` of the mesh.

    Carving leaves specks where silhouettes disagree by a pixel or two; they
    are noise, not parts of the object.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]]))
    n = len(vertices)
    count, labels = connected_components(
        coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
        directed=False,
    )
    if count <= 1:
        return vertices, faces
    sizes = np.bincount(labels, minlength=count)
    keep_label = sizes >= max(fraction * n, 1)
    keep_label[np.argmax(sizes)] = True
    keep_vertex = keep_label[labels]
    keep_face = keep_vertex[faces[:, 0]]
    remap = np.cumsum(keep_vertex) - 1
    return vertices[keep_vertex], remap[faces[keep_face]]


def rasterize_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project and rasterize a triangle mesh into a boolean camera mask."""
    width, height = intrinsics.image_width, intrinsics.image_height
    cam = vertices @ rotation.T + translation.reshape(1, 3)
    z = cam[:, 2]
    safe_z = np.maximum(z, 1e-6)
    uv = np.column_stack(
        (
            intrinsics.fx * (cam[:, 0] / safe_z) + intrinsics.cx,
            intrinsics.fy * (cam[:, 1] / safe_z) + intrinsics.cy,
        )
    )
    tri_uv = np.round(uv[faces]).astype(np.int32)
    tri_z = z[faces]
    in_front = np.all(tri_z > 1e-3, axis=1)

    silhouette = np.zeros((height, width), dtype=np.uint8)
    for tri in tri_uv[in_front]:
        cv2.fillConvexPoly(silhouette, tri, 255)
    return silhouette > 0


def _carve_top_cavity(
    points: np.ndarray,
    occupied: np.ndarray,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    masks: Sequence[np.ndarray],
    images: Sequence[Image.Image],
    mask_info: Sequence[tuple[float, float, float, float]],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    elevations = [
        float(Rotation.from_matrix(pose[0]).as_euler("zxy", degrees=True)[1])
        for pose in poses
    ]
    top_idx = int(np.argmax(np.abs(elevations)))
    if abs(elevations[top_idx]) <= 50.0:
        return occupied

    _, _, bw, bh = mask_info[top_idx]
    k_t = max(5, int(round(0.22 * min(bw, bh))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_t, k_t))
    body_top = (
        cv2.morphologyEx(masks[top_idx].astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        > 0
    )
    edt_top = distance_transform_edt(body_top)
    max_r = float(edt_top.max())
    if max_r <= 0:
        return occupied

    gray_top = cv2.cvtColor(
        np.asarray(images[top_idx].convert("RGB"), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    rim_band = (edt_top > 1.0) & (edt_top <= 0.14 * max_r)
    inner_core = edt_top > 0.28 * max_r
    if not (
        np.any(rim_band)
        and np.any(inner_core)
        and float(np.median(gray_top[inner_core]))
        < float(np.median(gray_top[rim_band])) - 5.0
    ):
        return occupied

    occ_pts = points[occupied]
    if len(occ_pts) == 0:
        return occupied

    r_top, t_top = poses[top_idx]
    y_min, y_max = float(occ_pts[:, 1].min()), float(occ_pts[:, 1].max())
    cam_top = occ_pts @ r_top.T + t_top.reshape(1, 3)
    near_ymin = float(cam_top[occ_pts[:, 1] < y_min + 0.2 * (y_max - y_min), 2].mean())
    near_ymax = float(cam_top[occ_pts[:, 1] > y_max - 0.2 * (y_max - y_min), 2].mean())

    if near_ymin < near_ymax:
        rim_mask_pts = occ_pts[:, 1] < (y_min + 0.22 * (y_max - y_min))
        in_depth_range = occ_pts[:, 1] < (y_max - 0.28 * (y_max - y_min))
    else:
        rim_mask_pts = occ_pts[:, 1] > (y_max - 0.22 * (y_max - y_min))
        in_depth_range = occ_pts[:, 1] > (y_min + 0.28 * (y_max - y_min))

    rim_pts = occ_pts[rim_mask_pts]
    if len(rim_pts) == 0:
        return occupied

    cam_rim = rim_pts @ r_top.T + t_top.reshape(1, 3)
    ur = np.round(
        intrinsics.fx * (cam_rim[:, 0] / cam_rim[:, 2]) + intrinsics.cx
    ).astype(np.int32)
    vr = np.round(
        intrinsics.fy * (cam_rim[:, 1] / cam_rim[:, 2]) + intrinsics.cy
    ).astype(np.int32)
    in_bounds = (
        (ur >= 0)
        & (ur < intrinsics.image_width)
        & (vr >= 0)
        & (vr < intrinsics.image_height)
    )
    core_rim_pts = rim_pts[in_bounds][inner_core[vr[in_bounds], ur[in_bounds]]]
    if len(core_rim_pts) <= 50:
        return occupied

    xz_center = np.mean(core_rim_pts[:, [0, 2]], axis=0)
    xz_radius = 0.78 * float(
        np.percentile(np.linalg.norm(core_rim_pts[:, [0, 2]] - xz_center, axis=1), 85)
    )
    frac = (occ_pts[:, 1] - y_min) / max(y_max - y_min, 1e-6)
    if near_ymin >= near_ymax:
        frac = 1.0 - frac
    taper_r = xz_radius * (1.0 - 0.30 * np.clip(frac, 0.0, 1.0))
    dist_xz = np.linalg.norm(occ_pts[:, [0, 2]] - xz_center, axis=1)
    carve_mask = in_depth_range & (dist_xz <= taper_r)

    updated = occupied.copy()
    occ_indices = np.flatnonzero(occupied)
    updated[occ_indices[carve_mask]] = False
    return updated


def _ortho_rotation(azimuth: float, elevation: float) -> np.ndarray:
    """World (Y up) to camera (x right, y down, z forward) rotation."""
    az, el = np.radians(azimuth), np.radians(elevation)
    forward = -np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    up = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up) > 0.999:
        up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.vstack((right, np.cross(forward, right), forward))


class _OrthoView:
    """A mask cropped to a small square for the global pose search."""

    def __init__(self, mask: np.ndarray, size: int = 48) -> None:
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        span = max(x1 - x0, y1 - y0) * 1.3 + 2
        self.offset = np.array(
            [0.5 * (x0 + x1) - 0.5 * span, 0.5 * (y0 + y1) - 0.5 * span]
        )
        self.k = size / span
        self.size = size
        h, w = mask.shape
        grid = (np.arange(size) + 0.5) / self.k
        sx = np.clip(np.round(self.offset[0] + grid).astype(int), 0, w - 1)
        sy = np.clip(np.round(self.offset[1] + grid).astype(int), 0, h - 1)
        inside_x = (self.offset[0] + grid >= 0) & (self.offset[0] + grid < w)
        inside_y = (self.offset[1] + grid >= 0) & (self.offset[1] + grid < h)
        crop = mask[np.ix_(sy, sx)] & inside_y[:, None] & inside_x[None, :]
        self.mask = crop
        cy, cx = np.nonzero(crop)
        self.bbox = (cx.min(), cx.max(), cy.min(), cy.max())

    def similarity(
        self, projected: np.ndarray, pad: float = 0.0, vertical: bool = False
    ) -> tuple[float, np.ndarray]:
        """Scale/centre mapping projected points' bbox onto the mask bbox.

        ``pad`` widens the points' bbox (in their units), e.g. by half a voxel
        when the points are voxel centres. ``vertical`` takes the scale from
        heights only, which thin sideways parts (arms) cannot disturb.
        """
        px0, py0 = projected.min(axis=0) - pad
        px1, py1 = projected.max(axis=0) + pad
        mx0, mx1, my0, my1 = self.bbox
        if vertical:
            scale = (my1 - my0 + 1) / max(py1 - py0, 1e-9)
        else:
            scale = 0.5 * (
                (mx1 - mx0 + 1) / max(px1 - px0, 1e-9)
                + (my1 - my0 + 1) / max(py1 - py0, 1e-9)
            )
        center = np.array([0.5 * (mx0 + mx1), 0.5 * (my0 + my1)]) - scale * 0.5 * (
            np.array([px0 + px1, py0 + py1])
        )
        return float(scale), center

    def pixels(self, uv: np.ndarray) -> np.ndarray:
        ui = np.round(uv[:, 0]).astype(np.int64)
        vi = np.round(uv[:, 1]).astype(np.int64)
        ok = (ui >= 0) & (ui < self.size) & (vi >= 0) & (vi < self.size)
        return np.where(ok, vi * self.size + ui, -1)

    def inside(self, pix: np.ndarray) -> np.ndarray:
        flat = self.mask.ravel()
        return np.where(pix >= 0, flat[np.maximum(pix, 0)], False)

    def iou(self, pix: np.ndarray) -> float:
        img = np.zeros(self.size * self.size, dtype=np.uint8)
        img[pix[pix >= 0]] = 1
        img = cv2.morphologyEx(
            img.reshape(self.size, self.size),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )
        sil = img > 0
        return float(
            np.count_nonzero(sil & self.mask)
            / max(np.count_nonzero(sil | self.mask), 1)
        )


def _known_view_poses(
    masks: Sequence[np.ndarray],
    angles: Sequence[tuple[float, float]],
    intrinsics: CameraIntrinsics,
    *,
    iterations: int = 4,
    vertical_scale: bool | None = None,
    refine_offsets: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera poses for known (azimuth, elevation) angles.

    Only each view's scale and centring are unknown. They are fitted by
    alternating carving with bbox matching of the carved hull's projection
    onto each silhouette, then converted to the perspective camera model.
    """
    # Placement errors of even a few pixels make the views carve away thin
    # parts (arms, legs, necks), so place at (near) full resolution.
    views = []
    for m in masks:
        ys, xs = np.nonzero(m)
        span = max(np.ptp(xs), np.ptp(ys)) * 1.3 + 2
        views.append(_OrthoView(m, size=int(min(span, 400))))
    lin = np.linspace(-0.9, 0.9, 72)
    pad = 0.5 * (lin[1] - lin[0])
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    rots = [_ortho_rotation(az, el) for az, el in angles]
    projected = [pts @ rot[:2].T for rot in rots]

    def carve(places) -> np.ndarray:
        kept = np.ones(len(pts), dtype=bool)
        for view, proj, (scale, center) in zip(views, projected, places, strict=True):
            kept &= view.inside(view.pixels(proj * scale + center))
        return kept

    def overlap(places, kept) -> float:
        if not kept.any():
            return 0.0
        return float(
            np.mean(
                [
                    view.iou(view.pixels(proj[kept] * scale + center))
                    for view, proj, (scale, center) in zip(
                        views, projected, places, strict=True
                    )
                ]
            )
        )

    def place(vertical: bool) -> tuple[list[tuple[float, np.ndarray]], float]:
        occ = np.ones(len(pts), dtype=bool)
        placements: list[tuple[float, np.ndarray]] = []
        for _ in range(iterations):
            placements = [
                view.similarity(proj[occ], pad=pad, vertical=vertical)
                for view, proj in zip(views, projected, strict=True)
            ]
            carved = carve(placements)
            if not carved.any():
                break
            occ = carved

        # Sideways offsets: a view shifted against the others carves away
        # parts they need, so pick each view's shift that keeps all fullest.
        best = overlap(placements, occ)
        for step in (4.0, 2.0, 1.0) if refine_offsets else ():
            for idx, view in enumerate(views):
                for direction in (-1.0, 1.0):
                    while True:
                        trial = list(placements)
                        scale, center = trial[idx]
                        shift = np.array([direction * step * view.k * 2.0, 0.0])
                        trial[idx] = (scale, center + shift)
                        kept = carve(trial)
                        score = overlap(trial, kept)
                        if score <= best + 1e-4:
                            break
                        best, placements, occ = score, trial, kept
        return placements, best

    # Whole-bbox scaling suits compact objects; height-only scaling survives
    # thin sideways parts (arms) that whole-bbox matching carves away. Keep
    # whichever leaves a hull that fills every silhouette best.
    modes = (False, True) if vertical_scale is None else (vertical_scale,)
    placements = max((place(mode) for mode in modes), key=lambda r: r[1])[0]

    poses = []
    for view, rot, (scale, center) in zip(views, rots, placements, strict=True):
        s_full = scale / view.k
        c_full = center / view.k + view.offset
        tz = intrinsics.fx / s_full
        trans = np.array(
            [
                (c_full[0] - intrinsics.cx) / s_full,
                (c_full[1] - intrinsics.cy) / s_full,
                tz,
            ]
        )
        poses.append((rot, trans))
    return poses


SHAPE_PRIORS = ("auto", "best", "revolution", "none")
_AUTO_REVOLUTION_MIN_IOU = 0.75


def _silhouette_scores(
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ious = []
    dices = []
    for (rotation, translation), mask in zip(poses, masks, strict=True):
        proj = rasterize_mesh_silhouette(
            vertices, faces, rotation, translation, intrinsics
        )
        inter = np.count_nonzero(proj & mask)
        union = np.count_nonzero(proj | mask)
        denom = np.count_nonzero(proj) + np.count_nonzero(mask)
        ious.append(float(inter / max(union, 1)))
        dices.append(float((2.0 * inter) / max(denom, 1)))
    return tuple(ious), tuple(dices)


def _try_revolution(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    resolution: int,
    hollow: bool | None,
    required: bool,
) -> VisualHullResult | None:
    from .revolve import analyze_view, is_surface_of_revolution, reconstruct_revolution

    try:
        analyses = [analyze_view(m) for m in masks]
        if not required and not is_surface_of_revolution(analyses):
            return None
        model = reconstruct_revolution(
            images,
            masks,
            intrinsics,
            analyses=analyses,
            resolution=resolution,
            hollow=hollow,
        )
    except ReconstructionError:
        if required:
            raise
        return None

    poses = list(zip(model.rotations, model.translations, strict=True))
    ious, dices = _silhouette_scores(
        model.vertices, model.faces, poses, masks, intrinsics
    )
    if not required and float(np.mean(ious)) < _AUTO_REVOLUTION_MIN_IOU:
        return None

    mesh = MeshResult(
        vertices=model.vertices,
        faces=model.faces,
        alpha=None,
        input_point_count=len(model.vertices),
        retained_point_count=len(model.vertices),
        vertex_colors=model.vertex_colors,
    )
    return VisualHullResult(
        mesh=mesh,
        camera_rotations=np.vstack(
            [Rotation.from_matrix(r).as_rotvec() for r in model.rotations]
        ),
        camera_translations=np.vstack([t.reshape(3) for t in model.translations]),
        intrinsics=intrinsics,
        masks=tuple(np.asarray(m, dtype=np.uint8) * 255 for m in masks),
        per_view_iou=ious,
        per_view_dice=dices,
        method="revolution",
    )


def reconstruct_visual_hull(
    images: Sequence[Image.Image],
    *,
    intrinsics: CameraIntrinsics | None = None,
    masks: Sequence[np.ndarray] | None = None,
    resolution: int = 88,
    carve_concavity: bool = True,
    shape_prior: str = "auto",
    view_angles: Sequence[tuple[float, float]] | None = None,
    prior_mesh: MeshResult | None = None,
    prior_up_axis: str | None = None,
    learned_backend: str | None = None,
    hollow: bool | None = None,
) -> VisualHullResult:
    """Reconstruct a watertight 3D mesh from multi-view silhouettes.

    ``shape_prior`` selects the model:

    - ``"auto"`` (default): use the surface-of-revolution model when the
      silhouettes look rotationally symmetric (cups, bowls, vases, bottles),
      otherwise fall back to visual-hull carving.
    - ``"revolution"``: always use the surface-of-revolution model.
    - ``"none"``: always use visual-hull carving.
    - ``"best"``: try every applicable strategy (revolution, carving, carving
      with ``view_angles``, ``learned_backend`` prior), score each by
      leave-one-out validation on held-out photos, build the winner and
      report a quality verdict with advice (see :mod:`img2mesh.validate`).
      Slower: several reconstructions per strategy.

    ``view_angles`` optionally gives each photo's camera (azimuth, elevation)
    in degrees, elevation positive when looking down (e.g. from a turntable
    rig). Silhouettes alone cannot pin down camera poses for arbitrary
    objects, so known angles make visual-hull carving far more accurate.

    ``prior_mesh`` supplies a shape prior, e.g. a mesh from a single-image 3D
    generator, in any frame and scale. Cameras are fitted against it and the
    result is the prior refined to match every silhouette (see
    :mod:`img2mesh.prior`). ``prior_up_axis`` names its up axis if known.
    ``learned_backend`` (e.g. ``"triposr"``) generates that prior from the
    photo with the largest silhouette (see :mod:`img2mesh.learned`).
    ``hollow`` forces the revolution model's top open (True) or closed
    (False); by default it is detected (and closed if ``carve_concavity`` is
    False).
    """
    if isinstance(images, (str, bytes)):
        raise InputError("Expected a sequence of decoded PIL images.")

    view_images = tuple(images)
    if len(view_images) < 2:
        raise ReconstructionError(
            "At least two images are required for silhouette reconstruction."
        )

    if any(not isinstance(img, Image.Image) for img in view_images):
        raise InputError("Every input view must be a PIL image.")

    width, height = view_images[0].size
    if any(img.size != (width, height) for img in view_images):
        raise ReconstructionError("All images must have identical dimensions.")

    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 24
    ):
        raise ConfigurationError("resolution must be an integer of at least 24.")

    if not isinstance(carve_concavity, bool):
        raise ConfigurationError("carve_concavity must be True or False.")

    if view_angles is not None:
        try:
            angles = np.asarray(view_angles, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("view_angles must be numeric pairs.") from exc
        if angles.shape != (len(view_images), 2) or not np.isfinite(angles).all():
            raise ConfigurationError(
                "view_angles needs one (azimuth, elevation) pair per image."
            )
        if np.any(np.abs(angles[:, 1]) > 90.0):
            raise ConfigurationError("Elevations must lie within [-90, 90] degrees.")
        view_angles = [tuple(map(float, a)) for a in angles]

    if shape_prior not in SHAPE_PRIORS:
        raise ConfigurationError(
            f"shape_prior must be one of: {', '.join(SHAPE_PRIORS)}."
        )

    if intrinsics is None:
        focal = float(max(width, height)) * 4.0
        intrinsics = CameraIntrinsics.from_focal_length(width, height, focal)
    elif (intrinsics.image_width, intrinsics.image_height) != (width, height):
        raise ReconstructionError(
            "Camera intrinsics dimensions must match the input images."
        )

    if masks is not None:
        if len(masks) != len(view_images):
            raise InputError("Mask count must match the number of images.")
        bool_masks = tuple(np.asarray(m) > 0 for m in masks)
    else:
        bool_masks = tuple(extract_foreground_mask(img) > 0 for img in view_images)

    if any(not np.any(m) for m in bool_masks):
        raise ReconstructionError(
            "Foreground segmentation produced an empty mask for one or more views."
        )

    if shape_prior == "best":
        from dataclasses import replace

        from .validate import quality_verdict, select_strategy

        best, scores = select_strategy(
            view_images,
            bool_masks,
            intrinsics,
            view_angles=view_angles,
            learned_backend=learned_backend,
            prior_up_axis=prior_up_axis,
            carve_concavity=carve_concavity,
            resolution=resolution,
        )
        result = reconstruct_visual_hull(
            view_images,
            masks=bool_masks,
            intrinsics=intrinsics,
            resolution=resolution,
            prior_mesh=prior_mesh,
            **best.strategy.kwargs,
        )
        quality, advice = quality_verdict(best, bool_masks, view_angles)
        return replace(
            result,
            method=f"best:{best.strategy.name}",
            holdout_iou=best.holdout_iou,
            quality=quality,
            advice=advice,
            candidates=tuple((s.strategy.name, s.holdout_iou) for s in scores),
        )

    prior_up_candidates = None
    prior_depth_axis = None
    prior_input = (None, None)
    if learned_backend is not None and prior_mesh is None:
        from .learned import (
            backend_depth_axis,
            backend_input_camera,
            backend_up_candidates,
            generate_prior_mesh,
        )
        from .revolve import analyze_view

        # Generators work in their input camera's frame, so feed the most
        # side-on photo: the object is then nearly upright in the output.
        elevations = [analyze_view(m).elevation_deg for m in bool_masks]
        best = int(np.argmin(elevations))
        prior_mesh = generate_prior_mesh(
            view_images[best], bool_masks[best], backend=learned_backend
        )
        if prior_up_axis is None:
            prior_up_candidates = backend_up_candidates(learned_backend)
        prior_depth_axis = backend_depth_axis(learned_backend)
        prior_input = (best, backend_input_camera(learned_backend))

    if prior_mesh is not None:
        from .prior import reconstruct_with_prior

        mesh, prior_poses, _ = reconstruct_with_prior(
            view_images,
            bool_masks,
            prior_mesh,
            intrinsics,
            resolution=max(resolution, 64),
            up_axis=prior_up_axis,
            up_candidates=prior_up_candidates,
            depth_axis=prior_depth_axis,
            input_view=prior_input[0],
            input_camera=prior_input[1],
        )
        ious, dices = _silhouette_scores(
            mesh.vertices, mesh.faces, prior_poses, bool_masks, intrinsics
        )
        return VisualHullResult(
            mesh=mesh,
            camera_rotations=np.vstack(
                [Rotation.from_matrix(r).as_rotvec() for r, _ in prior_poses]
            ),
            camera_translations=np.vstack([t.reshape(3) for _, t in prior_poses]),
            intrinsics=intrinsics,
            masks=tuple(m.astype(np.uint8) * 255 for m in bool_masks),
            per_view_iou=ious,
            per_view_dice=dices,
            method="prior",
        )

    if shape_prior != "none":
        revolved = _try_revolution(
            view_images,
            bool_masks,
            intrinsics,
            resolution=max(resolution + resolution // 2, 64),
            hollow=hollow
            if hollow is not None
            else (None if carve_concavity else False),
            required=shape_prior == "revolution",
        )
        if revolved is not None:
            return revolved

    dist_outside = [distance_transform_edt(~m) for m in bool_masks]
    mask_info = []
    for mask in bool_masks:
        ys, xs = np.nonzero(mask)
        mask_info.append(
            (
                float(xs.mean()),
                float(ys.mean()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            )
        )

    if view_angles is not None:
        resolved_poses = _known_view_poses(bool_masks, view_angles, intrinsics)
    else:
        aspect_ratios = [bh / bw for _, _, bw, bh in mask_info]
        anchor = int(np.argmin(aspect_ratios))
        base_tz = 2.0 * intrinsics.fx / mask_info[anchor][2]

        def init_translation(idx: int, tz: float) -> tuple[float, float, float]:
            mx, my, _, _ = mask_info[idx]
            return (
                (mx - intrinsics.cx) / intrinsics.fx * tz,
                (my - intrinsics.cy) / intrinsics.fy * tz,
                tz,
            )

        pts_coarse = _make_voxel_grid(44, 1.05)
        r_anchor, t_anchor = _pose_from_euler(
            0.0, 0.0, 0.0, *init_translation(anchor, base_tz)
        )
        cam_a = pts_coarse @ r_anchor.T + t_anchor.reshape(1, 3)
        u_a = np.round(
            intrinsics.fx * (cam_a[:, 0] / cam_a[:, 2]) + intrinsics.cx
        ).astype(np.int32)
        v_a = np.round(
            intrinsics.fy * (cam_a[:, 1] / cam_a[:, 2]) + intrinsics.cy
        ).astype(np.int32)
        in_a = (u_a >= 0) & (u_a < width) & (v_a >= 0) & (v_a < height)
        keep_a = np.zeros(len(pts_coarse), dtype=bool)
        keep_a[in_a] = bool_masks[anchor][v_a[in_a], u_a[in_a]]

        thick_map = _inflate_anchor_thickness(
            bool_masks[anchor], mask_info[anchor][2], mask_info[anchor][3]
        )
        z_radius = np.zeros(len(pts_coarse), dtype=np.float32)
        z_radius[keep_a] = thick_map[v_a[keep_a], u_a[keep_a]] * (
            base_tz / intrinsics.fx
        )
        hull_anchor = pts_coarse[keep_a & (np.abs(pts_coarse[:, 2]) <= z_radius)]

        poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(view_images)
        poses[anchor] = (r_anchor, t_anchor)

        for idx in sorted(range(len(view_images)), key=lambda i: -aspect_ratios[i]):
            if idx == anchor:
                continue
            registered = [j for j in range(len(view_images)) if poses[j] is not None]
            if len(registered) == 1:
                hull_pts = hull_anchor
            else:
                occ = _carve_voxels(
                    pts_coarse,
                    [poses[j] for j in registered if poses[j] is not None],
                    [bool_masks[j] for j in registered],
                    intrinsics,
                )
                hull_pts = pts_coarse[occ]
                if len(hull_pts) == 0:
                    hull_pts = hull_anchor

            if len(hull_pts) > 2500:
                hull_pts = hull_pts[np.linspace(0, len(hull_pts) - 1, 2500, dtype=int)]

            tx0, ty0, tz0 = init_translation(idx, base_tz)
            best = (1e9, 0.0, 0.0, 0.0)
            for az in range(-30, 31, 15):
                for el in range(-88, 89, 8):
                    for roll in (-8, 0, 8):
                        r_cand, t_cand = _pose_from_euler(az, el, roll, tx0, ty0, tz0)
                        score = _evaluate_pose_silhouette(
                            hull_pts,
                            r_cand,
                            t_cand,
                            bool_masks[idx],
                            dist_outside[idx],
                            intrinsics,
                            downsample=96,
                            base_tz=tz0,
                            roll_deg=float(roll),
                        )
                        if score < best[0]:
                            best = (score, float(az), float(el), float(roll))

            refined = _refine_pose_bounded(
                hull_pts,
                [best[1], best[2], best[3], tx0, ty0, tz0],
                bool_masks[idx],
                dist_outside[idx],
                intrinsics,
                base_tz=base_tz,
                angle_bound=15.0,
                trans_bound=0.4,
                maxiter=140,
            )
            poses[idx] = _pose_from_euler(*refined)

        resolved_poses = [pose for pose in poses if pose is not None]
        for refine_res in (56, 72):
            pts_r = _make_voxel_grid(refine_res, 1.1)
            occ_all = _carve_voxels(pts_r, resolved_poses, bool_masks, intrinsics)
            hull_all = pts_r[occ_all]
            if len(hull_all) == 0:
                break
            if len(hull_all) > 3500:
                hull_all = hull_all[np.linspace(0, len(hull_all) - 1, 3500, dtype=int)]
            for idx in range(len(view_images)):
                r_cur, t_cur = resolved_poses[idx]
                eul_cur = Rotation.from_matrix(r_cur).as_euler("zxy", degrees=True)
                x0 = [
                    float(eul_cur[2]),
                    float(eul_cur[1]),
                    float(eul_cur[0]),
                    float(t_cur[0]),
                    float(t_cur[1]),
                    float(t_cur[2]),
                ]
                refined = _refine_pose_bounded(
                    hull_all,
                    x0,
                    bool_masks[idx],
                    dist_outside[idx],
                    intrinsics,
                    base_tz=base_tz,
                    angle_bound=8.0,
                    trans_bound=0.25,
                    maxiter=120,
                )
                resolved_poses[idx] = _pose_from_euler(*refined)

    bound = 1.1
    pts_final = _make_voxel_grid(resolution, bound)
    occ_final = _carve_voxels(pts_final, resolved_poses, bool_masks, intrinsics)
    if not np.any(occ_final):
        raise ReconstructionError("Silhouette carving produced an empty volume.")

    if carve_concavity:
        occ_final = _carve_top_cavity(
            pts_final,
            occ_final,
            resolved_poses,
            bool_masks,
            view_images,
            mask_info,
            intrinsics,
        )

    # Signed distance (in voxels) of the carved volume, lightly smoothed:
    # a smoother iso-surface than blurring the binary occupancy.
    occ_grid = occ_final.reshape(resolution, resolution, resolution)
    sdf = distance_transform_edt(~occ_grid) - distance_transform_edt(occ_grid)
    field = np.clip(0.5 - gaussian_filter(sdf, sigma=1.0) / 2.0, 0.0, 1.0)
    voxel_size = (2.0 * bound) / (resolution - 1)
    vertices, faces = marching_tetrahedra(
        field,
        [-bound, -bound, -bound],
        voxel_size,
        level=0.5,
    )
    from .revolve import project_vertex_colors, taubin_smooth

    vertices, faces = drop_small_components(vertices, faces)
    vertices = taubin_smooth(vertices, faces)

    per_view_iou, per_view_dice = _silhouette_scores(
        vertices, faces, resolved_poses, bool_masks, intrinsics
    )

    rot_vecs = np.vstack(
        [Rotation.from_matrix(pose[0]).as_rotvec() for pose in resolved_poses]
    )
    trans_vecs = np.vstack([pose[1].reshape(3) for pose in resolved_poses])
    colors = project_vertex_colors(
        vertices,
        faces,
        view_images,
        bool_masks,
        [pose[0] for pose in resolved_poses],
        [pose[1].reshape(3) for pose in resolved_poses],
        intrinsics,
    )
    mesh = MeshResult(
        vertices=vertices,
        faces=faces,
        alpha=None,
        input_point_count=int(np.count_nonzero(occ_final)),
        retained_point_count=len(vertices),
        vertex_colors=colors,
    )

    return VisualHullResult(
        mesh=mesh,
        camera_rotations=rot_vecs,
        camera_translations=trans_vecs,
        intrinsics=intrinsics,
        masks=tuple(m.astype(np.uint8) * 255 for m in bool_masks),
        per_view_iou=per_view_iou,
        per_view_dice=per_view_dice,
    )
