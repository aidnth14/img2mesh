"""Reconstruction guided by an external (e.g. learned) shape prior.

Silhouettes alone cannot pin down camera poses for arbitrary objects, and
single-image 3D generators (TripoSR, InstantMesh, Hunyuan3D, ...) produce
plausible shapes in an arbitrary frame and scale that do not exactly match
the photos. Combining them works well:

1. find the prior's up axis and fit every photo's camera against it; a
   shared model frame makes all cameras mutually consistent,
2. carve the visual hull with those cameras and intersect it with the
   (slightly dilated) prior, so the result matches every silhouette while
   keeping the prior's concavities,
3. project vertex colours from the photos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import ReconstructionError
from .geometry import CameraIntrinsics
from .mesh import MeshResult
from .revolve import (
    ViewAnalysis,
    _fit_pose,
    _fit_pose_candidates,
    _ViewTarget,
    project_vertex_colors,
    taubin_smooth,
)

# Generic objects give no elevation cue: search every elevation, both signs
# (the camera-above constraint inside the fit rejects views from below).
_ALL_ELEVATIONS = tuple(float(e) for e in np.arange(-90.0, 90.1, 10.0))

# Candidate rotations taking each signed axis of the prior to world +Y.
_UP_CANDIDATES = {
    "+y": np.eye(3),
    "-y": Rotation.from_euler("x", 180, degrees=True).as_matrix(),
    "+z": Rotation.from_euler("x", -90, degrees=True).as_matrix(),
    "-z": Rotation.from_euler("x", 90, degrees=True).as_matrix(),
    "+x": Rotation.from_euler("z", 90, degrees=True).as_matrix(),
    "-x": Rotation.from_euler("z", -90, degrees=True).as_matrix(),
}


def normalize_prior(vertices: np.ndarray) -> np.ndarray:
    """Centre on the bounding box and scale the longest side to 1.6."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return (vertices - 0.5 * (lo + hi)) * (1.6 / max(float(np.max(hi - lo)), 1e-12))


def _surface_samples(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    keep = area > 0
    tri, normals, area = tri[keep], normals[keep], area[keep]
    chosen = rng.choice(len(tri), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[chosen]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    n = normals[chosen]
    return pts, n / np.linalg.norm(n, axis=1, keepdims=True)


def prior_signed_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    shape: tuple[int, int, int],
    voxel: float,
    samples: int = 200000,
) -> np.ndarray:
    """Approximate signed distance (negative inside, capped at 4 voxels).

    ``points`` are the grid's cell centres in C order with ``shape``. Cells
    within a voxel of the surface form a barrier; "outside" is everything
    reachable from the grid border without crossing it. Unlike signing by
    surface normals this tolerates noisy normals and cracks narrower than a
    voxel, which learned meshes often have.
    """
    from scipy.ndimage import binary_dilation, label

    surface, _ = _surface_samples(vertices, faces, samples)
    # Only distances within a few voxels matter (barrier band, dilation);
    # capping the search keeps far-away grid cells cheap.
    cap = 4.0 * voxel
    dist, _ = cKDTree(surface).query(points, distance_upper_bound=cap)
    dist = np.minimum(dist, cap)
    band = (dist < 0.9 * voxel).reshape(shape)
    band = binary_dilation(band, iterations=1)
    labels, _ = label(~band)
    border = np.unique(
        np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    outside = np.isin(labels, border[border > 0])
    sign = np.where(outside.ravel(), 1.0, -1.0)
    return sign * dist


def _rank_up_axes(
    vertices: np.ndarray,
    views: Sequence[ViewAnalysis],
    intrinsics: CameraIntrinsics,
) -> list[str]:
    """Up-axis candidates ordered by how well single-view fits explain views."""
    probe = list(views[: min(3, len(views))])
    scores = []
    for name, rot in _UP_CANDIDATES.items():
        pts = (vertices @ rot.T)[:: max(1, len(vertices) // 12000)]
        fits = [
            _fit_pose(pts, v, intrinsics, elevations=_ALL_ELEVATIONS)[2] for v in probe
        ]
        scores.append((float(np.mean(fits)), name))
    return [name for _, name in sorted(scores, reverse=True)]


def _consistent_poses(
    oriented: np.ndarray,
    faces: np.ndarray,
    views: Sequence[ViewAnalysis],
    intrinsics: CameraIntrinsics,
    *,
    candidates: int = 3,
    grid_res: int = 44,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    """Per-view pose candidates, then the mutually consistent combination.

    A symmetric shape fits one view equally well from several orientations,
    and views fitted independently may pick incompatible ones. Score a
    combination by carving the prior with all its cameras and measuring how
    well the result still fills every silhouette.
    """
    from .silhouette import _carve_voxels, _make_voxel_grid

    # Candidates only need to be told apart; the final fit refines the winner.
    pts = oriented[:: max(1, len(oriented) // 15000)]
    options = [
        _fit_pose_candidates(
            pts, v, intrinsics, elevations=_ALL_ELEVATIONS, count=candidates
        )
        for v in views
    ]
    bound = 1.1
    grid = _make_voxel_grid(grid_res, bound)
    voxel = 2.0 * bound / (grid_res - 1)
    inside = (
        prior_signed_distance(oriented, faces, grid, (grid_res,) * 3, voxel) < voxel
    )
    targets = [_ViewTarget(v.mask, grid=48) for v in views]
    masks = [v.mask for v in views]

    def consistency(choice: Sequence[int]) -> float:
        poses = [options[i][c][:2] for i, c in enumerate(choice)]
        occ = _carve_voxels(grid, poses, masks, intrinsics) & inside
        if not occ.any():
            return 0.0
        hull = grid[occ]
        return float(
            np.mean(
                [
                    t.iou(hull, rot, trans, intrinsics)
                    for t, (rot, trans) in zip(targets, poses)
                ]
            )
        )

    choice = [0] * len(views)
    best = consistency(choice)
    for _ in range(2):
        improved = False
        for i, opts in enumerate(options):
            for c in range(len(opts)):
                if c == choice[i]:
                    continue
                trial = list(choice)
                trial[i] = c
                score = consistency(trial)
                if score > best + 1e-4:
                    best, choice, improved = score, trial, True
        if not improved:
            break
    return [options[i][c][:2] for i, c in enumerate(choice)], best


def reconstruct_with_prior(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    prior: MeshResult,
    intrinsics: CameraIntrinsics,
    *,
    resolution: int = 110,
    up_axis: str | None = None,
    dilation: float = 0.02,
) -> tuple[MeshResult, list[tuple[np.ndarray, np.ndarray]], str]:
    """Fit cameras to an external prior mesh and refine it against the photos.

    ``up_axis`` ("+y", "-z", ...) names the prior's up direction if known;
    otherwise it is chosen by silhouette fit. ``dilation`` (in prior units,
    longest side 1.6) is how far the result may extend beyond the prior where
    every silhouette still allows it. Returns the mesh, the per-view cameras
    and the up axis used.
    """
    from .revolve import analyze_view
    from .silhouette import _carve_voxels, _make_voxel_grid, marching_tetrahedra

    if prior.face_count < 4:
        raise ReconstructionError("The prior mesh has too few faces.")
    views = [analyze_view(m) for m in masks]
    base = normalize_prior(np.asarray(prior.vertices, dtype=np.float64))
    faces = np.asarray(prior.faces, dtype=np.int64)

    if up_axis is None:
        ranked = _rank_up_axes(base, views, intrinsics)[:2]
    elif up_axis in _UP_CANDIDATES:
        ranked = [up_axis]
    else:
        raise ReconstructionError(
            f"up_axis must be one of {', '.join(_UP_CANDIDATES)}."
        )
    best = None
    for name in ranked:
        candidate = base @ _UP_CANDIDATES[name].T
        poses, score = _consistent_poses(candidate, faces, views, intrinsics)
        if best is None or score > best[0]:
            best = (score, name, candidate, poses)
    _, up_axis, oriented, poses = best

    bound = 1.1
    grid = _make_voxel_grid(resolution, bound)
    occ = _carve_voxels(grid, poses, [np.asarray(m) > 0 for m in masks], intrinsics)
    sdf_prior = prior_signed_distance(
        oriented,
        faces,
        grid,
        (resolution, resolution, resolution),
        (2.0 * bound) / (resolution - 1),
    )
    occ &= sdf_prior < dilation
    if not occ.any():
        raise ReconstructionError("The prior and the silhouettes do not overlap.")

    occ_grid = occ.reshape(resolution, resolution, resolution)
    sdf = distance_transform_edt(~occ_grid) - distance_transform_edt(occ_grid)
    field = np.clip(0.5 - gaussian_filter(sdf, sigma=1.0) / 2.0, 0.0, 1.0)
    voxel = (2.0 * bound) / (resolution - 1)
    verts, tris = marching_tetrahedra(field, [-bound] * 3, voxel, level=0.5)
    verts = taubin_smooth(verts, tris)

    # Final camera refinement against the refined shape.
    sample = verts[:: max(1, len(verts) // 30000)]
    poses = [
        _fit_pose(sample, view, intrinsics, init=pose)[:2]
        for view, pose in zip(views, poses, strict=True)
    ]
    colors = project_vertex_colors(
        verts,
        tris,
        images,
        [v.mask for v in views],
        [p[0] for p in poses],
        [p[1] for p in poses],
        intrinsics,
    )
    mesh = MeshResult(
        vertices=verts,
        faces=tris,
        alpha=None,
        input_point_count=prior.vertex_count,
        retained_point_count=len(verts),
        vertex_colors=colors,
    )
    return mesh, poses, up_axis


def silhouette_overlap(
    mesh: MeshResult,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> float:
    """Mean coarse silhouette IoU of ``mesh`` over the given cameras."""
    pts = mesh.vertices[:: max(1, mesh.vertex_count // 30000)]
    return float(
        np.mean(
            [
                _ViewTarget(np.asarray(m) > 0).iou(pts, rot, trans, intrinsics)
                for m, (rot, trans) in zip(masks, poses, strict=True)
            ]
        )
    )
