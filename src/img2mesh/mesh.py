from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from .config import ReconstructionError


@dataclass(frozen=True, slots=True)
class MeshResult:
    vertices: np.ndarray
    faces: np.ndarray
    alpha: float | None
    input_point_count: int
    retained_point_count: int

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


def _validate_points(points):
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(
            "Point cloud must contain numeric coordinates."
        ) from exc

    if array.ndim != 2 or array.shape[1] != 3:
        raise ReconstructionError("Point cloud must have shape (N, 3).")

    if not np.isfinite(array).all():
        raise ReconstructionError("Point cloud must contain finite coordinates.")

    return np.unique(array, axis=0)


def clean_point_cloud(
    points,
    *,
    neighbors: int = 8,
    std_ratio: float = 2.0,
) -> np.ndarray:
    points = _validate_points(points)

    if not isinstance(neighbors, int) or isinstance(neighbors, bool):
        raise ReconstructionError("neighbors must be an integer.")

    if neighbors < 2:
        raise ReconstructionError("neighbors must be at least 2.")

    if (
        isinstance(std_ratio, bool)
        or not isinstance(std_ratio, (int, float))
        or not np.isfinite(std_ratio)
        or std_ratio <= 0
    ):
        raise ReconstructionError("std_ratio must be positive.")

    if len(points) < 5:
        if len(points) < 4:
            raise ReconstructionError("At least four distinct 3D points are required.")
        return points.copy()

    k = min(neighbors, len(points) - 1)
    distances, _ = cKDTree(points).query(points, k=k + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    median = float(np.median(mean_distances))
    mad = float(np.median(np.abs(mean_distances - median)))

    threshold = median + std_ratio * 1.4826 * mad
    threshold += max(1e-12, abs(median) * 1e-12)

    retained = points[mean_distances <= threshold]
    if len(retained) < 4:
        raise ReconstructionError("Outlier removal left fewer than four points.")

    return retained.copy()


def _circumradius(tetrahedron: np.ndarray) -> float:
    origin = tetrahedron[0]
    matrix = 2.0 * (tetrahedron[1:] - origin)
    rhs = np.sum(tetrahedron[1:] ** 2, axis=1) - np.sum(origin**2)

    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return float("inf")

    return float(np.linalg.norm(center - origin))


def estimate_alpha(
    points,
    *,
    clean: bool = True,
    neighbors: int = 8,
    std_ratio: float = 2.0,
) -> float:
    """Estimate an adaptive alpha-shape radius from local point spacing."""
    cloud = _validate_points(points)
    if not isinstance(clean, bool):
        raise ReconstructionError("clean must be True or False.")

    if clean:
        cloud = clean_point_cloud(
            cloud,
            neighbors=neighbors,
            std_ratio=std_ratio,
        )

    if len(cloud) < 4:
        raise ReconstructionError(
            "Alpha estimation requires at least four distinct points."
        )

    k = min(4, len(cloud) - 1)
    distances, _ = cKDTree(cloud).query(cloud, k=k + 1)
    kth_distances = distances[:, k]
    spacing = float(np.percentile(kth_distances, 80))
    if not np.isfinite(spacing) or spacing <= 0:
        positive = distances[:, 1:][distances[:, 1:] > 0]
        if len(positive) == 0:
            raise ReconstructionError("Could not estimate alpha for degenerate points.")
        spacing = float(np.median(positive))

    candidate = 1.8 * spacing

    try:
        tetrahedralization = Delaunay(cloud)
    except QhullError as exc:
        raise ReconstructionError(
            "Could not tetrahedralize the point cloud. "
            "The points may be coplanar or geometrically degenerate."
        ) from exc

    radii = np.array(
        [
            _circumradius(cloud[simplex])
            for simplex in tetrahedralization.simplices
        ],
        dtype=np.float64,
    )
    finite_radii = radii[np.isfinite(radii) & (radii > 0)]
    if len(finite_radii) == 0:
        raise ReconstructionError("No finite tetrahedra found for alpha estimation.")

    min_feasible = float(np.percentile(finite_radii, 35))
    return float(max(candidate, min_feasible))


def generate_mesh(
    points,
    *,
    alpha: float | None = None,
    auto_alpha: bool = False,
    clean: bool = True,
    neighbors: int = 8,
    std_ratio: float = 2.0,
) -> MeshResult:
    original = _validate_points(points)
    input_count = len(np.asarray(points))

    if not isinstance(clean, bool):
        raise ReconstructionError("clean must be True or False.")

    if not isinstance(auto_alpha, bool):
        raise ReconstructionError("auto_alpha must be True or False.")

    if alpha is not None and auto_alpha:
        raise ReconstructionError("Cannot specify both alpha and auto_alpha.")

    if alpha is not None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ReconstructionError("alpha must be positive and finite.")

    if clean:
        cloud = clean_point_cloud(
            original,
            neighbors=neighbors,
            std_ratio=std_ratio,
        )
    else:
        cloud = original.copy()

    if len(cloud) < 4:
        raise ReconstructionError(
            "Mesh generation requires at least four distinct points."
        )

    effective_alpha = (
        estimate_alpha(cloud, clean=False) if auto_alpha else alpha
    )

    try:
        tetrahedralization = Delaunay(cloud)
    except QhullError as exc:
        raise ReconstructionError(
            "Could not tetrahedralize the point cloud. "
            "The points may be coplanar or geometrically degenerate."
        ) from exc

    face_map = {}

    for simplex in tetrahedralization.simplices:
        tetra = cloud[simplex]
        radius = _circumradius(tetra)

        if not np.isfinite(radius):
            continue

        if effective_alpha is not None and radius > effective_alpha:
            continue

        for opposite_index in range(4):
            face_indices = [
                int(simplex[index]) for index in range(4) if index != opposite_index
            ]

            a, b, c = cloud[face_indices]
            opposite = cloud[simplex[opposite_index]]
            normal = np.cross(b - a, c - a)

            if np.dot(normal, opposite - a) > 0:
                face_indices[1], face_indices[2] = (
                    face_indices[2],
                    face_indices[1],
                )

            key = tuple(sorted(face_indices))
            if key not in face_map:
                face_map[key] = [1, face_indices]
            else:
                face_map[key][0] += 1

    boundary_faces = [data[1] for data in face_map.values() if data[0] == 1]

    if boundary_faces:
        scale = float(np.max(np.ptp(cloud, axis=0)))
        area_threshold = np.finfo(np.float64).eps * max(scale * scale, 1e-300)
        boundary_faces = [
            face
            for face in boundary_faces
            if np.linalg.norm(
                np.cross(
                    cloud[face[1]] - cloud[face[0]],
                    cloud[face[2]] - cloud[face[0]],
                )
            )
            > area_threshold
        ]

    if not boundary_faces:
        raise ReconstructionError(
            "No mesh surface was generated. Increase alpha or check "
            "the point-cloud geometry."
        )

    used_indices = np.unique(np.asarray(boundary_faces).reshape(-1))
    remapping = {int(old): new for new, old in enumerate(used_indices)}

    faces = np.asarray(
        [[remapping[index] for index in face] for face in boundary_faces],
        dtype=np.int64,
    )
    vertices = cloud[used_indices].copy()

    if len(vertices) < 4 or len(faces) < 4:
        raise ReconstructionError("Generated surface contains insufficient geometry.")

    return MeshResult(
        vertices=vertices,
        faces=faces,
        alpha=None if effective_alpha is None else float(effective_alpha),
        input_point_count=input_count,
        retained_point_count=len(cloud),
    )
