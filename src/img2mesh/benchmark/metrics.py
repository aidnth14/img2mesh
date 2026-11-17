"""Shape-accuracy metrics for reconstructed meshes against ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0
) -> np.ndarray:
    """Area-weighted uniform point samples on a triangle mesh."""
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    chosen = rng.choice(len(faces), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[chosen]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def mesh_topology(faces: np.ndarray, vertex_count: int) -> dict[str, int | bool]:
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    euler = vertex_count - len(uniq) + len(faces)

    graph = coo_matrix(
        (np.ones(len(uniq)), (uniq[:, 0], uniq[:, 1])),
        shape=(vertex_count, vertex_count),
    )
    components, _ = connected_components(graph, directed=False)
    watertight = bool(np.all(counts == 2))
    genus = (2 * components - euler) // 2 if watertight else -1
    return {
        "euler": int(euler),
        "components": int(components),
        "watertight": watertight,
        "genus": int(genus),
    }


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    rot = u @ sign @ vt
    var = float((xs**2).sum() / len(src))
    scale = float(np.trace(np.diag(d) @ sign) / max(var, 1e-12))
    return scale, rot, mu_d - scale * rot @ mu_s


def align_similarity(
    source: np.ndarray,
    reference: np.ndarray,
    initial_rotation: np.ndarray | None = None,
    iterations: int = 40,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (scale, R, t) mapping ``source`` points onto ``reference``.

    Starts from ``initial_rotation`` with centroid/RMS-radius matching, then
    refines with symmetric nearest-neighbour ICP.
    """
    rot = np.eye(3) if initial_rotation is None else initial_rotation
    src_c, ref_c = source.mean(axis=0), reference.mean(axis=0)
    src_r = np.sqrt(((source - src_c) ** 2).sum(axis=1).mean())
    ref_r = np.sqrt(((reference - ref_c) ** 2).sum(axis=1).mean())
    scale = ref_r / max(src_r, 1e-12)
    trans = ref_c - scale * rot @ src_c

    ref_tree = cKDTree(reference)
    for _ in range(iterations):
        moved = scale * source @ rot.T + trans
        _, to_ref = ref_tree.query(moved)
        _, to_src = cKDTree(moved).query(reference)
        src_pairs = np.vstack((source, source[to_src]))
        dst_pairs = np.vstack((reference[to_ref], reference))
        new_scale, new_rot, new_trans = _umeyama(src_pairs, dst_pairs)
        change = abs(new_scale - scale) + np.abs(new_rot - rot).max()
        scale, rot, trans = new_scale, new_rot, new_trans
        if change < 1e-6:
            break
    return scale, rot, trans


@dataclass(frozen=True)
class ShapeScores:
    chamfer: float  # mean symmetric distance, as a fraction of GT diagonal
    precision: float  # recon surface within tau of GT
    recall: float  # GT surface within tau of recon
    fscore: float
    tau: float


def shape_scores(
    recon_points: np.ndarray, gt_points: np.ndarray, diagonal: float, tau: float
) -> ShapeScores:
    d_rg, _ = cKDTree(gt_points).query(recon_points)
    d_gr, _ = cKDTree(recon_points).query(gt_points)
    threshold = tau * diagonal
    precision = float(np.mean(d_rg < threshold))
    recall = float(np.mean(d_gr < threshold))
    fscore = 2 * precision * recall / max(precision + recall, 1e-12)
    chamfer = 0.5 * (d_rg.mean() + d_gr.mean()) / diagonal
    return ShapeScores(float(chamfer), precision, recall, float(fscore), tau)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))
