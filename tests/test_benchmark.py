import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from img2mesh.benchmark import (
    evaluate,
    get_object,
    ground_truth_mesh,
    mesh_topology,
    render_view,
    render_views,
    sample_surface,
    shape_scores,
)
from img2mesh.benchmark.metrics import align_similarity
from img2mesh.benchmark.objects import BUILDERS, polygon_sdf
from img2mesh.silhouette import marching_tetrahedra


def test_polygon_sdf_is_signed_euclidean_distance():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.array([[0.5, 0.5], [2.0, 0.5], [0.5, -0.25], [2.0, 2.0]])
    d = polygon_sdf(pts, square)
    assert d == pytest.approx([-0.5, 1.0, 0.25, np.sqrt(2.0)])


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_object_is_closed_with_expected_genus(name):
    obj = get_object(name)
    # Thin vessel walls need ~4+ voxels (the benchmark itself uses 150).
    verts, faces = ground_truth_mesh(obj, resolution=150 if obj.hollow else 90)
    topo = mesh_topology(faces, len(verts))
    assert topo["watertight"] and topo["components"] == 1
    assert topo["genus"] == obj.genus
    lo, hi = obj.bounds
    assert np.all(verts.min(axis=0) > lo - 0.06)
    assert np.all(verts.max(axis=0) < hi + 0.06)


def test_render_view_mask_matches_projected_ground_truth():
    obj = get_object("lidded_jar")
    view = render_view(obj, 20.0, 25.0, size=128)
    verts, faces = ground_truth_mesh(obj, resolution=80)
    from img2mesh.silhouette import rasterize_mesh_silhouette

    proj = rasterize_mesh_silhouette(
        verts, faces, view.rotation, view.translation, view.intrinsics
    )
    iou = np.count_nonzero(proj & view.mask) / np.count_nonzero(proj | view.mask)
    assert iou > 0.95
    rgb = np.asarray(view.image)
    assert np.all(rgb[~view.mask] == 255)
    assert rgb[view.mask].mean() < 200


def test_shape_scores_and_similarity_alignment():
    obj = get_object("teapot")
    verts, faces = ground_truth_mesh(obj, resolution=80)
    gt = sample_surface(verts, faces, 6000, seed=1)
    diagonal = float(np.linalg.norm(np.ptp(verts, axis=0)))

    perfect = shape_scores(
        sample_surface(verts, faces, 6000, seed=2), gt, diagonal, 0.02
    )
    assert perfect.fscore > 0.99 and perfect.chamfer < 0.01

    rot = Rotation.from_euler("xyz", [8.0, 30.0, -5.0], degrees=True).as_matrix()
    moved = 1.7 * sample_surface(verts, faces, 6000, seed=3) @ rot.T + [0.3, -1.0, 2.0]
    s, r, t = align_similarity(moved, gt, initial_rotation=rot.T)
    aligned = shape_scores(s * moved @ r.T + t, gt, diagonal, 0.02)
    assert s == pytest.approx(1.0 / 1.7, rel=0.01)
    assert aligned.fscore > 0.99


def test_marching_tetrahedra_stays_closed_with_values_near_level():
    lin = np.linspace(-1.0, 1.0, 30)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    field = np.clip(0.5 - (np.sqrt(gx**2 + gy**2 + gz**2) - 0.55) / 0.14, 0.0, 1.0)
    rng = np.random.default_rng(0)
    near = rng.random(field.shape) < 0.05
    field[near] = 0.5 + rng.choice([-1.0, 1.0], size=near.sum()) * 1e-8
    verts, faces = marching_tetrahedra(field, [-1.0, -1.0, -1.0], 2.0 / 29.0, level=0.5)
    assert mesh_topology(faces, len(verts))["watertight"]


def test_benchmark_evaluate_scores_round_prior_above_visual_hull():
    obj = get_object("bowl")
    views = render_views(obj, size=160)
    gt = ground_truth_mesh(obj, resolution=96)
    auto = evaluate(obj, views, gt, "auto", resolution=48)
    hull = evaluate(obj, views, gt, "none", resolution=48)
    assert auto.error == "" and hull.error == ""
    assert auto.used == "revolution" and hull.used == "visual_hull"
    assert auto.fscore > 0.85
    assert auto.fscore > hull.fscore + 0.2
    assert auto.genus == obj.genus and auto.watertight
    assert 0.0 <= auto.novel_iou <= 1.0
