import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from img2mesh.config import ConfigurationError, ReconstructionError
from img2mesh.geometry import CameraIntrinsics
from img2mesh.revolve import (
    analyze_view,
    camera_elevation,
    is_surface_of_revolution,
    taubin_smooth,
)
from img2mesh.silhouette import reconstruct_visual_hull

HEIGHT = 0.8
WALL = 0.05
FLOOR = 0.06
VIEWS = [(0.0, 0.0), (8.0, 6.0), (-12.0, 14.0), (10.0, 35.0), (0.0, 55.0), (4.0, 89.0)]


def _outer_radius(y):
    return 0.38 + 0.12 * (y / HEIGHT)


def _cup_points(step=0.0085):
    lin_x = np.arange(-0.56, 0.86, step)
    lin_y = np.arange(-0.02, HEIGHT + 0.02, step)
    lin_z = np.arange(-0.56, 0.56, step)
    gx, gy, gz = np.meshgrid(lin_x, lin_y, lin_z, indexing="ij")
    x, y, z = gx.ravel(), gy.ravel(), gz.ravel()
    radial = np.hypot(x, z)
    r_out = _outer_radius(np.clip(y, 0.0, HEIGHT))
    body = (y >= 0.0) & (y <= HEIGHT) & (radial <= r_out)
    cavity = (y > FLOOR) & (radial < r_out - WALL)
    ring = np.hypot(np.hypot(x - 0.55, y - 0.42) - 0.2, z)
    handle = (ring <= 0.06) & (x >= 0.42)
    solid = (body & ~cavity) | handle
    interior = solid & (y > FLOOR - 2 * step) & (radial < r_out - WALL + 1.5 * step)
    interior &= ~handle
    return np.column_stack((x, y, z))[solid], interior[solid]


def _view_rotation(az, el):
    flip = np.diag([1.0, -1.0, -1.0])
    for sign in (1.0, -1.0):
        rot = (
            Rotation.from_euler("zxy", [0.0, sign * el, az], degrees=True).as_matrix()
            @ flip
        )
        if rot[2, 1] <= 1e-9:  # camera looks down onto the cup
            return rot
    raise AssertionError("unreachable")


def _render_cup_views(size=256):
    """Z-buffered point renders of a hollow cup + handle on a white canvas."""
    focal = 2000.0
    intrinsics = CameraIntrinsics.from_focal_length(size, size, focal)
    points, interior = _cup_points()
    centered = points - np.array([0.0, 0.4, 0.0])
    colors = np.where(interior[:, None], [[60, 35, 20]], [[150, 90, 55]])

    images, masks = [], []
    for az, el in VIEWS:
        rot = _view_rotation(az, el)
        cam = centered @ rot.T + np.array([0.0, 0.0, 14.0])
        u = focal * cam[:, 0] / cam[:, 2] + intrinsics.cx
        v = focal * cam[:, 1] / cam[:, 2] + intrinsics.cy
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=bool)
        # Splat each point over a 2x2 footprint, keep the nearest per pixel.
        base_u = np.floor(u).astype(int)
        base_v = np.floor(v).astype(int)
        cand_u = np.concatenate([base_u + du for du in (0, 1) for _ in (0, 1)])
        cand_v = np.concatenate([base_v + dv for _ in (0, 1) for dv in (0, 1)])
        cand_z = np.tile(cam[:, 2], 4)
        cand_c = np.tile(colors, (4, 1))
        ok = (cand_u >= 0) & (cand_u < size) & (cand_v >= 0) & (cand_v < size)
        order = np.flatnonzero(ok)[np.argsort(cand_z[ok])]
        pix = cand_v[order] * size + cand_u[order]
        _, first = np.unique(pix, return_index=True)
        sel = order[first]
        canvas[cand_v[sel], cand_u[sel]] = cand_c[sel]
        mask[cand_v[sel], cand_u[sel]] = True
        images.append(Image.fromarray(canvas))
        masks.append(mask)
    return images, masks, intrinsics


@pytest.fixture(scope="module")
def cup_views():
    return _render_cup_views()


@pytest.fixture(scope="module")
def cup_result(cup_views):
    images, masks, intrinsics = cup_views
    return reconstruct_visual_hull(images, intrinsics=intrinsics, resolution=64)


def _edge_counts(faces):
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def _volume(vertices, faces):
    tri = vertices[faces]
    return (
        float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum())
        / 6.0
    )


def test_analyze_view_recovers_elevation_and_handle(cup_views):
    _, masks, _ = cup_views
    analyses = [analyze_view(m) for m in masks]
    for (_, el), analysis in zip(VIEWS, analyses, strict=True):
        # These estimates only seed the camera fit (checked separately).
        tolerance = 10.0 if el <= 35.0 else 20.0
        assert analysis.elevation_deg == pytest.approx(el, abs=tolerance)
        assert analysis.handle_side == 1
    assert is_surface_of_revolution(analyses)

    side = analyses[0]
    assert side.height_ratio == pytest.approx(HEIGHT, rel=0.06)
    assert side.profile[0] == pytest.approx(0.5, abs=0.02)
    assert side.profile[-1] == pytest.approx(0.38, abs=0.03)


def test_auto_prior_rebuilds_hollow_cup_with_handle(cup_result):
    result = cup_result
    mesh = result.mesh
    assert result.method == "revolution"
    assert result.mean_iou >= 0.92
    assert result.min_iou >= 0.88

    # Fitted cameras recover the true elevations (near-top views are least
    # constrained: their outline barely changes with elevation).
    for (_, el), rotvec in zip(VIEWS, result.camera_rotations, strict=True):
        fitted = camera_elevation(Rotation.from_rotvec(rotvec).as_matrix())
        assert fitted == pytest.approx(el, abs=10.0 if el < 60 else 15.0)

    counts = _edge_counts(mesh.faces)
    assert np.all(counts == 2), "mesh must be closed and manifold"
    euler = len(mesh.vertices) - len(counts) + len(mesh.faces)
    assert euler == 0, "a cup with one handle has genus 1"

    vertices = mesh.vertices
    body_diameter = -2.0 * vertices[:, 0].min()
    height = np.ptp(vertices[:, 1])
    assert height / body_diameter == pytest.approx(HEIGHT, rel=0.08)

    # Hollow: far less volume than the solid of revolution of the outer wall.
    ys = np.linspace(0.0, HEIGHT, 200)
    solid = np.pi * np.trapezoid(_outer_radius(ys) ** 2, ys)
    assert abs(_volume(vertices, mesh.faces)) < 0.5 * solid * body_diameter**3

    colors = mesh.vertex_colors
    assert colors is not None and colors.shape == (len(vertices), 3)
    assert np.all((colors >= 0.0) & (colors <= 1.0))
    # Inside of the bowl is darker than the outer wall, as in the renders.
    radial = np.hypot(vertices[:, 0], vertices[:, 2]) / body_diameter
    y_true = vertices[:, 1] / height * HEIGHT
    mid = (y_true > 0.3 * HEIGHT) & (y_true < 0.7 * HEIGHT)
    split = _outer_radius(y_true) - 0.5 * WALL
    inner = mid & (radial < split)
    outer = mid & (radial > split) & (vertices[:, 0] < 0)
    assert np.count_nonzero(inner) > 100 and np.count_nonzero(outer) > 100
    assert colors[inner].mean() < colors[outer].mean() - 0.1


def test_carve_concavity_false_builds_solid_body(cup_views):
    images, masks, intrinsics = cup_views
    result = reconstruct_visual_hull(
        images,
        intrinsics=intrinsics,
        resolution=48,
        carve_concavity=False,
        shape_prior="revolution",
    )
    counts = _edge_counts(result.mesh.faces)
    euler = len(result.mesh.vertices) - len(counts) + len(result.mesh.faces)
    assert euler == 0  # solid body + handle loop is still genus 1
    assert result.mean_iou >= 0.9


def test_shape_prior_validation_and_requirements():
    images = [Image.new("RGB", (64, 64), "white") for _ in range(2)]
    with pytest.raises(ConfigurationError):
        reconstruct_visual_hull(images, shape_prior="sphere")

    # Two top-down discs: no side-on view to measure a profile from.
    images = [Image.new("RGB", (96, 96), "white") for _ in range(2)]
    yy, xx = np.mgrid[0:96, 0:96]
    disc = ((xx - 48) / 40.0) ** 2 + ((yy - 48) / 40.0) ** 2 <= 1.0
    masks = [disc.copy(), disc.copy()]
    with pytest.raises(ReconstructionError, match="side-on"):
        reconstruct_visual_hull(images, masks=masks, shape_prior="revolution")


def test_taubin_smooth_reduces_noise_without_shrinking():
    from img2mesh.silhouette import marching_tetrahedra

    lin = np.linspace(-1.0, 1.0, 32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    field = np.clip(0.5 - (np.sqrt(gx**2 + gy**2 + gz**2) - 0.6) / 0.13, 0.0, 1.0)
    verts, faces = marching_tetrahedra(field, [-1.0, -1.0, -1.0], 2.0 / 31.0, level=0.5)
    noisy = verts * (
        1.0 + 0.02 * np.random.default_rng(0).standard_normal((len(verts), 1))
    )
    smoothed = taubin_smooth(noisy, faces, iterations=10)
    r_noisy = np.linalg.norm(noisy, axis=1)
    r_smooth = np.linalg.norm(smoothed, axis=1)
    assert np.std(r_smooth) < 0.5 * np.std(r_noisy)
    assert np.mean(r_smooth) == pytest.approx(0.6, rel=0.03)
