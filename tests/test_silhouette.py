import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from img2mesh import cli
from img2mesh.config import ConfigurationError, InputError, ReconstructionError
from img2mesh.geometry import CameraIntrinsics
from img2mesh.io import export_glb
from img2mesh.silhouette import (
    extract_foreground_mask,
    marching_tetrahedra,
    rasterize_mesh_silhouette,
    reconstruct_visual_hull,
)


def _make_synthetic_cup_views():
    """Render 6 multi-elevation views of a tapered cup + handle on white canvas."""
    width, height = 256, 256
    intrinsics = CameraIntrinsics.from_focal_length(width, height, 360.0)

    lin = np.linspace(-0.8, 0.8, 44)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    # Cup body centered at (-0.15, 0, 0) + curved handle loop at (+0.38..+0.78)
    r_xz = np.hypot(pts[:, 0] + 0.15, pts[:, 2])
    outer_r = 0.50 - 0.12 * ((pts[:, 1] + 0.45) / 0.90)
    body = (pts[:, 1] >= -0.45) & (pts[:, 1] <= 0.45) & (r_xz <= outer_r)

    handle_center_x = 0.35
    handle_center_y = 0.04
    handle_ring = np.hypot(
        np.hypot(pts[:, 0] - handle_center_x, pts[:, 1] - handle_center_y) - 0.24,
        pts[:, 2],
    )
    handle = (pts[:, 0] >= 0.25) & (handle_ring <= 0.085)

    occ_pts = pts[body | handle]

    view_angles = [
        (0.0, 0.0),
        (2.0, -8.0),
        (-3.0, -22.0),
        (5.0, -40.0),
        (0.0, -65.0),
        (-2.0, 4.0),
    ]
    images = []
    masks = []
    for az, el in view_angles:
        rot = Rotation.from_euler("zxy", [0.0, el, az], degrees=True).as_matrix()
        trans = np.array([0.0, 0.0, 2.8], dtype=np.float64)
        cam = occ_pts @ rot.T + trans.reshape(1, 3)
        u = np.round(intrinsics.fx * cam[:, 0] / cam[:, 2] + intrinsics.cx).astype(
            np.int32
        )
        v = np.round(intrinsics.fy * cam[:, 1] / cam[:, 2] + intrinsics.cy).astype(
            np.int32
        )
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        sil = np.zeros((height, width), dtype=np.uint8)
        sil[v[valid], u[valid]] = 255
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[sil > 0] = (120, 72, 42)
        images.append(Image.fromarray(canvas))
        masks.append(sil > 0)

    return images, masks, intrinsics


def test_extract_foreground_mask_separates_subject_and_fills_pinholes():
    canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.circle(canvas, (64, 64), 35, (110, 65, 35), -1)
    canvas[62:66, 62:66] = (255, 255, 255)  # small 16px pinhole

    mask = extract_foreground_mask(Image.fromarray(canvas))
    assert mask.shape == (128, 128)
    assert mask[64, 64] == 255
    assert mask[5, 5] == 0


def test_extract_foreground_mask_rejects_invalid_input():
    with pytest.raises(InputError):
        extract_foreground_mask("not an image")


def test_marching_tetrahedra_extracts_closed_sphere():
    lin = np.linspace(-1.0, 1.0, 24)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    field = (gx**2 + gy**2 + gz**2 <= 0.6**2).astype(np.float64)

    verts, faces = marching_tetrahedra(field, [-1.0, -1.0, -1.0], 2.0 / 23.0)
    assert len(verts) > 100
    assert len(faces) > 100
    radii = np.linalg.norm(verts, axis=1)
    assert np.mean(radii) == pytest.approx(0.6, abs=0.08)


def test_reconstruct_visual_hull_achieves_high_model_to_image_accuracy(tmp_path):
    images, gt_masks, intrinsics = _make_synthetic_cup_views()
    result = reconstruct_visual_hull(
        images,
        intrinsics=intrinsics,
        resolution=48,
        carve_concavity=False,
        shape_prior="none",
    )

    assert result.method == "visual_hull"
    assert result.mean_iou >= 0.90
    assert result.min_iou >= 0.85
    assert result.mean_dice >= 0.94
    assert result.mesh.vertex_count > 500
    assert result.mesh.face_count > 500

    for idx, gt_mask in enumerate(gt_masks):
        rot, _ = cv2.Rodrigues(result.camera_rotations[idx])
        proj = rasterize_mesh_silhouette(
            result.mesh.vertices,
            result.mesh.faces,
            rot,
            result.camera_translations[idx],
            result.intrinsics,
        )
        iou = np.count_nonzero(proj & gt_mask) / np.count_nonzero(proj | gt_mask)
        assert iou >= 0.85

    glb_path = export_glb(result.mesh, tmp_path / "cup.glb")
    assert glb_path.is_file() and glb_path.stat().st_size > 0


def test_reconstruct_visual_hull_rejects_invalid_options():
    images = [Image.new("RGB", (64, 64), "white") for _ in range(2)]
    with pytest.raises(ConfigurationError):
        reconstruct_visual_hull(images, resolution=8)
    with pytest.raises(ConfigurationError):
        reconstruct_visual_hull(images, carve_concavity="yes")
    with pytest.raises(ReconstructionError, match="empty mask"):
        reconstruct_visual_hull(images)


def test_cli_silhouette_and_auto_modes(tmp_path):
    images, _, _ = _make_synthetic_cup_views()
    for idx, image in enumerate(images):
        image.save(tmp_path / f"view_{idx:02d}.png")

    out_sil = tmp_path / "sil.glb"
    code = cli.main([str(tmp_path), str(out_sil), "--mode", "silhouette"])
    assert code == cli.EXIT_SUCCESS
    assert out_sil.is_file() and out_sil.stat().st_size > 0


def test_cli_shape_prior_option(tmp_path):
    images, _, _ = _make_synthetic_cup_views()
    for idx, image in enumerate(images):
        image.save(tmp_path / f"view_{idx:02d}.png")

    assert cli.build_parser().parse_args(["a", "b.glb"]).shape_prior == "auto"
    out = tmp_path / "hull.obj"
    code = cli.main(
        [str(tmp_path), str(out), "--mode", "silhouette", "--shape-prior", "none"]
    )
    assert code == cli.EXIT_SUCCESS
    first_vertex = next(
        line for line in out.read_text().splitlines() if line.startswith("v ")
    )
    assert len(first_vertex.split()) == 7  # x y z r g b: carved hulls are coloured


def test_cli_view_angles_parsing(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(["a", "b.glb", "--view-angles", "0:10,90:5.5"])
    assert args.view_angles == [(0.0, 10.0), (90.0, 5.5)]
    with pytest.raises(SystemExit):
        parser.parse_args(["a", "b.glb", "--view-angles", "0-10"])

    images, _, _ = _make_synthetic_cup_views()
    for idx, image in enumerate(images):
        image.save(tmp_path / f"view_{idx:02d}.png")
    code = cli.main(
        [
            str(tmp_path),
            str(tmp_path / "x.glb"),
            "--mode",
            "silhouette",
            "--view-angles",
            "0:0,10:10",
        ]
    )
    assert code == cli.EXIT_ERROR  # 2 angles for 6 images
