from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

import img2mesh.pipeline as pipeline_mod
from img2mesh.config import ConfigurationError, ReconstructionError
from img2mesh.features import (
    FeatureObservation,
    FeatureSet,
    FeatureTrack,
    MultiViewFeatureGraph,
    PairwiseMatches,
)
from img2mesh.geometry import CameraIntrinsics, TriangulationResult
from img2mesh.io import export_glb
from img2mesh.mesh import clean_point_cloud, estimate_alpha, generate_mesh
from img2mesh.pipeline import (
    _camera_matrix,
    _indexed_points,
    _is_valid_observation,
    _make_intrinsics,
    _triangulate_track_multiview,
    bundle_adjust,
    reconstruct_image_set,
)


def make_ba_problem():
    rng = np.random.default_rng(42)
    camera = CameraIntrinsics.from_focal_length(640, 480, 700)

    rotations = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0],
        ]
    )
    translations = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.8, 0.0, 0.0],
            [-0.4, 0.02, 0.0],
        ]
    )
    points = np.column_stack(
        [
            rng.uniform(-1, 1, 12),
            rng.uniform(-0.7, 0.7, 12),
            rng.uniform(4, 8, 12),
        ]
    )

    camera_indices = []
    point_indices = []
    observations = []

    for camera_id in range(3):
        rotated = Rotation.from_rotvec(rotations[camera_id]).apply(points)
        camera_points = rotated + translations[camera_id]
        projected = np.column_stack(
            [
                camera.fx * camera_points[:, 0] / camera_points[:, 2] + camera.cx,
                camera.fy * camera_points[:, 1] / camera_points[:, 2] + camera.cy,
            ]
        )
        camera_indices.extend([camera_id] * len(points))
        point_indices.extend(range(len(points)))
        observations.extend(projected)

    initial_rotations = rotations.copy()
    initial_translations = translations.copy()
    initial_points = points.copy()

    initial_rotations[2] += np.array([0.01, -0.02, 0.015])
    initial_translations[2] += np.array([0.03, -0.02, 0.02])
    initial_points += rng.normal(0, 0.04, points.shape)

    return {
        "camera": camera,
        "rotations": rotations,
        "translations": translations,
        "points": points,
        "initial_rotations": initial_rotations,
        "initial_translations": initial_translations,
        "initial_points": initial_points,
        "camera_indices": np.asarray(camera_indices),
        "point_indices": np.asarray(point_indices),
        "observations": np.asarray(observations),
    }


def run_adjustment(problem):
    return bundle_adjust(
        problem["initial_rotations"],
        problem["initial_translations"],
        problem["initial_points"],
        problem["camera_indices"],
        problem["point_indices"],
        problem["observations"],
        problem["camera"],
    )


def test_bundle_adjustment_reduces_reprojection_error():
    problem = make_ba_problem()
    result = run_adjustment(problem)

    assert result.success
    assert result.final_rmse < result.initial_rmse
    assert result.final_rmse < 0.01


def test_first_two_cameras_remain_fixed():
    problem = make_ba_problem()
    result = run_adjustment(problem)

    np.testing.assert_array_equal(
        result.camera_rotations[:2],
        problem["initial_rotations"][:2],
    )
    np.testing.assert_array_equal(
        result.camera_translations[:2],
        problem["initial_translations"][:2],
    )


def test_result_dimensions():
    problem = make_ba_problem()
    result = run_adjustment(problem)

    assert result.camera_rotations.shape == (3, 3)
    assert result.camera_translations.shape == (3, 3)
    assert result.points_3d.shape == (12, 3)
    assert result.function_evaluations > 0


def test_two_camera_adjustment():
    problem = make_ba_problem()
    result = bundle_adjust(
        problem["initial_rotations"][:2],
        problem["initial_translations"][:2],
        problem["initial_points"],
        problem["camera_indices"][problem["camera_indices"] < 2],
        problem["point_indices"][problem["camera_indices"] < 2],
        problem["observations"][problem["camera_indices"] < 2],
        problem["camera"],
    )

    assert result.success
    assert result.final_rmse < result.initial_rmse


def test_invalid_robust_loss():
    problem = make_ba_problem()
    with pytest.raises(ConfigurationError):
        bundle_adjust(
            problem["initial_rotations"],
            problem["initial_translations"],
            problem["initial_points"],
            problem["camera_indices"],
            problem["point_indices"],
            problem["observations"],
            problem["camera"],
            robust_loss="invalid",
        )


def test_invalid_maximum_evaluations():
    problem = make_ba_problem()
    with pytest.raises(ConfigurationError):
        bundle_adjust(
            problem["initial_rotations"],
            problem["initial_translations"],
            problem["initial_points"],
            problem["camera_indices"],
            problem["point_indices"],
            problem["observations"],
            problem["camera"],
            max_nfev=0,
        )


def test_invalid_observation_indices():
    problem = make_ba_problem()
    camera_indices = problem["camera_indices"].copy()
    camera_indices[0] = 100

    with pytest.raises(ReconstructionError):
        bundle_adjust(
            problem["initial_rotations"],
            problem["initial_translations"],
            problem["initial_points"],
            camera_indices,
            problem["point_indices"],
            problem["observations"],
            problem["camera"],
        )


def test_unobserved_point_is_rejected():
    problem = make_ba_problem()
    points = np.vstack([problem["initial_points"], [[0.0, 0.0, 5.0]]])

    with pytest.raises(ReconstructionError):
        bundle_adjust(
            problem["initial_rotations"],
            problem["initial_translations"],
            points,
            problem["camera_indices"],
            problem["point_indices"],
            problem["observations"],
            problem["camera"],
        )


def test_zero_baseline_is_rejected():
    problem = make_ba_problem()
    translations = problem["initial_translations"].copy()
    translations[1] = 0

    with pytest.raises(ConfigurationError):
        bundle_adjust(
            problem["initial_rotations"],
            translations,
            problem["initial_points"],
            problem["camera_indices"],
            problem["point_indices"],
            problem["observations"],
            problem["camera"],
        )


def test_invalid_observation_shape():
    problem = make_ba_problem()
    with pytest.raises(ReconstructionError):
        bundle_adjust(
            problem["initial_rotations"],
            problem["initial_translations"],
            problem["initial_points"],
            problem["camera_indices"],
            problem["point_indices"],
            problem["observations"][:, :1],
            problem["camera"],
        )


def test_camera_matrix_conversion():
    camera = CameraIntrinsics(800, 810, 320, 240, 640, 480)
    assert np.allclose(
        _camera_matrix(camera),
        [
            [800, 0, 320],
            [0, 810, 240],
            [0, 0, 1],
        ],
    )


def test_default_intrinsics():
    camera = _make_intrinsics(640, 480, None)
    assert camera.fx == 640
    assert camera.fy == 640
    assert camera.image_width == 640
    assert camera.image_height == 480


def test_intrinsics_dimension_mismatch():
    camera = CameraIntrinsics(800, 800, 320, 240, 600, 480)
    with pytest.raises(ReconstructionError, match="dimensions"):
        _make_intrinsics(640, 480, camera)


def test_index_full_triangulation_array():
    result = TriangulationResult(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        np.array([True, False, True]),
    )
    mapped = _indexed_points(result, 3)
    assert set(mapped) == {0, 2}
    assert np.allclose(mapped[2], [7, 8, 9])


def test_index_compact_triangulation_array():
    result = TriangulationResult(
        np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]),
        np.array([True, False, True]),
    )
    mapped = _indexed_points(result, 3)
    assert set(mapped) == {0, 2}


def test_reject_single_image():
    with pytest.raises(ReconstructionError, match="At least two"):
        reconstruct_image_set([Image.new("RGB", (64, 64))])


def test_reject_mixed_image_dimensions():
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (80, 64))]
    with pytest.raises(ReconstructionError, match="same dimensions"):
        reconstruct_image_set(images)


def test_reject_empty_pair_graph(monkeypatch):
    graph = SimpleNamespace(pairwise_matches=())
    monkeypatch.setattr(pipeline_mod, "analyze_image_set", lambda *a, **k: graph)

    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
    with pytest.raises(ReconstructionError, match="No pairwise"):
        reconstruct_image_set(images)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_pair_matches": True},
        {"min_seed_points": True},
        {"pnp_min_inliers": True},
        {"run_bundle_adjustment": "no"},
    ],
)
def test_reject_boolean_or_invalid_pipeline_parameters(kwargs):
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
    with pytest.raises(ReconstructionError):
        reconstruct_image_set(images, **kwargs)


def test_is_valid_observation_filters_depth_and_reprojection():
    intrinsics = CameraIntrinsics.from_focal_length(640, 480, 800.0)
    pose = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
    point = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    center_xy = np.array([intrinsics.cx, intrinsics.cy], dtype=np.float64)

    assert _is_valid_observation(point, pose, intrinsics, center_xy)
    assert not _is_valid_observation(
        np.array([0.0, 0.0, -5.0]), pose, intrinsics, center_xy
    )
    assert not _is_valid_observation(
        point, pose, intrinsics, center_xy + np.array([50.0, 0.0])
    )


def test_reconstruct_three_view_scene_filters_outlier_observations(monkeypatch):
    rng = np.random.default_rng(123)
    width, height = 640, 480
    intrinsics = CameraIntrinsics.from_focal_length(width, height, 800.0)
    k_mat = intrinsics.matrix

    points_gt = np.column_stack(
        [
            rng.uniform(-1.2, 1.2, 24),
            rng.uniform(-0.9, 0.9, 24),
            rng.uniform(4.5, 7.5, 24),
        ]
    )

    poses_gt = [
        (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)),
        (
            cv2.Rodrigues(np.array([0.0, 0.03, 0.0], dtype=np.float64))[0],
            np.array([[-0.8], [0.0], [0.02]], dtype=np.float64),
        ),
        (
            cv2.Rodrigues(np.array([0.01, -0.03, 0.01], dtype=np.float64))[0],
            np.array([[0.7], [0.05], [0.03]], dtype=np.float64),
        ),
    ]

    keypoints_per_view = []
    for r_mat, t_vec in poses_gt:
        cam_pts = (r_mat @ points_gt.T + t_vec).T
        proj = (k_mat @ cam_pts.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        keypoints_per_view.append(
            tuple(cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in uv)
        )

    kp_view2 = list(keypoints_per_view[2])
    kp_view2[0] = cv2.KeyPoint(5.0, 5.0, 1.0)
    keypoints_per_view[2] = tuple(kp_view2)

    features = tuple(
        FeatureSet(
            method="orb",
            keypoints=kps,
            descriptors=None,
            image_size=(width, height),
        )
        for kps in keypoints_per_view
    )

    n_pts = len(points_gt)
    matches = tuple(cv2.DMatch(i, i, 1.0) for i in range(n_pts))
    pairwise = (
        PairwiseMatches(0, 1, matches),
        PairwiseMatches(0, 2, matches),
        PairwiseMatches(1, 2, matches),
    )
    tracks = tuple(
        FeatureTrack(
            (
                FeatureObservation(0, i),
                FeatureObservation(1, i),
                FeatureObservation(2, i),
            )
        )
        for i in range(n_pts)
    )

    graph = MultiViewFeatureGraph(
        features=features,
        pairwise_matches=pairwise,
        tracks=tracks,
        image_sizes=((width, height),) * 3,
        pairing_strategy="all",
    )

    monkeypatch.setattr(pipeline_mod, "analyze_image_set", lambda *a, **k: graph)

    images = [Image.new("RGB", (width, height)) for _ in range(3)]
    result = reconstruct_image_set(images, intrinsics=intrinsics)

    assert result.camera_count == 3
    assert result.point_count == n_pts
    assert result.bundle_adjustment_result is not None
    assert result.bundle_adjustment_result.success
    assert result.bundle_adjustment_result.final_rmse < 1.0


def cube_points():
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )


def test_clean_point_cloud_removes_duplicates():
    points = np.vstack([cube_points(), cube_points()[0]])
    cleaned = clean_point_cloud(points)
    assert len(cleaned) == 8
    assert np.unique(cleaned, axis=0).shape[0] == 8


def test_clean_point_cloud_removes_isolated_outlier():
    points = np.vstack([cube_points(), [[30, 30, 30]]])
    cleaned = clean_point_cloud(points, neighbors=4, std_ratio=2.0)
    assert len(cleaned) == 8
    assert not np.any(np.all(cleaned == [30, 30, 30], axis=1))


def test_clean_point_cloud_rejects_invalid_dimensions():
    with pytest.raises(ReconstructionError, match="shape"):
        clean_point_cloud(np.zeros((8, 2)))


def test_generate_convex_hull_mesh():
    mesh = generate_mesh(cube_points(), clean=False)

    assert mesh.vertex_count == 8
    assert mesh.face_count == 12
    assert mesh.faces.shape[1] == 3
    assert mesh.vertices.shape[1] == 3
    assert mesh.input_point_count == 8


def test_generate_alpha_shape_mesh():
    mesh = generate_mesh(cube_points(), alpha=2.0, clean=False)

    assert mesh.vertex_count == 8
    assert mesh.face_count == 12
    assert mesh.alpha == 2.0


def test_rejects_alpha_that_is_too_small():
    with pytest.raises(ReconstructionError, match="No mesh surface"):
        generate_mesh(cube_points(), alpha=0.01, clean=False)


def test_rejects_coplanar_point_cloud():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ReconstructionError, match="coplanar"):
        generate_mesh(points, clean=False)


def test_rejects_invalid_alpha():
    with pytest.raises(ReconstructionError, match="alpha"):
        generate_mesh(cube_points(), alpha=0, clean=False)

    with pytest.raises(ReconstructionError, match="alpha"):
        generate_mesh(cube_points(), alpha=True, clean=False)


def test_rejects_invalid_clean_and_std_ratio():
    with pytest.raises(ReconstructionError, match="clean"):
        generate_mesh(cube_points(), clean="no")

    with pytest.raises(ReconstructionError, match="std_ratio"):
        clean_point_cloud(cube_points(), std_ratio=True)


def test_bundle_adjustment_unfixed_second_camera_preserves_baseline_norm():
    problem = make_ba_problem()
    init_r = problem["initial_rotations"].copy()
    init_t = problem["initial_translations"].copy()
    init_r[1] += np.array([0.015, -0.02, 0.01])
    target_norm = np.linalg.norm(init_t[1])
    init_t[1] += np.array([0.0, 0.04, -0.03])
    init_t[1] = init_t[1] / np.linalg.norm(init_t[1]) * target_norm

    result = bundle_adjust(
        init_r,
        init_t,
        problem["initial_points"],
        problem["camera_indices"],
        problem["point_indices"],
        problem["observations"],
        problem["camera"],
        fix_second_camera=False,
    )

    assert result.success
    assert result.final_rmse < 1e-4
    np.testing.assert_allclose(result.camera_rotations[0], 0.0)
    np.testing.assert_allclose(result.camera_translations[0], 0.0)
    assert np.linalg.norm(result.camera_translations[1]) == pytest.approx(target_norm)


def test_multiview_track_triangulation_salvages_single_outlier_view():
    intrinsics = CameraIntrinsics.from_focal_length(640, 480, 700.0)
    gt_point = np.array([0.4, -0.3, 5.0], dtype=np.float64)
    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (
            cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))[0],
            np.array([[-0.8], [0.0], [0.05]]),
        ),
        2: (
            cv2.Rodrigues(np.array([0.04, -0.1, 0.01]))[0],
            np.array([[0.7], [0.05], [0.1]]),
        ),
        3: (
            cv2.Rodrigues(np.array([-0.03, 0.05, -0.02]))[0],
            np.array([[0.1], [-0.6], [0.05]]),
        ),
    }

    kps = []
    for view in range(4):
        r_mat, t_vec = poses[view]
        cam = (r_mat @ gt_point.reshape(3, 1) + t_vec).reshape(3)
        u = intrinsics.fx * cam[0] / cam[2] + intrinsics.cx
        v = intrinsics.fy * cam[1] / cam[2] + intrinsics.cy
        if view == 2:
            u += 90.0
            v -= 70.0
        kps.append(
            FeatureSet(
                method="orb",
                keypoints=(cv2.KeyPoint(float(u), float(v), 1.0),),
                descriptors=None,
                image_size=(640, 480),
            )
        )

    track = FeatureTrack(tuple(FeatureObservation(v, 0) for v in range(4)))
    est = _triangulate_track_multiview(track, poses, tuple(kps), intrinsics)
    assert est is not None
    assert np.linalg.norm(est - gt_point) < 1e-4


def test_end_to_end_rendered_multiview_accuracy_and_auto_alpha_mesh(tmp_path):
    width, height = 640, 480
    intrinsics = CameraIntrinsics.from_focal_length(width, height, 700.0)
    rng = np.random.default_rng(123)
    tex = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (3, 3), 0.7)
    for i in range(0, 256, 32):
        cv2.rectangle(tex, (i, i), (i + 16, i + 16), (255, 220, 100), -1)
        cv2.circle(tex, (255 - i, i), 6, (40, 200, 255), -1)

    quads = [
        np.array(
            [[-1.0, -0.8, 4.2], [0.2, -0.8, 4.2], [0.2, 0.8, 4.2], [-1.0, 0.8, 4.2]]
        ),
        np.array(
            [[0.2, -0.8, 4.2], [0.2, -0.8, 5.4], [0.2, 0.8, 5.4], [0.2, 0.8, 4.2]]
        ),
        np.array(
            [[0.2, -0.8, 5.4], [1.4, -0.8, 5.4], [1.4, 0.8, 5.4], [0.2, 0.8, 5.4]]
        ),
        np.array(
            [[-1.0, -0.8, 4.2], [-1.0, 0.8, 4.2], [-1.0, 0.8, 5.4], [-1.0, -0.8, 5.4]]
        ),
    ]
    cam_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.02, 0.0, 0.0]),  # Near-zero baseline trap with view 0
        np.array([0.7, 0.05, 0.1]),
        np.array([1.3, -0.05, 0.25]),
        np.array([-0.6, 0.08, 0.15]),
    ]
    target = np.array([0.1, 0.0, 4.8])
    images, poses_gt = [], []
    src_pts = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)

    for pos in cam_positions:
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
        down = np.array([0.0, 1.0, 0.0])
        right = np.cross(down, fwd)
        right /= np.linalg.norm(right)
        down_cam = np.cross(fwd, right)
        r_mat = np.column_stack([right, down_cam, fwd]).T
        t_vec = (-r_mat @ pos).reshape(3, 1)
        poses_gt.append((r_mat, t_vec))

        canvas = np.full((height, width, 3), 20, dtype=np.uint8)
        quads_sorted = sorted(
            quads,
            key=lambda q: np.mean((r_mat @ q.T + t_vec)[2]),
            reverse=True,
        )
        for q in quads_sorted:
            cam_q = (r_mat @ q.T + t_vec).T
            proj = (intrinsics.matrix @ cam_q.T).T
            dst_pts = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
            h_mat, _ = cv2.findHomography(src_pts, dst_pts)
            warped = cv2.warpPerspective(tex, h_mat, (width, height))
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
            canvas[mask > 0] = warped[mask > 0]
        images.append(Image.fromarray(canvas))

    result = reconstruct_image_set(images, intrinsics=intrinsics, method="orb")
    assert result.camera_count == 5
    assert result.seed_pair != (0, 1)
    assert result.point_count >= 500
    assert result.bundle_adjustment_result is not None
    assert result.bundle_adjustment_result.final_rmse < 0.6

    est_centers = np.array(
        [
            -Rotation.from_rotvec(result.camera_rotations[i]).as_matrix().T
            @ result.camera_translations[i]
            for i in range(result.camera_count)
        ]
    )
    gt_centers = np.array(
        [-poses_gt[v][0].T @ poses_gt[v][1].reshape(3) for v in result.view_order]
    )
    mu_e, mu_g = est_centers.mean(axis=0), gt_centers.mean(axis=0)
    u_mat, s_vals, vt_mat = np.linalg.svd(
        (gt_centers - mu_g).T @ (est_centers - mu_e)
    )
    det_sign = np.linalg.det(u_mat @ vt_mat)
    r_align = u_mat @ np.diag([1.0, 1.0, det_sign]) @ vt_mat
    scale = np.sum(s_vals * np.array([1.0, 1.0, det_sign])) / np.sum(
        (est_centers - mu_e) ** 2
    )
    aligned = scale * ((est_centers - mu_e) @ r_align.T) + mu_g
    center_errors = np.linalg.norm(aligned - gt_centers, axis=1)
    assert float(np.mean(center_errors)) < 0.02

    alpha_est = estimate_alpha(result.points_3d)
    assert alpha_est > 0
    mesh = generate_mesh(result.points_3d, auto_alpha=True)
    assert mesh.alpha is not None and mesh.alpha > 0
    assert mesh.vertex_count >= 100
    assert mesh.face_count >= 100
    glb_path = export_glb(mesh, tmp_path / "scene.glb")
    assert glb_path.is_file() and glb_path.stat().st_size > 0
