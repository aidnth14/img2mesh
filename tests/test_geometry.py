import cv2
import numpy as np
import pytest

from img2mesh.config import ConfigurationError, ReconstructionError
from img2mesh.features import FeatureSet
from img2mesh.geometry import (
    CameraIntrinsics,
    RelativePose,
    estimate_relative_pose,
    matched_keypoint_coordinates,
    register_camera_pnp,
    triangulate_points,
)
from img2mesh.pipeline import _relative_pose, _sampson_inlier_mask


def test_camera_matrix():
    camera = CameraIntrinsics(
        fx=800,
        fy=800,
        cx=320,
        cy=240,
        image_width=640,
        image_height=480,
    )
    expected = np.array(
        [
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(camera.matrix, expected)


def test_camera_from_focal_length():
    camera = CameraIntrinsics.from_focal_length(
        image_width=640,
        image_height=480,
        focal_length_px=800,
    )
    assert camera.fx == 800
    assert camera.fy == 800
    assert camera.cx == 319.5
    assert camera.cy == 239.5


def test_camera_scaling():
    camera = CameraIntrinsics(
        fx=800,
        fy=800,
        cx=320,
        cy=240,
        image_width=640,
        image_height=480,
    )
    scaled = camera.scaled_to(320, 240)
    assert scaled.fx == 400
    assert scaled.fy == 400
    assert scaled.cx == 160
    assert scaled.cy == 120


@pytest.mark.parametrize("focal_length", [0, -1, float("inf"), float("nan"), True])
def test_invalid_focal_length(focal_length):
    with pytest.raises(ConfigurationError):
        CameraIntrinsics(
            fx=focal_length,
            fy=800,
            cx=320,
            cy=240,
            image_width=640,
            image_height=480,
        )


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_camera_dimensions(dimension):
    with pytest.raises(ConfigurationError):
        CameraIntrinsics(
            fx=800,
            fy=800,
            cx=100,
            cy=100,
            image_width=dimension,
            image_height=480,
        )


def test_principal_point_must_be_inside_image():
    with pytest.raises(ConfigurationError):
        CameraIntrinsics(
            fx=800,
            fy=800,
            cx=700,
            cy=240,
            image_width=640,
            image_height=480,
        )


def test_scaled_camera_rejects_invalid_dimensions():
    camera = CameraIntrinsics(
        fx=800,
        fy=800,
        cx=320,
        cy=240,
        image_width=640,
        image_height=480,
    )
    with pytest.raises(ConfigurationError):
        camera.scaled_to(0, 240)


def synthetic_two_view_scene():
    rng = np.random.default_rng(42)
    camera = CameraIntrinsics.from_focal_length(640, 480, 800)

    points_3d = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, 120),
            rng.uniform(-1.0, 1.0, 120),
            rng.uniform(4.0, 9.0, 120),
        ]
    )

    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.03, -0.015], dtype=np.float64))
    translation = np.array([-0.7, 0.02, 0.05], dtype=np.float64).reshape(3, 1)

    first_camera = points_3d.T
    second_camera = rotation @ first_camera + translation

    def project(points):
        homogeneous = camera.matrix @ points
        return (homogeneous[:2] / homogeneous[2]).T

    first = project(first_camera)
    second = project(second_camera)
    second += rng.normal(0, 0.15, second.shape)

    outlier_indices = rng.choice(len(second), size=20, replace=False)
    second[outlier_indices] = rng.uniform(
        [0, 0],
        [640, 480],
        size=(len(outlier_indices), 2),
    )

    return camera, first, second, rotation, translation


def test_relative_pose_estimation():
    camera, first, second, expected_rotation, expected_translation = (
        synthetic_two_view_scene()
    )
    pose = estimate_relative_pose(first, second, camera, min_inliers=40)

    assert pose.rotation.shape == (3, 3)
    assert pose.translation.shape == (3, 1)
    assert pose.essential_matrix.shape == (3, 3)
    assert pose.inlier_count >= 40

    rotation_error = pose.rotation @ expected_rotation.T
    angle_error = np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1, 1))
    assert angle_error < np.deg2rad(2)

    estimated_direction = pose.translation.ravel()
    expected_direction = expected_translation.ravel()
    cosine_similarity = np.dot(estimated_direction, expected_direction) / (
        np.linalg.norm(estimated_direction) * np.linalg.norm(expected_direction)
    )
    assert cosine_similarity > 0.95


def test_too_few_correspondences():
    camera, first, second, _, _ = synthetic_two_view_scene()
    with pytest.raises(ReconstructionError):
        estimate_relative_pose(first[:4], second[:4], camera)


def test_mismatched_correspondence_arrays():
    camera, first, second, _, _ = synthetic_two_view_scene()
    with pytest.raises(ReconstructionError):
        estimate_relative_pose(first, second[:-1], camera)


def test_nonfinite_correspondences():
    camera, first, second, _, _ = synthetic_two_view_scene()
    first[0, 0] = np.nan
    with pytest.raises(ReconstructionError):
        estimate_relative_pose(first, second, camera)


def test_invalid_ransac_threshold():
    camera, first, second, _, _ = synthetic_two_view_scene()
    with pytest.raises(ConfigurationError):
        estimate_relative_pose(first, second, camera, ransac_threshold_px=-1)


def test_match_keypoint_coordinates():
    first = FeatureSet(
        method="orb",
        keypoints=(cv2.KeyPoint(10, 20, 1), cv2.KeyPoint(30, 40, 1)),
        descriptors=None,
        image_size=(100, 100),
    )
    second = FeatureSet(
        method="orb",
        keypoints=(cv2.KeyPoint(12, 22, 1), cv2.KeyPoint(32, 42, 1)),
        descriptors=None,
        image_size=(100, 100),
    )
    matches = (cv2.DMatch(0, 0, 0), cv2.DMatch(1, 1, 0))

    points_first, points_second = matched_keypoint_coordinates(first, second, matches)
    np.testing.assert_allclose(points_first, [[10, 20], [30, 40]])
    np.testing.assert_allclose(points_second, [[12, 22], [32, 42]])


def test_empty_feature_matches():
    first = FeatureSet("orb", (), None, (100, 100))
    second = FeatureSet("orb", (), None, (100, 100))
    with pytest.raises(ReconstructionError):
        matched_keypoint_coordinates(first, second, ())


def test_invalid_feature_match_index():
    first = FeatureSet("orb", (), None, (100, 100))
    second = FeatureSet("orb", (), None, (100, 100))
    with pytest.raises(ReconstructionError):
        matched_keypoint_coordinates(first, second, (cv2.DMatch(0, 0, 0),))


def _essential_inputs():
    points = np.array(
        [
            [40.0, 60.0],
            [180.0, 75.0],
            [250.0, 190.0],
            [120.0, 330.0],
            [370.0, 270.0],
        ],
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(700.0, 700.0, 320.0, 240.0, 640, 480)
    return points, points + 2.0, intrinsics


def test_selects_best_of_stacked_essential_matrices(monkeypatch):
    first, second, intrinsics = _essential_inputs()
    candidate_one = np.eye(3, dtype=np.float64)
    candidate_two = np.eye(3, dtype=np.float64) * 2.0
    stacked = np.vstack((candidate_one, candidate_two))
    ransac_mask = np.ones((5, 1), dtype=np.uint8)

    monkeypatch.setattr(
        cv2,
        "findEssentialMat",
        lambda *args, **kwargs: (stacked, ransac_mask),
    )

    calls = []

    def fake_recover_pose(essential, *args, mask):
        calls.append(essential.copy())
        if np.allclose(essential, candidate_one):
            result_mask = np.array([[1], [1], [1], [0], [0]], dtype=np.uint8)
            return 3, np.eye(3), np.array([[1.0], [0.0], [0.0]]), result_mask

        result_mask = np.ones((5, 1), dtype=np.uint8)
        return 5, np.eye(3), np.array([[1.0], [0.0], [0.0]]), result_mask

    monkeypatch.setattr(cv2, "recoverPose", fake_recover_pose)
    result = estimate_relative_pose(first, second, intrinsics, min_inliers=5)

    assert len(calls) == 2
    assert result.inlier_count == 5
    np.testing.assert_array_equal(result.essential_matrix, candidate_two)


def test_accepts_single_essential_matrix(monkeypatch):
    first, second, intrinsics = _essential_inputs()
    essential = np.eye(3, dtype=np.float64)
    ransac_mask = np.ones((5, 1), dtype=np.uint8)

    monkeypatch.setattr(
        cv2,
        "findEssentialMat",
        lambda *args, **kwargs: (essential, ransac_mask),
    )
    monkeypatch.setattr(
        cv2,
        "recoverPose",
        lambda matrix, *args, mask: (
            5,
            np.eye(3),
            np.array([[1.0], [0.0], [0.0]]),
            mask.copy(),
        ),
    )

    result = estimate_relative_pose(first, second, intrinsics, min_inliers=5)
    assert result.inlier_count == 5
    np.testing.assert_array_equal(result.essential_matrix, essential)


def test_rejects_all_candidates_with_insufficient_cheirality(monkeypatch):
    first, second, intrinsics = _essential_inputs()
    stacked = np.vstack((np.eye(3), np.eye(3) * 2.0))
    ransac_mask = np.ones((5, 1), dtype=np.uint8)

    monkeypatch.setattr(
        cv2,
        "findEssentialMat",
        lambda *args, **kwargs: (stacked, ransac_mask),
    )
    monkeypatch.setattr(
        cv2,
        "recoverPose",
        lambda matrix, *args, mask: (
            4,
            np.eye(3),
            np.array([[1.0], [0.0], [0.0]]),
            np.array([[1], [1], [1], [1], [0]], dtype=np.uint8),
        ),
    )

    with pytest.raises(ReconstructionError, match="too few valid inliers"):
        estimate_relative_pose(first, second, intrinsics, min_inliers=5)


def test_rejects_malformed_essential_matrix(monkeypatch):
    first, second, intrinsics = _essential_inputs()
    malformed = np.zeros((4, 3), dtype=np.float64)
    ransac_mask = np.ones((5, 1), dtype=np.uint8)

    monkeypatch.setattr(
        cv2,
        "findEssentialMat",
        lambda *args, **kwargs: (malformed, ransac_mask),
    )
    with pytest.raises(ReconstructionError, match="vertically stacked"):
        estimate_relative_pose(first, second, intrinsics, min_inliers=5)


def make_triangulation_scene():
    rng = np.random.default_rng(7)
    camera = CameraIntrinsics.from_focal_length(640, 480, 800)

    points = np.column_stack(
        [
            rng.uniform(-1, 1, 30),
            rng.uniform(-1, 1, 30),
            rng.uniform(4, 8, 30),
        ]
    )

    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.02, 0.0], dtype=np.float64))
    translation = np.array([-0.8, 0.0, 0.02], dtype=np.float64).reshape(3, 1)

    def project(points_camera):
        homogeneous = camera.matrix @ points_camera
        return (homogeneous[:2] / homogeneous[2]).T

    first_camera = points.T
    second_camera = rotation @ first_camera + translation
    first = project(first_camera)
    second = project(second_camera)

    pose = RelativePose(
        rotation=rotation,
        translation=translation,
        essential_matrix=np.zeros((3, 3)),
        inlier_mask=np.ones(len(points), dtype=bool),
    )
    return camera, points, first, second, pose


def test_triangulate_known_3d_points():
    camera, expected, first, second, pose = make_triangulation_scene()
    result = triangulate_points(first, second, camera, pose)

    assert result.point_count == len(expected)
    assert result.valid_mask.all()
    np.testing.assert_allclose(result.points_3d, expected, rtol=1e-5, atol=1e-5)


def test_valid_mask_preserves_input_correspondence_count():
    camera, _, first, second, pose = make_triangulation_scene()
    result = triangulate_points(first, second, camera, pose)

    assert len(result.valid_mask) == len(first)
    assert result.valid_mask.dtype == bool


def test_triangulation_rejects_invalid_pose_rotation():
    camera, _, first, second, pose = make_triangulation_scene()
    invalid_pose = RelativePose(
        rotation=np.eye(2),
        translation=pose.translation,
        essential_matrix=pose.essential_matrix,
        inlier_mask=pose.inlier_mask,
    )
    with pytest.raises(ReconstructionError):
        triangulate_points(first, second, camera, invalid_pose)


def test_triangulation_requires_enough_points():
    camera, _, first, second, pose = make_triangulation_scene()
    with pytest.raises(ReconstructionError):
        triangulate_points(first[:4], second[:4], camera, pose)


def test_invalid_minimum_depth():
    camera, _, first, second, pose = make_triangulation_scene()
    with pytest.raises(ReconstructionError):
        triangulate_points(first, second, camera, pose, min_depth=-1)


def test_points_behind_both_cameras_are_rejected():
    camera, _, first, second, pose = make_triangulation_scene()
    behind_first = first.copy()
    behind_second = second.copy()

    negative_z_points = np.column_stack(
        [
            np.linspace(-1, 1, 30),
            np.linspace(-1, 1, 30),
            np.full(30, -5.0),
        ]
    )

    first_camera = negative_z_points.T
    second_camera = pose.rotation @ first_camera + pose.translation

    def project(points_camera):
        homogeneous = camera.matrix @ points_camera
        return (homogeneous[:2] / homogeneous[2]).T

    behind_first[:] = project(first_camera)
    behind_second[:] = project(second_camera)

    result = triangulate_points(behind_first, behind_second, camera, pose)
    assert result.point_count == 0
    assert not result.valid_mask.any()


def test_ransac_outliers_are_excluded():
    camera, _, first, second, pose = make_triangulation_scene()
    mask = pose.inlier_mask.copy()
    mask[0] = False

    filtered_pose = RelativePose(
        rotation=pose.rotation,
        translation=pose.translation,
        essential_matrix=pose.essential_matrix,
        inlier_mask=mask,
    )
    result = triangulate_points(first, second, camera, filtered_pose)
    assert not result.valid_mask[0]
    assert result.point_count == len(first) - 1


def test_pose_mask_must_match_correspondence_count():
    camera, _, first, second, pose = make_triangulation_scene()
    invalid_pose = RelativePose(
        rotation=pose.rotation,
        translation=pose.translation,
        essential_matrix=pose.essential_matrix,
        inlier_mask=pose.inlier_mask[:-1],
    )
    with pytest.raises(ReconstructionError):
        triangulate_points(first, second, camera, invalid_pose)


def _synthetic_pair(baseline=1.0):
    intrinsics = CameraIntrinsics(700.0, 700.0, 320.0, 240.0, 640, 480)
    points_3d = np.array(
        [
            [-0.8, -0.4, 4.0],
            [-0.3, 0.2, 4.5],
            [0.1, -0.2, 5.0],
            [0.4, 0.3, 5.5],
            [0.8, -0.1, 6.0],
            [0.2, 0.5, 6.5],
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([[-baseline], [0.0], [0.0]], dtype=np.float64)
    camera_matrix = intrinsics.matrix

    def project(points):
        camera_points = (rotation @ points.T + translation).T
        pixels_h = (camera_matrix @ camera_points.T).T
        return pixels_h[:, :2] / pixels_h[:, 2:3]

    first = (camera_matrix @ points_3d.T).T
    first = first[:, :2] / first[:, 2:3]
    second = project(points_3d)

    tx, ty, tz = translation.reshape(3)
    skew = np.array(
        [
            [0.0, -tz, ty],
            [tz, 0.0, -tx],
            [-ty, tx, 0.0],
        ],
        dtype=np.float64,
    )
    pose = RelativePose(
        rotation=rotation,
        translation=translation,
        essential_matrix=skew @ rotation,
        inlier_mask=np.ones(len(points_3d), dtype=bool),
    )
    return intrinsics, points_3d, first, second, pose, translation


def test_triangulation_recovers_clean_points():
    intrinsics, expected, first, second, pose, _ = _synthetic_pair()
    result = triangulate_points(first, second, intrinsics, pose)

    assert result.point_count == len(expected)
    assert result.valid_mask.all()
    np.testing.assert_allclose(result.points_3d, expected, atol=1e-5)


def test_triangulation_rejects_large_reprojection_outlier():
    intrinsics, expected, first, second, pose, _ = _synthetic_pair()
    corrupted = second.copy()
    corrupted[-1] += np.array([55.0, -40.0])

    result = triangulate_points(first, corrupted, intrinsics, pose)
    assert result.point_count == len(expected) - 1
    assert not result.valid_mask[-1]
    assert result.valid_mask[:-1].all()


def test_triangulation_rejects_low_parallax_points():
    intrinsics, _, first, second, pose, _ = _synthetic_pair(baseline=1e-4)
    result = triangulate_points(first, second, intrinsics, pose)

    assert result.point_count == 0
    assert not result.valid_mask.any()


def test_triangulation_preserves_original_pose_inlier_mask():
    intrinsics, expected, first, second, pose, _ = _synthetic_pair()
    mask = pose.inlier_mask.copy()
    mask[2] = False

    filtered_pose = RelativePose(
        rotation=pose.rotation,
        translation=pose.translation,
        essential_matrix=pose.essential_matrix,
        inlier_mask=mask,
    )
    result = triangulate_points(first, second, intrinsics, filtered_pose)
    assert result.point_count == len(expected) - 1
    assert not result.valid_mask[2]


def test_sampson_filter_rejects_inconsistent_epipolar_match():
    intrinsics, _, first, second, _, translation = _synthetic_pair()
    relative = _relative_pose(
        (np.eye(3), np.zeros((3, 1))),
        (np.eye(3), translation),
        len(first),
    )
    corrupted = second.copy()
    corrupted[-1, 1] += 60.0

    mask = _sampson_inlier_mask(
        first,
        corrupted,
        relative,
        intrinsics,
        threshold_px=2.5,
    )
    assert mask[:-1].all()
    assert not mask[-1]


def make_pnp_correspondences(count=40, seed=21):
    rng = np.random.default_rng(seed)
    objects = np.column_stack(
        (
            rng.uniform(-1.2, 1.2, count),
            rng.uniform(-0.9, 0.9, count),
            rng.uniform(-0.5, 0.8, count),
        )
    ).astype(np.float64)

    matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_rotation = np.array([[0.05], [-0.08], [0.025]], dtype=np.float64)
    expected_translation = np.array([[0.15], [-0.12], [5.5]], dtype=np.float64)

    images, _ = cv2.projectPoints(
        objects,
        expected_rotation,
        expected_translation,
        matrix,
        None,
    )
    return (
        objects,
        images.reshape(-1, 2),
        matrix,
        expected_rotation,
        expected_translation,
    )


def test_recovers_known_camera_pose():
    objects, images, matrix, expected_r, expected_t = make_pnp_correspondences()
    result = register_camera_pnp(objects, images, matrix)

    assert result.inlier_count == len(objects)
    assert result.total_correspondences == len(objects)
    assert result.reprojection_rmse < 1e-3
    assert np.allclose(result.rotation_vector, expected_r, atol=1e-3)
    assert np.allclose(result.translation_vector, expected_t, atol=1e-3)
    assert result.rotation_matrix.shape == (3, 3)


def test_ransac_rejects_outlier_correspondences():
    objects, images, matrix, _, _ = make_pnp_correspondences(count=50)
    rng = np.random.default_rng(42)
    images[:10] = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(10, 2))

    result = register_camera_pnp(
        objects,
        images,
        matrix,
        reprojection_error=2.0,
        min_inliers=20,
    )
    assert result.inlier_count >= 35
    assert result.inlier_count < len(objects)
    assert not np.any(np.isin(result.inlier_indices, np.arange(10)))
    assert result.reprojection_rmse < 1.0


def test_accepts_distortion_coefficients():
    objects, images, matrix, _, _ = make_pnp_correspondences()
    result = register_camera_pnp(
        objects,
        images,
        matrix,
        dist_coeffs=np.zeros(5, dtype=np.float64),
    )
    assert result.inlier_count == len(objects)


def test_rejects_mismatched_correspondence_counts():
    objects, images, matrix, _, _ = make_pnp_correspondences()
    with pytest.raises(ReconstructionError, match="same number"):
        register_camera_pnp(objects, images[:-1], matrix)


def test_rejects_too_few_correspondences():
    objects, images, matrix, _, _ = make_pnp_correspondences(count=3)
    with pytest.raises(ReconstructionError, match="at least four"):
        register_camera_pnp(objects, images, matrix)


def test_rejects_invalid_camera_matrix():
    objects, images, matrix, _, _ = make_pnp_correspondences()
    matrix[0, 0] = -1.0
    with pytest.raises(ReconstructionError, match="focal lengths"):
        register_camera_pnp(objects, images, matrix)


def test_rejects_non_finite_points():
    objects, images, matrix, _, _ = make_pnp_correspondences()
    objects[0, 0] = np.nan
    with pytest.raises(ReconstructionError, match="finite"):
        register_camera_pnp(objects, images, matrix)


def test_rejects_invalid_registration_settings():
    objects, images, matrix, _, _ = make_pnp_correspondences()

    with pytest.raises(ReconstructionError, match="confidence"):
        register_camera_pnp(objects, images, matrix, confidence=1.0)

    with pytest.raises(ReconstructionError, match="iterations"):
        register_camera_pnp(objects, images, matrix, iterations=0)

    with pytest.raises(ReconstructionError, match="min_inliers"):
        register_camera_pnp(objects, images, matrix, min_inliers=3)

    with pytest.raises(ReconstructionError, match="reprojection_error"):
        register_camera_pnp(objects, images, matrix, reprojection_error=True)

    with pytest.raises(ReconstructionError, match="confidence"):
        register_camera_pnp(objects, images, matrix, confidence=True)

    with pytest.raises(ReconstructionError, match="refine"):
        register_camera_pnp(objects, images, matrix, refine="yes")
