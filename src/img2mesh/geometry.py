from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import ConfigurationError, ReconstructionError
from .features import FeatureSet


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ConfigurationError(f"Camera parameter {name} must be finite.")

        if self.fx <= 0 or self.fy <= 0:
            raise ConfigurationError("Camera focal lengths must be positive.")

        for name in ("image_width", "image_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer.")

        if not (0 <= self.cx < self.image_width):
            raise ConfigurationError("Principal point cx must lie inside the image.")

        if not (0 <= self.cy < self.image_height):
            raise ConfigurationError("Principal point cy must lie inside the image.")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_focal_length(
        cls,
        image_width: int,
        image_height: int,
        focal_length_px: float,
    ) -> CameraIntrinsics:
        return cls(
            fx=focal_length_px,
            fy=focal_length_px,
            cx=(image_width - 1) / 2,
            cy=(image_height - 1) / 2,
            image_width=image_width,
            image_height=image_height,
        )

    def scaled_to(
        self,
        new_width: int,
        new_height: int,
    ) -> CameraIntrinsics:
        if (
            isinstance(new_width, bool)
            or not isinstance(new_width, int)
            or new_width <= 0
            or isinstance(new_height, bool)
            or not isinstance(new_height, int)
            or new_height <= 0
        ):
            raise ConfigurationError("New image dimensions must be positive integers.")

        scale_x = new_width / self.image_width
        scale_y = new_height / self.image_height

        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            image_width=new_width,
            image_height=new_height,
        )


@dataclass(frozen=True, slots=True)
class RelativePose:
    rotation: np.ndarray
    translation: np.ndarray
    essential_matrix: np.ndarray
    inlier_mask: np.ndarray

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


def matched_keypoint_coordinates(
    first: FeatureSet,
    second: FeatureSet,
    matches: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    if len(matches) == 0:
        raise ReconstructionError("No feature matches were provided.")

    points_first = []
    points_second = []

    for match in matches:
        query_index = match.queryIdx
        train_index = match.trainIdx

        if not (
            0 <= query_index < len(first.keypoints)
            and 0 <= train_index < len(second.keypoints)
        ):
            raise ReconstructionError(
                "A feature match contains an invalid keypoint index."
            )

        points_first.append(first.keypoints[query_index].pt)
        points_second.append(second.keypoints[train_index].pt)

    return (
        np.asarray(points_first, dtype=np.float32),
        np.asarray(points_second, dtype=np.float32),
    )


def _validate_correspondences(
    points_first: np.ndarray,
    points_second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(points_first, dtype=np.float64)
    second = np.asarray(points_second, dtype=np.float64)

    if first.ndim != 2 or first.shape[1] != 2:
        raise ReconstructionError("First correspondence array must have shape (N, 2).")

    if second.ndim != 2 or second.shape[1] != 2:
        raise ReconstructionError("Second correspondence array must have shape (N, 2).")

    if first.shape != second.shape:
        raise ReconstructionError("Correspondence arrays must have identical shapes.")

    if len(first) < 5:
        raise ReconstructionError("At least five point correspondences are required.")

    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ReconstructionError("Correspondences must contain finite coordinates.")

    return (
        np.ascontiguousarray(first, dtype=np.float64),
        np.ascontiguousarray(second, dtype=np.float64),
    )


def estimate_relative_pose(
    points_first: np.ndarray,
    points_second: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    ransac_threshold_px: float = 1.0,
    confidence: float = 0.999,
    max_iterations: int = 2000,
    min_inliers: int = 8,
) -> RelativePose:
    first, second = _validate_correspondences(points_first, points_second)

    if (
        isinstance(ransac_threshold_px, bool)
        or not isinstance(ransac_threshold_px, (int, float))
        or not math.isfinite(ransac_threshold_px)
        or ransac_threshold_px <= 0
    ):
        raise ConfigurationError("RANSAC threshold must be a positive finite number.")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ConfigurationError("Confidence must be greater than 0 and less than 1.")

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ConfigurationError("Maximum iterations must be a positive integer.")

    if (
        isinstance(min_inliers, bool)
        or not isinstance(min_inliers, int)
        or min_inliers < 5
    ):
        raise ConfigurationError("Minimum inliers must be an integer of at least 5.")

    try:
        essential, ransac_mask = cv2.findEssentialMat(
            first,
            second,
            intrinsics.matrix,
            cv2.RANSAC,
            confidence,
            ransac_threshold_px,
            max_iterations,
        )
    except cv2.error as exc:
        raise ReconstructionError(f"Essential matrix estimation failed: {exc}") from exc

    if essential is None or ransac_mask is None:
        raise ReconstructionError("Could not estimate an essential matrix.")

    try:
        essential = np.asarray(essential, dtype=np.float64)
        ransac_mask = np.asarray(ransac_mask).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(
            "Essential matrix or RANSAC mask has an invalid format."
        ) from exc

    if essential.ndim != 2 or essential.shape[1] != 3 or essential.shape[0] % 3 != 0:
        raise ReconstructionError(
            "Expected a single 3x3 essential matrix or vertically stacked "
            "3x3 candidates."
        )

    if ransac_mask.size != len(first):
        raise ReconstructionError("RANSAC mask must match the correspondence count.")

    if not np.isfinite(essential).all():
        raise ReconstructionError("Essential matrix contains non-finite values.")

    ransac_inlier_count = int(np.count_nonzero(ransac_mask))
    if ransac_inlier_count < min_inliers:
        raise ReconstructionError(
            "Not enough geometrically consistent feature matches."
        )

    candidates = [
        essential[start : start + 3] for start in range(0, essential.shape[0], 3)
    ]
    ransac_cv_mask = np.ascontiguousarray(ransac_mask.astype(np.uint8).reshape(-1, 1))

    best = None
    best_count = -1
    recovery_errors = []

    for candidate in candidates:
        if not np.isfinite(candidate).all():
            continue

        try:
            _, rotation, translation, pose_mask = cv2.recoverPose(
                candidate,
                first,
                second,
                intrinsics.matrix,
                mask=ransac_cv_mask.copy(),
            )
        except cv2.error as exc:
            recovery_errors.append(str(exc))
            continue

        if pose_mask is None or rotation is None or translation is None:
            continue

        pose_mask = np.asarray(pose_mask).reshape(-1)
        if pose_mask.size != len(first):
            continue

        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(-1)

        if (
            rotation.shape != (3, 3)
            or translation.size != 3
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
        ):
            continue

        valid_count = int(np.count_nonzero(pose_mask))
        if valid_count > best_count:
            best_count = valid_count
            best = (
                candidate.copy(),
                rotation.copy(),
                translation.reshape(3, 1).copy(),
                pose_mask.astype(bool),
            )

    if best is None or best_count < min_inliers:
        message = "Camera pose recovery produced too few valid inliers."
        if recovery_errors:
            message += " OpenCV: " + recovery_errors[0]
        raise ReconstructionError(message)

    selected_essential, rotation, translation, pose_mask = best

    return RelativePose(
        rotation=rotation,
        translation=translation,
        essential_matrix=selected_essential,
        inlier_mask=pose_mask,
    )


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    points_3d: np.ndarray
    valid_mask: np.ndarray

    @property
    def point_count(self) -> int:
        return len(self.points_3d)


def _refine_two_view_points(
    points_camera_one: np.ndarray,
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    depth_valid: np.ndarray,
    min_depth: float,
    iterations: int = 4,
) -> np.ndarray:
    """Refine two-view DLT points by minimizing geometric reprojection error."""
    indices = np.flatnonzero(depth_valid)
    if len(indices) == 0:
        return points_camera_one

    refined = points_camera_one.copy()
    pts = refined[indices]
    u1 = first_pixels[indices]
    u2 = second_pixels[indices]
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    t_flat = translation.reshape(1, 3)

    def _eval(x_pts):
        q1 = x_pts
        q2 = (rotation @ x_pts.T).T + t_flat
        z1 = q1[:, 2]
        z2 = q2[:, 2]
        valid = np.isfinite(q1).all(axis=1) & (z1 > min_depth) & (z2 > min_depth)
        err = np.full(len(x_pts), np.inf, dtype=np.float64)
        p1 = np.empty((len(x_pts), 2), dtype=np.float64)
        p2 = np.empty((len(x_pts), 2), dtype=np.float64)
        p1[valid, 0] = fx * q1[valid, 0] / z1[valid] + cx
        p1[valid, 1] = fy * q1[valid, 1] / z1[valid] + cy
        p2[valid, 0] = fx * q2[valid, 0] / z2[valid] + cx
        p2[valid, 1] = fy * q2[valid, 1] / z2[valid] + cy
        r1 = p1 - u1
        r2 = p2 - u2
        err[valid] = np.sum(r1[valid] ** 2, axis=1) + np.sum(r2[valid] ** 2, axis=1)
        return q1, q2, r1, r2, err, valid

    damping = 1e-4
    for _ in range(iterations):
        q1, q2, r1, r2, err, valid = _eval(pts)
        active = np.flatnonzero(valid)
        if len(active) == 0:
            break

        n = len(active)
        j_all = np.zeros((n, 4, 3), dtype=np.float64)
        z1 = q1[active, 2]
        z2 = q2[active, 2]

        j_all[:, 0, 0] = fx / z1
        j_all[:, 0, 2] = -fx * q1[active, 0] / (z1 * z1)
        j_all[:, 1, 1] = fy / z1
        j_all[:, 1, 2] = -fy * q1[active, 1] / (z1 * z1)

        j2_cam = np.zeros((n, 2, 3), dtype=np.float64)
        j2_cam[:, 0, 0] = fx / z2
        j2_cam[:, 0, 2] = -fx * q2[active, 0] / (z2 * z2)
        j2_cam[:, 1, 1] = fy / z2
        j2_cam[:, 1, 2] = -fy * q2[active, 1] / (z2 * z2)
        j_all[:, 2:, :] = j2_cam @ rotation

        res = np.concatenate((r1[active], r2[active]), axis=1)
        jtj = np.einsum("nji,njk->nik", j_all, j_all)
        jtj += damping * np.eye(3)[None, :, :]
        jtr = np.einsum("nji,nj->ni", j_all, res)

        try:
            step = -np.linalg.solve(jtj, jtr[:, :, None])[..., 0]
        except np.linalg.LinAlgError:
            break

        candidate = pts.copy()
        finite_step = np.isfinite(step).all(axis=1)
        candidate[active[finite_step]] += step[finite_step]
        _, _, _, _, new_err, new_valid = _eval(candidate)
        improved = new_valid & (new_err < err)
        if not np.any(improved):
            break
        pts[improved] = candidate[improved]

    refined[indices] = pts
    return refined


def triangulate_multiview_point(
    rotations: np.ndarray,
    translations: np.ndarray,
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float = 1e-6,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangulate a 3D point from M >= 2 calibrated views via DLT + Gauss-Newton."""
    view_count = len(rotations)
    if view_count < 2 or len(translations) != view_count or len(pixels) != view_count:
        return None

    k_mat = intrinsics.matrix
    design = np.empty((2 * view_count, 4), dtype=np.float64)

    for idx in range(view_count):
        proj = k_mat @ np.hstack((rotations[idx], translations[idx].reshape(3, 1)))
        row_x = pixels[idx, 0] * proj[2] - proj[0]
        row_y = pixels[idx, 1] * proj[2] - proj[1]
        norm_x = np.linalg.norm(row_x)
        norm_y = np.linalg.norm(row_y)
        design[2 * idx] = row_x / norm_x if norm_x > 1e-12 else row_x
        design[2 * idx + 1] = row_y / norm_y if norm_y > 1e-12 else row_y

    try:
        _, _, vt = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    homogeneous = vt[-1]
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) <= 1e-12:
        return None

    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        return None

    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    t_vecs = translations.reshape(view_count, 3)

    def _project_and_errors(x_pt: np.ndarray):
        cam = np.einsum("nij,j->ni", rotations, x_pt) + t_vecs
        z = cam[:, 2]
        if not np.isfinite(cam).all() or np.any(z <= min_depth):
            return None, None, None
        pred_x = fx * (cam[:, 0] / z) + cx
        pred_y = fy * (cam[:, 1] / z) + cy
        res = np.column_stack((pred_x - pixels[:, 0], pred_y - pixels[:, 1]))
        return cam, res, float(np.sum(res**2))

    cam, res, total_err = _project_and_errors(point)
    if cam is None:
        return None

    damping = 1e-4
    for _ in range(iterations):
        z = cam[:, 2]
        j_mat = np.empty((2 * view_count, 3), dtype=np.float64)
        for idx in range(view_count):
            inv_z = 1.0 / z[idx]
            inv_z2 = inv_z * inv_z
            j_cam = np.array(
                [
                    [fx * inv_z, 0.0, -fx * cam[idx, 0] * inv_z2],
                    [0.0, fy * inv_z, -fy * cam[idx, 1] * inv_z2],
                ],
                dtype=np.float64,
            )
            j_mat[2 * idx : 2 * idx + 2] = j_cam @ rotations[idx]

        lhs = j_mat.T @ j_mat + damping * np.eye(3)
        rhs = -(j_mat.T @ res.reshape(-1))
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break

        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-10:
            break

        candidate = point + delta
        new_cam, new_res, new_err = _project_and_errors(candidate)
        if new_cam is not None and new_err < total_err:
            point = candidate
            cam = new_cam
            res = new_res
            total_err = new_err
            damping = max(damping * 0.5, 1e-7)
        else:
            damping *= 10.0

    errors = np.linalg.norm(res, axis=1)
    return point, errors


def triangulate_points(
    points_first: np.ndarray,
    points_second: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: RelativePose,
    *,
    min_depth: float = 1e-6,
    max_reprojection_error_px: float = 3.0,
    min_parallax_degrees: float = 0.25,
) -> TriangulationResult:
    first, second = _validate_correspondences(points_first, points_second)

    parameters = (
        ("min_depth", min_depth, True),
        ("max_reprojection_error_px", max_reprojection_error_px, True),
        ("min_parallax_degrees", min_parallax_degrees, False),
    )

    for name, value, strictly_positive in parameters:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or (value <= 0 if strictly_positive else value < 0)
        ):
            raise ReconstructionError(f"{name} has an invalid value.")

    if min_parallax_degrees >= 90:
        raise ReconstructionError("min_parallax_degrees must be less than 90.")

    rotation = np.asarray(pose.rotation, dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64).reshape(-1)
    inlier_mask = np.asarray(pose.inlier_mask, dtype=bool).reshape(-1)

    if rotation.shape != (3, 3):
        raise ReconstructionError("Camera rotation must have shape (3, 3).")

    if translation.size != 3:
        raise ReconstructionError("Camera translation must contain three values.")

    if len(inlier_mask) != len(first):
        raise ReconstructionError(
            "Pose inlier mask must match the correspondence count."
        )

    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ReconstructionError("Camera pose must contain finite values.")

    inlier_indices = np.flatnonzero(inlier_mask)
    if len(inlier_indices) < 5:
        raise ReconstructionError("At least five geometric inliers are required.")

    first_inliers = first[inlier_indices]
    second_inliers = second[inlier_indices]

    translation = translation.reshape(3, 1)
    camera_matrix = intrinsics.matrix

    projection_first = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_second = camera_matrix @ np.hstack((rotation, translation))

    try:
        homogeneous = cv2.triangulatePoints(
            projection_first,
            projection_second,
            first_inliers.T,
            second_inliers.T,
        ).astype(np.float64)
    except cv2.error as exc:
        raise ReconstructionError(f"3D triangulation failed: {exc}") from exc

    denominators = homogeneous[3]
    finite = np.isfinite(denominators) & (np.abs(denominators) > 1e-12)

    safe_denominators = np.where(finite, denominators, 1.0)
    points_camera_one = (homogeneous[:3] / safe_denominators).T
    points_camera_two = (rotation @ points_camera_one.T + translation).T

    depth_valid = (
        finite
        & np.isfinite(points_camera_one).all(axis=1)
        & np.isfinite(points_camera_two).all(axis=1)
        & (points_camera_one[:, 2] > min_depth)
        & (points_camera_two[:, 2] > min_depth)
    )

    points_camera_one = _refine_two_view_points(
        points_camera_one,
        first_inliers,
        second_inliers,
        camera_matrix,
        rotation,
        translation,
        depth_valid,
        min_depth,
    )
    points_camera_two = (rotation @ points_camera_one.T + translation).T

    depth_valid = (
        finite
        & np.isfinite(points_camera_one).all(axis=1)
        & np.isfinite(points_camera_two).all(axis=1)
        & (points_camera_one[:, 2] > min_depth)
        & (points_camera_two[:, 2] > min_depth)
    )

    def project(points):
        projected_h = (camera_matrix @ points.T).T
        valid = np.isfinite(projected_h).all(axis=1) & (
            np.abs(projected_h[:, 2]) > 1e-12
        )
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        pixels[valid] = projected_h[valid, :2] / projected_h[valid, 2:3]
        return pixels, valid

    projected_first, valid_first = project(points_camera_one)
    projected_second, valid_second = project(points_camera_two)

    error_first = np.linalg.norm(projected_first - first_inliers, axis=1)
    error_second = np.linalg.norm(projected_second - second_inliers, axis=1)

    reprojection_valid = (
        valid_first
        & valid_second
        & np.isfinite(error_first)
        & np.isfinite(error_second)
        & (error_first <= max_reprojection_error_px)
        & (error_second <= max_reprojection_error_px)
    )

    camera_two_center = -rotation.T @ translation.reshape(3)
    ray_one = points_camera_one
    ray_two = points_camera_one - camera_two_center.reshape(1, 3)

    ray_one_norm = np.linalg.norm(ray_one, axis=1)
    ray_two_norm = np.linalg.norm(ray_two, axis=1)
    norm_product = ray_one_norm * ray_two_norm

    parallax_valid = norm_product > 1e-12
    cosine = np.ones(len(points_camera_one), dtype=np.float64)
    cosine[parallax_valid] = (
        np.sum(ray_one[parallax_valid] * ray_two[parallax_valid], axis=1)
        / norm_product[parallax_valid]
    )

    parallax_degrees = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    parallax_valid &= parallax_degrees >= min_parallax_degrees

    valid_inliers = depth_valid & reprojection_valid & parallax_valid
    valid_mask = np.zeros(len(first), dtype=bool)
    valid_mask[inlier_indices] = valid_inliers

    return TriangulationResult(
        points_3d=points_camera_one[valid_inliers],
        valid_mask=valid_mask,
    )


@dataclass(frozen=True, slots=True)
class CameraRegistrationResult:
    rotation_vector: NDArray[np.float64]
    translation_vector: NDArray[np.float64]
    rotation_matrix: NDArray[np.float64]
    inlier_indices: NDArray[np.int64]
    reprojection_rmse: float
    inlier_count: int
    total_correspondences: int


def register_camera_pnp(
    object_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    *,
    dist_coeffs: NDArray[np.float64] | None = None,
    reprojection_error: float = 4.0,
    confidence: float = 0.999,
    iterations: int = 2000,
    min_inliers: int = 6,
    refine: bool = True,
) -> CameraRegistrationResult:
    try:
        objects = np.asarray(object_points, dtype=np.float64)
        images = np.asarray(image_points, dtype=np.float64)
        intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(
            "Point arrays and camera matrix must contain numeric values."
        ) from exc

    if objects.ndim != 2 or objects.shape[1] != 3:
        raise ReconstructionError("object_points must have shape (N, 3).")

    if images.ndim != 2 or images.shape[1] != 2:
        raise ReconstructionError("image_points must have shape (N, 2).")

    if len(objects) != len(images):
        raise ReconstructionError(
            "object_points and image_points must contain the same number of rows."
        )

    if len(objects) < 4:
        raise ReconstructionError("PnP requires at least four correspondences.")

    if not np.isfinite(objects).all() or not np.isfinite(images).all():
        raise ReconstructionError("Point correspondences must contain finite values.")

    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ReconstructionError("camera_matrix must be a finite 3x3 matrix.")

    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ReconstructionError("Camera focal lengths must be positive.")

    if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ReconstructionError("camera_matrix has an invalid final row.")

    if not isinstance(min_inliers, int) or isinstance(min_inliers, bool):
        raise ReconstructionError("min_inliers must be an integer.")

    if min_inliers < 4 or min_inliers > len(objects):
        raise ReconstructionError(
            "min_inliers must be between 4 and the correspondence count."
        )

    if (
        isinstance(reprojection_error, bool)
        or not isinstance(reprojection_error, (int, float))
        or not np.isfinite(reprojection_error)
        or reprojection_error <= 0
    ):
        raise ReconstructionError("reprojection_error must be positive.")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not np.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ReconstructionError("confidence must be between 0 and 1.")

    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise ReconstructionError("iterations must be an integer.")

    if iterations < 1:
        raise ReconstructionError("iterations must be positive.")

    if not isinstance(refine, bool):
        raise ReconstructionError("refine must be True or False.")

    distortion = None
    if dist_coeffs is not None:
        try:
            distortion = np.asarray(dist_coeffs, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ReconstructionError(
                "dist_coeffs must contain numeric values."
            ) from exc

        if not np.isfinite(distortion).all():
            raise ReconstructionError("dist_coeffs must contain finite values.")

    objects = np.ascontiguousarray(objects)
    images = np.ascontiguousarray(images)
    intrinsics = np.ascontiguousarray(intrinsics)

    if distortion is not None:
        distortion = np.ascontiguousarray(distortion)

    try:
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            objectPoints=objects,
            imagePoints=images,
            cameraMatrix=intrinsics,
            distCoeffs=distortion,
            iterationsCount=iterations,
            reprojectionError=float(reprojection_error),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        raise ReconstructionError(f"OpenCV camera registration failed: {exc}") from exc

    if not success or rotation_vector is None or translation_vector is None:
        raise ReconstructionError("PnP could not estimate a camera pose.")

    if inliers is None:
        raise ReconstructionError("PnP did not identify any inliers.")

    inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    if len(inlier_indices) < min_inliers:
        raise ReconstructionError(
            f"Not enough reliable correspondences: "
            f"{len(inlier_indices)} found, {min_inliers} required."
        )

    rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)

    if refine and len(inlier_indices) >= 6:
        try:
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                objects[inlier_indices],
                images[inlier_indices],
                intrinsics,
                distortion,
                rotation_vector,
                translation_vector,
            )
        except cv2.error as exc:
            raise ReconstructionError(f"Camera pose refinement failed: {exc}") from exc

        rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
        translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(
            3, 1
        )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    camera_points = (
        rotation_matrix @ objects[inlier_indices].T
    ).T + translation_vector.reshape(1, 3)

    if not np.isfinite(camera_points).all():
        raise ReconstructionError("Camera registration produced invalid coordinates.")

    if np.any(camera_points[:, 2] <= 0):
        raise ReconstructionError(
            "Registered inlier points must lie in front of the camera."
        )

    try:
        projected, _ = cv2.projectPoints(
            objects[inlier_indices],
            rotation_vector,
            translation_vector,
            intrinsics,
            distortion,
        )
    except cv2.error as exc:
        raise ReconstructionError(
            f"Could not calculate reprojection error: {exc}"
        ) from exc

    projected = projected.reshape(-1, 2)
    differences = projected - images[inlier_indices]
    rmse = float(np.sqrt(np.mean(np.sum(differences**2, axis=1))))

    if not np.isfinite(rmse):
        raise ReconstructionError("Camera registration produced an invalid RMSE.")

    return CameraRegistrationResult(
        rotation_vector=rotation_vector.copy(),
        translation_vector=translation_vector.copy(),
        rotation_matrix=np.asarray(rotation_matrix).copy(),
        inlier_indices=np.sort(inlier_indices),
        reprojection_rmse=rmse,
        inlier_count=len(inlier_indices),
        total_correspondences=len(objects),
    )
