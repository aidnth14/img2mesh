from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .config import ConfigurationError, ReconstructionError
from .features import FeatureTrack, MultiViewFeatureGraph, analyze_image_set
from .geometry import (
    CameraIntrinsics,
    RelativePose,
    TriangulationResult,
    estimate_relative_pose,
    matched_keypoint_coordinates,
    register_camera_pnp,
    triangulate_multiview_point,
    triangulate_points,
)


@dataclass(frozen=True, slots=True)
class BundleAdjustmentResult:
    camera_rotations: np.ndarray
    camera_translations: np.ndarray
    points_3d: np.ndarray
    initial_rmse: float
    final_rmse: float
    success: bool
    message: str
    function_evaluations: int


def _validate_inputs(
    camera_rotations,
    camera_translations,
    points_3d,
    camera_indices,
    point_indices,
    observations_2d,
):
    rotations = np.asarray(camera_rotations, dtype=np.float64)
    translations = np.asarray(camera_translations, dtype=np.float64)
    points = np.asarray(points_3d, dtype=np.float64)
    observations = np.asarray(observations_2d, dtype=np.float64)

    raw_camera_indices = np.asarray(camera_indices)
    raw_point_indices = np.asarray(point_indices)

    for name, indices in (
        ("camera_indices", raw_camera_indices),
        ("point_indices", raw_point_indices),
    ):
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ReconstructionError(
                f"{name} must be a one-dimensional integer array."
            )

    camera_ids = raw_camera_indices.astype(np.int64)
    point_ids = raw_point_indices.astype(np.int64)

    if rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ReconstructionError("Camera rotations must have shape (N, 3).")

    if translations.shape != rotations.shape:
        raise ReconstructionError(
            "Camera translations must have the same shape as rotations."
        )

    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ReconstructionError(
            "3D points must have shape (N, 3) with at least one point."
        )

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ReconstructionError("2D observations must have shape (N, 2).")

    observation_count = len(observations)

    if (
        len(rotations) < 2
        or len(camera_ids) != observation_count
        or len(point_ids) != observation_count
        or observation_count < 4
    ):
        raise ReconstructionError(
            "Bundle adjustment requires at least two cameras "
            "and four valid observations."
        )

    for name, array in (
        ("camera rotations", rotations),
        ("camera translations", translations),
        ("3D points", points),
        ("2D observations", observations),
    ):
        if not np.isfinite(array).all():
            raise ReconstructionError(f"{name} must contain only finite values.")

    if (
        np.any(camera_ids < 0)
        or np.any(camera_ids >= len(rotations))
        or np.any(point_ids < 0)
        or np.any(point_ids >= len(points))
    ):
        raise ReconstructionError(
            "Observation indices reference an invalid camera or point."
        )

    for point_id in range(len(points)):
        observing_cameras = np.unique(camera_ids[point_ids == point_id])
        if len(observing_cameras) < 2:
            raise ReconstructionError(
                f"Point {point_id} must be observed by at least two different cameras."
            )

    if not np.allclose(rotations[0], 0, atol=1e-8):
        raise ConfigurationError("The first camera must define the identity rotation.")

    if not np.allclose(translations[0], 0, atol=1e-8):
        raise ConfigurationError("The first camera must be located at the origin.")

    if np.linalg.norm(translations[1]) < 1e-8:
        raise ConfigurationError(
            "The second camera must have a non-zero baseline "
            "to establish reconstruction scale."
        )

    return (
        rotations.copy(),
        translations.copy(),
        points.copy(),
        camera_ids,
        point_ids,
        observations,
    )


def _skew_matrices(vectors: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) skew-symmetric matrices for an (N, 3) array."""
    skew = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def _so3_left_jacobians(rotvecs: np.ndarray) -> np.ndarray:
    """Compute exact SO(3) left Jacobians for an (N, 3) array of rotvecs."""
    count = len(rotvecs)
    angles = np.linalg.norm(rotvecs, axis=1)
    k_mats = _skew_matrices(rotvecs)
    k_sq = k_mats @ k_mats
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), (count, 3, 3)).copy()

    small = angles < 1e-6
    coef1 = np.empty(count, dtype=np.float64)
    coef2 = np.empty(count, dtype=np.float64)

    coef1[small] = 0.5 - (angles[small] ** 2) / 24.0
    coef2[small] = (1.0 / 6.0) - (angles[small] ** 2) / 120.0

    nz = ~small
    if np.any(nz):
        th = angles[nz]
        th2 = th * th
        coef1[nz] = (1.0 - np.cos(th)) / th2
        coef2[nz] = (th - np.sin(th)) / (th2 * th)

    return eye + coef1[:, None, None] * k_mats + coef2[:, None, None] * k_sq


def _tangent_basis(vector: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (norm, unit_dir, (3, 2) orthonormal tangent basis) for vector."""
    norm = float(np.linalg.norm(vector))
    unit = vector / norm
    if abs(unit[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(unit, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    e2 /= np.linalg.norm(e2)
    return norm, unit, np.column_stack((e1, e2))


def bundle_adjust(
    camera_rotations: np.ndarray,
    camera_translations: np.ndarray,
    points_3d: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    observations_2d: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 200,
    fix_second_camera: bool = True,
) -> BundleAdjustmentResult:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise ConfigurationError("intrinsics must be a CameraIntrinsics instance.")

    if robust_loss not in {
        "linear",
        "soft_l1",
        "huber",
        "cauchy",
        "arctan",
    }:
        raise ConfigurationError("Unsupported robust loss function.")

    if (
        isinstance(f_scale, bool)
        or not isinstance(f_scale, (int, float))
        or not np.isfinite(f_scale)
        or f_scale <= 0
    ):
        raise ConfigurationError("f_scale must be a positive finite number.")

    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ConfigurationError("max_nfev must be a positive integer.")

    if not isinstance(fix_second_camera, bool):
        raise ConfigurationError("fix_second_camera must be True or False.")

    (
        initial_rotations,
        initial_translations,
        initial_points,
        camera_ids,
        point_ids,
        observations,
    ) = _validate_inputs(
        camera_rotations,
        camera_translations,
        points_3d,
        camera_indices,
        point_indices,
        observations_2d,
    )

    camera_count = len(initial_rotations)
    point_count = len(initial_points)
    obs_count = len(observations)

    t1_norm, t1_dir, t1_basis = _tangent_basis(initial_translations[1])
    cam1_param_count = 0 if fix_second_camera else 5
    other_cam_param_count = max(0, camera_count - 2) * 6
    pose_parameter_count = cam1_param_count + other_cam_param_count

    other_pose_params = np.column_stack(
        (
            initial_rotations[2:],
            initial_translations[2:],
        )
    ).ravel()

    if fix_second_camera:
        x0 = np.concatenate((other_pose_params, initial_points.ravel()))
    else:
        cam1_init = np.concatenate(
            (initial_rotations[1], np.zeros(2, dtype=np.float64))
        )
        x0 = np.concatenate((cam1_init, other_pose_params, initial_points.ravel()))

    total_params = len(x0)

    def unpack(parameters):
        rotations = initial_rotations.copy()
        translations = initial_translations.copy()

        if not fix_second_camera:
            rotations[1] = parameters[:3]
            w = t1_dir + t1_basis @ parameters[3:5]
            translations[1] = (t1_norm / np.linalg.norm(w)) * w

        if camera_count > 2:
            camera_parameters = parameters[
                cam1_param_count:pose_parameter_count
            ].reshape(-1, 6)
            rotations[2:] = camera_parameters[:, :3]
            translations[2:] = camera_parameters[:, 3:]

        points = parameters[pose_parameter_count:].reshape(point_count, 3)
        return rotations, translations, points

    def calculate_residuals(parameters):
        rotations, translations, points = unpack(parameters)

        rotation_matrices = Rotation.from_rotvec(rotations).as_matrix()
        camera_points = (
            np.einsum(
                "nij,nj->ni",
                rotation_matrices[camera_ids],
                points[point_ids],
            )
            + translations[camera_ids]
        )

        depth = camera_points[:, 2]
        safe_depth = np.maximum(depth, 1e-8)

        predicted = np.column_stack(
            (
                intrinsics.fx * camera_points[:, 0] / safe_depth + intrinsics.cx,
                intrinsics.fy * camera_points[:, 1] / safe_depth + intrinsics.cy,
            )
        )

        pixel_residuals = predicted - observations
        depth_penalties = np.minimum(depth - 1e-8, 0.0) * 100.0

        return np.concatenate((pixel_residuals.ravel(), depth_penalties))

    def calculate_jacobian(parameters):
        rotations, translations, points = unpack(parameters)
        rotation_matrices = Rotation.from_rotvec(rotations).as_matrix()
        left_jacs = _so3_left_jacobians(rotations)

        rotated_points = np.einsum(
            "nij,nj->ni",
            rotation_matrices[camera_ids],
            points[point_ids],
        )
        camera_points = rotated_points + translations[camera_ids]

        depth = camera_points[:, 2]
        clamped = depth <= 1e-8
        safe_depth = np.maximum(depth, 1e-8)
        inv_z = 1.0 / safe_depth
        inv_z2 = inv_z * inv_z

        # d(u, v) / d(q) with shape (M, 2, 3)
        d_uv_dq = np.zeros((obs_count, 2, 3), dtype=np.float64)
        d_uv_dq[:, 0, 0] = intrinsics.fx * inv_z
        d_uv_dq[:, 1, 1] = intrinsics.fy * inv_z
        active = ~clamped
        d_uv_dq[active, 0, 2] = (
            -intrinsics.fx * camera_points[active, 0] * inv_z2[active]
        )
        d_uv_dq[active, 1, 2] = (
            -intrinsics.fy * camera_points[active, 1] * inv_z2[active]
        )

        # d(q) / d(rvec) = -[R X]_x J_l(rvec)
        d_q_dr = -_skew_matrices(rotated_points) @ left_jacs[camera_ids]

        row_chunks = []
        col_chunks = []
        val_chunks = []

        obs_idx = np.arange(obs_count, dtype=np.int64)
        pixel_rows = np.column_stack((2 * obs_idx, 2 * obs_idx + 1))

        def _append_3d_block(mask, col_offsets, jac_3x3):
            if not np.any(mask):
                return
            sel_obs = obs_idx[mask]
            sel_rows = pixel_rows[sel_obs]
            sel_cols = col_offsets[mask]
            d_uv_dp = d_uv_dq[mask] @ jac_3x3

            # Pixel rows (2 x 3 per observation)
            r_pix = np.repeat(sel_rows, 3, axis=1).ravel()
            c_pix = np.tile(sel_cols[:, None] + np.arange(3), (1, 2)).ravel()
            row_chunks.append(r_pix)
            col_chunks.append(c_pix)
            val_chunks.append(d_uv_dp.reshape(-1))

            # Depth penalty row for clamped points
            clamped_sub = clamped[mask]
            if np.any(clamped_sub):
                d_rows = 2 * obs_count + sel_obs[clamped_sub]
                r_dep = np.repeat(d_rows, 3)
                c_dep = (sel_cols[clamped_sub, None] + np.arange(3)).ravel()
                v_dep = (100.0 * jac_3x3[clamped_sub, 2, :]).ravel()
                row_chunks.append(r_dep)
                col_chunks.append(c_dep)
                val_chunks.append(v_dep)

        # Point blocks (all observations)
        point_cols = pose_parameter_count + point_ids * 3
        _append_3d_block(
            np.ones(obs_count, dtype=bool),
            point_cols,
            rotation_matrices[camera_ids],
        )

        # Cameras >= 2 blocks
        cam2_mask = camera_ids >= 2
        if np.any(cam2_mask):
            cam_rot_cols = cam1_param_count + (camera_ids - 2) * 6
            _append_3d_block(cam2_mask, cam_rot_cols, d_q_dr[cam2_mask])
            eye3 = np.broadcast_to(
                np.eye(3, dtype=np.float64),
                (int(np.count_nonzero(cam2_mask)), 3, 3),
            )
            _append_3d_block(cam2_mask, cam_rot_cols + 3, eye3)

        # Camera 1 block (when unfixed)
        if not fix_second_camera:
            cam1_mask = camera_ids == 1
            if np.any(cam1_mask):
                cam1_rot_cols = np.zeros(obs_count, dtype=np.int64)
                _append_3d_block(cam1_mask, cam1_rot_cols, d_q_dr[cam1_mask])

                w = t1_dir + t1_basis @ parameters[3:5]
                w_norm = float(np.linalg.norm(w))
                proj_mat = (t1_norm / w_norm) * (
                    np.eye(3, dtype=np.float64) - np.outer(w, w) / (w_norm * w_norm)
                )
                d_t1_dab = proj_mat @ t1_basis  # (3, 2)

                sel_obs = obs_idx[cam1_mask]
                n1 = len(sel_obs)
                d_uv_dab = d_uv_dq[cam1_mask] @ d_t1_dab[None, :, :]  # (n1, 2, 2)
                r_pix = np.repeat(pixel_rows[sel_obs], 2, axis=1).ravel()
                c_pix = np.tile(np.array([3, 4], dtype=np.int64), (n1, 2)).ravel()
                row_chunks.append(r_pix)
                col_chunks.append(c_pix)
                val_chunks.append(d_uv_dab.reshape(-1))

                clamped1 = clamped[cam1_mask]
                if np.any(clamped1):
                    d_rows = 2 * obs_count + sel_obs[clamped1]
                    r_dep = np.repeat(d_rows, 2)
                    c_dep = np.tile(np.array([3, 4], dtype=np.int64), len(d_rows))
                    v_dep = np.tile(100.0 * d_t1_dab[2, :], len(d_rows))
                    row_chunks.append(r_dep)
                    col_chunks.append(c_dep)
                    val_chunks.append(v_dep)

        return coo_matrix(
            (
                np.concatenate(val_chunks),
                (np.concatenate(row_chunks), np.concatenate(col_chunks)),
            ),
            shape=(obs_count * 3, total_params),
        ).tocsr()

    def pixel_rmse(parameters):
        residuals = calculate_residuals(parameters)
        pixel_values = residuals[: len(observations) * 2]
        return float(np.sqrt(np.mean(pixel_values**2)))

    initial_rmse = pixel_rmse(x0)

    try:
        optimized = least_squares(
            calculate_residuals,
            x0,
            jac=calculate_jacobian,
            method="trf",
            tr_solver="lsmr",
            x_scale="jac",
            loss=robust_loss,
            f_scale=float(f_scale),
            max_nfev=max_nfev,
        )
    except (ValueError, FloatingPointError) as exc:
        raise ReconstructionError(f"Bundle adjustment failed: {exc}") from exc

    if not np.isfinite(optimized.x).all():
        raise ReconstructionError("Bundle adjustment produced non-finite parameters.")

    final_rotations, final_translations, final_points = unpack(optimized.x)

    return BundleAdjustmentResult(
        camera_rotations=final_rotations,
        camera_translations=final_translations,
        points_3d=final_points,
        initial_rmse=initial_rmse,
        final_rmse=pixel_rmse(optimized.x),
        success=bool(optimized.success),
        message=str(optimized.message),
        function_evaluations=int(optimized.nfev),
    )


@dataclass(frozen=True, slots=True)
class MultiViewReconstructionResult:
    seed_pair: tuple[int, int]
    view_order: tuple[int, ...]
    camera_rotations: np.ndarray
    camera_translations: np.ndarray
    points_3d: np.ndarray
    point_track_indices: np.ndarray
    intrinsics: CameraIntrinsics
    feature_graph: MultiViewFeatureGraph
    bundle_adjustment_result: BundleAdjustmentResult | None

    @property
    def camera_count(self) -> int:
        return len(self.view_order)

    @property
    def point_count(self) -> int:
        return len(self.points_3d)


def _make_intrinsics(width, height, intrinsics):
    if intrinsics is None:
        focal = float(max(width, height))
        return CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
            image_width=width,
            image_height=height,
        )

    if (intrinsics.image_width, intrinsics.image_height) != (width, height):
        raise ReconstructionError(
            "Camera intrinsics dimensions must match the input images."
        )

    values = np.array([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy])
    if not np.isfinite(values).all() or intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ReconstructionError("Camera intrinsics are invalid.")

    return intrinsics


def _camera_matrix(intrinsics):
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _indexed_points(result: TriangulationResult, count: int):
    mask = np.asarray(result.valid_mask, dtype=bool).reshape(-1)
    points = np.asarray(result.points_3d, dtype=np.float64)

    if len(mask) != count or points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionError("Invalid triangulation result dimensions.")

    indices = np.flatnonzero(mask)

    if len(points) == count:
        selected = points[indices]
    elif len(points) == len(indices):
        selected = points
    else:
        raise ReconstructionError("Triangulation mask and points do not match.")

    return {
        int(index): point.copy()
        for index, point in zip(indices, selected)
        if np.isfinite(point).all()
    }


def _track_lookup(tracks: tuple[FeatureTrack, ...]):
    lookup = {}
    for track_id, track in enumerate(tracks):
        for obs in track.observations:
            key = (int(obs.view_index), int(obs.keypoint_index))
            if key in lookup and lookup[key] != track_id:
                raise ReconstructionError(
                    "Feature observation belongs to multiple tracks."
                )
            lookup[key] = track_id
    return lookup


def _match_tracks(pair, lookup):
    ids = []
    for match in pair.matches:
        first = lookup.get((pair.first_view, int(match.queryIdx)))
        second = lookup.get((pair.second_view, int(match.trainIdx)))
        ids.append(first if first is not None and first == second else None)
    return ids


def _keypoint_xy(features, view, keypoint_index):
    point = np.asarray(
        features[view].keypoints[keypoint_index].pt,
        dtype=np.float64,
    )
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ReconstructionError("Invalid feature keypoint coordinates.")
    return point


def _relative_pose(first_pose, second_pose, count):
    r1, t1 = first_pose
    r2, t2 = second_pose
    rotation = r2 @ r1.T
    translation = t2.reshape(3, 1) - rotation @ t1.reshape(3, 1)
    x, y, z = translation.reshape(3)
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    return RelativePose(
        rotation=rotation,
        translation=translation,
        essential_matrix=skew @ rotation,
        inlier_mask=np.ones(count, dtype=bool),
    )


def _sampson_inlier_mask(
    points_first,
    points_second,
    relative_pose,
    intrinsics,
    *,
    threshold_px=2.5,
):
    first = np.asarray(points_first, dtype=np.float64)
    second = np.asarray(points_second, dtype=np.float64)

    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape != second.shape
        or first.shape[1] != 2
    ):
        raise ReconstructionError(
            "Epipolar correspondences must have matching shape (N, 2)."
        )

    if (
        isinstance(threshold_px, bool)
        or not isinstance(threshold_px, (int, float))
        or not np.isfinite(threshold_px)
        or threshold_px <= 0
    ):
        raise ReconstructionError(
            "Epipolar threshold must be a positive finite number."
        )

    essential = np.asarray(relative_pose.essential_matrix, dtype=np.float64)
    if essential.shape != (3, 3) or not np.isfinite(essential).all():
        raise ReconstructionError("Relative essential matrix is invalid.")

    if len(first) == 0:
        return np.zeros(0, dtype=bool)

    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ReconstructionError(
            "Epipolar correspondences must contain finite coordinates."
        )

    camera_matrix = np.asarray(intrinsics.matrix, dtype=np.float64)
    try:
        inverse_camera = np.linalg.inv(camera_matrix)
    except np.linalg.LinAlgError as exc:
        raise ReconstructionError("Camera intrinsic matrix is singular.") from exc

    fundamental = inverse_camera.T @ essential @ inverse_camera
    if not np.isfinite(fundamental).all():
        raise ReconstructionError("Fundamental matrix is invalid.")

    first_h = np.column_stack((first, np.ones(len(first), dtype=np.float64)))
    second_h = np.column_stack((second, np.ones(len(second), dtype=np.float64)))

    line_second = (fundamental @ first_h.T).T
    line_first = (fundamental.T @ second_h.T).T

    residual = np.sum(second_h * line_second, axis=1)
    denominator = (
        line_second[:, 0] ** 2
        + line_second[:, 1] ** 2
        + line_first[:, 0] ** 2
        + line_first[:, 1] ** 2
    )

    valid_denominator = np.isfinite(denominator) & (denominator > 1e-12)
    errors = np.full(len(first), np.inf, dtype=np.float64)
    errors[valid_denominator] = np.sqrt(
        residual[valid_denominator] ** 2 / denominator[valid_denominator]
    )

    return np.isfinite(errors) & (errors <= threshold_px)


def _max_pairwise_parallax_deg(
    point_3d: np.ndarray,
    centers: np.ndarray,
) -> float:
    rays = point_3d[None, :] - centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-12):
        return 0.0
    unit_rays = rays / norms[:, None]
    dots = np.clip(unit_rays @ unit_rays.T, -1.0, 1.0)
    min_dot = float(np.min(dots))
    return float(np.rad2deg(np.arccos(min_dot)))


def _triangulate_track_multiview(
    track: FeatureTrack,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    features,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float = 1e-6,
    max_reprojection_error_px: float = 3.5,
    min_parallax_degrees: float = 0.25,
) -> np.ndarray | None:
    obs_views = []
    obs_rotations = []
    obs_translations = []
    obs_pixels = []
    obs_centers = []

    for obs in track.observations:
        view = int(obs.view_index)
        if view not in poses:
            continue
        r_mat, t_vec = poses[view]
        t_flat = t_vec.reshape(3)
        obs_views.append(view)
        obs_rotations.append(r_mat)
        obs_translations.append(t_flat)
        obs_pixels.append(_keypoint_xy(features, view, obs.keypoint_index))
        obs_centers.append(-r_mat.T @ t_flat)

    view_count = len(obs_views)
    if view_count < 2:
        return None

    rotations_arr = np.asarray(obs_rotations, dtype=np.float64)
    translations_arr = np.asarray(obs_translations, dtype=np.float64)
    pixels_arr = np.asarray(obs_pixels, dtype=np.float64)
    centers_arr = np.asarray(obs_centers, dtype=np.float64)

    result = triangulate_multiview_point(
        rotations_arr,
        translations_arr,
        pixels_arr,
        intrinsics,
        min_depth=min_depth,
    )
    if result is not None:
        point, errors = result
        if np.all(errors <= max_reprojection_error_px):
            if (
                _max_pairwise_parallax_deg(point, centers_arr)
                >= min_parallax_degrees
            ):
                return point

    if view_count < 3:
        return None

    # A single bad observation in a 3+ view track can corrupt all-view DLT;
    # find the largest geometrically consistent view subset via pair consensus.
    max_search_views = min(view_count, 8)
    best_inliers = None
    best_mean_err = float("inf")
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

    for i, j in combinations(range(max_search_views), 2):
        pair_res = triangulate_multiview_point(
            rotations_arr[[i, j]],
            translations_arr[[i, j]],
            pixels_arr[[i, j]],
            intrinsics,
            min_depth=min_depth,
        )
        if pair_res is None:
            continue
        cand_pt, pair_err = pair_res
        if np.any(pair_err > max_reprojection_error_px):
            continue

        cam_pts = np.einsum("nij,j->ni", rotations_arr, cand_pt) + translations_arr
        z = cam_pts[:, 2]
        valid_z = np.isfinite(cam_pts).all(axis=1) & (z > min_depth)
        inliers = []
        errs = []
        for idx in np.flatnonzero(valid_z):
            px = fx * (cam_pts[idx, 0] / z[idx]) + cx
            py = fy * (cam_pts[idx, 1] / z[idx]) + cy
            err = float(np.hypot(px - pixels_arr[idx, 0], py - pixels_arr[idx, 1]))
            if err <= max_reprojection_error_px:
                inliers.append(int(idx))
                errs.append(err)

        if len(inliers) >= 2:
            mean_err = float(np.mean(errs))
            if best_inliers is None or (len(inliers), -mean_err) > (
                len(best_inliers),
                -best_mean_err,
            ):
                best_inliers = inliers
                best_mean_err = mean_err

    if best_inliers is None:
        return None

    subset_res = triangulate_multiview_point(
        rotations_arr[best_inliers],
        translations_arr[best_inliers],
        pixels_arr[best_inliers],
        intrinsics,
        min_depth=min_depth,
    )
    if subset_res is None:
        return None

    point, errors = subset_res
    if np.any(errors > max_reprojection_error_px):
        return None
    if (
        _max_pairwise_parallax_deg(point, centers_arr[best_inliers])
        < min_parallax_degrees
    ):
        return None
    return point


def _add_triangulated_tracks(
    graph,
    intrinsics,
    poses,
    lookup,
    points_by_track,
    *,
    target_view: int | None = None,
):
    if target_view is None:
        for pair in graph.pairwise_matches:
            a, b = pair.first_view, pair.second_view
            if a not in poses or b not in poses or len(pair.matches) < 5:
                continue

            try:
                xy_a, xy_b = matched_keypoint_coordinates(
                    graph.features[a],
                    graph.features[b],
                    pair.matches,
                )
                relative = _relative_pose(poses[a], poses[b], len(pair.matches))
                geometric_mask = _sampson_inlier_mask(
                    xy_a,
                    xy_b,
                    relative,
                    intrinsics,
                    threshold_px=2.5,
                )
                if np.count_nonzero(geometric_mask) < 5:
                    continue

                relative = RelativePose(
                    rotation=relative.rotation,
                    translation=relative.translation,
                    essential_matrix=relative.essential_matrix,
                    inlier_mask=geometric_mask,
                )
                result = triangulate_points(
                    xy_a,
                    xy_b,
                    intrinsics,
                    relative,
                    max_reprojection_error_px=3.0,
                    min_parallax_degrees=0.25,
                )
                indexed = _indexed_points(result, len(pair.matches))
            except (ReconstructionError, cv2.error, np.linalg.LinAlgError):
                continue

            track_ids = _match_tracks(pair, lookup)
            rotation_a, translation_a = poses[a]
            for match_index, point_camera_a in indexed.items():
                track_id = track_ids[match_index]
                if track_id is None or track_id in points_by_track:
                    continue
                world_point = (point_camera_a - translation_a.reshape(3)) @ rotation_a
                if np.isfinite(world_point).all():
                    points_by_track[track_id] = world_point.copy()
        return

    for track_id, track in enumerate(graph.tracks):
        views_in_track = [int(obs.view_index) for obs in track.observations]
        if target_view not in views_in_track:
            continue
        registered_views = [v for v in views_in_track if v in poses]
        if len(registered_views) < 2:
            continue

        point = _triangulate_track_multiview(
            track,
            poses,
            graph.features,
            intrinsics,
        )
        if point is not None and np.isfinite(point).all():
            points_by_track[track_id] = point.copy()


def _is_valid_observation(
    point_3d: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    intrinsics: CameraIntrinsics,
    pixel_xy: np.ndarray,
    *,
    min_depth: float = 1e-6,
    max_reprojection_error_px: float = 12.0,
) -> bool:
    rotation, translation = pose
    camera_point = rotation @ point_3d + translation.reshape(3)
    if not np.isfinite(camera_point).all() or camera_point[2] <= min_depth:
        return False

    predicted_x = intrinsics.fx * (camera_point[0] / camera_point[2]) + intrinsics.cx
    predicted_y = intrinsics.fy * (camera_point[1] / camera_point[2]) + intrinsics.cy
    error = np.hypot(predicted_x - pixel_xy[0], predicted_y - pixel_xy[1])
    return bool(np.isfinite(error) and error <= max_reprojection_error_px)


def _pair_median_parallax_deg(
    points_cam_a: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    if len(points_cam_a) == 0:
        return 0.0
    cam_b_center = (-rotation.T @ translation.reshape(3)).reshape(1, 3)
    ray_a = points_cam_a
    ray_b = points_cam_a - cam_b_center
    norm_prod = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
    valid = norm_prod > 1e-12
    if not np.any(valid):
        return 0.0
    cosines = np.sum(ray_a[valid] * ray_b[valid], axis=1) / norm_prod[valid]
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return float(np.median(angles))


def _seed_pair(
    graph,
    intrinsics,
    lookup,
    min_matches,
    min_points,
    *,
    refine_two_view: bool = True,
):
    candidates = sorted(
        graph.pairwise_matches,
        key=lambda pair: len(pair.matches),
        reverse=True,
    )

    best_choice = None
    best_rank = (-1, -1.0)
    max_candidates_to_evaluate = 24
    evaluated = 0

    for pair in candidates:
        if len(pair.matches) < min_matches:
            continue
        if evaluated >= max_candidates_to_evaluate and best_choice is not None:
            break
        evaluated += 1

        try:
            xy_a, xy_b = matched_keypoint_coordinates(
                graph.features[pair.first_view],
                graph.features[pair.second_view],
                pair.matches,
            )
            pose = estimate_relative_pose(
                xy_a, xy_b, intrinsics, min_inliers=min_matches
            )
            result = triangulate_points(xy_a, xy_b, intrinsics, pose)
            indexed = _indexed_points(result, len(pair.matches))
        except (ReconstructionError, cv2.error, np.linalg.LinAlgError):
            continue

        ids = _match_tracks(pair, lookup)
        points = {
            ids[index]: point.copy()
            for index, point in indexed.items()
            if ids[index] is not None
        }

        if len(points) < min_points:
            continue

        rotation = np.asarray(pose.rotation, dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64).reshape(3, 1)

        if np.linalg.norm(translation) < 1e-8:
            continue

        median_parallax = _pair_median_parallax_deg(
            np.asarray(list(points.values()), dtype=np.float64),
            rotation,
            translation,
        )
        # Prefer well-conditioned baselines (>= 2 deg parallax) over near-zero
        # baseline pairs even if the latter have slightly more raw matches.
        tier = 1 if median_parallax >= 2.0 else 0
        parallax_weight = (
            min(median_parallax / 6.0, 1.5)
            if tier == 1
            else min(median_parallax / 2.0, 1.0)
        )
        score = float(len(points)) * max(parallax_weight, 1e-3)
        rank = (tier, score)

        if rank > best_rank:
            best_rank = rank
            best_choice = (
                (pair.first_view, pair.second_view),
                {
                    pair.first_view: (np.eye(3), np.zeros((3, 1))),
                    pair.second_view: (rotation.copy(), translation.copy()),
                },
                points,
            )

    if best_choice is None:
        raise ReconstructionError(
            "No suitable starting camera pair found. Use photographs with "
            "more overlapping visual features."
        )

    seed, poses, points = best_choice
    if refine_two_view and len(points) >= 6:
        try:
            ba_res, selected_ids = _run_pose_and_point_ba(
                seed,
                poses,
                points,
                graph,
                intrinsics,
                max_observation_error_px=5.0,
                max_nfev=60,
            )
            if ba_res is not None and ba_res.final_rmse <= ba_res.initial_rmse:
                r1, _ = cv2.Rodrigues(ba_res.camera_rotations[1])
                poses[seed[1]] = (
                    r1.copy(),
                    ba_res.camera_translations[1].reshape(3, 1).copy(),
                )
                for idx, track_id in enumerate(selected_ids):
                    points[int(track_id)] = ba_res.points_3d[idx].copy()
        except ReconstructionError:
            pass

    return seed, poses, points


def _collect_ba_problem(
    order,
    poses,
    points_by_track,
    graph,
    intrinsics,
    *,
    max_observation_error_px: float,
):
    camera_index = {view: index for index, view in enumerate(order)}
    rotations = np.vstack(
        [cv2.Rodrigues(poses[view][0])[0].reshape(3) for view in order]
    )
    translations = np.vstack([poses[view][1].reshape(3) for view in order])

    selected_ids, point_list = [], []
    camera_indices, point_indices, pixels = [], [], []

    for track_id in sorted(points_by_track):
        track_point = points_by_track[track_id]
        observations = []
        for obs in graph.tracks[track_id].observations:
            if obs.view_index in camera_index:
                xy = _keypoint_xy(graph.features, obs.view_index, obs.keypoint_index)
                if _is_valid_observation(
                    track_point,
                    poses[obs.view_index],
                    intrinsics,
                    xy,
                    max_reprojection_error_px=max_observation_error_px,
                ):
                    observations.append((camera_index[obs.view_index], xy))

        if len(observations) < 2:
            continue

        selected_ids.append(track_id)
        point_list.append(track_point)
        for cam, xy in observations:
            camera_indices.append(cam)
            point_indices.append(len(point_list) - 1)
            pixels.append(xy)

    return (
        rotations,
        translations,
        np.asarray(point_list, dtype=np.float64),
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(camera_indices, dtype=np.int64),
        np.asarray(point_indices, dtype=np.int64),
        np.asarray(pixels, dtype=np.float64),
    )


def _run_pose_and_point_ba(
    order,
    poses,
    points_by_track,
    graph,
    intrinsics,
    *,
    max_observation_error_px: float = 5.0,
    max_nfev: int = 80,
):
    (
        rotations,
        translations,
        points_3d,
        selected_ids,
        camera_indices,
        point_indices,
        pixels,
    ) = _collect_ba_problem(
        order,
        poses,
        points_by_track,
        graph,
        intrinsics,
        max_observation_error_px=max_observation_error_px,
    )
    if len(points_3d) < 6 or len(pixels) < 4:
        return None, selected_ids

    ba_result = bundle_adjust(
        rotations,
        translations,
        points_3d,
        camera_indices,
        point_indices,
        pixels,
        intrinsics,
        max_nfev=max_nfev,
        fix_second_camera=False,
    )
    return ba_result, selected_ids


def reconstruct_image_set(
    images,
    *,
    intrinsics=None,
    method="orb",
    max_features=2000,
    ratio_threshold=0.75,
    pairing_strategy="all",
    min_track_length=2,
    min_pair_matches=8,
    min_seed_points=8,
    pnp_min_inliers=6,
    run_bundle_adjustment=True,
    bundle_adjustment_max_nfev=200,
    require_all_cameras=False,
    masks=None,
):
    images = tuple(images)

    if len(images) < 2:
        raise ReconstructionError("At least two images are required.")

    if any(not isinstance(image, Image.Image) for image in images):
        raise ReconstructionError("Every input must be a PIL image.")

    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ReconstructionError("All images must have the same dimensions.")

    if (
        isinstance(min_pair_matches, bool)
        or not isinstance(min_pair_matches, int)
        or min_pair_matches < 8
    ):
        raise ReconstructionError("min_pair_matches must be at least 8.")
    if (
        isinstance(min_seed_points, bool)
        or not isinstance(min_seed_points, int)
        or min_seed_points < 1
    ):
        raise ReconstructionError("min_seed_points must be positive.")
    if (
        isinstance(pnp_min_inliers, bool)
        or not isinstance(pnp_min_inliers, int)
        or pnp_min_inliers < 4
    ):
        raise ReconstructionError("pnp_min_inliers must be at least 4.")
    if not isinstance(run_bundle_adjustment, bool):
        raise ReconstructionError("run_bundle_adjustment must be True or False.")

    intrinsics = _make_intrinsics(width, height, intrinsics)

    graph = analyze_image_set(
        images,
        method=method,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        pairing_strategy=pairing_strategy,
        min_track_length=min_track_length,
        masks=masks,
    )

    if not graph.pairwise_matches:
        raise ReconstructionError("No pairwise feature matches were found.")

    lookup = _track_lookup(graph.tracks)
    seed, poses, points_by_track = _seed_pair(
        graph,
        intrinsics,
        lookup,
        min_pair_matches,
        min_seed_points,
        refine_two_view=run_bundle_adjustment,
    )

    matrix = _camera_matrix(intrinsics)

    while len(poses) < len(images):
        candidates = []

        for view in range(len(images)):
            if view in poses:
                continue

            objects, pixels = [], []
            for track_id, point in points_by_track.items():
                observation = next(
                    (
                        obs
                        for obs in graph.tracks[track_id].observations
                        if obs.view_index == view
                    ),
                    None,
                )
                if observation is not None:
                    objects.append(point)
                    pixels.append(
                        _keypoint_xy(graph.features, view, observation.keypoint_index)
                    )

            if len(objects) >= pnp_min_inliers:
                candidates.append((view, objects, pixels))

        candidates.sort(key=lambda item: len(item[1]), reverse=True)

        registered_view = None
        for view, objects, pixels in candidates:
            try:
                result = register_camera_pnp(
                    np.asarray(objects),
                    np.asarray(pixels),
                    matrix,
                    reprojection_error=3.5,
                    min_inliers=pnp_min_inliers,
                )
            except ReconstructionError:
                continue

            poses[view] = (
                result.rotation_matrix.copy(),
                result.translation_vector.copy(),
            )
            registered_view = view
            break

        if registered_view is None:
            if require_all_cameras:
                raise ReconstructionError(
                    "Could not register every camera. A remaining image needs "
                    "more reliable 2D-3D feature correspondences."
                )
            break

        _add_triangulated_tracks(
            graph,
            intrinsics,
            poses,
            lookup,
            points_by_track,
            target_view=registered_view,
        )

        if (
            run_bundle_adjustment
            and len(poses) >= 3
            and len(points_by_track) >= 6
            and (len(poses) <= 8 or len(poses) % 2 == 0)
        ):
            cur_order = (
                seed[0],
                seed[1],
                *(
                    v
                    for v in range(len(images))
                    if v in poses and v not in seed
                ),
            )
            try:
                inc_ba, inc_ids = _run_pose_and_point_ba(
                    cur_order,
                    poses,
                    points_by_track,
                    graph,
                    intrinsics,
                    max_observation_error_px=6.0,
                    max_nfev=60,
                )
                if inc_ba is not None and inc_ba.final_rmse <= inc_ba.initial_rmse:
                    for idx, view_id in enumerate(cur_order):
                        r_mat, _ = cv2.Rodrigues(inc_ba.camera_rotations[idx])
                        poses[view_id] = (
                            r_mat.copy(),
                            inc_ba.camera_translations[idx].reshape(3, 1).copy(),
                        )
                    for idx, track_id in enumerate(inc_ids):
                        points_by_track[int(track_id)] = inc_ba.points_3d[idx].copy()
            except ReconstructionError:
                pass

    order = (
        seed[0],
        seed[1],
        *(view for view in range(len(images)) if view in poses and view not in seed),
    )

    (
        rotations,
        translations,
        points_3d,
        selected_ids,
        camera_indices,
        point_indices,
        pixels,
    ) = _collect_ba_problem(
        order,
        poses,
        points_by_track,
        graph,
        intrinsics,
        max_observation_error_px=5.0,
    )

    if len(points_3d) == 0:
        raise ReconstructionError("No valid multi-view 3D tracks were produced.")

    ba_result = None

    if run_bundle_adjustment:
        if len(points_3d) < 6:
            raise ReconstructionError(
                "At least six reconstructed points are required for bundle adjustment."
            )

        ba_result = bundle_adjust(
            rotations,
            translations,
            points_3d,
            camera_indices,
            point_indices,
            pixels,
            intrinsics,
            max_nfev=bundle_adjustment_max_nfev,
            fix_second_camera=False,
        )

        if not ba_result.success and ba_result.final_rmse > ba_result.initial_rmse:
            raise ReconstructionError("Bundle adjustment failed: " + ba_result.message)

        rotations = ba_result.camera_rotations
        translations = ba_result.camera_translations
        points_3d = ba_result.points_3d

        # Post-BA outlier pruning and polish pass.
        refined_poses = {}
        for idx, view in enumerate(order):
            r_mat, _ = cv2.Rodrigues(rotations[idx])
            refined_poses[view] = (
                r_mat.copy(),
                translations[idx].reshape(3, 1).copy(),
            )
        refined_points = {
            int(track_id): points_3d[idx].copy()
            for idx, track_id in enumerate(selected_ids)
        }
        (
            p_rot,
            p_trans,
            p_pts,
            p_ids,
            p_cam_ids,
            p_pt_ids,
            p_pixels,
        ) = _collect_ba_problem(
            order,
            refined_poses,
            refined_points,
            graph,
            intrinsics,
            max_observation_error_px=3.5,
        )
        if len(p_pts) >= 6 and len(p_pixels) < len(pixels):
            polish = bundle_adjust(
                p_rot,
                p_trans,
                p_pts,
                p_cam_ids,
                p_pt_ids,
                p_pixels,
                intrinsics,
                max_nfev=min(bundle_adjustment_max_nfev, 80),
                fix_second_camera=False,
            )
            if polish.success or polish.final_rmse <= polish.initial_rmse:
                ba_result = polish
                rotations = polish.camera_rotations
                translations = polish.camera_translations
                points_3d = polish.points_3d
                selected_ids = p_ids

    return MultiViewReconstructionResult(
        seed_pair=seed,
        view_order=tuple(order),
        camera_rotations=np.asarray(rotations),
        camera_translations=np.asarray(translations),
        points_3d=np.asarray(points_3d),
        point_track_indices=selected_ids,
        intrinsics=intrinsics,
        feature_graph=graph,
        bundle_adjustment_result=ba_result,
    )
