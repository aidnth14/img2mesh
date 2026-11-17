from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
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


def _build_jacobian_sparsity(
    camera_count: int,
    point_count: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
):
    observation_count = len(camera_indices)
    pose_parameter_count = max(0, camera_count - 2) * 6

    sparsity = lil_matrix(
        (
            observation_count * 3,
            pose_parameter_count + point_count * 3,
        ),
        dtype=np.int8,
    )

    for observation_id, (camera_id, point_id) in enumerate(
        zip(camera_indices, point_indices, strict=True)
    ):
        pixel_rows = slice(observation_id * 2, observation_id * 2 + 2)
        depth_row = 2 * observation_count + observation_id

        if camera_id >= 2:
            camera_start = (camera_id - 2) * 6
            camera_columns = slice(camera_start, camera_start + 6)
            sparsity[pixel_rows, camera_columns] = 1
            sparsity[depth_row, camera_columns] = 1

        point_start = pose_parameter_count + point_id * 3
        point_columns = slice(point_start, point_start + 3)
        sparsity[pixel_rows, point_columns] = 1
        sparsity[depth_row, point_columns] = 1

    return sparsity.tocsr()


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

    pose_parameters = np.column_stack(
        (
            initial_rotations[2:],
            initial_translations[2:],
        )
    ).ravel()

    x0 = np.concatenate((pose_parameters, initial_points.ravel()))
    pose_parameter_count = len(pose_parameters)

    def unpack(parameters):
        rotations = initial_rotations.copy()
        translations = initial_translations.copy()

        if camera_count > 2:
            camera_parameters = parameters[:pose_parameter_count].reshape(-1, 6)
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

    def pixel_rmse(parameters):
        residuals = calculate_residuals(parameters)
        pixel_values = residuals[: len(observations) * 2]
        return float(np.sqrt(np.mean(pixel_values**2)))

    initial_rmse = pixel_rmse(x0)

    sparsity = _build_jacobian_sparsity(
        camera_count,
        point_count,
        camera_ids,
        point_ids,
    )

    try:
        optimized = least_squares(
            calculate_residuals,
            x0,
            jac_sparsity=sparsity,
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


def _add_triangulated_tracks(graph, intrinsics, poses, lookup, points_by_track):
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


def _seed_pair(graph, intrinsics, lookup, min_matches, min_points):
    candidates = sorted(
        graph.pairwise_matches,
        key=lambda pair: len(pair.matches),
        reverse=True,
    )

    for pair in candidates:
        if len(pair.matches) < min_matches:
            continue

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

        return (
            (pair.first_view, pair.second_view),
            {
                pair.first_view: (np.eye(3), np.zeros((3, 1))),
                pair.second_view: (rotation.copy(), translation.copy()),
            },
            points,
        )

    raise ReconstructionError(
        "No suitable starting camera pair found. Use photographs with "
        "more overlapping visual features."
    )


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
    )

    if not graph.pairwise_matches:
        raise ReconstructionError("No pairwise feature matches were found.")

    lookup = _track_lookup(graph.tracks)
    seed, poses, points_by_track = _seed_pair(
        graph, intrinsics, lookup, min_pair_matches, min_seed_points
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

        registered = False
        for view, objects, pixels in candidates:
            try:
                result = register_camera_pnp(
                    np.asarray(objects),
                    np.asarray(pixels),
                    matrix,
                    min_inliers=pnp_min_inliers,
                )
            except ReconstructionError:
                continue

            poses[view] = (
                result.rotation_matrix.copy(),
                result.translation_vector.copy(),
            )
            registered = True
            break

        if not registered:
            raise ReconstructionError(
                "Could not register every camera. A remaining image needs "
                "more reliable 2D-3D feature correspondences."
            )

        _add_triangulated_tracks(graph, intrinsics, poses, lookup, points_by_track)

    order = (
        seed[0],
        seed[1],
        *(view for view in range(len(images)) if view not in seed),
    )
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

    if not point_list:
        raise ReconstructionError("No valid multi-view 3D tracks were produced.")

    points_3d = np.asarray(point_list, dtype=np.float64)
    selected_ids = np.asarray(selected_ids, dtype=np.int64)
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
            np.asarray(camera_indices, dtype=np.int64),
            np.asarray(point_indices, dtype=np.int64),
            np.asarray(pixels, dtype=np.float64),
            intrinsics,
            max_nfev=bundle_adjustment_max_nfev,
        )

        if not ba_result.success:
            raise ReconstructionError("Bundle adjustment failed: " + ba_result.message)

        rotations = ba_result.camera_rotations
        translations = ba_result.camera_translations
        points_3d = ba_result.points_3d

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
