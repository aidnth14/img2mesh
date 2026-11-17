from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from .config import ConfigurationError, InputError, ReconstructionError


@dataclass(frozen=True, slots=True)
class FeatureSet:
    method: str
    keypoints: tuple
    descriptors: np.ndarray | None
    image_size: tuple[int, int]

    @property
    def count(self) -> int:
        return len(self.keypoints)


def extract_features(
    image: Image.Image,
    *,
    method: str = "orb",
    max_features: int = 2000,
    mask: np.ndarray | None = None,
) -> FeatureSet:
    if not isinstance(image, Image.Image):
        raise InputError("Feature extraction requires a PIL image.")

    if (
        isinstance(max_features, bool)
        or not isinstance(max_features, int)
        or max_features < 1
    ):
        raise ConfigurationError("Maximum features must be a positive integer.")

    if not isinstance(method, str):
        raise ConfigurationError("Feature method must be 'orb' or 'sift'.")

    normalized_method = method.lower()

    if normalized_method == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
    elif normalized_method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise InputError("SIFT is unavailable in this OpenCV installation.")
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        raise ConfigurationError("Feature method must be 'orb' or 'sift'.")

    cv_mask = None
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise InputError("Mask must be a two-dimensional NumPy array.")
        if mask.shape != (image.height, image.width):
            raise InputError("Mask dimensions must match the image dimensions.")
        cv_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    keypoints, descriptors = detector.detectAndCompute(gray, cv_mask)
    keypoints_tuple = tuple(keypoints) if keypoints is not None else ()

    if len(keypoints_tuple) < min(150, max_features // 2) and int(np.ptp(gray)) > 0:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if normalized_method == "orb":
            fallback = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=1.15,
                nlevels=10,
                fastThreshold=7,
            )
        else:
            fallback = cv2.SIFT_create(
                nfeatures=max_features,
                contrastThreshold=0.01,
                edgeThreshold=15,
            )
        alt_kp, alt_desc = fallback.detectAndCompute(enhanced, cv_mask)
        if alt_kp is not None and len(alt_kp) > len(keypoints_tuple):
            keypoints_tuple = tuple(alt_kp)
            descriptors = alt_desc

    if len(keypoints_tuple) > max_features:
        keypoints_tuple = keypoints_tuple[:max_features]
        if descriptors is not None:
            descriptors = descriptors[:max_features]

    if not keypoints_tuple:
        descriptors = None

    return FeatureSet(
        method=normalized_method,
        keypoints=keypoints_tuple,
        descriptors=descriptors,
        image_size=(image.width, image.height),
    )


def match_features(
    first: FeatureSet,
    second: FeatureSet,
    *,
    ratio_threshold: float = 0.75,
) -> tuple:
    if not isinstance(first, FeatureSet) or not isinstance(second, FeatureSet):
        raise InputError("match_features expects two FeatureSet instances.")

    if first.method != second.method:
        raise ConfigurationError("Feature sets must use the same detection method.")

    if (
        isinstance(ratio_threshold, bool)
        or not isinstance(ratio_threshold, (int, float))
        or not 0 < ratio_threshold < 1
    ):
        raise ConfigurationError(
            "Ratio threshold must be greater than 0 and less than 1."
        )

    if first.descriptors is None or second.descriptors is None:
        return ()

    if len(first.descriptors) < 2 or len(second.descriptors) < 2:
        return ()

    norm = cv2.NORM_HAMMING if first.method == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
    candidate_pairs = matcher.knnMatch(first.descriptors, second.descriptors, k=2)

    good_matches = [
        pair[0]
        for pair in candidate_pairs
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    ]
    good_matches.sort(key=lambda match: match.distance)

    return tuple(good_matches)


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    view_index: int
    keypoint_index: int


@dataclass(frozen=True, slots=True)
class FeatureTrack:
    observations: tuple[FeatureObservation, ...]

    @property
    def view_count(self) -> int:
        return len(self.observations)


class _DisjointSet:
    def __init__(self) -> None:
        self.parent = {}
        self.size = {}
        self.views = {}

    def add(self, node: tuple[int, int]) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            self.views[node] = {node[0]: node[1]}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, first: tuple[int, int], second: tuple[int, int]) -> bool:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return True

        first_views = self.views[root_first]
        second_views = self.views[root_second]

        for view_index in first_views.keys() & second_views.keys():
            if first_views[view_index] != second_views[view_index]:
                return False

        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = root_second, root_first
            first_views, second_views = second_views, first_views

        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]
        self.views[root_first] = {**first_views, **second_views}

        del self.size[root_second]
        del self.views[root_second]
        return True


def build_feature_tracks(
    feature_sets: Sequence[FeatureSet],
    pair_matches: Mapping[tuple[int, int], Sequence],
    *,
    min_track_length: int = 2,
) -> tuple[FeatureTrack, ...]:
    if len(feature_sets) < 2:
        raise ReconstructionError("At least two feature sets are required.")

    if (
        isinstance(min_track_length, bool)
        or not isinstance(min_track_length, int)
        or min_track_length < 2
    ):
        raise ConfigurationError(
            "Minimum track length must be an integer of at least 2."
        )

    sets = _DisjointSet()
    edges = []

    for pair, matches in pair_matches.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ReconstructionError("Each pair key must contain two view indices.")

        first_view, second_view = pair

        for view_index in pair:
            if (
                isinstance(view_index, bool)
                or not isinstance(view_index, int)
                or not 0 <= view_index < len(feature_sets)
            ):
                raise ReconstructionError(f"Invalid view index: {view_index}.")

        if first_view == second_view:
            raise ReconstructionError(
                "A feature pair cannot refer to the same view twice."
            )

        for match in matches:
            first_index = match.queryIdx
            second_index = match.trainIdx

            if not (
                0 <= first_index < len(feature_sets[first_view].keypoints)
                and 0 <= second_index < len(feature_sets[second_view].keypoints)
            ):
                raise ReconstructionError(
                    "Feature match contains an invalid keypoint index."
                )

            try:
                distance = float(match.distance)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ReconstructionError(
                    "Feature match must contain a numeric descriptor distance."
                ) from exc

            if not math.isfinite(distance) or distance < 0:
                raise ReconstructionError(
                    "Feature descriptor distance must be finite and non-negative."
                )

            edges.append(
                (
                    distance,
                    (first_view, first_index),
                    (second_view, second_index),
                )
            )

    edges.sort(key=lambda edge: edge[0])

    for _, first_node, second_node in edges:
        sets.add(first_node)
        sets.add(second_node)
        sets.union(first_node, second_node)

    components = {}
    for node in sets.parent:
        root = sets.find(node)
        components[root] = sets.views[root]

    tracks = []
    for observations in components.values():
        if len(observations) < min_track_length:
            continue

        ordered = tuple(
            FeatureObservation(view_index, keypoint_index)
            for view_index, keypoint_index in sorted(observations.items())
        )
        tracks.append(FeatureTrack(ordered))

    tracks.sort(
        key=lambda track: (
            track.observations[0].view_index,
            track.observations[0].keypoint_index,
        )
    )

    return tuple(tracks)


@dataclass(frozen=True, slots=True)
class PairwiseMatches:
    first_view: int
    second_view: int
    matches: tuple


@dataclass(frozen=True, slots=True)
class MultiViewFeatureGraph:
    features: tuple[FeatureSet, ...]
    pairwise_matches: tuple[PairwiseMatches, ...]
    tracks: tuple[FeatureTrack, ...]
    image_sizes: tuple[tuple[int, int], ...]
    pairing_strategy: str

    @property
    def view_count(self) -> int:
        return len(self.features)

    @property
    def pair_count(self) -> int:
        return len(self.pairwise_matches)

    @property
    def total_feature_count(self) -> int:
        return sum(feature.count for feature in self.features)

    @property
    def total_match_count(self) -> int:
        return sum(len(pair.matches) for pair in self.pairwise_matches)

    @property
    def track_count(self) -> int:
        return len(self.tracks)


def _build_view_pairs(
    view_count: int,
    strategy: str,
) -> tuple[tuple[int, int], ...]:
    if strategy == "all":
        return tuple(combinations(range(view_count), 2))
    if strategy == "sequential":
        return tuple((index, index + 1) for index in range(view_count - 1))
    raise ConfigurationError("Pairing strategy must be 'all' or 'sequential'.")


def analyze_image_set(
    images: Sequence[Image.Image],
    *,
    method: str = "orb",
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
    pairing_strategy: str = "all",
    min_track_length: int = 2,
    masks: Sequence | None = None,
) -> MultiViewFeatureGraph:
    if isinstance(images, (str, bytes)):
        raise InputError("Expected a sequence of decoded PIL images.")

    try:
        view_images = tuple(images)
    except TypeError as exc:
        raise InputError("Images must be provided as a sequence.") from exc

    if len(view_images) < 2:
        raise ReconstructionError(
            "At least two images are required for multi-view analysis."
        )

    if not all(isinstance(image, Image.Image) for image in view_images):
        raise InputError("Every input view must be a PIL image.")

    image_sizes = tuple((image.width, image.height) for image in view_images)
    reference_size = image_sizes[0]

    if any(size != reference_size for size in image_sizes):
        raise ReconstructionError(
            "All images must have identical dimensions. "
            "Resize them consistently and scale camera intrinsics "
            "accordingly before reconstruction."
        )

    if masks is not None:
        if isinstance(masks, (str, bytes)):
            raise InputError("Masks must be provided as a sequence.")
        try:
            view_masks = tuple(masks)
        except TypeError as exc:
            raise InputError("Masks must be provided as a sequence.") from exc
        if len(view_masks) != len(view_images):
            raise InputError("Mask count must match the number of images.")
        view_features = tuple(
            extract_features(
                image,
                method=method,
                max_features=max_features,
                mask=mask,
            )
            for image, mask in zip(view_images, view_masks, strict=True)
        )
    else:
        view_features = tuple(
            extract_features(
                image,
                method=method,
                max_features=max_features,
            )
            for image in view_images
        )

    view_pairs = _build_view_pairs(len(view_images), pairing_strategy)
    pairwise_results = []
    track_matches = {}

    for first_view, second_view in view_pairs:
        matches = match_features(
            view_features[first_view],
            view_features[second_view],
            ratio_threshold=ratio_threshold,
        )
        pair_result = PairwiseMatches(
            first_view=first_view,
            second_view=second_view,
            matches=matches,
        )
        pairwise_results.append(pair_result)
        track_matches[(first_view, second_view)] = matches

    tracks = build_feature_tracks(
        view_features,
        track_matches,
        min_track_length=min_track_length,
    )

    return MultiViewFeatureGraph(
        features=view_features,
        pairwise_matches=tuple(pairwise_results),
        tracks=tracks,
        image_sizes=image_sizes,
        pairing_strategy=pairing_strategy,
    )
