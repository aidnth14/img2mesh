import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from img2mesh.config import ConfigurationError, InputError, ReconstructionError
from img2mesh.features import (
    FeatureSet,
    analyze_image_set,
    build_feature_tracks,
    extract_features,
    match_features,
)


def make_checkerboard():
    image = Image.new("RGB", (256, 256), color="white")
    draw = ImageDraw.Draw(image)
    square_size = 32

    for row in range(8):
        for column in range(8):
            if (row + column) % 2 == 0:
                left = column * square_size
                top = row * square_size
                draw.rectangle(
                    [left, top, left + square_size - 1, top + square_size - 1],
                    fill="black",
                )
    return image


def make_textured_image(seed=42):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def make_features(count):
    return FeatureSet(
        method="orb",
        keypoints=tuple(
            cv2.KeyPoint(float(i * 10), float(i * 10), 1) for i in range(count)
        ),
        descriptors=None,
        image_size=(100, 100),
    )


def test_orb_feature_extraction():
    image = make_checkerboard()
    result = extract_features(image, method="orb", max_features=500)

    assert result.method == "orb"
    assert result.count > 0
    assert result.descriptors is not None
    assert result.descriptors.shape[0] == result.count
    assert result.image_size == (256, 256)


def test_feature_count_limit():
    result = extract_features(make_checkerboard(), method="orb", max_features=50)
    assert result.count <= 50


def test_blank_image_returns_no_features():
    image = Image.new("RGB", (100, 100), color="white")
    result = extract_features(image)

    assert result.count == 0
    assert result.descriptors is None


def test_matching_identical_images():
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)

    first = extract_features(image)
    second = extract_features(image)
    matches = match_features(first, second)

    assert len(matches) > 0
    assert all(match.distance >= 0 for match in matches)


def test_matching_requires_same_method():
    first = FeatureSet("orb", (), None, (100, 100))
    second = FeatureSet("sift", (), None, (100, 100))

    with pytest.raises(ConfigurationError):
        match_features(first, second)


def test_matching_without_descriptors():
    first = FeatureSet("orb", (), None, (100, 100))
    second = FeatureSet("orb", (), None, (100, 100))
    assert match_features(first, second) == ()


def test_invalid_feature_method():
    with pytest.raises(ConfigurationError):
        extract_features(make_checkerboard(), method="invalid")


def test_invalid_feature_image():
    with pytest.raises(InputError):
        extract_features("not an image")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_maximum_features(limit):
    with pytest.raises(ConfigurationError):
        extract_features(make_checkerboard(), max_features=limit)


@pytest.mark.parametrize("ratio", [0, 1, -0.5, 1.5, True])
def test_invalid_matching_ratio(ratio):
    empty = FeatureSet("orb", (), None, (100, 100))
    with pytest.raises(ConfigurationError):
        match_features(empty, empty, ratio_threshold=ratio)


def test_sift_feature_extraction_when_available():
    image = make_checkerboard()
    try:
        result = extract_features(image, method="sift")
    except InputError:
        pytest.skip("SIFT unavailable in this OpenCV build.")

    assert result.method == "sift"
    assert result.count > 0


def test_feature_extraction_with_mask():
    image = make_checkerboard()
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:128, :128] = 255

    result = extract_features(image, mask=mask)
    assert result.count > 0
    assert all(kp.pt[0] < 128 and kp.pt[1] < 128 for kp in result.keypoints)


def test_feature_extraction_rejects_invalid_mask():
    image = make_checkerboard()

    with pytest.raises(InputError, match="two-dimensional"):
        extract_features(image, mask=np.zeros((256, 256, 3), dtype=np.uint8))

    with pytest.raises(InputError, match="dimensions"):
        extract_features(image, mask=np.zeros((64, 64), dtype=np.uint8))


def test_match_features_rejects_non_feature_set():
    with pytest.raises(InputError, match="FeatureSet"):
        match_features("invalid", "invalid")


def test_track_across_two_views():
    features = [make_features(3), make_features(3)]
    matches = {(0, 1): (cv2.DMatch(0, 1, 0), cv2.DMatch(2, 2, 0))}

    tracks = build_feature_tracks(features, matches)
    assert len(tracks) == 2
    assert tracks[0].view_count == 2


def test_track_across_three_views():
    features = [make_features(2), make_features(2), make_features(2)]
    matches = {
        (0, 1): (cv2.DMatch(0, 0, 0),),
        (1, 2): (cv2.DMatch(0, 1, 0),),
    }

    tracks = build_feature_tracks(features, matches)
    assert len(tracks) == 1
    assert tracks[0].view_count == 3
    assert [
        (obs.view_index, obs.keypoint_index) for obs in tracks[0].observations
    ] == [(0, 0), (1, 0), (2, 1)]


def test_conflicting_same_view_observations_are_not_merged():
    features = [make_features(2), make_features(1), make_features(1)]
    matches = {
        (0, 1): (cv2.DMatch(0, 0, 0),),
        (1, 2): (cv2.DMatch(0, 0, 0),),
        (0, 2): (cv2.DMatch(1, 0, 0),),
    }

    tracks = build_feature_tracks(features, matches)
    assert len(tracks) == 1

    observations = {
        (obs.view_index, obs.keypoint_index) for obs in tracks[0].observations
    }
    assert observations == {(0, 0), (1, 0), (2, 0)}
    assert (0, 1) not in observations


@pytest.mark.parametrize("reverse", [False, True])
def test_stronger_matches_are_merged_first(reverse):
    features = [make_features(2), make_features(1), make_features(1)]
    entries = [
        ((0, 1), (cv2.DMatch(0, 0, 20.0),)),
        ((0, 2), (cv2.DMatch(1, 0, 1.0),)),
        ((1, 2), (cv2.DMatch(0, 0, 2.0),)),
    ]
    if reverse:
        entries.reverse()

    tracks = build_feature_tracks(features, dict(entries))
    assert len(tracks) == 1
    observations = {
        (obs.view_index, obs.keypoint_index) for obs in tracks[0].observations
    }
    assert observations == {(0, 1), (1, 0), (2, 0)}


def test_equal_distance_matches_preserve_insertion_order():
    features = [make_features(2), make_features(1), make_features(1)]
    matches = {
        (0, 1): (cv2.DMatch(0, 0, 0.0),),
        (1, 2): (cv2.DMatch(0, 0, 0.0),),
        (0, 2): (cv2.DMatch(1, 0, 0.0),),
    }

    tracks = build_feature_tracks(features, matches)
    assert len(tracks) == 1
    observations = {
        (obs.view_index, obs.keypoint_index) for obs in tracks[0].observations
    }
    assert observations == {(0, 0), (1, 0), (2, 0)}


def test_non_finite_descriptor_distance_is_rejected():
    features = [make_features(1), make_features(1)]
    with pytest.raises(ReconstructionError, match="descriptor distance"):
        build_feature_tracks(
            features,
            {(0, 1): (cv2.DMatch(0, 0, float("nan")),)},
        )


def test_short_tracks_are_filtered():
    features = [make_features(2), make_features(2), make_features(2)]
    matches = {(0, 1): (cv2.DMatch(0, 0, 0),)}
    tracks = build_feature_tracks(features, matches, min_track_length=3)
    assert tracks == ()


def test_invalid_view_index():
    features = [make_features(2), make_features(2)]
    with pytest.raises(ReconstructionError):
        build_feature_tracks(features, {(0, 3): (cv2.DMatch(0, 0, 0),)})


def test_invalid_keypoint_index():
    features = [make_features(2), make_features(2)]
    with pytest.raises(ReconstructionError):
        build_feature_tracks(features, {(0, 1): (cv2.DMatch(9, 0, 0),)})


def test_same_view_pair_is_rejected():
    features = [make_features(2), make_features(2)]
    with pytest.raises(ReconstructionError):
        build_feature_tracks(features, {(0, 0): (cv2.DMatch(0, 0, 0),)})


def test_requires_two_feature_sets():
    with pytest.raises(ReconstructionError):
        build_feature_tracks([make_features(2)], {})


def test_invalid_minimum_track_length():
    features = [make_features(2), make_features(2)]
    with pytest.raises(ConfigurationError):
        build_feature_tracks(features, {}, min_track_length=1)


def test_requires_at_least_two_views():
    with pytest.raises(ReconstructionError):
        analyze_image_set([make_textured_image()])


def test_rejects_non_pil_images():
    with pytest.raises(InputError):
        analyze_image_set([make_textured_image(), "not an image"])


def test_rejects_different_image_dimensions():
    first = Image.new("RGB", (320, 240))
    second = Image.new("RGB", (640, 480))
    with pytest.raises(ReconstructionError):
        analyze_image_set([first, second])


def test_all_pair_matching_builds_complete_pair_graph():
    image = make_textured_image()
    result = analyze_image_set(
        [image.copy(), image.copy(), image.copy()],
        pairing_strategy="all",
    )

    assert result.view_count == 3
    assert result.pair_count == 3
    assert {
        (pair.first_view, pair.second_view) for pair in result.pairwise_matches
    } == {(0, 1), (0, 2), (1, 2)}


def test_sequential_matching_uses_adjacent_views():
    image = make_textured_image()
    result = analyze_image_set(
        [image.copy(), image.copy(), image.copy()],
        pairing_strategy="sequential",
    )

    assert result.pair_count == 2
    assert {
        (pair.first_view, pair.second_view) for pair in result.pairwise_matches
    } == {(0, 1), (1, 2)}


def test_identical_views_produce_multi_view_tracks():
    image = make_textured_image()
    result = analyze_image_set(
        [image.copy(), image.copy(), image.copy()],
        pairing_strategy="all",
    )

    assert result.total_feature_count > 0
    assert result.total_match_count > 0
    assert result.track_count > 0
    assert any(track.view_count == 3 for track in result.tracks)


def test_blank_images_produce_empty_feature_graph():
    blank = Image.new("RGB", (320, 240), color="white")
    result = analyze_image_set([blank.copy(), blank.copy(), blank.copy()])

    assert result.total_feature_count == 0
    assert result.total_match_count == 0
    assert result.track_count == 0


def test_invalid_pairing_strategy():
    image = make_textured_image()
    with pytest.raises(ConfigurationError):
        analyze_image_set([image.copy(), image.copy()], pairing_strategy="random")


def test_invalid_scene_minimum_track_length():
    image = make_textured_image()
    with pytest.raises(ConfigurationError):
        analyze_image_set([image.copy(), image.copy()], min_track_length=1)


def test_image_sizes_are_recorded():
    image = make_textured_image()
    result = analyze_image_set([image.copy(), image.copy()])
    assert result.image_sizes == ((320, 240), (320, 240))


def test_analyze_image_set_with_masks():
    image = make_textured_image()
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:200, 40:280] = 255

    result = analyze_image_set([image.copy(), image.copy()], masks=[mask, mask])
    assert result.total_feature_count > 0
    assert result.track_count > 0


def test_analyze_image_set_rejects_mismatched_masks():
    image = make_textured_image()
    mask = np.ones((240, 320), dtype=np.uint8) * 255

    with pytest.raises(InputError, match="Mask count"):
        analyze_image_set([image.copy(), image.copy()], masks=[mask])
