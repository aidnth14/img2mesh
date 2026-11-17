import numpy as np
import pytest
from PIL import Image

from img2mesh.config import ConfigurationError, InputError
from img2mesh.image import (
    PreprocessingOptions,
    clean_mask,
    largest_component,
    preprocess_image,
    segment_foreground,
)


def test_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (300, 150), color="red")
    options = PreprocessingOptions(max_dimension=120)
    result = preprocess_image(image, options)
    assert result.size == (120, 60)


def test_resize_does_not_enlarge():
    image = Image.new("RGB", (20, 10), color="red")
    options = PreprocessingOptions(max_dimension=100)
    result = preprocess_image(image, options)
    assert result.size == (20, 10)


def test_preserve_original_color_mode():
    image = Image.new("RGBA", (20, 20), color=(255, 0, 0, 100))
    result = preprocess_image(image)
    assert result.mode == "RGBA"


def test_convert_to_rgb():
    image = Image.new("RGBA", (20, 20), color=(255, 0, 0, 100))
    options = PreprocessingOptions(color_mode="RGB")
    result = preprocess_image(image, options)
    assert result.mode == "RGB"


def test_convert_to_rgba():
    image = Image.new("RGB", (20, 20), color="red")
    options = PreprocessingOptions(color_mode="RGBA")
    result = preprocess_image(image, options)
    assert result.mode == "RGBA"


def test_convert_to_grayscale():
    image = Image.new("RGB", (20, 20), color="red")
    options = PreprocessingOptions(color_mode="L")
    result = preprocess_image(image, options)
    assert result.mode == "L"


def test_contrast_normalization():
    image = Image.new("L", (10, 10))
    image.putdata([50] * 50 + [200] * 50)
    options = PreprocessingOptions(normalize_contrast=True)
    result = preprocess_image(image, options)
    assert result.getextrema() == (0, 255)


def test_gaussian_denoising():
    image = Image.new("RGB", (20, 20), color="red")
    options = PreprocessingOptions(denoise="gaussian", denoise_radius=1.0)
    result = preprocess_image(image, options)
    assert result.size == image.size
    assert result.mode == "RGB"


def test_median_denoising():
    image = Image.new("RGB", (20, 20), color="red")
    options = PreprocessingOptions(denoise="median")
    result = preprocess_image(image, options)
    assert result.size == image.size
    assert result.mode == "RGB"


def test_denoising_preserves_alpha():
    image = Image.new("RGBA", (20, 20), color=(255, 0, 0, 100))
    options = PreprocessingOptions(denoise="gaussian")
    result = preprocess_image(image, options)
    assert result.mode == "RGBA"
    assert result.getchannel("A").getextrema() == (100, 100)


def test_original_image_is_not_modified():
    image = Image.new("RGB", (300, 150), color="red")
    original_size = image.size
    original_pixel = image.getpixel((0, 0))

    options = PreprocessingOptions(max_dimension=100, color_mode="L")
    preprocess_image(image, options)

    assert image.size == original_size
    assert image.getpixel((0, 0)) == original_pixel


def test_result_is_independent_copy():
    image = Image.new("RGB", (10, 10), color="red")
    result = preprocess_image(image)
    result.putpixel((0, 0), (0, 0, 0))
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_invalid_image_input():
    with pytest.raises(InputError):
        preprocess_image("image.png")


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_max_dimension(dimension):
    with pytest.raises(ConfigurationError):
        PreprocessingOptions(max_dimension=dimension)


def test_invalid_color_mode():
    with pytest.raises(ConfigurationError):
        PreprocessingOptions(color_mode="CMYK")


def test_invalid_denoising_method():
    with pytest.raises(ConfigurationError):
        PreprocessingOptions(denoise="bilateral")


@pytest.mark.parametrize("radius", [0, -1, float("inf"), float("nan"), True])
def test_invalid_denoise_radius(radius):
    with pytest.raises(ConfigurationError):
        PreprocessingOptions(denoise_radius=radius)


@pytest.mark.parametrize("filter_size", [0, 2, 4, -3, 3.5, True])
def test_invalid_median_filter_size(filter_size):
    with pytest.raises(ConfigurationError):
        PreprocessingOptions(median_filter_size=filter_size)


def test_otsu_segmentation():
    image = Image.new("L", (100, 100))
    image.putdata([0] * 5000 + [255] * 5000)

    mask = segment_foreground(image, method="otsu")
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)).issubset({0, 255})
    assert 255 in mask
    assert 0 in mask


def test_otsu_inversion():
    image = Image.new("L", (100, 100))
    image.putdata([0] * 5000 + [255] * 5000)

    normal = segment_foreground(image, method="otsu")
    inverted = segment_foreground(image, method="otsu", invert=True)
    assert np.array_equal(inverted, 255 - normal)


def test_adaptive_segmentation():
    image = Image.new("L", (100, 100), color=128)
    mask = segment_foreground(image, method="adaptive", adaptive_block_size=31)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8


def test_invalid_segmentation_method():
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ConfigurationError):
        segment_foreground(image, method="invalid")


def test_invalid_segmentation_image():
    with pytest.raises(InputError):
        segment_foreground("not an image")


@pytest.mark.parametrize("block_size", [0, 2, 4, True, 3.5])
def test_invalid_adaptive_block_size(block_size):
    with pytest.raises(ConfigurationError):
        segment_foreground(
            Image.new("L", (20, 20)),
            method="adaptive",
            adaptive_block_size=block_size,
        )


def test_invalid_adaptive_constant():
    with pytest.raises(ConfigurationError):
        segment_foreground(
            Image.new("L", (20, 20)),
            method="adaptive",
            adaptive_c=float("nan"),
        )


def test_clean_mask_removes_small_noise():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[1, 1] = 255

    result = clean_mask(
        mask,
        kernel_size=3,
        opening_iterations=1,
        closing_iterations=0,
    )
    assert result[1, 1] == 0
    assert result[15, 15] == 255


def test_clean_mask_preserves_binary_values():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    result = clean_mask(mask, opening_iterations=0, closing_iterations=0)
    assert set(np.unique(result)).issubset({0, 255})


def test_invalid_mask():
    with pytest.raises(InputError):
        clean_mask(np.zeros((10, 10, 3), dtype=np.uint8))


def test_invalid_kernel_size():
    with pytest.raises(ConfigurationError):
        clean_mask(np.zeros((10, 10), dtype=np.uint8), kernel_size=2)


def test_invalid_morphology_iterations():
    with pytest.raises(ConfigurationError):
        clean_mask(np.zeros((10, 10), dtype=np.uint8), opening_iterations=-1)


def test_largest_component():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:20, 5:20] = 255
    mask[30:35, 30:35] = 255

    result = largest_component(mask)
    assert result[10, 10] == 255
    assert result[32, 32] == 0


def test_largest_component_minimum_area():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:8] = 255

    result = largest_component(mask, min_area=20)
    assert not np.any(result)


def test_largest_component_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    result = largest_component(mask)
    assert not np.any(result)


def test_invalid_minimum_area():
    with pytest.raises(ConfigurationError):
        largest_component(np.zeros((10, 10), dtype=np.uint8), min_area=0)
