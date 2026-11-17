from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import ConfigurationError, InputError

ColorMode = Literal["preserve", "RGB", "RGBA", "L"]
DenoiseMethod = Literal["none", "median", "gaussian"]


@dataclass(frozen=True, slots=True)
class PreprocessingOptions:
    max_dimension: int | None = None
    color_mode: ColorMode = "preserve"
    normalize_contrast: bool = False
    denoise: DenoiseMethod = "none"
    denoise_radius: float = 1.0
    median_filter_size: int = 3

    def __post_init__(self) -> None:
        if self.max_dimension is not None:
            if (
                isinstance(self.max_dimension, bool)
                or not isinstance(self.max_dimension, int)
                or self.max_dimension <= 0
            ):
                raise ConfigurationError(
                    "Maximum dimension must be a positive integer."
                )

        if self.color_mode not in {"preserve", "RGB", "RGBA", "L"}:
            raise ConfigurationError("Color mode must be preserve, RGB, RGBA, or L.")

        if not isinstance(self.normalize_contrast, bool):
            raise ConfigurationError("normalize_contrast must be True or False.")

        if self.denoise not in {"none", "median", "gaussian"}:
            raise ConfigurationError(
                "Denoise method must be none, median, or gaussian."
            )

        if (
            isinstance(self.denoise_radius, bool)
            or not isinstance(self.denoise_radius, (int, float))
            or not math.isfinite(self.denoise_radius)
            or self.denoise_radius <= 0
        ):
            raise ConfigurationError("Denoise radius must be a positive finite number.")

        if (
            isinstance(self.median_filter_size, bool)
            or not isinstance(self.median_filter_size, int)
            or self.median_filter_size < 3
            or self.median_filter_size % 2 == 0
        ):
            raise ConfigurationError(
                "Median filter size must be an odd integer of at least 3."
            )


def _resize_image(image: Image.Image, max_dimension: int | None) -> Image.Image:
    if max_dimension is None:
        return image

    width, height = image.size
    largest_dimension = max(width, height)
    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _apply_color_transform(image: Image.Image, transform) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb = image.convert("RGB")
        transformed = transform(rgb)
        transformed.putalpha(alpha)
        return transformed

    if image.mode in {"RGB", "L"}:
        return transform(image)

    return transform(image.convert("RGB"))


def _normalize_contrast(image: Image.Image) -> Image.Image:
    return _apply_color_transform(image, lambda source: ImageOps.autocontrast(source))


def _denoise_image(image: Image.Image, options: PreprocessingOptions) -> Image.Image:
    if options.denoise == "none":
        return image

    if options.denoise == "median":
        image_filter = ImageFilter.MedianFilter(size=options.median_filter_size)
    else:
        image_filter = ImageFilter.GaussianBlur(radius=options.denoise_radius)

    return _apply_color_transform(image, lambda source: source.filter(image_filter))


def preprocess_image(
    image: Image.Image,
    options: PreprocessingOptions | None = None,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise InputError("preprocess_image expects a PIL.Image.Image instance.")

    if options is None:
        options = PreprocessingOptions()

    processed = image.copy()
    processed = _resize_image(processed, options.max_dimension)

    if options.color_mode != "preserve":
        processed = processed.convert(options.color_mode)

    if options.normalize_contrast:
        processed = _normalize_contrast(processed)

    return _denoise_image(processed, options)


def _to_grayscale(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise InputError("Expected a PIL image.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def segment_foreground(
    image: Image.Image,
    *,
    method: str = "otsu",
    invert: bool = False,
    adaptive_block_size: int = 31,
    adaptive_c: float = 5.0,
) -> np.ndarray:
    if not isinstance(invert, bool):
        raise ConfigurationError("invert must be True or False.")

    if method not in {"otsu", "adaptive"}:
        raise ConfigurationError("Segmentation method must be 'otsu' or 'adaptive'.")

    if (
        isinstance(adaptive_block_size, bool)
        or not isinstance(adaptive_block_size, int)
        or adaptive_block_size < 3
        or adaptive_block_size % 2 == 0
    ):
        raise ConfigurationError(
            "Adaptive block size must be an odd integer of at least 3."
        )

    if (
        isinstance(adaptive_c, bool)
        or not isinstance(adaptive_c, (int, float))
        or not math.isfinite(adaptive_c)
    ):
        raise ConfigurationError("Adaptive C must be a finite number.")

    gray = _to_grayscale(image)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            adaptive_block_size,
            float(adaptive_c),
        )

    return mask


def clean_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise InputError("Mask must be a two-dimensional NumPy array.")

    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ConfigurationError("Kernel size must be a positive odd integer.")

    for name, value in (
        ("opening_iterations", opening_iterations),
        ("closing_iterations", closing_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"{name} must be a non-negative integer.")

    result = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if opening_iterations:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_OPEN,
            kernel,
            iterations=opening_iterations,
        )

    if closing_iterations:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=closing_iterations,
        )

    return result


def largest_component(
    mask: np.ndarray,
    *,
    min_area: int = 1,
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise InputError("Mask must be a two-dimensional NumPy array.")

    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 1:
        raise ConfigurationError("Minimum component area must be a positive integer.")

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if count <= 1:
        return np.zeros_like(binary)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])

    if largest_area < min_area:
        return np.zeros_like(binary)

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)
