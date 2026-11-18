from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from .config import Img2MeshError, InputError, OutputError, ReconstructionError
from .geometry import CameraIntrinsics
from .image import (
    PreprocessingOptions,
    clean_mask,
    largest_component,
    preprocess_image,
    segment_foreground,
)
from .io import (
    SUPPORTED_IMAGE_EXTENSIONS,
    export_mesh,
    extract_exif_focal_length,
    load_image,
    load_mesh,
    validate_image_path,
)
from .mesh import generate_mesh
from .pipeline import reconstruct_image_set
from .silhouette import reconstruct_visual_hull

__version__ = "0.1.0"

LOGGER = logging.getLogger("img2mesh")

SUPPORTED_OUTPUT_FORMATS = {
    ".glb": "glb",
    ".obj": "obj",
    ".stl": "stl",
}

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INSUFFICIENT_VIEWS = 3
EXIT_RECONSTRUCTION_NOT_IMPLEMENTED = EXIT_INSUFFICIENT_VIEWS


def validate_input_image(path: str | Path) -> Path:
    return validate_image_path(path)


def discover_input_images(
    path: str | Path,
    *,
    recursive: bool = False,
) -> tuple[Path, ...]:
    source = Path(path).expanduser()

    if source.is_file():
        return (validate_input_image(source),)

    if not source.is_dir():
        raise InputError(f"Input image file or directory does not exist: {source}")

    try:
        candidates = source.rglob("*") if recursive else source.iterdir()
        image_paths = sorted(
            (
                item
                for item in candidates
                if item.is_file()
                and not item.name.startswith(".")
                and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda item: str(item).casefold(),
        )
    except OSError as exc:
        raise InputError(f"Could not read input directory '{source}': {exc}") from exc

    if not image_paths:
        raise InputError(
            f"No supported images found in '{source}'. "
            "Supported formats: PNG, JPEG, BMP, TIFF, and WEBP."
        )

    return tuple(validate_input_image(item) for item in image_paths)


def detect_output_format(path: str | Path) -> str:
    extension = Path(path).expanduser().suffix.lower()

    if extension not in SUPPORTED_OUTPUT_FORMATS:
        supported = ", ".join(SUPPORTED_OUTPUT_FORMATS.values())
        raise OutputError(
            f"Unsupported output format '{extension}'. Supported formats: {supported}."
        )

    return SUPPORTED_OUTPUT_FORMATS[extension]


def validate_output_path(
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(path).expanduser()

    if output_path.exists() and output_path.is_dir():
        raise OutputError(f"Output destination is a directory: {output_path}")

    detect_output_format(output_path)

    output_directory = output_path.parent
    if not output_directory.exists():
        raise OutputError(f"Output directory does not exist: {output_directory}")

    if not output_directory.is_dir():
        raise OutputError(f"Output parent is not a directory: {output_directory}")

    if output_path.exists() and not overwrite:
        raise OutputError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    return output_path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive integer.") from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")

    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be a positive number.") from exc

    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number.")

    return result


def _view_angles(value: str) -> list[tuple[float, float]]:
    angles = []
    for item in value.split(","):
        try:
            az, el = (float(part) for part in item.split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Use azimuth:elevation pairs separated by commas, e.g. 0:10,90:10."
            ) from exc
        if not (math.isfinite(az) and math.isfinite(el)) or abs(el) > 90.0:
            raise argparse.ArgumentTypeError("Elevations must lie within [-90, 90].")
        angles.append((az, el))
    return angles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img2mesh",
        description="Reconstruct a sparse 3D mesh from overlapping photographs.",
        epilog="Example: img2mesh ./photos ./model.glb",
    )

    parser.add_argument(
        "input_image",
        type=Path,
        help="Input image file or directory of overlapping photographs.",
    )
    parser.add_argument(
        "output_model",
        type=Path,
        help="Output mesh path (.glb, .obj, or .stl).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Decode and validate inputs without running reconstruction.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--method",
        choices=("orb", "sift"),
        default="orb",
        help="Feature detector (default: orb).",
    )
    parser.add_argument(
        "--max-features",
        type=_positive_int,
        default=2000,
        help="Maximum features detected per image (default: 2000).",
    )
    parser.add_argument(
        "--pairing-strategy",
        choices=("all", "sequential"),
        default="all",
        help="Image pairing strategy (default: all).",
    )
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--alpha",
        type=_positive_float,
        default=None,
        help="Alpha-shape radius (omit for convex hull).",
    )
    alpha_group.add_argument(
        "--auto-alpha",
        action="store_true",
        help="Automatically estimate alpha-shape radius from point spacing.",
    )
    parser.add_argument(
        "--segment-foreground",
        choices=("none", "otsu", "adaptive"),
        default="none",
        help="Foreground segmentation mask for feature extraction (default: none).",
    )
    parser.add_argument(
        "--mode",
        choices=("sfm", "silhouette", "auto"),
        default="sfm",
        help="Reconstruction mode (default: sfm).",
    )
    parser.add_argument(
        "--shape-prior",
        choices=("auto", "revolution", "none"),
        default="auto",
        help=(
            "Silhouette shape model: 'revolution' rebuilds rotationally "
            "symmetric objects (cups, bowls, vases) as a smooth hollow body "
            "plus handle; 'none' uses plain visual-hull carving; 'auto' picks "
            "revolution when the views look symmetric (default: auto)."
        ),
    )
    parser.add_argument(
        "--view-angles",
        type=_view_angles,
        default=None,
        metavar="AZ:EL,...",
        help=(
            "Known camera angles per photo in input order, as azimuth:elevation "
            "degrees (elevation > 0 looks down), e.g. 0:10,45:10,90:10. Makes "
            "silhouette carving far more accurate for non-round objects."
        ),
    )
    parser.add_argument(
        "--prior-mesh",
        type=Path,
        default=None,
        help=(
            "Shape prior (.obj/.stl/.glb), e.g. from an image-to-3D generator, "
            "in any frame and scale. Cameras are fitted to it and it is refined "
            "to match every photo's silhouette (silhouette mode)."
        ),
    )
    parser.add_argument(
        "--prior-up-axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default=None,
        help="Up axis of the prior mesh, if known (default: detect).",
    )
    parser.add_argument(
        "--learned-prior",
        default=None,
        metavar="BACKEND",
        help=(
            "Generate the shape prior with a learned backend (e.g. 'triposr'; "
            "needs its optional dependencies)."
        ),
    )
    parser.add_argument(
        "--no-bundle-adjustment",
        action="store_true",
        help="Skip bundle adjustment refinement.",
    )
    parser.add_argument(
        "--focal-length",
        type=_positive_float,
        default=None,
        help="Camera focal length in pixels.",
    )
    parser.add_argument(
        "--max-dimension",
        type=_positive_int,
        default=None,
        help="Maximum image width or height in pixels.",
    )
    parser.add_argument(
        "--normalize-contrast",
        action="store_true",
        help="Stretch image contrast before feature detection.",
    )
    parser.add_argument(
        "--denoise",
        choices=("none", "median", "gaussian"),
        default="none",
        help="Denoising filter (default: none).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _close_images(images) -> None:
    for image in images:
        try:
            image.close()
        except Exception:
            LOGGER.warning("Could not close an input image cleanly.")


def _run_silhouette(
    images,
    reconstruction_kwargs,
    shape_prior: str,
    view_angles=None,
    prior_options=None,
):
    vh_kwargs = {"shape_prior": shape_prior, **(prior_options or {})}
    if view_angles is not None:
        if len(view_angles) != len(images):
            raise InputError(
                f"--view-angles lists {len(view_angles)} angle(s) for "
                f"{len(images)} image(s)."
            )
        vh_kwargs["view_angles"] = view_angles
    if "intrinsics" in reconstruction_kwargs:
        vh_kwargs["intrinsics"] = reconstruction_kwargs["intrinsics"]
    if "masks" in reconstruction_kwargs:
        vh_kwargs["masks"] = reconstruction_kwargs["masks"]
    vh_result = reconstruct_visual_hull(images, **vh_kwargs)
    LOGGER.info(
        "Silhouette reconstruction (%s) completed: mean IoU=%.2f%%, "
        "min IoU=%.2f%%, mean Dice=%.2f%%.",
        vh_result.method,
        vh_result.mean_iou * 100.0,
        vh_result.min_iou * 100.0,
        vh_result.mean_dice * 100.0,
    )
    return vh_result.mesh


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )
    LOGGER.setLevel(log_level)

    images = []

    try:
        image_paths = discover_input_images(
            args.input_image,
            recursive=args.recursive,
        )
        output_path = validate_output_path(
            args.output_model,
            overwrite=args.overwrite,
        )
        output_format = detect_output_format(output_path)

        preprocessing_options = None
        if (
            args.max_dimension is not None
            or args.normalize_contrast
            or args.denoise != "none"
        ):
            preprocessing_options = PreprocessingOptions(
                max_dimension=args.max_dimension,
                normalize_contrast=args.normalize_contrast,
                denoise=args.denoise,
            )

        LOGGER.info("Found %d input image(s).", len(image_paths))
        LOGGER.info("Output destination: %s", output_path)
        LOGGER.info("Output format: %s", output_format)

        for image_path in image_paths:
            image = load_image(image_path)
            if preprocessing_options is not None:
                try:
                    preprocessed = preprocess_image(image, preprocessing_options)
                finally:
                    image.close()
                image = preprocessed
            images.append(image)
            LOGGER.debug(
                "Loaded %s: %d x %d pixels",
                image_path,
                image.width,
                image.height,
            )

        if args.validate_only:
            LOGGER.info(
                "Validation successful: %d image(s) decoded. No model was generated.",
                len(images),
            )
            return EXIT_SUCCESS

        if len(images) < 2:
            LOGGER.error(
                "Multi-view reconstruction requires at least two overlapping "
                "photographs. Provide a directory containing two or more images."
            )
            return EXIT_INSUFFICIENT_VIEWS

        LOGGER.info("Starting feature extraction and camera reconstruction.")

        reconstruction_kwargs = {
            "method": args.method,
            "max_features": args.max_features,
            "pairing_strategy": args.pairing_strategy,
            "run_bundle_adjustment": not args.no_bundle_adjustment,
        }
        if args.focal_length is not None:
            reconstruction_kwargs["intrinsics"] = CameraIntrinsics.from_focal_length(
                images[0].width,
                images[0].height,
                args.focal_length,
            )
        else:
            exif_focal = extract_exif_focal_length(
                image_paths[0],
                target_size=images[0].size,
            )
            if exif_focal is not None:
                LOGGER.debug("Using EXIF focal length: %.2f px", exif_focal)
                reconstruction_kwargs["intrinsics"] = (
                    CameraIntrinsics.from_focal_length(
                        images[0].width,
                        images[0].height,
                        exif_focal,
                    )
                )

        if args.segment_foreground != "none":
            masks = []
            for image in images:
                raw_mask = segment_foreground(image, method=args.segment_foreground)
                cleaned_mask = largest_component(clean_mask(raw_mask))
                if not cleaned_mask.any():
                    cleaned_mask = None
                masks.append(cleaned_mask)
            if all(mask is not None for mask in masks):
                reconstruction_kwargs["masks"] = masks

        prior_options = {}
        if args.prior_mesh is not None:
            prior_options["prior_mesh"] = load_mesh(args.prior_mesh)
        if args.prior_up_axis is not None:
            prior_options["prior_up_axis"] = args.prior_up_axis
        if args.learned_prior is not None:
            prior_options["learned_backend"] = args.learned_prior
        if prior_options and args.mode == "sfm":
            raise InputError(
                "--prior-mesh/--learned-prior need --mode silhouette or auto."
            )

        if args.mode == "silhouette":
            mesh = _run_silhouette(
                images,
                reconstruction_kwargs,
                args.shape_prior,
                args.view_angles,
                prior_options,
            )
        else:
            reconstruction = None
            sfm_error = None
            try:
                reconstruction = reconstruct_image_set(
                    images,
                    **reconstruction_kwargs,
                )
            except ReconstructionError as exc:
                if args.mode != "auto":
                    raise
                sfm_error = exc

            if args.mode == "auto" and (
                reconstruction is None
                or reconstruction.camera_count < len(images)
                or reconstruction.point_count < 32
            ):
                LOGGER.info(
                    "Falling back to silhouette visual-hull reconstruction%s.",
                    f" ({sfm_error})" if sfm_error else "",
                )
                mesh = _run_silhouette(
                    images,
                    reconstruction_kwargs,
                    args.shape_prior,
                    args.view_angles,
                    prior_options,
                )
            else:
                LOGGER.info(
                    "Reconstruction completed: %d cameras and %d sparse 3D points.",
                    reconstruction.camera_count,
                    reconstruction.point_count,
                )
                LOGGER.info("Generating triangle mesh.")
                mesh_kwargs = {"alpha": args.alpha}
                if args.auto_alpha:
                    mesh_kwargs["auto_alpha"] = True
                mesh = generate_mesh(
                    reconstruction.points_3d,
                    **mesh_kwargs,
                )

        LOGGER.info(
            "Mesh generated: %d vertices and %d faces.",
            mesh.vertex_count,
            mesh.face_count,
        )

        export_mesh(
            mesh,
            output_path,
            overwrite=args.overwrite,
        )

        LOGGER.info("Model successfully exported to %s", output_path)
        return EXIT_SUCCESS

    except Img2MeshError as exc:
        LOGGER.error("%s", exc)
        return EXIT_ERROR

    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Processing failed: %s", exc)
        return EXIT_ERROR

    finally:
        _close_images(images)


if __name__ == "__main__":
    raise SystemExit(main())
