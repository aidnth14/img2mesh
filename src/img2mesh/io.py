from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import (
    DEFAULT_CONFIG,
    ConfigurationError,
    ExportError,
    InputError,
)
from .mesh import MeshResult

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
)

SUPPORTED_PIL_FORMATS = {
    "PNG",
    "JPEG",
    "BMP",
    "TIFF",
    "WEBP",
}


def validate_image_path(path: str | Path) -> Path:
    image_path = Path(path).expanduser()

    if not image_path.exists():
        raise InputError(f"Input image does not exist: {image_path}")

    if not image_path.is_file():
        raise InputError(f"Input path is not a file: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise InputError(
            f"Unsupported image extension '{image_path.suffix}'. "
            f"Supported extensions: {supported}."
        )

    try:
        if image_path.stat().st_size == 0:
            raise InputError(f"Input image is empty: {image_path}")
    except OSError as exc:
        raise InputError(f"Cannot access input image: {image_path}") from exc

    return image_path


def load_image(
    path: str | Path,
    *,
    max_image_pixels: int | None = None,
) -> Image.Image:
    image_path = validate_image_path(path)

    pixel_limit = (
        DEFAULT_CONFIG.max_image_pixels
        if max_image_pixels is None
        else max_image_pixels
    )

    if (
        isinstance(pixel_limit, bool)
        or not isinstance(pixel_limit, int)
        or pixel_limit <= 0
    ):
        raise ConfigurationError("Maximum image pixels must be a positive integer.")

    previous_max_pixels = Image.MAX_IMAGE_PIXELS

    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)

            with Image.open(image_path) as probe:
                width, height = probe.size
                if width * height > pixel_limit:
                    raise InputError(
                        f"Image exceeds the pixel limit of "
                        f"{pixel_limit:,} pixels: {width:,} x {height:,}."
                    )

                if probe.format not in SUPPORTED_PIL_FORMATS:
                    raise InputError(
                        f"Unsupported actual image format: {probe.format}."
                    )

                probe.verify()

            with Image.open(image_path) as source:
                width, height = source.size
                if width * height > pixel_limit:
                    raise InputError(
                        f"Image exceeds the pixel limit of {pixel_limit:,} pixels."
                    )

                oriented = ImageOps.exif_transpose(source)
                try:
                    oriented.load()
                    has_alpha = (
                        "A" in oriented.getbands() or "transparency" in oriented.info
                    )
                    target_mode = "RGBA" if has_alpha else "RGB"
                    return oriented.convert(target_mode)
                finally:
                    if oriented is not source:
                        oriented.close()

    except InputError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise InputError(f"Unable to decode image '{image_path}': {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels


def _exif_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def extract_exif_focal_length(
    path: str | Path,
    *,
    target_size: tuple[int, int] | None = None,
) -> float | None:
    """Estimate camera focal length in pixels from EXIF metadata, if present."""
    try:
        image_path = validate_image_path(path)
        with Image.open(image_path) as source:
            exif = source.getexif()
            if not exif:
                return None
            oriented = ImageOps.exif_transpose(source)
            try:
                width, height = oriented.size
            finally:
                if oriented is not source:
                    oriented.close()
    except (InputError, OSError, UnidentifiedImageError, ValueError):
        return None

    exif_ifd = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}

    focal_35mm = _exif_float(exif_ifd.get(41989) or exif.get(41989))
    focal_px = None
    if focal_35mm is not None:
        focal_px = (focal_35mm / 36.0) * max(width, height)
    else:
        focal_mm = _exif_float(exif_ifd.get(37386) or exif.get(37386))
        x_res = _exif_float(exif_ifd.get(41486) or exif.get(41486))
        res_unit = exif_ifd.get(41488) or exif.get(41488) or 2
        unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(res_unit)
        if focal_mm is not None and x_res is not None and unit_mm is not None:
            focal_px = focal_mm * (x_res / unit_mm)

    if focal_px is None or not math.isfinite(focal_px) or focal_px <= 0:
        return None

    if target_size is not None and width > 0 and height > 0:
        target_w, target_h = target_size
        scale = max(target_w, target_h) / max(width, height)
        focal_px *= scale

    return float(focal_px)


def _prepare_mesh(mesh: MeshResult):
    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExportError("Invalid mesh data.") from exc

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ExportError("Mesh vertices must have shape (N, 3).")

    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ExportError("Mesh must contain at least three finite vertices.")

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ExportError("Mesh faces must have shape (M, 3) and be non-empty.")

    if not np.issubdtype(faces.dtype, np.integer):
        raise ExportError("Mesh face indices must be integers.")

    faces = faces.astype(np.int64, copy=False)

    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ExportError("Mesh face index is outside the vertex array.")

    if len(np.unique(faces)) != len(vertices):
        raise ExportError("Every exported vertex must belong to a face.")

    triangles = vertices[faces]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )

    scale = float(np.max(np.ptp(vertices, axis=0)))
    threshold = np.finfo(np.float64).eps * max(scale * scale, 1e-300)

    if scale <= 0 or np.any(np.linalg.norm(cross_products, axis=1) <= threshold):
        raise ExportError("Mesh contains degenerate triangles.")

    return vertices, faces


def _float32_vertices(vertices):
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(vertices, dtype="<f4")

    if not np.isfinite(converted).all():
        raise ExportError("Mesh coordinates exceed the supported float32 range.")

    return converted


def _atomic_write(path, data: bytes, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise ExportError("overwrite must be boolean.")

    destination = Path(path).expanduser()

    if not destination.name:
        raise ExportError("Output path must specify a filename.")

    if not destination.parent.is_dir():
        raise ExportError("Output directory does not exist.")

    if destination.exists() and not overwrite:
        raise ExportError(
            f"Output file already exists: {destination}. "
            "Use overwrite=True to replace it."
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()

        return destination

    except FileExistsError as exc:
        raise ExportError(f"Output file already exists: {destination}.") from exc
    except OSError as exc:
        raise ExportError(f"Could not write output file: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def export_obj(mesh: MeshResult, output_path, *, overwrite=False) -> Path:
    vertices, faces = _prepare_mesh(mesh)

    lines = [
        "# Img2Mesh OBJ export",
        f"# Vertices: {len(vertices)}",
        f"# Faces: {len(faces)}",
    ]

    for vertex in vertices:
        lines.append("v " + " ".join(format(float(value), ".17g") for value in vertex))

    for face in faces:
        lines.append("f " + " ".join(str(int(index) + 1) for index in face))

    data = ("\n".join(lines) + "\n").encode("utf-8")
    return _atomic_write(output_path, data, overwrite)


def export_stl(mesh: MeshResult, output_path, *, overwrite=False) -> Path:
    vertices, faces = _prepare_mesh(mesh)
    vertices32 = _float32_vertices(vertices)

    if len(faces) > 0xFFFFFFFF:
        raise ExportError("STL cannot contain more than 2^32-1 triangles.")

    header = b"Img2Mesh binary STL".ljust(80, b"\0")
    output = bytearray(header)
    output.extend(struct.pack("<I", len(faces)))

    for face in faces:
        triangle = vertices32[face].astype(np.float64)
        normal = np.cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        )
        normal_length = np.linalg.norm(normal)
        if normal_length <= 0:
            raise ExportError("Cannot calculate a triangle normal.")

        normal /= normal_length
        values = [
            *normal.tolist(),
            *triangle.reshape(-1).tolist(),
        ]
        output.extend(struct.pack("<12fH", *values, 0))

    return _atomic_write(output_path, bytes(output), overwrite)


def _build_glb(vertices, faces) -> bytes:
    vertices32 = _float32_vertices(vertices)

    triangles = vertices32[faces].astype(np.float64)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )

    normals = np.zeros_like(vertices32, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1)

    zero_mask = lengths <= 1e-12
    if np.any(zero_mask):
        face_normal_lengths = np.linalg.norm(face_normals, axis=1)
        valid_faces = face_normal_lengths > 0
        for face_idx, face in enumerate(faces):
            if not valid_faces[face_idx]:
                continue
            unit_face_normal = face_normals[face_idx] / face_normal_lengths[face_idx]
            for vertex_idx in face:
                if zero_mask[vertex_idx]:
                    normals[vertex_idx] = unit_face_normal
                    zero_mask[vertex_idx] = False
        lengths = np.linalg.norm(normals, axis=1)

    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ExportError("Could not compute valid vertex normals.")

    normals = (normals / lengths[:, None]).astype("<f4")
    positions = np.ascontiguousarray(vertices32, dtype="<f4")
    indices = np.ascontiguousarray(faces.reshape(-1), dtype="<u4")

    binary = bytearray()

    def add_buffer_view(array, target):
        while len(binary) % 4:
            binary.append(0)

        offset = len(binary)
        raw = array.tobytes(order="C")
        binary.extend(raw)

        return {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
            "target": target,
        }

    position_view = add_buffer_view(positions, 34962)
    normal_view = add_buffer_view(normals, 34962)
    index_view = add_buffer_view(indices, 34963)

    document = {
        "asset": {
            "version": "2.0",
            "generator": "Img2Mesh",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "name": "Img2Mesh Mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                        },
                        "indices": 2,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [position_view, normal_view, index_view],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
    }

    json_chunk = json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)

    binary_chunk = bytes(binary)
    binary_chunk += b"\0" * ((-len(binary_chunk)) % 4)

    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)

    return b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
            binary_chunk,
        ]
    )


def export_glb(mesh: MeshResult, output_path, *, overwrite=False) -> Path:
    vertices, faces = _prepare_mesh(mesh)
    data = _build_glb(vertices, faces)
    return _atomic_write(output_path, data, overwrite)


def export_mesh(mesh: MeshResult, output_path, *, overwrite=False) -> Path:
    suffix = Path(output_path).expanduser().suffix.lower()

    exporters = {
        ".obj": export_obj,
        ".stl": export_stl,
        ".glb": export_glb,
    }

    exporter = exporters.get(suffix)
    if exporter is None:
        raise ExportError("Unsupported output format. Choose .obj, .stl, or .glb.")

    return exporter(mesh, output_path, overwrite=overwrite)
