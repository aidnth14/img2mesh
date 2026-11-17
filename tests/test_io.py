import json
import struct

import numpy as np
import pytest
from PIL import Image

from img2mesh.config import (
    DEFAULT_CONFIG,
    AppConfig,
    ConfigurationError,
    ExportError,
    InputError,
)
from img2mesh.io import (
    export_glb,
    export_mesh,
    export_obj,
    export_stl,
    extract_exif_focal_length,
    load_image,
)
from img2mesh.mesh import MeshResult


def test_default_configuration():
    assert DEFAULT_CONFIG.log_level == "INFO"
    assert DEFAULT_CONFIG.max_image_pixels == 100_000_000
    assert DEFAULT_CONFIG.keep_intermediate_files is False


def test_custom_configuration(tmp_path):
    config = AppConfig(
        log_level="DEBUG",
        max_image_pixels=50_000_000,
        cache_dir=tmp_path,
        keep_intermediate_files=True,
    )
    assert config.log_level == "DEBUG"
    assert config.max_image_pixels == 50_000_000
    assert config.cache_dir == tmp_path
    assert config.keep_intermediate_files is True


def test_log_level_is_normalized():
    config = AppConfig(log_level="debug")
    assert config.log_level == "DEBUG"


@pytest.mark.parametrize("level", ["", "TRACE", "INVALID", None, 123])
def test_invalid_log_level(level):
    with pytest.raises(ConfigurationError):
        AppConfig(log_level=level)


@pytest.mark.parametrize("pixel_limit", [0, -1, True, 1.5])
def test_invalid_config_pixel_limit(pixel_limit):
    with pytest.raises(ConfigurationError):
        AppConfig(max_image_pixels=pixel_limit)


def test_invalid_intermediate_file_setting():
    with pytest.raises(ConfigurationError):
        AppConfig(keep_intermediate_files="yes")


@pytest.mark.parametrize(
    ("extension", "image_format"),
    [
        (".png", "PNG"),
        (".jpg", "JPEG"),
        (".bmp", "BMP"),
        (".tiff", "TIFF"),
    ],
)
def test_supported_image_formats(tmp_path, extension, image_format):
    path = tmp_path / f"sample{extension}"
    Image.new("RGB", (12, 8), color="red").save(path, format=image_format)

    with load_image(path) as image:
        assert image.size == (12, 8)
        assert image.mode == "RGB"


def test_transparency_is_preserved(tmp_path):
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (10, 10), color=(255, 0, 0, 120)).save(path)

    with load_image(path) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 120


def test_missing_image(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        load_image(tmp_path / "missing.png")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "image.gif"
    path.write_bytes(b"test data")

    with pytest.raises(InputError, match="Unsupported image extension"):
        load_image(path)


def test_corrupted_image(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not an actual image")

    with pytest.raises(InputError, match="Unable to decode"):
        load_image(path)


def test_unsupported_actual_image_format(tmp_path):
    path = tmp_path / "animated.png"
    Image.new("RGB", (5, 5)).save(path, format="GIF")

    with pytest.raises(InputError, match="Unsupported actual image format"):
        load_image(path)


def test_image_pixel_limit(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (100, 100)).save(path)

    with pytest.raises(InputError, match="exceeds the pixel limit"):
        load_image(path, max_image_pixels=1000)


def test_invalid_loader_pixel_limit(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (5, 5)).save(path)

    with pytest.raises(ConfigurationError):
        load_image(path, max_image_pixels=-1)


def tetrahedron():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    return MeshResult(
        vertices=vertices,
        faces=faces,
        alpha=None,
        input_point_count=4,
        retained_point_count=4,
    )


def test_export_obj(tmp_path):
    path = export_obj(tetrahedron(), tmp_path / "model.obj")
    content = path.read_text(encoding="utf-8")

    assert content.startswith("# Img2Mesh OBJ export")
    assert content.count("\nv ") == 4
    assert content.count("\nf ") == 4
    assert "f 1 3 2" in content


def test_export_binary_stl(tmp_path):
    path = export_stl(tetrahedron(), tmp_path / "model.stl")
    data = path.read_bytes()

    assert len(data) == 84 + 50 * 4
    assert data.startswith(b"Img2Mesh binary STL")
    assert struct.unpack_from("<I", data, 80)[0] == 4


def test_export_glb(tmp_path):
    path = export_glb(tetrahedron(), tmp_path / "model.glb")
    data = path.read_bytes()

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(data)

    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON"

    document = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    assert document["asset"]["version"] == "2.0"
    assert document["meshes"][0]["primitives"][0]["mode"] == 4
    assert document["accessors"][0]["count"] == 4
    assert document["accessors"][2]["count"] == 12


def test_export_mesh_dispatches_all_formats(tmp_path):
    mesh = tetrahedron()
    for extension in (".obj", ".stl", ".glb"):
        path = export_mesh(mesh, tmp_path / f"model{extension}")
        assert path.is_file()
        assert path.stat().st_size > 0


def test_prevents_overwrite_by_default(tmp_path):
    path = tmp_path / "existing.obj"
    path.write_text("original", encoding="utf-8")

    with pytest.raises(ExportError, match="already exists"):
        export_obj(tetrahedron(), path)

    assert path.read_text(encoding="utf-8") == "original"


def test_allows_explicit_overwrite(tmp_path):
    path = tmp_path / "existing.obj"
    path.write_text("original", encoding="utf-8")

    export_obj(tetrahedron(), path, overwrite=True)
    assert path.read_text(encoding="utf-8").startswith("# Img2Mesh OBJ export")


def test_rejects_unsupported_format(tmp_path):
    with pytest.raises(ExportError, match="Unsupported"):
        export_mesh(tetrahedron(), tmp_path / "model.ply")


def test_rejects_invalid_face_indices(tmp_path):
    mesh = tetrahedron()
    invalid = MeshResult(
        vertices=mesh.vertices,
        faces=np.array([[0, 1, 9]]),
        alpha=None,
        input_point_count=4,
        retained_point_count=4,
    )

    with pytest.raises(ExportError, match="outside"):
        export_obj(invalid, tmp_path / "bad.obj")


def test_rejects_non_finite_vertices(tmp_path):
    mesh = tetrahedron()
    vertices = mesh.vertices.copy()
    vertices[0, 0] = np.nan

    invalid = MeshResult(
        vertices=vertices,
        faces=mesh.faces,
        alpha=None,
        input_point_count=4,
        retained_point_count=4,
    )

    with pytest.raises(ExportError, match="finite"):
        export_glb(invalid, tmp_path / "bad.glb")


def test_rejects_degenerate_triangles(tmp_path):
    mesh = tetrahedron()
    faces = mesh.faces.copy()
    faces[0] = [0, 1, 1]

    invalid = MeshResult(
        vertices=mesh.vertices,
        faces=faces,
        alpha=None,
        input_point_count=4,
        retained_point_count=4,
    )

    with pytest.raises(ExportError, match="degenerate"):
        export_stl(invalid, tmp_path / "bad.stl")


def test_export_glb_handles_cancelling_vertex_normals(tmp_path):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    mesh = MeshResult(
        vertices=vertices,
        faces=faces,
        alpha=None,
        input_point_count=3,
        retained_point_count=3,
    )

    path = export_glb(mesh, tmp_path / "double_sided.glb")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_extract_exif_focal_length_and_scaling(tmp_path):
    path = tmp_path / "photo.jpg"
    img = Image.new("RGB", (400, 300), color="gray")
    exif = Image.Exif()
    exif[0xA405] = 28  # FocalLengthIn35mmFilm
    img.save(path, format="JPEG", exif=exif.tobytes())

    focal = extract_exif_focal_length(path)
    assert focal is not None
    assert focal == pytest.approx((28.0 / 36.0) * 400.0)

    scaled = extract_exif_focal_length(path, target_size=(200, 150))
    assert scaled == pytest.approx((28.0 / 36.0) * 200.0)
