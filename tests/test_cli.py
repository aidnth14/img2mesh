from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from img2mesh import cli
from img2mesh.cli import (
    EXIT_ERROR,
    EXIT_RECONSTRUCTION_NOT_IMPLEMENTED,
    EXIT_SUCCESS,
    detect_output_format,
    main,
    validate_input_image,
    validate_output_path,
)
from img2mesh.config import InputError, OutputError


def make_image(path: Path, color=(120, 80, 40)):
    Image.new("RGB", (64, 64), color).save(path)
    return path


@pytest.mark.parametrize(
    "extension",
    [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
)
def test_supported_input_images(tmp_path, extension):
    image = tmp_path / f"image{extension}"
    Image.new("RGB", (8, 8), color="red").save(image)
    assert validate_input_image(image) == image


def test_missing_input_image(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        validate_input_image(tmp_path / "missing.png")


def test_input_path_must_be_file(tmp_path):
    with pytest.raises(InputError, match="not a file"):
        validate_input_image(tmp_path)


def test_unsupported_input_format(tmp_path):
    image = tmp_path / "image.gif"
    image.write_bytes(b"test data")
    with pytest.raises(InputError, match="Unsupported image extension"):
        validate_input_image(image)


def test_empty_input_image(tmp_path):
    image = tmp_path / "empty.png"
    image.touch()
    with pytest.raises(InputError, match="empty"):
        validate_input_image(image)


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        ("model.glb", "glb"),
        ("model.obj", "obj"),
        ("model.stl", "stl"),
    ],
)
def test_output_format_detection(tmp_path, filename, expected_format):
    output = tmp_path / filename
    assert detect_output_format(output) == expected_format


def test_output_format_detection_is_case_insensitive(tmp_path):
    assert detect_output_format(tmp_path / "MODEL.GLB") == "glb"


def test_unsupported_output_format(tmp_path):
    with pytest.raises(OutputError, match="Unsupported output format"):
        detect_output_format(tmp_path / "model.fbx")


def test_output_directory_must_exist(tmp_path):
    output = tmp_path / "missing" / "model.glb"
    with pytest.raises(OutputError, match="does not exist"):
        validate_output_path(output)


def test_existing_output_requires_overwrite(tmp_path):
    output = tmp_path / "model.glb"
    output.write_bytes(b"existing model")
    with pytest.raises(OutputError, match="already exists"):
        validate_output_path(output)


def test_overwrite_allows_existing_output(tmp_path):
    output = tmp_path / "model.glb"
    output.write_bytes(b"existing model")
    assert validate_output_path(output, overwrite=True) == output


def test_cli_validate_only(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="red").save(image)
    output = tmp_path / "output.glb"

    result = main([str(image), str(output), "--validate-only"])
    assert result == EXIT_SUCCESS
    assert not output.exists()


def test_cli_reports_insufficient_views(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="red").save(image)
    output = tmp_path / "output.glb"

    result = main([str(image), str(output)])
    assert result == EXIT_RECONSTRUCTION_NOT_IMPLEMENTED
    assert not output.exists()


def test_cli_rejects_invalid_input(tmp_path):
    output = tmp_path / "output.glb"
    result = main([str(tmp_path / "missing.png"), str(output), "--validate-only"])
    assert result == EXIT_ERROR


def test_discovers_images_in_directory(tmp_path):
    make_image(tmp_path / "b.png")
    make_image(tmp_path / "a.jpg")
    (tmp_path / "._a.jpg").write_bytes(b"appledouble metadata")
    (tmp_path / ".hidden.png").write_bytes(b"hidden file")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    images = cli.discover_input_images(tmp_path)
    assert [p.name for p in images] == ["a.jpg", "b.png"]


def test_discovers_images_recursively(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    make_image(nested / "inside.png")

    assert cli.discover_input_images(tmp_path, recursive=True)[0].name == "inside.png"
    with pytest.raises(InputError, match="No supported images"):
        cli.discover_input_images(tmp_path, recursive=False)


def test_discovers_single_image_file(tmp_path):
    image = make_image(tmp_path / "one.png")
    assert cli.discover_input_images(image) == (image,)


def test_missing_input_is_rejected(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        cli.discover_input_images(tmp_path / "missing")


def test_directory_without_images_is_rejected(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing", encoding="utf-8")
    with pytest.raises(InputError, match="No supported images"):
        cli.discover_input_images(tmp_path)


def test_validate_only_decodes_directory_images(tmp_path):
    make_image(tmp_path / "one.png")
    make_image(tmp_path / "two.png")
    output = tmp_path / "model.glb"

    code = cli.main([str(tmp_path), str(output), "--validate-only"])
    assert code == cli.EXIT_SUCCESS
    assert not output.exists()


def test_single_photo_requires_multi_view_input(tmp_path):
    image = make_image(tmp_path / "one.png")
    output = tmp_path / "model.glb"

    code = cli.main([str(image), str(output)])
    assert code == cli.EXIT_INSUFFICIENT_VIEWS
    assert not output.exists()


def test_cli_connects_reconstruction_mesh_and_export(tmp_path, monkeypatch):
    make_image(tmp_path / "one.png")
    make_image(tmp_path / "two.png")
    output = tmp_path / "model.glb"

    calls = {}

    def fake_reconstruct(images, **kwargs):
        calls["image_count"] = len(images)
        calls["reconstruction_options"] = kwargs
        return SimpleNamespace(
            camera_count=2,
            point_count=20,
            points_3d=np.ones((8, 3), dtype=np.float64),
        )

    def fake_mesh(points, **kwargs):
        calls["mesh_points"] = len(points)
        calls["mesh_options"] = kwargs
        return SimpleNamespace(vertex_count=8, face_count=12)

    def fake_export(mesh, path, **kwargs):
        calls["export_path"] = Path(path)
        calls["export_options"] = kwargs
        Path(path).write_bytes(b"test model")
        return Path(path)

    monkeypatch.setattr(cli, "reconstruct_image_set", fake_reconstruct)
    monkeypatch.setattr(cli, "generate_mesh", fake_mesh)
    monkeypatch.setattr(cli, "export_mesh", fake_export)

    code = cli.main(
        [
            str(tmp_path),
            str(output),
            "--method",
            "sift",
            "--max-features",
            "1200",
            "--pairing-strategy",
            "sequential",
            "--alpha",
            "2.5",
            "--no-bundle-adjustment",
        ]
    )

    assert code == cli.EXIT_SUCCESS
    assert output.is_file()
    assert calls["image_count"] == 2
    assert calls["reconstruction_options"]["method"] == "sift"
    assert calls["reconstruction_options"]["max_features"] == 1200
    assert calls["reconstruction_options"]["pairing_strategy"] == "sequential"
    assert calls["reconstruction_options"]["run_bundle_adjustment"] is False
    assert calls["mesh_options"]["alpha"] == 2.5


def test_cli_rejects_existing_output_without_overwrite(tmp_path):
    make_image(tmp_path / "one.png")
    make_image(tmp_path / "two.png")
    output = tmp_path / "model.obj"
    output.write_text("existing", encoding="utf-8")

    code = cli.main([str(tmp_path), str(output), "--validate-only"])
    assert code == cli.EXIT_ERROR
    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_accepts_explicit_overwrite_for_validation(tmp_path):
    make_image(tmp_path / "one.png")
    output = tmp_path / "model.obj"
    output.write_text("existing", encoding="utf-8")

    code = cli.main(
        [str(tmp_path / "one.png"), str(output), "--validate-only", "--overwrite"]
    )
    assert code == cli.EXIT_SUCCESS
    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_applies_preprocessing_and_focal_length(tmp_path, monkeypatch):
    Image.new("RGB", (120, 80), (100, 120, 140)).save(tmp_path / "one.png")
    Image.new("RGB", (120, 80), (110, 130, 150)).save(tmp_path / "two.png")
    output = tmp_path / "model.glb"

    captured = {}

    def fake_reconstruct(images, **kwargs):
        captured["sizes"] = [img.size for img in images]
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            camera_count=2,
            point_count=10,
            points_3d=np.ones((8, 3), dtype=np.float64),
        )

    monkeypatch.setattr(cli, "reconstruct_image_set", fake_reconstruct)
    monkeypatch.setattr(
        cli,
        "generate_mesh",
        lambda points, **kwargs: SimpleNamespace(vertex_count=8, face_count=12),
    )
    monkeypatch.setattr(
        cli,
        "export_mesh",
        lambda mesh, path, **kwargs: Path(path).write_bytes(b"glb") or Path(path),
    )

    code = cli.main(
        [
            str(tmp_path),
            str(output),
            "--max-dimension",
            "60",
            "--normalize-contrast",
            "--denoise",
            "gaussian",
            "--focal-length",
            "500",
        ]
    )

    assert code == cli.EXIT_SUCCESS
    assert captured["sizes"] == [(60, 40), (60, 40)]
    intrinsics = captured["kwargs"]["intrinsics"]
    assert intrinsics.image_width == 60
    assert intrinsics.image_height == 40
    assert intrinsics.fx == 500.0


def test_cli_auto_alpha_and_segment_foreground(tmp_path, monkeypatch):
    Image.new("RGB", (64, 64), (10, 10, 10)).save(tmp_path / "one.png")
    Image.new("RGB", (64, 64), (15, 15, 15)).save(tmp_path / "two.png")
    output = tmp_path / "model.glb"

    captured = {}

    def fake_reconstruct(images, **kwargs):
        captured["recon"] = kwargs
        return SimpleNamespace(
            camera_count=2,
            point_count=10,
            points_3d=np.ones((8, 3), dtype=np.float64),
        )

    def fake_mesh(points, **kwargs):
        captured["mesh"] = kwargs
        return SimpleNamespace(vertex_count=8, face_count=12)

    monkeypatch.setattr(cli, "reconstruct_image_set", fake_reconstruct)
    monkeypatch.setattr(cli, "generate_mesh", fake_mesh)
    monkeypatch.setattr(
        cli,
        "export_mesh",
        lambda mesh, path, **kwargs: Path(path).write_bytes(b"glb") or Path(path),
    )

    code = cli.main(
        [
            str(tmp_path),
            str(output),
            "--auto-alpha",
            "--segment-foreground",
            "otsu",
        ]
    )
    assert code == cli.EXIT_SUCCESS
    assert captured["mesh"]["auto_alpha"] is True
    assert captured["mesh"]["alpha"] is None


def test_cli_rejects_alpha_and_auto_alpha_together(tmp_path):
    make_image(tmp_path / "one.png")
    make_image(tmp_path / "two.png")
    output = tmp_path / "model.glb"
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path), str(output), "--alpha", "1.0", "--auto-alpha"])
