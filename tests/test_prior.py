import importlib.util

import numpy as np
import pytest
from PIL import Image

from img2mesh import cli
from img2mesh.benchmark import get_object, ground_truth_mesh, render_views
from img2mesh.benchmark.metrics import mesh_topology, sample_surface, shape_scores
from img2mesh.benchmark.run import _align, simulated_learned_prior
from img2mesh.config import ConfigurationError, InputError
from img2mesh.io import export_glb, export_obj, export_stl, load_mesh
from img2mesh.learned import (
    available_backends,
    foreground_rgba,
    generate_prior_mesh,
    register_backend,
)
from img2mesh.mesh import MeshResult
from img2mesh.silhouette import reconstruct_visual_hull


@pytest.fixture(scope="module")
def duck():
    obj = get_object("toy_duck")
    views = render_views(obj, size=160)
    gt = ground_truth_mesh(obj, resolution=96)
    return obj, views, gt


def _mesh(vertices, faces):
    return MeshResult(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        alpha=None,
        input_point_count=len(vertices),
        retained_point_count=len(vertices),
    )


def test_load_mesh_round_trips_every_export_format(tmp_path, duck):
    _, _, (verts, faces) = duck
    mesh = _mesh(verts, faces)
    for exporter, name in (
        (export_obj, "m.obj"),
        (export_stl, "m.stl"),
        (export_glb, "m.glb"),
    ):
        loaded = load_mesh(exporter(mesh, tmp_path / name))
        assert loaded.face_count == mesh.face_count
        assert loaded.vertex_count == mesh.vertex_count
        # Same triangles (STL re-welds vertices, so compare by position).
        a = np.sort(np.round(verts[faces], 4).reshape(len(faces), -1), axis=0)
        b = np.sort(
            np.round(loaded.vertices[loaded.faces], 4).reshape(len(faces), -1), axis=0
        )
        np.testing.assert_allclose(a, b, atol=2e-4)


def test_load_obj_handles_polygons_negative_indices_and_errors(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\n"
        "f 1/1 2/2 3/3 4/4\nf -1 1 2\nf -1 2 3\nf -1 3 4\n"
    )
    mesh = load_mesh(path)
    assert mesh.face_count == 5
    assert mesh.faces.max() == 4
    with pytest.raises(InputError):
        load_mesh(tmp_path / "missing.glb")
    (tmp_path / "bad.ply").write_text("ply")
    with pytest.raises(InputError, match="Unsupported"):
        load_mesh(tmp_path / "bad.ply")


def test_foreground_rgba_crops_and_masks():
    image = Image.new("RGB", (64, 48), "white")
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:30, 20:40] = True
    rgba = foreground_rgba(image, mask)
    alpha = np.asarray(rgba)[:, :, 3]
    assert rgba.mode == "RGBA"
    assert rgba.width < 64 and rgba.height < 48
    assert alpha.max() == 255 and alpha.min() == 0


def test_learned_backend_registry_and_missing_dependencies():
    assert "triposr" in available_backends()
    image = Image.new("RGB", (32, 32), "white")
    mask = np.ones((32, 32), dtype=bool)
    with pytest.raises(ConfigurationError, match="Unknown"):
        generate_prior_mesh(image, mask, backend="no-such-model")
    if importlib.util.find_spec("tsr") is None:
        with pytest.raises(ConfigurationError, match="TripoSR"):
            generate_prior_mesh(image, mask, backend="triposr")


def test_prior_from_learned_backend_beats_blind_carving(duck):
    obj, views, (gt_v, gt_f) = duck
    train = views[:-1]
    calls = []

    def fake_factory():
        def run(rgba):
            calls.append(rgba.size)
            return simulated_learned_prior(gt_v, gt_f)

        return run

    register_backend("fake-generator", fake_factory)
    images = [v.image for v in train]
    prior = reconstruct_visual_hull(
        images, resolution=56, learned_backend="fake-generator", prior_up_axis="+z"
    )
    blind = reconstruct_visual_hull(images, resolution=56, shape_prior="none")
    assert prior.method == "prior" and len(calls) == 1
    assert prior.mesh.vertex_colors is not None
    assert mesh_topology(prior.mesh.faces, prior.mesh.vertex_count)["genus"] == 0

    gt_pts = sample_surface(gt_v, gt_f, 8000, seed=1)
    diagonal = float(np.linalg.norm(np.ptp(gt_v, axis=0)))

    def fscore(result):
        pts = sample_surface(result.mesh.vertices, result.mesh.faces, 8000, seed=2)
        s, r, t = _align(pts, gt_pts, np.eye(3), diagonal)
        return shape_scores(s * pts @ r.T + t, gt_pts, diagonal, 0.02).fscore

    assert fscore(prior) > 0.8
    assert fscore(prior) > fscore(blind) + 0.2


def test_cli_prior_options(tmp_path, duck):
    _, views, (gt_v, gt_f) = duck
    prior_path = export_obj(_mesh(gt_v, gt_f), tmp_path / "prior.obj")
    for idx, view in enumerate(views[:3]):
        view.image.save(tmp_path / f"v{idx}.png")
    args = cli.build_parser().parse_args(
        ["in", "out.glb", "--prior-mesh", str(prior_path), "--prior-up-axis", "+y"]
    )
    assert args.prior_up_axis == "+y"
    code = cli.main(
        [str(tmp_path), str(tmp_path / "o.glb"), "--prior-mesh", str(prior_path)]
    )
    assert code == cli.EXIT_ERROR  # default sfm mode does not take a prior
