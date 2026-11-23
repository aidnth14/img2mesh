import json
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from img2mesh.benchmark.meshdata import load_textured_obj, render_mesh_view
from img2mesh.benchmark.meshref import _decide, build, calibrate

CUBE_OBJ = """\
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3 4/4
f 5/1 8/4 7/3 6/2
f 1/1 5/2 6/3 2/4
f 2/1 6/2 7/3 3/4
f 3/1 7/2 8/3 4/4
f 5/1 1/2 4/3 8/4
"""


@pytest.fixture()
def raw_cube(tmp_path):
    folder = tmp_path / "raw" / "Test_Cube"
    folder.mkdir(parents=True)
    (folder / "model.obj").write_text(CUBE_OBJ)
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:, :4] = (200, 40, 40)
    texture[:, 4:] = (40, 40, 200)
    Image.fromarray(texture).save(folder / "texture.png")
    return folder


def test_load_textured_obj_normalises_and_renders(raw_cube):
    mesh = load_textured_obj(raw_cube / "model.obj", raw_cube / "texture.png")
    assert len(mesh.faces) == 12  # quads fan-triangulated
    assert mesh.vertices[:, 1].min() == pytest.approx(0.0)  # stands on y = 0
    assert np.max(np.ptp(mesh.vertices, axis=0)) == pytest.approx(1.0)

    view = render_mesh_view(mesh, 30.0, 20.0, size=96)
    rgb = np.asarray(view.image)
    assert view.mask.mean() > 0.1
    # White background, apart from anti-aliased pixels at the outline.
    near = cv2.dilate(view.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.all(rgb[~near] >= 250)
    inside = rgb[view.mask].astype(float)
    assert (inside[:, 0] > inside[:, 2] + 40).any() and (
        inside[:, 2] > inside[:, 0] + 40
    ).any()


def test_build_writes_photos_truth_metadata_and_split(raw_cube, tmp_path):
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps([{"name": "Test_Cube", "category": "Toys"}]))
    out = tmp_path / "meshref"
    build(raw_cube.parent, chosen, out, size=96)
    folder = out / "001_Test_Cube"
    assert len(list((folder / "images").glob("view_*.png"))) == 6
    assert (folder / "true_model.glb").is_file()
    assert (folder / "eval" / "heldout.png").is_file()
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["license"].startswith("CC BY 4.0") and len(meta["views"]) == 6
    split = json.loads((out / "split.json").read_text())
    assert split["train"] + split["test"] == ["001_Test_Cube"]


def _record(name, mirror, margin, f_hollow, f_solid, f_carve):
    return {
        "object": name,
        "category": "x",
        "features": {
            "has_side_view": 1.0,
            "mirror_median": mirror,
            "mirror_min": mirror,
            "height_spread": 0.05,
            "top_aspect_min": None,
            "open_top_margin": margin,
        },
        "outcomes": {
            "revolution_hollow": {"fscore": f_hollow},
            "revolution_solid": {"fscore": f_solid},
            "carving": {"fscore": f_carve},
            "carving_angles": {"fscore": f_carve},
        },
    }


def test_decision_rule_and_calibration_use_only_training_objects(tmp_path):
    params = {
        "mirror_median": 0.93,
        "mirror_min": 0.88,
        "height_spread": 0.15,
        "top_aspect": 0.85,
        "open_top_margin": 5.0,
    }
    assert (
        _decide(_record("a", 0.99, 20.0, 1, 0, 0)["features"], params)
        == "revolution_hollow"
    )
    assert (
        _decide(_record("a", 0.99, -9.0, 1, 0, 0)["features"], params)
        == "revolution_solid"
    )
    assert _decide(_record("a", 0.80, 20.0, 1, 0, 0)["features"], params) == "carving"

    # Training objects: round ones at mirror 0.90 really are round, so the
    # tuned mirror threshold should drop below 0.90.
    train = [_record(f"t{i}", 0.90, 20.0, 0.95, 0.5, 0.3) for i in range(4)]
    train += [_record(f"n{i}", 0.70, -9.0, 0.2, 0.2, 0.6) for i in range(4)]
    test = [_record("x0", 0.91, 20.0, 0.97, 0.5, 0.3)]
    records = tmp_path / "eval_records"
    records.mkdir()
    for r in train + test:
        (records / f"{r['object']}.json").write_text(json.dumps(r))
    (tmp_path / "split.json").write_text(
        json.dumps({"train": [r["object"] for r in train], "test": ["x0"]})
    )
    report = calibrate(tmp_path)
    assert report["tuned_params"]["mirror_median"] <= 0.90
    assert report["train"]["tuned"] > report["train"]["current"]
    assert report["test"]["tuned"] == pytest.approx(0.97)
    assert report["test"]["current"] == pytest.approx(0.3)


def test_calibration_tolerates_objects_with_no_outcomes(tmp_path):
    # A scan whose every reconstruction failed has outcomes that are all
    # None; the oracle must skip it instead of crashing on max([]).
    good = _record("g0", 0.70, -9.0, 0.2, 0.2, 0.6)
    empty = _record("e0", 0.70, -9.0, 0.2, 0.2, 0.6)
    empty["outcomes"] = {k: None for k in empty["outcomes"]}
    records = tmp_path / "eval_records"
    records.mkdir()
    for r in (good, empty):
        (records / f"{r['object']}.json").write_text(json.dumps(r))
    (tmp_path / "split.json").write_text(
        json.dumps({"train": ["g0", "e0"], "test": ["e0"]})
    )
    report = calibrate(tmp_path)
    assert report["train"]["oracle"] == pytest.approx(0.6)
    assert math.isnan(report["test"]["oracle"])
