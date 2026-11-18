import numpy as np

from img2mesh import cli
from img2mesh.benchmark import get_object
from img2mesh.benchmark.render import render_view
from img2mesh.silhouette import reconstruct_visual_hull
from img2mesh.validate import (
    Strategy,
    StrategyScore,
    candidate_strategies,
    quality_verdict,
)


def _views(name, angles, size=112):
    obj = get_object(name)
    return [render_view(obj, az, el, size=size).image for az, el in angles]


def test_candidate_strategies_follow_available_information():
    names = [s.name for s in candidate_strategies()]
    assert names == ["revolution", "carving"]
    names = [s.name for s in candidate_strategies(round_object=False)]
    assert names == ["carving"]
    angles = [(0.0, 10.0), (90.0, 10.0)]
    names = [s.name for s in candidate_strategies(view_angles=angles)]
    assert names == ["revolution", "carving+angles"]  # known angles replace guessing
    names = [s.name for s in candidate_strategies(learned_backend="x")]
    assert names[-1] == "prior:x"


def test_quality_verdict_is_strategy_aware_and_gives_advice():
    masks = [np.zeros((20, 20), dtype=bool) for _ in range(6)]
    for m in masks:
        m[5:15, 5:15] = True
    revolution = Strategy("revolution", "")
    carving = Strategy("carving", "")
    angled = Strategy("carving+angles", "")
    assert (
        quality_verdict(StrategyScore(revolution, 0.95, ()), masks, None)[0] == "good"
    )
    # Carving can never certify hollows, so it tops out at "fair".
    quality, advice = quality_verdict(
        StrategyScore(angled, 0.97, ()), masks, [(a, 10.0) for a in range(6)]
    )
    assert quality == "fair"
    assert any("same height" in a for a in advice)
    assert any("Only 6 photos" in a for a in advice)
    quality, advice = quality_verdict(StrategyScore(carving, 0.9, ()), masks, None)
    assert quality == "poor"
    assert any("--view-angles" in a for a in advice)

    masks[2][0, 3] = True  # object touching the image edge
    _, advice = quality_verdict(StrategyScore(revolution, 0.95, ()), masks, None)
    assert any("Photo 3" in a and "edge" in a for a in advice)


def test_best_mode_picks_revolution_for_a_round_object():
    angles = [
        (0.0, 4.0),
        (25.0, 14.0),
        (-30.0, 9.0),
        (55.0, 32.0),
        (-15.0, 52.0),
        (8.0, 86.0),
    ]
    result = reconstruct_visual_hull(
        _views("bowl", angles), resolution=48, shape_prior="best"
    )
    assert result.method == "best:revolution"
    assert result.quality == "good"
    names = dict(result.candidates)
    assert names["revolution"] > names["carving"]


def test_best_mode_uses_known_angles_for_a_non_round_object():
    angles = [(float(a), 10.0) for a in range(0, 360, 60)]
    result = reconstruct_visual_hull(
        _views("box", angles), resolution=48, shape_prior="best", view_angles=angles
    )
    assert result.method == "best:carving+angles"
    assert result.quality in ("fair", "poor")
    assert result.holdout_iou is not None and 0.0 < result.holdout_iou <= 1.0
    assert any("same height" in a for a in result.advice)


def test_cli_prints_quality_report(tmp_path, caplog):
    angles = [(0.0, 4.0), (25.0, 14.0), (-30.0, 9.0), (55.0, 32.0)]
    for idx, image in enumerate(_views("bowl", angles, size=96)):
        image.save(tmp_path / f"v{idx}.png")
    with caplog.at_level("INFO", logger="img2mesh"):
        code = cli.main(
            [
                str(tmp_path),
                str(tmp_path / "out.glb"),
                "--mode",
                "silhouette",
                "--shape-prior",
                "best",
            ]
        )
    assert code == cli.EXIT_SUCCESS
    text = caplog.text
    assert "Strategies (held-out photo IoU)" in text
    assert "Quality:" in text
