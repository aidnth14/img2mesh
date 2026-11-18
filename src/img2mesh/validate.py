"""Self-validation and automatic strategy selection.

A reconstruction's fit to its own input photos says little about accuracy:
carving and pose fitting optimise exactly that, so wrong shapes score high.
Leave-one-out validation is an honest proxy instead. Rebuild without one
photo, then ask whether the result can explain that unseen photo: fit a
camera for it (orientation known when view angles are given) and measure
silhouette overlap. Shapes that are wrong away from the input views fail.

:func:`select_strategy` runs this for every applicable strategy and returns
the best, with a quality verdict and advice for improving the photo set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import Img2MeshError
from .geometry import CameraIntrinsics

# Verdict thresholds on held-out silhouette IoU, calibrated on the
# benchmark. Outlines cannot reveal missing hollows or concave regions, so
# only the revolution model (which reasons about interiors) can rate "good";
# carving-based results top out at "fair" however well they fit.
GOOD_HOLDOUT_IOU = 0.92
FAIR_HOLDOUT_IOU = 0.85


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyScore:
    strategy: Strategy
    holdout_iou: float
    per_fold: tuple[float, ...]
    error: str = ""


def _fit_heldout(
    vertices: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    angle: tuple[float, float] | None,
) -> float:
    """Best silhouette overlap of the mesh with an unseen photo."""
    from .prior import _ALL_ELEVATIONS, _pose_from_rotation
    from .revolve import _fit_pose, analyze_view
    from .silhouette import _ortho_rotation

    view = analyze_view(mask)
    pts = vertices[:: max(1, len(vertices) // 20000)]
    if angle is not None:
        rot = _ortho_rotation(*angle)
        return float(_pose_from_rotation(pts, view, rot, intrinsics)[2])
    best = 0.0
    # Meshes are Y-up, except blind carving's frame, which is Y-down.
    for flip in (np.eye(3), np.diag([1.0, -1.0, -1.0])):
        score = _fit_pose(pts @ flip.T, view, intrinsics, elevations=_ALL_ELEVATIONS)[2]
        best = max(best, float(score))
    return best


def _folds(count: int, exclude: Sequence[int] = ()) -> list[int]:
    candidates = [i for i in range(count) if i not in exclude]
    if len(candidates) <= 3:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, 3).round().astype(int)
    return [candidates[i] for i in sorted(set(picks))]


def holdout_score(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    strategy: Strategy,
    *,
    resolution: int = 88,
    folds: Sequence[int] | None = None,
) -> StrategyScore:
    """Leave-one-out silhouette IoU of ``strategy`` on up to three folds.

    Use the final build's resolution: coarser validation builds lose thin
    parts (limbs, handles) and then misjudge the strategy.
    """
    from .silhouette import reconstruct_visual_hull

    angles = strategy.kwargs.get("view_angles")
    folds = list(folds) if folds is not None else _folds(len(images))
    scores = []
    errors = []
    for held in folds:
        keep = [i for i in range(len(images)) if i != held]
        kwargs = dict(strategy.kwargs)
        if angles is not None:
            kwargs["view_angles"] = [angles[i] for i in keep]
        try:
            result = reconstruct_visual_hull(
                [images[i] for i in keep],
                masks=[masks[i] for i in keep],
                intrinsics=intrinsics,
                resolution=resolution,
                **kwargs,
            )
            if strategy.name == "revolution" and result.method != "revolution":
                raise Img2MeshError("not a surface of revolution without this view")
            scores.append(
                _fit_heldout(
                    result.mesh.vertices,
                    masks[held],
                    intrinsics,
                    None if angles is None else tuple(angles[held]),
                )
            )
        except (Img2MeshError, ValueError) as exc:
            errors.append(str(exc))
            scores.append(0.0)
    return StrategyScore(
        strategy,
        float(np.mean(scores)) if scores else 0.0,
        tuple(scores),
        "; ".join(errors),
    )


def candidate_strategies(
    *,
    view_angles: Sequence[tuple[float, float]] | None = None,
    learned_backend: str | None = None,
    prior_up_axis: str | None = None,
    carve_concavity: bool = True,
    round_object: bool = True,
) -> list[Strategy]:
    """Strategies worth comparing.

    The revolution model is only offered when the views pass the roundness
    test: forced onto a box it can still explain held-out outlines with a
    freely fitted camera. Known angles replace blind pose estimation rather
    than compete with it (a free camera flatters any held-out score).
    """
    base = {"carve_concavity": carve_concavity}
    strategies = []
    if round_object:
        strategies.append(
            Strategy(
                "revolution",
                "rotationally symmetric body (+ handles, spouts)",
                {**base, "shape_prior": "revolution"},
            )
        )
    if view_angles is None:
        strategies.append(
            Strategy(
                "carving",
                "visual hull with estimated camera angles",
                {**base, "shape_prior": "none"},
            )
        )
    else:
        strategies.append(
            Strategy(
                "carving+angles",
                "visual hull with the given camera angles",
                {**base, "shape_prior": "none", "view_angles": list(view_angles)},
            )
        )
    if learned_backend is not None:
        strategies.append(
            Strategy(
                f"prior:{learned_backend}",
                "learned shape prior refined against the photos",
                {
                    **base,
                    "shape_prior": "none",
                    "learned_backend": learned_backend,
                    "prior_up_axis": prior_up_axis,
                },
            )
        )
    return strategies


def quality_verdict(
    best: StrategyScore,
    masks: Sequence[np.ndarray],
    view_angles: Sequence[tuple[float, float]] | None,
) -> tuple[str, tuple[str, ...]]:
    """("good" | "fair" | "poor", advice for a better photo set)."""
    score = best.holdout_iou
    name = best.strategy.name
    if name == "revolution" and score >= GOOD_HOLDOUT_IOU:
        quality = "good"
    elif name == "carving":
        # Guessed camera angles: outline-consistent but often wrong shapes.
        quality = "fair" if score >= 0.95 else "poor"
    elif score >= FAIR_HOLDOUT_IOU:
        quality = "fair"
    else:
        quality = "poor"
    advice = []
    for idx, mask in enumerate(masks):
        m = np.asarray(mask) > 0
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            advice.append(
                f"Photo {idx + 1}: the object touches the image edge; leave a margin."
            )
    if quality != "good":
        if name != "revolution":
            advice.append(
                "Carved shapes cannot show hollows or deep concave areas; the "
                "outlines match, but check those regions."
            )
        if name == "carving" and view_angles is None:
            advice.append(
                "Camera angles had to be guessed. If they are known (turntable), "
                "pass --view-angles; otherwise try --learned-prior."
            )
        if view_angles is not None:
            elevations = [el for _, el in view_angles]
            if max(elevations) - min(elevations) < 20.0:
                advice.append(
                    "All photos are from about the same height: add a few from "
                    "higher up (30-60 degrees) to carve tops and gaps."
                )
        if len(masks) < 10:
            advice.append(
                f"Only {len(masks)} photos: 10-16 evenly around the object "
                "carve concave regions much better."
            )
    return quality, tuple(advice)


def select_strategy(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    view_angles: Sequence[tuple[float, float]] | None = None,
    learned_backend: str | None = None,
    prior_up_axis: str | None = None,
    carve_concavity: bool = True,
    resolution: int = 88,
) -> tuple[StrategyScore, list[StrategyScore]]:
    """Score every applicable strategy by held-out views; return the best."""
    from .revolve import analyze_view, is_surface_of_revolution

    round_object = is_surface_of_revolution([analyze_view(m) for m in masks])
    scores = [
        holdout_score(images, masks, intrinsics, strategy, resolution=resolution)
        for strategy in candidate_strategies(
            view_angles=view_angles,
            learned_backend=learned_backend,
            prior_up_axis=prior_up_axis,
            carve_concavity=carve_concavity,
            round_object=round_object,
        )
    ]
    best = max(scores, key=lambda s: s.holdout_iou)
    return best, scores
