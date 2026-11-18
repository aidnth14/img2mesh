"""Benchmark runner: reconstruct synthetic product shots and score the shapes.

Usage::

    python -m img2mesh.benchmark                       # all objects, both priors
    python -m img2mesh.benchmark --objects cup,box --methods auto
    python -m img2mesh.benchmark --json results.json --save-meshes out/

Each object is rendered from seven product-style views. Six are given to the
reconstruction; the seventh is held out. The reconstructed mesh is aligned to
the ground truth with a similarity transform (initialised from the fitted
cameras, refined by ICP) and scored with:

- F-score / precision / recall at 2% of the ground-truth diagonal
  (precision: no extra material; recall: nothing missing, e.g. a hollow),
- symmetric Chamfer distance (% of the diagonal),
- genus and watertightness against the ground truth,
- silhouette IoU from the held-out view (never seen by the method),
- the method's self-reported IoU on its own input views, for contrast.

Methods: ``auto`` / ``revolution`` / ``none`` are the shape priors; ``angles``
is visual-hull carving given the true view angles; ``prior`` refines a
simulated learned prior (see :func:`simulated_learned_prior`) whose up axis is
found automatically, ``prior_up`` the same prior with its up axis given, and
``learned`` a real TripoSR prior (optional dependencies required).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..io import export_glb
from ..mesh import MeshResult
from ..silhouette import (
    marching_tetrahedra,
    rasterize_mesh_silhouette,
    reconstruct_visual_hull,
)
from .metrics import (
    align_similarity,
    mean_rotation,
    mesh_topology,
    sample_surface,
    shape_scores,
    silhouette_iou,
)
from .objects import BUILDERS, BenchObject, get_object
from .render import DEFAULT_VIEWS, RenderedView, render_views

TAU = 0.02
SAMPLES = 30000


@dataclass
class BenchResult:
    object: str
    category: str
    method: str
    used: str = ""
    fscore: float = float("nan")
    fscore_1pct: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    chamfer_pct: float = float("nan")
    genus: int = -1
    gt_genus: int = -1
    watertight: bool = False
    novel_iou: float = float("nan")
    self_iou: float = float("nan")
    seconds: float = float("nan")
    error: str = ""


def ground_truth_mesh(
    obj: BenchObject, resolution: int = 150
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = obj.bounds
    pad = 0.05 * float(np.max(hi - lo))
    lo, hi = lo - pad, hi + pad
    voxel = float(np.max(hi - lo)) / (resolution - 1)
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    axes = [lo[i] + voxel * np.arange(shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    sdf = obj.sdf(grid).reshape(shape)
    field = np.clip(0.5 - sdf / (2.0 * voxel), 0.0, 1.0)
    return marching_tetrahedra(field, lo, voxel, level=0.5)


def simulated_learned_prior(
    vertices: np.ndarray, faces: np.ndarray, seed: int = 0
) -> MeshResult:
    """Stand-in for a single-image 3D generator's output.

    The ground truth, over-smoothed (plain Laplacian smoothing shrinks and
    blurs details), perturbed by noise of ~1.5% of its size, and put in
    an arbitrary frame: Z up, rotated about the up axis, rescaled, offset.
    It measures how well a decent-but-imperfect prior is exploited; it is not
    a measurement of any particular learned model.
    """
    rng = np.random.default_rng(seed)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    counts = np.maximum(np.bincount(edges.ravel(), minlength=len(vertices)), 1)
    v = vertices.copy()
    for _ in range(12):
        summed = np.zeros_like(v)
        np.add.at(summed, edges[:, 0], v[edges[:, 1]])
        np.add.at(summed, edges[:, 1], v[edges[:, 0]])
        v = v + 0.5 * (summed / counts[:, None] - v)
    size = float(np.max(np.ptp(vertices, axis=0)))
    v = v + rng.normal(scale=0.015 * size / np.sqrt(3), size=v.shape)
    frame = Rotation.from_euler("zx", [37.0, 90.0], degrees=True).as_matrix()
    v = 3.1 * v @ frame.T + np.array([0.4, -1.2, 2.0])
    return MeshResult(
        vertices=v,
        faces=faces,
        alpha=None,
        input_point_count=len(v),
        retained_point_count=len(v),
    )


def _align(
    recon_pts: np.ndarray,
    gt_pts: np.ndarray,
    camera_rotation: np.ndarray,
    diagonal: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Best similarity over a few yaw-rotated starts (shape, not pose, is scored)."""
    sub_r = recon_pts[:: max(1, len(recon_pts) // 4000)]
    sub_g = gt_pts[:: max(1, len(gt_pts) // 4000)]
    best = None
    for yaw in (0.0, 90.0, 180.0, 270.0):
        spin = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        s, r, t = align_similarity(sub_r, sub_g, spin @ camera_rotation)
        score = shape_scores(s * sub_r @ r.T + t, sub_g, diagonal, TAU).chamfer
        if best is None or score < best[0]:
            best = (score, s, r, t)
    _, s, r, t = best
    return s, r, t


def evaluate(
    obj: BenchObject,
    views: Sequence[RenderedView],
    gt: tuple[np.ndarray, np.ndarray],
    method: str,
    *,
    resolution: int,
    save_dir: Path | None = None,
) -> BenchResult:
    result = BenchResult(obj.name, obj.category, method, gt_genus=obj.genus)
    train, held = list(views[:-1]), views[-1]
    gt_v, gt_f = gt
    gt_pts = sample_surface(gt_v, gt_f, SAMPLES, seed=1)
    diagonal = float(np.linalg.norm(np.ptp(gt_v, axis=0)))

    kwargs = {"resolution": resolution, "shape_prior": method}
    if method in ("prior", "prior_up"):
        kwargs["shape_prior"] = "none"
        kwargs["prior_mesh"] = simulated_learned_prior(gt_v, gt_f)
        if method == "prior_up":
            # Generators output a fixed, documented orientation (Z up here).
            kwargs["prior_up_axis"] = "+z"
    elif method == "learned":
        # A real learned generator (needs its optional dependencies).
        kwargs["shape_prior"] = "none"
        kwargs["learned_backend"] = "triposr"
    elif method == "angles":
        # Visual hull given the true view angles (as from a turntable rig).
        kwargs["shape_prior"] = "none"
        kwargs["view_angles"] = [(v.azimuth, v.elevation) for v in train]
    start = time.perf_counter()
    try:
        recon = reconstruct_visual_hull([v.image for v in train], **kwargs)
    except Exception as exc:  # report, keep benchmarking
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - start
        return result
    result.seconds = time.perf_counter() - start
    result.used = recon.method
    result.self_iou = recon.mean_iou

    mesh = recon.mesh
    fitted = [Rotation.from_rotvec(r).as_matrix() for r in recon.camera_rotations]
    cam_rot = mean_rotation([v.rotation.T @ f for v, f in zip(train, fitted)])
    recon_pts = sample_surface(mesh.vertices, mesh.faces, SAMPLES, seed=2)
    s, r, t = _align(recon_pts, gt_pts, cam_rot, diagonal)
    aligned_pts = s * recon_pts @ r.T + t

    main = shape_scores(aligned_pts, gt_pts, diagonal, TAU)
    fine = shape_scores(aligned_pts, gt_pts, diagonal, TAU / 2)
    result.fscore, result.precision, result.recall = (
        main.fscore,
        main.precision,
        main.recall,
    )
    result.fscore_1pct = fine.fscore
    result.chamfer_pct = 100.0 * main.chamfer

    topo = mesh_topology(mesh.faces, len(mesh.vertices))
    result.genus = int(topo["genus"])
    result.watertight = bool(topo["watertight"])

    aligned_v = s * mesh.vertices @ r.T + t
    proj = rasterize_mesh_silhouette(
        aligned_v, mesh.faces, held.rotation, held.translation, held.intrinsics
    )
    result.novel_iou = silhouette_iou(proj, held.mask)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        aligned_mesh = MeshResult(
            vertices=aligned_v,
            faces=mesh.faces,
            alpha=None,
            input_point_count=mesh.input_point_count,
            retained_point_count=mesh.retained_point_count,
            vertex_colors=mesh.vertex_colors,
        )
        export_glb(aligned_mesh, save_dir / f"{obj.name}_{method}.glb", overwrite=True)
    return result


def _fmt(value: float, spec: str) -> str:
    return "-" if value != value else format(value, spec)


def format_table(results: Sequence[BenchResult]) -> str:
    header = (
        f"{'object':11s} {'method':8s} {'used':11s} {'F@2%':>5s} {'P':>5s} "
        f"{'R':>5s} {'F@1%':>5s} {'CD%':>5s} {'genus':>6s} {'novel':>6s} "
        f"{'self':>5s} {'sec':>5s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        if r.error:
            lines.append(f"{r.object:11s} {r.method:8s} ERROR {r.error}")
            continue
        genus = f"{r.genus}/{r.gt_genus}" if r.watertight else f"open/{r.gt_genus}"
        lines.append(
            f"{r.object:11s} {r.method:8s} {r.used:11s} {_fmt(r.fscore, '.3f'):>5s} "
            f"{_fmt(r.precision, '.2f'):>5s} {_fmt(r.recall, '.2f'):>5s} "
            f"{_fmt(r.fscore_1pct, '.3f'):>5s} {_fmt(r.chamfer_pct, '.2f'):>5s} "
            f"{genus:>6s} {_fmt(r.novel_iou, '.3f'):>6s} "
            f"{_fmt(r.self_iou, '.3f'):>5s} {_fmt(r.seconds, '.1f'):>5s}"
        )

    lines.append("")
    lines.append("Mean by category (F@2% / Chamfer% / novel IoU / genus correct):")
    methods = sorted({r.method for r in results})
    for category in ("round", "other", "all"):
        for method in methods:
            rows = [
                r
                for r in results
                if r.method == method
                and not r.error
                and (category == "all" or r.category == category)
            ]
            if not rows:
                continue
            genus_ok = sum(r.watertight and r.genus == r.gt_genus for r in rows)
            lines.append(
                f"  {category:5s} {method:8s} "
                f"F={np.mean([r.fscore for r in rows]):.3f}  "
                f"CD={np.mean([r.chamfer_pct for r in rows]):.2f}%  "
                f"novel={np.mean([r.novel_iou for r in rows]):.3f}  "
                f"genus {genus_ok}/{len(rows)}"
            )
    return "\n".join(lines)


def run(
    objects: Sequence[str],
    methods: Sequence[str],
    *,
    size: int = 256,
    resolution: int = 88,
    save_dir: Path | None = None,
    progress: bool = True,
) -> list[BenchResult]:
    results = []
    for name in objects:
        obj = get_object(name)
        gt = ground_truth_mesh(obj)
        views = render_views(obj, DEFAULT_VIEWS, size=size)
        for method in methods:
            res = evaluate(
                obj, views, gt, method, resolution=resolution, save_dir=save_dir
            )
            results.append(res)
            if progress:
                note = f" {res.error}" if res.error else ""
                print(
                    f"  {name}/{method}: F={_fmt(res.fscore, '.3f')} "
                    f"({_fmt(res.seconds, '.1f')}s){note}",
                    file=sys.stderr,
                )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m img2mesh.benchmark")
    parser.add_argument("--objects", default=",".join(BUILDERS))
    parser.add_argument("--methods", default="none,auto")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=88)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--save-meshes", type=Path, default=None)
    args = parser.parse_args(argv)

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = run(
        objects,
        methods,
        size=args.size,
        resolution=args.resolution,
        save_dir=args.save_meshes,
    )
    print(format_table(results))
    if args.json is not None:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return 0
