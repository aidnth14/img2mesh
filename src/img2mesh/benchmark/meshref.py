"""meshref: a reference set of real scanned objects for img2mesh.

Commands::

    python -m img2mesh.benchmark.meshref build RAW_DIR CHOSEN_JSON OUT_DIR
    python -m img2mesh.benchmark.meshref evaluate OUT_DIR [--workers 4]
    python -m img2mesh.benchmark.meshref calibrate OUT_DIR
    python -m img2mesh.benchmark.meshref reconstruct OUT_DIR

``build`` renders 6 product-style photos of every textured scan (plus a 7th
held-out photo kept under ``eval/``), writes the true model and attribution,
and a stratified 80/20 train/test split. ``evaluate`` records, per object,
the measurements img2mesh's decisions use and the true accuracy of every
outcome those decisions can lead to. ``calibrate`` tunes the decision
thresholds on the training objects only and reports accuracy on the test
objects, before and after. ``reconstruct`` saves img2mesh's model for every
object next to its photos.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import export_glb, load_mesh
from ..mesh import MeshResult
from .render import DEFAULT_VIEWS, RenderedView

PHOTO_VIEWS = DEFAULT_VIEWS[:6]
HELDOUT_VIEW = DEFAULT_VIEWS[6]
OUTCOMES = ("revolution_hollow", "revolution_solid", "carving", "carving_angles")


@dataclass(frozen=True)
class _Subject:
    """Minimal stand-in for BenchObject in :func:`run.evaluate`."""

    name: str
    category: str
    genus: int = -1


# ----------------------------------------------------------------- build ---


def _view_record(view: RenderedView) -> dict:
    return {
        "azimuth": view.azimuth,
        "elevation": view.elevation,
        "rotation": view.rotation.tolist(),
        "translation": view.translation.tolist(),
        "focal_px": view.intrinsics.fx,
        "size": view.intrinsics.image_width,
    }


def _clean_true_mesh(mesh) -> MeshResult:
    """Make a scan exportable: exporters need every vertex to have a normal.

    Scans contain zero-area triangles, duplicated (double-sided) surfaces
    whose normals cancel, and triangles that only collapse at float32
    precision. Weld coincident vertices, drop duplicate and float32-degenerate
    faces, then drop faces touching vertices whose normals still cancel.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    colors = mesh.vertex_colors()
    scale = float(np.max(np.ptp(v, axis=0))) or 1.0
    keys = np.round(v / scale * 1e6).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    v, colors = v[first], colors[first]
    f = inverse.reshape(-1)[np.asarray(mesh.faces)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, uniq = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(uniq)]
    for _ in range(3):
        v32 = v.astype(np.float32).astype(np.float64)
        tri = v32[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        f = f[np.linalg.norm(fn, axis=1) > 1e-12 * scale * scale]
        tri = v32[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = np.zeros_like(v)
        for corner in range(3):
            np.add.at(normals, f[:, corner], fn)
        bad = np.linalg.norm(normals, axis=1) <= 1e-12 * scale * scale
        touched = bad[f].any(axis=1)
        if not touched.any():
            break
        f = f[~touched]
    used = np.unique(f)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return MeshResult(
        vertices=v[used],
        faces=remap[f],
        alpha=None,
        input_point_count=len(mesh.vertices),
        retained_point_count=len(used),
        vertex_colors=colors[used],
    )


def build(raw_dir: Path, chosen_json: Path, out_dir: Path, size: int = 512) -> None:
    chosen = json.loads(chosen_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for number, item in enumerate(chosen, start=1):
        name = item["name"]
        src = raw_dir / name
        if not (src / "model.obj").is_file():
            print(f"skip {name}: mesh not downloaded", file=sys.stderr)
            continue
        folder = out_dir / f"{number:03d}_{name}"
        try:
            meta = _build_one(src, folder, item, size)
        except Exception as exc:  # one bad scan must not stop the set
            print(f"skip {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        built.append((folder.name, meta["category"]))
        print(f"[{len(built):3d}] {folder.name}", flush=True)
    _write_split(out_dir, built)


def _build_one(src: Path, folder: Path, item: dict, size: int) -> dict:
    from .meshdata import load_textured_obj, render_mesh_view

    name = item["name"]
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "eval").mkdir(exist_ok=True)
    texture = src / "texture.png"
    mesh = load_textured_obj(src / "model.obj", texture if texture.is_file() else None)
    records = []
    for idx, (az, el) in enumerate(PHOTO_VIEWS, start=1):
        view = render_mesh_view(mesh, az, el, size=size)
        view.image.save(folder / "images" / f"view_{idx}_az{az:+.0f}_el{el:.0f}.png")
        records.append(_view_record(view))
    held = render_mesh_view(mesh, *HELDOUT_VIEW, size=size)
    held.image.save(folder / "eval" / "heldout.png")
    Image.fromarray(held.mask.astype(np.uint8) * 255).save(
        folder / "eval" / "heldout_mask.png"
    )
    export_glb(_clean_true_mesh(mesh), folder / "true_model.glb", overwrite=True)
    meta = {
        "name": name,
        "category": item.get("category", ""),
        "description": item.get("description", ""),
        "textured": texture.is_file(),
        "source": f"https://app.gazebosim.org/GoogleResearch/fuel/models/{name}",
        "license": "CC BY 4.0 (Google Scanned Objects, Google Research)",
        "views": records,
        "heldout_view": _view_record(held),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _write_split(out_dir: Path, built: list[tuple[str, str]]) -> None:
    rng = random.Random(7)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for folder_name, category in built:
        by_cat[category].append(folder_name)
    train, test = [], []
    for category in sorted(by_cat):
        names = sorted(by_cat[category])
        rng.shuffle(names)
        n_test = max(1, round(0.2 * len(names))) if len(names) >= 3 else 0
        test += names[:n_test]
        train += names[n_test:]
    split_path = out_dir / "split.json"
    extra = []
    if split_path.is_file():  # keep artifacts added with add-artifact
        extra = json.loads(split_path.read_text()).get("extra", [])
    split = {"train": sorted(train), "test": sorted(test)}
    if extra:
        split["extra"] = sorted(extra)
    split_path.write_text(json.dumps(split, indent=1))
    print(f"built {len(built)} objects: {len(train)} train / {len(test)} test")


def add_artifact(name: str, out_dir: Path, number: int, size: int = 512) -> Path:
    """Add a code-defined object (``objects.ARTIFACTS``) in meshref layout.

    These are reconstructions modelled in code, not scans; their metadata
    says so. They are kept out of the calibration split.
    """
    from .objects import get_object
    from .render import render_view
    from .run import ground_truth_mesh

    obj = get_object(name)
    folder = out_dir / f"{number:03d}_{name}"
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "eval").mkdir(exist_ok=True)
    records = []
    for idx, (az, el) in enumerate(PHOTO_VIEWS, start=1):
        view = render_view(obj, az, el, size=size)
        view.image.save(folder / "images" / f"view_{idx}_az{az:+.0f}_el{el:.0f}.png")
        records.append(_view_record(view))
    held = render_view(obj, *HELDOUT_VIEW, size=size)
    held.image.save(folder / "eval" / "heldout.png")
    Image.fromarray(held.mask.astype(np.uint8) * 255).save(
        folder / "eval" / "heldout_mask.png"
    )
    verts, faces = ground_truth_mesh(obj, resolution=200)
    export_glb(
        MeshResult(
            vertices=verts,
            faces=faces,
            alpha=None,
            input_point_count=len(verts),
            retained_point_count=len(verts),
            vertex_colors=obj.albedo(verts),
        ),
        folder / "true_model.glb",
        overwrite=True,
    )
    meta = {
        "name": name,
        "category": "Historical artifact (modelled)",
        "description": (obj.__doc__ or "").strip() or obj.notes,
        "notes": obj.notes,
        "textured": True,
        "source": "modelled in code: img2mesh.benchmark.objects",
        "license": "same as img2mesh (Apache-2.0)",
        "views": records,
        "heldout_view": _view_record(held),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    split_path = out_dir / "split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text())
        extra = set(split.get("extra", [])) | {folder.name}
        split["extra"] = sorted(extra)
        split_path.write_text(json.dumps(split, indent=1))
    return folder


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def build_from_selection(
    selection_json: Path,
    glb_dir: Path,
    out_dir: Path,
    *,
    size: int = 512,
    workers: int = 4,
) -> None:
    """Build meshref from a category list matched to GLB models.

    ``selection_json`` holds one entry per category: ``id``, ``item``,
    ``section``, ``status`` and, for matched ones, ``uid``, ``name``,
    ``license``, ``author``, ``url``, ``score``. Every category gets a folder;
    unmatched ones get a NOT_AVAILABLE.txt saying why.
    """
    entries = json.loads(selection_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for e in entries:
        folder = out_dir / f"{e['id']:03d}_{_slug(e['item'])}"
        folder.mkdir(exist_ok=True)
        glb = glb_dir / f"{e.get('uid', '')}.glb"
        if e["status"] != "matched" or not glb.is_file():
            reason = (
                e["status"] if e["status"] != "matched" else "model failed to download"
            )
            (folder / "NOT_AVAILABLE.txt").write_text(
                f"{e['item']} ({e['section']})\n\nNo images: {reason}.\n"
                + (f"Closest model found: {e['closest']}\n" if e.get("closest") else "")
            )
            continue
        if (folder / "meta.json").is_file():  # already built: keep (incremental)
            continue
        jobs.append((e, folder, glb))
    built = [
        (p.name, json.loads((p / "meta.json").read_text())["category"])
        for p in sorted(out_dir.iterdir())
        if (p / "meta.json").is_file()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_build_glb_one, e, folder, glb, size): (e, folder)
            for e, folder, glb in jobs
        }
        for future in as_completed(futures):
            e, folder = futures[future]
            try:
                future.result()
            except Exception as exc:
                (folder / "NOT_AVAILABLE.txt").write_text(
                    f"{e['item']} ({e['section']})\n\nNo images: the model could "
                    f"not be rendered ({type(exc).__name__}: {exc}).\n"
                )
                print(f"skip {folder.name}: {exc}", file=sys.stderr, flush=True)
                continue
            built.append((folder.name, e["section"]))
            print(f"[{len(built):3d}] {folder.name}", flush=True)
    _write_split(out_dir, built)


def _build_glb_one(entry: dict, folder: Path, glb: Path, size: int) -> None:
    from .meshdata import load_glb_textured, render_mesh_view

    mesh = load_glb_textured(glb)
    (folder / "images").mkdir(exist_ok=True)
    (folder / "eval").mkdir(exist_ok=True)
    records = []
    for idx, (az, el) in enumerate(PHOTO_VIEWS, start=1):
        view = render_mesh_view(mesh, az, el, size=size)
        view.image.save(folder / "images" / f"view_{idx}_az{az:+.0f}_el{el:.0f}.png")
        records.append(_view_record(view))
    held = render_mesh_view(mesh, *HELDOUT_VIEW, size=size)
    held.image.save(folder / "eval" / "heldout.png")
    Image.fromarray(held.mask.astype(np.uint8) * 255).save(
        folder / "eval" / "heldout_mask.png"
    )
    export_glb(_clean_true_mesh(mesh), folder / "true_model.glb", overwrite=True)
    meta = {
        "name": entry["item"],
        "category": entry["section"],
        "item_id": entry["id"],
        "model_name": entry.get("name"),
        "description": f"{entry['item']}: represented by '{entry.get('name')}'",
        "match_confidence": "good" if entry.get("score", 0) >= 12 else "approximate",
        "match_score": entry.get("score"),
        "textured": True,
        "source": entry.get("url"),
        "author": entry.get("author"),
        "license": f"CC {entry.get('license', '').upper()} (Sketchfab via Objaverse)",
        "views": records,
        "heldout_view": _view_record(held),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    (folder / "NOT_AVAILABLE.txt").unlink(missing_ok=True)  # from a failed pass


# -------------------------------------------------------------- evaluate ---


def _load_views(folder: Path) -> tuple[list[RenderedView], dict]:
    from ..geometry import CameraIntrinsics

    meta = json.loads((folder / "meta.json").read_text())
    views = []
    images = sorted((folder / "images").glob("view_*.png"))
    for path, rec in zip(images, meta["views"], strict=True):
        views.append(
            _rendered(Image.open(path).convert("RGB"), None, rec, CameraIntrinsics)
        )
    held = meta["heldout_view"]
    mask = np.asarray(Image.open(folder / "eval" / "heldout_mask.png")) > 127
    views.append(
        _rendered(
            Image.open(folder / "eval" / "heldout.png").convert("RGB"),
            mask,
            held,
            CameraIntrinsics,
        )
    )
    return views, meta


def _rendered(image, mask, rec, intrinsics_cls) -> RenderedView:
    from ..silhouette import extract_foreground_mask

    if mask is None:
        mask = extract_foreground_mask(image) > 0
    return RenderedView(
        image=image,
        mask=mask,
        rotation=np.asarray(rec["rotation"]),
        translation=np.asarray(rec["translation"]),
        intrinsics=intrinsics_cls.from_focal_length(
            rec["size"], rec["size"], rec["focal_px"]
        ),
        azimuth=rec["azimuth"],
        elevation=rec["elevation"],
    )


def evaluate_object(folder: Path, resolution: int = 88, masker: str = "color") -> dict:
    """Decision features plus the true accuracy of every possible outcome."""
    import importlib

    import cv2

    from ..revolve import analyze_view, open_top_margin, roundness_features

    run = importlib.import_module("img2mesh.benchmark.run")

    from dataclasses import replace as _replace

    views, meta = _load_views(folder)
    if masker == "rembg":
        from ..learned import rembg_mask

        views = [_replace(v, mask=rembg_mask(v.image)) for v in views[:-1]] + views[-1:]
    train = views[:-1]
    masks = [v.mask for v in train]
    true = load_mesh(folder / "true_model.glb")
    gt = (true.vertices, true.faces)
    subject = _Subject(folder.name, meta["category"])

    analyses = [analyze_view(v.mask) for v in train]
    grays = [cv2.cvtColor(np.asarray(v.image), cv2.COLOR_RGB2GRAY) for v in train]
    record = {
        "object": folder.name,
        "category": meta["category"],
        "features": {
            **roundness_features(analyses),
            "open_top_margin": open_top_margin(analyses, grays),
        },
        "outcomes": {},
        "seconds": {},
    }
    plans = {
        "revolution_hollow": ("revolution", {"hollow": True}),
        "revolution_solid": ("revolution", {"hollow": False}),
        "carving": ("none", {}),
        "carving_angles": ("angles", {}),
    }
    for outcome, (method, extra) in plans.items():
        start = time.perf_counter()
        res = run.evaluate(
            subject,
            views,
            gt,
            method,
            resolution=resolution,
            extra={**extra, "masks": masks},
        )
        record["outcomes"][outcome] = (
            None
            if res.error
            else {
                "fscore": res.fscore,
                "chamfer_pct": res.chamfer_pct,
                "novel_iou": res.novel_iou,
            }
        )
        record["seconds"][outcome] = round(time.perf_counter() - start, 1)
    return record


def evaluate(out_dir: Path, workers: int = 4, masker: str = "color") -> None:
    name = "eval_records" if masker == "color" else f"eval_records_{masker}"
    cache = out_dir / name
    cache.mkdir(exist_ok=True)
    folders = sorted(p for p in out_dir.iterdir() if (p / "meta.json").is_file())
    todo = [f for f in folders if not (cache / f"{f.name}.json").is_file()]
    print(f"evaluating {len(todo)} of {len(folders)} objects with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_object, f, 88, masker): f for f in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            folder = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # keep going; report at the end
                record = {
                    "object": folder.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            (cache / f"{folder.name}.json").write_text(json.dumps(record, indent=1))
            print(f"[{done}/{len(todo)}] {folder.name}", flush=True)


# ------------------------------------------------------------- calibrate ---


def _decide(features: dict, params: dict) -> str:
    """The outcome img2mesh's 'auto' path would produce under ``params``."""
    f = features
    round_object = (
        f["has_side_view"]
        and f["mirror_median"] >= params["mirror_median"]
        and f["mirror_min"] >= params["mirror_min"]
        and f["height_spread"] <= params["height_spread"]
        and (f["top_aspect_min"] is None or f["top_aspect_min"] >= params["top_aspect"])
    )
    if not round_object:
        return "carving"
    margin = f["open_top_margin"]
    hollow = margin is not None and margin > params["open_top_margin"]
    return "revolution_hollow" if hollow else "revolution_solid"


def _score(records: Sequence[dict], params: dict) -> float:
    total = 0.0
    for r in records:
        outcome = r["outcomes"].get(_decide(r["features"], params))
        total += outcome["fscore"] if outcome else 0.0
    return total / max(len(records), 1)


def calibrate(out_dir: Path, masker: str = "color") -> dict:
    from .. import revolve

    split = json.loads((out_dir / "split.json").read_text())
    records = {}
    records_dir = "eval_records" if masker == "color" else f"eval_records_{masker}"
    for path in (out_dir / records_dir).glob("*.json"):
        rec = json.loads(path.read_text())
        if "error" not in rec:
            records[rec["object"]] = rec
    train = [records[n] for n in split["train"] if n in records]
    test = [records[n] for n in split["test"] if n in records]

    current = {
        "mirror_median": revolve.ROUND_MIRROR_MEDIAN,
        "mirror_min": revolve.ROUND_MIRROR_MIN,
        "height_spread": revolve.ROUND_HEIGHT_SPREAD,
        "top_aspect": revolve.ROUND_TOP_ASPECT,
        "open_top_margin": revolve.OPEN_TOP_MARGIN,
    }
    grid = {
        "mirror_median": [0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
        "mirror_min": [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92],
        "height_spread": [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5],
        "top_aspect": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        "open_top_margin": [-5.0, 0.0, 2.0, 5.0, 8.0, 12.0, 20.0, 40.0],
    }
    best_params, best_score = current, _score(train, current)
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        score = _score(train, params)
        # Prefer the current setting on ties, so changes need evidence.
        if score > best_score + 1e-9:
            best_params, best_score = params, score

    def oracle(rs):
        vals = [
            max(
                (
                    o["fscore"]
                    for k, o in r["outcomes"].items()
                    if o and k != "carving_angles"
                ),
                default=None,
            )
            for r in rs
        ]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def mean(rs, key):
        vals = [r["outcomes"][key]["fscore"] for r in rs if r["outcomes"].get(key)]
        return float(np.mean(vals)) if vals else float("nan")

    report = {
        "objects": {"train": len(train), "test": len(test)},
        "current_params": current,
        "tuned_params": best_params,
        "train": {
            "current": _score(train, current),
            "tuned": best_score,
            "oracle": oracle(train),
        },
        "test": {
            "current": _score(test, current),
            "tuned": _score(test, best_params),
            "oracle": oracle(test),
            "always_carving": mean(test, "carving"),
            "carving_with_angles": mean(test, "carving_angles"),
        },
        "test_choices": {
            r["object"]: {
                "current": _decide(r["features"], current),
                "tuned": _decide(r["features"], best_params),
                "fscores": {
                    k: (o["fscore"] if o else None) for k, o in r["outcomes"].items()
                },
            }
            for r in test
        },
    }
    report["masker"] = masker
    name = "calibration.json" if masker == "color" else f"calibration_{masker}.json"
    (out_dir / name).write_text(json.dumps(report, indent=1))
    return report


# ----------------------------------------------------------- reconstruct ---


def reconstruct(out_dir: Path) -> None:
    from ..silhouette import reconstruct_visual_hull

    for folder in sorted(p for p in out_dir.iterdir() if (p / "meta.json").is_file()):
        if (folder / "img2mesh_model.glb").is_file():
            continue  # incremental
        images = [
            Image.open(p).convert("RGB")
            for p in sorted((folder / "images").glob("view_*.png"))
        ]
        try:
            result = reconstruct_visual_hull(images)
        except Exception as exc:
            print(f"{folder.name}: failed ({exc})", flush=True)
            continue
        export_glb(result.mesh, folder / "img2mesh_model.glb", overwrite=True)
        print(f"{folder.name}: {result.method}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m img2mesh.benchmark.meshref")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("raw_dir", type=Path)
    b.add_argument("chosen_json", type=Path)
    b.add_argument("out_dir", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("out_dir", type=Path)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--masker", choices=("color", "rembg"), default="color")
    c = sub.add_parser("calibrate")
    c.add_argument("out_dir", type=Path)
    c.add_argument("--masker", choices=("color", "rembg"), default="color")
    r = sub.add_parser("reconstruct")
    r.add_argument("out_dir", type=Path)
    bs = sub.add_parser("build-selection")
    bs.add_argument("selection_json", type=Path)
    bs.add_argument("glb_dir", type=Path)
    bs.add_argument("out_dir", type=Path)
    bs.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("add-artifact")
    a.add_argument("name")
    a.add_argument("out_dir", type=Path)
    a.add_argument("number", type=int)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.raw_dir, args.chosen_json, args.out_dir)
    elif args.command == "evaluate":
        evaluate(args.out_dir, args.workers, args.masker)
    elif args.command == "build-selection":
        build_from_selection(
            args.selection_json, args.glb_dir, args.out_dir, workers=args.workers
        )
    elif args.command == "add-artifact":
        print(add_artifact(args.name, args.out_dir, args.number))
    elif args.command == "calibrate":
        print(json.dumps(calibrate(args.out_dir, args.masker), indent=1))
    else:
        reconstruct(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
