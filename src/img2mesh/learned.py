"""Optional single-image 3D generators that supply a shape prior.

Backends turn one photo into a rough mesh, which :mod:`img2mesh.prior` then
fits to all the photos. Heavy dependencies (PyTorch, model weights) are only
imported when a backend is actually used, so the core package stays light.

Built in:

- ``"triposr"``: TripoSR (MIT licence). TripoSR is not a pip package: clone
  https://github.com/VAST-AI-Research/TripoSR, put it on ``PYTHONPATH``, and
  install ``torch omegaconf einops transformers==4.35.0 trimesh
  huggingface-hub imageio rembg onnxruntime`` plus ``torchmcubes`` (built
  from git with ``--no-build-isolation`` after ``scikit-build-core``). The
  weights (``stabilityai/TripoSR``, ~1.7 GB) download on first use. Python
  3.12 is the safest choice.

Register others with :func:`register_backend`: a factory returning a callable
``(PIL RGBA image) -> MeshResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from .config import ConfigurationError, ReconstructionError
from .mesh import MeshResult

Backend = Callable[[Image.Image], MeshResult]
_FACTORIES: dict[str, Callable[..., Backend]] = {}
_UP_HINTS: dict[str, tuple[np.ndarray, ...]] = {}
_DEPTH_AXES: dict[str, np.ndarray] = {}
_INPUT_CAMERAS: dict[str, np.ndarray] = {}
_CACHE: dict[tuple, Backend] = {}


def register_backend(
    name: str,
    factory: Callable[..., Backend],
    up_candidates: Sequence[Sequence[float]] | None = None,
    depth_axis: Sequence[float] | None = None,
    input_camera: Sequence[Sequence[float]] | None = None,
) -> None:
    """Make ``factory(**options) -> backend`` available under ``name``.

    ``up_candidates`` lists plausible directions of the real-world up axis in
    the backend's output frame (tried in turn); ``None`` means unknown.
    ``depth_axis`` is the input camera's viewing axis in that frame: the
    direction along which a single-image model's thickness guess is least
    reliable, so it is rescaled to fit the other photos. ``input_camera`` is
    the rotation (output frame to camera: x right, y down, z forward) from
    which the generator sees its input photo; that photo's camera is then
    known instead of searched for.
    """
    _FACTORIES[name] = factory
    if up_candidates is None:
        _UP_HINTS.pop(name, None)
    else:
        _UP_HINTS[name] = tuple(np.asarray(u, dtype=np.float64) for u in up_candidates)
    if depth_axis is None:
        _DEPTH_AXES.pop(name, None)
    else:
        _DEPTH_AXES[name] = np.asarray(depth_axis, dtype=np.float64)
    if input_camera is None:
        _INPUT_CAMERAS.pop(name, None)
    else:
        _INPUT_CAMERAS[name] = np.asarray(input_camera, dtype=np.float64)
    for key in [k for k in _CACHE if k[0] == name]:
        del _CACHE[key]


def backend_up_candidates(name: str) -> tuple[np.ndarray, ...] | None:
    return _UP_HINTS.get(name)


def backend_depth_axis(name: str) -> np.ndarray | None:
    return _DEPTH_AXES.get(name)


def backend_input_camera(name: str) -> np.ndarray | None:
    return _INPUT_CAMERAS.get(name)


def available_backends() -> list[str]:
    return sorted(_FACTORIES)


def foreground_rgba(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """The photo with its background made transparent, cropped to the subject."""
    rgb = np.asarray(image.convert("RGB"))
    alpha = (np.asarray(mask) > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise ReconstructionError("The foreground mask is empty.")
    pad = int(0.05 * max(np.ptp(xs), np.ptp(ys))) + 1
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    rgba = np.dstack((rgb, alpha))[y0:y1, x0:x1]
    return Image.fromarray(rgba, mode="RGBA")


def clean_mesh(mesh: MeshResult) -> MeshResult:
    """Drop zero-area triangles and unreferenced vertices (common in
    generator output, and rejected by the exporters)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    scale = float(np.max(np.ptp(vertices, axis=0))) if len(vertices) else 1.0
    faces = faces[area2 > 1e-12 * scale * scale]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return MeshResult(
        vertices=vertices[used],
        faces=remap[faces],
        alpha=None,
        input_point_count=len(vertices),
        retained_point_count=len(used),
    )


def generate_prior_mesh(
    image: Image.Image,
    mask: np.ndarray,
    backend: str = "triposr",
    **options,
) -> MeshResult:
    """Generate a rough mesh of the masked subject with a learned backend."""
    if backend not in _FACTORIES:
        raise ConfigurationError(
            f"Unknown learned backend '{backend}'. "
            f"Available: {', '.join(available_backends()) or 'none'}."
        )
    key = (backend, tuple(sorted(options.items())))
    if key not in _CACHE:
        _CACHE[key] = _FACTORIES[backend](**options)
    mesh = clean_mesh(_CACHE[key](foreground_rgba(image, mask)))
    if mesh.face_count < 4:
        raise ReconstructionError(f"The '{backend}' backend returned no surface.")
    return mesh


def _triposr_factory(
    device: str | None = None,
    resolution: int = 256,
    foreground_ratio: float = 0.85,
) -> Backend:
    try:
        import torch
        from tsr.system import TSR
        from tsr.utils import resize_foreground
    except ImportError as exc:
        raise ConfigurationError(
            "The TripoSR backend needs PyTorch, torchmcubes and a TripoSR "
            "checkout on PYTHONPATH (see img2mesh.learned for setup steps)."
        ) from exc

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)

    def run(rgba: Image.Image) -> MeshResult:
        framed = np.asarray(resize_foreground(rgba, foreground_ratio), dtype=np.float32)
        framed = framed / 255.0
        rgb = framed[:, :, :3] * framed[:, :, 3:4] + (1.0 - framed[:, :, 3:4]) * 0.5
        model_input = Image.fromarray((rgb * 255.0).astype(np.uint8))
        with torch.no_grad():
            scene_codes = model([model_input], device=device)
            meshes = model.extract_mesh(scene_codes, True, resolution=resolution)
        result = meshes[0]
        vertices = np.asarray(result.vertices, dtype=np.float64)
        return MeshResult(
            vertices=vertices,
            faces=np.asarray(result.faces, dtype=np.int64),
            alpha=None,
            input_point_count=len(vertices),
            retained_point_count=len(vertices),
        )

    return run


# TripoSR reconstructs in its input camera's frame: Z up, camera on +X looking
# at the origin. A photo taken from elevation e therefore shows the object
# with its true up axis tilted from +Z towards the camera by e.
_TRIPOSR_UP = [
    (float(np.sin(np.radians(e))), 0.0, float(np.cos(np.radians(e))))
    for e in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
]
# Verified against real output: the input photo's outline matches the mesh
# seen from +X with +Y to the image right and +Z up, within a few degrees.
register_backend(
    "triposr",
    _triposr_factory,
    up_candidates=_TRIPOSR_UP,
    depth_axis=(1.0, 0.0, 0.0),
    input_camera=((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
)


_REMBG_SESSIONS: dict[str, object] = {}


def rembg_mask(image: Image.Image, model: str = "u2net") -> np.ndarray:
    """Foreground mask from rembg (U^2-Net), optional dependency.

    Learned segmentation copes with white or pale objects on white
    backgrounds, where colour-distance rules fail (on meshref: mean IoU
    0.986 vs 0.963, and 3 vs 26 photos below 0.95).
    """
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise ConfigurationError(
            "The rembg masker needs: pip install rembg onnxruntime"
        ) from exc
    if model not in _REMBG_SESSIONS:
        _REMBG_SESSIONS[model] = new_session(model)
    alpha = remove(image.convert("RGB"), session=_REMBG_SESSIONS[model], only_mask=True)
    return np.asarray(alpha) > 127
