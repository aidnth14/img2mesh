"""Optional single-image 3D generators that supply a shape prior.

Backends turn one photo into a rough mesh, which :mod:`img2mesh.prior` then
fits to all the photos. Heavy dependencies (PyTorch, model weights) are only
imported when a backend is actually used, so the core package stays light.

Built in:

- ``"triposr"``: TripoSR (MIT licence). Needs ``torch`` and the TripoSR
  package (``pip install git+https://github.com/VAST-AI-Research/TripoSR``);
  the weights (``stabilityai/TripoSR``) download on first use.

Register others with :func:`register_backend`: a factory returning a callable
``(PIL RGBA image) -> MeshResult``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from .config import ConfigurationError, ReconstructionError
from .mesh import MeshResult

Backend = Callable[[Image.Image], MeshResult]
_FACTORIES: dict[str, Callable[..., Backend]] = {}
_CACHE: dict[tuple, Backend] = {}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    """Make ``factory(**options) -> backend`` available under ``name``."""
    _FACTORIES[name] = factory
    for key in [k for k in _CACHE if k[0] == name]:
        del _CACHE[key]


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
    mesh = _CACHE[key](foreground_rgba(image, mask))
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
            "The TripoSR backend needs PyTorch and TripoSR: pip install torch "
            "git+https://github.com/VAST-AI-Research/TripoSR"
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


register_backend("triposr", _triposr_factory)
