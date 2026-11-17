"""Synthetic accuracy benchmark for img2mesh reconstructions."""

from .metrics import align_similarity, mesh_topology, sample_surface, shape_scores
from .objects import BUILDERS, BenchObject, get_object
from .render import DEFAULT_VIEWS, RenderedView, render_view, render_views
from .run import BenchResult, evaluate, format_table, ground_truth_mesh, run

__all__ = [
    "BUILDERS",
    "BenchObject",
    "BenchResult",
    "DEFAULT_VIEWS",
    "RenderedView",
    "align_similarity",
    "evaluate",
    "format_table",
    "get_object",
    "ground_truth_mesh",
    "mesh_topology",
    "render_view",
    "render_views",
    "run",
    "sample_surface",
    "shape_scores",
]
