# Changelog

## Unreleased

- Surface-of-revolution shape prior for silhouette reconstruction
  (`img2mesh.revolve`, `reconstruct_visual_hull(shape_prior=...)`,
  `--shape-prior`). Rotationally symmetric objects become a smooth,
  hollow, watertight body plus swept handle instead of a faceted carved hull.
- Per-view camera pose fitting against the known shape, constrained to
  views from the side or above.
- Per-vertex colour projected from the photographs (`MeshResult.vertex_colors`),
  exported as OBJ `v x y z r g b` and glTF `COLOR_0`.
- `VisualHullResult.method` reports which model produced the mesh.

## 0.1.0

- Multi-view structure-from-motion pipeline (`img2mesh.pipeline`, `img2mesh.geometry`, `img2mesh.features`).
- Image loading, EXIF normalization, preprocessing, and segmentation (`img2mesh.io`, `img2mesh.image`).
- Point-cloud MAD outlier filtering and Delaunay alpha-shape meshing (`img2mesh.mesh`).
- Wavefront OBJ, binary STL, and glTF 2.0 binary (`.glb`) exporters (`img2mesh.io`).
- Command-line interface (`img2mesh.cli`).
