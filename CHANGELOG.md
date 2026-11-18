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
- Accuracy benchmark (`python -m img2mesh.benchmark`): ten SDF objects
  rendered as product shots, scored against ground truth (F-score, Chamfer,
  genus, held-out-view IoU).
- Revolution prior generalised beyond cups: lip/neck/knob-aware elevation,
  symmetric radius robust to handles and spouts, several appendages per view,
  roundness test that rejects boxes and other non-round objects, open-top
  detection from ~30 degree views, camera-elevation feedback, per-part
  out-of-plane thickness chosen by silhouette overlap. Benchmark (round
  objects): F@2% 0.42 -> 0.99, Chamfer 3.3% -> 0.6%, genus 5/7 -> 7/7.
- Known camera angles (`view_angles=`, `--view-angles`) for silhouette
  carving: non-round benchmark objects F@2% 0.35 -> 0.63, Chamfer 3.3% -> 1.9%.
- Carved hulls now use a smoothed signed-distance surface plus Taubin
  smoothing and carry projected vertex colours.
- Silhouette mode assumes a longer lens by default (focal = 4x image size),
  matching product photography (real cup sample IoU 0.94 -> 0.96).
- Shape priors from learned generators (`prior_mesh=`, `--prior-mesh`,
  `--prior-up-axis`): cameras are fitted to the prior (up axis detected or
  given, mutually consistent poses chosen by carving consistency), the
  visual hull is intersected with the prior and coloured from the photos.
  Optional `learned_backend=` / `--learned-prior triposr` generates the prior
  (`img2mesh.learned`, lazy optional dependencies, pluggable registry).
  Benchmark with a simulated prior: non-round objects F@2% 0.36 -> 0.84.
- `img2mesh.io.load_mesh` reads .obj, .stl (binary/ASCII, welded) and .glb.
- Marching tetrahedra no longer leaves pinholes when grid values sit within
  ~1e-7 of the iso-level.

## 0.1.0

- Multi-view structure-from-motion pipeline (`img2mesh.pipeline`, `img2mesh.geometry`, `img2mesh.features`).
- Image loading, EXIF normalization, preprocessing, and segmentation (`img2mesh.io`, `img2mesh.image`).
- Point-cloud MAD outlier filtering and Delaunay alpha-shape meshing (`img2mesh.mesh`).
- Wavefront OBJ, binary STL, and glTF 2.0 binary (`.glb`) exporters (`img2mesh.io`).
- Command-line interface (`img2mesh.cli`).
