# img2mesh

Sparse multi-view 3D reconstruction and surface meshing from overlapping photographs.

## Overview

`img2mesh` reconstructs camera poses and sparse 3D geometry from an image set using a structure-from-motion pipeline and Delaunay alpha-shape surface extraction:

- Image decoding, EXIF orientation handling, resizing, contrast stretching, and denoising (`img2mesh.io`, `img2mesh.image`)
- Otsu and adaptive foreground segmentation (`img2mesh.image`)
- ORB / SIFT keypoint extraction, Lowe ratio matching, and conflict-free multi-view track graph construction (`img2mesh.features`)
- Five-point essential matrix estimation, cheirality pose recovery, Sampson epipolar filtering, two-view triangulation, and EPnP + Levenberg-Marquardt camera registration (`img2mesh.geometry`)
- Incremental multi-view reconstruction with sparse Jacobian bundle adjustment (`img2mesh.pipeline`)
- MAD outlier filtering and Delaunay alpha-shape / convex-hull meshing (`img2mesh.mesh`)
- Silhouette reconstruction (`img2mesh.silhouette`): visual-hull carving, or a surface-of-revolution shape prior (`img2mesh.revolve`) that rebuilds cups, bowls, vases and bottles as a smooth, hollow, watertight body with a swept handle, fits a camera pose per view, and projects vertex colours from the photos
- Wavefront OBJ, binary STL, and glTF 2.0 binary (`.glb`) export (`img2mesh.io`)

## Installation

```bash
git clone https://github.com/aidnth14/img2mesh.git
cd img2mesh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

Place two or more overlapping photographs of the same subject (with matching dimensions) into a directory and run:

```bash
img2mesh ./photos ./model.glb
```

Export as `.obj` or binary `.stl`:

```bash
img2mesh ./photos ./model.obj
img2mesh ./photos ./model.stl
```

Silhouette reconstruction suits product shots on a plain background, where
feature matching finds too little texture for structure-from-motion:

```bash
img2mesh ./photos ./model.glb --mode silhouette                       # auto shape prior
img2mesh ./photos ./model.glb --mode silhouette --shape-prior revolution
img2mesh ./photos ./model.glb --mode silhouette --shape-prior none     # plain visual hull
img2mesh ./photos ./model.glb --mode silhouette --view-angles 0:10,60:10,120:10,180:10,240:10,300:10
```

For objects that are not round, silhouettes alone cannot pin down the camera
poses (many wrong pose sets carve shapes consistent with every outline). If
the angles are known, for example from a turntable, pass them with
`--view-angles` (azimuth:elevation per photo in input order, elevation > 0
looking down): on the benchmark this roughly doubles shape accuracy for the
non-round objects.

The revolution prior handles rims, necks, lids, knobs and caps, any number of
handles and spouts on either side, and open (hollow) or closed tops. It needs
at least one near side-on view (elevation below about 25°) and assumes the
photos look at the object from the side or from above. GLB and OBJ exports
carry per-vertex colour (STL has none).

### Automatic strategy choice and quality report

`--shape-prior best` tries every applicable strategy (revolution when the
object looks round, carving, carving with `--view-angles`, a
`--learned-prior`), scores each by leave-one-out validation (rebuild without
a photo, then check how well the model explains it), builds the winner, and
prints a verdict with advice:

```text
Strategies (held-out photo IoU):
  revolution         0.968
  carving            0.938
Quality: GOOD (held-out IoU 0.968).
```

Carved results never rate better than "fair": silhouettes cannot show
hollows or deep concave areas. Advice covers objects cut off at the image
edge, missing view angles, photos all taken from one height, and too few
photos (12 turntable photos instead of 6 lift the mannequin from F 0.80 to
0.93). It is slower than `auto` (several reconstructions per strategy).

### Learned shape priors

For arbitrary objects, a rough mesh from a single-image 3D generator
(TripoSR, InstantMesh, Hunyuan3D, ...) removes the camera-pose ambiguity:
img2mesh fits every photo's camera to the prior, carves with those cameras,
intersects the result with the prior so it matches every silhouette, and
projects colour from the photos.

```bash
img2mesh ./photos ./model.glb --mode silhouette --prior-mesh generated.glb --prior-up-axis +y
img2mesh ./photos ./model.glb --mode silhouette --learned-prior triposr   # optional deps
```

The prior can be in any frame and scale (`--prior-up-axis` helps when its up
direction is known; otherwise it is detected, less reliably for symmetric
shapes). Other generators can be registered with
`img2mesh.learned.register_backend`. The built-in `triposr` backend needs a
separate environment (Python 3.12 recommended):

```bash
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR ~/TripoSR
uv venv --python 3.12 .venv-learned && source .venv-learned/bin/activate
uv pip install torch omegaconf einops "transformers==4.35.0" trimesh \
    huggingface-hub imageio rembg onnxruntime scikit-build-core pybind11 cmake ninja
uv pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
uv pip install -e .
export PYTHONPATH=~/TripoSR   # TripoSR is not a pip package
```

Its weights (~1.7 GB) download from Hugging Face on first use.

### Accuracy benchmark

`python -m img2mesh.benchmark` renders ten synthetic objects with known
shapes (cup, bowl, vase, capped bottle, lidded jar, teapot, amphora, box,
toy duck, shoe) as product shots, reconstructs each from six views, aligns
the result to the ground truth and reports F-score / precision / recall at
2% of the object size, Chamfer distance, genus, and silhouette IoU from a
seventh held-out view. Self-reported IoU on the input views is shown only for
contrast: carving-style methods score well on it even when the shape is wrong.

```bash
python -m img2mesh.benchmark                                  # everything (11 objects)
python -m img2mesh.benchmark --objects cup,box --methods auto,none
python -m img2mesh.benchmark --json results.json --save-meshes meshes/
python -m img2mesh.benchmark --methods none,auto,angles,prior_up
```

Current results (F-score at 2% of object size; Chamfer as % of size):

| Method | Round objects (7) | Other objects (3) |
|---|---|---|
| `none` (plain visual hull) | F 0.41, CD 3.3% | F 0.36, CD 3.3% |
| `auto` (revolution prior) | **F 0.99, CD 0.6%** | F 0.36, CD 3.3% (falls back) |
| `angles` (hull, known view angles) | F 0.68, CD 2.0% | F 0.72, CD 1.6% |
| `prior_up` (simulated learned prior) | F 0.76, CD 1.4% | **F 0.84, CD 1.2%** |

`prior_up` uses a degraded copy of the ground truth as a stand-in for a
generator's output; it measures how well a decent prior is exploited, not
the quality of any particular learned model. With the real TripoSR backend
(`--methods learned`, optional environment) the non-round objects reach
F 0.45 (shoe 0.67, duck 0.41, box 0.26; plain hull 0.36): TripoSR's single-view
guesses on these renders are far weaker than the stand-in, mostly in
thickness, which silhouettes cannot correct. On the real cup photos the
TripoSR path matches the revolution prior (silhouette IoU 0.957).

### CLI Options

```text
img2mesh <input_image_or_dir> <output_model> [options]

Options:
  -v, --verbose             Enable debug logging
  --overwrite               Overwrite an existing output file
  --validate-only           Decode and validate inputs without running reconstruction
  --recursive               Search input directories recursively
  --method {orb,sift}       Feature detector (default: orb)
  --max-features INT        Maximum features per image (default: 2000)
  --pairing-strategy {all,sequential}
                            Image pairing strategy (default: all)
  --alpha FLOAT             Alpha-shape radius (omit for convex hull)
  --mode {sfm,silhouette,auto}
                            Reconstruction mode (default: sfm)
  --shape-prior {auto,best,revolution,none}
                            Silhouette shape model; 'best' validates and reports
                            (default: auto)
  --view-angles AZ:EL,...   Known camera angles per photo (silhouette mode)
  --prior-mesh PATH         Shape prior mesh (.obj/.stl/.glb) to fit and refine
  --prior-up-axis {+x,-x,+y,-y,+z,-z}
                            Up axis of the prior mesh, if known
  --learned-prior BACKEND   Generate the prior with a learned backend
  --no-bundle-adjustment    Skip bundle adjustment refinement
  --focal-length FLOAT      Camera focal length in pixels
  --max-dimension INT       Resize images so max(width, height) <= INT
  --normalize-contrast      Stretch image contrast before feature detection
  --denoise {none,median,gaussian}
                            Denoising filter (default: none)
  --version                 Show version and exit
```

## Project Layout

```text
src/img2mesh/
  __init__.py   Public package exports
  cli.py        Command-line interface
  config.py     Configuration and exception hierarchy
  features.py   Feature detection, matching, tracks, and view graph
  geometry.py   Camera intrinsics, two-view pose, triangulation, and PnP
  image.py      Preprocessing and foreground segmentation
  io.py         Image loading and OBJ / STL / GLB exporters
  mesh.py       Point-cloud filtering and alpha-shape meshing
  pipeline.py   Bundle adjustment and incremental SfM pipeline
  revolve.py    Surface-of-revolution shape prior, pose fitting, vertex colours
  prior.py      Reconstruction guided by an external/learned prior mesh
  learned.py    Optional learned backends (TripoSR) producing prior meshes
  validate.py   Leave-one-out validation, strategy selection, quality report
  benchmark/    Synthetic objects, renderer, shape metrics, benchmark runner
  silhouette.py Foreground masks, visual hull, marching tetrahedra
tests/
  test_cli.py
  test_features.py
  test_geometry.py
  test_image.py
  test_io.py
  test_benchmark.py
  test_pipeline.py
  test_prior.py
  test_revolve.py
  test_silhouette.py
  test_validate.py
```

## Testing & Linting

```bash
pytest
ruff check .
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
