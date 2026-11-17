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
tests/
  test_cli.py
  test_features.py
  test_geometry.py
  test_image.py
  test_io.py
  test_pipeline.py
```

## Testing & Linting

```bash
pytest
ruff check .
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
