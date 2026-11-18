"""Textured scanned meshes (e.g. Google Scanned Objects) as benchmark subjects.

Loads an OBJ with UVs plus its texture, normalises it (Y up, standing on
y = 0, longest side 1), and renders product-style photos with the same
camera and lighting model as :mod:`img2mesh.benchmark.render`, using a
vectorised z-buffer rasteriser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ..geometry import CameraIntrinsics
from .render import _FILL_LIGHT, _KEY_LIGHT, RenderedView, look_at

Image.MAX_IMAGE_PIXELS = None  # scanned-object textures are large


@dataclass(frozen=True)
class TexturedMesh:
    vertices: np.ndarray  # (N, 3), Y up, base on y = 0
    faces: np.ndarray  # (F, 3) vertex indices
    uvs: np.ndarray  # (M, 2)
    face_uvs: np.ndarray  # (F, 3) uv indices
    texture: np.ndarray  # (H, W, 3) float in [0, 1]
    vertex_normals: np.ndarray
    # Multi-material models (glTF scenes): per-face texture index into
    # ``textures`` (-1 = untextured, coloured by ``face_colors``).
    textures: tuple[np.ndarray, ...] = ()
    face_tex: np.ndarray | None = None
    face_colors: np.ndarray | None = None

    def albedo(self, face_idx: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Surface colour for faces ``face_idx`` at texture coordinates ``uv``."""
        if self.face_tex is None:
            return sample_texture(self.texture, uv)
        out = np.empty((len(face_idx), 3))
        tex_id = self.face_tex[face_idx]
        for t in np.unique(tex_id):
            sel = tex_id == t
            if t < 0:
                out[sel] = self.face_colors[face_idx[sel]]
            else:
                out[sel] = sample_texture(self.textures[t], uv[sel])
        return out

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        lo, hi = self.bounds
        return 0.5 * (lo + hi)

    @property
    def radius(self) -> float:
        lo, hi = self.bounds
        return float(0.5 * np.linalg.norm(hi - lo))

    def vertex_colors(self) -> np.ndarray:
        """Per-vertex colour from the texture (first UV seen per vertex)."""
        if self.face_tex is not None:
            corner_faces = np.repeat(np.arange(len(self.faces)), 3)
            corner_uv = self.uvs[self.face_uvs.ravel()]
            corner_col = self.albedo(corner_faces, corner_uv)
            total = np.zeros((len(self.vertices), 3))
            count = np.zeros(len(self.vertices))
            np.add.at(total, self.faces.ravel(), corner_col)
            np.add.at(count, self.faces.ravel(), 1.0)
            return total / np.maximum(count, 1.0)[:, None]
        colors = np.zeros((len(self.vertices), 3))
        uv_for_vertex = np.full(len(self.vertices), -1, dtype=np.int64)
        uv_for_vertex[self.faces.ravel()[::-1]] = self.face_uvs.ravel()[::-1]
        valid = uv_for_vertex >= 0
        colors[valid] = sample_texture(self.texture, self.uvs[uv_for_vertex[valid]])
        return colors


def sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = (1.0 - np.mod(uv[:, 1], 1.0)) * (h - 1)
    coords = np.vstack((v, u))
    return np.column_stack(
        [
            map_coordinates(texture[:, :, c], coords, order=1, mode="nearest")
            for c in range(3)
        ]
    )


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], fn)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _normalise(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = 1.0 / max(float(np.max(hi - lo)), 1e-12)
    return (v - np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])) * scale


def _rgb_factor(material) -> np.ndarray:
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        factor = getattr(material, "main_color", None)
    if factor is None:
        return np.ones(3)
    factor = np.asarray(factor, dtype=np.float64)[:3]
    return factor / 255.0 if factor.max() > 1.0 else factor


def load_glb_textured(path: str | Path, *, max_texture: int = 2048) -> TexturedMesh:
    """Load a glTF/GLB scene (node transforms, several materials) as one mesh.

    glTF is Y-up, so no axis swap; the result is normalised like
    :func:`load_textured_obj`. Needs ``trimesh`` (dataset building only).
    """
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    meshes = scene.dump(concatenate=False) if hasattr(scene, "dump") else [scene]
    verts, faces, uvs, face_uvs, face_tex, face_cols = (
        [],
        [],
        [np.zeros((1, 2))],
        [],
        [],
        [],
    )
    textures: list[np.ndarray] = []
    tex_index: dict[int, int] = {}
    v_off, uv_off = 0, 1
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        visual = mesh.visual
        tex_id, colour = -1, np.full(3, 0.6)
        kind = getattr(visual, "kind", None)
        uv = getattr(visual, "uv", None) if kind == "texture" else None
        if kind == "texture":
            material = visual.material
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            factor = _rgb_factor(material)
            if image is not None and uv is not None and len(uv) == len(v):
                key = id(image)
                if key not in tex_index:
                    img = image.convert("RGB")
                    if max(img.size) > max_texture:
                        img.thumbnail(
                            (max_texture, max_texture), Image.Resampling.LANCZOS
                        )
                    arr = np.asarray(img, dtype=np.float64) / 255.0
                    tex_index[key] = len(textures)
                    textures.append(np.clip(arr * factor[None, None, :], 0.0, 1.0))
                tex_id = tex_index[key]
            else:
                colour = factor
                uv = None
        elif kind == "vertex":
            vc = np.asarray(visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            colour = None
            per_face = vc[f].mean(axis=1)
        else:
            colour = _rgb_factor(getattr(visual, "material", None))
        verts.append(v)
        faces.append(f + v_off)
        if tex_id >= 0:
            uvs.append(np.asarray(uv, dtype=np.float64))
            face_uvs.append(f + uv_off)
            uv_off += len(uv)
        else:
            face_uvs.append(np.zeros_like(f))
        face_tex.append(np.full(len(f), tex_id, dtype=np.int64))
        face_cols.append(per_face if colour is None else np.tile(colour, (len(f), 1)))
        v_off += len(v)
    if not verts:
        raise ValueError(f"No triangle meshes in {path}")
    v = _normalise(np.vstack(verts))
    f = np.vstack(faces)
    return TexturedMesh(
        vertices=v,
        faces=f,
        uvs=np.vstack(uvs),
        face_uvs=np.vstack(face_uvs),
        texture=textures[0] if textures else np.full((2, 2, 3), 0.6),
        vertex_normals=_vertex_normals(v, f),
        textures=tuple(textures),
        face_tex=np.concatenate(face_tex),
        face_colors=np.vstack(face_cols),
    )


def load_textured_obj(
    obj_path: str | Path,
    texture_path: str | Path | None,
    *,
    z_up: bool = True,
) -> TexturedMesh:
    """Load and normalise a textured OBJ (Z-up by default, as in Gazebo)."""
    positions, uvs, faces, face_uvs = [], [], [], []
    with open(obj_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                vi, ti = [], []
                for token in tokens:
                    parts = token.split("/")
                    vi.append(int(parts[0]))
                    ti.append(int(parts[1]) if len(parts) > 1 and parts[1] else 0)
                for k in range(1, len(vi) - 1):
                    faces.append([vi[0], vi[k], vi[k + 1]])
                    face_uvs.append([ti[0], ti[k], ti[k + 1]])
    v = np.asarray(positions, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    f = np.where(f > 0, f - 1, len(v) + f)
    t = np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((1, 2))
    ft = np.asarray(face_uvs, dtype=np.int64)
    ft = np.where(ft > 0, ft - 1, np.where(ft < 0, len(t) + ft, 0))

    if z_up:  # (x, y, z) Z-up -> (x, z, -y) Y-up
        v = np.column_stack((v[:, 0], v[:, 2], -v[:, 1]))
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = 1.0 / max(float(np.max(hi - lo)), 1e-12)
    v = (v - np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])) * scale

    if texture_path is not None and Path(texture_path).is_file():
        with Image.open(texture_path) as img:
            img = img.convert("RGB")
            if max(img.size) > 2048:
                img = img.resize((2048, 2048), Image.Resampling.LANCZOS)
            tex = np.asarray(img, dtype=np.float64) / 255.0
    else:
        tex = np.full((2, 2, 3), 0.6)
    return TexturedMesh(v, f, t, ft, tex, _vertex_normals(v, f))


def render_mesh_view(
    mesh: TexturedMesh,
    azimuth: float,
    elevation: float,
    *,
    size: int = 256,
    fill: float = 0.78,
    distance_factor: float = 12.0,
    supersample: int = 2,
) -> RenderedView:
    """Product-style photo of a textured mesh (white background)."""
    target = mesh.center
    radius = mesh.radius
    distance = distance_factor * radius
    focal = fill * 0.5 * size * distance / radius
    intrinsics = CameraIntrinsics.from_focal_length(size, size, focal)
    rotation, translation, eye = look_at(azimuth, elevation, distance, target)

    big = size * supersample
    f_big = focal * supersample
    cam = mesh.vertices @ rotation.T + translation
    z = cam[:, 2]
    px = f_big * cam[:, 0] / z + 0.5 * big - 0.5
    py = f_big * cam[:, 1] / z + 0.5 * big - 0.5

    faces = mesh.faces
    fx, fy, fz = px[faces], py[faces], z[faces]
    x0 = np.clip(np.floor(fx.min(axis=1)), 0, big - 1).astype(np.int64)
    x1 = np.clip(np.ceil(fx.max(axis=1)), 0, big - 1).astype(np.int64)
    y0 = np.clip(np.floor(fy.min(axis=1)), 0, big - 1).astype(np.int64)
    y1 = np.clip(np.ceil(fy.max(axis=1)), 0, big - 1).astype(np.int64)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    area = w * h
    keep = (area > 0) & np.all(fz > 1e-6, axis=1)

    image = np.ones((big * big, 3))
    covered = np.zeros(big * big, dtype=bool)
    idx_all = np.flatnonzero(keep)
    # Process faces in chunks to bound memory.
    best_depth = np.full(big * big, np.inf)
    best_face = np.full(big * big, -1, dtype=np.int64)
    best_bary = np.zeros((big * big, 3))
    chunk = 20000
    for start in range(0, len(idx_all), chunk):
        sel = idx_all[start : start + chunk]
        counts = area[sel]
        face_of = np.repeat(sel, counts)
        offsets = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        ww = w[face_of]
        qx = x0[face_of] + offsets % ww
        qy = y0[face_of] + offsets // ww
        ax, ay = fx[face_of, 0], fy[face_of, 0]
        bx, by = fx[face_of, 1], fy[face_of, 1]
        cx, cy = fx[face_of, 2], fy[face_of, 2]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(denom) > 1e-12
        safe = np.where(ok, denom, 1.0)
        l0 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / safe
        l1 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / safe
        l2 = 1.0 - l0 - l1
        inside = ok & (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        face_of, qx, qy = face_of[inside], qx[inside], qy[inside]
        bary = np.column_stack((l0[inside], l1[inside], l2[inside]))
        depth = np.einsum("ij,ij->i", bary, fz[face_of])
        pix = qy * big + qx
        order = np.lexsort((depth, pix))
        pix, depth, face_of, bary = (
            pix[order],
            depth[order],
            face_of[order],
            bary[order],
        )
        first = np.concatenate(([True], pix[1:] != pix[:-1]))
        pix, depth, face_of, bary = (
            pix[first],
            depth[first],
            face_of[first],
            bary[first],
        )
        nearer = depth < best_depth[pix]
        pix = pix[nearer]
        best_depth[pix] = depth[nearer]
        best_face[pix] = face_of[nearer]
        best_bary[pix] = bary[nearer]

    hit = best_face >= 0
    covered[hit] = True
    pix = np.flatnonzero(hit)
    fidx = best_face[pix]
    bary = best_bary[pix]
    uv = np.einsum("ij,ijk->ik", bary, mesh.uvs[mesh.face_uvs[fidx]])
    albedo = mesh.albedo(fidx, uv)
    normal = np.einsum("ij,ijk->ik", bary, mesh.vertex_normals[faces[fidx]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    points = np.einsum("ij,ijk->ik", bary, mesh.vertices[faces[fidx]])
    view_dir = eye - points
    flip = np.sum(normal * view_dir, axis=1) < 0
    normal[flip] *= -1.0
    key = np.clip(normal @ _KEY_LIGHT, 0.0, 1.0)
    fill_l = np.clip(normal @ _FILL_LIGHT, 0.0, 1.0)
    light = 0.32 + 0.72 * key + 0.22 * fill_l
    image[pix] = np.clip(albedo * light[:, None] * 1.1, 0.0, 1.0)

    rgb = image.reshape(big, big, 3)
    mask = covered.reshape(big, big)
    if supersample > 1:
        rgb = rgb.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
        mask = (
            mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3)) >= 0.5
        )
    return RenderedView(
        image=Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)),
        mask=mask,
        rotation=rotation,
        translation=translation,
        intrinsics=intrinsics,
        azimuth=azimuth,
        elevation=elevation,
    )
