"""A stand-in for LATO.2's WhiteModelRenderer (dataset/mesh_render.py) on nvdiffrast's CUDA rasterizer, or on the
CPU (NumPy) where nvdiffrast is not installed, e.g. on Windows, which has no nvdiffrast build to install.

LATO.2 conditions its vertex flow on a DINOv2 view of the input: a light grey model on black, lit by a sun from
above with shadows, seen from an orbit camera (distance 1.8, 50 degree field of view) and cropped to the object.
The original renders with Open3D/Filament, which needs a working EGL or Vulkan display stack and silently returns a
black image in some containers. This renderer keeps the same camera, crop, colours and light direction; the shading
is a plain Lambert sun plus ambient term with a shadow map, close to but not the same as Filament's.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def _look_at(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _perspective(fov_deg, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2.0 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj


def _orthographic(half, near, far):
    proj = np.eye(4)
    proj[0, 0] = proj[1, 1] = 1.0 / half
    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)
    return proj


BACKEND = "auto"  # "nvdiffrast" (CUDA), "numpy" (CPU, nothing to build) or "auto": nvdiffrast when it works
SHADOW_RES = 2048


def _nvdiffrast_context():
    """nvdiffrast's CUDA rasterizer, or None where it is not installed or cannot start."""
    try:
        import nvdiffrast.torch as dr
        if not torch.cuda.is_available():
            return None
        return dr, dr.RasterizeCudaContext()
    except Exception:
        return None


def _screen_barycentrics(px, py, triangles, ids, size):
    """For the pixels of an ID buffer (ids, flat, -1 where empty) that a triangle covers: their flat indices, the
    triangle and its screen-space barycentric weights at the pixel centre (i + 0.5)."""
    pixel = np.nonzero(ids >= 0)[0]
    t = ids[pixel]
    cx = (pixel % size) + 0.5
    cy = (pixel // size) + 0.5
    x, y = px[triangles[t]], py[triangles[t]]
    ex1, ey1 = x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]
    ex2, ey2 = x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]
    inv = 1.0 / (ex1 * ey2 - ex2 * ey1)
    l1 = (ey2 * (cx - x[:, 0]) - ex2 * (cy - y[:, 0])) * inv
    l2 = (ex1 * (cy - y[:, 0]) - ey1 * (cx - x[:, 0])) * inv
    return pixel, t, np.stack([1.0 - l1 - l2, l1, l2], axis=1)


class WhiteModelRenderer:
    def __init__(self, img_res: int = 518, mesh_color=(0.78, 0.78, 0.82), bg_color=(0.0, 0.0, 0.0),
                 up_axis: str = "y", add_ground: bool = False, shadow: bool = True, camera_distance: float = 1.8,
                 fov: float = 50.0, crop_to_object: bool = True, crop_padding: float = 1.2, backend: str = None,
                 **_ignored):
        backend = backend or BACKEND
        found = _nvdiffrast_context() if backend in ("auto", "nvdiffrast") else None
        if backend == "nvdiffrast" and found is None:
            raise RuntimeError("nvdiffrast is not available (install it, or use the numpy renderer)")
        self.backend = "nvdiffrast" if found is not None else "numpy"
        if found is not None:
            self.dr, self.ctx = found
        print(f"[lato2] conditioning view renderer: {self.backend}", flush=True)
        self.img_res = int(img_res)
        self.mesh_color = np.asarray(mesh_color, dtype=np.float32)
        self.bg_color = np.asarray(bg_color, dtype=np.float32)
        self.axis = {"x": 0, "y": 1, "z": 2}[up_axis.lower()]
        self.shadow = shadow
        self.camera_distance = float(camera_distance)
        self.fov = float(fov)
        self.crop_to_object = crop_to_object
        self.crop_padding = float(crop_padding)

    def _eye(self, center, azimuth, elevation):
        az, el = np.radians(azimuth), np.radians(elevation)
        plane = [i for i in range(3) if i != self.axis]
        offset = np.zeros(3)
        offset[plane[0]] = self.camera_distance * np.cos(el) * np.cos(az)
        offset[plane[1]] = self.camera_distance * np.cos(el) * np.sin(az)
        offset[self.axis] = self.camera_distance * np.sin(el)
        up = np.zeros(3)
        up[self.axis] = 1.0
        return center + offset, up

    def _clip(self, verts, mvp):
        homo = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
        return (homo @ torch.as_tensor(mvp.T, dtype=torch.float32, device=verts.device))[None]

    def render(self, vertices: np.ndarray, faces: np.ndarray, num_views: int = 1, mesh_color=None,
               azimuths: Optional[Sequence[float]] = None, elevations: Optional[Sequence[float]] = None,
               seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[dict]]:
        if self.backend == "numpy":
            return self._render_numpy(vertices, faces, num_views, mesh_color, azimuths, elevations)
        dr = self.dr
        device = "cuda"
        verts = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
        tris = torch.as_tensor(np.asarray(faces, dtype=np.int32), device=device)
        a, b, c = verts[tris[:, 0].long()], verts[tris[:, 1].long()], verts[tris[:, 2].long()]
        face_n = torch.cross(b - a, c - a, dim=1)
        vert_n = torch.zeros_like(verts)
        for k in range(3):
            vert_n.index_add_(0, tris[:, k].long(), face_n)
        vert_n = torch.nn.functional.normalize(vert_n, dim=1)
        lo = verts.min(dim=0).values.cpu().numpy()
        hi = verts.max(dim=0).values.cpu().numpy()
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0 + 1e-6
        color = np.asarray(mesh_color, dtype=np.float32) if mesh_color is not None else self.mesh_color

        # Sun as in mesh_render.py: (0.35, 0.35, 0.35) with the up component -1, i.e. falling from above.
        sun = np.array([0.35, 0.35, 0.35])
        sun[self.axis] = -1.0
        sun = sun / np.linalg.norm(sun)
        res = self.img_res
        shadow_res = SHADOW_RES
        light_eye = center - sun * radius * 3.0
        light_up = np.zeros(3)
        light_up[(self.axis + 1) % 3] = 1.0
        light_mvp = _orthographic(radius * 1.05, radius, radius * 5.0) @ _look_at(light_eye, center, light_up)
        light_clip = self._clip(verts, light_mvp)
        light_rast, _ = dr.rasterize(self.ctx, light_clip, tris, resolution=[shadow_res, shadow_res])
        light_depth = torch.where(light_rast[..., 3:4] > 0, light_rast[..., 2:3], torch.ones_like(light_rast[..., 2:3]))

        azimuths = list(azimuths) if azimuths is not None else list(np.linspace(0, 360, num_views, endpoint=False))
        elevations = list(elevations) if elevations is not None else [30.0] * len(azimuths)
        images, infos = [], []
        for azimuth, elevation in zip(azimuths, elevations):
            eye, up = self._eye(center, azimuth, elevation)
            mvp = _perspective(self.fov, 0.01, 20.0) @ _look_at(eye, center, up)
            clip = self._clip(verts, mvp)
            rast, _ = dr.rasterize(self.ctx, clip, tris, resolution=[res, res])
            mask = rast[..., 3:4] > 0
            normal, _ = dr.interpolate(vert_n[None], rast, tris)
            normal = torch.nn.functional.normalize(normal, dim=-1)
            position, _ = dr.interpolate(verts[None], rast, tris)
            view_dir = torch.as_tensor(eye, dtype=torch.float32, device=device) - position
            # Two-sided: flip normals that face away from the camera.
            facing = (normal * view_dir).sum(-1, keepdim=True) < 0
            normal = torch.where(facing, -normal, normal)
            to_sun = torch.as_tensor(-sun, dtype=torch.float32, device=device)
            lambert = (normal * to_sun).sum(-1, keepdim=True).clamp(min=0.0)
            lit = torch.ones_like(lambert)
            if self.shadow:
                pos_clip = self._clip(position.reshape(-1, 3), light_mvp)[0]
                ndc = pos_clip[:, :3] / pos_clip[:, 3:4]
                uv = (ndc[:, :2] * 0.5 + 0.5).reshape(1, res, res, 2)
                stored = dr.texture(light_depth.contiguous(), uv.contiguous(), filter_mode="nearest")
                own = ndc[:, 2].reshape(1, res, res, 1)
                lit = (own <= stored + 0.004).float()
            # Filament's image-based light fills the side the sun misses; approximate it with sky light from
            # above and a weak fill from the camera so the shape stays readable there.
            up = torch.zeros(3, device=device)
            up[self.axis] = 1.0
            sky = 0.5 + 0.5 * (normal * up).sum(-1, keepdim=True)
            fill = (normal * torch.nn.functional.normalize(view_dir, dim=-1)).sum(-1, keepdim=True).clamp(min=0.0)
            shade = 0.12 + 0.22 * sky + 0.30 * fill + 0.65 * lambert * lit
            rgb = torch.as_tensor(color, device=device) * shade
            rgb = rgb.clamp(0, 1) ** (1.0 / 2.2)
            rgb = torch.where(mask, rgb, torch.as_tensor(self.bg_color, device=device))
            image = (rgb[0].flip(0).cpu().numpy() * 255).astype(np.uint8)
            object_mask = mask[0, ..., 0].flip(0).cpu().numpy()
            if self.crop_to_object:
                image = self._crop(image, object_mask)
            images.append(np.ascontiguousarray(image))
            infos.append({"azimuth": float(azimuth), "elevation": float(elevation)})
        return images, infos

    def _render_numpy(self, vertices, faces, num_views, mesh_color, azimuths, elevations):
        """The same camera, light, shadow test and shading on the CPU (exterior_fill's NumPy ID-buffer rasterizer, with
        perspective-correct interpolation), for machines without nvdiffrast."""
        from exterior_fill import _nearest_triangles
        verts = np.asarray(vertices, dtype=np.float32)
        tris = np.asarray(faces, dtype=np.int64)
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        face_n = np.cross(b - a, c - a)
        vert_n = np.zeros_like(verts)
        for k in range(3):
            np.add.at(vert_n, tris[:, k], face_n)
        vert_n /= np.maximum(np.linalg.norm(vert_n, axis=1, keepdims=True), 1e-12)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0 + 1e-6
        color = np.asarray(mesh_color, dtype=np.float32) if mesh_color is not None else self.mesh_color
        homo = np.concatenate([verts, np.ones_like(verts[:, :1])], axis=1)

        def project(mvp, size):
            clip = (homo @ mvp.T.astype(np.float32)).astype(np.float64)
            ndc = clip[:, :3] / clip[:, 3:4]
            return (ndc[:, 0] * 0.5 + 0.5) * size, (ndc[:, 1] * 0.5 + 0.5) * size, ndc[:, 2], clip[:, 3]

        sun = np.array([0.35, 0.35, 0.35])
        sun[self.axis] = -1.0
        sun = sun / np.linalg.norm(sun)
        res = self.img_res
        light_eye = center - sun * radius * 3.0
        light_up = np.zeros(3)
        light_up[(self.axis + 1) % 3] = 1.0
        light_mvp = _orthographic(radius * 1.05, radius, radius * 5.0) @ _look_at(light_eye, center, light_up)
        light_depth = None
        if self.shadow:
            lx, ly, lz, _ = project(light_mvp, SHADOW_RES)
            ids = _nearest_triangles(lx, ly, lz, tris, SHADOW_RES)
            pixel, t, bary = _screen_barycentrics(lx, ly, tris, ids, SHADOW_RES)
            light_depth = np.ones(SHADOW_RES * SHADOW_RES)
            light_depth[pixel] = (bary * lz[tris[t]]).sum(axis=1)

        azimuths = list(azimuths) if azimuths is not None else list(np.linspace(0, 360, num_views, endpoint=False))
        elevations = list(elevations) if elevations is not None else [30.0] * len(azimuths)
        up = np.zeros(3)
        up[self.axis] = 1.0
        images, infos = [], []
        for azimuth, elevation in zip(azimuths, elevations):
            eye, cam_up = self._eye(center, azimuth, elevation)
            px, py, pz, pw = project(_perspective(self.fov, 0.01, 20.0) @ _look_at(eye, center, cam_up), res)
            ids = _nearest_triangles(px, py, pz, tris, res)
            pixel, t, bary = _screen_barycentrics(px, py, tris, ids, res)
            # Perspective-correct weights, as nvdiffrast interpolates.
            bary = bary / pw[tris[t]]
            bary /= bary.sum(axis=1, keepdims=True)
            normal = (bary[:, :, None] * vert_n[tris[t]]).sum(axis=1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            position = (bary[:, :, None] * verts[tris[t]]).sum(axis=1)
            view_dir = eye[None, :] - position
            normal = np.where((normal * view_dir).sum(axis=1, keepdims=True) < 0, -normal, normal)
            lambert = np.clip(normal @ -sun, 0.0, None)
            lit = np.ones(len(pixel))
            if light_depth is not None:
                clip = np.concatenate([position, np.ones((len(position), 1))], axis=1) @ light_mvp.T
                ndc = clip[:, :3] / clip[:, 3:4]
                tx = np.floor((ndc[:, 0] * 0.5 + 0.5) * SHADOW_RES).astype(np.int64) % SHADOW_RES
                ty = np.floor((ndc[:, 1] * 0.5 + 0.5) * SHADOW_RES).astype(np.int64) % SHADOW_RES
                lit = (ndc[:, 2] <= light_depth[ty * SHADOW_RES + tx] + 0.004).astype(np.float64)
            sky = 0.5 + 0.5 * (normal @ up)
            unit_view = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-12)
            fill = np.clip((normal * unit_view).sum(axis=1), 0.0, None)
            shade = 0.12 + 0.22 * sky + 0.30 * fill + 0.65 * lambert * lit
            rgb = np.tile(self.bg_color, (res * res, 1)).astype(np.float64)
            rgb[pixel] = np.clip(color[None, :] * shade[:, None], 0.0, 1.0) ** (1.0 / 2.2)
            image = (rgb.reshape(res, res, 3)[::-1] * 255).astype(np.uint8)
            object_mask = (ids >= 0).reshape(res, res)[::-1]
            if self.crop_to_object:
                image = self._crop(image, object_mask)
            images.append(np.ascontiguousarray(image))
            infos.append({"azimuth": float(azimuth), "elevation": float(elevation)})
        return images, infos

    def _crop(self, rgb, mask):
        from PIL import Image
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.asarray(Image.fromarray(rgb).resize((self.img_res, self.img_res), Image.LANCZOS))
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        size = max(int(max(x1 - x0, y1 - y0) * self.crop_padding), 1)
        half = size // 2
        bx0, by0 = int(round(cx - half)), int(round(cy - half))
        h, w = rgb.shape[:2]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        sx0, sy0, sx1, sy1 = max(0, bx0), max(0, by0), min(w, bx0 + size), min(h, by0 + size)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = rgb[sy0:sy1, sx0:sx1]
        return np.asarray(Image.fromarray(canvas).resize((self.img_res, self.img_res), Image.LANCZOS))
