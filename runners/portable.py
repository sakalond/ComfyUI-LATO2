"""Pure NumPy / PyTorch stand-ins for LATO.2's compiled dependencies, so the LATO.2 runner also works where they
are not installed (Windows has no builds of o_voxel and only some of the others for a given PyTorch):

- o_voxel.convert.mesh_to_flexible_dual_grid: only the voxel coordinates, which is all LATO.2 reads, from the same
  scan-line pass in the same float32/float64 arithmetic as o_voxel's C++ (src/convert/flexible_dual_grid.cpp,
  intersect_qef). The order of the voxels differs; LATO.2 sorts them.
- torch_scatter.scatter_mean / scatter_sum (LATO.2's PointNet encoder).
- xformers.ops.memory_efficient_attention with fmha.BlockDiagonalMask, on PyTorch's scaled_dot_product_attention,
  for LATO.2's sparse attention when neither flash-attn nor xformers imports.

install() registers a stand-in only for a package that does not import, then picks LATO.2's sparse attention
backend (SPARSE_ATTN_BACKEND) to match. Call it before LATO.2's modules are imported.
"""
import importlib
import os
import sys
import types

import numpy as np

VOXEL_CHUNK = 100_000  # triangles per batch in the voxel pass (bounds the expanded row and column arrays)


def _imports(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# --- o_voxel ------------------------------------------------------------------------------------------------------

def _lerp(a, b, t, va, vb):
    """o_voxel's lerp: val_a where a == b, else (1 - alpha) * val_a + alpha * val_b with alpha = (t - a) / (b - a)."""
    same = a == b
    alpha = (t - a) / np.where(same, 1.0, b - a)
    if va.ndim > alpha.ndim:
        alpha = alpha[:, None]
        same = same[:, None]
    return np.where(same, va, (1.0 - alpha) * va + alpha * vb)


def _cell(value, size, upper):
    """int(value / size) clamped to [0, upper - 1], as the C++ does (truncation toward zero)."""
    return np.clip(np.trunc(value / np.float64(size)), 0, upper - 1).astype(np.int64)


def _expand(counts):
    """For runs of the given lengths: the run of each element and its position within the run."""
    owner = np.repeat(np.arange(len(counts)), counts)
    local = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, local


def _scan_axis(tri, voxel, grid, ax2):
    """Voxel keys marked by scanning along axis ax2 (o_voxel's scan_line_fill), for triangles tri (F, 3, 3) float64
    in grid units of voxel (float32 per axis) from the grid's corner."""
    ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
    p = tri[:, :, [ax0, ax1, ax2]]  # (F, 3, [x, y, z]) in the scan's frame
    order = np.argsort(p[:, :, 1], axis=1, kind='stable')
    p = np.take_along_axis(p, order[:, :, None], axis=1)
    t0, t1, t2 = p[:, 0], p[:, 1], p[:, 2]
    v0, v1, v2 = np.float32(voxel[ax0]), np.float32(voxel[ax1]), np.float32(voxel[ax2])
    g0, g1, g2 = int(grid[ax0]), int(grid[ax1]), int(grid[ax2])
    start, mid, end = _cell(t0[:, 1], v1, g1), _cell(t1[:, 1], v1, g1), _cell(t2[:, 1], v1, g1)
    keys = []
    for row_lo, row_hi, a, b, c in ((start, mid, t0, t1, t2), (mid, end, t2, t1, t0)):
        tri_of_row, k = _expand(np.maximum(row_hi - row_lo, 0))
        if not len(tri_of_row):
            continue
        row = row_lo[tri_of_row] + k
        y = ((row + 1).astype(np.float32) * v1).astype(np.float64)  # int * float in float, as in the C++
        a_, b_, c_ = a[tri_of_row], b[tri_of_row], c[tri_of_row]
        t3 = _lerp(a_[:, 1], b_[:, 1], y, a_[:, [0, 2]], b_[:, [0, 2]])
        t4 = _lerp(a_[:, 1], c_[:, 1], y, a_[:, [0, 2]], c_[:, [0, 2]])
        swap = t3[:, 0] > t4[:, 0]
        t3[swap], t4[swap] = t4[swap].copy(), t3[swap].copy()
        line_lo, line_hi = _cell(t3[:, 0], v0, g0), _cell(t4[:, 0], v0, g0)
        row_of_col, k = _expand(np.maximum(line_hi - line_lo, 0))
        if not len(row_of_col):
            continue
        x_idx = line_lo[row_of_col] + k
        x = ((x_idx + 1).astype(np.float32) * v0).astype(np.float64)
        z = _lerp(t3[row_of_col, 0], t4[row_of_col, 0], x, t3[row_of_col, 1], t4[row_of_col, 1])
        z_idx = np.trunc(z / np.float64(v2)).astype(np.int64)
        inside = (z_idx >= 0) & (z_idx < g2)
        x_idx, y_idx, z_idx = x_idx[inside], row[row_of_col][inside], z_idx[inside]
        for dx in (0, 1):
            for dy in (0, 1):
                coord = [None, None, None]
                coord[ax0], coord[ax1], coord[ax2] = x_idx + dx, y_idx + dy, z_idx
                keys.append((coord[0] * (grid[1] + 1) + coord[1]) * (grid[2] + 1) + coord[2])
    return np.unique(np.concatenate(keys)) if keys else np.empty(0, dtype=np.int64)


def flexible_dual_grid_voxels(vertices, faces, grid_size, aabb):
    """(N, 3) int32 coordinates of the voxels o_voxel's mesh_to_flexible_dual_grid returns for a grid of grid_size
    cells over aabb ([[min], [max]]): every voxel around a grid edge that a triangle crosses."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    aabb = np.asarray(aabb, dtype=np.float32)
    grid = np.broadcast_to(np.asarray(grid_size, dtype=np.int64), (3,)).copy()
    voxel = ((aabb[1] - aabb[0]) / grid.astype(np.float32)).astype(np.float32)
    shifted = (vertices - aabb[0]).astype(np.float32)
    keys = []
    for start in range(0, len(faces), VOXEL_CHUNK):
        tri = shifted[faces[start:start + VOXEL_CHUNK]].astype(np.float64)
        for ax2 in range(3):
            keys.append(_scan_axis(tri, voxel, grid, ax2))
    keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, dtype=np.int64)
    z = keys % (grid[2] + 1)
    y = (keys // (grid[2] + 1)) % (grid[1] + 1)
    x = keys // ((grid[2] + 1) * (grid[1] + 1))
    return np.stack([x, y, z], axis=1).astype(np.int32)


def mesh_to_flexible_dual_grid(vertices, faces, voxel_size=None, grid_size=None, aabb=None, **_unused):
    """o_voxel's signature; returns (coords, None, None): LATO.2 reads only the coordinates."""
    import torch
    if grid_size is None or aabb is None:
        raise NotImplementedError("the o_voxel stand-in needs grid_size and aabb (as LATO.2 passes them)")
    as_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    coords = flexible_dual_grid_voxels(as_numpy(vertices), as_numpy(faces), as_numpy(grid_size), as_numpy(aabb))
    return torch.from_numpy(coords), None, None


def _install_o_voxel():
    package = types.ModuleType('o_voxel')
    package.__path__ = []
    convert = types.ModuleType('o_voxel.convert')
    convert.mesh_to_flexible_dual_grid = mesh_to_flexible_dual_grid
    package.convert = convert
    sys.modules['o_voxel'] = package
    sys.modules['o_voxel.convert'] = convert


# --- torch_scatter ------------------------------------------------------------------------------------------------

def _broadcast(index, src, dim):
    """torch_scatter.utils.broadcast."""
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    return index.expand(src.size())


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    dim = dim if dim >= 0 else src.dim() + dim
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        size[dim] = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
        out = src.new_zeros(size)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim = dim if dim >= 0 else src.dim() + dim
    count = scatter_sum(src.new_ones(src.size()), index, dim, None, out.size(dim)).clamp_(min=1)
    if out.is_floating_point():
        return out.div_(count)
    return out.div_(count, rounding_mode='floor')


def _install_torch_scatter():
    module = types.ModuleType('torch_scatter')
    module.scatter_mean = scatter_mean
    module.scatter_sum = module.scatter_add = scatter_sum
    sys.modules['torch_scatter'] = module


# --- xformers.ops on scaled_dot_product_attention ---------------------------------------------------------------

class BlockDiagonalMask:
    """xformers.ops.fmha.BlockDiagonalMask.from_seqlens: tokens attend within their own segment only."""

    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen = [int(n) for n in q_seqlen]
        self.kv_seqlen = [int(n) for n in (kv_seqlen if kv_seqlen is not None else q_seqlen)]

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        return cls(q_seqlen, kv_seqlen)


def _pad(x, lengths):
    """(1, sum(lengths), H, C) -> (S, max(lengths), H, C) padded with zeros, plus each token's flat padded index."""
    import torch
    lengths_t = torch.as_tensor(lengths, device=x.device)
    longest = int(lengths_t.max())
    segment = torch.repeat_interleave(torch.arange(len(lengths), device=x.device), lengths_t)
    offsets = torch.cumsum(lengths_t, 0) - lengths_t
    position = torch.arange(x.shape[1], device=x.device) - offsets[segment]
    flat = segment * longest + position
    padded = x.new_zeros((len(lengths) * longest,) + tuple(x.shape[2:]))
    padded[flat] = x[0]
    return padded.view((len(lengths), longest) + tuple(x.shape[2:])), flat, longest


def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
    """xformers' layout: q, k, v (B, N, H, C) -> (B, N, H, C)."""
    import torch.nn.functional as F
    if attn_bias is None:
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
        return out.transpose(1, 2)
    if not isinstance(attn_bias, BlockDiagonalMask):
        raise NotImplementedError(f"attention bias {type(attn_bias).__name__} is not supported by the stand-in")
    qp, q_flat, _ = _pad(q, attn_bias.q_seqlen)
    kp, _, k_longest = _pad(k, attn_bias.kv_seqlen)
    vp, _, _ = _pad(v, attn_bias.kv_seqlen)
    import torch
    lengths = torch.as_tensor(attn_bias.kv_seqlen, device=q.device)
    keep = torch.arange(k_longest, device=q.device)[None, :] < lengths[:, None]  # (S, Lk) valid keys
    out = F.scaled_dot_product_attention(qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2),
                                         attn_mask=keep[:, None, None, :], scale=scale).transpose(1, 2)
    return out.reshape((-1,) + tuple(out.shape[2:]))[q_flat].unsqueeze(0)


def _install_xformers():
    package = types.ModuleType('xformers')
    package.__path__ = []
    ops = types.ModuleType('xformers.ops')
    fmha = types.ModuleType('xformers.ops.fmha')
    fmha.BlockDiagonalMask = BlockDiagonalMask
    ops.fmha = fmha
    ops.memory_efficient_attention = memory_efficient_attention
    package.ops = ops
    sys.modules['xformers'] = package
    sys.modules['xformers.ops'] = ops
    sys.modules['xformers.ops.fmha'] = fmha


def install(log=print):
    """Register stand-ins for the packages that do not import; returns their names. LATO2_PORTABLE=1 uses every
    stand-in even where the real package is installed (to compare the two)."""
    force = os.environ.get('LATO2_PORTABLE') == '1'
    used = []
    if force or not _imports('o_voxel.convert'):
        _install_o_voxel()
        used.append('o_voxel')
    if force or not _imports('torch_scatter'):
        _install_torch_scatter()
        used.append('torch_scatter')
    if force:
        _install_xformers()
        used.append('xformers')
        os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
    elif 'SPARSE_ATTN_BACKEND' not in os.environ:
        if _imports('flash_attn'):
            os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
        else:
            if not _imports('xformers.ops'):
                _install_xformers()
                used.append('xformers')
            os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
    if used:
        log(f"[lato2] portable stand-ins for {', '.join(used)}; sparse attention: {os.environ['SPARSE_ATTN_BACKEND']}")
    return used
