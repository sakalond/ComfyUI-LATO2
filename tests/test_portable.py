"""The portable stand-ins (runners/portable.py) against references: the vectorized voxel pass against a loop-for-loop
transcription of o_voxel's C++ scan-line pass, scatter_mean against a loop, the attention stand-in against attention
per segment. (On a GPU machine with o_voxel and flash-attn installed, the voxel sets were also equal to o_voxel's own
at grid sizes 64 and 1024 on five TRELLIS.2 meshes.)"""
import numpy as np
import pytest

import portable

trimesh = pytest.importorskip("trimesh")


def literal_voxels(vertices, faces, grid, aabb):
    """intersect_qef's voxel marking, transcribed loop for loop (float32 voxel size and products, double lerps)."""
    voxel = ((aabb[1] - aabb[0]) / np.float32(grid)).astype(np.float32)
    shifted = (np.asarray(vertices, np.float32) - aabb[0]).astype(np.float32)
    out = set()

    def lerp(a, b, t, va, vb):
        if a == b:
            return va
        alpha = (t - a) / (b - a)
        return (1 - alpha) * va + alpha * vb

    def clamp(v, lo, hi):
        return min(max(v, lo), hi)

    for f in faces:
        tri = [shifted[i].astype(np.float64) for i in f]
        for ax2 in range(3):
            ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
            t = sorted([np.array([p[ax0], p[ax1], p[ax2]]) for p in tri], key=lambda p: p[1])
            start = clamp(int(t[0][1] / np.float64(voxel[ax1])), 0, grid - 1)
            mid = clamp(int(t[1][1] / np.float64(voxel[ax1])), 0, grid - 1)
            end = clamp(int(t[2][1] / np.float64(voxel[ax1])), 0, grid - 1)
            for r0, r1, t0, t1, t2 in ((start, mid, t[0], t[1], t[2]), (mid, end, t[2], t[1], t[0])):
                for y_idx in range(r0, r1):
                    y = float(np.float32(y_idx + 1) * voxel[ax1])
                    t3 = lerp(t0[1], t1[1], y, np.array([t0[0], t0[2]]), np.array([t1[0], t1[2]]))
                    t4 = lerp(t0[1], t2[1], y, np.array([t0[0], t0[2]]), np.array([t2[0], t2[2]]))
                    if t3[0] > t4[0]:
                        t3, t4 = t4, t3
                    ls = clamp(int(t3[0] / np.float64(voxel[ax0])), 0, grid - 1)
                    le = clamp(int(t4[0] / np.float64(voxel[ax0])), 0, grid - 1)
                    for x_idx in range(ls, le):
                        x = float(np.float32(x_idx + 1) * voxel[ax0])
                        z = lerp(t3[0], t4[0], x, t3[1], t4[1])
                        z_idx = int(z / np.float64(voxel[ax2]))
                        if 0 <= z_idx < grid:
                            for dx in (0, 1):
                                for dy in (0, 1):
                                    c = [0, 0, 0]
                                    c[ax0], c[ax1], c[ax2] = x_idx + dx, y_idx + dy, z_idx
                                    out.add(tuple(c))
    return out


@pytest.mark.parametrize("shape", ["sphere", "box", "torus"])
@pytest.mark.parametrize("grid", [16, 64])
def test_voxel_pass_matches_the_cpp_loop(shape, grid):
    mesh = {"sphere": lambda: trimesh.creation.icosphere(3), "box": lambda: trimesh.creation.box((0.9, 0.3, 0.6)),
            "torus": lambda: trimesh.creation.torus(0.3, 0.1)}[shape]()
    v = np.asarray(mesh.vertices)
    v = (v - v.min(0)) / (v.max(0) - v.min(0)).max() - 0.5
    v = v + np.random.default_rng(0).normal(0, 1e-3, v.shape)
    v = np.clip(v, -0.5, 0.5).astype(np.float32) * np.float32(0.99999)
    f = np.asarray(mesh.faces)
    aabb = np.array([[-0.5] * 3, [0.5] * 3], np.float32)
    fast = {tuple(c) for c in portable.flexible_dual_grid_voxels(v, f, grid, aabb).tolist()}
    assert fast == literal_voxels(v, f, grid, aabb)


def test_scatter_mean_matches_a_loop():
    torch = pytest.importorskip("torch")
    src = torch.randn(2, 5, 40)
    index = torch.randint(0, 7, (2, 1, 40)).expand(2, 5, 40)
    out = portable.scatter_mean(src, index, out=src.new_zeros(2, 5, 7))
    ref = torch.zeros(2, 5, 7)
    for b in range(2):
        for c in range(5):
            for j in range(7):
                sel = src[b, c][index[b, c] == j]
                ref[b, c, j] = sel.mean() if len(sel) else 0
    assert torch.allclose(out, ref, atol=1e-5)


def test_block_diagonal_attention_matches_per_segment_attention():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    torch.manual_seed(0)
    lengths = [5, 11, 3]
    q, k, v = (torch.randn(1, sum(lengths), 4, 8) for _ in range(3))
    out = portable.memory_efficient_attention(q, k, v, portable.BlockDiagonalMask.from_seqlens(lengths))
    pieces, start = [], 0
    for n in lengths:
        s = slice(start, start + n)
        pieces.append(F.scaled_dot_product_attention(q[:, s].transpose(1, 2), k[:, s].transpose(1, 2),
                                                     v[:, s].transpose(1, 2)).transpose(1, 2))
        start += n
    assert torch.allclose(out, torch.cat(pieces, dim=1), atol=1e-5)
    q2 = torch.randn(2, 6, 4, 8)
    plain = portable.memory_efficient_attention(q2, q2, q2)
    ref = F.scaled_dot_product_attention(q2.transpose(1, 2), q2.transpose(1, 2), q2.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(plain, ref, atol=1e-6)
