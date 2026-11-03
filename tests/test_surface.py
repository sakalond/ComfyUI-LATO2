"""The surface rebuild (runners/lato2_surface.py) on synthetic edge predictions: a known mesh's edges with LATO.2-like
noise (both diagonals of quads, dropped edges, long spurious edges) must come back as one closed, consistently and
outward-wound surface."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
from lato2_surface import rebuild  # noqa: E402


def noisy_edges(mesh, drop=0.0, diagonals=True, spurious=0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(mesh.vertices)
    edges = np.asarray(mesh.edges_unique)
    logits = rng.uniform(1.0, 6.0, len(edges))
    dropped = rng.random(len(edges)) < drop
    logits[dropped] = rng.uniform(-2.5, -0.2, dropped.sum())
    pairs, values = [edges], [logits]
    if diagonals:
        f = np.asarray(mesh.faces)
        adjacency = mesh.face_adjacency
        a = np.stack([np.setdiff1d(f[i], f[j])[0] for i, j in adjacency])
        b = np.stack([np.setdiff1d(f[j], f[i])[0] for i, j in adjacency])
        pick = rng.random(len(adjacency)) < 0.3
        pairs.append(np.stack([a[pick], b[pick]], axis=1))
        values.append(rng.uniform(0.5, 4.0, pick.sum()))
    if spurious:
        pairs.append(rng.integers(0, len(v), (spurious, 2)))
        values.append(rng.uniform(0.0, 2.0, spurious))
    pairs, values = np.concatenate(pairs), np.concatenate(values)
    keep = pairs[:, 0] != pairs[:, 1]
    return v, pairs[keep], values[keep]


CASES = {
    "sphere": (lambda: trimesh.creation.icosphere(subdivisions=3), dict(diagonals=False)),
    "sphere, both diagonals": (lambda: trimesh.creation.icosphere(subdivisions=3), dict()),
    "sphere, dropped edges": (lambda: trimesh.creation.icosphere(subdivisions=3), dict(drop=0.08)),
    "sphere, dropped and spurious edges": (lambda: trimesh.creation.icosphere(subdivisions=3),
                                           dict(drop=0.08, spurious=300)),
    "box, dropped edges": (lambda: trimesh.creation.box(extents=(1, 0.4, 0.6)).subdivide().subdivide(),
                           dict(drop=0.08)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_rebuild_closes_and_winds_outward(name):
    make, noise = CASES[name]
    mesh = make()
    vertices, pairs, logits = noisy_edges(mesh, **noise)
    faces, info = rebuild(vertices, pairs, logits)
    out = trimesh.Trimesh(vertices, faces, process=False)
    assert info["open"] == 0
    assert out.is_watertight
    assert out.is_winding_consistent
    assert out.volume > 0  # outward
    assert abs(out.volume - mesh.volume) < 0.03 * mesh.volume
