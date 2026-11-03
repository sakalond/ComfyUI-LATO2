"""One clean surface from LATO.2's edge predictions.

LATO.2 predicts which vertex pairs are joined and takes every triangle of that edge graph as a face. Extra edges
stack overlapping faces (both diagonals of a quad give four, half the edges end up with three or more faces) and
missing edges leave holes. This builds the faces again from the predicted edge logits:

1. Candidate triangles are the triangles of a looser edge graph (logit above CANDIDATE_LOGIT, so near misses can
   still close a gap), scored by how likely all three edges are.
2. Best first, a triangle joins while each of its edges has fewer than two faces, it does not fold onto the face
   already on a shared edge, it can be wound consistently with the faces it touches (no one-sided twists), and its
   corners do not overlap corners already at its vertices (in each vertex's tangent plane), so the result is
   edge-manifold without stacked faces.
3. Boundary cycles left over are filled by a minimum-weight triangulation (triangle area, how unlikely each new
   edge is and how sharply the patch bends against its neighbours), never reusing an edge the surface already has
   and never folding onto a neighbour, as in step 2. Triangles left without any neighbour are dropped.
4. Winding: a minimum cut weighs, per face, which side touches empty space connected to the outside (a voxel flood
   fill of the now closed surface) against keeping neighbours wound the same way, which counts most where they lie
   flat, so neighbours mostly disagree only where the surface folds back on itself or crosses itself (thin parts);
   the source surface's winding breaks ties.

    faces, info = rebuild(vertices, pairs, logits, source=(source_vertices, source_faces))
"""
import numpy as np

from exterior_fill import _dilate, _flood_exterior, _grid, _surface_occupancy

CANDIDATE_LOGIT = -3.0
FOLD_DEGREES = 30.0  # a triangle closer than this to its neighbour across an edge would overlap it
MAX_HOLE = 200  # boundary cycles longer than this stay open (a missing region, not a gap)
NOTCH_PASSES = 3  # passes that let leftover triangles close notches in holes without the corner test
NOTCH_MIN_SCORE = 3 * -np.log(2.0)  # ... if their edges are as likely as not on average (all three logits at 0)
FILL_ROUNDS = 3  # a hole that cannot be filled is widened by its border faces and tried again, this many times
FILL_BEND = 1.0  # fill cost of an edge folded flat, in typical triangle areas (falls off quadratically to 0 when flat)


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


class EdgeLogits:
    """Logit lookup for vertex pairs; pairs that were not kept get the `unknown` logit."""

    def __init__(self, n, pairs, logits, unknown):
        self.n = n
        lo, hi = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
        keys = lo.astype(np.int64) * n + hi
        order = np.argsort(keys)
        self.keys = keys[order]
        self.values = np.asarray(logits, dtype=np.float64)[order]
        self.unknown = unknown

    def __call__(self, a, b):
        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
        keys = np.minimum(a, b) * self.n + np.maximum(a, b)
        pos = np.clip(np.searchsorted(self.keys, keys), 0, max(len(self.keys) - 1, 0))
        found = (self.keys[pos] == keys) if len(self.keys) else np.zeros(keys.shape, dtype=bool)
        return np.where(found, self.values[pos] if len(self.keys) else 0.0, self.unknown)


def candidate_triangles(n, pairs, logits, threshold=CANDIDATE_LOGIT):
    """Every triangle of the graph of pairs with a logit above threshold (u < v < w)."""
    keep = logits > threshold
    u = np.minimum(pairs[keep, 0], pairs[keep, 1])
    v = np.maximum(pairs[keep, 0], pairs[keep, 1])
    neighbours = [set() for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        neighbours[a].add(b)
    out = []
    for a, b in zip(u.tolist(), v.tolist()):
        for c in neighbours[a] & neighbours[b]:
            out.append((a, b, c))
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


def _wing(vertices, a, b, c):
    """Unit vector from edge ab towards c, perpendicular to ab."""
    axis = vertices[b] - vertices[a]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    w = vertices[c] - vertices[a]
    w = w - axis * np.dot(w, axis)
    return w / max(np.linalg.norm(w), 1e-12)


def _fold_cos(a, b, c, d):
    """Cosine of the angle between triangles abc and abd at their edge ab (points, broadcast over rows): -1 when
    they lie flat, 1 when one is folded onto the other."""
    axis = b - a
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    wc = c - a
    wc = wc - axis * np.sum(wc * axis, axis=-1, keepdims=True)
    wd = d - a
    wd = wd - axis * np.sum(wd * axis, axis=-1, keepdims=True)
    return np.sum(wc * wd, axis=-1) / np.maximum(np.linalg.norm(wc, axis=-1) * np.linalg.norm(wd, axis=-1), 1e-20)


def _principal_axes(normals, owners, weights, n):
    """Per vertex, the main axis of the unit normals around it (sign-free: the top eigenvector of sum w n n^T)."""
    tensor = np.zeros((n, 3, 3))
    np.add.at(tensor, owners, weights[:, None, None] * normals[:, :, None] * normals[:, None, :])
    values, vectors = np.linalg.eigh(tensor)
    return vectors[:, :, 2], values[:, 2] > 0


def tangent_normals(vertices, triangles, weights, source=None, radius=None):
    """A tangent-plane normal per vertex: the main normal axis of the source surface within radius when the source
    mesh is given (vertices, faces), else that of the candidate triangles at the vertex weighted by score."""
    n = len(vertices)
    if source is not None and len(source[1]):
        from scipy.spatial import cKDTree
        sv, sf = np.asarray(source[0], dtype=np.float64), np.asarray(source[1], dtype=np.int64)
        p = sv[sf]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = np.linalg.norm(cross, axis=1)
        good = area > 1e-20
        found = cKDTree(p[good].mean(axis=1)).query_ball_point(vertices, r=radius)
        owners = np.repeat(np.arange(n), [len(f) for f in found])
        near = np.concatenate([np.asarray(f, dtype=np.int64) for f in found]) if len(owners) else np.empty(0, int)
        axis, ok = _principal_axes(cross[good][near] / area[good][near, None], owners, area[good][near], n)
        if ok.all():
            return axis
        return np.where(ok[:, None], axis, tangent_normals(vertices, triangles, weights))
    p = vertices[triangles]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    unit = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-20)
    axis, ok = _principal_axes(np.repeat(unit, 3, axis=0), triangles.reshape(-1), np.repeat(weights, 3), n)
    return np.where(ok[:, None], axis, np.array([0.0, 0.0, 1.0]))


WEDGE_TOLERANCE = np.radians(4.0)  # corners at a vertex may overlap this much in its tangent plane
MIN_WEDGE = np.radians(8.0)  # corners narrower than this in the tangent plane (seen edge-on) are not checked


class _Fans:
    """The corners of the accepted triangles at each vertex, as angle intervals in the vertex's tangent plane."""

    def __init__(self, vertices, normals):
        self.vertices = vertices
        helper = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        self.u = np.cross(normals, helper)
        self.u /= np.linalg.norm(self.u, axis=1, keepdims=True)
        self.w = np.cross(normals, self.u)
        self.corners = {}

    def corner(self, v, a, b):
        """(start, width) of the corner at v between rays va and vb, or None when it is too narrow to judge."""
        da = self.vertices[a] - self.vertices[v]
        db = self.vertices[b] - self.vertices[v]
        ta = np.arctan2(da @ self.w[v], da @ self.u[v])
        tb = np.arctan2(db @ self.w[v], db @ self.u[v])
        width = (tb - ta) % (2 * np.pi)
        start = ta
        if width > np.pi:
            start, width = tb, 2 * np.pi - width
        return (start, width) if width >= MIN_WEDGE else None

    def overlaps(self, v, corner):
        start, width = corner
        for other_start, other_width in self.corners.get(v, ()):
            # Arcs [start, start + width] and [other_start, other_start + width'] on the circle.
            ahead = (other_start - start) % (2 * np.pi)
            behind = (start - other_start) % (2 * np.pi)
            overlap = max(0.0, min(width, ahead + other_width) - ahead) + \
                max(0.0, min(other_width, behind + width) - behind)
            if overlap > WEDGE_TOLERANCE:
                return True
        return False

    def add(self, v, corner):
        self.corners.setdefault(v, []).append(corner)


class _Windings:
    """Union-find over the accepted triangles with each one's winding relative to its component's root (1: reversed
    from the stored order), so the accepted faces can always be wound consistently. A triangle that would need both
    windings at once, to agree with two neighbours in one component, closes an odd cycle: the surface would turn
    one-sided there (a twist, or a fold through itself), and it is refused."""

    def __init__(self, triangles):
        self.triangles = triangles
        self.parent = np.arange(len(triangles))
        self.parity = np.zeros(len(triangles), dtype=np.int8)

    def find(self, x):
        """(root, winding of x relative to the root)."""
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        acc = 0
        for y in reversed(path):  # nearest the root first: accumulate outward and point each at the root
            acc ^= int(self.parity[y])
            self.parity[y] = acc
            self.parent[y] = x
        return x, (int(self.parity[path[0]]) if path else 0)

    def _runs(self, t, u, v):
        x, y, z = (int(i) for i in self.triangles[t])
        return (u, v) in ((x, y), (y, z), (z, x))

    def place(self, t, shared):
        """The winding t needs relative to each component it touches ({root: parity}; shared: (u, v, face) per
        shared edge), or None when it cannot agree with all of them."""
        need = {}
        for u, v, other in shared:
            root, parity = self.find(other)
            # Wound consistently, the two faces run their shared edge in opposite directions.
            want = int(self._runs(t, u, v)) ^ int(self._runs(other, u, v)) ^ parity ^ 1
            if need.setdefault(root, want) != want:
                return None
        return need

    def join(self, t, need):
        """Make t part of the components it touches (merging them), with the windings place() found."""
        if not need:
            return
        (root0, parity0), *others = need.items()
        self.parent[t], self.parity[t] = root0, parity0
        for root, parity in others:
            self.parent[root], self.parity[root] = root0, parity ^ parity0


def select_manifold(vertices, triangles, scores, normals=None, fold_degrees=FOLD_DEGREES, notch_passes=NOTCH_PASSES,
                    consistent=True):
    """Best score first, a triangle joins when none of its edges has two faces yet, it does not fold onto the face
    already on a shared edge, (consistent) it can be wound consistently with every face it shares an edge with,
    and (given tangent-plane normals) none of its corners overlaps a corner already at that vertex, which catches
    overlapping faces that share only a vertex. Returns the accepted faces, wound consistently within each
    connected piece.

    The corner test also turns away good triangles where the tangent plane is a poor guess (corners, thin parts),
    leaving holes. Later passes take the remaining likely triangles (scores of at least NOTCH_MIN_SCORE: near misses
    would bridge real gaps) that close a notch in such a hole, two of their edges already on the boundary, without
    the corner test."""
    n = len(vertices)
    edge_faces = {}
    accepted = []
    cos_fold = np.cos(np.radians(fold_degrees))
    fans = _Fans(vertices, normals) if normals is not None else None
    windings = _Windings(triangles) if consistent else None
    order = np.argsort(-scores, kind='stable')
    taken = np.zeros(len(triangles), dtype=bool)
    for t in order:
        if _try_add(vertices, triangles, t, n, edge_faces, cos_fold, fans, windings):
            accepted.append(t)
            taken[t] = True
    for _ in range(notch_passes):
        added = 0
        for t in order:
            if taken[t] or scores[t] < NOTCH_MIN_SCORE:
                continue
            a, b, c = (int(x) for x in triangles[t])
            open_sides = sum(len(edge_faces.get(min(u, v) * n + max(u, v), ())) == 1 for u, v in ((a, b), (b, c), (a, c)))
            if open_sides >= 2 and _try_add(vertices, triangles, t, n, edge_faces, cos_fold, None, windings):
                accepted.append(t)
                taken[t] = True
                added += 1
        if not added:
            break
    accepted = np.asarray(accepted, dtype=np.int64)
    faces = triangles[accepted].copy()
    if windings is not None and len(accepted):
        flip = np.array([windings.find(int(t))[1] for t in accepted], dtype=bool)
        faces[flip] = faces[flip][:, ::-1]
    return faces


def _try_add(vertices, triangles, t, n, edge_faces, cos_fold, fans, windings=None):
    """Add triangle t to the surface (edge_faces, fans, windings) when it passes the edge, fold, winding and corner
    tests."""
    a, b, c = (int(x) for x in triangles[t])
    sides = ((a, b, c), (b, c, a), (a, c, b))
    keys = [min(u, v) * n + max(u, v) for u, v, _ in sides]
    if any(len(edge_faces.get(k, ())) >= 2 for k in keys):
        return False
    for (u, v, w), k in zip(sides, keys):
        for other in edge_faces.get(k, ()):
            o = next(int(x) for x in triangles[other] if x != u and x != v)
            if np.dot(_wing(vertices, u, v, w), _wing(vertices, u, v, o)) > cos_fold:
                return False
    need = None
    if windings is not None:
        need = windings.place(t, [(u, v, other) for (u, v, _), k in zip(sides, keys) for other in edge_faces.get(k, ())])
        if need is None:
            return False
    if fans is not None:
        corners = [(a, fans.corner(a, b, c)), (b, fans.corner(b, c, a)), (c, fans.corner(c, a, b))]
        if any(corner is not None and fans.overlaps(v, corner) for v, corner in corners):
            return False
        for v, corner in corners:
            if corner is not None:
                fans.add(v, corner)
    if windings is not None:
        windings.join(t, need)
    for k in keys:
        edge_faces.setdefault(k, []).append(t)
    return True


def _edge_keys(faces, n):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return edges, edges[:, 0].astype(np.int64) * n + edges[:, 1]


def boundary_cycles(faces, n):
    """Simple cycles of boundary edges (edges with one face). A pinched boundary (a vertex with four boundary
    edges) splits into separate cycles."""
    edges, key = _edge_keys(faces, n)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    adjacent = {}
    for a, b in np.unique(edges[counts[inverse] == 1], axis=0):
        adjacent.setdefault(int(a), set()).add(int(b))
        adjacent.setdefault(int(b), set()).add(int(a))
    cycles = []
    while adjacent:
        start = next(iter(adjacent))
        path = [start]
        position = {start: 0}
        current = start
        while current in adjacent:
            options = adjacent[current]
            nxt = options.pop()
            if not options:
                del adjacent[current]
            adjacent[nxt].discard(current)
            if not adjacent[nxt]:
                del adjacent[nxt]
            if nxt in position:
                cut = position[nxt]
                cycles.append(path[cut:])
                for v in path[cut + 1:]:
                    del position[v]
                path = path[:cut + 1]
            else:
                position[nxt] = len(path)
                path.append(nxt)
            current = nxt
    return [c for c in cycles if len(c) >= 3]


def fill_cycle(vertices, cycle, existing, edge_cost=None, border=None, fold_degrees=FOLD_DEGREES, bend_cost=0.0):
    """Minimum-weight triangulation of one boundary cycle (Liepa's dynamic programme over the polygon): triangle
    area plus an optional cost per new edge. Chords that are already mesh edges are not allowed (the surface would
    turn non-manifold); returns no faces when every triangulation needs one.

    border[i] is the far corner of the surface's face on the cycle edge (i, i + 1) (border[-1]: on (m - 1, 0)), -1
    if unknown. With it, no patch triangle may fold closer than fold_degrees onto that face or onto its neighbour in
    the patch (None turns the test off), and each such edge costs bend_cost * ((1 + cos) / 2) ** 2, 0 when flat. As
    in Liepa's method, a triangle is checked against the best triangulations of the sub-polygons it closes."""
    n = len(vertices)
    m = len(cycle)
    idx = np.asarray(cycle, dtype=np.int64)
    p = vertices[idx]
    keys = np.minimum(idx[:, None], idx[None, :]) * n + np.maximum(idx[:, None], idx[None, :])
    chord = np.zeros((m, m)) if edge_cost is None else edge_cost(idx[:, None], idx[None, :])
    ring = np.zeros((m, m), dtype=bool)
    ring[np.arange(m - 1), np.arange(1, m)] = True
    ring[0, m - 1] = True
    ring |= ring.T
    chord = np.where(ring, 0.0, np.where(np.isin(keys, existing), np.inf, chord))
    folds = border is not None and (fold_degrees is not None or bend_cost > 0)
    if folds:
        border = np.asarray(border, dtype=np.int64)
        known = border >= 0
        far = np.where(known[:, None], vertices[np.maximum(border, 0)], np.nan)
        cos_fold = np.cos(np.radians(fold_degrees)) if fold_degrees is not None else np.inf

    def bend(cos):
        """Cost of each neighbour angle (inf past the fold limit; nan, an unknown neighbour, costs nothing)."""
        with np.errstate(invalid='ignore'):
            cost = bend_cost * ((1.0 + cos) / 2.0) ** 2
            return np.where(np.isnan(cos), 0.0, np.where(cos > cos_fold, np.inf, cost))

    weight = np.full((m, m), np.inf)
    choice = np.zeros((m, m), dtype=np.int64)
    weight[np.arange(m - 1), np.arange(1, m)] = 0.0
    for span in range(2, m):
        for i in range(0, m - span):
            j = i + span
            k = np.arange(i + 1, j)
            area = 0.5 * np.linalg.norm(np.cross(p[k] - p[i], p[j] - p[i]), axis=1)
            total = weight[i, k] + weight[k, j] + area + chord[i, k] + chord[k, j]
            if folds:
                # Across (i, k): the surface's face when k follows i, else the triangle closing sub-polygon (i, k).
                left = np.where((k == i + 1)[:, None], far[i], p[choice[i, k]])
                right = np.where((j == k + 1)[:, None], far[k], p[choice[k, j]])
                total = total + bend(_fold_cos(p[i], p[k], p[j], left)) + bend(_fold_cos(p[k], p[j], p[i], right))
                if i == 0 and j == m - 1:
                    total = total + bend(_fold_cos(p[0], p[j], p[k], np.broadcast_to(far[m - 1], p[k].shape)))
            best = int(np.argmin(total))
            weight[i, j] = total[best]
            choice[i, j] = k[best]
    if not np.isfinite(weight[0, m - 1]):
        return np.empty((0, 3), dtype=np.int64)
    out = []
    stack = [(0, m - 1)]
    while stack:
        i, j = stack.pop()
        if j - i >= 2:
            k = choice[i, j]
            out.append((idx[i], idx[k], idx[j]))
            stack.extend(((i, k), (k, j)))
    return np.asarray(out, dtype=np.int64)


def _border_corners(faces, n):
    """For every boundary edge (one face), the far corner of that face: {edge key: vertex}."""
    edges, key = _edge_keys(faces, n)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    corner = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])  # opposite (0,1), (1,2), (2,0)
    boundary = counts[inverse] == 1
    return dict(zip(key[boundary].tolist(), corner[boundary].tolist()))


def fill_holes(vertices, faces, max_hole=MAX_HOLE, edge_cost=None, rounds=FILL_ROUNDS, fold_degrees=FOLD_DEGREES,
               bend_cost=0.0):
    """Fill every boundary cycle up to max_hole edges (fill_cycle, folds checked against the faces around the hole).
    A cycle with no triangulation inside the fold limit takes the least bent one past it (info['relaxed']: widening
    such holes threw away good faces and the bigger patches bridged gaps); one that cannot be filled without reusing
    an edge is widened by the faces along it and filled again in the next round. The fill faces end the returned
    array."""
    n = len(vertices)
    is_fill = np.zeros(len(faces), dtype=bool)
    info = {'holes': 0, 'filled': 0, 'widened': 0, 'relaxed': 0}
    for round_index in range(rounds):
        cycles = boundary_cycles(faces, n)
        existing = np.unique(_edge_keys(faces, n)[1])
        face_keys = set(map(tuple, np.sort(faces, axis=1).tolist()))
        corners = _border_corners(faces, n)
        # A lone triangle's own outline is not a hole.
        cycles = [c for c in cycles if not (len(c) == 3 and tuple(sorted(c)) in face_keys)]
        if not round_index:
            info['holes'] = len(cycles)
        patches, stuck = [], []
        for cycle in sorted(cycles, key=len):
            if len(cycle) > max_hole:
                continue
            border = [corners.get(min(u, v) * n + max(u, v), -1) for u, v in zip(cycle, cycle[1:] + cycle[:1])]
            patch = fill_cycle(vertices, cycle, existing, edge_cost, border, fold_degrees, bend_cost)
            if not len(patch) and fold_degrees is not None:
                patch = fill_cycle(vertices, cycle, existing, edge_cost, border, None, bend_cost)
                info['relaxed'] += bool(len(patch))
            if len(patch):
                patches.append(patch)
                existing = np.union1d(existing, _edge_keys(patch, n)[1])
            else:
                stuck.append(cycle)
        info['filled'] += len(patches)
        if patches:
            faces = np.concatenate([faces] + patches)
            is_fill = np.concatenate([is_fill, np.ones(sum(len(p) for p in patches), dtype=bool)])
        if not stuck or round_index == rounds - 1:
            break
        ring = set()
        for cycle in stuck:
            for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                ring.add(min(u, v) * n + max(u, v))
        _, keys = _edge_keys(faces, n)
        along = np.isin(keys, list(ring)).reshape(3, -1).any(axis=0)
        info['widened'] += int(along.sum())
        faces, is_fill = faces[~along], is_fill[~along]
    order = np.argsort(is_fill, kind='stable')
    info['open'] = len([c for c in boundary_cycles(faces, n) if len(c) > 3])
    info['fill_faces'] = int(is_fill.sum())
    return faces[order], info


SURE = 0.5  # an outside test at least this one-sided counts as decided (info['unsure_faces'] counts the rest)
CONSISTENCY = 0.5  # cost of flat neighbours wound against each other, per unit of their mean area (higher turned
# recesses such as wheel hubs inward: across thin parts and self-crossing sheets the outside really flips)
FLAT_POWER = 2  # how fast that cost falls off as the two faces bend away from flat
PRIOR_WEIGHT = 0.1  # weight of the source winding next to the outside test
ORIENT_RESOLUTION = 192  # voxels along the longest side for the outside test
ORIENT_CLOSE = 2  # voxels the surface is thickened by for it, sealing holes the filling left


def _face_samples(vertices, faces):
    """Unit normals and four points per face (the centroid and three points towards the corners)."""
    p = vertices[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    center = p.mean(axis=1)
    return normal, np.concatenate([center[:, None], (2 * center[:, None] + p) / 3], axis=1)  # (F, 4, 3)


def _exterior_votes(vertices, faces, resolution=ORIENT_RESOLUTION, close=ORIENT_CLOSE):
    """Per face, which side faces empty space connected to the outside (exterior_fill's flood fill on this mesh): +1
    when the front does at every sample point, -1 when the back does, near 0 where both or neither side does (thin
    parts). The surface is thickened by `close` voxels first, so a hole the filling left open does not flood the
    inside; samples step off the face until they leave that band."""
    origin, pitch, shape = _grid(vertices, resolution)
    occupied = _surface_occupancy(vertices, faces, origin, pitch, shape)
    for _ in range(close):
        occupied = _dilate(occupied)
    exterior = _flood_exterior(occupied)
    normal, points = _face_samples(vertices, faces)
    limit = np.array(shape) - 1

    def outside(direction):
        result = np.zeros(points.shape[:2], dtype=np.int8)  # 1 outside, -1 enclosed, 0 not found yet
        for step in np.arange(1.0, close + 4.0):
            index = np.floor((points + direction[:, None] * (step * pitch) - origin) / pitch).astype(np.int64)
            np.clip(index, 0, limit, out=index)
            cell = (index[..., 0], index[..., 1], index[..., 2])
            free = ~occupied[cell] & (result == 0)
            result[free] = np.where(exterior[cell][free], 1, -1)
        return result == 1

    front = outside(normal)
    back = outside(-normal)
    return (front.astype(np.float64) - back).mean(axis=1)


def _source_votes(vertices, faces, source):
    """Per face, how well its normal agrees with the outward source surface nearby: in [-1, 1], negative when the
    face is wound inward. Four points per face; at each, of the nearest source faces the one whose normal is most
    parallel (either way) decides, so the far side of a thin part does not."""
    from scipy.spatial import cKDTree
    sv, sf = np.asarray(source[0], dtype=np.float64), np.asarray(source[1], dtype=np.int64)
    sp = sv[sf]
    s_normal = np.cross(sp[:, 1] - sp[:, 0], sp[:, 2] - sp[:, 0])
    s_normal /= np.maximum(np.linalg.norm(s_normal, axis=1, keepdims=True), 1e-20)
    tree = cKDTree(sp.mean(axis=1))
    normal, points = _face_samples(vertices, faces)
    _, near = tree.query(points.reshape(-1, 3), k=8)
    dots = np.einsum('qkc,qc->qk', s_normal[near], np.repeat(normal, 4, axis=0))
    best = dots[np.arange(len(dots)), np.argmax(np.abs(dots), axis=1)]
    return best.reshape(-1, 4).mean(axis=1)


def _neighbour_edges(vertices, faces):
    """Every edge with exactly two faces: (face i, face j), whether they run it the same way (wound against each
    other) and how flat they lie there (1 flat, 0 folded onto each other)."""
    n = len(vertices)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    corner = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    key = np.minimum(directed[:, 0], directed[:, 1]).astype(np.int64) * n + np.maximum(directed[:, 0], directed[:, 1])
    order = np.argsort(key, kind='stable')
    key, directed, owner, corner = key[order], directed[order], owner[order], corner[order]
    _, start, counts = np.unique(key, return_index=True, return_counts=True)
    first = start[counts == 2]
    second = first + 1
    same = directed[first, 0] == directed[second, 0]
    a, b = vertices[directed[first, 0]], vertices[directed[first, 1]]
    flat = (1.0 - _fold_cos(a, b, vertices[corner[first]], vertices[corner[second]])) / 2.0
    return owner[first], owner[second], same, flat


def _gauge(count, i, j, same, order):
    """Per face, a flip that winds the faces across the edges of a spanning forest (edges taken in `order`, so the
    flattest first) the same way: union-find with the parity of each face relative to its root."""
    parent = np.arange(count)
    parity = np.zeros(count, dtype=np.int8)

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root, acc = x, 0
        for y in reversed(path):  # nearest the root first: accumulate parity outward and point each at the root
            acc ^= parity[y]
            parity[y] = acc
            parent[y] = root
        return root

    for e in order:
        a, b = int(i[e]), int(j[e])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            parity[rb] = parity[a] ^ parity[b] ^ int(same[e])
    for x in range(count):
        find(x)
    return parity.astype(bool)


def _orient_by_cut(vertices, faces, vote, prior):
    """Flip faces to minimise, per face, its area times how much the outside test (vote, in [-1, 1]; prior breaks
    ties) disagrees with its side, plus, per edge whose two faces end up wound against each other, CONSISTENCY times
    their mean area times flatness ** FLAT_POWER: neighbours that lie flat stay consistent, folds may disagree so both
    faces still point out.

    Consistency is not a submodular term in general, so the faces are first flipped to agree across a maximum
    spanning forest of the flattest edges (_gauge). In that frame most edges ask for equal labels (a graph cut
    solves it exactly); the few still wound against each other close odd cycles (the surface does not have two
    sides there) and are left out of the cut."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    count = len(faces)
    i, j, same, flat = _neighbour_edges(vertices, faces)
    p = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    weight = CONSISTENCY * 0.5 * (area[i] + area[j]) * flat ** FLAT_POWER
    gauge = _gauge(count, i, j, same, np.argsort(-weight, kind='stable'))
    agree = (same ^ gauge[i] ^ gauge[j]) == 0
    data = area * np.clip(vote + PRIOR_WEIGHT * prior, -1.0, 1.0) * np.where(gauge, -1.0, 1.0)
    source, sink = count, count + 1
    rows = np.concatenate([np.full(count, source), np.arange(count), i[agree], j[agree]])
    cols = np.concatenate([np.arange(count), np.full(count, sink), j[agree], i[agree]])
    caps = np.concatenate([np.maximum(data, 0.0), np.maximum(-data, 0.0), weight[agree], weight[agree]])
    scale = 1e9 / max(caps.sum(), 1e-30)
    graph = csr_matrix((np.round(caps * scale).astype(np.int32), (rows, cols)), shape=(count + 2, count + 2))
    flow = maximum_flow(graph, source, sink).flow
    residual = (graph - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    kept = np.zeros(count + 2, dtype=bool)  # the source side keeps the gauge winding
    kept[breadth_first_order(residual, source, directed=True, return_predecessors=False)] = True
    flip = gauge ^ ~kept[:count]
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return faces


def orient(vertices, faces, source=None):
    """Outward winding that stays consistent where the surface is flat (_orient_by_cut): each face's front should
    face the outside of this (closed) surface (_exterior_votes); the source surface nearby (TRELLIS.2 meshes are
    mostly, not always, wound outward) breaks ties. Returns faces and the number of faces the outside test could
    not decide."""
    vote = _exterior_votes(vertices, faces)
    prior = _source_votes(vertices, faces, source) if source is not None and len(source[1]) else np.zeros(len(faces))
    return _orient_by_cut(vertices, faces, vote, prior), int((np.abs(vote) < SURE).sum())


def rebuild(vertices, pairs, logits, source=None, threshold=CANDIDATE_LOGIT, max_hole=MAX_HOLE, fill=True):
    """Faces for vertices from LATO.2's edge logits (pairs: (E, 2) vertex indices, logits: (E,), every pair with a
    logit above threshold). source: the mesh LATO.2 re-meshed, (vertices, faces) in the same frame, for tangent
    planes and to break winding ties. Returns (faces, info)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = len(vertices)
    lookup = EdgeLogits(n, pairs, logits, unknown=threshold - 2.0)
    triangles = candidate_triangles(n, pairs, logits, threshold)
    info = {'candidates': len(triangles)}
    if not len(triangles):
        return np.empty((0, 3), dtype=np.int64), info
    scores = (_log_sigmoid(lookup(triangles[:, 0], triangles[:, 1])) + _log_sigmoid(lookup(triangles[:, 1], triangles[:, 2]))
              + _log_sigmoid(lookup(triangles[:, 0], triangles[:, 2])))
    p = vertices[triangles]
    edge = float(np.median(np.linalg.norm(p[:, 1] - p[:, 0], axis=1)))
    normals = tangent_normals(vertices, triangles, np.exp(scores), source, radius=0.5 * edge)
    faces = select_manifold(vertices, triangles, scores, normals)
    info['selected'] = len(faces)
    if fill and len(faces):
        p = vertices[faces]
        typical = float(np.median(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)))
        faces, fill_info = fill_holes(vertices, faces, max_hole,
                                      edge_cost=lambda a, b: typical * np.logaddexp(0.0, -lookup(a, b)),
                                      bend_cost=FILL_BEND * typical)
        info.update(fill_info)
        # A triangle that shares no edge with any other after filling floats on its own (an overlap that slipped
        # through, or a scrap): drop it.
        _, key = _edge_keys(faces, n)
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        lone = (counts[inverse].reshape(3, -1) == 1).all(axis=0)
        faces = faces[~lone]
        info['lone_dropped'] = int(lone.sum())
    faces, info['unsure_faces'] = orient(vertices, faces, source)
    return faces, info
