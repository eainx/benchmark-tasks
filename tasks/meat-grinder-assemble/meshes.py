"""Small triangle-mesh builders for the visual surfaces (NumPy only).

Every builder returns (points (N, 3), triangles (M, 3)) in metres, wound outward. ``merge`` joins
meshes; ``place`` rotates and translates one. Lathe profiles turn about the local z axis.
"""
from __future__ import annotations

import math

import numpy as np


def merge(*meshes):
    pts, tris, off = [], [], 0
    for p, t in meshes:
        pts.append(np.asarray(p, float))
        tris.append(np.asarray(t, int) + off)
        off += len(p)
    return np.vstack(pts), np.vstack(tris)


def rot(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def place(mesh, R=None, t=(0.0, 0.0, 0.0)):
    p, tr = mesh
    p = np.asarray(p, float)
    if R is not None:
        p = p @ np.asarray(R).T
    return p + np.asarray(t, float), tr


Z_TO_X = rot(pitch=math.pi / 2)          # a lathe made about z, turned to run along +x


def lathe(profile, n=64, a0=0.0, a1=2 * math.pi, radial=None):
    """Surface of revolution of ``profile`` [(r, z), ...] about z. ``radial(a, k)`` may scale r by angle
    (knurls). Ends with r == 0 close to a point."""
    closed = abs(a1 - a0 - 2 * math.pi) < 1e-9
    m = n if closed else n + 1
    angs = [a0 + (a1 - a0) * i / n for i in range(m)]
    pts = []
    for a in angs:
        for k, (r, z) in enumerate(profile):
            rr = r * (radial(a, k) if radial else 1.0)
            pts.append((rr * math.cos(a), rr * math.sin(a), z))
    L = len(profile)
    tris = []
    for i in range(n):
        i2 = (i + 1) % m if closed else i + 1
        for k in range(L - 1):
            a, b, c, d = i * L + k, i2 * L + k, i2 * L + k + 1, i * L + k + 1
            tris += [(a, b, c), (a, c, d)]
    return np.array(pts), np.array(tris)


def helix_flight(r_in, r_out, pitch, length, thick, per_turn=48, hand=1):
    """A continuous helical flight (a screw thread) along +z from 0 to ``length``: a ribbon from
    ``r_in`` to ``r_out`` with an axial thickness ``thick`` (thinner at the rim)."""
    turns = length / pitch
    n = max(8, int(turns * per_turn))
    pts, tris = [], []
    for i in range(n + 1):
        a = hand * 2 * math.pi * turns * i / n
        z = length * i / n
        c, s = math.cos(a), math.sin(a)
        for r, dz in ((r_in, -thick / 2), (r_out, -thick / 4), (r_out, thick / 4), (r_in, thick / 2)):
            pts.append((r * c, r * s, z + dz))
    for i in range(n):
        for k in range(4):
            a, b = i * 4 + k, i * 4 + (k + 1) % 4
            c, d = (i + 1) * 4 + (k + 1) % 4, (i + 1) * 4 + k
            tris += [(a, d, c), (a, c, b)] if hand > 0 else [(a, b, c), (a, c, d)]
    # end caps
    for i0 in (0, n * 4):
        tris += [(i0, i0 + 1, i0 + 2), (i0, i0 + 2, i0 + 3)]
    return np.array(pts), np.array(tris)


def extrude(poly, z0, z1, top_inset=0.0):
    """A prism from the 2D outline ``poly`` [(x, y), ...] (counter-clockwise, star-shaped about its
    centroid) between z0 and z1; ``top_inset`` shrinks the top face toward the centroid (a bevel)."""
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    top = c + (poly - c) * (1.0 - top_inset)
    n = len(poly)
    pts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in top] + [(c[0], c[1], z0), (c[0], c[1], z1)]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + j), (i, n + j, n + i)]
        tris += [(2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return np.array(pts), np.array(tris)


def box(size, center=(0.0, 0.0, 0.0), bevel=0.0):
    """A box, its top and bottom edges bevelled by ``bevel``."""
    sx, sy, sz = (s / 2 for s in size)
    b = bevel
    ring = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]
    inner = [(-sx + b, -sy + b), (sx - b, -sy + b), (sx - b, sy - b), (-sx + b, sy - b)]
    zs = [(-sz, inner), (-sz + b, ring), (sz - b, ring), (sz, inner)]
    pts = [(x, y, z) for z, r in zs for x, y in r]
    tris = []
    for L in range(3):
        for i in range(4):
            j = (i + 1) % 4
            a, bb, c, d = L * 4 + i, L * 4 + j, (L + 1) * 4 + j, (L + 1) * 4 + i
            tris += [(a, bb, c), (a, c, d)]
    tris += [(0, 2, 1), (0, 3, 2), (12, 13, 14), (12, 14, 15)]
    return np.array(pts) + np.asarray(center, float), np.array(tris)


def lumpy_box(size, seed, n=6, amp=0.12):
    """A rough-cut cube (a chunk of meat): a subdivided box with its corners rounded and its faces
    pushed in and out a little."""
    rng = np.random.default_rng(seed)
    pts, tris = [], []
    sx, sy, sz = (s / 2 for s in size)
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    bumps = rng.normal(0, 1, (4, 3))
    for nrm, u, v in faces:
        base = len(pts)
        nrm, u, v = np.array(nrm, float), np.array(u, float), np.array(v, float)
        for i in range(n + 1):
            for j in range(n + 1):
                q = nrm + u * (2 * i / n - 1) + v * (2 * j / n - 1)
                q = q / np.max(np.abs(q))
                q = q * 0.55 + 0.45 * q / np.linalg.norm(q) * 1.15          # rounded corners
                w = 1 + amp * sum(math.sin(3.1 * np.dot(q, bb) + k) for k, bb in enumerate(bumps)) / 4
                pts.append((q[0] * sx * w, q[1] * sy * w, q[2] * sz * w))
        for i in range(n):
            for j in range(n):
                a, b, c, d = base + i * (n + 1) + j, base + (i + 1) * (n + 1) + j, base + (i + 1) * (n + 1) + j + 1, base + i * (n + 1) + j + 1
                tris += [(a, b, c), (a, c, d)]
    return np.array(pts), np.array(tris)


def cloth(size, folds, nx=40, ny=32, seed=0, droop=0.003):
    """A flat cloth sheet (top surface at z = 0) with soft folds, its edges turned down by ``droop``."""
    rng = np.random.default_rng(seed)
    sx, sy = size
    pts, tris = [], []
    ph = rng.uniform(0, 6.28, 4)
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = -sx / 2 + sx * i / nx, -sy / 2 + sy * j / ny
            z = folds * (0.6 * math.sin(2 * math.pi * x / sx * 1.5 + ph[0]) * math.sin(2 * math.pi * y / sy + ph[1])
                         + 0.4 * math.sin(2 * math.pi * (x + y) / sx * 2.3 + ph[2]))
            z = z * 0.5 + folds * 0.5
            if i in (0, nx) or j in (0, ny):
                z = -droop
            pts.append((x, y, z))
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1
            tris += [(a, b, c), (a, c, d)]
    return np.array(pts), np.array(tris)


def tapered_bar(p0, p1, w0, w1, t, up=(0.0, 0.0, 1.0)):
    """A flat bar from p0 to p1, width w0 tapering to w1 (across ``side``), thickness t."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d /= np.linalg.norm(d)
    side = np.cross(d, up)
    side /= np.linalg.norm(side)
    nrm = np.cross(side, d)
    pts = []
    for p, w in ((p0, w0), (p1, w1)):
        for s_ in (-1, 1):
            for n_ in (-1, 1):
                pts.append(p + side * s_ * w / 2 + nrm * n_ * t / 2)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d_ in q:
        tris += [(a, b, c), (a, c, d_)]
    return np.array(pts), np.array(tris)
