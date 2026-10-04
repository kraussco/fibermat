#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp
import warnings
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from tqdm import tqdm

from fibermat import *
from fibermat import Mat


def _periodic_cloud(xyz, uvw, length, width, size, periodic):
    """Replicate fibers across the periodic neighborhood.

    Each ghost is a pure in-plane translation of an original fiber. The
    returned label array maps every row back to the original fiber index.
    Copies are created only for fibers close enough to the wrapped boundary
    that their center could lie within a bounding radius of another fiber.
    All eight neighbor shifts are included, not only ``(+x)``, ``(+y)`` and
    one diagonal.
    """
    n = len(xyz)
    empty_i = np.zeros(0, dtype=int)
    if n == 0:
        return (xyz, uvw, length, width, empty_i)

    parts = [(xyz, uvw, length, width, np.arange(n, dtype=int))]
    if periodic and size > 0:
        reach = float(np.max(length) + np.max(width))
        half = 0.5 * float(size)
        x = xyz[:, 0]
        y = xyz[:, 1]
        left = x < -half + reach
        right = x > half - reach
        bottom = y < -half + reach
        top = y > half - reach
        specs = (
            ((size, 0.), left),
            ((-size, 0.), right),
            ((0., size), bottom),
            ((0., -size), top),
            ((size, size), left & bottom),
            ((size, -size), left & top),
            ((-size, size), right & bottom),
            ((-size, -size), right & top),
        )
        for (dx, dy), mask in specs:
            if not np.any(mask):
                continue
            ghost = xyz[mask].copy()
            ghost[:, 0] += dx
            ghost[:, 1] += dy
            parts.append((
                ghost, uvw[mask], length[mask], width[mask],
                np.flatnonzero(mask).astype(int),
            ))

    return tuple(np.concatenate(col) for col in zip(*parts))


def _candidate_pairs(xy, length, width, pairs=None):
    """Fiber pairs whose centers can still touch, without building all pairs.

    ``pairs`` indexes rows of the periodic cloud. When it is omitted, candidates
    come from a KD-tree ball of radius ``max(length) + max(width)``, which is
    the farthest two centers can be while their segments remain within a
    half-width of each other. Self-pairs are left out; fiber endpoints are
    added separately.
    """
    n = len(xy)
    if pairs is not None:
        cand = np.asarray(pairs, dtype=int).reshape(-1, 2)
        cand = cand[cand[:, 0] != cand[:, 1]]
    elif n < 2:
        cand = np.empty((0, 2), dtype=int)
    else:
        reach = float(np.max(length) + np.max(width))
        if reach <= 0:
            cand = np.empty((0, 2), dtype=int)
        else:
            cand = cKDTree(xy).query_pairs(reach, output_type="ndarray")
            cand = np.asarray(cand, dtype=int)
            if cand.size == 0:
                cand = np.empty((0, 2), dtype=int)
            elif cand.ndim == 1:
                cand = cand.reshape(-1, 2)

    if len(cand) == 0:
        return np.empty((0, 2), dtype=int)

    i = cand[:, 0]
    j = cand[:, 1]
    limit = 0.5 * (length[i] + length[j] + width[i] + width[j])
    dist = np.linalg.norm(xy[i] - xy[j], axis=1)
    return cand[dist < limit]


def _dedupe_contacts(lab_a, lab_b, s_a, s_b):
    """Keep one row for each fiber pair and curvilinear position."""
    if len(lab_a) == 0:
        return lab_a, lab_b, s_a, s_b
    order = np.lexsort((s_b, s_a, lab_b, lab_a))
    lab_a, lab_b = lab_a[order], lab_b[order]
    s_a, s_b = s_a[order], s_b[order]
    key = np.stack((lab_a, lab_b, np.round(s_a, 6), np.round(s_b, 6)))
    _, keep = np.unique(key, axis=1, return_index=True)
    keep.sort()
    return lab_a[keep], lab_b[keep], s_a[keep], s_b[keep]


def _collect_contacts(xyz, uvw, length, width, size, periodic, pairs):
    """Contacts between originals and, one shift at a time, their images.

    Each shift is tested against the original fibers only. Ghost-ghost pairs
    repeat those same interactions, so they are not materialized.
    """
    n = len(xyz)
    empty = (
        np.zeros(0, dtype=int), np.zeros(0, dtype=int),
        np.zeros(0), np.zeros(0),
    )
    if n == 0:
        return empty

    if pairs is not None:
        xyz_i, uvw_i, length_i, width_i, labels = _periodic_cloud(
            xyz, uvw, length, width, size, periodic
        )
        cand = _candidate_pairs(xyz_i[:, :2], length_i, width_i, pairs=pairs)
        return _contacts_from_pairs(
            xyz_i, uvw_i, length_i, width_i, labels, cand[:, 0], cand[:, 1]
        )

    chunks = []
    labels0 = np.arange(n, dtype=int)
    reach = float(np.max(length) + np.max(width))
    if n < 2 or reach <= 0:
        return empty

    def consume(block_xyz, block_uvw, block_l, block_b, block_lab, pair_i, pair_j):
        if len(pair_i) == 0:
            return
        limit = 0.5 * (
            block_l[pair_i] + block_l[pair_j] + block_b[pair_i] + block_b[pair_j]
        )
        dist = np.linalg.norm(block_xyz[pair_i, :2] - block_xyz[pair_j, :2], axis=1)
        keep = dist < limit
        if not np.any(keep):
            return
        chunks.append(_contacts_from_pairs(
            block_xyz, block_uvw, block_l, block_b, block_lab,
            pair_i[keep], pair_j[keep],
        ))

    tree = cKDTree(xyz[:, :2])
    direct = np.asarray(tree.query_pairs(reach, output_type="ndarray"), dtype=int)
    if direct.size == 0:
        direct = np.empty((0, 2), dtype=int)
    elif direct.ndim == 1:
        direct = direct.reshape(-1, 2)
    consume(xyz, uvw, length, width, labels0, direct[:, 0], direct[:, 1])

    if periodic and size > 0:
        half = 0.5 * float(size)
        x = xyz[:, 0]
        y = xyz[:, 1]
        left = x < -half + reach
        right = x > half - reach
        bottom = y < -half + reach
        top = y > half - reach
        specs = (
            ((size, 0.), left),
            ((-size, 0.), right),
            ((0., size), bottom),
            ((0., -size), top),
            ((size, size), left & bottom),
            ((size, -size), left & top),
            ((-size, size), right & bottom),
            ((-size, -size), right & top),
        )
        for (dx, dy), mask in specs:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            ghost = xyz[idx].copy()
            ghost[:, 0] += dx
            ghost[:, 1] += dy
            found = tree.sparse_distance_matrix(
                cKDTree(ghost[:, :2]), reach, output_type="coo_matrix"
            )
            if found.nnz == 0:
                continue
            consume(
                np.concatenate((xyz, ghost)),
                np.concatenate((uvw, uvw[idx])),
                np.concatenate((length, length[idx])),
                np.concatenate((width, width[idx])),
                np.concatenate((labels0, idx.astype(int))),
                found.row, found.col + n,
            )

    if not chunks:
        return empty
    return _dedupe_contacts(
        np.concatenate([c[0] for c in chunks]),
        np.concatenate([c[1] for c in chunks]),
        np.concatenate([c[2] for c in chunks]),
        np.concatenate([c[3] for c in chunks]),
    )


def _segment_closest_xy(c1, d1, r1, c2, d2, r2):
    """Closest points of two finite segments in the plane.

    A point on the first segment is ``c1 + s * d1`` with ``|s| <= r1``.
    ``d1`` and ``d2`` are the in-plane direction components and are not
    required to be unit vectors. Returns ``s``, ``t`` and the distance.
    """
    n = len(c1)
    s = np.zeros(n)
    t = np.zeros(n)
    if n == 0:
        return s, t, np.zeros(0)

    w0 = c1 - c2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = np.einsum("ij,ij->i", d1, w0)
    e = np.einsum("ij,ij->i", d2, w0)
    denom = a * c - b * b

    both = (a > 1e-16) & (c > 1e-16)
    parallel = both & (np.abs(denom) <= 1e-12 * np.maximum(a * c, 1e-30))
    cross = both & ~parallel

    if np.any(cross):
        s[cross] = (b[cross] * e[cross] - c[cross] * d[cross]) / denom[cross]
        s[cross] = np.clip(s[cross], -r1[cross], r1[cross])
        t[cross] = np.clip(
            (b[cross] * s[cross] + e[cross]) / c[cross],
            -r2[cross], r2[cross],
        )
        s[cross] = np.clip(
            (b[cross] * t[cross] - d[cross]) / a[cross],
            -r1[cross], r1[cross],
        )
        t[cross] = np.clip(
            (b[cross] * s[cross] + e[cross]) / c[cross],
            -r2[cross], r2[cross],
        )

    if np.any(parallel):
        s[parallel] = np.clip(-d[parallel] / a[parallel], -r1[parallel], r1[parallel])
        t[parallel] = np.clip(
            (b[parallel] * s[parallel] + e[parallel]) / c[parallel],
            -r2[parallel], r2[parallel],
        )
        s[parallel] = np.clip(
            (b[parallel] * t[parallel] - d[parallel]) / a[parallel],
            -r1[parallel], r1[parallel],
        )

    point1 = (a <= 1e-16) & (c > 1e-16)
    if np.any(point1):
        t[point1] = np.clip(e[point1] / c[point1], -r2[point1], r2[point1])

    point2 = (c <= 1e-16) & (a > 1e-16)
    if np.any(point2):
        s[point2] = np.clip(-d[point2] / a[point2], -r1[point2], r1[point2])

    p1 = c1 + s[:, None] * d1
    p2 = c2 + t[:, None] * d2
    return s, t, np.linalg.norm(p1 - p2, axis=1)


def _infinite_line_abscissa(xyz, uvw, length, pair_i, pair_j):
    """Curvilinear abscissae of the infinite-line closest points.

    The parameter is the same one used historically: ``s`` is measured from
    the fiber center along the unit direction, and the second component is
    stored with a flipped sign inside the normal equations. ``inside`` is
    true when both feet lie in the open segment and the lines are not parallel.
    """
    n = len(pair_i)
    X = np.stack((xyz[pair_i], xyz[pair_j]), axis=1)
    U = np.stack((uvw[pair_i], uvw[pair_j]), axis=1)
    L = np.stack((length[pair_i], length[pair_j]), axis=1)
    r = np.diff(np.swapaxes(X, 1, 2), axis=2)
    gram = U @ np.swapaxes(U, 1, 2)
    rhs = U @ r
    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
    valid = det != 0
    s = np.column_stack((0.5 * L[:, 0], -0.5 * L[:, 1]))
    if np.any(valid):
        solved = np.linalg.solve(gram[valid], rhs[valid])
        s[valid, 0] = solved[:, 0, 0]
        s[valid, 1] = -solved[:, 1, 0]
    radius = 0.5 * L
    inside = valid & np.all(np.abs(s) < radius, axis=1)
    return s, inside


def _contacts_from_pairs(xyz, uvw, length, width, labels, pair_i, pair_j):
    """Accept overlapping pairs and return label-ordered contact rows.

    A pair is a contact when the infinite-line feet lie on both finite fibers
    (a centerline crossing, including former periodic images) or when the
    in-plane distance between the finite segments is below ``(b_i + b_j) / 2``.
    The second test covers parallel fibers and passes that miss the centerline
    but still overlap across the fiber width. Abscissae of true crossings keep
    the infinite-line solution.
    """
    if len(pair_i) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0), np.zeros(0))

    s_inf, inside = _infinite_line_abscissa(xyz, uvw, length, pair_i, pair_j)
    s_xy, t_xy, dist = _segment_closest_xy(
        xyz[pair_i, :2], uvw[pair_i, :2], 0.5 * length[pair_i],
        xyz[pair_j, :2], uvw[pair_j, :2], 0.5 * length[pair_j],
    )
    gap = 0.5 * (width[pair_i] + width[pair_j])
    accept = inside | (dist < gap)
    if not np.any(accept):
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0), np.zeros(0))

    s = np.column_stack((s_xy, t_xy))
    s[inside] = s_inf[inside]
    s = s[accept]
    lab = np.column_stack((labels[pair_i], labels[pair_j]))[accept]

    swap = lab[:, 0] > lab[:, 1]
    lab[swap] = lab[swap, ::-1]
    s[swap] = s[swap, ::-1]

    # Drop the same geometric contact discovered through several images.
    order = np.lexsort((s[:, 1], s[:, 0], lab[:, 1], lab[:, 0]))
    lab = lab[order]
    s = s[order]
    key = np.stack((
        lab[:, 0], lab[:, 1], np.round(s[:, 0], 6), np.round(s[:, 1], 6),
    ))
    _, keep = np.unique(key, axis=1, return_index=True)
    keep.sort()
    lab = lab[keep]
    s = s[keep]
    return lab[:, 0], lab[:, 1], s[:, 0], s[:, 1]


def _wrap_xy(xy, size, periodic):
    """Keep centers inside the cell. Periodic mats wrap; others are clipped."""
    if size <= 0:
        return xy
    half = 0.5 * float(size)
    if periodic:
        return (xy + half) % size - half
    return np.clip(xy, -half, half)


def _slip_pairs(xy, reach, size, periodic):
    """Index pairs that can still meet, including across the periodic boundary."""
    n = len(xy)
    if n < 2 or reach <= 0:
        return np.empty((0, 2), dtype=int)
    tree = cKDTree(xy)
    parts = []
    base = np.asarray(tree.query_pairs(reach, output_type="ndarray"), dtype=int)
    base = base.reshape(-1, 2)
    if len(base):
        parts.append(base)
    if periodic and size > 0:
        for dx, dy in _slip_shifts(size):
            if dx == 0.0 and dy == 0.0:
                continue
            other = cKDTree(xy + (dx, dy))
            coo = tree.sparse_distance_matrix(other, reach, output_type="coo_matrix")
            if coo.nnz == 0:
                continue
            i = coo.row.astype(int).copy()
            j = coo.col.astype(int).copy()
            swap = i > j
            i[swap], j[swap] = j[swap], i[swap]
            parts.append(np.column_stack((i, j)))
    if not parts:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.vstack(parts), axis=0)


def _slip_shifts(size):
    return tuple(
        (dx, dy)
        for dx in (-size, 0.0, size)
        for dy in (-size, 0.0, size)
    )


def _rectangle_mtv(c1, u1, length1, width1, c2, u2, length2, width2):
    """Translation of body 2 out of body 1. NaN where the rectangles miss."""
    m = len(c1)
    mtv = np.full((m, 2), np.nan)
    if m == 0:
        return mtv
    n1 = np.column_stack((-u1[:, 1], u1[:, 0]))
    n2 = np.column_stack((-u2[:, 1], u2[:, 0]))
    delta = c2 - c1
    best = np.full(m, np.inf)
    separated = np.zeros(m, dtype=bool)
    half_l1, half_l2 = 0.5 * length1, 0.5 * length2
    half_b1, half_b2 = 0.5 * width1, 0.5 * width2
    specs = (
        (u1, half_l1,
         np.abs(np.einsum("ij,ij->i", u2, u1)) * half_l2
         + np.abs(np.einsum("ij,ij->i", n2, u1)) * half_b2),
        (n1, half_b1,
         np.abs(np.einsum("ij,ij->i", u2, n1)) * half_l2
         + np.abs(np.einsum("ij,ij->i", n2, n1)) * half_b2),
        (u2,
         np.abs(np.einsum("ij,ij->i", u1, u2)) * half_l1
         + np.abs(np.einsum("ij,ij->i", n1, u2)) * half_b1,
         half_l2),
        (n2,
         np.abs(np.einsum("ij,ij->i", u1, n2)) * half_l1
         + np.abs(np.einsum("ij,ij->i", n1, n2)) * half_b1,
         half_b2),
    )
    for axis, extent1, extent2 in specs:
        dist = np.einsum("ij,ij->i", delta, axis)
        pen = extent1 + extent2 - np.abs(dist)
        separated |= pen <= 1e-8
        better = (~separated) & (pen < best)
        best = np.where(better, pen, best)
        sign = np.where(dist >= 0.0, 1.0, -1.0)
        mtv = np.where(better[:, None], (sign * pen)[:, None] * axis, mtv)
    mtv[separated] = np.nan
    return mtv


def _layer_overlaps(xy, direction, length, width, size, periodic):
    """Overlapping pairs and the translation that takes fiber j off fiber i."""
    reach = float(np.max(length) + np.max(width)) + 1e-6
    pairs = _slip_pairs(xy, reach, size, periodic)
    if len(pairs) == 0:
        return pairs, np.empty((0, 2))
    best = np.full((len(pairs), 2), np.nan)
    best_pen = np.full(len(pairs), np.inf)
    ii, jj = pairs[:, 0], pairs[:, 1]
    for dx, dy in _slip_shifts(size) if periodic and size > 0 else ((0.0, 0.0),):
        mtv = _rectangle_mtv(
            xy[ii], direction[ii], length[ii], width[ii],
            xy[jj] + (dx, dy), direction[jj], length[jj], width[jj],
        )
        valid = np.isfinite(mtv[:, 0])
        pen = np.full(len(pairs), np.inf)
        if np.any(valid):
            pen[valid] = np.linalg.norm(mtv[valid], axis=1)
        take = valid & (pen < best_pen)
        best_pen = np.where(take, pen, best_pen)
        best = np.where(take[:, None], mtv, best)
    keep = np.isfinite(best[:, 0])
    return pairs[keep], best[keep]


def _relax_positions(xy, direction, length, width, size, periodic, rounds, rng=None):
    """Slide each fiber out of its worst overlap. Angles stay fixed."""
    stall = 0
    previous = np.inf
    for _ in range(rounds):
        pairs, mtv = _layer_overlaps(xy, direction, length, width, size, periodic)
        if len(pairs) == 0:
            return xy
        pen = np.linalg.norm(mtv, axis=1)
        energy = float(pen.sum())
        disp = np.zeros_like(xy)
        worst = np.full(len(xy), -1.0)
        for k, (i, j) in enumerate(pairs):
            if pen[k] >= worst[i]:
                worst[i] = pen[k]
                disp[i] = -mtv[k]
            if pen[k] >= worst[j]:
                worst[j] = pen[k]
                disp[j] = mtv[k]
        xy = _wrap_xy(xy + disp, size, periodic)
        if energy >= previous - 1e-2:
            stall += 1
        else:
            stall = 0
        previous = min(previous, energy)
        if stall >= 4 and rng is not None:
            fiber = int(np.argmax(worst))
            half = 0.5 * size
            saved = xy[fiber].copy()
            best_xy = saved
            best_energy = np.inf
            for _trial in range(12):
                xy[fiber] = rng.uniform(-half, half, size=2)
                trial_pairs, trial_mtv = _layer_overlaps(
                    xy, direction, length, width, size, periodic
                )
                trial_energy = 0.0 if len(trial_pairs) == 0 else float(
                    np.linalg.norm(trial_mtv, axis=1).sum()
                )
                if trial_energy < best_energy:
                    best_energy = trial_energy
                    best_xy = xy[fiber].copy()
                    if trial_energy <= 1e-8:
                        break
            xy[fiber] = best_xy
            stall = 0
    return xy


def _separate_layer(xy, direction, length, width, size, periodic, rng):
    """Clear one layer. Fibers that still overlap are reported as spilled."""
    active = np.arange(len(xy))
    for scale in np.linspace(0.5, 1.0, 8):
        xy = _relax_positions(
            xy, direction, scale * length, scale * width, size, periodic, 18, rng
        )
    for _ in range(len(xy) + 1):
        xy[active] = _relax_positions(
            xy[active], direction[active], length[active], width[active],
            size, periodic, 18, rng,
        )
        pairs, mtv = _layer_overlaps(
            xy[active], direction[active], length[active], width[active],
            size, periodic,
        )
        if len(pairs) == 0 or len(active) <= 1:
            break
        pen = np.linalg.norm(mtv, axis=1)
        load = np.zeros(len(active))
        np.add.at(load, pairs[:, 0], pen)
        np.add.at(load, pairs[:, 1], pen)
        active = np.delete(active, int(np.argmax(load)))
    return active, xy


def _newcomer_fits(xy, direction, length, width, size, periodic, host, fiber, rng):
    """True if ``fiber`` can be parked in ``host`` without moving the host."""
    half = 0.5 * size
    home = xy[fiber].copy()
    host = np.asarray(host, dtype=int)
    for _trial in range(8):
        xy[fiber] = rng.uniform(-half, half, size=2)
        for _step in range(4):
            index = np.append(host, fiber)
            pairs, mtv = _layer_overlaps(
                xy[index], direction[index], length[index], width[index],
                size, periodic,
            )
            if len(pairs) == 0:
                return True
            last = len(index) - 1
            involved = (pairs[:, 0] == last) | (pairs[:, 1] == last)
            if not np.any(involved):
                return True
            disp = np.zeros(2)
            for k in np.flatnonzero(involved):
                if pairs[k, 0] == last:
                    disp -= mtv[k]
                else:
                    disp += mtv[k]
            xy[fiber] = _wrap_xy(xy[fiber] + disp, size, periodic)
    xy[fiber] = home
    return False


def _absorb_leftovers(layers, xy, direction, length, width, size, periodic, rng):
    """Move fibers from a thin top layer down into an earlier clear layer."""
    if len(layers) < 2 or len(layers[-1]) > 12:
        return layers
    hosts = [np.asarray(layer, dtype=int) for layer in layers[:-1]]
    pending = [int(fiber) for fiber in np.asarray(layers[-1], dtype=int)]
    still = []
    for fiber in pending:
        parked = False
        for host_index, host in enumerate(hosts):
            if _newcomer_fits(
                xy, direction, length, width, size, periodic, host, fiber, rng
            ):
                hosts[host_index] = np.append(host, fiber)
                parked = True
                break
        if not parked:
            still.append(fiber)
    if still:
        hosts.append(np.asarray(still, dtype=int))
    return hosts


def _nested_gap(dist, width1, thick1, width2, thick2):
    """Center separation where two elliptical cross-sections just miss.

    ``dist`` is the distance between centerlines. The required gap is the
    largest sum of the two half-thicknesses over the lateral overlap, so a
    touch near the thin edge needs less height than a crossing on the
    centerline.
    """
    dist = np.asarray(dist, dtype=float)
    gap = np.zeros(len(dist))
    lo = np.maximum(-0.5 * width1, dist - 0.5 * width2)
    hi = np.minimum(0.5 * width1, dist + 0.5 * width2)
    overlap = hi > lo + 1e-12
    if not np.any(overlap):
        return gap
    for weight in np.linspace(0.0, 1.0, 11):
        lateral = lo + weight * (hi - lo)
        left = np.clip(2.0 * lateral / width1, -1.0, 1.0)
        right = np.clip(2.0 * (lateral - dist) / width2, -1.0, 1.0)
        thick = (
            0.5 * thick1 * np.sqrt(np.maximum(0.0, 1.0 - left * left))
            + 0.5 * thick2 * np.sqrt(np.maximum(0.0, 1.0 - right * right))
        )
        gap = np.maximum(gap, np.where(overlap, thick, 0.0))
    return gap


def _contact_gaps(xy, direction, length, width, thickness, size, periodic):
    """Pairs whose planforms meet, and the vertical gap their centers need."""
    reach = float(np.max(length) + np.max(width)) + 1e-6
    pairs = _slip_pairs(xy, reach, size, periodic)
    if len(pairs) == 0:
        return pairs, np.zeros(0)
    best = np.zeros(len(pairs))
    ii, jj = pairs[:, 0], pairs[:, 1]
    shifts = _slip_shifts(size) if periodic and size > 0 else ((0.0, 0.0),)
    for dx, dy in shifts:
        origin = xy[jj] + (dx, dy)
        sat = _rectangle_mtv(
            xy[ii], direction[ii], length[ii], width[ii],
            origin, direction[jj], length[jj], width[jj],
        )
        _s, _t, dist = _segment_closest_xy(
            xy[ii], direction[ii], 0.5 * length[ii],
            origin, direction[jj], 0.5 * length[jj],
        )
        nested = _nested_gap(
            dist, width[ii], thickness[ii], width[jj], thickness[jj]
        )
        full = 0.5 * (thickness[ii] + thickness[jj])
        # A corner overlap that the lateral sample misses still needs a gap.
        gap = np.where(nested > 1e-8, nested, full)
        gap = np.where(np.isfinite(sat[:, 0]), gap, 0.0)
        best = np.maximum(best, gap)
    keep = best > 1e-6
    return pairs[keep], best[keep]


def _lowest_free(floor, blocks):
    """Lowest height at or above ``floor`` outside every open interval.

    The interval edges are allowed: that is the two ellipses just touching.
    """
    if not blocks:
        return floor
    blocks = sorted(blocks)
    merged = [list(blocks[0])]
    for start, end in blocks[1:]:
        if start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    height = floor
    for start, end in merged:
        if height <= start + 1e-9:
            return height
        if height < end - 1e-9:
            height = end
    return height


def _settle_heights(thickness, pairs, gaps, order=None):
    """Drop each fiber into the lowest slot that clears the fibers it meets.

    ``order`` is the sequence used in each pass. Fibers earlier in that
    sequence keep the lower slots; later ones nest into whatever is left.
    """
    n = len(thickness)
    z = 0.5 * np.asarray(thickness, dtype=float).copy()
    sequence = np.arange(n) if order is None else np.asarray(order, dtype=int)
    rank = np.full(n, n, dtype=int)
    rank[sequence] = np.arange(len(sequence))
    neighbors = [[] for _ in range(n)]
    for (i, j), gap in zip(pairs, gaps):
        neighbors[int(i)].append((int(j), float(gap)))
        neighbors[int(j)].append((int(i), float(gap)))
    # Only fibers already placed (earlier in ``order``) hold a slot. Later
    # fibers drop into the gaps those leave, including between full sheets.
    for fiber in sequence:
        blocks = [
            (z[other] - gap, z[other] + gap)
            for other, gap in neighbors[fiber]
            if rank[other] < rank[fiber]
        ]
        z[fiber] = _lowest_free(0.5 * thickness[fiber], blocks)
    return z


def slip(mat, periodic=True, seed=0):
    """Slide fibers apart without changing their orientation, then stack the rest.

    Orientations stay the ones on the mat, so a random mat stays random.
    Centers move in the plane until each sheet is free of overlaps. Vertically,
    a fiber is not assigned to an integer sheet: it drops into the lowest
    slot that clears the fibers it actually meets. A glancing contact, where
    only the thin edge of the ellipse is involved, needs less than a full
    thickness, so the fiber can sit between the heights of a full stack.

    Parameters
    ----------
    mat : pandas.DataFrame
        Set of fibers represented by a :class:`~.Mat` object.
    periodic : bool, optional
        Wrap motion across the cell when True. Default is True.
    seed : int, optional
        Seed for the order in which fibers are offered to a layer.
        Default is 0.

    Returns
    -------
    mat : pandas.DataFrame
        The same mat, after the rearrangement.

    """
    if mat is None or len(mat) == 0:
        return mat

    xy = np.array(mat[["x", "y"]].to_numpy(dtype=float), copy=True)
    uvw = mat[["u", "v", "w"]].to_numpy(dtype=float)
    length = mat["l"].to_numpy(dtype=float)
    width = mat["b"].to_numpy(dtype=float) + 1e-3
    thickness = mat["h"].to_numpy(dtype=float)
    horiz = np.linalg.norm(uvw[:, :2], axis=1)
    span = np.where(horiz > 1e-8, length * horiz, width)
    direction = np.zeros((len(mat), 2))
    flat = horiz > 1e-8
    direction[flat] = uvw[flat, :2] / horiz[flat, None]
    direction[~flat] = (1.0, 0.0)
    size = float(mat.attrs["size"])
    # Random orientations jam near half the cell area. Stay under that.
    budget = 0.42 * size * size
    rng = np.random.default_rng(seed)
    pool = rng.permutation(len(mat)).tolist()

    layers = []
    while pool:
        batch = []
        rest = []
        area = 0.0
        for fiber in pool:
            piece = float(span[fiber] * width[fiber])
            if batch and area + piece > budget:
                rest.append(fiber)
            else:
                batch.append(fiber)
                area += piece
        index = np.asarray(batch, dtype=int)
        # Extend each rectangle by half a width at either end so the rounded
        # tip, which the contact test counts, is kept clear as well.
        kept_local, placed = _separate_layer(
            xy[index].copy(), direction[index],
            span[index] + width[index], width[index],
            size, periodic, rng,
        )
        kept = index[kept_local]
        spilled = [int(fiber) for fiber in index if fiber not in set(kept.tolist())]
        xy[kept] = placed[kept_local]
        layers.append(kept)
        pool = spilled + rest

    collision = span + width
    layers = _absorb_leftovers(
        layers, xy, direction, collision, width, size, periodic, rng
    )

    # Real ellipse, not the slightly widened shape used while sliding.
    true_width = width - 1e-3
    pairs, gaps = _contact_gaps(
        xy, direction, span, true_width, thickness, size, periodic
    )
    placed = [int(fiber) for layer in layers for fiber in np.asarray(layer, dtype=int)]
    seen = set(placed)
    placed.extend(fiber for fiber in range(len(mat)) if fiber not in seen)
    z = _settle_heights(thickness, pairs, gaps, order=placed)
    order = np.argsort(z, kind="mergesort")

    data = np.array(mat.to_numpy(), dtype=float, copy=True)
    data[:, 3:5] = xy
    data[:, 5] = z
    mat.iloc[:, :] = data[np.asarray(order, dtype=int)]
    return mat


# `Stack.init` takes a parameter named slip, which would hide this function.
_slip_mat = slip


class Net(pd.DataFrame):
    r"""
    A class inherited from pandas.DataFrame_ **to build a fiber network**.

    It describes nodes and connections between fibers within a :class:`~.Mat` object:
        - *nodes* are defined as the nearest points between pairs of fibers.
        - *connections* link pairs of nodes to define relative positions between fibers.

    Parameters:
    -----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`~.Mat` object.
    periodic : bool, optional
        If True, fibers are duplicated for periodicity. Default is True.
    pairs : numpy.ndarray, optional
        Candidate pairs indexing the periodic replication (original fibers
        come first). By default, a KD-tree supplies the centers close enough
        to meet. Size: (m x 2).

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters.
        Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    :Use:
        >>> # Generate a set of fibers
        >>> mat = Mat(100)
        >>> # Build the fiber network
        >>> net = Net(mat)
        >>> net
              A   B         sA         sB         xA         yA         zA         xB         yB         zB
        0     0   0  12.500000 -12.500000   6.057752  20.856058 -24.338157  -1.176401  -3.074404 -24.338157
        1     0   2   3.938063  -1.799582   3.580217  12.660413 -24.338157   3.580217  12.660413 -23.967450
        2     0   3   6.509881   8.253676   4.324414  15.122205 -24.338157   4.324414  15.122205 -23.766064
        3     0   5   0.269800  -7.165082   2.518746   9.149084 -24.338157   2.518746   9.149084 -21.802237
        4     0   6 -10.466114   6.264470  -0.587864  -1.127531 -24.338157  -0.587864  -1.127531 -21.637518
        ..   ..  ..        ...        ...        ...        ...        ...        ...        ...        ...
        862  95  95  12.500000 -12.500000 -14.234141  11.919304  23.096819 -17.446723 -12.873423  23.096819
        863  96  96  12.500000 -12.500000  15.557356 -19.115497  23.645974  -6.906063  -8.143040  23.645974
        864  97  97  12.500000 -12.500000 -12.739951 -17.721142  23.874757 -35.249295  -6.843209  23.874757
        865  98  98  12.500000 -12.500000  17.087939 -34.582099  24.091469  15.806064  -9.614985  24.091469
        866  99  99  12.500000 -12.500000 -17.894817 -13.721749  24.516947 -31.635635   7.163412  24.516947
        <BLANKLINE>
        [867 rows x 10 columns]

    Data:
    -----
    + index : pandas.Index
        Connection label. Each label refers to a unique connection.
    + Pair of fibers:
        - A : pandas.Series
            1ˢᵗ fiber label. It must satisfy `net.A` ≤ `net.B`.
        - B : pandas.Series
            2ⁿᵈ fiber label. It must satisfy `net.A` ≤ `net.B`.
    + Curvilinear abscissa:
        - sA : pandas.Series
            Curvilinear abscissa of node along the first fiber (mm).
        - sB : pandas.Series
            Curvilinear abscissa of node along the second fiber (mm).
    + Relative node positions:
        - xA : pandas.Series
            X-coordinate of node along the first fiber (mm).
        - yA : pandas.Series
            Y-coordinate of node along the first fiber (mm).
        - zA : pandas.Series
            Z-coordinate of node along the first fiber (mm).
        - xB : pandas.Series
            X-coordinate of node along the second fiber (mm).
        - yB : pandas.Series
            Y-coordinate of node along the second fiber (mm).
        - zB : pandas.Series
            Z-coordinate of node along the second fiber (mm).

    ----

    Attributes
    ----------
    :attr:`attrs` :
        Global attributes of DataFrame.

    Methods
    -------
    :meth:`init` :
        Build a fiber network.
    :meth:`check` :
        Check that a :class:`Net` object is defined correctly.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Net` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.
        kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        `Net.init`.

        """
        if len(args) and Net._is(args[0]):
            # Initialize the DataFrame from argument
            super().__init__(*args, **kwargs)
            # Copy global attributes from argument
            self.attrs = args[0].attrs
            # Copy global flags from argument
            self.flags.mat = args[0].flags.mat

        else:
            # Initialize the DataFrame from parameters
            self.__init__(Net.init(*args, **kwargs))

        # Check `Net` object
        assert Net.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(mat=None, periodic=True, pairs=None, **_):
        """
        Build a fiber network.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a :class:`~.Mat` object.
        periodic : bool, optional
            If True, fibers are duplicated for periodicity. Default is True.
        pairs : numpy.ndarray, optional
            Candidate pairs indexing the periodic replication (original fibers
            come first). By default, candidates are the centers a KD-tree
            reports within a bounding radius, instead of every fiber pair.

        Returns
        -------
        net : pandas.DataFrame
            Initialized :class:`Net` object.

        """
        # Optional
        if mat is None:
            mat = Mat()

        assert Mat.check(mat)

        xyz = mat[[*"xyz"]].to_numpy(dtype=float)
        uvw = mat[[*"uvw"]].to_numpy(dtype=float)
        length = mat["l"].to_numpy(dtype=float)
        width = mat["b"].to_numpy(dtype=float)
        n = len(mat)

        # Periodic images are tested one shift at a time. Candidate pairs are
        # the centers a KD-tree places inside a bounding radius, not every pair.
        A, B, sA, sB = _collect_contacts(
            xyz, uvw, length, width, mat.attrs["size"], periodic, pairs
        )

        # Fiber endpoints (one row per fiber, s = ±l/2).
        if n:
            end_s = np.column_stack((0.5 * length, -0.5 * length))
            end_id = np.arange(n, dtype=int)
            A = np.concatenate((A, end_id))
            B = np.concatenate((B, end_id))
            sA = np.concatenate((sA, end_s[:, 0]))
            sB = np.concatenate((sB, end_s[:, 1]))
            A, B, sA, sB = _dedupe_contacts(A, B, sA, sB)

        if len(A):
            abscissa = np.column_stack((sA, sB)).reshape(-1, 2, 1)
            pair = np.column_stack((A, B))
            points = xyz[pair] + abscissa * uvw[pair]
            data = np.c_[pair, abscissa.reshape(-1, 2), points.reshape(-1, 6)]
        else:
            data = np.zeros((0, 10))

        # Initialize net DataFrame
        net = pd.DataFrame(
            data=data,
            columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"]
        )
        # Convert type to int
        net[[*"AB"]] = net[[*"AB"]].astype(int)

        # Set attributes
        net.attrs = mat.attrs
        net.attrs["periodic"] = periodic

        # Set flags
        net.flags.mat = mat

        # Return the `Net` object
        return net

    # ~~~ Public properties ~~~ #

    @property
    def attrs(self):
        """
        Global attributes of DataFrame:
            - n : int
                Number of fibers. By default, it is empty (n = 0).
            - size : float
                Box dimensions (mm). By default, the domain is a 50 mm square cube.
            - periodic : bool
                Boundary periodicity. By default, the domain is periodic.

        """
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = attrs

    # ~~~ Public methods ~~~ #

    def check(self: pd.DataFrame):
        """
        Check that a :class:`Net` object is defined correctly.

        This method is called when a :class:`Net` object is initialized.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of :class:`Net` object.
        AttributeError
            If any attributes are missing from the dictionary :attr:`attrs`.
        IndexError
            If row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., m-1] where m is the number of connections.
                - Connection labels are not sorted.
        TypeError
            If labels are not integers.
        ValueError
            If any of the following conditions are not met:
                - Fiber labels are incorrect.
                - There are duplicate connections.
                - Fiber labels are not ordered.

        Returns
        -------
        bool
            Indicates whether the object can be instantiated as :class:`Net`.

        .. TIP::
            - If `self` is None, it returns an empty :class:`Net` object.
            - If a `"skip_check"` flag is True in :attr:`attrs`, the check is passed.

        """
        if self is None:
            net = Net()
        else:
            net = self

        if "skip_check" in net.attrs.keys() and net.attrs["skip_check"]:
            warnings.warn("{}.attrs['skip_check'] is active."
                          " Delete it or set it to False.".format(net.__class__),
                          UserWarning)
            return True

        assert Mat.check(net.flags.mat)

        # Keys
        try:
            net[["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"]]
        except KeyError as e:
            raise KeyError(e)

        # Attributes
        if not ("n" in net.attrs.keys()):
            raise AttributeError("'n' is not in attribute dictionary.")
        if not ("size" in net.attrs.keys()):
            raise AttributeError("'size' is not in attribute dictionary.")
        if not ("periodic" in net.attrs.keys()):
            raise AttributeError("'periodic' is not in attribute dictionary.")

        # Indices
        if not np.all(np.unique(net.index) == np.arange(len(net))):
            raise IndexError("Row indices must be unique in [0,..., {}]."
                             .format(len(net) - 1))
        if not np.all(net.index == np.arange(len(net))):
            raise IndexError("Connection labels must be sorted.")

        # Types
        if net[[*"AB"]].values.dtype != int:
            raise TypeError("Fiber labels are not integers.")

        # Data
        if len(net) and not (0 <= net[[*"AB"]].values.min()
                             and net[[*"AB"]].values.max() < net.attrs["n"]):
            raise ValueError("Fiber labels must be in [0,..., {}]."
                             .format(net.attrs["n"] - 1))
        if (len(net) != len(
                net[["A", "B", "sA", "sB"]].drop_duplicates(
                    ignore_index=True))):
            raise ValueError("Connections must be unique.")
        if not np.all(net.A <= net.B):
            raise ValueError("Pairs of fiber labels must be ordered: A ≤ B.")

        # Return True if the test is correct
        return True

    # ~~~ Private methods ~~~ #

    def _is(self: pd.DataFrame) -> bool:
        if self is None:
            return False
        try:
            __class__.check(self)
            return True
        except (KeyError, AttributeError, IndexError, TypeError, ValueError):
            return False


class Stack(Net):
    """
    A class inherited from :class:`Net` **to stack a set of fibers**.

    It solves the *linear programming system*:

    .. MATH::
        \min_{z} (-\mathbf{f} \cdot \mathbf{z})
        \quad s.t. \quad \mathbb{C} \, \mathbf{z} \leq \mathbf{H}
        \quad and \quad \mathbf{z} \geq \mathbf{h} / 2
    .. MATH::
        with \quad \mathbf{f} = -\mathbf{m} \, g \quad and \quad \mathbf{h} > 0

    where:
        - 𝐳 is the vector of fiber vertical positions (*unknowns of the problem*).
        - 𝐟 is the vector of fiber weights (with 𝐦 : fiber masses, 𝑔 gravity).
        - 𝐡 is the vector of fiber thicknesses.
        - ℂ is the matrix of inequality constraints that positions must satisfy to prevent the fibers from crossing each other.
        - -𝐇 is the vector of minimum distances between the pairs of fibers.

    *Non-penetration conditions* between two fibers give expressions for the rows of ℂ and 𝐇:

    .. MATH::
        z_B - z_A \geq (h_A + h_B) \, / \, 2
        \quad \Leftrightarrow \quad z_A - z_B \leq - (h_A + h_B) \, / \, 2

    Parameters:
    -----------
    net : pandas.DataFrame, optional
        Fiber network represented by a :class:`Net` object.
    threshold : float, optional
        Threshold distance value for proximity detection (mm).
    kwargs :
        Additional keyword arguments passed to the solver.

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters.
        Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

        :Use:
            >>> # Generate a set of fibers
            >>> mat = Mat(100)
            >>> # Build the fiber network
            >>> net = Net(mat)
            >>> # Stack fibers
            >>> stack = Stack(net)
            >>> stack
                  A   B         sA         sB         xA         yA    zA         xB         yB    zB
            0     0   0  12.500000 -12.500000   6.057752  20.856058   0.5  -1.176401  -3.074404   0.5
            1     0   2   3.938063  -1.799582   3.580217  12.660413   0.5   3.580217  12.660413   1.5
            2     0   3   6.509881   8.253676   4.324414  15.122205   0.5   4.324414  15.122205   2.5
            3     0   5   0.269800  -7.165082   2.518746   9.149084   0.5   2.518746   9.149084   1.5
            4     0   6 -10.466114   6.264470  -0.587864  -1.127531   0.5  -0.587864  -1.127531   1.5
            ..   ..  ..        ...        ...        ...        ...   ...        ...        ...   ...
            862  95  95  12.500000 -12.500000 -14.234141  11.919304  27.5 -17.446723 -12.873423  27.5
            863  96  96  12.500000 -12.500000  15.557356 -19.115497  27.5  -6.906063  -8.143040  27.5
            864  97  97  12.500000 -12.500000 -12.739951 -17.721142  27.5 -35.249295  -6.843209  27.5
            865  98  98  12.500000 -12.500000  17.087939 -34.582099  27.5  15.806064  -9.614985  27.5
            866  99  99  12.500000 -12.500000 -17.894817 -13.721749  26.5 -31.635635   7.163412  26.5
            <BLANKLINE>
            [867 rows x 10 columns]

    ----

    Methods
    -------
    :meth:`init` :
        Stack a fiber network by gravity.
    :meth:`solve` :
        Solve the stacking problem.
    :meth:`constraint` :
        Assemble the linear system.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Stack` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.
        kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        `Stack.init`.

        """
        super().__init__(Stack.init(*args, **kwargs))

        # Check `Stack` object
        assert Stack.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(net=None, threshold=None, slip=False, **kwargs):
        """
        Stack a fiber network by gravity.

        Parameters
        ----------
        net : pandas.DataFrame, optional
            Fiber network represented by a :class:`Net` object.
        threshold : float, optional
            Threshold distance value for proximity detection (mm).
        slip : bool, optional
            When True, fibers slide in the plane, keeping their orientation,
            and each one then rests at the lowest height that clears the
            fibers it meets. Default is False, which keeps the layout and
            stacks by a full thickness.
        kwargs :
            Additional keyword arguments passed to the solver.

        Returns
        -------
        stack : pandas.DataFrame
            Stacked :class:`Net` object.

        .. WARNING::
            :class:`~.Mat` object is modified during execution.
            With ``slip=True`` the in-plane pose and the fiber order change too.

        """
        # Optional
        if net is None:
            net = Net()

        assert Net.check(net)

        # Get material data
        mat = net.flags.mat
        if slip:
            _slip_mat(mat, periodic=net.attrs.get("periodic", True))
            net = Net(mat, periodic=net.attrs.get("periodic", True))

        # The slipped mat already rests each fiber in the lowest open slot.
        # The gravity program would lift those contacts by a full thickness.
        linsol = None if slip else Stack.solve(net, **kwargs)
        stack = net

        if linsol:
            # Update DataFrames
            mat.z = linsol.x
            stack = Net(mat, **net.attrs)

        if (linsol or slip) and threshold is not None:
            # Remove nodes based on threshold distances between nodes
            mask = np.zeros(len(stack))
            # : 0 if removed node
            # : 1 if kept node
            # : 2 if end node
            mask[np.abs(stack.zB.values - stack.zA.values) <= threshold] = 1
            mask[stack.A == stack.B] = 2
            stack = stack[mask > 0]
            stack.reset_index(drop=True, inplace=True)
            # Add global flags
            stack.flags.mat = mat

        # Set attributes
        stack.attrs["threshold"] = threshold

        # Return the `Stack` object
        return stack

    # ~~~ Public methods ~~~ #

    @staticmethod
    def solve(net=None, **_):
        """
        Solve the stacking problem.

        Parameters
        ----------
        net : pandas.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Returns
        -------
        linsol : OptimizeResult
            Results of linear programming solver.

        .. SEEALSO::
            The solver is based on scipy.optimize.linprog_.

        .. _scipy.optimize.linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

        """
        # Optional
        if net is None:
            net = Net()

        assert Net.check(net)

        # Assemble the linear programming system
        C, f, H, h = Stack.constraint(net)

        if net.attrs["n"]:
            # Linear programming solver
            bounds = np.c_[0.5 * h, np.full(len(h), np.inf)]
            linsol = sp.optimize.linprog(f, C, H, bounds=bounds, method='highs')
        else:
            linsol = None

        return linsol

    @staticmethod
    def constraint(net=None, **_):
        """
        Assemble the linear programming system.

        Parameters
        ----------
        net : pd.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Returns
        -------
            C : sparse matrix
                Constraint matrix.
            mg : numpy.ndarray
                Force vector.
            H : numpy.ndarray
                Minimum distance vector.
            h : numpy.ndarray
                Thickness vector.

        """
        # Optional
        if net is None:
            net = Net()

        assert Net.check(net)

        # Get network data
        mask = net.A.values < net.B.values
        i = net.A[mask].values
        j = net.B[mask].values
        k = 1 * np.arange(len(i))
        O = i * 0  # : zero
        I = O + 1  # : one

        # Get material data
        mat = net.flags.mat
        h = mat.h.values

        # Create constraint data
        row = np.array([k, k]).ravel()
        col = np.array([i, j]).ravel()
        data = np.array([I, -I]).ravel()

        # Initialize ℂ matrix
        C = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(1 * len(net[mask]), 1 * len(mat)))

        # Initialize 𝒇 and 𝑯 vectors
        mg = np.pi / 4 * mat[[*"lbh"]].prod(axis=1)  # : potential field
        H = np.zeros(C.shape[0])
        # X₂ - X₁ ≥ ½(h₁ + h₂) ⟺ X₁ - X₂ ≤ -½(h₁ + h₂)
        H -= 0.5 * (h[i] + h[j])

        return C, mg, H, h


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # from fibermat import *

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)

    # Check data
    Stack.check(stack)  # or `stack.check()`
    # -> returns True if correct, otherwise it raises an error.

    # Get the linear system
    C, mg, H, h = Stack.constraint(stack)
    linsol = Stack.solve(stack)
    # Contact force
    f = linsol.ineqlin.marginals
    # Resulting force
    load = 0.5 * f @ np.abs(C) + 0.5 * f @ C

    # Normalize by fiber weight
    load /= np.pi / 4 * mat[[*"lbh"]].prod(axis=1).mean()
    # Get loaded nodes
    points = (stack[stack.A < stack.B][["xA", "yA", "zA", "xB", "yB", "zB"]]
              .values.reshape(-1, 2, 3))
    # Prepare color scale
    cmap = plt.cm.viridis
    color = interp1d([np.min(load), np.max(load)], [0, 1])

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mat):
        # Draw fibers
        for i in tqdm(range(len(mat)), desc="Draw fibers"):
            # Get fiber data
            fiber = mat.iloc[i]
            # Calculate fiber end points
            A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
            B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
            plt.plot(*np.c_[A, B], c=cmap(color(load[i])))
    if len(points):
        # Draw contacts
        for point in tqdm(points[~np.isclose(f, 0)], desc="Draw nodes"):
            plt.plot(*point.T, '--ok', lw=1, mfc='none', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * stack.attrs["size"], 0.5 * stack.attrs["size"])
    ax.set_ylim(-0.5 * stack.attrs["size"], 0.5 * stack.attrs["size"])
    # Add a color bar
    norm = plt.Normalize(vmin=np.min(load), vmax=np.max(load))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_label("Load / $mg$ ($N\,/\,N$)")
    plt.show()
