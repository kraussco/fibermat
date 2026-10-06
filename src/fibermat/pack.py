#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from fibermat.mat import Mat
from fibermat.net import _lowest_free, _nested_gap, _rectangle_mtv, _segment_closest_xy


def pack(box=(50.0, 10.0, 15.0), length=12.5, diameter=1.0,
         width=None, thickness=None, volume_fraction=0.6, sweeps=5, seed=0,
         bend=False, section="ellipse"):
    """Drop straight fibers at random angles into a mat, as in SMC.

    The box is ``(length, width, height)`` in millimetres, centered in the
    plane and standing on ``z = 0``. It is periodic in the plane, so a fiber
    may leave through one side and come back on the other. Each fiber is
    given a random in-plane angle and then lowered onto the fibers already
    in place. An elliptical section thins at the edge, so a glancing
    contact needs less than a full thickness. A rectangular section keeps
    that thickness across the whole width.

    ``width`` and ``thickness`` are the in-plane width and the vertical
    thickness of the cross-section. Leave both unset for a round fiber of
    the given ``diameter``.     A flat tow is wide and much thinner than it is
    wide, so a crossing costs only that small thickness.

    With ``bend=True`` each tow follows the surface already in the mat.
    It rises where it crosses another tow and sits down in the gaps.
    The centerline of each tow is stored in ``mat.attrs["centerlines"]``.

    ``volume_fraction`` is a cap. The achieved fraction is stored in
    ``mat.attrs["volume_fraction"]``.

    Parameters
    ----------
    box : tuple of float, optional
        Box length, width and height (mm). Default is ``(50, 10, 15)``.
    length : float, optional
        Fiber length (mm). Default is 12.5 mm.
    diameter : float, optional
        Round-fiber diameter (mm), used when ``width`` and ``thickness``
        are omitted. Default is 1 mm.
    width : float, optional
        In-plane width of the cross-section (mm).
    thickness : float, optional
        Vertical thickness of the cross-section (mm).
    volume_fraction : float, optional
        Maximum fiber volume over box volume. Default is 0.6.
    sweeps : int, optional
        In-plane nudges tried when a fiber does not fit at first. Default is 5.
    seed : int, optional
        Random seed. Default is 0.
    bend : bool, optional
        Drape each tow over the mat instead of keeping it straight.
        Default is False.
    section : {"ellipse", "rectangle"}, optional
        Cross-section used for the vertical clearance. ``ellipse`` thins
        toward the edge. ``rectangle`` keeps the full thickness across the
        whole width. Default is ``ellipse``.

    Returns
    -------
    mat : Mat
        Sedimented fibers. ``mat.attrs["box"]`` is the box size and
        ``mat.attrs["periodic"]`` is True.

    """
    lx, ly, lz = (float(v) for v in box)
    length = float(length)
    width, thickness = _section(diameter, width, thickness)
    if min(lx, ly, lz, length, width, thickness) <= 0:
        raise ValueError("Box size and fiber dimensions must be positive.")
    if not 0 < volume_fraction <= 1:
        raise ValueError("Volume fraction must be between 0 and 1.")

    if section not in ("ellipse", "rectangle"):
        raise ValueError("section must be 'ellipse' or 'rectangle'.")
    if section == "rectangle":
        fiber_volume = length * width * thickness
    else:
        fiber_volume = 0.25 * np.pi * length * width * thickness
    n_max = int(round(volume_fraction * lx * ly * lz / fiber_volume))
    half = 0.5 * thickness
    rng = np.random.default_rng(seed)

    xy = np.zeros((n_max, 2))
    z = np.zeros(n_max)
    theta = np.zeros(n_max)
    grid = _Grid(max(width, thickness), lx, ly)
    hmap = _HeightMap(lx, ly, min(0.4, max(width / 10.0, 0.2))) if bend else None
    centerlines = []
    placed = 0
    misses = 0
    # A taller box has more open pockets, so keep trying in proportion to it.
    miss_limit = max(200, n_max // 2)
    while placed < n_max and misses < miss_limit:
        ang = float(rng.uniform(0.0, np.pi))
        if _hits_own_image(ang, length, width, lx, ly):
            continue
        x = float(rng.uniform(-0.5 * lx, 0.5 * lx))
        y = float(rng.uniform(-0.5 * ly, 0.5 * ly))
        line = None
        if bend:
            line = _drape_centerline(
                x, y, ang, length, width, thickness, hmap,
            )
            if float(line[:, 2].max()) + half > lz + 1e-8:
                misses += 1
                continue
            _paint_tow(x, y, ang, length, width, thickness, line, hmap)
            pose = (x, y, float(line[:, 2].mean()))
        else:
            pose = _fit(
                x, y, ang, xy[:placed], z[:placed], theta[:placed],
                grid, lx, ly, lz, length, width, thickness, half, sweeps,
                section,
            )
        if pose is None:
            misses += 1
            continue
        x, y, height = pose
        xy[placed] = (x, y)
        z[placed] = height
        theta[placed] = ang
        if not bend:
            grid.add(placed, x, y)
        else:
            centerlines.append(line)
        placed += 1
        misses = 0

    xy, z, theta = xy[:placed], z[:placed], theta[:placed]
    if placed == 0:
        mat = Mat(0, size=max(lx, ly))
        mat.attrs["box"] = (lx, ly, lz)
        mat.attrs["periodic"] = True
        mat.attrs["volume_fraction"] = 0.0
        return mat

    frame = pd.DataFrame(
        np.column_stack((
            np.full(placed, length),
            np.full(placed, width),
            np.full(placed, thickness),
            xy[:, 0], xy[:, 1], z,
            np.cos(theta), np.sin(theta), np.zeros(placed),
            np.ones(placed), np.full(placed, np.inf),
        )),
        columns=list("lbhxyzuvwGE"),
    )
    frame.attrs["n"] = placed
    frame.attrs["size"] = float(max(lx, ly))
    frame.attrs["box"] = (lx, ly, lz)
    frame.attrs["periodic"] = True
    frame.attrs["volume_fraction"] = placed * fiber_volume / (lx * ly * lz)
    frame.attrs["bent"] = bool(bend)
    frame.attrs["section"] = section
    if bend:
        frame.attrs["centerlines"] = centerlines
    return Mat(frame)


def max_penetration(mat):
    """Largest overlap between any two fibers, in millimetres.

    Periodic in the plane when ``mat.attrs["periodic"]`` is true. The box
    is ``mat.attrs["box"]`` when that is set, and a square of ``size``
    otherwise.
    """
    if len(mat) == 0:
        return 0.0
    xy = mat[["x", "y"]].to_numpy(dtype=float)
    z = mat["z"].to_numpy(dtype=float)
    theta = np.arctan2(mat["v"].to_numpy(dtype=float), mat["u"].to_numpy(dtype=float))
    length = float(np.max(mat["l"].to_numpy(dtype=float)))
    width = float(np.max(mat["b"].to_numpy(dtype=float)))
    thickness = float(np.max(mat["h"].to_numpy(dtype=float)))
    if "box" in mat.attrs:
        lx, ly, _lz = mat.attrs["box"]
    else:
        lx = ly = float(mat.attrs["size"])
    periodic = bool(mat.attrs.get("periodic", False))
    grid = _Grid(max(width, thickness), lx, ly)
    for i in range(len(mat)):
        grid.add(i, xy[i, 0], xy[i, 1])
    worst = 0.0
    for fiber in range(len(mat)):
        idx, origin = _candidates(
            xy[fiber, 0], xy[fiber, 1], theta[fiber],
            xy, theta, grid, lx, ly, length, width, periodic,
        )
        own = idx == fiber
        idx, origin = idx[~own], origin[~own]
        if len(idx) == 0:
            continue
        gap, _mtv = _clearance(
            xy[fiber], theta[fiber], origin, theta[idx], length, width, thickness,
            mat.attrs.get("section", "ellipse"),
        )
        pen = gap - np.abs(z[idx] - z[fiber])
        worst = max(worst, float(np.max(pen, initial=0.0)))
    return max(0.0, worst)


class _Grid:
    """Periodic bins of fiber centers. The cell size is one diameter."""

    def __init__(self, cell, lx, ly):
        self.cell = float(cell)
        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = max(1, int(np.ceil(self.lx / self.cell)))
        self.ny = max(1, int(np.ceil(self.ly / self.cell)))
        self.bins = {}

    def add(self, index, x, y):
        key = self._key(x, y)
        self.bins.setdefault(key, []).append(int(index))

    def _key(self, x, y):
        ux = (x + 0.5 * self.lx) % self.lx
        uy = (y + 0.5 * self.ly) % self.ly
        return (min(int(ux / self.cell), self.nx - 1),
                min(int(uy / self.cell), self.ny - 1))

    def nearby(self, x, y, reach_x, reach_y):
        """Fiber indices whose centers can meet ``(x, y)``, and the image shift."""
        ix0 = int(np.floor((x - reach_x + 0.5 * self.lx) / self.cell)) - 1
        ix1 = int(np.floor((x + reach_x + 0.5 * self.lx) / self.cell)) + 1
        iy0 = int(np.floor((y - reach_y + 0.5 * self.ly) / self.cell)) - 1
        iy1 = int(np.floor((y + reach_y + 0.5 * self.ly) / self.cell)) + 1
        found = {}
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bucket = self.bins.get((ix % self.nx, iy % self.ny))
                if not bucket:
                    continue
                shift = (np.floor(ix / self.nx) * self.lx,
                         np.floor(iy / self.ny) * self.ly)
                for index in bucket:
                    previous = found.get(index)
                    if previous is None:
                        found[index] = [shift]
                    elif shift not in previous:
                        previous.append(shift)
        indices = []
        shifts = []
        for index, images in found.items():
            for shift in images:
                indices.append(index)
                shifts.append(shift)
        if not indices:
            return np.empty(0, dtype=int), np.zeros((0, 2))
        return np.asarray(indices, dtype=int), np.asarray(shifts, dtype=float)


class _HeightMap:
    """Top surface of the mat, sampled on a periodic grid."""

    def __init__(self, lx, ly, pitch):
        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = max(1, int(np.ceil(self.lx / pitch)))
        self.ny = max(1, int(np.ceil(self.ly / pitch)))
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.top = np.zeros((self.ny, self.nx))

    def sample(self, x, y):
        """Highest painted corner near each point, so a tow does not cut the mat."""
        u = np.mod((np.asarray(x, dtype=float) + 0.5 * self.lx) / self.dx, self.nx)
        v = np.mod((np.asarray(y, dtype=float) + 0.5 * self.ly) / self.dy, self.ny)
        u0 = np.floor(u).astype(int) % self.nx
        v0 = np.floor(v).astype(int) % self.ny
        u1 = (u0 + 1) % self.nx
        v1 = (v0 + 1) % self.ny
        return np.maximum(
            np.maximum(self.top[v0, u0], self.top[v0, u1]),
            np.maximum(self.top[v1, u0], self.top[v1, u1]),
        )

    def paint(self, x, y, z):
        u = np.mod((np.asarray(x, dtype=float) + 0.5 * self.lx) / self.dx, self.nx)
        v = np.mod((np.asarray(y, dtype=float) + 0.5 * self.ly) / self.dy, self.ny)
        ui = np.floor(u).astype(int) % self.nx
        vi = np.floor(v).astype(int) % self.ny
        np.maximum.at(self.top, (vi, ui), np.asarray(z, dtype=float))


def _half_thickness(lateral, width, thickness):
    clipped = np.clip(2.0 * np.asarray(lateral, dtype=float) / width, -1.0, 1.0)
    return 0.5 * thickness * np.sqrt(np.maximum(0.0, 1.0 - clipped * clipped))


def _drape_centerline(x, y, theta, length, width, thickness, hmap, spacing=0.5):
    """Centerline that rests on ``hmap``. Straight in the plane, bent in height."""
    n_along = max(8, int(np.ceil(length / spacing)) + 1)
    n_across = max(5, int(np.ceil(width / spacing)) + 1)
    station = np.linspace(-0.5 * length, 0.5 * length, n_along)
    lateral = np.linspace(-0.5 * width, 0.5 * width, n_across)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    half = _half_thickness(lateral, width, thickness)
    height = np.full(n_along, 0.5 * thickness)
    for i, s in enumerate(station):
        origin = np.array([x, y]) + s * direction
        point = origin + lateral[:, None] * normal
        support = hmap.sample(point[:, 0], point[:, 1])
        height[i] = max(height[i], float(np.max(support + half)))
    return np.column_stack((
        x + station * direction[0],
        y + station * direction[1],
        height,
    ))


def _paint_tow(x, y, theta, length, width, thickness, centerline, hmap, spacing=0.35):
    """Write the top surface of a draped tow onto the height map."""
    n_along = max(8, int(np.ceil(length / spacing)) + 1)
    n_across = max(5, int(np.ceil(width / spacing)) + 1)
    station = np.linspace(-0.5 * length, 0.5 * length, n_along)
    lateral = np.linspace(-0.5 * width, 0.5 * width, n_across)
    known = np.linspace(-0.5 * length, 0.5 * length, len(centerline))
    height = np.interp(station, known, centerline[:, 2])
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    ss, tt = np.meshgrid(station, lateral, indexing="ij")
    top = height[:, None] + _half_thickness(tt, width, thickness)
    point_x = x + ss * direction[0] + tt * normal[0]
    point_y = y + ss * direction[1] + tt * normal[1]
    hmap.paint(point_x.ravel(), point_y.ravel(), top.ravel())


def _section(diameter, width, thickness):
    """In-plane width and vertical thickness. A round fiber uses ``diameter``."""
    if width is None and thickness is None:
        return float(diameter), float(diameter)
    if width is None or thickness is None:
        raise ValueError("Set both width and thickness, or neither.")
    return float(width), float(thickness)


def _hits_own_image(theta, length, width, lx, ly):
    """True when a fiber's planform cuts one of its own periodic copies."""
    direction = np.array([np.cos(theta), np.sin(theta)])
    shifts = [(sx, sy)
              for sx in (-lx, 0.0, lx)
              for sy in (-ly, 0.0, ly)
              if sx != 0.0 or sy != 0.0]
    origin = np.asarray(shifts, dtype=float)
    direc = np.repeat(direction[None], len(shifts), 0)
    _s, _t, dist = _segment_closest_xy(
        np.zeros((len(shifts), 2)), direc, np.full(len(shifts), 0.5 * length),
        origin, direc, np.full(len(shifts), 0.5 * length),
    )
    return bool(np.any(dist < width - 1e-8))


def _candidates(x, y, theta, xy, thetas, grid, lx, ly, length, width, periodic):
    # Centers of two touching fibers can be a full length apart.
    reach = length + width
    idx, shifts = grid.nearby(x, y, reach, reach)
    if len(idx) == 0:
        return idx, np.zeros((0, 2))
    origin = xy[idx] + shifts
    if not periodic:
        # Drop images that actually leave the cell. Centers are stored inside.
        keep = np.max(np.abs(shifts), axis=1) <= 1e-8
        idx, origin = idx[keep], origin[keep]
    return idx, origin


def subdivide(mat):
    """Replace each tow by parallel round fibers of diameter equal to its thickness.

    The fibers touch each other and stay inside the tow's rectangle. A tow
    of width ``b`` and thickness ``h`` becomes ``floor(b / h)`` fibers.
    """
    if len(mat) == 0:
        return mat

    rows = []
    tow_id = []
    for label, tow in mat.iterrows():
        diameter = float(tow.h)
        count = max(1, int(np.floor(float(tow.b) / diameter + 1e-8)))
        offsets = (np.arange(count) - 0.5 * (count - 1)) * diameter
        normal = np.array([-float(tow.v), float(tow.u)])
        for offset in offsets:
            place = np.array([float(tow.x), float(tow.y)]) + offset * normal
            rows.append((
                float(tow.l), diameter, diameter,
                place[0], place[1], float(tow.z),
                float(tow.u), float(tow.v), float(tow.w),
                float(tow.G), float(tow.E),
            ))
            tow_id.append(int(label))

    data = np.asarray(rows, dtype=float)
    frame = pd.DataFrame(data, columns=list("lbhxyzuvwGE"))
    frame.attrs["n"] = len(frame)
    span = float(np.max(np.abs(data[:, 3:5]))) if len(data) else 0.0
    frame.attrs["size"] = max(float(mat.attrs.get("size", 0.0)), 2.0 * span + 1.0)
    if "box" in mat.attrs:
        frame.attrs["box"] = mat.attrs["box"]
    frame.attrs["periodic"] = bool(mat.attrs.get("periodic", False))
    frame.attrs["section"] = "ellipse"
    frame.attrs["tow"] = np.asarray(tow_id, dtype=int)
    if "box" in mat.attrs:
        lx, ly, lz = mat.attrs["box"]
        volume = float(np.sum(0.25 * np.pi * frame.l * frame.b * frame.h))
        frame.attrs["volume_fraction"] = volume / (lx * ly * lz)
    return Mat(frame)


def roll_ring(points, box, scale=1.0):
    """Bend flat-stack coordinates into a ring about the y-axis.

    ``box`` is ``(length, width, height)``: x along the length, y along the
    width, and z up from the bottom. The mid-surface becomes a cylinder of
    radius ``scale`` times the closed-ring radius (the box length over
    2π). y stays the cylinder axis. A fiber off the mid-surface moves onto
    its own radius, so the line follows the roll instead of staying straight.
    With ``scale`` above 1 the sheet covers less than a full turn.
    """
    points = np.asarray(points, dtype=float)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive.")
    lx, _ly, lz = (float(v) for v in box)
    radius = scale * lx / (2.0 * np.pi)
    if radius <= 0.5 * lz:
        raise ValueError("The stack is too thick to roll into a ring about the y-axis.")
    theta = points[:, 0] / radius
    radial = radius + (points[:, 2] - 0.5 * lz)
    return np.column_stack((
        radial * np.sin(theta),
        points[:, 1],
        radial * np.cos(theta),
    ))


def line_mesh(mat, n=10, roll=False, scale=1.0):
    """Split each fiber into ``n`` colinear line elements sharing their nodes.

    A fiber of length ``l`` becomes ``n`` segments of length ``l / n``. The
    first and last nodes are the fiber ends. Consecutive elements share the
    node between them, and fibers do not share nodes with each other.

    With ``roll=True`` those nodes are then bent into a ring about the
    y-axis. The elements stay straight chords of that curve.

    Parameters
    ----------
    mat : Mat
        Fibers to discretize. Each row is one straight fiber.
    n : int, optional
        Number of line elements along each fiber. Default is 10.
    roll : bool, optional
        Bend the stack into a ring about the y-axis. Needs
        ``mat.attrs["box"]``. Default is False.
    scale : float, optional
        Mid-surface radius as a multiple of the closed-ring radius.
        Default is 1, a full turn. 3 is three times that radius.

    Returns
    -------
    meshio.Mesh
        Line mesh. Cell data holds the fiber index, its diameter and its
        in-plane angle in degrees. A tow index is included when the mat
        carries one.

    """
    import meshio

    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if len(mat) == 0:
        return meshio.Mesh(
            np.zeros((0, 3)),
            [("line", np.zeros((0, 2), dtype=np.int64))],
        )

    count = len(mat)
    centers = mat[["x", "y", "z"]].to_numpy(dtype=float)
    direction = np.array(mat[["u", "v", "w"]].to_numpy(dtype=float), copy=True)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = mat["l"].to_numpy(dtype=float)
    stations = np.linspace(-0.5, 0.5, n + 1)
    points = (
        centers[:, None, :]
        + stations[None, :, None] * length[:, None, None] * direction[:, None, :]
    ).reshape(-1, 3)
    if roll:
        if "box" not in mat.attrs:
            raise ValueError("Rolling into a ring needs mat.attrs['box'].")
        points = roll_ring(points, mat.attrs["box"], scale=scale)

    nodes = n + 1
    base = np.arange(count, dtype=np.int64)[:, None] * nodes
    local = np.arange(n, dtype=np.int64)
    cells = np.stack((base + local, base + local + 1), axis=-1).reshape(-1, 2)
    fiber = np.repeat(np.arange(count, dtype=np.int64), n)
    cell_data = {
        "fiber": [fiber],
        "diameter": [np.repeat(mat["h"].to_numpy(dtype=float), n)],
        "angle": [np.repeat(np.degrees(np.mod(
            np.arctan2(direction[:, 1], direction[:, 0]), np.pi,
        )), n)],
    }
    tow = mat.attrs.get("tow")
    if tow is not None:
        cell_data["tow"] = [np.repeat(np.asarray(tow, dtype=np.int64), n)]
    return meshio.Mesh(points, [("line", cells)], cell_data=cell_data)


def write_lines(mat, path, n=10, roll=False, scale=1.0):
    """Write each fiber as ``n`` colinear line elements using meshio.

    The file format follows the extension of ``path`` (for example ``.vtk``
    or ``.xdmf``). ``roll=True`` bends the stack into a ring about the y-axis
    before writing. ``scale`` multiplies that ring's mid-surface radius.
    """
    import meshio

    mesh = line_mesh(mat, n=n, roll=roll, scale=scale)
    meshio.write(path, mesh)
    return mesh


def _clearance(center, theta, origin, other_theta, length, width, thickness,
               section="ellipse"):
    """Vertical gap each neighbor needs, and the in-plane push off that neighbor."""
    m = len(origin)
    direction = np.array([np.cos(theta), np.sin(theta)])
    other = np.column_stack((np.cos(other_theta), np.sin(other_theta)))
    wide = np.full(m, width)
    sat = _rectangle_mtv(
        origin, other, np.full(m, length), wide,
        np.repeat(np.asarray(center, dtype=float)[None], m, 0),
        np.repeat(direction[None], m, 0),
        np.full(m, length), wide,
    )
    _s, _t, dist = _segment_closest_xy(
        np.repeat(np.asarray(center, dtype=float)[None], m, 0),
        np.repeat(direction[None], m, 0),
        np.full(m, 0.5 * length),
        origin, other, np.full(m, 0.5 * length),
    )
    overlap = np.isfinite(sat[:, 0])
    if section == "rectangle":
        # The full thickness stands across the whole width.
        gap = np.where(overlap, thickness, 0.0)
    else:
        thick = np.full(m, thickness)
        nested = _nested_gap(dist, wide, thick, wide, thick)
        # A corner the lateral sample misses still needs a full thickness.
        gap = np.where(overlap, np.where(nested > 1e-8, nested, thickness), 0.0)
    return gap, sat


def _fit(x, y, theta, xy, z, thetas, grid, lx, ly, lz, length, width, thickness, half, sweeps,
         section="ellipse"):
    """Lowest pose at this angle, after a few slides if the stack is too tall."""
    for _attempt in range(sweeps + 1):
        x, y = _wrap(x, y, lx, ly)
        idx, origin = _candidates(
            x, y, theta, xy, thetas, grid, lx, ly, length, width, True,
        )
        if len(idx) == 0:
            return _wrap(x, y, lx, ly) + (half,)
        gap, mtv = _clearance(
            (x, y), theta, origin, thetas[idx], length, width, thickness, section,
        )
        blocks = [(z[j] - g, z[j] + g) for j, g in zip(idx, gap) if g > 1e-8]
        height = _lowest_free(half, blocks)
        if height <= lz - half + 1e-9:
            return x, y, float(height)
        blocking = np.where(gap > 1e-8, z[idx] + gap, -np.inf)
        blame = int(np.argmax(blocking))
        push = mtv[blame]
        if not np.isfinite(push[0]):
            return None
        x += 0.85 * float(push[0])
        y += 0.85 * float(push[1])
        x, y = _wrap(x, y, lx, ly)
    return None


def _wrap(x, y, lx, ly):
    x = (x + 0.5 * lx) % lx - 0.5 * lx
    y = (y + 0.5 * ly) % ly - 0.5 * ly
    return float(x), float(y)
