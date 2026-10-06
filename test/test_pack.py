import time

import numpy as np

import pandas as pd

from fibermat import Mat
from fibermat.pack import (
    _clearance, line_mesh, max_penetration, pack, subdivide, write_lines,
)


def test_pack_drops_fibers_like_a_random_mat():
    """Fibers fall at random angles and rest on the mat already in the box."""
    started = time.perf_counter()
    mat = pack(box=(50.0, 10.0, 5.0), length=12.5, diameter=1.0,
               volume_fraction=0.6, seed=1)
    elapsed = time.perf_counter() - started

    assert len(mat) > 20
    assert mat.attrs["periodic"] is True
    fraction = mat.attrs["volume_fraction"]
    assert fraction > 0.15
    # In-plane orientation is spread out, not lined up with the box.
    along = float(np.mean(mat.u.to_numpy() ** 2))
    assert 0.35 < along < 0.70
    assert np.ptp(np.arctan2(mat.v.to_numpy(), mat.u.to_numpy())) > 2.0
    lx, ly, lz = mat.attrs["box"]
    assert mat.z.min() >= 0.5 - 1e-8
    assert mat.z.max() <= lz - 0.5 + 1e-6
    assert mat.z.max() > 0.5 * lz
    assert max_penetration(mat) <= 1e-4
    assert Mat.check(mat)
    assert elapsed < 30.0


def test_tow_bends_down_between_crossings():
    """A tow rests on the floor except where another tow passes underneath."""
    from fibermat.pack import _HeightMap, _drape_centerline, _paint_tow

    surface = _HeightMap(20.0, 20.0, pitch=0.25)
    lower = _drape_centerline(0.0, 0.0, 0.0, 10.0, 2.0, 0.2, surface)
    assert np.allclose(lower[:, 2], 0.1)
    _paint_tow(0.0, 0.0, 0.0, 10.0, 2.0, 0.2, lower, surface)
    upper = _drape_centerline(0.0, 0.0, 0.5 * np.pi, 10.0, 2.0, 0.2, surface)
    assert upper[0, 2] < 0.15
    assert upper[len(upper) // 2, 2] > upper[0, 2] + 0.05


def test_pack_flat_tows_stack_on_their_thickness():
    """A wide thin tow only needs its thickness of clearance at a crossing."""
    mat = pack(
        box=(36.0, 36.0, 1.0), length=12.0, width=4.0, thickness=0.2,
        volume_fraction=0.45, sweeps=3, seed=1,
    )
    assert float(mat.b.iloc[0]) == 4.0
    assert float(mat.h.iloc[0]) == 0.2
    assert mat.attrs["volume_fraction"] > 0.2
    along = float(np.mean(mat.u.to_numpy() ** 2))
    assert 0.35 < along < 0.70
    assert mat.z.min() >= 0.1 - 1e-8
    assert mat.z.max() <= 1.0 - 0.1 + 1e-6
    assert max_penetration(mat) <= 1e-4
    assert Mat.check(mat)


def test_rectangle_keeps_full_thickness_at_a_glancing_overlap():
    """A rectangular tow stays full height out to its edge."""
    gap, _mtv = _clearance(
        (0.0, 0.0), 0.0,
        np.array([[0.0, 3.5]]), np.array([0.0]),
        10.0, 4.0, 0.2,
        section="rectangle",
    )
    assert abs(gap[0] - 0.2) < 1e-8


def test_subdivide_lays_touching_fibers_across_the_tow():
    """Each tow becomes a row of round fibers, one tow-thickness apart."""
    frame = pd.DataFrame(
        [[10.0, 1.0, 0.25, 0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 1.0, np.inf]],
        columns=list("lbhxyzuvwGE"),
    )
    frame.attrs["n"] = 1
    frame.attrs["size"] = 20.0
    frame.attrs["box"] = (20.0, 20.0, 1.0)
    fibers = subdivide(Mat(frame))
    assert len(fibers) == 4
    assert np.allclose(fibers.b, 0.25)
    assert np.allclose(fibers.h, 0.25)
    assert np.allclose(np.diff(np.sort(fibers.y.to_numpy())), 0.25)
    assert np.allclose(fibers.u, 1.0)
    assert Mat.check(fibers)


def test_line_mesh_splits_each_fiber_into_connected_segments(tmp_path):
    """Ten colinear line elements run from one fiber end to the other."""
    frame = pd.DataFrame(
        [
            [10.0, 0.2, 0.2, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, np.inf],
            [4.0, 0.2, 0.2, 0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, np.inf],
        ],
        columns=list("lbhxyzuvwGE"),
    )
    frame.attrs["n"] = 2
    frame.attrs["size"] = 20.0
    mesh = line_mesh(Mat(frame))
    assert mesh.cells[0].type == "line"
    assert len(mesh.points) == 2 * 11
    assert len(mesh.cells[0].data) == 20
    first = mesh.points[:11]
    assert np.allclose(first[:, 0], np.linspace(-5.0, 5.0, 11))
    assert np.allclose(first[:, 1], 0.0)
    assert np.allclose(first[:, 2], 1.0)
    # Consecutive elements share a node. The two fibers do not.
    pairs = mesh.cells[0].data
    assert pairs[0, 1] == pairs[1, 0]
    assert pairs[9, 1] == 10
    assert pairs[10, 0] == 11
    path = tmp_path / "fibers.vtk"
    written = write_lines(Mat(frame), path, n=4)
    assert path.stat().st_size > 0
    assert len(written.cells[0].data) == 8


def test_roll_bends_fibers_into_a_ring_about_y():
    """A fiber along the box length becomes an arc about the y-axis."""
    lx, lz = 20.0, 2.0
    radius = lx / (2.0 * np.pi)
    frame = pd.DataFrame(
        [
            [4.0, 0.2, 0.2, 0.0, 1.0, 0.5 * lz, 1.0, 0.0, 0.0, 1.0, np.inf],
            [4.0, 0.2, 0.2, 0.0, 0.0, 0.5 * lz, 0.0, 1.0, 0.0, 1.0, np.inf],
        ],
        columns=list("lbhxyzuvwGE"),
    )
    frame.attrs["n"] = 2
    frame.attrs["size"] = 40.0
    frame.attrs["box"] = (lx, 10.0, lz)
    mesh = line_mesh(Mat(frame), n=8, roll=True)
    arc, axial = mesh.points[:9], mesh.points[9:]
    assert np.allclose(np.hypot(arc[:, 0], arc[:, 2]), radius)
    assert np.allclose(arc[:, 1], 1.0)
    chord = arc[-1] - arc[0]
    rise = arc[len(arc) // 2] - arc[0]
    assert np.linalg.norm(np.cross(chord, rise)) > 0.05
    assert np.allclose(axial[:, 0], 0.0)
    assert np.allclose(axial[:, 2], radius)
    assert np.allclose(axial[:, 1], np.linspace(-2.0, 2.0, 9))
    wider = line_mesh(Mat(frame), n=8, roll=True, scale=3)
    assert np.allclose(np.hypot(wider.points[:9, 0], wider.points[:9, 2]), 3 * radius)
