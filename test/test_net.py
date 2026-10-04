import numpy as np
import pandas as pd

from fibermat import *


def equal(df1, df2):
    """
    Test the equality between two DataFrames.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame.
    df2 : pandas.DataFrame
        Second DataFrame.

    """
    assert np.allclose(df1.values, df2.values)
    assert np.all(df1.index == df2.index)
    assert np.all(df1.columns == df2.columns)
    assert np.all(df1.dtypes == df2.dtypes)
    assert np.all(df1.attrs == df2.attrs)


def segment_distance(c1, d1, r1, c2, d2, r2):
    """In-plane distance between two finite centerlines.

    Independent of the network builder, so a missed contact fails the
    stacking test instead of being hidden by the same implementation.
    """
    w0 = c1 - c2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b
    if a <= 1e-16 and c <= 1e-16:
        s = t = 0.0
    elif a <= 1e-16:
        s, t = 0.0, np.clip(e / c, -r2, r2)
    elif c <= 1e-16:
        s, t = np.clip(-d / a, -r1, r1), 0.0
    elif abs(denom) <= 1e-12 * max(a * c, 1e-30):
        s = np.clip(-d / a, -r1, r1)
        t = np.clip((b * s + e) / c, -r2, r2)
        s = np.clip((b * t - d) / a, -r1, r1)
    else:
        s = np.clip((b * e - c * d) / denom, -r1, r1)
        t = np.clip((b * s + e) / c, -r2, r2)
        s = np.clip((b * t - d) / a, -r1, r1)
        t = np.clip((b * s + e) / c, -r2, r2)
    return np.linalg.norm((c1 + s * d1) - (c2 + t * d2))


################################################################################
# Tests
################################################################################

def test_emptyNet():
    """
    Test the initialization of an empty `Net` object.

    """
    net = pd.DataFrame(data=[], index=[], columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"], dtype=float)
    net[[*"AB"]] = net[[*"AB"]].astype(int)
    net.attrs = dict(n=0, size=50., periodic=True)
    net.flags.mat = Mat()
    equal(Net(), net)

    assert Net().check()
    assert Net(None).check()
    assert Net(net).check()


def test_Net():
    """
    Test the initialization of a `Net` object.

    """
    data = np.array([[  0.        ,   0.        ,  12.5       , -12.5       ,
         11.67431009,   6.16069633, -19.08627871,  -6.7929597 ,
         23.01180748, -19.08627871],
       [  0.        ,   3.        ,  -8.06427839,  -5.89623413,
         -3.51633299,  20.02193397, -19.08627871,  -3.51633299,
         20.02193397,  -1.92603189],
       [  0.        ,   4.        ,  -7.77355184,   8.74105637,
         -3.30157597,  19.82597135, -19.08627871,  -3.30157597,
        -30.17402865,   1.09241609],
       [  0.        ,   9.        ,  -8.03317106,   2.77512029,
         -3.49335429,  20.00096624, -19.08627871,  -3.49335429,
         20.00096624,  23.93091711],
       [  1.        ,   1.        ,  12.5       , -12.5       ,
         18.90042823,  10.93024923, -17.83233563,   2.61850841,
         -8.04075725, -17.83233563],
       [  1.        ,   2.        ,   1.38749949,   6.58735173,
         11.66311454,   2.49763646, -17.83233563,  11.66311454,
          2.49763646,  -4.266903  ],
       [  2.        ,   2.        ,  12.5       , -12.5       ,
         17.51974869,   1.68569683,  -4.266903  ,  -7.24341109,
          5.11875928,  -4.266903  ],
       [  3.        ,   3.        ,  12.5       , -12.5       ,
         14.4563865 ,  23.94657208,  -1.92603189,  -9.9680682 ,
         18.61309175,  -1.92603189],
       [  3.        ,   4.        ,  -5.69092012,   8.50087389,
         -3.31574568,  20.06573549,  -1.92603189,  -3.31574568,
        -29.93426451,   1.09241609],
       [  3.        ,   9.        ,  -5.81002817,   2.84791988,
         -3.43211164,  20.04032508,  -1.92603189,  -3.43211164,
         20.04032508,  23.93091711],
       [  4.        ,   4.        ,  12.5       , -12.5       ,
         -3.07981474, -33.92642511,   1.09241609,  -4.55470532,
         -8.96996907,   1.09241609],
       [  4.        ,   5.        ,   3.42731873, -11.69966786,
         -3.61506323, -24.86954626,   1.09241609,  -3.61506323,
        -24.86954626,   6.99605107],
       [  4.        ,   6.        ,   2.7422018 ,  -0.56981195,
         -3.65548213, -24.18562263,   1.09241609,  -3.65548213,
        -24.18562263,  14.02645881],
       [  4.        ,   9.        ,   8.45322302,   2.98290327,
         -3.31855687, -29.88669664,   1.09241609,  -3.31855687,
         20.11330336,  23.93091711],
       [  5.        ,   5.        ,  12.5       , -12.5       ,
         18.95077253, -16.12843756,   6.99605107,  -4.36136122,
        -25.15863247,   6.99605107],
       [  5.        ,   6.        ,   1.5775133 ,  12.5       ,
          8.7657137 , -20.07372491,   6.99605107,   8.61223773,
        -19.67751506,  14.02645881],
       [  6.        ,   6.        ,  12.5       , -12.5       ,
          8.61223773, -19.67751506,  14.02645881, -14.85351661,
        -28.30064519,  14.02645881],
       [  7.        ,   7.        ,  12.5       , -12.5       ,
         31.2546391 ,  21.12039082,  14.95792821,   7.92266098,
         12.14159373,  14.95792821],
       [  7.        ,   8.        ,   4.69521175,   4.4792103 ,
         23.97059315,  18.31728642,  14.95792821,  23.97059315,
         18.31728642,  22.23344585],
       [  8.        ,   8.        ,  12.5       , -12.5       ,
         25.38066581,  26.21315638,  22.23344585,  20.98561024,
          1.60251871,  22.23344585],
       [  9.        ,   9.        ,  12.5       , -12.5       ,
          4.68770015,  25.25868758,  23.93091711, -16.34354827,
         11.74252725,  23.93091711]])
    index = np.arange(len(data))

    net = pd.DataFrame(data=data, index=index, columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"])
    net[[*"AB"]] = net[[*"AB"]].astype(int)
    net.attrs = dict(n=10, size=50., periodic=True)
    net.flags.mat = Mat(10)
    equal(Net(Mat(10)), net)

    assert Net(net).check()


def test_emptyStack():
    """
    Test the initialization of an empty `Stack` object.

    """
    stack = pd.DataFrame(data=[], index=[], columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"], dtype=float)
    stack[[*"AB"]] = stack[[*"AB"]].astype(int)
    stack.attrs = dict(n=0, size=50., periodic=True, threshold=None)
    stack.flags.mat = Mat()
    equal(Stack(), stack)

    assert Stack().check()
    assert Stack(None).check()
    assert Stack(stack).check()


def test_Stack():
    """
    Test the initialization of a `Stack` object.

    """
    data = np.array([[  0.        ,   0.        ,  12.5       , -12.5       ,
         11.67431009,   6.16069633,   0.5       ,  -6.7929597 ,
         23.01180748,   0.5       ],
       [  0.        ,   3.        ,  -8.06427839,  -5.89623413,
         -3.51633299,  20.02193397,   0.5       ,  -3.51633299,
         20.02193397,   1.5       ],
       [  0.        ,   4.        ,  -7.77355184,   8.74105637,
         -3.30157597,  19.82597135,   0.5       ,  -3.30157597,
        -30.17402865,   2.5       ],
       [  0.        ,   9.        ,  -8.03317106,   2.77512029,
         -3.49335429,  20.00096624,   0.5       ,  -3.49335429,
         20.00096624,   3.5       ],
       [  1.        ,   1.        ,  12.5       , -12.5       ,
         18.90042823,  10.93024923,   0.5       ,   2.61850841,
         -8.04075725,   0.5       ],
       [  1.        ,   2.        ,   1.38749949,   6.58735173,
         11.66311454,   2.49763646,   0.5       ,  11.66311454,
          2.49763646,   1.5       ],
       [  2.        ,   2.        ,  12.5       , -12.5       ,
         17.51974869,   1.68569683,   1.5       ,  -7.24341109,
          5.11875928,   1.5       ],
       [  3.        ,   3.        ,  12.5       , -12.5       ,
         14.4563865 ,  23.94657208,   1.5       ,  -9.9680682 ,
         18.61309175,   1.5       ],
       [  3.        ,   4.        ,  -5.69092012,   8.50087389,
         -3.31574568,  20.06573549,   1.5       ,  -3.31574568,
        -29.93426451,   2.5       ],
       [  3.        ,   9.        ,  -5.81002817,   2.84791988,
         -3.43211164,  20.04032508,   1.5       ,  -3.43211164,
         20.04032508,   3.5       ],
       [  4.        ,   4.        ,  12.5       , -12.5       ,
         -3.07981474, -33.92642511,   2.5       ,  -4.55470532,
         -8.96996907,   2.5       ],
       [  4.        ,   5.        ,   3.42731873, -11.69966786,
         -3.61506323, -24.86954626,   2.5       ,  -3.61506323,
        -24.86954626,   3.5       ],
       [  4.        ,   6.        ,   2.7422018 ,  -0.56981195,
         -3.65548213, -24.18562263,   2.5       ,  -3.65548213,
        -24.18562263,   4.5       ],
       [  4.        ,   9.        ,   8.45322302,   2.98290327,
         -3.31855687, -29.88669664,   2.5       ,  -3.31855687,
         20.11330336,   3.5       ],
       [  5.        ,   5.        ,  12.5       , -12.5       ,
         18.95077253, -16.12843756,   3.5       ,  -4.36136122,
        -25.15863247,   3.5       ],
       [  5.        ,   6.        ,   1.5775133 ,  12.5       ,
          8.7657137 , -20.07372491,   3.5       ,   8.61223773,
        -19.67751506,   4.5       ],
       [  6.        ,   6.        ,  12.5       , -12.5       ,
          8.61223773, -19.67751506,   4.5       , -14.85351661,
        -28.30064519,   4.5       ],
       [  7.        ,   7.        ,  12.5       , -12.5       ,
         31.2546391 ,  21.12039082,   0.5       ,   7.92266098,
         12.14159373,   0.5       ],
       [  7.        ,   8.        ,   4.69521175,   4.4792103 ,
         23.97059315,  18.31728642,   0.5       ,  23.97059315,
         18.31728642,   1.5       ],
       [  8.        ,   8.        ,  12.5       , -12.5       ,
         25.38066581,  26.21315638,   1.5       ,  20.98561024,
          1.60251871,   1.5       ],
       [  9.        ,   9.        ,  12.5       , -12.5       ,
          4.68770015,  25.25868758,   3.5       , -16.34354827,
         11.74252725,   3.5       ]])
    index = np.arange(len(data))

    stack = pd.DataFrame(data=data, index=index, columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"])
    stack[[*"AB"]] = stack[[*"AB"]].astype(int)
    stack.attrs = dict(n=10, size=50., periodic=True, threshold=None)
    stack.flags.mat = Mat(10)
    equal(Stack(Net(Mat(10))), stack)

    assert Stack(stack).check()


def test_stack_finite():
    """
    Test the stacking algorithm.

    """
    mat = Mat(100, thickness=0.1, psi=0, seed=None)
    mat.l = np.random.normal(mat.l.mean(), 0.04 * mat.l.mean(), len(mat))
    mat.h = np.random.normal(mat.h.mean(), 0.1 * mat.h.mean(), len(mat))
    net = Net(mat)
    stack = Stack(net)

    # Get material data
    h = mat.h.values
    z = 0.5 * h

    # Stack fibers
    for i, j in stack[[*"AB"]][stack.A < stack.B].values:
        z[j] = max(z[i] + 0.5 * (h[i] + h[j]), z[j])

    if not np.all(mat.z == z):
        raise ValueError("Stacking algorithm error.")


def test_stack_periodic():
    """
    Test the stacking algorithm.

    """
    mat = Mat(100, thickness=0.1, psi=0, seed=None)
    mat.l = np.random.normal(mat.l.mean(), 0.04 * mat.l.mean(), len(mat))
    mat.h = np.random.normal(mat.h.mean(), 0.1 * mat.h.mean(), len(mat))
    net = Net(mat)
    stack = Stack(net)

    # Get material data
    h = mat.h.values
    z = 0.5 * h

    # Stack fibers
    for i, j in stack[[*"AB"]][stack.A < stack.B].values:
        z[j] = max(z[i] + 0.5 * (h[i] + h[j]), z[j])

    if not np.all(mat.z == z):
        raise ValueError("Stacking algorithm error.")


def test_centerline_crossing_keeps_center_abscissa():
    """Perpendicular centerlines still meet at the fiber centers."""
    mat = Mat(2, length=10, width=1, thickness=1, size=50, seed=0)
    mat.loc[0, ["x", "y", "z", "u", "v", "w"]] = [0, 0, 0, 1, 0, 0]
    mat.loc[1, ["x", "y", "z", "u", "v", "w"]] = [0, 0, 1, 0, 1, 0]
    net = Net(mat, periodic=False)
    row = net[(net.A == 0) & (net.B == 1)].iloc[0]
    assert abs(row.sA) < 1e-8
    assert abs(row.sB) < 1e-8


def test_parallel_fibers_within_width_contact():
    """Parallel fibers closer than the summed half-widths must be contacts."""
    mat = Mat(2, length=10, width=1, thickness=1, size=50, seed=0)
    mat.loc[0, ["x", "y", "z", "u", "v", "w"]] = [0, 0, 0, 1, 0, 0]
    mat.loc[1, ["x", "y", "z", "u", "v", "w"]] = [0, 0.4, 0, 1, 0, 0]
    net = Net(mat, periodic=False)
    assert np.any((net.A == 0) & (net.B == 1))


def test_near_parallel_fibers_within_width_contact():
    """A shallow pass inside the width gap is a contact even if centerlines miss."""
    mat = Mat(2, length=10, width=2, thickness=1, size=50, seed=0)
    mat.loc[0, ["x", "y", "z", "u", "v", "w"]] = [0, 0, 0, 1, 0, 0]
    direction = np.array([1.0, 0.05, 0.0])
    direction /= np.linalg.norm(direction)
    mat.loc[1, ["x", "y", "z", "u", "v", "w"]] = [6, 0.3, 0, *direction]
    net = Net(mat, periodic=False)
    assert np.any((net.A == 0) & (net.B == 1))


def test_parallel_fibers_outside_width_do_not_contact():
    """Parallel fibers separated by more than their widths stay disconnected."""
    mat = Mat(2, length=10, width=1, thickness=1, size=50, seed=0)
    mat.loc[0, ["x", "y", "z", "u", "v", "w"]] = [0, 0, 0, 1, 0, 0]
    mat.loc[1, ["x", "y", "z", "u", "v", "w"]] = [0, 3, 0, 1, 0, 0]
    net = Net(mat, periodic=False)
    assert not np.any((net.A == 0) & (net.B == 1))


def test_periodic_corner_and_edge_are_stacked_apart():
    """Images across a missing corner and the right edge must not share a volume."""
    corner = Mat(2, length=4, width=3, thickness=1, size=20, seed=0)
    corner.loc[0, ["x", "y", "z", "u", "v", "w"]] = [9, -9, 0, 1, 0, 0]
    corner.loc[1, ["x", "y", "z", "u", "v", "w"]] = [-9, 9, 0, 1, 0, 0]
    Stack(Net(corner, periodic=True))
    assert abs(corner.z.iloc[0] - corner.z.iloc[1]) >= 1 - 1e-6

    edge = Mat(2, length=6, width=1, thickness=1, size=20, seed=0)
    edge.loc[0, ["x", "y", "z", "u", "v", "w"]] = [9, -9, 0, 1, 0, 0]
    edge.loc[1, ["x", "y", "z", "u", "v", "w"]] = [-9, -9, 0, 1, 0, 0]
    Stack(Net(edge, periodic=True))
    assert abs(edge.z.iloc[0] - edge.z.iloc[1]) >= 1 - 1e-6


def _no_volume_overlap(mat):
    """True when elliptical cross-sections do not occupy the same volume."""
    from fibermat.net import _contact_gaps

    xy = mat[["x", "y"]].to_numpy()
    direction = mat[["u", "v"]].to_numpy()
    horiz = np.linalg.norm(direction, axis=1)
    flat = horiz > 1e-8
    direction = direction.copy()
    direction[flat] /= horiz[flat, None]
    span = mat.l.to_numpy() * horiz
    pairs, gaps = _contact_gaps(
        xy, direction, span, mat.b.to_numpy(), mat.h.to_numpy(),
        mat.attrs["size"], True,
    )
    if len(pairs) == 0:
        return True
    opened = np.abs(mat.z.to_numpy()[pairs[:, 0]] - mat.z.to_numpy()[pairs[:, 1]])
    return bool(np.all(opened + 1e-4 >= gaps))


def test_glancing_overlap_sits_between_full_thicknesses():
    """A touch near the thin edge needs less height than a centerline crossing."""
    from fibermat.net import _nested_gap, _settle_heights

    full = 1.0
    gap = _nested_gap(
        np.array([1.6]), np.array([2.0]), np.array([1.0]),
        np.array([2.0]), np.array([1.0]),
    )
    assert 0.05 < gap[0] < full - 0.05
    z = _settle_heights(
        np.array([1.0, 1.0]), np.array([[0, 1]]), gap, order=[0, 1],
    )
    assert abs(z[0] - 0.5) < 1e-8
    assert abs(abs(z[1] - z[0]) - gap[0]) < 1e-8


def test_slip_clears_a_crossed_bundle():
    """In-plane rearrangement puts a piled bundle into one layer."""
    def bundle():
        mat = Mat(8, length=10, width=1, thickness=1, size=40, psi=0, seed=0)
        for i in range(8):
            ang = -0.5 * np.pi + i * np.pi / 8
            mat.loc[i, ["x", "y", "z", "u", "v", "w"]] = [
                0, 0, 0, np.cos(ang), np.sin(ang), 0,
            ]
        return mat

    rigid = bundle()
    Stack(Net(rigid, periodic=True))
    slipped = bundle()
    slip(slipped, periodic=True, seed=0)
    assert slipped.z.max() < rigid.z.max() - 1
    assert _no_volume_overlap(slipped)
    assert Mat.check(slipped)


def test_slip_keeps_random_orientations():
    """Sliding leaves a random mat random, in a cell much larger than a fiber."""
    size = 30.0
    length = 5.0
    mat = Mat(80, length=length, width=1.0, thickness=0.5, size=size, psi=0, seed=1)
    assert size / length >= 5
    before = np.sort(np.arctan2(mat.v.to_numpy(), mat.u.to_numpy()))
    slip(mat, periodic=True, seed=1)
    after = np.sort(np.arctan2(mat.v.to_numpy(), mat.u.to_numpy()))
    assert np.allclose(before, after)
    assert np.std(after) > 0.4
    assert np.ptp(after) > 1.0
    assert _no_volume_overlap(mat)
    assert Mat.check(mat)


def test_stacked_network_has_no_volume_overlap():
    """After stacking, no periodic image pair occupies the same volume."""
    mat = Mat(30, length=15, width=1.0, thickness=0.25, size=40, psi=0, seed=1)
    Stack(Net(mat, periodic=True))
    center = mat[["x", "y"]].to_numpy()
    direction = mat[["u", "v"]].to_numpy()
    length = mat.l.to_numpy()
    width = mat.b.to_numpy()
    thick = mat.h.to_numpy()
    height = mat.z.to_numpy()
    size = mat.attrs["size"]
    shifts = [(sx * size, sy * size) for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
    for sx, sy in shifts:
        for i in range(len(mat)):
            j_start = i + 1 if sx == 0 and sy == 0 else 0
            for j in range(j_start, len(mat)):
                dist = segment_distance(
                    center[i], direction[i], 0.5 * length[i],
                    center[j] + (sx, sy), direction[j], 0.5 * length[j],
                )
                separated = abs(height[i] - height[j]) + 1e-5 >= 0.5 * (thick[i] + thick[j])
                clear = dist + 1e-5 >= 0.5 * (width[i] + width[j])
                assert separated or clear, (i, j, sx, sy, dist)


################################################################################
# Main
################################################################################

if __name__ == '__main__':

    # Empy net
    test_emptyNet()

    # Net initialization
    test_Net()

    # Empy stack
    test_emptyStack()

    # Stack initialization
    test_Stack()

    # Stacking algorithm
    test_stack_finite()
    test_stack_periodic()
