#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from sklearn.neighbors import KDTree
from tqdm import tqdm

from fibermat import *
from fibermat import Mat, Mesh


def _rotation_from_x(direction):
    """Rotation taking the local +X axis onto ``direction``."""
    src = np.array([1.0, 0.0, 0.0])
    dst = np.asarray(direction, dtype=float)
    dst = dst / np.linalg.norm(dst)
    cross = np.cross(src, dst)
    cosine = float(np.dot(src, dst))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        # 180 degrees about Y: +X goes to -X, thickness (local Z) stays vertical.
        return np.diag([-1.0, 1.0, -1.0])
    axis = cross / sine
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _place_fiber(mesh, center, direction):
    """Rotate a fiber from the local X axis onto ``direction`` and translate it."""
    transform = np.eye(4)
    transform[:3, :3] = _rotation_from_x(direction)
    transform[:3, 3] = np.asarray(center, dtype=float)
    mesh.transform(transform, inplace=True)


def vtk_fiber(length=25., width=1., thickness=1., x=0., y=0., z=0.,
              u=1., v=0., w=0., shear=1., tensile=np.inf, index=None,
              r_resolution=1, theta_resolution=8, z_resolution=20, **_):
    """
    Export a fiber as VTK mesh using `pyvista.CylinderStructured
    <https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.CylinderStructured.html>`_.

    Parameters
    ----------
    length : float, optional
        Fiber length (mm). Default is 25 mm.
    width : float, optional
        Fiber width (mm). Default is 1 mm.
    thickness : float, optional
        Fiber thickness (mm). Default is 1 mm.
    x : float, optional
        Fiber position: X-coordinate (mm). Default is 0 mm.
    y : float, optional
        Fiber position: Y-coordinate (mm). Default is 0 mm.
    z : float, optional
        Fiber position: Z-coordinate (mm). Default is 0 mm.
    u : float, optional
        Fiber orientation: X-component. Default is 1.
    v : float, optional
        Fiber orientation: Y-component. Default is 0.
    w : float, optional
        Fiber orientation: Z-component. Default is 0.
    shear : float, optional
        Shear modulus (MPa). Default is 1 MPa.
    tensile : float, optional
        Tensile modulus (MPa). Default is ∞ MPa.
    index : int, optional
        Fiber label.
    r_resolution : int, optional
        Number of elements along the radius of the fiber. Default is 1.
    theta_resolution : int, optional
        Number of points on the circular face of the fiber. Default is 8.
    z_resolution : int, optional
        Number of points along the length of the fiber. Default is 20.

    Returns
    -------
    pyvista.StructuredGrid
        VTK mesh.

    .. NOTE::
        If `index` is not None, the following fields are added to the VTK mesh:
            - `"fiber"` : fiber index
            - `"lbh"` : fiber dimensions (mm)
            - `"xyz"` : local fiber coordinates (mm)
            - `"uvw"` : fiber orientation vector
            - `"G"` : shear modulus (MPa)
            - `"E"` : tensile modulus (MPa)

    """
    # Create the VTK mesh (cylindrical structured grid)
    # Outer radius is 0.5 so a later scale by (length, width, thickness)
    # gives an elliptical cross-section. The inner radius stays positive
    # because current PyVista rejects a zero radius.
    radius = np.linspace(1e-6, 0.5, r_resolution + 1)
    msh = pv.CylinderStructured(radius=radius,
                                theta_resolution=theta_resolution,
                                z_resolution=z_resolution)

    l, b, h = length, width, thickness

    # Add fields to mesh data
    if index is not None:
        msh["fiber"] = np.full(len(msh.points), index)
        msh["lbh"] = np.tile([l, b, h], (len(msh.points), 1))
        msh["xyz"] = msh.points * np.array([[l, b, h]])
        msh["uvw"] = np.tile([u, v, w], (len(msh.points), 1))
        msh["G"] = np.full(len(msh.points), shear)
        msh["E"] = np.full(len(msh.points), tensile)

    # Transform the mesh (scale, rotate, and translate). The cylinder axis is
    # the local X axis; rotate it onto the fiber direction, then move it.
    msh.scale([l, b, h], inplace=True)
    _place_fiber(msh, (x, y, z), (u, v, w))

    # Return VTK mesh
    return msh


def vtk_mat(mat=None, func=None, verbose=True, **kwargs):
    """
    Export a :class:`~.Mat` object as VTK mesh.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`~.Mat` object.
    func : callable, optional
        Function called for each fiber to modify the mesh or add fields.
        It takes as arguments the VTK mesh and the label of the fiber.
    verbose : bool, optional
        If True, a progress bar is displayed. Default is True.
    kwargs :
        Additional keyword arguments passed to :meth:`vtk_fiber` function.

    Returns
    -------
    pyvista.UnstructuredGrid
        VTK mesh.

    .. NOTE::
        The following fields are added to the VTK mesh:
            - `"fiber"` : fiber index
            - `"lbh"` : fiber dimensions (mm)
            - `"xyz"` : local fiber coordinates (mm)
            - `"uvw"` : fiber orientation vector
            - `"G"` : shear modulus (MPa)
            - `"E"` : tensile modulus (MPa)

    """
    # Optional
    if mat is None:
        mat = Mat()

    assert Mat.check(mat)

    fibers = []  # : list to store individual fiber meshes

    for i in tqdm(mat.index, desc="Create VTK mat", disable=not verbose):
        # Get fiber
        fiber = mat.loc[i].astype(float)
        # Create the VTK mesh (cylindrical structured grid)
        msh = vtk_fiber(*fiber[[*"lbhxyzuvwGE"]].values,
                        index=i,
                        **kwargs)
        if func is not None:
            # Create additional fields
            func(msh, i)
        # Append fiber mesh to list
        fibers.append(msh)

    # Combine all individual fiber meshes into a single VTK mesh
    return pv.MultiBlock(fibers).combine()


def vtk_tows(mat=None, theta_resolution=12, verbose=True):
    """Export draped tows.

    Straight fibers are drawn as usual. When ``mat.attrs["centerlines"]`` is
    set, each tow is a ribbon swept along that bent centerline.
    """
    if mat is None:
        mat = Mat()
    lines = mat.attrs.get("centerlines")
    if not lines:
        return vtk_mat(mat, verbose=verbose, theta_resolution=theta_resolution)

    assert Mat.check(mat)
    meshes = []
    width = mat["b"].to_numpy(dtype=float)
    thickness = mat["h"].to_numpy(dtype=float)
    for i, centerline in tqdm(list(enumerate(lines)), desc="Create VTK tows",
                              disable=not verbose):
        mesh = _swept_tow(
            centerline, float(width[i]), float(thickness[i]),
            theta_resolution, index=i,
        )
        meshes.append(mesh)
    return pv.MultiBlock(meshes).combine()


def _swept_tow(centerline, width, thickness, theta_resolution, index=None):
    """Elliptical ribbon following ``centerline``."""
    line = np.asarray(centerline, dtype=float)
    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    side = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
    length = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.divide(side, np.maximum(length, 1e-12))
    flat = length[:, 0] < 1e-8
    side[flat] = np.array([0.0, 1.0, 0.0])
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    normal = np.cross(side, tangent)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    angle = np.linspace(0.0, 2.0 * np.pi, theta_resolution, endpoint=False)
    ring = (
        line[:, None, :]
        + np.cos(angle)[None, :, None] * side[:, None, :] * (0.5 * width)
        + np.sin(angle)[None, :, None] * normal[:, None, :] * (0.5 * thickness)
    )
    ring = np.concatenate((ring, ring[:, :1, :]), axis=1)
    n_along, n_around, _ = ring.shape
    grid = pv.StructuredGrid()
    grid.points = ring.reshape(-1, 3)
    grid.dimensions = (n_around, n_along, 1)
    if index is not None:
        grid.point_data["fiber"] = np.full(grid.n_points, index)
    return grid


def vtk_mesh(mesh=None,
             displacement=None,
             rotation=None,
             force=None,
             torque=None,
             verbose=True, **kwargs):
    """
    Export a :class:`~.Mesh` object as VTK mesh.

    Parameters
    ----------
    mesh : pandas.DataFrame, optional
        Fiber mesh represented by a :class:`~.Mesh` object.
    displacement : numpy.ndarray, optional
        Nodal displacements.
    rotation : numpy.ndarray, optional
        Nodal rotations.
    force : numpy.ndarray, optional
        Nodal forces.
    torque : numpy.ndarray, optional
        Nodal torques.
    verbose : bool, optional
        If True, a progress bar is displayed. Default is True.
    kwargs :
        Additional keyword arguments passed to :meth:`vtk_fiber` function.

    Returns
    -------
    pyvista.UnstructuredGrid
        VTK mesh.

    .. HINT::
        The following fields are added to the VTK mesh:
            - `"fiber"` : fiber index
            - `"lbh"` : fiber dimensions (mm)
            - `"xyz"` : local fiber coordinates (mm)
            - `"uvw"` : fiber orientation vector
            - `"G"` : shear modulus (MPa)
            - `"E"` : tensile modulus (MPa)
        If `displacement` is not None:
            - `"displacement"` : displacement field (mm)
            - `"rotation"` : rotation field (rad)
            - `"curvature"` : curvature field (1 / mm)
        If `force` is not None:
            - `"force"` : force field (N)

    """
    # Optional
    if mesh is None:
        mesh = Mesh()

    assert Mesh.check(mesh)

    # Group nodes by fiber
    by_fiber = mesh.groupby("fiber")

    def interpolation_field(msh, i):
        # Prepare interpolation data
        fiber = by_fiber.get_group(i)
        s = fiber.s.values[:, None]
        x = msh["xyz"][:, [0]] * 0.9999
        k = KDTree(s).query(x, return_distance=False).ravel()
        s, x = s.ravel(), x.ravel()
        # Correct indices (s_k <= x_i < s_{k+1}, k \in [-1, n])
        k = k * (x > s[k]) + (k - 1) * (x <= s[k])
        # Indices containing relative distances (`np.floor(j) == k`)
        j = ((x - s[k]) / (s[k + 1] - s[k]))
        # Correct issues for periodic mesh
        j = np.array([*j])
        j[j == np.inf] = 0
        # Add to vtk_fiber
        j += fiber.index[k]
        msh["node"] = j

    # Create a VTK mesh with interpolation data
    mat = mesh.flags.mat
    msh = vtk_mat(mat, func=interpolation_field, verbose=verbose, **kwargs)

    if len(mat):
        # Interpolate fields
        s = np.arange(len(mesh))
        x = msh["node"]
        if displacement is not None:
            if rotation is None:
                rotation = np.zeros_like(displacement)
            displacement = CubicHermiteSpline(s, displacement, rotation)
            msh["displacement"] = np.zeros(msh.points.shape)
            msh["displacement"][:, 2] = displacement(x)
            msh["rotation"] = displacement.derivative()(x)
            msh["curvature"] = displacement.derivative(2)(x)
            msh.points += msh["displacement"]
        if force is not None:
            if torque is None:
                torque = 0 * force
            force = CubicHermiteSpline(s, force, torque)
            msh["force"] = force(x)

    # Periodic boundary conditions (optional)
    if len(mat) and mesh.attrs["periodic"]:
        X = Y = mat.attrs["size"]
        Z1, Z2 = np.min(msh.points), np.max(msh.points)
        # Duplicate mesh for periodic conditions
        msh = pv.MultiBlock([
            msh,
            msh.copy().translate([-X, 0, 0]),
            msh.copy().translate([X, 0, 0]),
            msh.copy().translate([0, -Y, 0]),
            msh.copy().translate([0, Y, 0]),
            msh.copy().translate([-X, -Y, 0]),
            msh.copy().translate([-X, Y, 0]),
            msh.copy().translate([X, -Y, 0]),
            msh.copy().translate([X, Y, 0]),
        ]).combine().clip_box([-X, X, -Y, Y, Z1, Z2], invert=False)

    # Return VTK mesh
    return msh


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    # from fibermat import *

    # Create a VTK fiber
    vtk_fiber().plot()

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    sol = solve(Model(mesh), packing=4)

    # Create a VTK mat
    vtk_mat(mat).plot()

    # Create a VTK mesh
    vtk_mesh(mesh).plot()

    # Export as VTK
    msh = vtk_mesh(
        mesh,
        sol.displacement(1),
        sol.rotation(1),
        sol.force(1),
        sol.torque(1),
    )
    msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
