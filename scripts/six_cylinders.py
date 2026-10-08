#!/usr/bin/env python3
"""CF-SMC rolls and cuboid stacks with 25.5 mm tows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from shapely.geometry import box

from fibermat.mat import Mat
from fibermat.pack import clip_polygon, pack, subdivide, write_lines

OUTER_D = 29.0
INNER_D = 25.0
WIDTH = 30.0
CLIP_WIDTH = 18.0

outer_r = 0.5 * OUTER_D
inner_r = 0.5 * INNER_D
wall = outer_r - inner_r
mid_r = 0.5 * (outer_r + inner_r)
lx = 2.0 * np.pi * mid_r
ly = WIDTH
lz = wall
half_clip = 0.5 * CLIP_WIDTH

CUBOID_BOX = (120.0, 50.0, 9.0)
CUBOID_XY = (90.0, 33.0)
CUBOID_Z = (9.0, 8.0)

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

TOW = dict(
    length=25.5,
    width=4.0,
    thickness=0.103,
    section="rectangle",
    volume_fraction=0.4,
)
TOW_VOLUME = TOW["length"] * TOW["width"] * TOW["thickness"]


def compress_z(mat, lx, ly, lz):
    """Scale tow heights so the stack fits in ``lz`` without dropping tows."""
    if len(mat) == 0:
        return mat
    top = float((mat.z + 0.5 * mat.h).max())
    scale = float(lz) / max(top, 1e-12)
    data = np.array(mat[list("lbhxyzuvwGE")].to_numpy(dtype=float), copy=True)
    data[:, 5] *= scale
    frame = pd.DataFrame(data, columns=list("lbhxyzuvwGE"))
    frame.attrs = dict(mat.attrs)
    frame.attrs["n"] = len(frame)
    frame.attrs["box"] = (float(lx), float(ly), float(lz))
    frame.attrs["volume_fraction"] = len(frame) * TOW_VOLUME / (lx * ly * lz)
    frame.attrs["compress_scale"] = scale
    for key in ("tow", "source_length"):
        if key in mat.attrs:
            frame.attrs[key] = np.asarray(mat.attrs[key])
    return Mat(frame)


def pack_at_vf(box, seed, target_vf=None):
    """Sediment straight tows; ``volume_fraction`` is only a cap, no compaction."""
    tows = pack(box=box, seed=seed, bend=False, **TOW)
    print(
        f"    placed {len(tows)}  vf={float(tows.attrs['volume_fraction']):.4f}  "
        f"(cap {TOW['volume_fraction']}, no draping/compress)"
    )
    return tows


def clip_z(mat, zmax):
    """Keep fibers whose thickness stays at or below ``zmax``."""
    if len(mat) == 0:
        return mat
    top = mat["z"].to_numpy(dtype=float) + 0.5 * mat["h"].to_numpy(dtype=float)
    keep = top <= float(zmax) + 1e-9
    data = mat.loc[keep, list("lbhxyzuvwGE")].to_numpy(dtype=float)
    if len(data) == 0:
        frame = pd.DataFrame(columns=list("lbhxyzuvwGE"))
        frame.attrs["n"] = 0
        return Mat(frame)
    frame = pd.DataFrame(data, columns=list("lbhxyzuvwGE"))
    frame.attrs["n"] = len(frame)
    frame.attrs["size"] = float(mat.attrs.get("size", 1.0))
    for key in ("box", "periodic", "section", "volume_fraction"):
        if key in mat.attrs:
            frame.attrs[key] = mat.attrs[key]
    if "box" in mat.attrs:
        bx, by, _bz = mat.attrs["box"]
        frame.attrs["box"] = (bx, by, float(zmax))
    idx = np.flatnonzero(keep)
    for key in ("tow", "source_length"):
        if key in mat.attrs:
            frame.attrs[key] = np.asarray(mat.attrs[key])[idx]
    return Mat(frame)


def as_tubes(path):
    mesh = pv.read(path)
    if "diameter" in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_geometry()
    radius = np.asarray(mesh["diameter"], dtype=float) * 0.5
    mesh["radius"] = radius
    return mesh.tube(scalars="radius", absolute=True, n_sides=6, capping=True)


def screenshot(tubes, path, camera, *, window_size=(1600, 1100), zoom=1.0, title=None):
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("white")
    plotter.add_mesh(
        tubes,
        scalars="angle" if "angle" in tubes.array_names else None,
        cmap="viridis",
        clim=[0.0, 180.0],
        show_scalar_bar=True,
        scalar_bar_args={"title": "angle (°)", "color": "black"},
        smooth_shading=True,
    )
    plotter.add_axes(color="black")
    if title:
        plotter.add_text(title, font_size=12, color="black")
    plotter.camera_position = camera
    plotter.camera.zoom(zoom)
    plotter.show(screenshot=str(path), auto_close=True)
    plotter.close()
    print(f"  preview {path.name}")


def make_rolls():
    print(
        f"rolls box lx={lx:.4f} ly={ly:.1f} lz={lz:.2f}  "
        f"mid_r={mid_r:.2f} inner={inner_r:.2f} outer={outer_r:.2f}"
    )
    span = -0.5 * lx - 50.0, 0.5 * lx + 50.0
    clip_30 = box(span[0], -0.5 * WIDTH, span[1], 0.5 * WIDTH)
    clip_18 = box(span[0], -half_clip, span[1], half_clip)

    for seed in range(1, 7):
        tows = pack_at_vf((lx, ly, lz), seed=seed)
        fibers = subdivide(tows)
        vf = float(tows.attrs.get("volume_fraction", 0.0))
        print(f"roll {seed}: tows={len(tows)} fibers={len(fibers)} vf={vf:.4f}")

        wide = clip_polygon(fibers, clip_30)
        full_path = OUT / f"roll_{seed:02d}_w30.vtk"
        write_lines(wide, full_path, n=10, roll=True)
        print(f"  wrote {full_path.name}  fibers={len(wide)}")

        clipped = clip_polygon(fibers, clip_18)
        clip_path = OUT / f"roll_{seed:02d}_w18.vtk"
        write_lines(clipped, clip_path, n=10, roll=True)
        y = np.abs(clipped[["y"]].to_numpy(dtype=float))
        print(
            f"  wrote {clip_path.name}  fibers={len(clipped)}  "
            f"|y|_max={float(y.max()) if len(clipped) else 0:.3f}"
        )


def make_cuboids():
    bx, by, bz = CUBOID_BOX
    cx, cy = CUBOID_XY
    poly = box(-0.5 * cx, -0.5 * cy, 0.5 * cx, 0.5 * cy)
    print(f"cuboids pack {CUBOID_BOX}  clip xy {CUBOID_XY}  z {CUBOID_Z}")

    for seed in range(1, 4):
        tows = pack_at_vf(CUBOID_BOX, seed=seed)
        fibers = subdivide(tows)
        vf = float(tows.attrs.get("volume_fraction", 0.0))
        print(f"cuboid pack {seed}: tows={len(tows)} fibers={len(fibers)} vf={vf:.4f}")
        xy = clip_polygon(fibers, poly)
        for zmax in CUBOID_Z:
            cut = xy if zmax >= bz else clip_z(xy, zmax)
            if len(cut):
                z_top = float((cut.z + 0.5 * cut.h).max())
                xspan = float(np.abs(cut.x).max())
                yspan = float(np.abs(cut.y).max())
            else:
                z_top = xspan = yspan = 0.0
            path = OUT / f"cuboid_{seed:02d}_90x33x{int(zmax)}.vtk"
            write_lines(cut, path, n=10, roll=False)
            print(
                f"  wrote {path.name}  fibers={len(cut)}  "
                f"|x|<={xspan:.2f} |y|<={yspan:.2f} z_top={z_top:.3f}"
            )


def render_previews():
    roll = as_tubes(OUT / "roll_01_w18.vtk")
    iso = [
        (45.0, 28.0, 45.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    axial = [
        (0.0, 55.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    screenshot(roll, OUT / "preview_roll01_w18_iso.png", iso, title="roll 1, 18 mm, L=25.5 mm")
    screenshot(roll, OUT / "preview_roll01_w18_axial.png", axial, title="roll 1, axial, L=25.5 mm")

    thumbs = []
    for seed in range(1, 7):
        tubes = as_tubes(OUT / f"roll_{seed:02d}_w18.vtk")
        path = OUT / f"preview_roll{seed:02d}_w18_iso.png"
        screenshot(tubes, path, iso, window_size=(900, 700), title=f"roll {seed}, 18 mm")
        thumbs.append(path)

    from matplotlib import pyplot as plt
    from matplotlib.image import imread

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    for ax, path in zip(axes.ravel(), thumbs):
        ax.imshow(imread(path))
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    montage = OUT / "preview_six_w18.png"
    fig.savefig(montage, dpi=120)
    plt.close(fig)
    print(f"  preview {montage.name}")

    cuboid_cam = [
        (80.0, 70.0, 55.0),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, 1.0),
    ]
    cuboid_thumbs = []
    for seed in range(1, 4):
        for zmax in (9, 8):
            tubes = as_tubes(OUT / f"cuboid_{seed:02d}_90x33x{zmax}.vtk")
            path = OUT / f"preview_cuboid_{seed:02d}_90x33x{zmax}.png"
            screenshot(
                tubes, path, cuboid_cam, window_size=(1100, 800),
                title=f"cuboid {seed}, 90x33x{zmax} mm",
            )
            cuboid_thumbs.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    for ax, path in zip(axes.ravel(), cuboid_thumbs):
        ax.imshow(imread(path))
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    montage = OUT / "preview_six_cuboids.png"
    fig.savefig(montage, dpi=120)
    plt.close(fig)
    print(f"  preview {montage.name}")


def main(parts=None):
    parts = set(parts or ("rolls", "cuboids", "preview"))
    if "rolls" in parts:
        make_rolls()
    if "cuboids" in parts:
        make_cuboids()
    if "preview" in parts:
        render_previews()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
