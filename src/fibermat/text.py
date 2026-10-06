#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.affinity import scale, translate
from shapely.geometry import Polygon
from shapely.ops import unary_union


def text_polygon(text, height=40.0):
    """Outline of ``text`` as a shapely polygon, centered on the origin.

    ``height`` is the letter height in millimetres. Counters, such as the
    holes in A and R, stay open. The polygon lies in the xy plane, so
    :func:`fibermat.pack.clip_polygon` extrudes it through the stack.
    """
    if not text:
        raise ValueError("text must not be empty.")
    height = float(height)
    if height <= 0:
        raise ValueError("height must be positive.")

    path = TextPath(
        (0, 0), text, size=height,
        prop=FontProperties(family="DejaVu Sans", weight="bold"),
    )
    rings = []
    for contour in path.to_polygons(closed_only=True):
        coords = np.asarray(contour, dtype=float)
        if len(coords) < 4:
            continue
        ring = Polygon(coords)
        if not ring.is_valid:
            ring = ring.buffer(0)
        if ring.is_empty or ring.area <= 1e-6:
            continue
        rings.append(ring)
    if not rings:
        raise ValueError("The text did not produce an outline.")

    rings.sort(key=lambda ring: ring.area, reverse=True)
    exteriors = []
    holes = {}
    for ring in rings:
        probe = ring.representative_point()
        parent = next((outer for outer in exteriors if outer.contains(probe)), None)
        if parent is None:
            exteriors.append(ring)
            holes[id(ring)] = []
        else:
            holes[id(parent)].append(list(ring.exterior.coords))

    letters = [
        Polygon(outer.exterior.coords, holes[id(outer)])
        for outer in exteriors
    ]
    word = unary_union(letters).buffer(0)
    minx, miny, maxx, maxy = word.bounds
    word = translate(word, xoff=-0.5 * (minx + maxx), yoff=-0.5 * (miny + maxy))
    fitted = word.bounds
    current = fitted[3] - fitted[1]
    if current <= 0:
        raise ValueError("The text outline has no height.")
    factor = height / current
    return scale(word, xfact=factor, yfact=factor, origin=(0, 0))
