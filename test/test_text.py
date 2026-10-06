import numpy as np
import pandas as pd

from fibermat import Mat
from fibermat.pack import clip_polygon
from fibermat.text import text_polygon


def test_text_polygon_keeps_letter_counters():
    """MARCEL is six letters, and the counters in A and R stay open."""
    word = text_polygon("MARCEL", height=30.0)
    parts = list(word.geoms) if word.geom_type == "MultiPolygon" else [word]
    assert len(parts) == 6
    assert sum(len(part.interiors) for part in parts) >= 2
    assert abs(word.bounds[0] + word.bounds[2]) < 1e-6
    assert abs(word.bounds[1] + word.bounds[3]) < 1e-6
    assert abs((word.bounds[3] - word.bounds[1]) - 30.0) < 1e-6


def test_clip_polygon_drops_a_fiber_outside_the_word():
    word = text_polygon("MARCEL", height=30.0)
    frame = pd.DataFrame(
        [[4.0, 0.2, 0.2, 0.0, 40.0, 0.1, 1.0, 0.0, 0.0, 1.0, np.inf]],
        columns=list("lbhxyzuvwGE"),
    )
    frame.attrs["n"] = 1
    frame.attrs["size"] = 120.0
    assert len(clip_polygon(Mat(frame), word)) == 0
