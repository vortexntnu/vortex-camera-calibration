"""What the pixels say, as opposed to what the corners say.

Everything else in this package works from detected corners. By the time a
corner exists the image has already been reduced to twelve bytes, and the two
failure modes that ruin a calibration without ever failing detection are gone
with it:

  clipping    a corner sitting in a blown highlight or a crushed shadow still
              gets found, and still gets localised -- to the wrong place. The
              subpixel refinement fits a saddle to the intensity surface, and a
              surface that has been flattened by the sensor rail no longer has
              its saddle where the corner is. The residual stays small because
              the fit is self-consistent; it is just biased.

  blur        motion smears the intensity profile along the direction of travel.
              A smeared corner is localised precisely and wrongly in the same
              way, and worse, the bias is *systematic per view*, so it looks
              like a pose error rather than like noise.

Neither shows up as a large RMS. Both show up here.

The measures are relative: comparable between images of one dataset, not
between datasets. That is the question worth asking anyway -- which of *these*
frames should not be in the solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Where a pixel counts as clipped. 8-bit sensors rarely reach a true 0 or 255 on
# real optics, so a couple of counts of margin catches the pixels that have
# effectively hit the rail without waiting for the exact endpoint.
BLACK_LEVEL = 2
WHITE_LEVEL = 253

# Fraction of a region clipped before it is worth mentioning. Well below this is
# dead pixels and dust; above it is exposure.
CLIP_NOTABLE = 0.001

HIST_BINS = 64


@dataclass
class ImageStats:
    """Pixel-level facts about one image, and about the board inside it."""

    # -- exposure, whole frame ---------------------------------------------
    mean: float = 0.0
    p01: float = 0.0
    p99: float = 0.0
    black_frac: float = 0.0
    white_frac: float = 0.0
    # Whole frame, not the board region: this one is for judging the exposure
    # of the shot. The board_* fields below are the ones that bear on corners.
    hist: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, np.int64))

    # -- exposure, board region only ---------------------------------------
    # The frame can be beautifully exposed while the board is a white slab. Only
    # this pair bears on corner localisation.
    board_mean: float = 0.0
    board_black_frac: float = 0.0
    board_white_frac: float = 0.0
    board_contrast: float = 0.0     # p99 - p01 inside the board box, counts
    # The distribution that actually bears on corner accuracy. `hist` above is
    # the whole shot; this is the part corners are found in, and the two differ
    # whenever the background is darker or brighter than the target.
    board_hist: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, np.int64))

    # -- sharpness and motion ----------------------------------------------
    # Variance of the Laplacian over the board box, divided by the intensity
    # variance there. Dividing is what makes it a focus measure rather than a
    # contrast measure: a dim board and a bright board at the same focus land in
    # the same place.
    sharpness: float = 0.0
    # Structure-tensor anisotropy over the board box, 0 isotropic .. 1 fully
    # directional. A sharp checker pattern carries gradients in two
    # perpendicular directions and sits low. Motion along one axis erases the
    # gradients along it and leaves those across it, so this climbs.
    anisotropy: float = 0.0
    # Direction of travel implied by that anisotropy, degrees, 0 = +x, in image
    # coordinates. Meaningless when anisotropy is low; cross-check it against
    # the velocity the detections give before believing it.
    blur_angle_deg: float = 0.0

    ok: bool = False
    # Whether the board_* fields above really describe the board. Without a
    # detection there is no board region to measure, so `measure` falls back to
    # the whole frame -- useful for exposure, but it is not the board, and
    # anything reporting "% of the board" from it would be lying.
    board_ok: bool = False

    @property
    def clipped(self) -> bool:
        """Is any part of the board clipped?"""
        return (self.board_ok
                and max(self.board_black_frac, self.board_white_frac) > CLIP_NOTABLE)

    def notes(self) -> list[str]:
        out = []
        if self.board_white_frac > CLIP_NOTABLE:
            out.append(f"{self.board_white_frac * 100:.2f}% of the board is >= {WHITE_LEVEL}")
        if self.board_black_frac > CLIP_NOTABLE:
            out.append(f"{self.board_black_frac * 100:.2f}% of the board is <= {BLACK_LEVEL}")
        if self.board_contrast and self.board_contrast < 40:
            out.append(f"board contrast only {self.board_contrast:.0f} counts")
        return out


def _box(corners: np.ndarray, shape, pad: float = 0.02) -> tuple[int, int, int, int]:
    """Bounding box of the detected corners, padded, clipped to the image."""
    h, w = shape[:2]
    lo = corners.min(0)
    hi = corners.max(0)
    m = (hi - lo).max() * pad
    x0 = int(max(0, np.floor(lo[0] - m)))
    y0 = int(max(0, np.floor(lo[1] - m)))
    x1 = int(min(w, np.ceil(hi[0] + m)))
    y1 = int(min(h, np.ceil(hi[1] + m)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, w, h
    return x0, y0, x1, y1


def measure(gray: np.ndarray, corners: np.ndarray | None = None) -> ImageStats:
    """Statistics for one grayscale image, and for the board region within it.

    `corners` is the detection for this image, used only to find the region
    worth measuring. Without it every measure is taken over the whole frame,
    which is weaker but still tells you about exposure.
    """
    if gray is None or gray.size == 0:
        return ImageStats()
    g = gray if gray.dtype == np.uint8 else cv2.normalize(
        gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    st = ImageStats(ok=True)
    hist = cv2.calcHist([g], [0], None, [HIST_BINS], [0, 256]).ravel().astype(np.int64)
    st.hist = hist
    total = int(g.size)
    st.mean = float(g.mean())
    st.p01, st.p99 = (float(v) for v in np.percentile(g, [1, 99]))
    st.black_frac = float(np.count_nonzero(g <= BLACK_LEVEL) / total)
    st.white_frac = float(np.count_nonzero(g >= WHITE_LEVEL) / total)

    if corners is not None and len(corners) >= 4:
        x0, y0, x1, y1 = _box(np.asarray(corners, np.float64), g.shape)
        st.board_ok = True
    else:
        x0, y0, x1, y1 = 0, 0, g.shape[1], g.shape[0]
    roi = g[y0:y1, x0:x1]
    if roi.size < 64:
        return st

    n = int(roi.size)
    st.board_mean = float(roi.mean())
    st.board_black_frac = float(np.count_nonzero(roi <= BLACK_LEVEL) / n)
    st.board_white_frac = float(np.count_nonzero(roi >= WHITE_LEVEL) / n)
    lo, hi = np.percentile(roi, [1, 99])
    st.board_contrast = float(hi - lo)
    st.board_hist = cv2.calcHist([roi], [0], None, [HIST_BINS], [0, 256]).ravel().astype(np.int64)

    f = roi.astype(np.float32)
    var = float(f.var())
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    st.sharpness = float(lap.var() / var) if var > 1e-6 else 0.0

    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    tr = jxx + jyy
    if tr > 1e-9:
        # Closed-form eigenvalues of the 2x2 structure tensor.
        d = np.hypot(jxx - jyy, 2 * jxy)
        st.anisotropy = float(d / tr)
        # Dominant gradient orientation; travel is across it, hence the +90.
        ang = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
        st.blur_angle_deg = float((ang + 90.0) % 180.0)
    return st


def summarise(stats: dict) -> dict:
    """Dataset-level rollup of a {key: ImageStats} mapping."""
    good = [s for s in stats.values() if s.ok]
    if not good:
        return {}
    sharp = np.array([s.sharpness for s in good])
    return {
        "images": len(good),
        "clipped_views": sum(1 for s in good if s.clipped),
        "board_white_frac_max": max(s.board_white_frac for s in good),
        "board_black_frac_max": max(s.board_black_frac for s in good),
        "board_contrast_min": min(s.board_contrast for s in good),
        "sharpness_median": float(np.median(sharp)),
        "sharpness_p05": float(np.percentile(sharp, 5)),
        # Softest frames relative to the set's own median -- the ones to look at.
        "soft_views": sum(1 for s in good
                          if s.sharpness < 0.5 * float(np.median(sharp))),
    }
