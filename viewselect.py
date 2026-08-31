"""Choosing which views to solve on.

A 423-frame recording is not 423 useful views. Most of them are the board in
almost the same place as the frame before, and a near-duplicate view adds
nothing to the fit -- it only adds a pose to estimate. That matters more than it
sounds: OpenCV's calibration solves a dense system over 9 + 6N parameters and
inverts it every iteration, so cost grows roughly with the cube of the view
count. On this machine 50 views solve in about a second and 150 take fourteen,
which is the difference between an interactive reject-and-resolve loop and
walking away from the terminal.

So the default is to pick a spread rather than take everything. The greedy rule
is: first cover the frame -- keep taking the view that lights up the most
not-yet-covered cells of an image-space grid, because distortion is only
constrained where corners actually landed -- and once the grid is full, keep
taking the view most unlike everything already chosen, in a space of centroid,
apparent size (a stand-in for distance), tilt and in-plane rotation.

The features are all image-space, deliberately: this runs before there is a
calibration, so it cannot use one.

`--views all` is always there if you want to prove to yourself it changes
nothing.
"""

from __future__ import annotations

import numpy as np

from .detect import Detection

GRID = (12, 9)


def _features(corners: np.ndarray, image_size) -> np.ndarray:
    """A compact description of where and how the board sat, from pixels alone."""
    w, h = image_size
    diag = float(np.hypot(w, h))
    c = corners.mean(axis=0)
    centred = corners - c
    # Principal axes of the corner cloud: the larger one gives in-plane rotation,
    # their ratio gives foreshortening, which is tilt seen edge-on.
    cov = centred.T @ centred / max(len(corners), 1)
    evals, evecs = np.linalg.eigh(cov)
    major, minor = np.sqrt(max(evals[1], 1e-9)), np.sqrt(max(evals[0], 1e-9))
    angle = np.arctan2(evecs[1, 1], evecs[0, 1])
    return np.array([
        c[0] / w, c[1] / h,                    # where in the frame
        2.0 * major / diag,                    # apparent size ~ 1 / distance
        minor / major,                         # foreshortening ~ tilt
        np.cos(2 * angle) * 0.5, np.sin(2 * angle) * 0.5,   # in-plane, mod 180 deg
    ], float)


def _cells(corners: np.ndarray, image_size, grid=GRID) -> set[int]:
    w, h = image_size
    gx = np.clip((corners[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((corners[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
    return set((gy * grid[0] + gx).tolist())


def select_views(
    detections_by_camera: dict[str, dict[str, Detection]],
    image_size,
    keys,
    target: int = 60,
    min_corners: int = 8,
    require_all_cameras: bool = True,
    grid=GRID,
) -> tuple[list[str], str]:
    """Return (chosen keys, one-line explanation). `target <= 0` means all."""
    cameras = list(detections_by_camera)
    candidates = []
    for key in keys:
        dets = [detections_by_camera[c].get(key) for c in cameras]
        ok = [d is not None and d.count >= min_corners for d in dets]
        if (all(ok) if require_all_cameras else any(ok)):
            candidates.append(key)

    if target is None or target <= 0 or len(candidates) <= target:
        return candidates, (f"all {len(candidates)} detected views "
                            f"({len(keys) - len(candidates)} had no usable detection)")

    cells, feats = {}, {}
    for key in candidates:
        cs, fs = set(), []
        for n, cam in enumerate(cameras):
            det = detections_by_camera[cam].get(key)
            if det is None or not det.ok:
                continue
            cs |= {n * grid[0] * grid[1] + c for c in _cells(det.corners, image_size, grid)}
            fs.append(_features(det.corners, image_size))
        cells[key] = cs
        feats[key] = np.concatenate(fs)

    chosen: list[str] = []
    covered: set[int] = set()
    remaining = list(candidates)
    chosen_feats: list[np.ndarray] = []

    while remaining and len(chosen) < target:
        if chosen_feats:
            F = np.stack(chosen_feats)
            novelty = {k: float(np.linalg.norm(F - feats[k], axis=1).min()) for k in remaining}
        else:
            novelty = {k: 0.0 for k in remaining}
        best = max(remaining, key=lambda k: (len(cells[k] - covered), novelty[k]))
        chosen.append(best)
        covered |= cells[best]
        chosen_feats.append(feats[best])
        remaining.remove(best)

    total_cells = len(set().union(*cells.values())) if cells else 0
    why = (f"{len(chosen)} of {len(candidates)} detected views, greedily spread; "
           f"they reach {len(covered)}/{total_cells} of the image cells any view reaches")
    order = {k: i for i, k in enumerate(candidates)}
    return sorted(chosen, key=lambda k: order[k]), why
