"""The questions to ask when the RMS is bad and you do not know why.

`report.py` plots what a solve produced. This module works out *why* it came
out that way, by testing the residual against everything that could be driving
it. Each function answers one question and returns numbers, not pictures;
`pdfreport.py` draws them.

The tests are ordered by what they eliminate:

  pose_dependence   does the error track where the board was -- distance, tilt,
                    position in frame? Then the lens model is wrong, or the
                    poses never constrained it. This is the common case and the
                    cheapest to fix: shoot better views.

  motion_dependence does it track how fast the board was moving between frames?
                    Then the two cameras are not exposing at the same instant,
                    and no amount of solving fixes a timing bug. Stereo only --
                    a mono solve has nothing to be out of sync with.

  temporal_structure does it drift smoothly over the run rather than scattering?
                    A correlated residual is a physical process. Combined with
                    the test below it separates a moving mount from a bad model.

  rig_drift         stereo only, and the one that matters most when the mono
                    solves are clean and the pair is not. Refit the rig per view
                    instead of once. If the epipolar error collapses, the
                    correspondences are fine and the cameras moved relative to
                    each other during the capture; there is no single R, T to
                    find and the capture has to be redone on a stiffer mount.

  time_vs_pose      the discriminator between the two explanations that both
                    produce a smooth residual. Predict each view's error from
                    its neighbours in time, then from its neighbours in board
                    pose. Whichever predicts better is what is driving it. This
                    is what separates "the mount moved" from "the lens model is
                    wrong", and nothing else does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import (MonoCalibration, StereoCalibration, _sampson,
                        _undistort_pixels, solve_pose)

# Below this many usable views a correlation is noise dressed as a finding.
MIN_FOR_CORRELATION = 12


def _numeric(keys) -> np.ndarray | None:
    """View keys as frame numbers, or None if they are not numeric.

    Anything time-ordered here depends on the key being a frame index. When the
    dataset is a directory of arbitrarily named images it is not, and every
    test that needs an ordering has to be skipped rather than faked.
    """
    try:
        return np.array([int(k) for k in keys], float)
    except (TypeError, ValueError):
        return None


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_FOR_CORRELATION or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------------------


@dataclass
class ViewFacts:
    """One row per view: the error, and everything that might explain it."""

    keys: list = field(default_factory=list)
    index: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    distance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    corners: np.ndarray = field(default_factory=lambda: np.zeros(0))
    span: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sharpness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anisotropy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    board_white: np.ndarray = field(default_factory=lambda: np.zeros(0))
    board_black: np.ndarray = field(default_factory=lambda: np.zeros(0))
    board_contrast: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Board travel between the previous and next view, pixels per frame. Only
    # meaningful when the keys are consecutive frame numbers.
    speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    speed_y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return len(self.keys)


def view_facts(cal: MonoCalibration, detections=None, camera: str | None = None) -> ViewFacts:
    """Assemble the per-view table the tests below all run against."""
    keys = sorted(cal.views, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
    f = ViewFacts(keys=list(keys))
    n = len(keys)
    if not n:
        return f
    cam = camera or cal.camera
    f.rms = np.array([cal.views[k].rms for k in keys])
    f.distance = np.array([cal.views[k].distance for k in keys])
    f.tilt = np.array([cal.views[k].tilt_deg for k in keys])
    f.centroid = np.array([cal.views[k].observed.mean(0) for k in keys])
    f.corners = np.array([len(cal.views[k].observed) for k in keys], float)
    f.span = np.array([np.ptp(cal.views[k].observed, axis=0).max() for k in keys])

    idx = _numeric(keys)
    f.index = idx if idx is not None else np.arange(n, dtype=float)

    for name in ("sharpness", "anisotropy", "board_white", "board_black", "board_contrast"):
        setattr(f, name, np.full(n, np.nan))
    if detections is not None:
        for i, k in enumerate(keys):
            s = detections.stats_for(cam, k)
            if not s.ok:
                continue
            f.sharpness[i] = s.sharpness
            f.anisotropy[i] = s.anisotropy
            f.board_white[i] = s.board_white_frac
            f.board_black[i] = s.board_black_frac
            f.board_contrast[i] = s.board_contrast

    # Board travel, from a centred difference over the neighbouring frames. Only
    # defined where both neighbours are actually adjacent frames: across a gap
    # the difference measures the gap, not the motion.
    f.speed = np.full(n, np.nan)
    f.speed_y = np.full(n, np.nan)
    if idx is not None:
        pos = {int(k): f.centroid[i] for i, k in enumerate(keys)}
        for i, k in enumerate(keys):
            a, b = pos.get(int(k) - 1), pos.get(int(k) + 1)
            if a is None or b is None:
                continue
            d = (b - a) / 2.0
            f.speed[i] = float(np.hypot(*d))
            f.speed_y[i] = float(d[1])
    return f


# ---------------------------------------------------------------------------


def pose_dependence(f: ViewFacts) -> dict:
    """How much of the per-view error is explained by where the board was.

    A residual that rises with distance, or with tilt, is the lens model failing
    where the views stopped constraining it. A residual that tracks position in
    frame is distortion. A residual that tracks nothing here is either noise or
    something time-varying -- see `temporal_structure`.
    """
    if len(f) < MIN_FOR_CORRELATION:
        return {}
    out = {
        "board distance": _corr(f.distance, f.rms),
        "board tilt": _corr(f.tilt, f.rms),
        "corners seen": _corr(f.corners, f.rms),
        "board size in frame": _corr(f.span, f.rms),
        "distance from image centre": _corr(
            np.hypot(*(f.centroid - f.centroid.mean(0)).T), f.rms),
    }
    if np.isfinite(f.sharpness).sum() >= MIN_FOR_CORRELATION:
        out["sharpness"] = _corr(f.sharpness, f.rms)
        out["motion anisotropy"] = _corr(f.anisotropy, f.rms)
        out["board clipping"] = _corr(np.nan_to_num(f.board_white + f.board_black), f.rms)
    return {k: v for k, v in out.items() if np.isfinite(v)}


def motion_dependence(f: ViewFacts, offsets: np.ndarray | None = None,
                      frame_ms: float | None = None) -> dict:
    """Does the error track board speed? If so, the two cameras disagree on when.

    `offsets` is the per-view mean rectified dy -- a bulk vertical shift between
    the two rectified images. If the cameras expose at different instants, that
    shift is the board's own travel during the gap, so it is proportional to the
    board's vertical velocity, and the slope of one against the other *is* the
    timing error. If the slope is flat, the shift is not timing.

    The control matters as much as the test: an error that also fails to track
    speed when the board is nearly still is not a motion artefact at all.
    """
    # A couple of frames that happen to have both neighbours is not a speed
    # measurement; reporting a median over two views invites a wrong conclusion.
    if len(f) < MIN_FOR_CORRELATION or np.isfinite(f.speed).sum() < MIN_FOR_CORRELATION:
        return {}
    out = {
        "corr(rms, board speed)": _corr(f.speed, f.rms),
        "median speed px/frame": float(np.nanmedian(f.speed)),
        "max speed px/frame": float(np.nanmax(f.speed)),
    }
    if offsets is not None and len(offsets) == len(f):
        m = np.isfinite(f.speed_y) & np.isfinite(offsets)
        if m.sum() >= MIN_FOR_CORRELATION and np.std(f.speed_y[m]) > 1e-9:
            slope = float(np.polyfit(f.speed_y[m], offsets[m], 1)[0])
            out["corr(dy offset, vertical speed)"] = _corr(f.speed_y, offsets)
            out["implied time offset (frames)"] = slope
            if frame_ms:
                out["implied time offset (ms)"] = slope * frame_ms
        # The control: still frames should be clean if this is timing.
        slow = m & (np.abs(f.speed_y) < np.nanpercentile(np.abs(f.speed_y[m]), 25))
        fast = m & (np.abs(f.speed_y) > np.nanpercentile(np.abs(f.speed_y[m]), 75))
        if slow.sum() > 4 and fast.sum() > 4:
            out["dy rms, slowest quartile"] = float(np.sqrt(np.mean(offsets[slow] ** 2)))
            out["dy rms, fastest quartile"] = float(np.sqrt(np.mean(offsets[fast] ** 2)))
    return out


def temporal_structure(f: ViewFacts, values: np.ndarray | None = None,
                       lags=(1, 2, 3, 5, 10, 20)) -> dict:
    """Autocorrelation of the per-view error against frame lag.

    Independent per-frame noise decorrelates at lag 1. Anything that stays
    correlated for seconds is a physical process -- a mount settling, focus
    breathing, the light changing -- and it will not average out by adding
    views, because neighbouring views carry the same error.
    """
    v = f.rms if values is None else values
    idx = _numeric(f.keys)
    if idx is None or len(f) < MIN_FOR_CORRELATION:
        return {}
    at = {int(k): val for k, val in zip(f.keys, v) if np.isfinite(val)}
    out = {}
    for lag in lags:
        pairs = [(at[k], at[k + lag]) for k in at if k + lag in at]
        if len(pairs) >= MIN_FOR_CORRELATION:
            a, b = np.array(pairs).T
            out[lag] = _corr(a, b)
    return {k: val for k, val in out.items() if np.isfinite(val)}


def time_vs_pose(f: ViewFacts, values: np.ndarray | None = None,
                 k: int = 3, min_gap: int = 25) -> dict:
    """Is the error a function of *when* the view was taken, or of *how* it looked?

    Both a moving mount and a wrong lens model produce a residual that varies
    smoothly, so smoothness alone cannot tell them apart. This can. Predict each
    view's error from its k nearest neighbours in time, then from its k nearest
    neighbours in board pose -- deliberately excluding pose-neighbours that are
    also close in time, or the two predictors measure the same thing.

    Time wins   -> something changed between frames. The rig moved.
    Pose wins   -> the error is a property of where the board was. The model is
                   wrong, or the poses never pinned it down.
    Neither     -> noise, and the residual is as good as this data gets.
    """
    v = f.rms if values is None else values
    idx = _numeric(f.keys)
    if idx is None or len(f) < 3 * MIN_FOR_CORRELATION:
        return {}
    good = np.isfinite(v)
    if good.sum() < 3 * MIN_FOR_CORRELATION:
        return {}
    t = idx[good]
    y = np.asarray(v, float)[good]
    # Pose descriptor, each component scaled to a comparable range so no single
    # axis dominates the neighbour search.
    feat = np.column_stack([
        f.centroid[good, 0] / 100.0,
        f.centroid[good, 1] / 100.0,
        f.distance[good] * 5.0,
        f.tilt[good] / 5.0,
    ])
    dt = np.abs(t[:, None] - t[None, :])
    dp = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=2)
    np.fill_diagonal(dt, np.inf)
    dp_far = dp.copy()
    dp_far[dt < min_gap] = np.inf     # pose-neighbours must not also be time-neighbours

    def predict(metric):
        pred = np.full(len(y), np.nan)
        for i in range(len(y)):
            j = np.argsort(metric[i])[:k]
            j = j[np.isfinite(metric[i][j])]
            if len(j):
                pred[i] = y[j].mean()
        return pred

    out = {}
    for name, metric in (("time", dt), ("pose", dp_far)):
        p = predict(metric)
        m = np.isfinite(p)
        if m.sum() >= MIN_FOR_CORRELATION:
            out[f"{name} corr"] = _corr(p[m], y[m])
            out[f"{name} rms error"] = float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
    out["baseline rms error"] = float(np.std(y))

    # How far apart the solved views actually are. The time predictor can only
    # see as far as the nearest kept view, so a set thinned to every fifth frame
    # handicaps it against pose and can flip the verdict. Report the spacing so
    # the reader can discount the answer, and withhold the verdict when the gap
    # is wide enough to make the comparison unfair.
    gaps = np.diff(np.sort(t))
    out["median view gap (frames)"] = float(np.median(gaps)) if len(gaps) else float("nan")
    if "time corr" in out and "pose corr" in out:
        if out["median view gap (frames)"] > 2:
            out["verdict"] = "inconclusive (views too sparse; re-run with --views all)"
        else:
            out["verdict"] = ("time" if out["time rms error"] < out["pose rms error"]
                              else "pose")
    return out


# ---------------------------------------------------------------------------
# Coverage


def coverage(cal: MonoCalibration, grid=(8, 6)) -> dict:
    """Where the corners actually landed, and where they never did.

    Distortion is only constrained where corners went. Undistorting a region the
    board never visited is extrapolation, and the coefficients that behave
    inside the covered region can do anything outside it.
    """
    pts = cal.all_observed()
    w, h = cal.image_size
    if not len(pts):
        return {}
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=grid,
                                range=[[0, w], [0, h]])
    cells = int(hist.size)
    return {
        "corners": int(len(pts)),
        "u range": (float(pts[:, 0].min()), float(pts[:, 0].max())),
        "v range": (float(pts[:, 1].min()), float(pts[:, 1].max())),
        "u margin left": float(pts[:, 0].min() / w),
        "u margin right": float(1 - pts[:, 0].max() / w),
        "v margin top": float(pts[:, 1].min() / h),
        "v margin bottom": float(1 - pts[:, 1].max() / h),
        "empty cells": int((hist == 0).sum()),
        "cells": cells,
        "thin cells": int((hist < max(1, 0.1 * hist.mean())).sum()),
        "grid": hist,
    }


def board_coverage(cal: MonoCalibration, board) -> np.ndarray | None:
    """How often each board corner was detected, laid out as the board is.

    A board whose outer ring is rarely read is a board that was too close to the
    frame edge, or lit too unevenly at the margins. Those corners carry most of
    the distortion information, so losing them is expensive.
    """
    cols = board.columns - 1 if board.kind == "charuco" else board.columns
    rows = board.rows - 1 if board.kind == "charuco" else board.rows
    if cols * rows != board.n_points:
        return None
    count = np.zeros(board.n_points, int)
    for v in cal.views.values():
        count[np.asarray(v.ids, int)] += 1
    return count.reshape(rows, cols)


# ---------------------------------------------------------------------------
# Stereo only


def dy_offsets(s: StereoCalibration) -> tuple[list, np.ndarray, dict]:
    """Per-view mean rectified dy, and how the row error splits up.

    A rectified pair should put every correspondence on the same row whatever
    its depth, so dy is a pure error signal. Splitting its variance says what
    kind of error it is: a bulk shift that differs view to view is the rig
    changing between views, while structure *within* each view is the lens
    model or the rectification.
    """
    keys = sorted(s.views, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
    if not keys:
        return [], np.zeros(0), {}
    per = [np.asarray(s.views[k].rect_dy, float) for k in keys]
    means = np.array([p.mean() for p in per])
    allv = np.concatenate(per)
    n = len(allv)
    # Law of total variance, weighted by view size so the two shares sum to one.
    # An unweighted mean-of-means does not decompose exactly when views carry
    # different corner counts, and quietly reports shares above 100%.
    grand = float(allv.mean())
    within_ss = float(sum(((p - p.mean()) ** 2).sum() for p in per))
    between_ss = float(sum(len(p) * (p.mean() - grand) ** 2 for p in per))
    total_ss = within_ss + between_ss
    split = {
        "between std px": float(np.sqrt(between_ss / n)),
        "within std px": float(np.sqrt(within_ss / n)),
        "between share": between_ss / total_ss if total_ss else float("nan"),
        "within share": within_ss / total_ss if total_ss else float("nan"),
        "dy rms px": float(np.sqrt((allv ** 2).mean())),
    }
    return keys, means, split


def rig_drift(s: StereoCalibration) -> dict:
    """Refit the rig per view and see how much of the epipolar error survives.

    The global R, T is one rigid transform for the whole capture. Each view also
    implies its own, from the board solved independently in the two cameras and
    with no stereo constraint anywhere. Comparing the epipolar error under each
    separates two very different failures:

      both large  -> the correspondences or the intrinsics are wrong.
      global large, per-view small -> the correspondences are fine and the
                    cameras moved relative to each other during the capture.
                    There is no single R, T to find. Re-shoot on a stiffer
                    mount; nothing in software recovers this.

    The per-view transform is *derived* from the two poses, not fitted to
    minimise the epipolar error, which is what makes a low value here evidence
    rather than a tautology.
    """
    if not s.views:
        return {}
    K1, d1 = s.left.K, s.left.dist
    K2, d2 = s.right.K, s.right.dist
    fish = s.model.fisheye

    def F_of(R, t):
        t = np.asarray(t, float).reshape(3, 1)
        tx = np.array([[0, -t[2, 0], t[1, 0]],
                       [t[2, 0], 0, -t[0, 0]],
                       [-t[1, 0], t[0, 0], 0]])
        return np.linalg.inv(K2).T @ (tx @ R) @ np.linalg.inv(K1)

    Fg = F_of(np.asarray(s.R), np.asarray(s.T))
    glob, pv, rot, keys = [], [], [], []
    for key, v in s.views.items():
        pl = np.asarray(v.left.observed, float)
        pr = np.asarray(v.right.observed, float)
        if len(pl) < 8:
            continue
        obj = s.board.object_points_for(v.ids).astype(np.float64)
        try:
            rl, tl = solve_pose(obj, pl, K1, d1, fish)
            rr, tr = solve_pose(obj, pr, K2, d2, fish)
        except Exception:
            continue
        Rl, Rr = cv2.Rodrigues(rl)[0], cv2.Rodrigues(rr)[0]
        Rrel = Rr @ Rl.T
        trel = tr.reshape(3) - Rrel @ tl.reshape(3)
        x1 = _undistort_pixels(pl, K1, d1, fish)
        x2 = _undistort_pixels(pr, K2, d2, fish)
        glob.append(_sampson(Fg, x1, x2))
        pv.append(_sampson(F_of(Rrel, trel), x1, x2))
        rot.append(np.degrees(cv2.Rodrigues(Rrel)[0].ravel()))
        keys.append(key)
    if not glob:
        return {}
    g = np.concatenate(glob)
    p = np.concatenate(pv)
    rot = np.array(rot)
    g_rms, p_rms = float(np.sqrt((g ** 2).mean())), float(np.sqrt((p ** 2).mean()))
    return {
        "keys": keys,
        "global rms px": g_rms,
        "per view rms px": p_rms,
        "explained": float(1 - p_rms / g_rms) if g_rms else float("nan"),
        "rotation": rot,
        "rotation mad deg": 1.4826 * np.median(np.abs(rot - np.median(rot, 0)), 0),
        "rotation ptp deg": np.ptp(rot, axis=0),
    }


def dense_stereo_series(s: StereoCalibration, detections,
                        left: str = "left", right: str = "right"):
    """Row error and board pose for *every* detected view, not just solved ones.

    The solve runs on a spread of views chosen for coverage, which is usually a
    thinned subset of the capture -- `--views 60` out of several hundred. That
    is right for fitting and wrong for every test that needs a time axis: with
    only every fifth frame kept there are no adjacent frames left, so
    autocorrelation, board speed and the time-versus-pose comparison all have
    nothing to work with, and quietly return nothing.

    So: fit on the subset, evaluate on everything. The calibration is held
    fixed and simply applied to each detected pair, which costs one rectify and
    two PnP solves per view and makes the whole recording visible.

    Returns (ViewFacts, per-view mean dy). The facts carry the *dy offset* in
    `rms`, not a reprojection error -- these views were never in the solve, so
    there is no residual for them.
    """
    from .calibrate import rectify_points
    L = detections.per_camera.get(left, {})
    R = detections.per_camera.get(right, {})
    keys = sorted(set(L) & set(R), key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
    K1, d1 = s.left.K, s.left.dist
    K2, d2 = s.right.K, s.right.dist
    fish = s.model.fisheye

    kept, offs, dist, tilt, cen, spans, sharp, aniso = [], [], [], [], [], [], [], []
    for k in keys:
        dl, dr = L[k], R[k]
        if not (dl.ok and dr.ok):
            continue
        shared = np.intersect1d(dl.ids, dr.ids)
        if len(shared) < 12:
            continue
        li = {int(i): n for n, i in enumerate(dl.ids)}
        ri = {int(i): n for n, i in enumerate(dr.ids)}
        pl = dl.corners[[li[int(i)] for i in shared]].astype(np.float64)
        pr = dr.corners[[ri[int(i)] for i in shared]].astype(np.float64)
        yl = rectify_points(pl, K1, d1, s.rect.R1, s.rect.P1, fish)
        yr = rectify_points(pr, K2, d2, s.rect.R2, s.rect.P2, fish)
        obj = s.board.object_points_for(shared).astype(np.float64)
        try:
            rvec, tvec = solve_pose(obj, pl, K1, d1, fish)
        except Exception:
            continue
        z = float(np.linalg.norm(tvec))
        if not (0.05 < z < 50):
            continue
        kept.append(k)
        offs.append(float(np.mean(yl[:, 1] - yr[:, 1])))
        dist.append(z)
        tilt.append(float(np.degrees(np.arccos(
            np.clip(cv2.Rodrigues(rvec)[0][2, 2], -1, 1)))))
        cen.append(yl.mean(0))
        spans.append(float(np.ptp(pl, axis=0).max()))
        st = detections.stats_for(left, k)
        sharp.append(st.sharpness if st.ok else np.nan)
        aniso.append(st.anisotropy if st.ok else np.nan)

    f = ViewFacts(keys=kept)
    n = len(kept)
    if not n:
        return f, np.zeros(0)
    idx = _numeric(kept)
    f.index = idx if idx is not None else np.arange(n, dtype=float)
    f.rms = np.array(offs)
    f.distance = np.array(dist)
    f.tilt = np.array(tilt)
    f.centroid = np.array(cen)
    f.span = np.array(spans)
    f.corners = np.full(n, np.nan)
    f.sharpness = np.array(sharp)
    f.anisotropy = np.array(aniso)
    for name in ("board_white", "board_black", "board_contrast"):
        setattr(f, name, np.full(n, np.nan))
    f.speed = np.full(n, np.nan)
    f.speed_y = np.full(n, np.nan)
    if idx is not None:
        pos = {int(k): f.centroid[i] for i, k in enumerate(kept)}
        for i, k in enumerate(kept):
            a, b = pos.get(int(k) - 1), pos.get(int(k) + 1)
            if a is None or b is None:
                continue
            d = (b - a) / 2.0
            f.speed[i] = float(np.hypot(*d))
            f.speed_y[i] = float(d[1])
    return f, np.array(offs)


def dense_mono_facts(detections, camera: str) -> ViewFacts:
    """Sharpness, clipping and board travel for every detected view of a camera.

    The same fit-on-a-subset, evaluate-on-everything argument as
    `dense_stereo_series`, for the measures that need no calibration at all:
    they come from the pixels and the corner centroid, so every detected view
    can contribute whether or not it was in the solve. There is no reprojection
    error here -- those views were never fitted -- so `rms` stays NaN.
    """
    per = detections.per_camera.get(camera, {})
    keys = sorted((k for k, d in per.items() if d.ok),
                  key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
    f = ViewFacts(keys=list(keys))
    n = len(keys)
    if not n:
        return f
    idx = _numeric(keys)
    f.index = idx if idx is not None else np.arange(n, dtype=float)
    f.rms = np.full(n, np.nan)
    f.centroid = np.array([per[k].corners.mean(0) for k in keys], float)
    f.span = np.array([np.ptp(per[k].corners, axis=0).max() for k in keys], float)
    f.corners = np.array([per[k].count for k in keys], float)
    f.distance = np.full(n, np.nan)
    f.tilt = np.full(n, np.nan)
    for name in ("sharpness", "anisotropy", "board_white", "board_black", "board_contrast"):
        setattr(f, name, np.full(n, np.nan))
    for i, k in enumerate(keys):
        st = detections.stats_for(camera, k)
        if not st.ok:
            continue
        f.sharpness[i] = st.sharpness
        f.anisotropy[i] = st.anisotropy
        f.board_white[i] = st.board_white_frac
        f.board_black[i] = st.board_black_frac
        f.board_contrast[i] = st.board_contrast
    f.speed = np.full(n, np.nan)
    f.speed_y = np.full(n, np.nan)
    if idx is not None:
        pos = {int(k): f.centroid[i] for i, k in enumerate(keys)}
        for i, k in enumerate(keys):
            a, b = pos.get(int(k) - 1), pos.get(int(k) + 1)
            if a is None or b is None:
                continue
            d = (b - a) / 2.0
            f.speed[i] = float(np.hypot(*d))
            f.speed_y[i] = float(d[1])
    return f
