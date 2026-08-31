"""Ground-truth tests.

Real images can only tell you that a calibration came out badly, never whose
fault that is. So the chain -- id bookkeeping, shared-corner matching, pose
propagation, rectification, the epipolar and row-error metrics -- is checked
against a rig that is known exactly, by projecting a board through it and
handing the result back to the solver. If these pass and a real dataset still
solves badly, the problem is in front of the lens.

    python3 vortex-camera-calibration/tests/test_synthetic.py
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from ..board import Board
from ..calibrate import Model, calibrate_mono, calibrate_stereo
from ..detect import Detection

SIZE = (1440, 1080)
K1 = np.array([[1200.0, 0, 728.0], [0, 1200.0, 536.0], [0, 0, 1.0]])
K2 = np.array([[1190.0, 0, 712.0], [0, 1190.0, 548.0], [0, 0, 1.0]])
D1 = np.array([-0.21, 0.09, 0.0008, -0.0005, -0.014])
D2 = np.array([-0.19, 0.07, -0.0006, 0.0009, -0.010])
R_TRUE = cv2.Rodrigues(np.deg2rad(np.array([0.4, -1.2, 0.3])))[0]
T_TRUE = np.array([[-0.120], [0.002], [-0.004]])


def _poses(n: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for _ in range(n):
        rvec = np.deg2rad(rng.uniform(-28, 28, 3)).reshape(3, 1)
        tvec = np.array([[rng.uniform(-0.30, 0.10)],
                         [rng.uniform(-0.22, 0.22)],
                         [rng.uniform(1.0, 2.0)]])
        out.append((rvec, tvec))
    return out


def _observe(board, rvec, tvec, K, D, noise, rng, drop):
    """Project the board and keep the corners that land inside the frame."""
    objp = board.object_points()
    pts, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, K, D)
    pts = pts.reshape(-1, 2)
    inside = ((pts[:, 0] > 4) & (pts[:, 0] < SIZE[0] - 4)
              & (pts[:, 1] > 4) & (pts[:, 1] < SIZE[1] - 4))
    ids = np.flatnonzero(inside).astype(np.int32)
    if drop and len(ids) > 20:
        ids = np.sort(rng.choice(ids, size=len(ids) - drop, replace=False))
    pts = pts[ids] + rng.normal(0, noise, (len(ids), 2))
    return Detection(ids=ids, corners=pts.astype(np.float32))


def _check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got:.5f} (want {want:.5f} +- {tol})")
    return ok


def run(noise: float = 0.05, n_views: int = 30, seed: int = 7) -> bool:
    rng = np.random.default_rng(seed)
    board = Board(kind="charuco", columns=12, rows=9, square_size=0.060, marker_size=0.045)
    poses = _poses(n_views, rng)

    left, right, keys = {}, {}, []
    for i, (rvec, tvec) in enumerate(poses):
        key = str(i)
        # Different corners drop out of each eye, which is exactly the case the
        # shared-id matching has to get right.
        dl = _observe(board, rvec, tvec, K1, D1, noise, rng, drop=3)
        rvec_r = cv2.Rodrigues(R_TRUE @ cv2.Rodrigues(rvec)[0])[0]
        tvec_r = R_TRUE @ tvec + T_TRUE
        dr = _observe(board, rvec_r, tvec_r, K2, D2, noise, rng, drop=5)
        if dl.count < 20 or dr.count < 20:
            continue
        left[key], right[key] = dl, dr
        keys.append(key)
    print(f"synthetic rig: {len(keys)} views, {noise} px corner noise")

    ok = True
    mono = calibrate_mono(board, left, SIZE, keys, Model(), camera="left")
    ok &= _check("mono rms", mono.rms, noise, max(0.02, noise * 0.6))
    ok &= _check("mono fx", mono.K[0, 0], K1[0, 0], 6.0)
    ok &= _check("mono cx", mono.K[0, 2], K1[0, 2], 6.0)
    ok &= _check("mono k1", mono.dist[0], D1[0], 0.02)

    st = calibrate_stereo(board, left, right, SIZE, keys, Model())
    ok &= _check("stereo rms", st.rms, noise, max(0.03, noise * 0.8))
    ok &= _check("baseline mm", st.baseline * 1000, np.linalg.norm(T_TRUE) * 1000, 1.0)
    rod_true = np.degrees(cv2.Rodrigues(R_TRUE)[0].ravel())
    for i, axis in enumerate("xyz"):
        ok &= _check(f"rotation {axis} deg", st.rotation_deg[i], rod_true[i], 0.06)
    ok &= _check("epipolar rms", st.epipolar_rms(), 0.0, max(0.15, noise * 3))
    ok &= _check("rectified dy rms", st.rect_dy_rms(), 0.0, max(0.15, noise * 3))

    # The rectified baseline has to survive the trip through stereoRectify.
    ok &= _check("rectified baseline mm", st.rect.baseline * 1000,
                 np.linalg.norm(T_TRUE) * 1000, 1.0)

    # A checkerboard, where every view is the full grid and ids are positions.
    cb = Board(kind="checkerboard", columns=9, rows=6, square_size=0.030)
    cl, cr, ck = {}, {}, []
    for i, (rvec, tvec) in enumerate(_poses(24, np.random.default_rng(11))):
        objp = cb.object_points()
        p1, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, K1, D1)
        rvec_r = cv2.Rodrigues(R_TRUE @ cv2.Rodrigues(rvec)[0])[0]
        p2, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec_r, R_TRUE @ tvec + T_TRUE, K2, D2)
        p1, p2 = p1.reshape(-1, 2), p2.reshape(-1, 2)
        if not (p1 > 4).all() or not (p2 > 4).all():
            continue
        if p1[:, 0].max() > SIZE[0] - 4 or p2[:, 0].max() > SIZE[0] - 4:
            continue
        if p1[:, 1].max() > SIZE[1] - 4 or p2[:, 1].max() > SIZE[1] - 4:
            continue
        ids = np.arange(cb.n_points, dtype=np.int32)
        cl[str(i)] = Detection(ids, (p1 + rng.normal(0, noise, p1.shape)).astype(np.float32))
        cr[str(i)] = Detection(ids, (p2 + rng.normal(0, noise, p2.shape)).astype(np.float32))
        ck.append(str(i))
    print(f"checkerboard: {len(ck)} full-grid views")
    st_cb = calibrate_stereo(cb, cl, cr, SIZE, ck, Model())
    ok &= _check("checkerboard baseline mm", st_cb.baseline * 1000,
                 np.linalg.norm(T_TRUE) * 1000, 1.0)
    ok &= _check("checkerboard rectified dy", st_cb.rect_dy_rms(), 0.0, max(0.15, noise * 3))

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
