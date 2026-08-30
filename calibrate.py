"""Solving, and the numbers you need to distrust the solve.

The headline RMS from `calibrateCamera` is a single number over every corner of
every view, and it is the least interesting thing calibration produces. It goes
down when you delete views and it says nothing about *where* the model is wrong.
So everything here is built to hand back per-view and per-corner residuals
alongside the parameters: `MonoCalibration.views[key].residual` is a (K, 2)
array of pixel errors you can draw as arrows, and the arrows are what tell you
whether a view is genuinely bad or whether your distortion model is too small
to describe the corners of the frame.

For stereo there are three separate error numbers and they answer different
questions:

  reprojection      does the board pose explain both images? Catches a bad view.
  epipolar (Sampson) do the two eyes agree on the geometry, independent of pose?
  rectified dy      after rectification, how far off the same row does a corner
                    land? This is the one that decides whether a block matcher
                    will work, and it is the number to quote.

A rig can post a fine reprojection RMS and still have a rectified dy of two
pixels, and stereo matching will be soft everywhere without ever looking broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import Board
from .detect import Detection

TERM = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-8)

# Above this many views, skip the "Extended" calibration entry points. They also
# return parameter standard deviations, and getting those means inverting the
# full dense (9 + 6N) x (9 + 6N) normal-equation matrix -- which turns a
# one-second solve into a minutes-long one somewhere around a hundred views. The
# parameters themselves are unaffected; you only lose the 1-sigma column.
STD_MAX_VIEWS = 80

# Least a view's corners may collapse toward a single line before it is unusable.
# See `_spread_ratio`: 0 is a line, 1 is a disc. Every full board view sits far
# above this -- one row of a 9-wide board scores 0, and two rows score 0.19 --
# so the default only rejects the genuinely degenerate.
MIN_SPREAD = 0.05


class CalibrationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    """Which lens model to fit, as a set of independent decisions.

    Defaults are the plain 5-coefficient pinhole model, which is what you want
    for most machine-vision lenses. Reach for more only when the residual arrows
    show a systematic pattern -- a radial swirl at the frame edge means the
    radial terms are under-powered, and adding coefficients you cannot see the
    need for just fits noise and makes undistortion misbehave outside the
    corner coverage you actually had.
    """

    fisheye: bool = False            # the equidistant model; excludes the rest
    rational: bool = False           # k4..k6
    thin_prism: bool = False         # s1..s4
    tilted: bool = False             # taux, tauy
    fix_k3: bool = False
    zero_tangent: bool = False
    fix_principal_point: bool = False
    fix_aspect_ratio: bool = False

    def flags(self) -> int:
        if self.fisheye:
            # CHECK_COND is deliberately off: it raises on an ill-conditioned
            # view instead of reporting it, and finding those is this tool's job.
            f = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
            if self.fix_principal_point:
                f |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
            return f
        f = 0
        if self.rational:
            f |= cv2.CALIB_RATIONAL_MODEL
        if self.thin_prism:
            f |= cv2.CALIB_THIN_PRISM_MODEL
        if self.tilted:
            f |= cv2.CALIB_TILTED_MODEL
        if self.fix_k3:
            f |= cv2.CALIB_FIX_K3
        if self.zero_tangent:
            f |= cv2.CALIB_ZERO_TANGENT_DIST
        if self.fix_principal_point:
            f |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if self.fix_aspect_ratio:
            f |= cv2.CALIB_FIX_ASPECT_RATIO
        return f

    @property
    def n_dist(self) -> int:
        """How many distortion coefficients this model actually uses.

        OpenCV hands back a 14-long vector whenever any extended term is on,
        zero-padded past the ones it fitted. Writing all fourteen out implies a
        tilted-sensor model was fitted when it was not, so results are trimmed
        to this length. The order is fixed:
        k1 k2 p1 p2 k3 | k4 k5 k6 | s1 s2 s3 s4 | taux tauy.
        """
        if self.fisheye:
            return 4
        if self.tilted:
            return 14
        if self.thin_prism:
            return 12
        if self.rational:
            return 8
        return 5

    def describe(self) -> str:
        if self.fisheye:
            return "fisheye (equidistant, k1..k4)"
        parts = ["pinhole k1,k2,p1,p2,k3"]
        if self.rational:
            parts.append("rational k4..k6")
        if self.thin_prism:
            parts.append("thin prism s1..s4")
        if self.tilted:
            parts.append("tilted taux,tauy")
        for name, on in (("fix_k3", self.fix_k3), ("zero_tangent", self.zero_tangent),
                         ("fix_principal_point", self.fix_principal_point),
                         ("fix_aspect_ratio", self.fix_aspect_ratio)):
            if on:
                parts.append(name)
        return " + ".join(parts)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------


def project(objp: np.ndarray, rvec, tvec, K, dist, fisheye: bool) -> np.ndarray:
    """Object points -> pixels, for whichever model is in play."""
    objp = np.asarray(objp, np.float64).reshape(-1, 1, 3)
    if fisheye:
        pts, _ = cv2.fisheye.projectPoints(objp, np.asarray(rvec, np.float64).reshape(3, 1),
                                           np.asarray(tvec, np.float64).reshape(3, 1),
                                           K, np.asarray(dist, np.float64).reshape(4, 1))
    else:
        pts, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    return np.asarray(pts, np.float64).reshape(-1, 2)


def solve_pose(objp, imgp, K, dist, fisheye: bool):
    """Board pose from one view. Fisheye goes through normalised coordinates,
    because solvePnP has no equidistant model of its own."""
    objp = np.asarray(objp, np.float64).reshape(-1, 1, 3)
    imgp = np.asarray(imgp, np.float64).reshape(-1, 1, 2)
    if fisheye:
        norm = cv2.fisheye.undistortPoints(imgp, K, np.asarray(dist, np.float64).reshape(4, 1))
        ok, rvec, tvec = cv2.solvePnP(objp, norm, np.eye(3), np.zeros(4),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise CalibrationError("solvePnP failed")
    return rvec, tvec


@dataclass
class ViewResidual:
    """One view in one camera, after the solve."""

    key: str
    ids: np.ndarray
    observed: np.ndarray     # (K, 2)
    projected: np.ndarray    # (K, 2)
    rvec: np.ndarray
    tvec: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        return self.observed - self.projected

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.residual, axis=1)

    @property
    def rms(self) -> float:
        r = self.residual
        return float(np.sqrt((r ** 2).sum() / len(r))) if len(r) else float("nan")

    @property
    def max(self) -> float:
        m = self.magnitude
        return float(m.max()) if len(m) else float("nan")

    @property
    def distance(self) -> float:
        """Board distance from the camera, metres -- coverage in depth matters
        as much as coverage in the frame."""
        return float(np.linalg.norm(self.tvec))

    @property
    def tilt_deg(self) -> float:
        """Angle between the board normal and the optical axis. All-frontal
        views leave focal length and distance traded off against each other."""
        R = cv2.Rodrigues(np.asarray(self.rvec, np.float64))[0]
        return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


@dataclass
class MonoCalibration:
    camera: str
    board: Board
    model: Model
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray
    rms: float
    views: dict[str, ViewResidual] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    std_intrinsics: np.ndarray | None = None
    # Set only when this camera came out of a stereo solve: the rms this camera
    # reached on its own, before the rig's geometry was imposed on it. The gap
    # between the two is the interesting number -- it is what the stereo
    # constraint costs, and a large one means the pair does not agree.
    mono_rms: float | None = None

    @property
    def keys(self) -> list[str]:
        return list(self.views)

    @property
    def fov_deg(self) -> tuple[float, float]:
        w, h = self.image_size
        return (float(np.degrees(2 * np.arctan(w / (2 * self.K[0, 0])))),
                float(np.degrees(2 * np.arctan(h / (2 * self.K[1, 1])))))

    @property
    def n_corners(self) -> int:
        return sum(len(v.ids) for v in self.views.values())

    def per_view_rms(self) -> dict[str, float]:
        return {k: v.rms for k, v in self.views.items()}

    def all_residuals(self) -> np.ndarray:
        if not self.views:
            return np.zeros((0, 2))
        return np.concatenate([v.residual for v in self.views.values()])

    def all_observed(self) -> np.ndarray:
        if not self.views:
            return np.zeros((0, 2))
        return np.concatenate([v.observed for v in self.views.values()])


def _spread_ratio(pts: np.ndarray) -> float:
    """How far a set of coplanar points is from lying on one line: 0 line, 1 disc.

    The ratio of the two singular values of the mean-centred coordinates -- the
    minor extent of the point cloud over its major one.

    A corner count is not enough to qualify a view. OpenCV bootstraps the focal
    length in `initIntrinsicParams2D` by fitting one homography per view, and a
    homography needs points spanning two dimensions. Nine corners along a single
    board row clear a `min_corners` of 8 and still have no unique board->image
    mapping, so `findHomography` hands back an empty matrix and the assertion
    behind it takes down the whole solve. Cheaper to drop the view here and say
    why.
    """
    p = np.asarray(pts, np.float64).reshape(len(pts), -1)[:, :2]
    if len(p) < 3:
        return 0.0
    s = np.linalg.svd(p - p.mean(0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def _view_inputs(board: Board, detections: dict[str, Detection], keys, min_corners: int,
                 min_spread: float = MIN_SPREAD):
    """Split the requested views into (usable, reason-they-are-not)."""
    used, obj, img, skipped = [], [], [], {}
    for key in keys:
        det = detections.get(key)
        if det is None or not det.ok:
            skipped[key] = det.error if det is not None and det.error else "no detection"
            continue
        if det.count < min_corners:
            skipped[key] = f"only {det.count} corners (min {min_corners})"
            continue
        o = board.object_points_for(det.ids).astype(np.float32).reshape(-1, 1, 3)
        c = det.corners.astype(np.float32).reshape(-1, 1, 2)
        # Both ends: the board patch can span two dimensions and still project to
        # a line when the board is seen edge-on.
        spread = min(_spread_ratio(o), _spread_ratio(c))
        if spread < min_spread:
            skipped[key] = f"corners nearly collinear (spread {spread:.3f}, min {min_spread})"
            continue
        used.append(key)
        obj.append(o)
        img.append(c)
    return used, obj, img, skipped


def calibrate_mono(
    board: Board,
    detections: dict[str, Detection],
    image_size: tuple[int, int],
    keys=None,
    model: Model | None = None,
    min_corners: int = 8,
    camera: str = "cam",
    min_spread: float = MIN_SPREAD,
) -> MonoCalibration:
    """Intrinsics for one camera from its detections."""
    model = model or Model()
    keys = list(detections) if keys is None else [k for k in keys if k in detections]
    used, obj, img, skipped = _view_inputs(board, detections, keys, min_corners, min_spread)
    if len(used) < 3:
        raise CalibrationError(
            f"{camera}: {len(used)} usable view(s), need at least 3. "
            f"{len(skipped)} were skipped."
        )

    if model.fisheye:
        obj64 = [o.astype(np.float64).reshape(1, -1, 3) for o in obj]
        img64 = [i.astype(np.float64).reshape(1, -1, 2) for i in img]
        K = np.eye(3)
        dist = np.zeros((4, 1))
        rms, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj64, img64, image_size, K, dist, flags=model.flags(), criteria=TERM
        )
        std = None
    elif len(used) <= STD_MAX_VIEWS:
        rms, K, dist, rvecs, tvecs, std, _, _ = cv2.calibrateCameraExtended(
            obj, img, image_size, None, None, flags=model.flags(), criteria=TERM
        )
    else:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, image_size, None, None, flags=model.flags(), criteria=TERM
        )
        std = None

    result = MonoCalibration(
        camera=camera, board=board, model=model, image_size=tuple(image_size),
        K=np.asarray(K, np.float64),
        dist=np.asarray(dist, np.float64).ravel()[:model.n_dist],
        rms=float(rms), skipped=skipped,
        std_intrinsics=None if std is None else np.asarray(std, np.float64).ravel(),
    )
    for key, o, i, rvec, tvec in zip(used, obj, img, rvecs, tvecs):
        result.views[key] = ViewResidual(
            key=key,
            ids=detections[key].ids,
            observed=i.reshape(-1, 2).astype(np.float64),
            projected=project(o, rvec, tvec, result.K, result.dist, model.fisheye),
            rvec=np.asarray(rvec, np.float64).reshape(3),
            tvec=np.asarray(tvec, np.float64).reshape(3),
        )
    return result


# ---------------------------------------------------------------------------


@dataclass
class Rectification:
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: tuple
    roi2: tuple

    # What was asked of stereoRectify. Recorded because P1, P2, Q and the ROIs
    # are all functions of these two, so a reader who wants to reproduce or
    # re-crop the rectification needs to know what produced the numbers above.
    # Provenance only -- nothing downstream should re-derive P or Q from them.
    alpha: float = 0.0
    rectified_size: tuple = ()

    @property
    def baseline(self) -> float:
        """Rectified baseline in metres: -Tx' / fx."""
        return abs(self.P2[0, 3] / self.P2[0, 0])


@dataclass
class StereoViewResidual:
    key: str
    ids: np.ndarray
    left: ViewResidual
    right: ViewResidual
    epipolar: np.ndarray   # (K,) Sampson distance, pixels
    rect_dy: np.ndarray    # (K,) row disagreement after rectification, pixels
    # What this one view, on its own, thinks the rig looks like: the board solved
    # independently in each camera, then one pose composed onto the other. It
    # uses no stereo constraint at all, which is what makes it worth having --
    # see `rig_spread`.
    rel_rvec_deg: np.ndarray | None = None   # (3,)
    rel_t_mm: np.ndarray | None = None       # (3,)

    @property
    def rms(self) -> float:
        r = np.concatenate([self.left.residual, self.right.residual])
        return float(np.sqrt((r ** 2).sum() / len(r)))

    @property
    def epipolar_rms(self) -> float:
        return float(np.sqrt(np.mean(self.epipolar ** 2)))

    @property
    def rect_dy_rms(self) -> float:
        return float(np.sqrt(np.mean(self.rect_dy ** 2)))

    @property
    def rect_dy_max(self) -> float:
        return float(np.abs(self.rect_dy).max())


@dataclass
class StereoCalibration:
    board: Board
    model: Model
    image_size: tuple[int, int]
    left: MonoCalibration
    right: MonoCalibration
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rms: float
    rect: Rectification
    views: dict[str, StereoViewResidual] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    fixed_intrinsics: bool = True
    # The two per-camera solves as they stood *before* the rig was imposed,
    # residuals and all. Kept rather than reduced to `mono_rms` because the
    # comparison is the diagnosis: `left`/`right` above carry stereo-constrained
    # residuals over shared corners, these carry each camera's own fit over
    # every corner it saw, and a large gap between them is the rig failing, not
    # the lens.
    mono_left: "MonoCalibration | None" = None
    mono_right: "MonoCalibration | None" = None

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def rotation_deg(self) -> np.ndarray:
        """Right camera orientation relative to left, as roll/pitch/yaw degrees.
        A well-built rig is a couple of degrees at most; more is worth knowing
        about, because rectification has to throw away field of view to fix it."""
        rvec = cv2.Rodrigues(self.R)[0].ravel()
        return np.degrees(rvec)

    def epipolar_rms(self) -> float:
        if not self.views:
            return float("nan")
        e = np.concatenate([v.epipolar for v in self.views.values()])
        return float(np.sqrt(np.mean(e ** 2)))

    def rect_dy_rms(self) -> float:
        if not self.views:
            return float("nan")
        d = np.concatenate([v.rect_dy for v in self.views.values()])
        return float(np.sqrt(np.mean(d ** 2)))

    def rig_spread(self) -> dict:
        """How much each view disagrees about where the two cameras are.

        Every view gets its own answer for the rig geometry, from two
        independent PnP solves and no stereo constraint. On a rigid rig with
        sound intrinsics those answers pile up on one value, and the spread is
        just corner noise -- for a board a thousand pixels across that is on the
        order of a thousandth of a degree.

        A spread far above that means no single R, T can satisfy every view, and
        the stereo solve is being asked for something impossible. Read the
        rotation spread in pixels before judging it: a hundredth of a degree is
        nothing, until you multiply by a 2300 px focal length and it is half a
        pixel of unavoidable reprojection error.
        """
        rot = np.array([v.rel_rvec_deg for v in self.views.values()
                        if v.rel_rvec_deg is not None])
        trans = np.array([v.rel_t_mm for v in self.views.values()
                          if v.rel_t_mm is not None])
        if not len(rot):
            return {}
        focal = float((self.left.K[0, 0] + self.right.K[0, 0]) / 2)
        # Robust spread: outlying views should not set the scale of the answer.
        rot_mad = 1.4826 * np.median(np.abs(rot - np.median(rot, axis=0)), axis=0)
        return {
            "views": len(rot),
            "rotation_median_deg": np.median(rot, axis=0),
            "rotation_spread_deg": rot_mad,
            "translation_median_mm": np.median(trans, axis=0),
            "translation_spread_mm": 1.4826 * np.median(
                np.abs(trans - np.median(trans, axis=0)), axis=0),
            # The worst axis, converted to the reprojection error it forces.
            "rotation_spread_px": float(np.deg2rad(rot_mad.max()) * focal),
        }


def _sampson(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Distance from each correspondence to satisfying x2^T F x1 = 0, in pixels.

    The points must already be undistorted: F is a projective relation between
    pinhole rays, and feeding it raw pixels off a lens with any real k1 reports
    the distortion back to you as epipolar error. See `_undistort_pixels`.
    """
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.abs(np.einsum("ij,ij->i", x2, Fx1))
    den = np.sqrt(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
    return num / np.maximum(den, 1e-12)


def _undistort_pixels(pts, K, dist, fisheye: bool) -> np.ndarray:
    """Detected pixels moved to where an ideal pinhole would have put them."""
    src = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if fisheye:
        out = cv2.fisheye.undistortPoints(src, K, np.asarray(dist, np.float64).reshape(4, 1), P=K)
    else:
        out = cv2.undistortPoints(src, K, dist, P=K)
    return np.asarray(out, np.float64).reshape(-1, 2)


def rectify_points(pts: np.ndarray, K, dist, R_rect, P, fisheye: bool) -> np.ndarray:
    """Where a detected corner lands in the rectified image."""
    src = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if fisheye:
        out = cv2.fisheye.undistortPoints(src, K, np.asarray(dist, np.float64).reshape(4, 1),
                                          R=R_rect, P=P)
    else:
        out = cv2.undistortPoints(src, K, dist, R=R_rect, P=P)
    return np.asarray(out, np.float64).reshape(-1, 2)


def calibrate_stereo(
    board: Board,
    left_detections: dict[str, Detection],
    right_detections: dict[str, Detection],
    image_size: tuple[int, int],
    keys=None,
    model: Model | None = None,
    min_corners: int = 8,
    fix_intrinsics: bool = True,
    mono_left: MonoCalibration | None = None,
    mono_right: MonoCalibration | None = None,
    alpha: float = 0.0,
    min_spread: float = MIN_SPREAD,
) -> StereoCalibration:
    """Extrinsics for the pair, plus the rectification they imply.

    By default each camera's intrinsics are solved on its own first and then held
    fixed (`fix_intrinsics`). That is usually right: the mono solve gets every
    view where that camera saw the board, while the stereo solve only gets views
    where *both* did, so letting stereo re-fit the intrinsics throws away data
    and lets a lens error hide inside the extrinsics. Unfix it when the two
    cameras' coverage is nearly identical and you want the joint optimum.
    """
    model = model or Model()
    all_keys = list(keys) if keys is not None else sorted(
        set(left_detections) | set(right_detections)
    )

    mono_left = mono_left or calibrate_mono(
        board, left_detections, image_size, all_keys, model, min_corners, "left",
        min_spread=min_spread)
    mono_right = mono_right or calibrate_mono(
        board, right_detections, image_size, all_keys, model, min_corners, "right",
        min_spread=min_spread)

    obj, img_l, img_r, used, skipped = [], [], [], [], {}
    for key in all_keys:
        dl, dr = left_detections.get(key), right_detections.get(key)
        if dl is None or dr is None or not dl.ok or not dr.ok:
            skipped[key] = "not detected in both cameras"
            continue
        shared = np.intersect1d(dl.ids, dr.ids)
        if len(shared) < min_corners:
            skipped[key] = f"only {len(shared)} corners seen by both (min {min_corners})"
            continue
        li = {int(i): n for n, i in enumerate(dl.ids)}
        ri = {int(i): n for n, i in enumerate(dr.ids)}
        o = board.object_points_for(shared).astype(np.float32).reshape(-1, 1, 3)
        cl = dl.corners[[li[int(i)] for i in shared]].astype(np.float32).reshape(-1, 1, 2)
        cr = dr.corners[[ri[int(i)] for i in shared]].astype(np.float32).reshape(-1, 1, 2)
        # The intersection can be collinear even when neither camera's own view
        # was -- two overlapping detections meeting along one board row.
        spread = min(_spread_ratio(o), _spread_ratio(cl), _spread_ratio(cr))
        if spread < min_spread:
            skipped[key] = f"shared corners nearly collinear (spread {spread:.3f})"
            continue
        obj.append(o)
        img_l.append(cl)
        img_r.append(cr)
        used.append((key, shared))
    if len(used) < 3:
        raise CalibrationError(
            f"{len(used)} view(s) where both cameras saw at least {min_corners} shared "
            f"corners; need at least 3."
        )

    K1, d1 = mono_left.K.copy(), mono_left.dist.copy()
    K2, d2 = mono_right.K.copy(), mono_right.dist.copy()

    if model.fisheye:
        flags = model.flags() | (cv2.fisheye.CALIB_FIX_INTRINSIC if fix_intrinsics else 0)
        rms, K1, d1, K2, d2, R, T = cv2.fisheye.stereoCalibrate(
            [o.astype(np.float64).reshape(1, -1, 3) for o in obj],
            [i.astype(np.float64).reshape(1, -1, 2) for i in img_l],
            [i.astype(np.float64).reshape(1, -1, 2) for i in img_r],
            K1, d1.reshape(4, 1), K2, d2.reshape(4, 1), image_size,
            flags=flags, criteria=TERM,
        )
        R, T = np.asarray(R, np.float64), np.asarray(T, np.float64).reshape(3, 1)
        d1 = np.asarray(d1).ravel()[:model.n_dist]
        d2 = np.asarray(d2).ravel()[:model.n_dist]
        # No essential/fundamental matrix from the fisheye path; build them from
        # the pinhole-equivalent K so the epipolar metric still means something.
        rvecs = tvecs = None
    else:
        flags = model.flags() | (
            cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else cv2.CALIB_USE_INTRINSIC_GUESS
        )
        rms, K1, d1, K2, d2, R, T, E, F, rvecs, tvecs, _ = cv2.stereoCalibrateExtended(
            obj, img_l, img_r, K1, d1, K2, d2, image_size,
            np.eye(3), np.zeros((3, 1)), flags=flags, criteria=TERM,
        )
        d1 = np.asarray(d1).ravel()[:model.n_dist]
        d2 = np.asarray(d2).ravel()[:model.n_dist]

    Tx = np.array([[0, -T[2, 0], T[1, 0]], [T[2, 0], 0, -T[0, 0]], [-T[1, 0], T[0, 0], 0]])
    E = Tx @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    if model.fisheye:
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, d1.reshape(4, 1), K2, d2.reshape(4, 1), image_size, R, T,
            cv2.CALIB_ZERO_DISPARITY, newImageSize=image_size, balance=alpha, fov_scale=1.0,
        )
        roi1 = roi2 = (0, 0, image_size[0], image_size[1])
    else:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, d1, K2, d2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
        )
    rect = Rectification(np.asarray(R1), np.asarray(R2), np.asarray(P1), np.asarray(P2),
                         np.asarray(Q), tuple(roi1), tuple(roi2),
                         alpha=float(alpha), rectified_size=tuple(image_size))

    # Re-derive the mono containers so their intrinsics match what stereo settled
    # on -- otherwise the residual arrows you look at are not the ones the
    # extrinsics were fitted against.
    left = MonoCalibration("left", board, model, tuple(image_size), np.asarray(K1), d1,
                           float("nan"), skipped=dict(mono_left.skipped),
                           std_intrinsics=mono_left.std_intrinsics, mono_rms=mono_left.rms)
    right = MonoCalibration("right", board, model, tuple(image_size), np.asarray(K2), d2,
                            float("nan"), skipped=dict(mono_right.skipped),
                            std_intrinsics=mono_right.std_intrinsics, mono_rms=mono_right.rms)

    result = StereoCalibration(
        board=board, model=model, image_size=tuple(image_size), left=left, right=right,
        R=np.asarray(R), T=np.asarray(T).reshape(3, 1), E=E, F=F, rms=float(rms),
        rect=rect, skipped=skipped, fixed_intrinsics=fix_intrinsics,
        mono_left=mono_left, mono_right=mono_right,
    )

    for n, ((key, shared), o, li, ri) in enumerate(zip(used, obj, img_l, img_r)):
        pl = li.reshape(-1, 2).astype(np.float64)
        pr = ri.reshape(-1, 2).astype(np.float64)
        if rvecs is not None:
            rvec_l, tvec_l = np.asarray(rvecs[n], np.float64).reshape(3, 1), \
                np.asarray(tvecs[n], np.float64).reshape(3, 1)
        else:
            rvec_l, tvec_l = solve_pose(o, pl, K1, d1, model.fisheye)
        Rl = cv2.Rodrigues(rvec_l)[0]
        rvec_r = cv2.Rodrigues(np.asarray(R) @ Rl)[0]
        tvec_r = np.asarray(R) @ tvec_l.reshape(3, 1) + np.asarray(T).reshape(3, 1)

        vl = ViewResidual(key, shared, pl, project(o, rvec_l, tvec_l, K1, d1, model.fisheye),
                          rvec_l.ravel(), tvec_l.ravel())
        vr = ViewResidual(key, shared, pr, project(o, rvec_r, tvec_r, K2, d2, model.fisheye),
                          rvec_r.ravel(), tvec_r.ravel())
        left.views[key] = vl
        right.views[key] = vr

        yl = rectify_points(pl, K1, d1, rect.R1, rect.P1, model.fisheye)
        yr = rectify_points(pr, K2, d2, rect.R2, rect.P2, model.fisheye)
        # Independent per-camera pose for this view: no stereo constraint, so
        # the spread across views measures the rig, not the fit. See rig_spread.
        try:
            rl_i, tl_i = solve_pose(o, pl, K1, d1, model.fisheye)
            rr_i, tr_i = solve_pose(o, pr, K2, d2, model.fisheye)
            R_rel = cv2.Rodrigues(rr_i)[0] @ cv2.Rodrigues(rl_i)[0].T
            rel_rvec = np.degrees(cv2.Rodrigues(R_rel)[0].ravel())
            rel_t = (tr_i.reshape(3) - R_rel @ tl_i.reshape(3)) * 1000.0
        except CalibrationError:
            rel_rvec = rel_t = None

        result.views[key] = StereoViewResidual(
            key=key, ids=shared, left=vl, right=vr,
            rel_rvec_deg=rel_rvec, rel_t_mm=rel_t,
            epipolar=_sampson(F, _undistort_pixels(pl, K1, d1, model.fisheye),
                              _undistort_pixels(pr, K2, d2, model.fisheye)),
            rect_dy=yl[:, 1] - yr[:, 1],
        )
    # The headline rms has to describe the same residuals the per-view numbers
    # do, or the two disagree in the report and neither can be trusted.
    for cal in (left, right):
        res = cal.all_residuals()
        cal.rms = float(np.sqrt((res ** 2).sum() / len(res))) if len(res) else float("nan")
    return result


# ---------------------------------------------------------------------------


def outliers(per_view: dict[str, float], threshold: float | None = None,
             sigma: float = 3.0, floor_ratio: float = 2.0) -> list[str]:
    """Views to consider dropping.

    With an explicit threshold this is just "above that". Without one it is a
    MAD-based cut, which is the honest default: the mean and standard deviation
    of a set of errors are themselves dragged around by the outliers you are
    looking for, so the median absolute deviation does the job instead.

    `floor_ratio` stops the cut eating a healthy set. On a tight distribution the
    MAD is tiny and three sigma lands barely above the median, so a view a few
    hundredths of a pixel worse than its neighbours gets flagged -- which is
    noise, not a bad image. Nothing is flagged unless it is also at least
    `floor_ratio` times the median -- twice as bad as a typical view -- so
    "unusual" always means "and meaningfully worse".
    """
    if not per_view:
        return []
    keys = list(per_view)
    vals = np.array([per_view[k] for k in keys], float)
    if threshold is None:
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        threshold = med + sigma * (1.4826 * mad if mad > 0 else vals.std() or 1e-9)
        threshold = max(threshold, med * floor_ratio)
    return [k for k, v in zip(keys, vals) if v > threshold]
