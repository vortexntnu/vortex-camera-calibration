"""Corner detection, cached.

Detection is the slow half of calibrating (a few hundred milliseconds per
1440x1080 charuco image) and the half you never want to repeat. Everything the
inspector does -- rejecting a view, re-solving, sweeping a distortion model --
runs off the cache, so the loop between "that one looks wrong" and "here is the
calibration without it" stays interactive.

The cache is keyed on the board spec, the detector settings and the size and
mtime of every image. Change any of those and it is silently rebuilt; change
nothing and it loads in milliseconds.
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import pathlib
import pickle
import signal
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import CHARUCO, CHECKERBOARD, Board
from .dataset import Dataset
from .imagestats import ImageStats
from .imagestats import measure as measure_image

CACHE_VERSION = 6


@dataclass
class Detection:
    """What one image gave us. `ids` indexes into `Board.object_points()`."""

    ids: np.ndarray                       # (K,) int32
    corners: np.ndarray                   # (K, 2) float32, pixels
    marker_ids: np.ndarray | None = None  # charuco only, for display
    marker_corners: np.ndarray | None = None  # (M, 4, 2), charuco only
    error: str = ""                       # why nothing was found

    @property
    def count(self) -> int:
        return 0 if self.ids is None else int(len(self.ids))

    @property
    def ok(self) -> bool:
        return self.count > 0


EMPTY = Detection(np.zeros(0, np.int32), np.zeros((0, 2), np.float32))


@dataclass
class DetectorSettings:
    """Knobs that change what comes out of detection, so they key the cache."""

    # charuco: a chessboard corner is only interpolated if at least this many of
    # its neighbouring markers were read. 2 is OpenCV's default and is a good
    # trade; 1 finds more corners at the board edge and trusts them less.
    min_markers: int = 2
    # charuco: re-find markers the first pass missed, using the board layout as a
    # prior. Costs time, and on a board that is mostly visible it earns it back.
    refine_markers: bool = True
    # checkerboard: findChessboardCornersSB is slower than the classic detector
    # and much better on blur and glare. Turn it off only if SB is failing.
    use_sb: bool = True
    # Drop a view outright below this many corners -- too few to constrain a pose.
    min_corners: int = 8

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Per-image detection. Runs in a worker process, so it must build its own cv2
# objects: detectors are not picklable.

_WORKER: dict = {}


def _worker_init(board: Board, settings: DetectorSettings) -> None:
    # Ctrl+C goes to the whole process group, so without this every worker
    # raises KeyboardInterrupt at once, the pool breaks mid-map, and the parent
    # gets a BrokenProcessPool instead of a clean stop. Workers ignore it; the
    # parent decides when to shut them down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER.clear()
    _WORKER["board"] = board
    _WORKER["settings"] = settings
    if board.kind == CHARUCO:
        cv_board = board._cv_board()
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = settings.min_markers
        charuco_params.tryRefineMarkers = settings.refine_markers
        detector_params = cv2.aruco.DetectorParameters()
        # Charuco corners are refined from the chessboard, not the markers, so we
        # can afford a permissive marker pass: more markers read means more
        # corners survive the min_markers test.
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        detector_params.adaptiveThreshWinSizeMax = 43
        _WORKER["detector"] = cv2.aruco.CharucoDetector(
            cv_board, charuco_params, detector_params, cv2.aruco.RefineParameters()
        )


def detect_image(gray: np.ndarray, board: Board, settings: DetectorSettings,
                 detector=None) -> Detection:
    """Detect on a single grayscale image. Pure -- no cache, no I/O."""
    if board.kind == CHARUCO:
        if detector is None:
            cparams = cv2.aruco.CharucoParameters()
            cparams.minMarkers = settings.min_markers
            cparams.tryRefineMarkers = settings.refine_markers
            detector = cv2.aruco.CharucoDetector(board._cv_board(), cparams)
        corners, ids, m_corners, m_ids = detector.detectBoard(gray)
        if ids is None or len(ids) < settings.min_corners:
            n = 0 if ids is None else len(ids)
            markers = 0 if m_ids is None else len(m_ids)
            return Detection(
                np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                error=f"{n} corners from {markers} markers (need {settings.min_corners})",
            )
        return Detection(
            ids=np.asarray(ids, np.int32).ravel(),
            corners=np.asarray(corners, np.float32).reshape(-1, 2),
            marker_ids=None if m_ids is None else np.asarray(m_ids, np.int32).ravel(),
            marker_corners=(
                None if m_corners is None
                else np.asarray(m_corners, np.float32).reshape(-1, 4, 2)
            ),
        )

    # Checkerboard. SB first; the classic detector is the fallback because it
    # still occasionally wins on very low-contrast prints.
    pattern = board.inner_corners
    found, corners = False, None
    if settings.use_sb:
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
            )
    if not found:
        return Detection(np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                         error="chessboard not found")
    return Detection(
        ids=np.arange(board.n_points, dtype=np.int32),
        corners=np.asarray(corners, np.float32).reshape(-1, 2),
    )


def _detect_path(path: str):
    gray = load_gray(path)
    if gray is None:
        return path, Detection(np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                               error="unreadable"), None, ImageStats()
    det = detect_image(gray, _WORKER["board"], _WORKER["settings"], _WORKER.get("detector"))
    # Measured here because this is the only place the pixels exist. Reading
    # every image a second time at report time would cost more than the whole
    # detection pass, and would go stale against the cache.
    stats = measure_image(gray, det.corners if det.ok else None)
    return path, det, (gray.shape[1], gray.shape[0]), stats


def load_gray(path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


# ---------------------------------------------------------------------------


@dataclass
class DetectionSet:
    """Detections for one dataset, one per (camera, view key)."""

    board: Board
    settings: DetectorSettings
    image_size: dict[str, tuple[int, int]] = field(default_factory=dict)
    per_camera: dict[str, dict[str, Detection]] = field(default_factory=dict)
    # Pixel statistics alongside the corners, same keying. See imagestats: this
    # is where exposure and blur live, and neither is visible in a corner list.
    stats: dict[str, dict[str, ImageStats]] = field(default_factory=dict)
    seconds: float = 0.0

    def stats_for(self, camera: str, key: str) -> ImageStats:
        # getattr, not self.stats: a cache pickled before this field existed
        # unpickles without it, and a missing measurement is not an error.
        return getattr(self, "stats", {}).get(camera, {}).get(key, ImageStats())

    def get(self, camera: str, key: str) -> Detection:
        return self.per_camera.get(camera, {}).get(key, EMPTY)

    def counts(self, camera: str) -> dict[str, int]:
        return {k: d.count for k, d in self.per_camera.get(camera, {}).items()}

    def summary(self, dataset: Dataset) -> str:
        lines = []
        for cam in dataset.cameras:
            counts = np.array([self.get(cam, v.key).count for v in dataset.views])
            good = counts > 0
            lines.append(
                f"  {cam:>5}: {good.sum()}/{len(counts)} images with corners, "
                f"median {int(np.median(counts[good])) if good.any() else 0}"
                f"/{self.board.n_points} corners"
            )
        if dataset.is_stereo:
            both = sum(
                1 for v in dataset.views
                if self.get("left", v.key).ok and self.get("right", v.key).ok
            )
            lines.append(f"  {'both':>5}: {both}/{len(dataset.views)} views usable for stereo")
        return "\n".join(lines)


def _fingerprint(dataset: Dataset, board: Board, settings: DetectorSettings) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(
        {"v": CACHE_VERSION, "board": board.to_dict(), "settings": settings.to_dict()},
        sort_keys=True,
    ).encode())
    for view in dataset.views:
        for cam in sorted(view.paths):
            p = view.paths[cam]
            st = p.stat()
            h.update(f"{cam}|{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def run(
    dataset: Dataset,
    board: Board,
    settings: DetectorSettings | None = None,
    cache_path: str | pathlib.Path | None = None,
    workers: int | None = None,
    progress=None,
    force: bool = False,
) -> DetectionSet:
    """Detect over the whole dataset, using and updating the cache."""
    settings = settings or DetectorSettings()
    cache_path = pathlib.Path(cache_path) if cache_path else None
    fingerprint = _fingerprint(dataset, board, settings)

    if cache_path and cache_path.exists() and not force:
        try:
            with cache_path.open("rb") as fh:
                blob = pickle.load(fh)
            if blob.get("fingerprint") == fingerprint:
                if progress:
                    progress(len(dataset.views) * len(dataset.cameras),
                             len(dataset.views) * len(dataset.cameras), "cached")
                return blob["detections"]
        except Exception:
            pass  # a corrupt cache is not worth a traceback; just redo the work

    jobs = [(cam, v.key, str(v.paths[cam])) for v in dataset.views for cam in dataset.cameras
            if cam in v.paths]
    result = DetectionSet(board=board, settings=settings,
                          per_camera={c: {} for c in dataset.cameras},
                          stats={c: {} for c in dataset.cameras})
    by_path = {path: (cam, key) for cam, key, path in jobs}

    started = time.time()
    workers = workers or min(8, len(jobs)) or 1
    # Not a `with` block: its __exit__ calls shutdown(wait=True), which on an
    # interrupt blocks until every worker has finished the chunk it is in. That
    # wait is the hang -- Ctrl+C looks ignored for several seconds, or forever
    # if a worker is wedged on a bad file.
    pool = futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(board, settings))
    try:
        for done, (path, det, size, stats) in enumerate(
            pool.map(_detect_path, [p for _, _, p in jobs], chunksize=4), start=1
        ):
            cam, key = by_path[path]
            result.per_camera[cam][key] = det
            result.stats[cam][key] = stats
            if size and cam not in result.image_size:
                result.image_size[cam] = size
            elif size and result.image_size[cam] != size:
                raise ValueError(
                    f"{path}: {size} but {cam} started at {result.image_size[cam]} -- "
                    f"one calibration cannot cover two image sizes"
                )
            if progress:
                progress(done, len(jobs), pathlib.Path(path).name)
    except BaseException:
        # Kill rather than wait. The workers ignore SIGINT by design, so asking
        # politely means asking them to finish the whole queue first.
        #
        # Deliberately not shutdown(cancel_futures=True): cancelling races the
        # executor's own management thread, which then tries to complete
        # futures it has already cancelled and dumps an InvalidStateError
        # traceback over the exit. Killing the workers ends the work just as
        # dead, and quietly.
        for proc in list(getattr(pool, "_processes", {}).values()):
            for stop in (proc.terminate, proc.kill):
                try:
                    stop()
                except Exception:
                    pass
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
        try:
            pool.shutdown(wait=False)
        except Exception:
            pass
        raise
    pool.shutdown(wait=True)
    result.seconds = time.time() - started

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"fingerprint": fingerprint, "detections": result}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    return result
