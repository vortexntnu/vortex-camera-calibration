"""Board geometry -- the one place that knows what the target looks like.

Two kinds are supported, and they are deliberately made to look identical to
everything downstream:

  checkerboard  plain OpenCV chessboard. Detection is all-or-nothing: either the
                full grid of inner corners is found or the view is useless.
                Corner ids are just positions in that grid.
  charuco       chessboard with an ArUco marker in every white square. Corners
                carry a globally unique id, so a partial view still contributes
                and a half-occluded board is not a wasted frame.

Downstream code only ever sees `(ids, corners)` plus `object_points_for(ids)`,
so a partial charuco view and a full checkerboard view take exactly the same
path through detection, calibration and the residual maths.

Sizes are metres. The unit only scales translation -- intrinsics and reprojection
errors come out identical whatever you use -- but having the baseline pop out in
metres is worth the discipline.

One asymmetry worth knowing before you pick: a plain checkerboard with an even
number of inner corners on both axes is 180-degree ambiguous, and nothing in the
image resolves it. Within one camera that is harmless (the object points rotate
with it). Across a stereo pair it is not: if the two eyes order the same physical
board differently, the correspondence is silently wrong and the extrinsics are
garbage that still reports a plausible mono RMS. Use an odd x even inner-corner
count, or use charuco, which has no such failure mode.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

CHECKERBOARD = "checkerboard"
CHARUCO = "charuco"

# Every DICT_* constant the installed OpenCV exposes, by name.
ARUCO_DICTS = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


@dataclass(frozen=True)
class Board:
    """A calibration target.

    For a checkerboard, `columns` and `rows` count *inner corners* -- the OpenCV
    convention, and one less on each axis than the squares you can see.
    For charuco they count *squares*, which is what board generators print on
    the sheet, so the number you read off the board is the number you type.
    """

    kind: str = CHARUCO
    columns: int = 12
    rows: int = 9
    square_size: float = 0.060           # metres, checker pitch
    marker_size: float = 0.045           # metres, charuco only
    dictionary: str = "DICT_5X5_100"     # charuco only
    legacy_pattern: bool = False         # charuco boards generated pre-OpenCV 4.6

    def __post_init__(self) -> None:
        if self.kind not in (CHECKERBOARD, CHARUCO):
            raise ValueError(f"unknown board kind {self.kind!r}")
        if self.columns < 2 or self.rows < 2:
            raise ValueError("board needs at least 2 columns and 2 rows")
        if self.square_size <= 0:
            raise ValueError("square_size must be positive")
        if self.kind == CHARUCO:
            if self.dictionary not in ARUCO_DICTS:
                raise ValueError(
                    f"unknown aruco dictionary {self.dictionary!r}; "
                    f"have {', '.join(sorted(ARUCO_DICTS))}"
                )
            if not 0 < self.marker_size < self.square_size:
                raise ValueError("marker_size must be positive and smaller than square_size")

    # -- geometry ---------------------------------------------------------

    @property
    def inner_corners(self) -> tuple[int, int]:
        """(cols, rows) of chessboard corners that can carry an id."""
        if self.kind == CHECKERBOARD:
            return self.columns, self.rows
        return self.columns - 1, self.rows - 1

    @property
    def n_points(self) -> int:
        c, r = self.inner_corners
        return c * r

    @property
    def size_metres(self) -> tuple[float, float]:
        if self.kind == CHECKERBOARD:
            return (self.columns + 1) * self.square_size, (self.rows + 1) * self.square_size
        return self.columns * self.square_size, self.rows * self.square_size

    def object_points(self) -> np.ndarray:
        """(N, 3) board-frame coordinates, indexed by corner id."""
        if self.kind == CHARUCO:
            return np.asarray(self._cv_board().getChessboardCorners(), dtype=np.float32)
        c, r = self.inner_corners
        grid = np.zeros((r * c, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2) * self.square_size
        return grid

    def object_points_for(self, ids: np.ndarray) -> np.ndarray:
        return self.object_points()[np.asarray(ids).ravel()]

    # -- OpenCV objects ---------------------------------------------------
    #
    # Built on demand and never stored: cv2 detector objects do not pickle, and
    # these have to survive a trip to a worker process.

    def _cv_dictionary(self):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[self.dictionary])

    def _cv_board(self):
        if self.kind != CHARUCO:
            raise TypeError("only a charuco board has a cv2.aruco.CharucoBoard")
        board = cv2.aruco.CharucoBoard(
            (self.columns, self.rows), self.square_size, self.marker_size, self._cv_dictionary()
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.kind == CHECKERBOARD:
            for k in ("marker_size", "dictionary", "legacy_pattern"):
                d.pop(k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown board key(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    def describe(self) -> str:
        c, r = self.inner_corners
        w, h = self.size_metres
        head = (
            f"{self.kind} {self.columns}x{self.rows}"
            f"{' squares' if self.kind == CHARUCO else ' inner corners'}, "
            f"{self.square_size * 1000:g} mm pitch"
        )
        if self.kind == CHARUCO:
            head += f", {self.marker_size * 1000:g} mm markers, {self.dictionary}"
        return f"{head}\n  {c}x{r} = {self.n_points} corner ids, board {w * 1000:.0f}x{h * 1000:.0f} mm"
