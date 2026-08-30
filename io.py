"""Writing the result out, in the two forms something else will want it.

`calibration.yaml` is plain YAML for people and for yaml-cpp -- the same shape
as the rest of this repo's configs, readable in a diff, and carrying the
provenance (which views, which board, which model, what the errors were) that
turns a number into evidence six months later.

`calibration_opencv.yml` is the same parameters in cv::FileStorage form, so C++
can do `fs["K1"] >> K1` and stop. Both are written; they are small.

The selection file is separate on purpose. Which views you kept is an input to
calibration, not an output of it, and keeping it apart means you can re-solve
with a different model against exactly the set you curated.
"""

from __future__ import annotations

import datetime
import json
import pathlib

import cv2
import numpy as np
import yaml

from .board import Board
from .calibrate import Model, MonoCalibration, StereoCalibration


def _m(a) -> list:
    return np.asarray(a, np.float64).tolist()


def _stats(values) -> dict:
    a = np.asarray(list(values), float)
    if not len(a):
        return {}
    return {
        "rms": float(np.sqrt(np.mean(a ** 2))),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _mono_block(c: MonoCalibration) -> dict:
    return {
        "image_size": [int(c.image_size[0]), int(c.image_size[1])],
        "model": "fisheye" if c.model.fisheye else "pinhole",
        "camera_matrix": _m(c.K),
        "distortion": _m(c.dist),
        "fx": float(c.K[0, 0]), "fy": float(c.K[1, 1]),
        "cx": float(c.K[0, 2]), "cy": float(c.K[1, 2]),
        "fov_deg": [round(v, 3) for v in c.fov_deg],
        "reprojection": {
            "rms_px": float(c.rms),
            **({} if c.mono_rms is None else {"rms_px_this_camera_alone": float(c.mono_rms)}),
            "views": len(c.views),
            "corners": int(c.n_corners),
            "per_view": _stats(c.per_view_rms().values()),
        },
    }


def _provenance(board: Board, model: Model, dataset=None, keys=None) -> dict:
    p = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "opencv": cv2.__version__,
        "board": board.to_dict(),
        "model": model.to_dict(),
        "model_description": model.describe(),
    }
    if dataset is not None:
        p["dataset"] = str(dataset.root)
        p["views_available"] = len(dataset.views)
    if keys is not None:
        p["views_used"] = len(keys)
    return p


def mono_to_dict(c: MonoCalibration, dataset=None) -> dict:
    return {
        "calibration": {"type": "mono", **_provenance(c.board, c.model, dataset, c.keys)},
        "camera": _mono_block(c),
        "views": {
            "used": sorted(c.views, key=_sortkey),
            "skipped": {k: v for k, v in sorted(c.skipped.items(), key=lambda kv: _sortkey(kv[0]))},
            "per_view_rms_px": {k: round(v, 4) for k, v in
                                sorted(c.per_view_rms().items(), key=lambda kv: _sortkey(kv[0]))},
        },
    }


def stereo_to_dict(s: StereoCalibration, dataset=None) -> dict:
    return {
        "calibration": {"type": "stereo",
                        **_provenance(s.board, s.model, dataset, list(s.views))},
        "left": _mono_block(s.left),
        "right": _mono_block(s.right),
        "extrinsics": {
            "note": "R, T map a point in the LEFT camera frame into the RIGHT camera frame.",
            "R": _m(s.R),
            "T": _m(s.T.ravel()),
            "rotation_rodrigues_deg": [round(v, 5) for v in s.rotation_deg],
            "baseline_m": float(s.baseline),
            "essential": _m(s.E),
            "fundamental": _m(s.F),
            "intrinsics_held_fixed": bool(s.fixed_intrinsics),
        },
        "rectification": {
            # Provenance: what stereoRectify was asked for. P1, P2, Q and the
            # ROIs below are functions of these, so they are recorded to make
            # the rectification reproducible -- not for anything downstream to
            # re-derive P or Q from. One producer, one set of numbers.
            "alpha": float(s.rect.alpha),
            "rectified_size": [int(v) for v in (s.rect.rectified_size or s.image_size)],
            "R1": _m(s.rect.R1), "R2": _m(s.rect.R2),
            "P1": _m(s.rect.P1), "P2": _m(s.rect.P2), "Q": _m(s.rect.Q),
            "valid_roi_left": [int(v) for v in s.rect.roi1],
            "valid_roi_right": [int(v) for v in s.rect.roi2],
            "rectified_baseline_m": float(s.rect.baseline),
            "rectified_fx": float(s.rect.P1[0, 0]),
        },
        "errors": {
            "stereo_rms_px": float(s.rms),
            "reprojection_left": _stats([v.left.rms for v in s.views.values()]),
            "reprojection_right": _stats([v.right.rms for v in s.views.values()]),
            "epipolar_sampson_px": _stats(
                np.concatenate([v.epipolar for v in s.views.values()]) if s.views else []),
            "rectified_row_error_px": _stats(
                np.abs(np.concatenate([v.rect_dy for v in s.views.values()])) if s.views else []),
        },
        "rig_consistency": _rig_block(s),
        "views": {
            "used": sorted(s.views, key=_sortkey),
            "skipped": {k: v for k, v in sorted(s.skipped.items(), key=lambda kv: _sortkey(kv[0]))},
            "per_view": {
                k: {"rms_px": round(v.rms, 4),
                    "epipolar_px": round(v.epipolar_rms, 4),
                    "rect_dy_px": round(v.rect_dy_rms, 4),
                    "corners": int(len(v.ids))}
                for k, v in sorted(s.views.items(), key=lambda kv: _sortkey(kv[0]))
            },
        },
    }


def _rig_block(s: StereoCalibration) -> dict:
    """Spread of the per-view, stereo-free estimate of the rig geometry.
    Large rotation spread means no single R, T fits every view; see
    StereoCalibration.rig_spread."""
    spread = s.rig_spread()
    if not spread:
        return {}
    return {
        "note": "each view solved independently in both cameras, no stereo constraint",
        "views": int(spread["views"]),
        "rotation_median_deg": _m(spread["rotation_median_deg"]),
        "rotation_spread_deg": _m(spread["rotation_spread_deg"]),
        "translation_median_mm": _m(spread["translation_median_mm"]),
        "translation_spread_mm": _m(spread["translation_spread_mm"]),
        "rotation_spread_as_px": float(spread["rotation_spread_px"]),
    }


def _sortkey(k: str):
    return (0, int(k)) if str(k).isdigit() else (1, str(k))


class _Dumper(yaml.SafeDumper):
    """SafeDumper that also knows numpy scalars.

    Without this, a stray np.float64 -- and they leak in from anything that
    touched an array -- fails the dump with "cannot represent an object" after
    the whole calibration has already been computed.
    """


_Dumper.add_multi_representer(
    np.floating, lambda d, v: d.represent_float(float(v)))
_Dumper.add_multi_representer(
    np.integer, lambda d, v: d.represent_int(int(v)))
_Dumper.add_multi_representer(
    np.bool_, lambda d, v: d.represent_bool(bool(v)))
_Dumper.add_representer(
    np.ndarray, lambda d, v: d.represent_list(v.tolist()))


def write_yaml(data: dict, path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(data, fh, Dumper=_Dumper, sort_keys=False,
                  default_flow_style=None, width=100)
    return path


def write_opencv(result, path) -> pathlib.Path:
    """The same numbers in cv::FileStorage form."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        if isinstance(result, StereoCalibration):
            fs.write("image_width", int(result.image_size[0]))
            fs.write("image_height", int(result.image_size[1]))
            fs.write("K1", result.left.K)
            fs.write("D1", result.left.dist.reshape(1, -1))
            fs.write("K2", result.right.K)
            fs.write("D2", result.right.dist.reshape(1, -1))
            fs.write("R", result.R)
            fs.write("T", result.T.reshape(3, 1))
            fs.write("E", result.E)
            fs.write("F", result.F)
            fs.write("R1", result.rect.R1)
            fs.write("R2", result.rect.R2)
            fs.write("P1", result.rect.P1)
            fs.write("P2", result.rect.P2)
            fs.write("Q", result.rect.Q)
            fs.write("baseline_m", float(result.baseline))
            fs.write("model", "fisheye" if result.model.fisheye else "pinhole")
        else:
            fs.write("image_width", int(result.image_size[0]))
            fs.write("image_height", int(result.image_size[1]))
            fs.write("K", result.K)
            fs.write("D", result.dist.reshape(1, -1))
            fs.write("model", "fisheye" if result.model.fisheye else "pinhole")
    finally:
        fs.release()
    return path


def write_maps(s: StereoCalibration, path) -> pathlib.Path:
    """Precomputed remap LUTs, so the runtime does not have to re-derive them."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    undistort = cv2.fisheye.initUndistortRectifyMap if s.model.fisheye \
        else cv2.initUndistortRectifyMap
    d1 = s.left.dist.reshape(4, 1) if s.model.fisheye else s.left.dist
    d2 = s.right.dist.reshape(4, 1) if s.model.fisheye else s.right.dist
    l1, l2 = undistort(s.left.K, d1, s.rect.R1, s.rect.P1, s.image_size, cv2.CV_32FC1)
    r1, r2 = undistort(s.right.K, d2, s.rect.R2, s.rect.P2, s.image_size, cv2.CV_32FC1)
    np.savez_compressed(path, left_map_x=l1, left_map_y=l2, right_map_x=r1, right_map_y=r2,
                        Q=s.rect.Q, roi1=np.array(s.rect.roi1), roi2=np.array(s.rect.roi2))
    return path


# -- selection --------------------------------------------------------------


def write_selection(path, excluded, reasons=None) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"excluded": sorted(excluded, key=_sortkey), "reasons": reasons or {}}, indent=2))
    return path


def read_selection(path) -> set[str]:
    path = pathlib.Path(path)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("excluded", []))


def read_stereo_yaml(path) -> StereoCalibration:
    """Load a written calibration back into a StereoCalibration.

    Only what the file actually holds: intrinsics, extrinsics and the
    rectification. Per-view residuals are not stored and do not come back, so
    the result is enough to re-derive rectification and to inspect geometry, and
    not enough to re-run the diagnostics -- those need the detections.
    """
    import yaml as _yaml
    from .board import Board
    from .calibrate import Model, MonoCalibration, Rectification, StereoCalibration

    path = pathlib.Path(path)
    data = _yaml.safe_load(path.read_text()) or {}
    if data.get("calibration", {}).get("type") != "stereo":
        raise ValueError(f"{path}: not a stereo calibration")

    board = Board.from_dict(data["calibration"]["board"])
    spec = data["calibration"].get("model", {})
    model = Model(**{f: bool(spec.get(f, False)) for f in Model.__dataclass_fields__})
    size = tuple(int(v) for v in data["left"]["image_size"])

    def mono(name):
        b = data[name]
        return MonoCalibration(
            name, board, model, size,
            np.array(b["camera_matrix"], float), np.array(b["distortion"], float),
            float(b.get("reprojection", {}).get("rms_px", float("nan"))),
            mono_rms=b.get("reprojection", {}).get("rms_px_this_camera_alone"),
        )

    ex, rc = data["extrinsics"], data["rectification"]
    rect = Rectification(
        np.array(rc["R1"], float), np.array(rc["R2"], float),
        np.array(rc["P1"], float), np.array(rc["P2"], float), np.array(rc["Q"], float),
        tuple(rc.get("valid_roi_left", (0, 0, *size))),
        tuple(rc.get("valid_roi_right", (0, 0, *size))),
        alpha=float(rc.get("alpha", 0.0)),
        rectified_size=tuple(rc.get("rectified_size", size)),
    )
    return StereoCalibration(
        board=board, model=model, image_size=size, left=mono("left"), right=mono("right"),
        R=np.array(ex["R"], float), T=np.array(ex["T"], float).reshape(3, 1),
        E=np.array(ex["essential"], float), F=np.array(ex["fundamental"], float),
        rms=float(data.get("errors", {}).get("stereo_rms_px", float("nan"))),
        rect=rect, fixed_intrinsics=bool(ex.get("intrinsics_held_fixed", True)),
    )
