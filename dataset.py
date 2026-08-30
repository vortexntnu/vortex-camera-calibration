"""Finding the images and deciding which of them belong together.

A *view* is one instant: one image for mono, one per camera for stereo. Views
are the unit everything else works in -- detection results, per-view errors, and
the include/exclude decision you make in the inspector all hang off a view key.

Pairing is by the digits in the filename, not by sort order. `left_000042.png`
pairs with `right_000042.png` and a gap on one side shifts nothing, which is the
whole point: a dropped frame on one camera silently offsetting every later pair
is the kind of bug that produces a beautiful mono calibration and a baseline
that is quietly wrong.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}

# Common names for the two eyes, in the order we prefer to find them.
_LEFT_NAMES = ("left", "cam0", "l", "0")
_RIGHT_NAMES = ("right", "cam1", "r", "1")

_DIGITS = re.compile(r"(\d+)")


def _key_of(path: pathlib.Path) -> str:
    """The trailing number in a filename, which is what identifies the instant."""
    found = _DIGITS.findall(path.stem)
    return found[-1].lstrip("0") or "0" if found else path.stem


def _images_in(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class View:
    key: str
    paths: dict[str, pathlib.Path]

    @property
    def label(self) -> str:
        return next(iter(self.paths.values())).stem


@dataclass
class Dataset:
    cameras: list[str]
    views: list[View]
    root: pathlib.Path
    unpaired: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_stereo(self) -> bool:
        return len(self.cameras) == 2

    def __len__(self) -> int:
        return len(self.views)

    def paths_for(self, camera: str) -> list[pathlib.Path]:
        return [v.paths[camera] for v in self.views if camera in v.paths]

    def describe(self) -> str:
        kind = "stereo" if self.is_stereo else "mono"
        out = f"{kind} dataset at {self.root}: {len(self.views)} views, cameras {self.cameras}"
        for cam, keys in self.unpaired.items():
            if keys:
                shown = ", ".join(keys[:6]) + (" ..." if len(keys) > 6 else "")
                out += f"\n  {len(keys)} unpaired on {cam}: {shown}"
        return out


def _subdir(root: pathlib.Path, names) -> pathlib.Path | None:
    for n in names:
        for candidate in (root / n, root / n.upper(), root / n.capitalize()):
            if candidate.is_dir() and _images_in(candidate):
                return candidate
    return None


def discover(
    root: str | pathlib.Path,
    left: str | pathlib.Path | None = None,
    right: str | pathlib.Path | None = None,
) -> Dataset:
    """Work out what kind of dataset `root` is and pair up the views.

    Explicit `left`/`right` directories win. Otherwise: a directory holding
    left/right (or cam0/cam1) subdirectories is stereo, a directory of images is
    mono, and a recording directory is redirected into its `calib/` subdirectory
    so you can point this at the recording itself.
    """
    root = pathlib.Path(root).expanduser().resolve()

    if left or right:
        if not (left and right):
            raise ValueError("give both --left and --right, or neither")
        return _pair(pathlib.Path(left).resolve(), pathlib.Path(right).resolve(), root)

    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        root = root.parent

    if (root / "calib").is_dir():
        root = root / "calib"

    ldir, rdir = _subdir(root, _LEFT_NAMES), _subdir(root, _RIGHT_NAMES)
    if ldir and rdir:
        return _pair(ldir, rdir, root)

    images = _images_in(root)
    if images:
        return Dataset(
            cameras=["cam"],
            views=[View(_key_of(p), {"cam": p}) for p in images],
            root=root,
        )

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir() and _images_in(d)]
    if len(subdirs) == 1:
        images = _images_in(subdirs[0])
        return Dataset(["cam"], [View(_key_of(p), {"cam": p}) for p in images], subdirs[0])

    raise FileNotFoundError(
        f"no images under {root}. Expected either image files directly, or "
        f"left/ and right/ subdirectories."
    )


def _pair(ldir: pathlib.Path, rdir: pathlib.Path, root: pathlib.Path) -> Dataset:
    lmap = {_key_of(p): p for p in _images_in(ldir)}
    rmap = {_key_of(p): p for p in _images_in(rdir)}
    shared = sorted(set(lmap) & set(rmap), key=lambda k: (len(k), k))
    if not shared:
        raise ValueError(
            f"{ldir.name}/ and {rdir.name}/ share no frame numbers -- filenames must "
            f"carry a matching trailing number for pairing to mean anything"
        )
    return Dataset(
        cameras=["left", "right"],
        views=[View(k, {"left": lmap[k], "right": rmap[k]}) for k in shared],
        root=root,
        unpaired={
            "left": sorted(set(lmap) - set(rmap)),
            "right": sorted(set(rmap) - set(lmap)),
        },
    )
