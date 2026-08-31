#!/usr/bin/env python3
"""Entry point, runnable from anywhere:

    python3 /home/jorgen/vortex/vortex-camera-calibration/main.py run <dataset>

This directory's name contains hyphens, so it is not a legal Python module
name: it can be neither imported by name nor started with `python -m`. Load it
from its path as `calibtool` instead, so the package-relative imports inside it
resolve normally.
"""
import importlib.util
import pathlib
import sys

pkg_dir = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "calibtool", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
)
package = importlib.util.module_from_spec(spec)
sys.modules["calibtool"] = package
spec.loader.exec_module(package)

from calibtool.cli import main  # noqa: E402

sys.exit(main())
