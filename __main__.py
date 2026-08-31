"""Runs when this directory is given to the interpreter: `python3 <dir> ...`.

Delegates to main.py, which loads the package by path — the directory name
contains hyphens, so relative imports cannot resolve from here directly.
"""
import pathlib
import runpy
import sys

sys.argv[0] = str(pathlib.Path(__file__).resolve().parent / "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")
