"""Command line: detect, solve, report, and hand off to the inspector."""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import signal
import sys
import time

import numpy as np
import yaml

from . import detect as detect_mod
from . import io as io_mod
from . import report as report_mod
from . import select as select_mod
from .board import ARUCO_DICTS, CHARUCO, CHECKERBOARD, Board
from .calibrate import CalibrationError, Model, calibrate_mono, calibrate_stereo
from .dataset import discover


# -- argument plumbing ------------------------------------------------------


def add_board_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("board")
    g.add_argument("--board", metavar="YAML",
                   help="board spec file; anything else on this line overrides it")
    g.add_argument("--board-kind", choices=[CHARUCO, CHECKERBOARD])
    g.add_argument("--columns", type=int,
                   help="charuco: squares across. checkerboard: INNER corners across")
    g.add_argument("--rows", type=int, help="same convention as --columns")
    g.add_argument("--square", type=float, metavar="MM", help="checker pitch in millimetres")
    g.add_argument("--marker", type=float, metavar="MM", help="charuco marker size in millimetres")
    g.add_argument("--dictionary", choices=sorted(ARUCO_DICTS))
    g.add_argument("--legacy-charuco", action="store_true",
                   help="board image was generated before OpenCV 4.6 changed the marker layout")


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("lens model")
    g.add_argument("--fisheye", action="store_true", help="equidistant model instead of pinhole")
    g.add_argument("--rational", action="store_true", help="add k4..k6")
    g.add_argument("--thin-prism", action="store_true", help="add s1..s4")
    g.add_argument("--tilted", action="store_true", help="add taux, tauy")
    g.add_argument("--fix-k3", action="store_true")
    g.add_argument("--zero-tangent", action="store_true")
    g.add_argument("--fix-principal-point", action="store_true")
    g.add_argument("--fix-aspect-ratio", action="store_true")


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("dataset", help="image directory, left/right parent, or a recording directory")
    p.add_argument("--left", help="explicit left image directory")
    p.add_argument("--right", help="explicit right image directory")
    p.add_argument("-o", "--out", default=None, metavar="DIR",
                   help="output directory (default: <dataset>/calibration)")
    p.add_argument("--views", default="60", metavar="N|all",
                   help="how many views to solve on; a spread is chosen greedily. "
                        "Cost grows about cubically with view count (default: 60)")
    p.add_argument("--min-corners", type=int, default=8,
                   help="drop a view with fewer corners than this (default: 8)")
    p.add_argument("--exclude", metavar="JSON",
                   help="selection file from the inspector; those views are left out")
    p.add_argument("--jobs", type=int, default=None, help="detection worker processes")
    p.add_argument("--redetect", action="store_true", help="ignore the detection cache")
    add_board_args(p)
    add_model_args(p)


def board_from_args(args) -> Board:
    spec: dict = {}
    if args.board:
        loaded = yaml.safe_load(pathlib.Path(args.board).read_text()) or {}
        spec = loaded.get("board", loaded)
    if args.board_kind:
        spec["kind"] = args.board_kind
    if args.columns:
        spec["columns"] = args.columns
    if args.rows:
        spec["rows"] = args.rows
    if args.square:
        spec["square_size"] = args.square / 1000.0
    if args.marker:
        spec["marker_size"] = args.marker / 1000.0
    if args.dictionary:
        spec["dictionary"] = args.dictionary
    if args.legacy_charuco:
        spec["legacy_pattern"] = True
    if spec.get("kind") == CHECKERBOARD:
        for k in ("marker_size", "dictionary", "legacy_pattern"):
            spec.pop(k, None)
    return Board.from_dict(spec) if spec else Board()


def model_from_args(args) -> Model:
    return Model(
        fisheye=args.fisheye, rational=args.rational, thin_prism=args.thin_prism,
        tilted=args.tilted, fix_k3=args.fix_k3, zero_tangent=args.zero_tangent,
        fix_principal_point=args.fix_principal_point, fix_aspect_ratio=args.fix_aspect_ratio,
    )


def _target_views(value: str) -> int:
    if str(value).lower() in ("all", "0", "none"):
        return 0
    return int(value)


_TTY = sys.stderr.isatty()


def _progress(done, total, name):
    """A redrawing bar on a terminal; a few plain lines when piped to a file."""
    if _TTY:
        bar = int(30 * done / max(total, 1))
        sys.stderr.write(f"\r  [{'#' * bar}{'.' * (30 - bar)}] {done}/{total} {name[:34]:<34}")
        if done >= total:
            sys.stderr.write("\n")
    elif total and (done == total or done % max(1, total // 5) == 0):
        sys.stderr.write(f"  {done}/{total}\n")
    sys.stderr.flush()


def prepare(args):
    """Everything up to the solve: dataset, board, detections, chosen views."""
    dataset = discover(args.dataset, args.left, args.right)
    board = board_from_args(args)
    out = pathlib.Path(args.out) if args.out else dataset.root / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    print(dataset.describe())
    print(board.describe())
    print("detecting...")
    detections = detect_mod.run(
        dataset, board, detect_mod.DetectorSettings(min_corners=args.min_corners),
        cache_path=out / "detections.pkl", workers=args.jobs,
        progress=_progress, force=args.redetect,
    )
    print(detections.summary(dataset))

    excluded = io_mod.read_selection(args.exclude) if args.exclude else set()
    if not args.exclude and (out / "selection.json").exists():
        excluded = io_mod.read_selection(out / "selection.json")
        if excluded:
            print(f"honouring {len(excluded)} exclusion(s) from {out / 'selection.json'}")
    keys = [v.key for v in dataset.views if v.key not in excluded]

    chosen, why = select_mod.select_views(
        detections.per_camera, detections.image_size[dataset.cameras[0]], keys,
        target=_target_views(args.views), min_corners=args.min_corners,
        require_all_cameras=dataset.is_stereo,
    )
    print(f"views: {why}")
    return dataset, board, detections, chosen, out


@contextlib.contextmanager
def interruptible():
    """Let Ctrl+C kill the process outright for the duration of this block.

    Python only runs a signal handler between bytecodes, and an OpenCV solve is
    one long call into C -- `stereoCalibrate` on a few hundred views does not
    come back for minutes. A Ctrl+C during it sits queued for the whole solve,
    which looks exactly like the key doing nothing. Handing SIGINT back to the
    kernel's default makes it immediate.

    Safe here because a solve writes nothing: there is no half-written file to
    unwind, and the detection cache was already flushed before this point.
    """
    previous = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except ValueError:
        previous = None      # not the main thread; leave the handler alone
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def solve(args, dataset, board, detections, keys):
    model = model_from_args(args)
    size = detections.image_size[dataset.cameras[0]]
    started = time.time()
    with interruptible():
        result = _run_solve(args, dataset, board, detections, keys, model, size)
    print(f"solved in {time.time() - started:.1f}s\n")
    return result


def _run_solve(args, dataset, board, detections, keys, model, size):
    if dataset.is_stereo and not getattr(args, "mono", False):
        return calibrate_stereo(
            board, detections.per_camera["left"], detections.per_camera["right"], size,
            keys=keys, model=model, min_corners=args.min_corners,
            fix_intrinsics=not getattr(args, "refine_intrinsics", False),
            alpha=getattr(args, "alpha", 0.0),
        )
    cam = dataset.cameras[0]
    return calibrate_mono(board, detections.per_camera[cam], size, keys=keys,
                          model=model, min_corners=args.min_corners, camera=cam)


def write_everything(result, out: pathlib.Path, dataset, args, detections=None) -> None:
    from .calibrate import StereoCalibration
    stereo = isinstance(result, StereoCalibration)
    data = (io_mod.stereo_to_dict(result, dataset) if stereo
            else io_mod.mono_to_dict(result, dataset))
    paths = [io_mod.write_yaml(data, out / "calibration.yaml"),
             io_mod.write_opencv(result, out / "calibration_opencv.yml")]
    if not args.no_plots:
        paths += report_mod.save_figures(result, out)
    if getattr(args, "pdf", False):
        from . import pdfreport
        paths.append(pdfreport.build(result, dataset, detections,
                                     out / "calibration_report.pdf"))
    if stereo and args.maps:
        paths.append(io_mod.write_maps(result, out / "rectify_maps.npz"))
    if stereo and not args.no_plots:
        import cv2
        key = next(iter(result.views))
        view = next(v for v in dataset.views if v.key == key)
        preview = report_mod.rectified_preview(
            result, cv2.imread(str(view.paths["left"])), cv2.imread(str(view.paths["right"])))
        p = out / "rectified_preview.jpg"
        cv2.imwrite(str(p), preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    print("wrote:")
    for p in paths:
        print(f"  {p}")


def report_suspects(result, args) -> None:
    flagged = report_mod.suspects(result, sigma=args.sigma)
    if not flagged:
        print("\nno views stand out from the rest.")
        return
    print(f"\n{len(flagged)} view(s) unlike the rest (>{args.sigma} robust sigma):")
    for key, why in flagged[:20]:
        print(f"  {key:>8}  {'; '.join(why)}")
    if len(flagged) > 20:
        print(f"  ... and {len(flagged) - 20} more")
    print("  -> inspect them with:  calib gui <dataset>")


# -- subcommands ------------------------------------------------------------


def cmd_detect(args) -> int:
    dataset, board, detections, chosen, out = prepare(args)
    print(f"\ndetection cache: {out / 'detections.pkl'} "
          f"({detections.seconds:.1f}s of work)" if detections.seconds
          else f"\ndetection cache: {out / 'detections.pkl'} (reused)")
    return 0


def cmd_calibrate(args) -> int:
    dataset, board, detections, chosen, out = prepare(args)
    try:
        result = solve(args, dataset, board, detections, chosen)
    except CalibrationError as exc:
        print(f"\ncalibration failed: {exc}", file=sys.stderr)
        return 1
    print(report_mod.summary(result))
    report_suspects(result, args)
    print()
    write_everything(result, out, dataset, args, detections)
    return 0


def cmd_auto(args) -> int:
    """Solve, drop what the robust cut flags, solve again. Twice, at most."""
    dataset, board, detections, chosen, out = prepare(args)
    dropped: dict[str, str] = {}
    result = None
    for round_no in range(1, args.rounds + 1):
        result = solve(args, dataset, board, detections, chosen)
        flagged = report_mod.suspects(result, sigma=args.sigma)
        print(f"round {round_no}: {len(chosen)} views, "
              f"rms {result.rms:.4f} px, {len(flagged)} flagged")
        if not flagged or round_no == args.rounds:
            break
        for key, why in flagged:
            dropped[key] = "; ".join(why)
        chosen = [k for k in chosen if k not in dropped]
        if len(chosen) < 8:
            print("  stopping: too few views left to keep cutting")
            break
    print()
    print(report_mod.summary(result))
    if dropped:
        print(f"\ndropped {len(dropped)} view(s):")
        for key, why in sorted(dropped.items())[:20]:
            print(f"  {key:>8}  {why}")
        io_mod.write_selection(out / "selection.json", dropped, dropped)
        print(f"  recorded in {out / 'selection.json'}")
    print()
    write_everything(result, out, dataset, args, detections)
    return 0


def cmd_alpha(args) -> int:
    """Show what each rectify alpha would cost, from an existing calibration.

    Choosing alpha needs an eye on the result, but the crop itself belongs to
    the solve: P1, P2, Q and the ROI are all functions of it. So this reads a
    calibration, shows the trade, and tells you what to re-run -- it never
    writes a rectification of its own.
    """
    from . import pdfreport
    path = pathlib.Path(args.calibration)
    if path.is_dir():
        path = path / "calibration.yaml"
    result = io_mod.read_stereo_yaml(path)
    sweep = pdfreport.alpha_sweep(result, alphas=args.alphas)
    if not sweep:
        print("alpha does not apply to a fisheye rectification (it takes balance instead)")
        return 1
    print(f"{path}\n")
    print(f"  {'alpha':>6}  {'rect fx':>9}  {'usable both eyes':>18}  valid roi (left)")
    for d in sweep:
        mark = "  <- in the file" if abs(d["alpha"] - result.rect.alpha) < 1e-9 else ""
        print(f"  {d['alpha']:>6.2f}  {d['fx']:>9.2f}  "
              f"{d['common'][0]:>5d}x{d['common'][1]:<5d} {d['common_frac'] * 100:5.1f}%  "
              f"{d['roi_left']}{mark}")
    print(f"\n  re-run the solve with the alpha you want:\n"
          f"    calib run <dataset> --alpha <value> --pdf")
    return 0


def cmd_gui(args) -> int:
    from .gui import launch
    return launch(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calib",
        description="Camera calibration for mono and stereo rigs, checkerboard or charuco.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  calib gui  recordings/recording-2026-08-22T13-00-20
  calib run  recordings/recording-2026-08-22T13-00-20 --views 60
  calib auto recordings/recording-2026-08-22T13-00-20 --sigma 3
  calib run  images/ --board-kind checkerboard --columns 9 --rows 6 --square 25
""")
    subs = p.add_subparsers(dest="command", required=True)

    d = subs.add_parser("detect", help="detect corners and fill the cache, nothing else")
    add_common_args(d)
    d.set_defaults(func=cmd_detect)

    for name, help_text in (("run", "detect, solve, report, write"),
                            ("auto", "run, then drop flagged views and re-solve")):
        c = subs.add_parser(name, help=help_text)
        add_common_args(c)
        c.add_argument("--mono", action="store_true",
                       help="calibrate the left camera alone even if the dataset is stereo")
        c.add_argument("--refine-intrinsics", action="store_true",
                       help="let the stereo solve re-fit intrinsics instead of holding "
                            "the per-camera solves fixed")
        c.add_argument("--alpha", type=float, default=0.0,
                       help="stereoRectify alpha: 0 crops to valid pixels, 1 keeps all "
                            "(default: 0)")
        c.add_argument("--sigma", type=float, default=3.0,
                       help="robust cut for flagging odd views (default: 3)")
        c.add_argument("--maps", action="store_true", help="also write rectify_maps.npz")
        c.add_argument("--no-plots", action="store_true", help="skip the diagnostic figures")
        c.add_argument("--pdf", action="store_true",
                       help="also write calibration_report.pdf: verdict, exposure and "
                            "clipping, coverage, sharpness and motion, pose dependence, "
                            "and for stereo the rig and alpha-sweep pages")
        if name == "auto":
            c.add_argument("--rounds", type=int, default=3, help="max solve/cut rounds")
        c.set_defaults(func=cmd_auto if name == "auto" else cmd_calibrate)

    g = subs.add_parser("gui", help="interactive inspector: images, residuals, reject, re-solve")
    add_common_args(g)
    g.add_argument("--mono", action="store_true")
    g.add_argument("--refine-intrinsics", action="store_true")
    g.add_argument("--alpha", type=float, default=0.0)
    g.add_argument("--sigma", type=float, default=3.0)
    g.add_argument("--maps", action="store_true")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--pdf", action="store_true")
    g.set_defaults(func=cmd_gui)

    a = subs.add_parser("alpha", help="what each rectify alpha would cost, from a solve")
    a.add_argument("calibration", help="calibration.yaml, or the directory holding it")
    a.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0], help="alphas to compare")
    a.set_defaults(func=cmd_alpha)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
