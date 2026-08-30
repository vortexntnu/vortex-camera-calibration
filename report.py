"""Turning a solve into something you can argue with.

Every plot here exists to answer a question the RMS cannot:

  coverage        did the board ever visit the corners of the frame? Distortion
                  is only constrained where you put corners, and undistorting
                  outside that region is extrapolation.
  residual field  binned mean residual across the image. Random speckle means
                  the model fits. A coherent swirl or a radial bloom at the edge
                  means it does not, and no amount of dropping views will help.
  per-view error  which frames are dragging the fit, and by how much over the
                  rest -- the input to deciding what to reject.
  pose spread     distance and tilt of the board across the set. All-frontal,
                  all-at-one-distance views leave focal length and depth traded
                  off against each other, and that shows up later as a scale
                  error rather than as a large RMS.
  rectified dy    stereo only, and the number that actually predicts whether a
                  block matcher will work.
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np

from .calibrate import MonoCalibration, StereoCalibration, outliers


def _mpl():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg",):
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


# -- text -------------------------------------------------------------------


def _fmt_stats(a) -> str:
    a = np.asarray(list(a), float)
    if not len(a):
        return "n/a"
    return (f"rms {np.sqrt(np.mean(a ** 2)):.3f}  median {np.median(a):.3f}  "
            f"p95 {np.percentile(a, 95):.3f}  max {a.max():.3f}")


def mono_summary(c: MonoCalibration) -> str:
    fx, fy, cx, cy = c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]
    w, h = c.image_size
    lines = [
        f"[{c.camera}] {c.model.describe()}",
        f"  views used     {len(c.views)}  ({len(c.skipped)} skipped)   corners {c.n_corners}",
        f"  reprojection   rms {c.rms:.4f} px" + (
            "" if c.mono_rms is None
            else f"   [this camera alone: {c.mono_rms:.4f} px]"),
        f"  per-view       {_fmt_stats(c.per_view_rms().values())}",
        f"  focal          fx {fx:.2f}  fy {fy:.2f}  (aspect {fy / fx:.5f})",
        f"  principal      cx {cx:.2f}  cy {cy:.2f}"
        f"   [offset from centre {cx - w / 2:+.1f}, {cy - h / 2:+.1f} px]",
        f"  fov            {c.fov_deg[0]:.1f} x {c.fov_deg[1]:.1f} deg",
        f"  distortion     {np.array2string(c.dist, precision=5, suppress_small=True)}",
    ]
    if c.std_intrinsics is not None and len(c.std_intrinsics) >= 4:
        s = c.std_intrinsics
        lines.append(f"  std (1 sigma)  fx {s[0]:.2f}  fy {s[1]:.2f}  cx {s[2]:.2f}  cy {s[3]:.2f}")
    if c.views:
        d = np.array([v.distance for v in c.views.values()])
        t = np.array([v.tilt_deg for v in c.views.values()])
        lines.append(f"  board distance {d.min():.2f} - {d.max():.2f} m (median {np.median(d):.2f})")
        lines.append(f"  board tilt     {t.min():.0f} - {t.max():.0f} deg (median {np.median(t):.0f})")
        lines.append("  coverage       " + coverage_text(c))
    return "\n".join(lines)


def coverage_text(c: MonoCalibration, bins: int = 8) -> str:
    pts = c.all_observed()
    if not len(pts):
        return "no corners"
    w, h = c.image_size
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[[0, w], [0, h]])
    filled = int((hist > 0).sum())
    edge = np.concatenate([hist[0], hist[-1], hist[:, 0], hist[:, -1]])
    return (f"{filled}/{bins * bins} cells of the frame reached"
            f"{'' if (edge > 0).all() else '  [WARNING: frame border not covered]'}")


def stereo_summary(s: StereoCalibration) -> str:
    r = s.rotation_deg
    lines = [
        mono_summary(s.left), "", mono_summary(s.right), "",
        "[stereo]",
        f"  views used     {len(s.views)}  ({len(s.skipped)} skipped)",
        f"  intrinsics     {'held fixed from the mono solves' if s.fixed_intrinsics else 're-fitted jointly'}",
        f"  stereo rms     {s.rms:.4f} px",
        f"  baseline       {s.baseline * 1000:.2f} mm",
        f"  T (l->r)       [{s.T[0, 0] * 1000:+.2f}, {s.T[1, 0] * 1000:+.2f}, {s.T[2, 0] * 1000:+.2f}] mm",
        f"  R (l->r)       rodrigues [{r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f}] deg",
        f"  reproj left    {_fmt_stats([v.left.rms for v in s.views.values()])}",
        f"  reproj right   {_fmt_stats([v.right.rms for v in s.views.values()])}",
        f"  epipolar       {_fmt_stats(np.concatenate([v.epipolar for v in s.views.values()]) if s.views else [])} px",
        f"  rectified dy   {_fmt_stats(np.abs(np.concatenate([v.rect_dy for v in s.views.values()])) if s.views else [])} px",
        f"  rectified fx   {s.rect.P1[0, 0]:.2f} px   baseline*fx {s.rect.baseline * s.rect.P1[0, 0]:.1f} px.m",
        f"  valid roi      left {s.rect.roi1}  right {s.rect.roi2}",
    ]
    spread = s.rig_spread()
    if spread:
        rot_med, rot_sp = spread["rotation_median_deg"], spread["rotation_spread_deg"]
        t_med, t_sp = spread["translation_median_mm"], spread["translation_spread_mm"]
        lines += [
            "",
            "  rig consistency -- what each view alone says the geometry is,",
            "  from two independent pose solves and no stereo constraint:",
            f"    rotation    [{rot_med[0]:+.3f}, {rot_med[1]:+.3f}, {rot_med[2]:+.3f}] deg"
            f"  +- [{rot_sp[0]:.3f}, {rot_sp[1]:.3f}, {rot_sp[2]:.3f}]",
            f"    translation [{t_med[0]:+.2f}, {t_med[1]:+.2f}, {t_med[2]:+.2f}] mm"
            f"  +- [{t_sp[0]:.2f}, {t_sp[1]:.2f}, {t_sp[2]:.2f}]",
            f"    the rotation spread alone forces {spread['rotation_spread_px']:.2f} px "
            f"of reprojection error",
        ]
        if spread["rotation_spread_px"] > 0.5:
            lines.append(
                "    -> the views do not agree on the rig. No single R, T can fit them all,\n"
                "       so the stereo rms below is a floor set by the capture, not by the\n"
                "       solver. Suspect a mount that moves, or intrinsics that are only\n"
                "       weakly constrained by this set of poses.")

    dy = np.abs(np.concatenate([v.rect_dy for v in s.views.values()])) if s.views else np.zeros(1)
    verdict = ("good" if np.sqrt(np.mean(dy ** 2)) < 0.3 else
               "usable" if np.sqrt(np.mean(dy ** 2)) < 0.7 else "too soft for reliable matching")
    lines.append(f"  -> rectified rows agree to {np.sqrt(np.mean(dy ** 2)):.3f} px rms: {verdict}")
    return "\n".join(lines)


def summary(result) -> str:
    return stereo_summary(result) if isinstance(result, StereoCalibration) else mono_summary(result)


def suspects(result, sigma: float = 3.0) -> list[tuple[str, str]]:
    """Views a robust cut says are unlike the rest, with why."""
    out: dict[str, list[str]] = {}
    if isinstance(result, StereoCalibration):
        checks = [("reprojection", {k: v.rms for k, v in result.views.items()}),
                  ("epipolar", {k: v.epipolar_rms for k, v in result.views.items()}),
                  ("rectified dy", {k: v.rect_dy_rms for k, v in result.views.items()})]
    else:
        checks = [("reprojection", result.per_view_rms())]
    for name, per_view in checks:
        for key in outliers(per_view, sigma=sigma):
            out.setdefault(key, []).append(f"{name} {per_view[key]:.3f}")
    return sorted(out.items(), key=lambda kv: kv[0])


# -- figures ----------------------------------------------------------------


def _residual_field(ax, cal: MonoCalibration, bins: int = 14):
    pts, res = cal.all_observed(), cal.all_residuals()
    w, h = cal.image_size
    if not len(pts):
        return
    xi = np.clip((pts[:, 0] / w * bins).astype(int), 0, bins - 1)
    yi = np.clip((pts[:, 1] / h * bins).astype(int), 0, bins - 1)
    flat = yi * bins + xi
    count = np.bincount(flat, minlength=bins * bins)
    ux = np.bincount(flat, weights=res[:, 0], minlength=bins * bins)
    uy = np.bincount(flat, weights=res[:, 1], minlength=bins * bins)
    ok = count > 0
    gx, gy = np.meshgrid((np.arange(bins) + 0.5) * w / bins, (np.arange(bins) + 0.5) * h / bins)
    mx, my = ux[ok] / count[ok], uy[ok] / count[ok]
    mag = np.hypot(mx, my)
    q = ax.quiver(gx.ravel()[ok], gy.ravel()[ok], mx, my, mag,
                  cmap="viridis", angles="xy", scale_units="xy",
                  scale=max(mag.max(), 1e-6) / (0.45 * w / bins), width=0.004)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")
    ax.set_title(f"mean residual per cell ({cal.camera})\nmax {mag.max():.3f} px")
    return q




def _error_cmap():
    """Magma with the pale top trimmed off.

    Error magnitude is one-sided, so the ramp has to be sequential -- but on a
    white page a full sequential ramp puts the worst corners in near-white,
    which is exactly backwards: the points that matter most become the hardest
    to see. Stopping short of the light end keeps the high values saturated
    while the dark low end stays legible, so both extremes read.
    """
    import matplotlib
    base = matplotlib.colormaps["magma"]
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "magma_trimmed", base(np.linspace(0.08, 0.80, 256)))


_ERROR_CMAP = None


def _coverage_scatter(ax, fig, pts, magnitude, image_size, plt, label="reprojection error px"):
    """Where the corners landed, coloured by how badly each one fitted.

    A plain density plot answers "did the board go there"; this answers "did the
    model work there", which is the question the coverage was a proxy for all
    along. Error magnitude is one-sided, so the ramp is sequential -- a
    diverging map here would invent a midpoint that means nothing.

    The top of the scale is the 99th percentile rather than the maximum: a
    single bad corner otherwise flattens every other point to the same colour.
    """
    global _ERROR_CMAP
    if _ERROR_CMAP is None:
        _ERROR_CMAP = _error_cmap()
    w, h = image_size
    if len(pts):
        top = float(np.percentile(magnitude, 99)) if len(magnitude) else 1.0
        # Worst corners last, so they are drawn on top. Without this the point
        # you need to see is whichever happened to be plotted last, and a bad
        # corner can sit invisibly under a good one.
        order = np.argsort(magnitude)
        sc = ax.scatter(pts[order, 0], pts[order, 1], c=magnitude[order], s=5,
                        cmap=_ERROR_CMAP, vmin=0.0, vmax=max(top, 1e-6), lw=0)
        fig.colorbar(sc, ax=ax, label=label)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")


def _figure_mono(cal: MonoCalibration, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    w, h = cal.image_size
    # `mono_rms` is set only when this camera came out of a stereo solve, which
    # is also when everything on this figure changes meaning: the residuals are
    # then the stereo bundle's, not this camera's own fit, and they cover only
    # the corners both cameras saw. Saying so matters -- the two numbers differ
    # by an order of magnitude on a rig that does not agree with itself, and a
    # reader who assumes these are the mono residuals will blame the lens.
    stereo = cal.mono_rms is not None
    tag = "stereo-constrained" if stereo else "mono fit"

    ax = axes[0, 0]
    pts = cal.all_observed()
    mag = np.linalg.norm(cal.all_residuals(), axis=1) if len(pts) else np.zeros(0)
    _coverage_scatter(ax, fig, pts, mag, (w, h), plt)
    ax.set_title(f"reprojection error, by where the corner was seen\n"
                 f"({len(pts)} corners"
                 + (", seen by both cameras)" if stereo else ")"))

    _residual_field(axes[0, 1], cal)

    ax = axes[1, 0]
    res = cal.all_residuals()
    ax.scatter(res[:, 0], res[:, 1], s=2, alpha=0.25, edgecolors="none")
    lim = max(np.percentile(np.abs(res), 99.8), 1e-3) * 1.2
    for k in (1, 2, 3):
        ax.add_artist(plt.Circle((0, 0), k * cal.rms, fill=False, ls="--", lw=0.8, color="crimson"))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_title(f"residual scatter, {tag}, rings at 1/2/3 x rms ({cal.rms:.3f} px)")
    ax.set_xlabel("dx px"); ax.set_ylabel("dy px")

    ax = axes[1, 1]
    per = cal.per_view_rms()
    keys = sorted(per, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    vals = [per[k] for k in keys]
    bad = set(outliers(per))
    ax.bar(range(len(keys)), vals,
           color=["crimson" if k in bad else "steelblue" for k in keys], width=1.0)
    ax.axhline(cal.rms, color="k", ls="--", lw=1, label=f"overall {cal.rms:.3f}")
    ax.set_title(f"per-view rms, {tag} ({len(bad)} flagged)")
    ax.set_xlabel("view"); ax.set_ylabel("px"); ax.legend(fontsize=8)

    fig.suptitle(f"{cal.camera}: {cal.model.describe()}", y=0.985)
    if stereo:
        note = (f"Residuals on this page come from the STEREO solve: one board pose per "
                f"view, shared between the cameras through R and T, over the corners both "
                f"saw. This camera fitted on its own reached {cal.mono_rms:.4f} px -- the "
                f"gap to {cal.rms:.4f} px is what the rig constraint costs, not lens error.")
    else:
        note = "Residuals on this page come from this camera's own calibration."
    # Wrapped explicitly: this figure is 13 inches wide, and matplotlib's own
    # wrap=True measures against the figure rather than the text's anchor, so a
    # centred line of this length runs off both edges.
    fig.text(0.5, 0.962, "\n".join(textwrap.wrap(note, 118)), ha="center", va="top",
             fontsize=8.5, color="0.35", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return fig


def _figure_stereo(s: StereoCalibration, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    w, h = s.image_size
    dy = np.concatenate([v.rect_dy for v in s.views.values()])
    pl = np.concatenate([v.left.observed for v in s.views.values()])
    epi = np.concatenate([v.epipolar for v in s.views.values()])

    ax = axes[0, 0]
    sc = ax.scatter(pl[:, 0], pl[:, 1], c=dy, s=4, cmap="coolwarm",
                    vmin=-np.percentile(np.abs(dy), 99), vmax=np.percentile(np.abs(dy), 99))
    fig.colorbar(sc, ax=ax, label="dy px")
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")
    ax.set_title("rectified row error, by where the corner was seen")

    ax = axes[0, 1]
    ax.hist(dy, bins=80, color="steelblue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"rectified dy: rms {np.sqrt(np.mean(dy ** 2)):.3f} px, "
                 f"max |dy| {np.abs(dy).max():.2f} px")
    ax.set_xlabel("px")

    ax = axes[1, 0]
    keys = sorted(s.views, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    ax.plot([s.views[k].left.rms for k in keys], lw=0.9, label="reproj left")
    ax.plot([s.views[k].right.rms for k in keys], lw=0.9, label="reproj right")
    ax.plot([s.views[k].rect_dy_rms for k in keys], lw=0.9, label="rectified dy")
    ax.plot([s.views[k].epipolar_rms for k in keys], lw=0.9, label="epipolar")
    ax.set_title("per-view errors"); ax.set_xlabel("view"); ax.set_ylabel("px")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    d = np.array([s.views[k].left.distance for k in keys])
    t = np.array([s.views[k].left.tilt_deg for k in keys])
    e = np.array([s.views[k].rms for k in keys])
    sc = ax.scatter(d, t, c=e, s=18, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="view rms px")
    ax.set_xlabel("board distance (m)"); ax.set_ylabel("board tilt (deg)")
    ax.set_title("pose spread -- clusters here are gaps in the calibration")

    fig.suptitle(f"stereo: baseline {s.baseline * 1000:.1f} mm, "
                 f"rectified dy rms {np.sqrt(np.mean(dy ** 2)):.3f} px, "
                 f"epipolar rms {np.sqrt(np.mean(epi ** 2)):.3f} px")
    fig.tight_layout()
    return fig


def figures(result) -> dict:
    plt = _mpl()
    figs = {}
    if isinstance(result, StereoCalibration):
        figs["left"] = _figure_mono(result.left, plt)
        figs["right"] = _figure_mono(result.right, plt)
        figs["stereo"] = _figure_stereo(result, plt)
    else:
        figs[result.camera] = _figure_mono(result, plt)
    return figs


def save_figures(result, outdir, dpi: int = 110) -> list[pathlib.Path]:
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    plt = _mpl()
    for name, fig in figures(result).items():
        path = outdir / f"diagnostics_{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written


def rectified_preview(s: StereoCalibration, left_image, right_image,
                      rows: int = 24) -> np.ndarray:
    """The two eyes rectified and stacked side by side with horizontal rules.

    If the calibration is right, every feature crosses each rule at the same
    height in both halves. It is a crude check and it is the one that catches
    a sign error nothing else notices."""
    import cv2
    undistort = cv2.fisheye.initUndistortRectifyMap if s.model.fisheye \
        else cv2.initUndistortRectifyMap
    d1 = s.left.dist.reshape(4, 1) if s.model.fisheye else s.left.dist
    d2 = s.right.dist.reshape(4, 1) if s.model.fisheye else s.right.dist
    m1 = undistort(s.left.K, d1, s.rect.R1, s.rect.P1, s.image_size, cv2.CV_32FC1)
    m2 = undistort(s.right.K, d2, s.rect.R2, s.rect.P2, s.image_size, cv2.CV_32FC1)
    l = cv2.remap(left_image, m1[0], m1[1], cv2.INTER_LINEAR)
    r = cv2.remap(right_image, m2[0], m2[1], cv2.INTER_LINEAR)
    if l.ndim == 2:
        l, r = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([l, r])
    for i in range(1, rows):
        y = int(i * canvas.shape[0] / rows)
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 220, 0), 1)
    cv2.line(canvas, (l.shape[1], 0), (l.shape[1], canvas.shape[0]), (0, 0, 255), 2)
    return canvas
