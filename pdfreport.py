"""One PDF you can hand to someone else.

The PNGs `report.py` writes are for the person who just ran the solve and
already knows what they were looking for. This is for the person who did not:
it leads with a verdict, then shows the evidence for it, then the raw plots.

The order is the order you would work through it yourself:

  1  verdict        did this calibration come out usable, and if not, which of
                    the known failures is it?
  2  exposure       were the pixels good enough for the corners to mean anything?
  3  coverage       did the board go everywhere the model claims to describe?
  4  motion         was anything moving, or soft, in a way that biases corners?
  5  pose           does the residual track where the board was?
  6  undistortion   the frame before and after the model, big enough to judge.
  7  mono residuals stereo only: each camera alone, before the rig was imposed.
  8  rig            stereo only: is there one rig, or did it move?
  9  rectification  stereo only: what the alpha choice costs, at a glance.
 10  rectified pair stereo only: full size, with rules to sight along.
 11  per camera     the same cameras under the stereo constraint.

Every page states what a good answer looks like, because a number without its
threshold is not a diagnosis.
"""

from __future__ import annotations

import datetime
import pathlib
import textwrap

import numpy as np

from . import diagnostics as dg
from . import report as report_mod
from .calibrate import MonoCalibration, StereoCalibration
from .imagestats import BLACK_LEVEL, WHITE_LEVEL

PAGE = (11.69, 8.27)          # A4 landscape, inches

# What "good" means, in the units the reader sees.
DY_GOOD, DY_USABLE = 0.3, 0.7          # rectified row error, px
CLIP_FRAC = 0.001                      # fraction of the board clipped
EMPTY_CELL_FRAC = 0.15                 # share of the frame the board never visited
RIG_EXPLAINED = 0.5                    # per-view refit removing this much = moving rig


def _mpl():
    return report_mod._mpl()


def _page(plt, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(title, fontsize=15, x=0.045, ha="left", weight="bold")
    if subtitle:
        # Wrapped by hand: matplotlib's wrap=True measures against the whole
        # figure, so a left-anchored line still runs off the right edge.
        fig.text(0.045, 0.945, "\n".join(textwrap.wrap(subtitle, 132)),
                 fontsize=9.5, va="top", color="0.35", linespacing=1.4)
    return fig


def _note(fig, text: str, y: float = 0.018):
    fig.text(0.045, y, "\n".join(textwrap.wrap(text, 148)), fontsize=8.5,
             va="bottom", color="0.35", linespacing=1.45)


def _table(ax, rows, col_w=(0.62, 0.38), fontsize=9.5):
    """Left-aligned label/value rows. Cheaper to read than matplotlib's table."""
    ax.axis("off")
    for i, (label, value, *rest) in enumerate(rows):
        y = 1.0 - (i + 1) * (1.0 / (len(rows) + 1))
        colour = rest[0] if rest else "0.15"
        ax.text(0.0, y, str(label), fontsize=fontsize, va="center", color="0.35")
        text = str(value)
        # Shrink rather than overflow: these panels are narrow and a value that
        # runs past the edge is worse than a smaller one that fits.
        size = fontsize if len(text) <= 13 else fontsize * max(0.58, 13 / len(text))
        ax.text(col_w[0], y, text, fontsize=size, va="center",
                color=colour, family="monospace", weight="bold")


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# Page 1: the verdict


def _findings(result, detections) -> list[tuple[str, str, str]]:
    """(severity, headline, detail) for everything worth saying up front."""
    out = []
    stereo = isinstance(result, StereoCalibration)
    cams = ([("left", result.left), ("right", result.right)] if stereo
            else [(result.camera, result)])

    for name, cal in cams:
        alone = cal.mono_rms if cal.mono_rms is not None else cal.rms
        if alone is not None and np.isfinite(alone) and alone > 1.0:
            out.append(("bad", f"{name}: intrinsics do not fit",
                        f"reprojection {alone:.3f} px on its own images; expect < 0.5"))
        cv = dg.coverage(cal)
        if cv and cv["empty cells"] / cv["cells"] > EMPTY_CELL_FRAC:
            out.append(("warn", f"{name}: {cv['empty cells']} of {cv['cells']} "
                                f"frame cells never saw a corner",
                        "distortion is extrapolated there"))
        if detections is not None:
            st = [detections.stats_for(name, k) for k in cal.views]
            st = [s for s in st if s.ok]
            clipped = sum(1 for s in st if s.clipped)
            if clipped:
                out.append(("warn", f"{name}: {clipped} view(s) with the board clipped",
                            "clipped corners localise confidently and wrongly"))

    if stereo:
        dy = np.sqrt(np.mean(np.concatenate(
            [v.rect_dy for v in result.views.values()]) ** 2)) if result.views else np.nan
        if np.isfinite(dy):
            sev = "ok" if dy < DY_GOOD else "warn" if dy < DY_USABLE else "bad"
            out.append((sev, f"rectified rows agree to {dy:.3f} px",
                        f"< {DY_GOOD} good, < {DY_USABLE} usable for block matching"))
        rd = dg.rig_drift(result)
        if rd and rd.get("explained", 0) > RIG_EXPLAINED:
            mad = rd["rotation mad deg"]
            out.append(("bad", "the rig moved during the capture",
                        f"per-view refit drops epipolar {rd['global rms px']:.3f} -> "
                        f"{rd['per view rms px']:.3f} px ({100 * rd['explained']:.0f}%); "
                        f"relative rotation wanders {mad.max():.3f} deg"))
        spread = result.rig_spread()
        if spread and spread.get("rotation_spread_px", 0) > 0.5:
            out.append(("bad", "the views disagree about the rig",
                        f"rotation spread alone forces "
                        f"{spread['rotation_spread_px']:.2f} px of error"))
    if not out:
        out.append(("ok", "nothing stands out", "every check below is within tolerance"))
    return out


def _page_verdict(plt, result, detections, dataset):
    stereo = isinstance(result, StereoCalibration)
    kind = "stereo" if stereo else "mono"
    fig = _page(plt, f"{kind} calibration report",
                f"{getattr(dataset, 'root', '')}  ·  "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    findings = _findings(result, detections)
    colour = {"ok": "#1b6e3c", "warn": "#9a6100", "bad": "#a11d1d"}
    ax = fig.add_axes([0.045, 0.44, 0.60, 0.44]); ax.axis("off")
    ax.text(0, 1.0, "findings", fontsize=10, color="0.35", va="top")
    for i, (sev, head, detail) in enumerate(findings[:8]):
        y = 0.90 - i * 0.115
        ax.text(0.0, y, "●", fontsize=11, color=colour[sev], va="top")
        ax.text(0.035, y, head, fontsize=10.5, va="top", weight="bold",
                color=colour[sev])
        ax.text(0.035, y - 0.045, detail, fontsize=9, va="top", color="0.35")

    rows = [("views used", len(result.views)),
            ("model", result.model.describe())]
    if stereo:
        rows += [("baseline", f"{result.baseline * 1000:.2f} mm"),
                 ("stereo rms", f"{result.rms:.4f} px"),
                 ("left alone", f"{result.left.mono_rms:.4f} px"
                  if result.left.mono_rms else "n/a"),
                 ("right alone", f"{result.right.mono_rms:.4f} px"
                  if result.right.mono_rms else "n/a"),
                 ("rectify alpha", f"{result.rect.alpha:g}")]
    else:
        rows += [("reprojection rms", f"{result.rms:.4f} px"),
                 ("fx, fy", f"{result.K[0, 0]:.2f}, {result.K[1, 1]:.2f}"),
                 ("cx, cy", f"{result.K[0, 2]:.2f}, {result.K[1, 2]:.2f}")]
    ax2 = fig.add_axes([0.68, 0.44, 0.28, 0.44])
    ax2.text(0, 1.0, "numbers", fontsize=10, color="0.35", va="top",
             transform=ax2.transAxes)
    _table(ax2, rows)

    ax3 = fig.add_axes([0.045, 0.05, 0.915, 0.33]); ax3.axis("off")
    ax3.text(0, 1.0, "text summary", fontsize=10, color="0.35", va="top")
    text = report_mod.summary(result).strip("\n").split("\n")
    # Two columns, split on the blank line between blocks nearest the middle, so
    # a long summary stays on the page instead of running off the bottom.
    breaks = [i for i, ln in enumerate(text) if not ln.strip()] or [len(text) // 2]
    cut = min(breaks, key=lambda i: abs(i - len(text) / 2))
    for col, chunk in enumerate((text[:cut], text[cut + 1:])):
        ax3.text(col * 0.5, 0.90, "\n".join(chunk), fontsize=6.2, va="top",
                 family="monospace", color="0.2", linespacing=1.35)
    return fig


# ---------------------------------------------------------------------------


def _page_exposure(plt, result, detections, cams):
    fig = _page(plt, "exposure and clipping",
                "a corner sitting in a clipped highlight is still found, and still "
                "localised -- to the wrong place. Clipping biases corners without "
                "raising the RMS.")
    n = len(cams)
    for i, (name, cal) in enumerate(cams):
        keys = list(cal.views)
        st = [detections.stats_for(name, k) for k in keys]
        good = [s for s in st if s.ok]

        ax = fig.add_subplot(n, 3, i * 3 + 1)
        if good:
            h = np.sum([s.board_hist for s in good], axis=0).astype(float)
            centres = np.linspace(0, 255, len(h))
            ax.fill_between(centres, h, color="#3a6ea5", alpha=0.85, lw=0)
            ax.axvspan(0, 3, color="#a11d1d", alpha=0.35, zorder=3)
            ax.axvspan(252, 255, color="#a11d1d", alpha=0.35, zorder=3)
            ax.set_yscale("log")
        ax.set_title(f"{name}: intensity, BOARD REGION, all views", fontsize=10)
        ax.set_xlabel(f"pixel value (red: <= {BLACK_LEVEL} or >= {WHITE_LEVEL})")
        ax.set_xlim(0, 255)

        ax = fig.add_subplot(n, 3, i * 3 + 2)
        if good:
            w = np.array([s.board_white_frac for s in good]) * 100
            b = np.array([s.board_black_frac for s in good]) * 100
            x = np.arange(len(good))
            ax.plot(x, w, lw=1.2, color="#9a6100", label=f">= {WHITE_LEVEL}")
            ax.plot(x, b, lw=1.2, color="#3a6ea5", label=f"<= {BLACK_LEVEL}")
            ax.axhline(CLIP_FRAC * 100, ls="--", lw=0.8, color="0.5")
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{name}: BOARD REGION clipped, per view", fontsize=10)
        ax.set_ylabel("% of board"); ax.set_xlabel("view")

        ax = fig.add_subplot(n, 3, i * 3 + 3)
        rows = [("views measured", len(good))]
        if good:
            worst_w = max(s.board_white_frac for s in good) * 100
            worst_b = max(s.board_black_frac for s in good) * 100
            lo_c = min(s.board_contrast for s in good)
            rows += [
                ("clipped views", sum(1 for s in good if s.clipped),
                 "#a11d1d" if any(s.clipped for s in good) else "#1b6e3c"),
                (f"worst >= {WHITE_LEVEL}", f"{worst_w:.2f}% of board"),
                (f"worst <= {BLACK_LEVEL}", f"{worst_b:.2f}% of board"),
                ("lowest board contrast", f"{lo_c:.0f} counts",
                 "#9a6100" if lo_c < 40 else "#1b6e3c"),
                ("median board level", f"{np.median([s.board_mean for s in good]):.0f}"),
            ]
        else:
            rows += [("", "no pixel statistics in the cache"),
                     ("", "re-run detection to measure")]
        _table(ax, rows, fontsize=9)
        ax.set_title(f"{name}: verdict", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    _note(fig, f"Clipped means a pixel stuck at the end of the sensor's range: <= "
               f"{BLACK_LEVEL} (shadow, no detail left) or >= {WHITE_LEVEL} (highlight, "
               f"same). A clipped pixel has no gradient, and corner refinement fits a "
               f"saddle to the gradient -- so the corner is still found, and found in the "
               f"wrong place, without raising the RMS. Every panel here is the BOARD "
               f"REGION -- the corner bounding box, padded -- because that is the part "
               f"that bears on corner accuracy; background exposure does not. More than "
               f"{CLIP_FRAC * 100:g}% of the board "
               f"clipped is worth re-shooting at a shorter exposure; board contrast under "
               f"~40 counts makes refinement noisy whatever the exposure.")
    return fig


def _page_coverage(plt, result, cams, board):
    fig = _page(plt, "coverage",
                "distortion is only constrained where corners went. Outside that region "
                "the model is extrapolating, however small the RMS.")
    n = len(cams)
    for i, (name, cal) in enumerate(cams):
        cv = dg.coverage(cal)
        w, h = cal.image_size

        ax = fig.add_subplot(n, 3, i * 3 + 1)
        pts = cal.all_observed()
        mag = np.linalg.norm(cal.all_residuals(), axis=1) if len(pts) else np.zeros(0)
        report_mod._coverage_scatter(ax, fig, pts, mag, (w, h), plt)
        ax.set_title(f"{name}: reprojection error,\nby where the corner was seen",
                     fontsize=9.5)

        ax = fig.add_subplot(n, 3, i * 3 + 2)
        bc = dg.board_coverage(cal, board)
        if bc is not None:
            im = ax.imshow(bc, cmap="viridis", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{name}: detections per board corner", fontsize=10)
        else:
            ax.axis("off")
            ax.set_title(f"{name}: board layout not griddable", fontsize=10)

        ax = fig.add_subplot(n, 3, i * 3 + 3)
        rows = []
        if cv:
            empty = cv["empty cells"] / cv["cells"]
            rows = [("corners", cv["corners"]),
                    ("frame cells never visited", f"{cv['empty cells']}/{cv['cells']}",
                     "#a11d1d" if empty > EMPTY_CELL_FRAC else "#1b6e3c"),
                    ("unused margin, left", f"{cv['u margin left'] * 100:.1f}%"),
                    ("unused margin, right", f"{cv['u margin right'] * 100:.1f}%"),
                    ("unused margin, top", f"{cv['v margin top'] * 100:.1f}%"),
                    ("unused margin, bottom", f"{cv['v margin bottom'] * 100:.1f}%")]
            if bc is not None:
                border = np.ones(bc.shape, bool); border[1:-1, 1:-1] = False
                rows.append(("board edge vs middle",
                             f"{bc[border].mean() / max(bc[~border].mean(), 1e-9):.2f}x"))
        _table(ax, rows, fontsize=9)
        ax.set_title(f"{name}: verdict", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    _note(fig, "In a stereo run these plots show only the corners both cameras saw, so the "
               "non-overlapping margins read as uncovered. That band is stereo geometry, "
               "not a gap in the capture -- its width is roughly the disparity.")
    return fig


def _page_motion(plt, result, detections, cams, facts):
    fig = _page(plt, "sharpness and motion",
                "motion smears a corner along the direction of travel, and the smear "
                "biases it the same way in every corner of that view -- which reads as "
                "a pose error, not as noise.")
    n = len(cams)
    for i, (name, cal) in enumerate(cams):
        solved = facts[name]
        # Sharpness and board travel need no calibration, so every detected view
        # can contribute -- and must, because the solve keeps a thinned spread
        # with no adjacent frames left to difference.
        f = solved
        if detections is not None:
            try:
                dense = dg.dense_mono_facts(detections, name)
                if len(dense) > len(solved):
                    f = dense
            except Exception:
                pass
        scope = f"{len(f)} detected views" if f is not solved else f"{len(f)} solved views"
        have = np.isfinite(f.sharpness).any()
        enough_speed = int(np.isfinite(f.speed).sum()) >= dg.MIN_FOR_CORRELATION

        ax = fig.add_subplot(n, 3, i * 3 + 1)
        if have:
            ax.plot(f.index, f.sharpness, lw=1.0, color="#3a6ea5")
            med = np.nanmedian(f.sharpness)
            ax.axhline(med, ls="--", lw=0.8, color="0.5", label="median")
            ax.axhline(0.5 * med, ls=":", lw=0.9, color="#a11d1d", label="half median")
            ax.legend(fontsize=7.5, frameon=False)
            ax.set_ylabel("normalised Laplacian variance")
        else:
            _empty(ax, "no pixel statistics in the cache\nre-run detection to measure")
        ax.set_title(f"{name}: sharpness per view ({scope})", fontsize=9.5)
        ax.set_xlabel("view")

        ax = fig.add_subplot(n, 3, i * 3 + 2)
        if have and enough_speed:
            m = np.isfinite(f.speed) & np.isfinite(f.sharpness)
            sc = ax.scatter(f.speed[m], f.sharpness[m], s=13,
                            c=np.nan_to_num(f.anisotropy[m]), cmap="magma", lw=0,
                            vmin=0, vmax=1)
            fig.colorbar(sc, ax=ax, fraction=0.046, label="motion anisotropy")
            ax.set_xlabel("board speed px/frame"); ax.set_ylabel("sharpness")
        elif have:
            m = np.isfinite(f.anisotropy) & np.isfinite(f.sharpness)
            ax.scatter(f.anisotropy[m], f.sharpness[m], s=13, color="#3a6ea5", lw=0)
            ax.set_xlabel("motion anisotropy"); ax.set_ylabel("sharpness")
        else:
            _empty(ax, "no pixel statistics")
        ax.set_title(f"{name}: does speed cost sharpness?", fontsize=9.5)

        ax = fig.add_subplot(n, 3, i * 3 + 3)
        rows = []
        if have:
            sh = f.sharpness[np.isfinite(f.sharpness)]
            med = float(np.median(sh))
            soft = int((sh < 0.5 * med).sum())
            rows += [("median sharpness", f"{med:.2f}"),
                     ("softest view", f"{sh.min():.2f}"),
                     ("views under half median", soft,
                      "#9a6100" if soft else "#1b6e3c")]
            a = f.anisotropy[np.isfinite(f.anisotropy)]
            if len(a):
                rows.append(("max motion anisotropy", f"{a.max():.2f}",
                             "#9a6100" if a.max() > 0.5 else "#1b6e3c"))
        if enough_speed:
            rows += [("median board speed", f"{np.nanmedian(f.speed):.1f} px/frame"),
                     ("fastest view", f"{np.nanmax(f.speed):.0f} px/frame"),
                     ("corr(sharpness, speed)", _fmt(dg._corr(f.speed, f.sharpness)))]
            c = dg._corr(solved.speed, solved.rms)
            if np.isfinite(c):
                rows.append(("corr(view rms, speed)", _fmt(c)))
        else:
            rows.append(("board speed", "needs consecutive frames"))
        _table(ax, rows, fontsize=9)
        ax.set_title(f"{name}: verdict", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    _note(fig, "Defocus lowers sharpness and leaves anisotropy near zero; motion lowers "
               "sharpness and raises it. Anisotropy above ~0.5 on a board that should "
               "carry gradients in two directions means the frame was moving.")
    return fig


def _page_pose(plt, result, cams, facts):
    fig = _page(plt, "pose dependence",
                "if the residual tracks where the board was, the lens model is wrong or "
                "the poses never constrained it -- and dropping views will not help.")
    n = len(cams)
    for i, (name, cal) in enumerate(cams):
        f = facts[name]
        ax = fig.add_subplot(n, 3, i * 3 + 1)
        if len(f):
            sc = ax.scatter(f.distance, f.tilt, c=f.rms, s=16, cmap="magma", lw=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, label="view rms px")
        ax.set_xlabel("board distance (m)"); ax.set_ylabel("board tilt (deg)")
        ax.set_title(f"{name}: pose spread", fontsize=10)

        ax = fig.add_subplot(n, 3, i * 3 + 2)
        pd = dg.pose_dependence(f)
        if pd:
            labels = list(pd)[::-1]
            vals = [pd[k] for k in labels]
            colours = ["#a11d1d" if abs(v) > 0.4 else "#9a6100" if abs(v) > 0.25
                       else "#7c8b96" for v in vals]
            ax.barh(labels, vals, color=colours, height=0.6)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.set_xlim(-1, 1); ax.tick_params(labelsize=8)
        ax.set_title(f"{name}: corr(view rms, X)", fontsize=10)

        ax = fig.add_subplot(n, 3, i * 3 + 3)
        rows = [("tilt median", f"{np.median(f.tilt):.1f} deg" if len(f) else "n/a",
                 "#9a6100" if len(f) and np.median(f.tilt) < 15 else "#1b6e3c"),
                ("tilt p95", f"{np.percentile(f.tilt, 95):.1f} deg" if len(f) else "n/a"),
                ("distance range",
                 f"{f.distance.min():.2f} - {f.distance.max():.2f} m" if len(f) else "n/a"),
                ("strongest driver",
                 max(pd, key=lambda k: abs(pd[k])) if pd else "none")]
        ac = dg.temporal_structure(f)
        if ac:
            rows.append(("autocorr at lag 1", _fmt(ac.get(1)),
                         "#a11d1d" if ac.get(1, 0) > 0.5 else "#1b6e3c"))
        _table(ax, rows, fontsize=9)
        ax.set_title(f"{name}: verdict", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    _note(fig, "Tilt under about 15 deg median with a narrow lens leaves focal length "
               "trading against board distance: the fit stays tight and the scale is "
               "free. Aim for views at 30-45 deg. A lag-1 autocorrelation above ~0.5 "
               "means the residual is a drifting process, not per-view noise.")
    return fig


def _empty(ax, why: str):
    """An axis with nothing in it should say why, not look broken."""
    ax.text(0.5, 0.5, why, ha="center", va="center", fontsize=9, color="0.5",
            wrap=True, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def _page_rig(plt, s, facts, detections=None):
    fig = _page(plt, "is there one rig?",
                "a stereo solve assumes one rigid transform for the whole capture. "
                "This page tests that assumption directly.")
    keys, offsets, split = dg.dy_offsets(s)
    f = facts["left"]
    rd = dg.rig_drift(s)

    # Fit on the solved subset, evaluate on every detected view. The solve keeps
    # a spread of views chosen for coverage, which usually leaves no adjacent
    # frames -- and every test below needs a time axis.
    dense = None
    if detections is not None:
        try:
            df, doff = dg.dense_stereo_series(s, detections)
            if len(df) > len(f):
                dense = (df, doff)
        except Exception:
            dense = None
    tf, toff = dense if dense else (f, offsets if len(offsets) == len(f) else None)
    scope = (f"all {len(tf)} detected views" if dense
             else f"the {len(tf)} solved views")

    ax = fig.add_subplot(2, 3, 1)
    if toff is not None and len(toff):
        ax.axhspan(-DY_USABLE, DY_USABLE, color="#1b6e3c", alpha=0.25, zorder=0,
                   label=f"+-{DY_USABLE} px usable")
        ax.plot(tf.index, toff, lw=1.0, color="#a11d1d", zorder=2)
        ax.axhline(0, color="0.4", lw=0.8, zorder=1)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    else:
        _empty(ax, "no rectified pairs")
    ax.set_title(f"bulk row shift per view\n({scope})", fontsize=9.5)
    ax.set_xlabel("view"); ax.set_ylabel("mean dy (px)")

    ax = fig.add_subplot(2, 3, 2)
    ac = dg.temporal_structure(tf, toff)
    if ac:
        ax.bar([str(k) for k in ac], list(ac.values()), color="#3a6ea5", width=0.6)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axhline(0.5, ls="--", lw=0.8, color="#a11d1d")
        ax.set_ylim(min(-0.4, min(ac.values()) - 0.1), 1)
    else:
        _empty(ax, "needs consecutive frame numbers\n(solve with --views all, or "
                   "name images by frame)")
    ax.set_title("autocorrelation of that shift", fontsize=9.5)
    ax.set_xlabel("lag (frames)")

    ax = fig.add_subplot(2, 3, 3)
    if rd and len(rd.get("rotation", [])):
        rot = rd["rotation"]
        x = dg._numeric(rd["keys"])
        x = x if x is not None else np.arange(len(rot))
        for j, (lab, col) in enumerate((("pitch", "#a11d1d"), ("yaw", "#1b6e3c"),
                                        ("roll", "#3a6ea5"))):
            ax.plot(x, rot[:, j] - np.median(rot[:, j]), lw=1.0, color=col, label=lab)
        ax.legend(fontsize=8, frameon=False); ax.axhline(0, color="0.4", lw=0.8)
    else:
        _empty(ax, "no per-view rig estimate")
    ax.set_title("per-view rig rotation, centred", fontsize=9.5)
    ax.set_ylabel("deg"); ax.set_xlabel("view")

    ax = fig.add_subplot(2, 3, 4)
    rows = []
    if rd:
        rows = [("epipolar, one global R,T", f"{rd['global rms px']:.3f} px"),
                ("epipolar, R,T per view", f"{rd['per view rms px']:.3f} px"),
                ("explained by refitting", f"{100 * rd['explained']:.0f}%",
                 "#a11d1d" if rd["explained"] > RIG_EXPLAINED else "#1b6e3c"),
                ("rotation wander (MAD)",
                 " / ".join(f"{v:.3f}" for v in rd["rotation mad deg"]) + " deg"),
                ("rotation range",
                 " / ".join(f"{v:.3f}" for v in rd["rotation ptp deg"]) + " deg")]
    _table(ax, rows, fontsize=9)
    ax.set_title("does one rigid transform fit?", fontsize=10)

    ax = fig.add_subplot(2, 3, 5)
    rows = [(k, _fmt(v)) for k, v in split.items()]
    _table(ax, rows, fontsize=9)
    ax.set_title("what the row error is made of", fontsize=10)

    ax = fig.add_subplot(2, 3, 6)
    tvp = dg.time_vs_pose(tf, toff)
    md = dg.motion_dependence(tf, toff)
    rows = [(k, _fmt(v),
             "#a11d1d" if k == "verdict" and str(v).startswith("time") else "0.15")
            for k, v in tvp.items()]
    if md:
        rows += [(k, _fmt(v)) for k, v in list(md.items())[:5]]
    if not rows:
        rows = [("", f"needs consecutive frames over"), ("", "at least 48 views")]
    _table(ax, rows, fontsize=8)
    ax.set_title("driven by time, or by pose?", fontsize=9.5)

    fig.tight_layout(rect=(0, 0.11, 1, 0.90))
    _note(fig, "If refitting per view collapses the epipolar error, the correspondences "
               "and intrinsics are fine and the cameras moved relative to each other: "
               "there is no single R, T to find, and the capture has to be redone on a "
               "stiffer mount. If the shift instead tracks board speed, the two cameras "
               "are exposing at different instants -- a timing bug, not a rig one.")
    return fig


# ---------------------------------------------------------------------------
# Rectification and the alpha sweep


def alpha_sweep(s: StereoCalibration, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """What each alpha would cost, without re-solving anything.

    stereoRectify's alpha trades field of view against black border: 0 crops to
    pixels that are valid in both views, 1 keeps every source pixel and pads the
    rest. P1, P2, Q and the valid ROI are all functions of it, which is why this
    is computed here from the calibration rather than applied afterwards to an
    already-rectified image.
    """
    import cv2
    if s.model.fisheye:
        return []
    w, h = s.image_size
    out = []
    for a in alphas:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            s.left.K, s.left.dist, s.right.K, s.right.dist, s.image_size,
            s.R, s.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(a))
        r1, r2 = tuple(int(v) for v in roi1), tuple(int(v) for v in roi2)
        # The usable rectangle is the intersection of the two ROIs: a row is
        # only matchable where both eyes have real pixels.
        x0 = max(r1[0], r2[0]); y0 = max(r1[1], r2[1])
        x1 = min(r1[0] + r1[2], r2[0] + r2[2]); y1 = min(r1[1] + r1[3], r2[1] + r2[3])
        common = (max(0, x1 - x0), max(0, y1 - y0))
        out.append({
            "alpha": float(a),
            "fx": float(P1[0, 0]),
            "roi_left": r1, "roi_right": r2,
            "common": common,
            "common_frac": common[0] * common[1] / float(w * h),
            "fov_scale": float(P1[0, 0]) and float(s.left.K[0, 0] / P1[0, 0]),
        })
    return out


def _page_rectification(plt, s, dataset):
    import cv2
    fig = _page(plt, "rectification and the alpha choice",
                "alpha trades field of view against black border. P1, P2, Q and the "
                "valid ROI are all functions of it, so it is a calibration input, not "
                "something to apply to a rectified image afterwards.")
    sweep = alpha_sweep(s)

    ax = fig.add_subplot(2, 3, 1)
    if sweep:
        a = [d["alpha"] for d in sweep]
        ax.plot(a, [d["common_frac"] * 100 for d in sweep], "o-", color="#3a6ea5")
        ax.axvline(s.rect.alpha, ls="--", color="#a11d1d", lw=1)
        ax.set_xlabel("alpha"); ax.set_ylabel("% of frame usable in both eyes")
    ax.set_title("what survives rectification", fontsize=10)

    ax = fig.add_subplot(2, 3, 2)
    if sweep:
        ax.plot([d["alpha"] for d in sweep], [d["fx"] for d in sweep], "o-",
                color="#1b6e3c")
        ax.axvline(s.rect.alpha, ls="--", color="#a11d1d", lw=1)
        ax.set_xlabel("alpha"); ax.set_ylabel("rectified fx (px)")
    ax.set_title("focal length after rectification", fontsize=10)

    ax = fig.add_subplot(2, 3, 3)
    rows = [("alpha used", f"{s.rect.alpha:g}"),
            ("rectified size", "x".join(str(v) for v in
                                        (s.rect.rectified_size or s.image_size))),
            ("rectified fx", f"{s.rect.P1[0, 0]:.2f}"),
            ("baseline", f"{s.rect.baseline * 1000:.2f} mm"),
            ("valid roi, left", str(s.rect.roi1)),
            ("valid roi, right", str(s.rect.roi2))]
    _table(ax, rows, fontsize=8.5)
    ax.set_title("what was written", fontsize=10)

    # A real pair through each alpha, so the trade is visible rather than tabular.
    view = None
    if dataset is not None and s.views:
        for key in s.views:
            view = next((v for v in dataset.views if v.key == key
                         and "left" in v.paths and "right" in v.paths), None)
            if view:
                break
    shown = [d for d in sweep if d["alpha"] in (0.0, 0.5, 1.0)][:3]
    for i, d in enumerate(shown):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.axis("off")
        ax.set_title(f"alpha {d['alpha']:g}  ({d['common_frac'] * 100:.0f}% usable)",
                     fontsize=10)
        if view is None:
            ax.text(0.5, 0.5, "no image pair available", ha="center", va="center",
                    fontsize=9, color="0.5")
            continue
        li = cv2.imread(str(view.paths["left"]), cv2.IMREAD_GRAYSCALE)
        ri = cv2.imread(str(view.paths["right"]), cv2.IMREAD_GRAYSCALE)
        if li is None or ri is None:
            continue
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            s.left.K, s.left.dist, s.right.K, s.right.dist, s.image_size,
            s.R, s.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=d["alpha"])
        m1 = cv2.initUndistortRectifyMap(s.left.K, s.left.dist, R1, P1,
                                         s.image_size, cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(s.right.K, s.right.dist, R2, P2,
                                         s.image_size, cv2.CV_32FC1)
        lr = cv2.remap(li, m1[0], m1[1], cv2.INTER_LINEAR)
        rr = cv2.remap(ri, m2[0], m2[1], cv2.INTER_LINEAR)
        canvas = np.hstack([lr, rr])
        ax.imshow(canvas, cmap="gray", aspect="equal")
        for k in range(1, 8):
            ax.axhline(k * canvas.shape[0] / 8, color="#22c55e", lw=0.35, alpha=0.65)
        ax.axvline(lr.shape[1], color="#a11d1d", lw=0.7)
        x, y, ww, hh = (int(v) for v in roi1)
        ax.add_patch(plt.Rectangle((x, y), ww, hh, fill=False, ec="#facc15", lw=0.9))
    fig.tight_layout(rect=(0, 0.11, 1, 0.90))
    _note(fig, "Yellow is the left valid ROI; green rules should cross features at the "
               "same height in both halves. Pick the largest alpha whose border you are "
               "willing to carry, then re-run the solve with it -- the crop belongs in "
               "the calibration, so that P, Q and the ROI stay one self-consistent set.")
    return fig


# ---------------------------------------------------------------------------


def build(result, dataset=None, detections=None, path="calibration_report.pdf"):
    """Write the whole report. Returns the path."""
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    stereo = isinstance(result, StereoCalibration)
    cams = ([("left", result.left), ("right", result.right)] if stereo
            else [(result.camera, result)])
    facts = {name: dg.view_facts(cal, detections, name) for name, cal in cams}

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = []
    try:
        pages.append(_page_verdict(plt, result, detections, dataset))
        if detections is not None:
            pages.append(_page_exposure(plt, result, detections, cams))
        pages.append(_page_coverage(plt, result, cams, result.board))
        pages.append(_page_motion(plt, result, detections, cams, facts))
        pages.append(_page_pose(plt, result, cams, facts))
        pages.append(_page_undistort(plt, result, dataset, cams))
        if stereo:
            pages.append(_page_mono_residuals(plt, result))
            pages.append(_page_rig(plt, result, facts, detections))
            pages.append(_page_rectification(plt, result, dataset))
            pages.append(_page_rectified_big(plt, result, dataset))
        for name, fig in report_mod.figures(result).items():
            pages.append(fig)
        with PdfPages(path) as pdf:
            for fig in pages:
                pdf.savefig(fig)
            info = pdf.infodict()
            info["Title"] = f"{'stereo' if stereo else 'mono'} calibration report"
            info["Subject"] = str(getattr(dataset, "root", ""))
            info["Creator"] = "calib"
    finally:
        for fig in pages:
            plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Big pictures: what the lens model actually does to an image.


def _warp_magnitude(K, dist, size, fisheye: bool) -> tuple[float, float]:
    """How far pixels move when the lens model is removed, in pixels.

    A single number for "how much does this lens warp": sample a grid over the
    frame, undistort it, and measure the displacement. The maximum lands at the
    frame corners, which is also where the model is least constrained -- so a
    large number here and thin corner coverage is the combination to worry
    about.
    """
    w, h = size
    xs = np.linspace(0, w - 1, 33)
    ys = np.linspace(0, h - 1, 33)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float64)
    import cv2
    if fisheye:
        und = cv2.fisheye.undistortPoints(grid, K, np.asarray(dist, float).reshape(4, 1), P=K)
    else:
        und = cv2.undistortPoints(grid, K, dist, P=K)
    d = np.linalg.norm(und.reshape(-1, 2) - grid.reshape(-1, 2), axis=1)
    return float(d.max()), float(d.mean())


def _new_camera(K, dist, size, alpha: float, fisheye: bool):
    """The undistorted camera matrix at this alpha, and the rectangle worth keeping."""
    import cv2
    if fisheye:
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, np.asarray(dist, float).reshape(4, 1), size, np.eye(3), balance=alpha)
        return newK, None
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return newK, tuple(int(v) for v in roi)


def _undistorted(img, K, dist, size, alpha: float, fisheye: bool):
    import cv2
    newK, roi = _new_camera(K, dist, size, alpha, fisheye)
    if fisheye:
        m = cv2.fisheye.initUndistortRectifyMap(
            K, np.asarray(dist, float).reshape(4, 1), np.eye(3), newK, size, cv2.CV_32FC1)
    else:
        m = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_32FC1)
    return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR), roi


def _shrink(img, max_w: int = 900):
    """Downsample for the page. A 1440 px frame printed 3.5 inches wide gains
    nothing from full resolution and costs a megabyte a picture."""
    import cv2
    if img is None or img.shape[1] <= max_w:
        return img
    scale = max_w / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _pick_view(result, dataset, cams):
    """The view that shows the warp best: the one reaching furthest into the corners.

    Distortion is smallest at the centre and largest at the edge, so a board
    sitting in the middle of the frame demonstrates nothing. This picks the view
    whose corners span the most of the frame.
    """
    if dataset is None:
        return None
    name, cal = cams[0]
    best, best_span = None, -1.0
    for key, v in cal.views.items():
        view = next((vw for vw in dataset.views
                     if vw.key == key and all(c in vw.paths for c, _ in cams)), None)
        if view is None:
            continue
        span = float(np.ptp(v.observed, axis=0).prod())
        if span > best_span:
            best, best_span = view, span
    return best


def _page_undistort(plt, result, dataset, cams):
    """Original against undistorted, at both ends of the alpha range."""
    import cv2
    stereo = isinstance(result, StereoCalibration)
    fig = _page(plt, "undistortion",
                "the same frame before and after the lens model is removed. alpha 0 keeps "
                "only pixels that stay valid, so the frame is cropped and scaled; alpha 1 "
                "keeps every source pixel, so the corners pull in and leave black.")
    view = _pick_view(result, dataset, cams)
    if view is None:
        ax = fig.add_axes([0.045, 0.2, 0.9, 0.6]); ax.axis("off")
        _empty(ax, "no source images available\n(the report needs the dataset to draw these)")
        return fig

    n = len(cams)
    # Three panels across, as wide as the page allows: the whole point of this
    # page is that the pictures are big enough to judge by eye. The numbers go
    # in a caption under each row rather than a fourth column.
    #
    # The row height is derived from the image aspect rather than fixed. An axes
    # taller than its picture letterboxes it, which on a one-camera report left
    # the single row stranded in the middle of the page.
    probe = cv2.imread(str(view.paths[cams[0][0]]), cv2.IMREAD_GRAYSCALE)
    aspect = (probe.shape[1] / probe.shape[0]) if probe is not None else 4 / 3
    top, floor, gap = (0.86 if n > 1 else 0.80), 0.125, 0.072
    # Start from the widest panel that fits three across, then clamp it to the
    # vertical budget so two rows plus their captions still clear the footnote.
    panel_w = 0.295
    panel_h = panel_w * PAGE[0] / (aspect * PAGE[1])
    budget = (top - floor) / n - gap
    if panel_h > budget:
        panel_h = budget
        panel_w = panel_h * aspect * PAGE[1] / PAGE[0]
    step = panel_w + 0.02
    left = (1.0 - (3 * panel_w + 2 * 0.02)) / 2
    row = panel_h + gap
    for i, (name, cal) in enumerate(cams):
        img = cv2.imread(str(view.paths[name]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = tuple(cal.image_size)
        mx, mean = _warp_magnitude(cal.K, cal.dist, size, cal.model.fisheye)
        scale = 900 / img.shape[1] if img.shape[1] > 900 else 1.0
        panels = [("original", _shrink(img), None)]
        for a in (0.0, 1.0):
            und, roi = _undistorted(img, cal.K, cal.dist, size, a, cal.model.fisheye)
            # Only worth drawing at alpha 1, where it marks the largest rectangle
            # with no black in it. At alpha 0 it is the frame border already.
            keep = None if (roi is None or a == 0.0) else tuple(v * scale for v in roi)
            panels.append((f"undistorted, alpha {a:g}", _shrink(und), keep))
        for j, (label, im, roi) in enumerate(panels):
            ax = fig.add_axes([left + j * step, top - (i + 1) * row, panel_w, panel_h])
            ax.imshow(im, cmap="gray", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}: {label}", fontsize=9.5)
            if roi:
                x, y, w, h = roi
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#facc15", lw=1.3))
        keep0 = _new_camera(cal.K, cal.dist, size, 0.0, cal.model.fisheye)[1]
        caption = (f"{name}:  pixels move up to {mx:.1f} px (mean {mean:.1f} px) when the "
                   f"model is removed   ·   k1 {cal.dist[0]:+.4f}")
        if keep0:
            caption += (f"   ·   alpha 0 keeps "
                        f"{100 * keep0[2] * keep0[3] / (size[0] * size[1]):.0f}% of the frame")
        fig.text(left, top - (i + 1) * row - 0.030, caption, fontsize=8.5,
                 color="0.35", va="bottom")
    _note(fig, "Yellow marks the largest rectangle with no black in it -- what you would "
               "crop to at alpha 1. Max pixel shift peaks at the frame corners, which is "
               "where corner coverage is usually thinnest: read it against that page.")
    return fig


def _page_rectified_big(plt, s, dataset):
    """The rectified pair, large enough to sight along the rules."""
    import cv2
    fig = _page(plt, "rectified pair",
                "if the calibration is right, every feature crosses each rule at the same "
                "height in both halves. The top row is the alpha this calibration was "
                "solved with; the bottom keeps every source pixel, so the black is what "
                "fitting everything would cost.")
    view = _pick_view(s, dataset, [("left", s.left), ("right", s.right)])
    if view is None:
        ax = fig.add_axes([0.045, 0.2, 0.9, 0.6]); ax.axis("off")
        _empty(ax, "no source image pair available")
        return fig
    li = cv2.imread(str(view.paths["left"]), cv2.IMREAD_GRAYSCALE)
    ri = cv2.imread(str(view.paths["right"]), cv2.IMREAD_GRAYSCALE)
    if li is None or ri is None:
        return fig

    for i, alpha in enumerate((s.rect.alpha, 1.0)):
        if s.model.fisheye:
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                s.left.K, s.left.dist.reshape(4, 1), s.right.K, s.right.dist.reshape(4, 1),
                s.image_size, s.R, s.T, cv2.CALIB_ZERO_DISPARITY,
                newImageSize=s.image_size, balance=alpha, fov_scale=1.0)
            roi1 = None
            m1 = cv2.fisheye.initUndistortRectifyMap(
                s.left.K, s.left.dist.reshape(4, 1), R1, P1, s.image_size, cv2.CV_32FC1)
            m2 = cv2.fisheye.initUndistortRectifyMap(
                s.right.K, s.right.dist.reshape(4, 1), R2, P2, s.image_size, cv2.CV_32FC1)
        else:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                s.left.K, s.left.dist, s.right.K, s.right.dist, s.image_size,
                s.R, s.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(alpha))
            m1 = cv2.initUndistortRectifyMap(s.left.K, s.left.dist, R1, P1,
                                             s.image_size, cv2.CV_32FC1)
            m2 = cv2.initUndistortRectifyMap(s.right.K, s.right.dist, R2, P2,
                                             s.image_size, cv2.CV_32FC1)
        lr = cv2.remap(li, m1[0], m1[1], cv2.INTER_LINEAR)
        rr = cv2.remap(ri, m2[0], m2[1], cv2.INTER_LINEAR)
        canvas = _shrink(np.hstack([lr, rr]), max_w=1800)
        scale = canvas.shape[1] / (lr.shape[1] * 2)

        # Height-limited, not width-limited: the pair is 2.7:1, so giving the
        # axes more height is the only thing that makes the picture bigger.
        ax = fig.add_axes([0.045, 0.475 - i * 0.395, 0.91, 0.37])
        ax.imshow(canvas, cmap="gray", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for k in range(1, 16):
            ax.axhline(k * canvas.shape[0] / 16, color="#22c55e", lw=0.5, alpha=0.7)
        ax.axvline(canvas.shape[1] / 2, color="#a11d1d", lw=1.0)
        if roi1:
            x, y, w, h = (v * scale for v in roi1)
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#facc15", lw=1.2))
        tag = ("as solved" if i == 0 else "everything kept")
        black = float(np.count_nonzero(canvas == 0)) / canvas.size * 100
        ax.set_title(f"alpha {alpha:g}  ({tag}) -- {black:.0f}% of the frame is black",
                     fontsize=10)
    _note(fig, "Yellow is the left valid ROI. Black is where no source pixel maps; a block "
               "matcher will find nothing there, so alpha 1 buys field of view you cannot "
               "actually match in. Pick the alpha you want and re-run the solve with it.")
    return fig


def _page_mono_residuals(plt, s):
    """Each camera's own fit, before the rig was imposed on it.

    The per-camera pages elsewhere in this report show stereo-constrained
    residuals: one board pose per view shared through R and T, over the corners
    both cameras saw. That is the right thing to look at when judging the pair,
    and the wrong thing when judging a lens. This page is the other half -- each
    camera solved alone, on every corner it saw -- and the gap between the two
    is the cost of forcing one rig onto the views.
    """
    monos = [(n, m) for n, m in (("left", s.mono_left), ("right", s.mono_right))
             if m is not None and m.views]
    fig = _page(plt, "mono residuals",
                "each camera fitted on its own, over every corner it saw. Compare with "
                "the per-camera pages, which show the same cameras under the stereo "
                "constraint: the difference is what the rig costs, not what the lens does.")
    if not monos:
        ax = fig.add_axes([0.045, 0.2, 0.9, 0.6]); ax.axis("off")
        _empty(ax, "the mono solves were not retained\n(a calibration loaded from YAML "
                   "carries no per-view residuals)")
        return fig

    n = len(monos)
    for i, (name, m) in enumerate(monos):
        w, h = m.image_size
        pts = m.all_observed()
        res = m.all_residuals()

        ax = fig.add_subplot(n, 4, i * 4 + 1)
        report_mod._coverage_scatter(ax, fig, pts, np.linalg.norm(res, axis=1)
                                     if len(res) else np.zeros(0), (w, h), plt,
                                     label="mono reprojection error px")
        ax.set_title(f"{name}: mono reprojection error,\nby where the corner was seen "
                     f"({len(pts)} corners)", fontsize=9.5)

        ax = fig.add_subplot(n, 4, i * 4 + 2)
        if len(res):
            ax.scatter(res[:, 0], res[:, 1], s=2, alpha=0.25, edgecolors="none",
                       color="#3a6ea5")
            lim = max(np.percentile(np.abs(res), 99.8), 1e-3) * 1.2
            for k in (1, 2, 3):
                ax.add_artist(plt.Circle((0, 0), k * m.rms, fill=False, ls="--",
                                         lw=0.8, color="#a11d1d"))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
        ax.set_title(f"{name}: mono residual scatter\nrings at 1/2/3 x rms "
                     f"({m.rms:.4f} px)", fontsize=9.5)
        ax.set_xlabel("dx px"); ax.set_ylabel("dy px")

        ax = fig.add_subplot(n, 4, i * 4 + 3)
        per = m.per_view_rms()
        keys = sorted(per, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))
        ax.bar(range(len(keys)), [per[k] for k in keys], color="#3a6ea5", width=1.0)
        ax.axhline(m.rms, color="k", ls="--", lw=1)
        ax.set_title(f"{name}: per-view rms, mono fit", fontsize=9.5)
        ax.set_xlabel("view"); ax.set_ylabel("px")

        ax = fig.add_subplot(n, 4, i * 4 + 4)
        stereo_cal = s.left if name == "left" else s.right
        rows = [("mono rms", f"{m.rms:.4f} px", "#1b6e3c"),
                ("under the rig", f"{stereo_cal.rms:.4f} px",
                 "#a11d1d" if stereo_cal.rms > 5 * max(m.rms, 1e-9) else "#0e6e76"),
                ("cost of the rig", f"{stereo_cal.rms / max(m.rms, 1e-9):.0f}x"),
                ("views, mono", len(m.views)),
                ("views, stereo", len(stereo_cal.views)),
                ("corners, mono", int(m.n_corners)),
                ("corners, stereo", int(stereo_cal.n_corners))]
        _table(ax, rows, fontsize=8.5)
        ax.set_title(f"{name}: mono against stereo", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))
    _note(fig, "A mono rms far below the stereo one means each camera describes its own "
               "images well and the two cannot be reconciled by a single R, T -- look at "
               "the rig page, not at the lens model. Mono and stereo corner counts differ "
               "because the stereo solve keeps only corners both cameras saw.")
    return fig
