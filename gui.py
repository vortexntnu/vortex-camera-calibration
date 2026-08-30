"""The inspector: look at the images, see where the error is, throw views out.

The loop this exists to make fast is: sort by error, look at the worst view,
decide whether it is a bad image or a bad model, drop it if it is a bad image,
re-solve, look again. Everything is arranged around that.

What you are looking at in the image pane:

  green dots      corners as detected
  magenta dots    where the current calibration says those corners should be
  yellow lines    the residual between them, drawn at an exaggeration factor --
                  at 1x a good calibration is invisible, which is the point of
                  the slider

The residual arrows are the reason to look at an image at all. A view where they
all point the same way is a pose the solver could not fit -- usually motion blur
or a board that moved during the exposure, and worth dropping. A view where they
swirl outward at the frame edge is not that view's fault: it is the distortion
model running out of terms, and dropping views will not fix it. A single corner
with a long arrow among short ones is a misdetection.

Rejections are written to selection.json beside the results, so the CLI can
re-solve against exactly the set you curated.
"""

from __future__ import annotations

import os
import pathlib
import signal
import sys

import cv2
import numpy as np

# opencv-python ships its own copy of the Qt platform plugins and points
# QT_QPA_PLATFORM_PLUGIN_PATH at them when it is imported. PyQt5 then tries to
# load that copy against its own Qt build, fails to initialise xcb, and the
# process aborts before a window ever appears. Dropping the variable -- only
# when it is the one cv2 set -- sends PyQt5 back to its own plugins.
_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
if _plugin_path and (os.sep + "cv2" + os.sep) in _plugin_path:
    del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from . import detect as detect_mod
from . import io as io_mod
from . import report as report_mod
from . import select as select_mod
from .calibrate import (CalibrationError, Model, StereoCalibration, calibrate_mono,
                        calibrate_stereo, outliers)
from .cli import board_from_args, model_from_args, _target_views
from .dataset import discover

GREEN = QtGui.QColor(70, 230, 90)
MAGENTA = QtGui.QColor(255, 80, 220)
YELLOW = QtGui.QColor(255, 210, 40)
BLUE = QtGui.QColor(90, 170, 255)


# ---------------------------------------------------------------------------


class ImageView(QtWidgets.QGraphicsView):
    """Zoom/pan image pane with the overlay drawn in scene coordinates.

    Overlay geometry lives in the scene, not in the pixmap, so zooming in gives
    you more precision instead of bigger pixels -- which matters when the thing
    you are judging is a third of a pixel long.
    """

    def __init__(self, title: str):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor(24, 24, 28))
        self.pixmap_item = None
        self._fitted = False
        self.title = title

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def set_image(self, bgr: np.ndarray | None):
        self.scene().clear()
        self.pixmap_item = None
        if bgr is None:
            self.scene().addText("no image").setDefaultTextColor(QtGui.QColor(180, 180, 180))
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else \
            cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap.fromImage(image))
        self.setSceneRect(0, 0, w, h)
        if not self._fitted:
            self.fit()
            self._fitted = True

    def fit(self):
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)

    def draw_overlay(self, detection, residual, gain: float, show):
        if self.pixmap_item is None:
            return
        scene = self.scene()
        if show.get("markers") and detection is not None and detection.marker_corners is not None:
            pen = QtGui.QPen(BLUE, 0)
            for quad in detection.marker_corners:
                poly = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in quad])
                scene.addPolygon(poly, pen)
        if show.get("detected") and detection is not None and detection.ok:
            pen = QtGui.QPen(GREEN, 0)
            for x, y in detection.corners:
                scene.addEllipse(float(x) - 3, float(y) - 3, 6, 6, pen)
        if residual is None:
            return
        obs, proj = residual.observed, residual.projected
        if show.get("projected"):
            pen = QtGui.QPen(MAGENTA, 0)
            for x, y in proj:
                scene.addLine(x - 4, y, x + 4, y, pen)
                scene.addLine(x, y - 4, x, y + 4, pen)
        if show.get("residuals"):
            pen = QtGui.QPen(YELLOW, 0)
            for (ox, oy), (px, py) in zip(obs, proj):
                scene.addLine(px, py, px + (ox - px) * gain, py + (oy - py) * gain, pen)
        if show.get("ids") and residual is not None:
            for (x, y), i in zip(obs, residual.ids):
                t = scene.addSimpleText(str(int(i)))
                t.setBrush(QtGui.QBrush(GREEN))
                t.setPos(float(x) + 5, float(y) - 16)
                t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, False)


# ---------------------------------------------------------------------------


class SolveWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), "")
        except Exception as exc:  # noqa: BLE001 -- surfaced in the status bar
            self.done.emit(None, str(exc))


class DetectWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object, str)

    def __init__(self, dataset, board, settings, cache, jobs, force):
        super().__init__()
        self.args = (dataset, board, settings, cache, jobs, force)

    def run(self):
        dataset, board, settings, cache, jobs, force = self.args
        try:
            result = detect_mod.run(
                dataset, board, settings, cache_path=cache, workers=jobs,
                progress=lambda d, t, n: self.progress.emit(d, t, n), force=force)
            self.done.emit(result, "")
        except Exception as exc:  # noqa: BLE001
            self.done.emit(None, str(exc))


# ---------------------------------------------------------------------------

COLUMNS = ["view", "keep", "corners", "rms", "epipolar", "dy", "dist m", "tilt"]


class Inspector(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = discover(args.dataset, args.left, args.right)
        self.board = board_from_args(args)
        self.out = pathlib.Path(args.out) if args.out else self.dataset.root / "calibration"
        self.out.mkdir(parents=True, exist_ok=True)

        self.detections = None
        self.result = None
        self.excluded = io_mod.read_selection(self.out / "selection.json")
        self.solving = False
        self._image_cache: dict[str, np.ndarray] = {}

        self.setWindowTitle(f"calib -- {self.dataset.root}")
        self.resize(1800, 1000)
        self._build()
        self.start_detection(force=args.redetect)

    # -- construction -----------------------------------------------------

    def _build(self):
        self.views = {}
        for cam in self.dataset.cameras:
            self.views[cam] = ImageView(cam)

        images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        for cam in self.dataset.cameras:
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(cam)
            label.setStyleSheet("font-weight: bold; padding: 2px;")
            lay.addWidget(label)
            lay.addWidget(self.views[cam])
            images.addWidget(box)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.table.setColumnWidth(0, 70)
        for c in range(1, len(COLUMNS)):
            self.table.setColumnWidth(c, 68)

        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QtGui.QFont("monospace", 9))

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.table)
        right.addWidget(self.summary)
        right.setSizes([620, 380])

        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(images)
        main.addWidget(right)
        main.setSizes([1250, 550])
        self.setCentralWidget(main)

        self._toolbar()
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(240)
        self.status.addPermanentWidget(self.progress)

        for key, slot in (("Space", self.toggle_selected), ("Left", lambda: self.step(-1)),
                          ("Right", lambda: self.step(1)), ("F", self.fit_all)):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, slot)

    def _toolbar(self):
        tb = self.addToolBar("main")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        self.act_solve = tb.addAction("Solve", self.start_solve)
        self.act_solve.setShortcut("Ctrl+R")
        tb.addAction("Drop flagged", self.drop_flagged)
        tb.addAction("Keep all", self.keep_all)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" views "))
        self.spin_views = QtWidgets.QSpinBox()
        self.spin_views.setRange(0, 5000)
        self.spin_views.setValue(_target_views(self.args.views))
        self.spin_views.setSpecialValueText("all")
        self.spin_views.setToolTip(
            "How many views to solve on. Cost grows about cubically with this, "
            "so 40-80 keeps the loop interactive. 0 means all of them.")
        tb.addWidget(self.spin_views)

        tb.addWidget(QtWidgets.QLabel("  sigma "))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 20.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(self.args.sigma)
        self.spin_sigma.setToolTip("Robust (median-absolute-deviation) cut for flagging views.")
        self.spin_sigma.valueChanged.connect(self.refresh_table)
        tb.addWidget(self.spin_sigma)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" residual x"))
        self.slider_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_gain.setRange(1, 300)
        self.slider_gain.setValue(40)
        self.slider_gain.setMaximumWidth(150)
        self.slider_gain.valueChanged.connect(self.redraw)
        tb.addWidget(self.slider_gain)
        self.label_gain = QtWidgets.QLabel("40x")
        self.slider_gain.valueChanged.connect(lambda v: self.label_gain.setText(f"{v}x"))
        tb.addWidget(self.label_gain)
        tb.addSeparator()

        self.toggles = {}
        for name, text, on in (("detected", "detected", True), ("projected", "projected", True),
                               ("residuals", "residuals", True), ("markers", "markers", False),
                               ("ids", "ids", False)):
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(on)
            cb.stateChanged.connect(self.redraw)
            self.toggles[name] = cb
            tb.addWidget(cb)
        tb.addSeparator()

        tb.addAction("Plots", self.show_plots)
        tb.addAction("Rectified", self.show_rectified)
        tb.addAction("Save", self.save)
        tb.addAction("PDF report", self.save_pdf)

    # -- pipeline ---------------------------------------------------------

    def start_detection(self, force=False):
        self.status.showMessage("detecting corners...")
        settings = detect_mod.DetectorSettings(min_corners=self.args.min_corners)
        self.detect_worker = DetectWorker(
            self.dataset, self.board, settings, self.out / "detections.pkl",
            self.args.jobs, force)
        self.detect_worker.progress.connect(self._on_detect_progress)
        self.detect_worker.done.connect(self._on_detected)
        self.detect_worker.start()

    def _on_detect_progress(self, done, total, name):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_detected(self, detections, error):
        if error:
            QtWidgets.QMessageBox.critical(self, "detection failed", error)
            return
        self.detections = detections
        self.status.showMessage(
            f"{len(self.dataset.views)} views detected. "
            f"{len(self.excluded)} excluded from a previous session." if self.excluded
            else f"{len(self.dataset.views)} views detected.")
        self.refresh_table()
        self.start_solve()

    def chosen_keys(self) -> list[str]:
        keys = [v.key for v in self.dataset.views if v.key not in self.excluded]
        size = self.detections.image_size[self.dataset.cameras[0]]
        chosen, _ = select_mod.select_views(
            self.detections.per_camera, size, keys, target=self.spin_views.value(),
            min_corners=self.args.min_corners, require_all_cameras=self.dataset.is_stereo)
        return chosen

    def start_solve(self):
        if self.detections is None or self.solving:
            return
        keys = self.chosen_keys()
        if len(keys) < 4:
            self.status.showMessage(f"only {len(keys)} usable views -- nothing to solve")
            return
        self.solving = True
        self.act_solve.setEnabled(False)
        self.status.showMessage(f"solving on {len(keys)} views...")
        self.progress.setRange(0, 0)

        board, model = self.board, model_from_args(self.args)
        size = self.detections.image_size[self.dataset.cameras[0]]
        stereo = self.dataset.is_stereo and not self.args.mono
        per_cam = self.detections.per_camera
        min_corners = self.args.min_corners
        fix = not self.args.refine_intrinsics
        alpha = self.args.alpha

        if stereo:
            def fn():
                return calibrate_stereo(board, per_cam["left"], per_cam["right"], size,
                                        keys=keys, model=model, min_corners=min_corners,
                                        fix_intrinsics=fix, alpha=alpha)
        else:
            cam = self.dataset.cameras[0]

            def fn():
                return calibrate_mono(board, per_cam[cam], size, keys=keys, model=model,
                                      min_corners=min_corners, camera=cam)

        self.solve_worker = SolveWorker(fn)
        self.solve_worker.done.connect(self._on_solved)
        self.solve_worker.start()

    def _on_solved(self, result, error):
        self.solving = False
        self.act_solve.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if error:
            self.status.showMessage(f"solve failed: {error}")
            self.summary.setPlainText(f"solve failed:\n\n{error}")
            return
        self.result = result
        self.summary.setPlainText(report_mod.summary(result))
        self.status.showMessage(
            f"solved on {len(self._solved_keys())} views -- rms {result.rms:.4f} px")
        self.refresh_table()

    def _solved_keys(self):
        if self.result is None:
            return []
        return list(self.result.views) if isinstance(self.result, StereoCalibration) \
            else list(self.result.views)

    # -- table ------------------------------------------------------------

    def _row_data(self, key):
        """(corners, rms, epipolar, dy, distance, tilt) for one view."""
        n = min((self.detections.get(c, key).count for c in self.dataset.cameras), default=0)
        if self.result is None:
            return n, None, None, None, None, None
        if isinstance(self.result, StereoCalibration):
            v = self.result.views.get(key)
            if v is None:
                return n, None, None, None, None, None
            return (n, v.rms, v.epipolar_rms, v.rect_dy_rms,
                    v.left.distance, v.left.tilt_deg)
        v = self.result.views.get(key)
        if v is None:
            return n, None, None, None, None, None
        return n, v.rms, None, None, v.distance, v.tilt_deg

    def flagged(self) -> set[str]:
        if self.result is None:
            return set()
        return {k for k, _ in report_mod.suspects(self.result, sigma=self.spin_sigma.value())}

    def refresh_table(self):
        if self.detections is None:
            return
        current = self.current_key()
        flagged = self.flagged()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.dataset.views))
        for row, view in enumerate(self.dataset.views):
            key = view.key
            n, rms, epi, dy, dist, tilt = self._row_data(key)
            values = [key, "" if key in self.excluded else "keep", n, rms, epi, dy, dist, tilt]
            for col, value in enumerate(values):
                if col == 0:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.DisplayRole, int(key) if key.isdigit() else key)
                elif col == 1:
                    item = QtWidgets.QTableWidgetItem("keep" if key not in self.excluded else "drop")
                elif value is None:
                    item = QtWidgets.QTableWidgetItem("")
                elif isinstance(value, (int, np.integer)):
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.DisplayRole, int(value))
                else:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(QtCore.Qt.DisplayRole, round(float(value), 3))
                item.setData(QtCore.Qt.UserRole, key)
                if key in self.excluded:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(130, 130, 130)))
                    f = item.font()
                    f.setStrikeOut(True)
                    item.setFont(f)
                elif key in flagged:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(90, 30, 30)))
                elif n == 0:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(160, 120, 60)))
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        if current:
            self.select_key(current)
        elif self.table.rowCount():
            self.table.selectRow(0)

    def current_key(self):
        items = self.table.selectedItems()
        return items[0].data(QtCore.Qt.UserRole) if items else None

    def selected_keys(self):
        return sorted({i.data(QtCore.Qt.UserRole) for i in self.table.selectedItems()},
                      key=lambda k: (0, int(k)) if k.isdigit() else (1, k))

    def select_key(self, key):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(QtCore.Qt.UserRole) == key:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return

    def step(self, delta):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            return
        self.table.selectRow(max(0, min(self.table.rowCount() - 1, rows[0] + delta)))

    # -- actions ----------------------------------------------------------

    def toggle_selected(self):
        keys = self.selected_keys()
        if not keys:
            return
        # One gesture, one meaning: if anything in the selection is kept, drop
        # the lot; only when all of it is already dropped does this put it back.
        if any(k not in self.excluded for k in keys):
            self.excluded.update(keys)
        else:
            self.excluded.difference_update(keys)
        self.persist_selection()
        self.refresh_table()
        self.status.showMessage(
            f"{len(self.excluded)} view(s) excluded -- press Ctrl+R to re-solve")

    def drop_flagged(self):
        flagged = self.flagged()
        if not flagged:
            self.status.showMessage("nothing flagged at this sigma")
            return
        self.excluded |= flagged
        self.persist_selection()
        self.refresh_table()
        self.status.showMessage(f"dropped {len(flagged)} flagged view(s); re-solving")
        self.start_solve()

    def keep_all(self):
        self.excluded.clear()
        self.persist_selection()
        self.refresh_table()
        self.status.showMessage("all views back in; press Ctrl+R to re-solve")

    def persist_selection(self):
        io_mod.write_selection(self.out / "selection.json", self.excluded)

    def save(self):
        if self.result is None:
            self.status.showMessage("nothing solved yet")
            return
        from .cli import write_everything
        self.persist_selection()
        write_everything(self.result, self.out, self.dataset, self.args, self.detections)
        extra = "\ncalibration_report.pdf" if getattr(self.args, "pdf", False) else ""
        QtWidgets.QMessageBox.information(
            self, "saved",
            f"Wrote calibration.yaml, calibration_opencv.yml, diagnostics{extra} and\n"
            f"selection.json ({len(self.excluded)} excluded) to\n\n{self.out}")

    def save_pdf(self):
        """The full report, on demand -- it costs a few seconds, so it is a
        separate button rather than part of every save."""
        if self.result is None:
            self.status.showMessage("nothing solved yet")
            return
        from . import pdfreport
        self.status.showMessage("building report...")
        QtWidgets.QApplication.processEvents()
        try:
            path = pdfreport.build(self.result, self.dataset, self.detections,
                                   self.out / "calibration_report.pdf")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "report failed", str(exc))
            self.status.showMessage("report failed")
            return
        self.status.showMessage(f"wrote {path}")
        QtWidgets.QMessageBox.information(self, "report", f"Wrote\n\n{path}")

    def show_plots(self):
        if self.result is None:
            return
        import matplotlib
        matplotlib.use("Qt5Agg", force=True)
        import matplotlib.pyplot as plt
        for fig in report_mod.figures(self.result).values():
            fig.show()
        plt.show(block=False)

    def show_rectified(self):
        if not isinstance(self.result, StereoCalibration):
            self.status.showMessage("rectified preview needs a stereo solve")
            return
        key = self.current_key() or next(iter(self.result.views))
        view = next(v for v in self.dataset.views if v.key == key)
        canvas = report_mod.rectified_preview(
            self.result, cv2.imread(str(view.paths["left"])),
            cv2.imread(str(view.paths["right"])))
        window = QtWidgets.QMainWindow(self)
        window.setWindowTitle(f"rectified pair -- view {key} "
                              f"(features should cross each green rule at the same height)")
        pane = ImageView("rectified")
        pane.set_image(canvas)
        window.setCentralWidget(pane)
        window.resize(1600, 700)
        window.show()

    def fit_all(self):
        for v in self.views.values():
            v.fit()

    # -- drawing ----------------------------------------------------------

    def on_select(self):
        self.redraw()

    def _image(self, path) -> np.ndarray | None:
        key = str(path)
        if key not in self._image_cache:
            if len(self._image_cache) > 12:
                self._image_cache.clear()
            self._image_cache[key] = cv2.imread(key)
        return self._image_cache[key]

    def redraw(self):
        key = self.current_key()
        if key is None or self.detections is None:
            return
        view = next((v for v in self.dataset.views if v.key == key), None)
        if view is None:
            return
        gain = self.slider_gain.value()
        show = {n: cb.isChecked() for n, cb in self.toggles.items()}
        for cam in self.dataset.cameras:
            pane = self.views[cam]
            pane.set_image(self._image(view.paths[cam]))
            residual = None
            if self.result is not None:
                if isinstance(self.result, StereoCalibration):
                    sv = self.result.views.get(key)
                    residual = (sv.left if cam == "left" else sv.right) if sv else None
                else:
                    residual = self.result.views.get(key)
            pane.draw_overlay(self.detections.get(cam, key), residual, gain, show)

        det = self.detections.get(self.dataset.cameras[0], key)
        bits = [f"view {key}", f"{det.count} corners"]
        if det.error:
            bits.append(det.error)
        n, rms, epi, dy, dist, tilt = self._row_data(key)
        if rms is not None:
            bits.append(f"rms {rms:.3f} px")
            if dy is not None:
                bits.append(f"rectified dy {dy:.3f} px")
            if dist is not None:
                bits.append(f"{dist:.2f} m, {tilt:.0f} deg tilt")
        elif self.result is not None:
            bits.append("not in the solve")
        if key in self.excluded:
            bits.append("EXCLUDED")
        self.status.showMessage("   |   ".join(bits))


def _install_sigint(app) -> QtCore.QTimer:
    """Make Ctrl+C in the launching terminal close the inspector.

    Two things stop it working by default. `app.exec_()` sits inside Qt's C
    event loop, and Python only runs a signal handler between bytecodes, so the
    handler never gets a turn -- hence a timer that fires often enough to hand
    the interpreter control, doing nothing else. And the solve runs on a
    QThread inside OpenCV, which cannot be asked to stop, so a graceful quit
    would block on joining it.

    So: first Ctrl+C asks the window to close, which is the polite path when
    nothing is running. A second one leaves immediately, because the first
    failing means a worker is wedged in C and no amount of waiting will help.
    """
    pressed = {"once": False}

    def handler(_signum, _frame):
        if pressed["once"]:
            print("\nforced exit", file=sys.stderr)
            os._exit(130)
        pressed["once"] = True
        print("\ninterrupted -- closing. Press Ctrl+C again to exit now.",
              file=sys.stderr)
        app.quit()

    signal.signal(signal.SIGINT, handler)
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(150)
    return timer


def launch(args) -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = Inspector(args)
    window.show()
    timer = _install_sigint(app)          # kept alive by the reference
    code = app.exec_()
    timer.stop()
    # Worker threads may still be inside an OpenCV call that cannot be
    # interrupted; waiting on them would hang the exit the user just asked for.
    for worker in (getattr(window, "solve_worker", None),
                   getattr(window, "detect_worker", None)):
        if worker is not None and worker.isRunning():
            os._exit(code)
    return code
