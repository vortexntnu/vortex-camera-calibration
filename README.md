# calibtool

Camera calibration for mono and stereo rigs, checkerboard or ChArUco, built on
OpenCV. It solves, but mostly it exists to let you *disbelieve* the solve: look
at every image, see where the error actually is, throw out the frames that
deserve it, and re-solve in a second.

```
./calibtool/calib gui  perception_pipeline/recordings/recording-2026-08-22T13-00-20
./calibtool/calib run  perception_pipeline/recordings/recording-2026-08-22T13-00-20
./calibtool/calib auto perception_pipeline/recordings/recording-2026-08-22T13-00-20
```

Needs `opencv-python` (4.7+, for the modern `CharucoDetector`), `numpy`,
`pyyaml`, `matplotlib`, and `PyQt5` for the inspector. All present on this
machine.


## Commands

| | |
| --- | --- |
| `calib gui <dataset>` | the inspector. Detect, solve, look, reject, re-solve |
| `calib run <dataset>` | detect, solve, report, write results |
| `calib auto <dataset>` | `run`, then drop flagged views and re-solve, up to `--rounds` |
| `calib detect <dataset>` | fill the detection cache and stop |
| `calib alpha <calibration>` | what each rectify alpha would cost, from a finished solve |

`<dataset>` can be:

- a **recording directory** — it looks inside for `calib/`
- a directory holding **`left/` and `right/`** (or `cam0/`/`cam1/`) → stereo
- a directory of **images** → mono
- explicit `--left DIR --right DIR`

Stereo pairing is by the **digits in the filename**, not sort order:
`left_000042.png` pairs with `right_000042.png`.

### The board

```bash
--board calibtool/boards/charuco_12x9_60mm.yaml       # a preset
--board-kind charuco --columns 12 --rows 9 --square 60 --marker 45 \
  --dictionary DICT_5X5_100                            # or spell it out
--board-kind checkerboard --columns 9 --rows 6 --square 25
```

Sizes on the command line are **millimetres**; in a board YAML they are metres.

> **`--columns`/`--rows` mean different things per board kind.** For charuco they
> count **squares** — the numbers printed on the sheet. For a checkerboard they
> count **inner corners**, one less on each axis than the squares you can see.
> This is the OpenCV convention and the most common way to get a calibration
> silently wrong.

Prefer charuco. Beyond tolerating partial views, a plain checkerboard with an
even inner-corner count on both axes is 180°-ambiguous, and if the two eyes
resolve that ambiguity differently the stereo correspondence is wrong while
every mono number still looks healthy. If you must use a checkerboard, make it
odd × even.

### The lens model

Default is the plain 5-coefficient pinhole model. Reach for more only when the
residual field shows you the need:

```
--rational        add k4..k6          --fix-k3
--thin-prism      add s1..s4          --zero-tangent
--tilted          add taux, tauy      --fix-principal-point
--fisheye         equidistant model   --fix-aspect-ratio
```

Coefficients you cannot see the need for fit noise and make undistortion
misbehave outside the coverage you actually had.

### How many views

```
--views 60     # default: a spread, chosen greedily
--views all    # everything that detected
```

A 423-frame recording is not 423 useful views; most are the board in almost the
same place as the frame before. That matters more than it sounds, because
OpenCV solves a dense system over `9 + 6N` parameters and inverts it every
iteration — **cost grows roughly with the cube of the view count**. Measured
here: 50 views solve in ~1 s, 150 take ~14 s, 400 is minutes.

So the default picks a spread: first cover the frame (distortion is only
constrained where corners actually landed), then keep taking the view most
unlike everything already chosen, in a space of centroid, apparent size, tilt
and in-plane rotation. All image-space, because this runs before there is a
calibration to use.

---

## The inspector

```
./calibtool/calib gui perception_pipeline/recordings/recording-2026-08-22T13-00-20
```

Left/right panes with the overlay drawn in **scene** coordinates, so zooming in
buys precision rather than bigger pixels:

- **green circles** — corners as detected
- **magenta crosses** — where the calibration says they should be
- **yellow lines** — the residual, at the exaggeration on the `residual ×`
  slider. At 1× a good calibration is invisible; that is the point of the slider.

The table is sortable — click `rms`, `epipolar` or `dy` to bring the worst views
to the top. The panel below carries the full report, live.

| key / control | |
| --- | --- |
| `←` `→` | previous / next view |
| `Space` | keep ⇄ drop the selected views (multi-select works) |
| `Ctrl+R` | re-solve on what is currently kept |
| `F` | fit both panes |
| scroll | zoom, drag to pan |
| **Drop flagged** | exclude everything past the robust cut, then re-solve |
| **Keep all** | put everything back |
| **Plots** | the diagnostic figures, live |
| **Rectified** | this pair rectified, side by side with rules to sight along |
| **Save** | write the results |

Rejections persist to `selection.json` the moment you make them, and every
command honours that file, so the CLI re-solves against exactly the set you
curated.

---

## Output

Written to `<dataset>/calibration/` unless you pass `-o`:

| file | |
| --- | --- |
| `calibration.yaml` | everything, plain YAML for people and yaml-cpp, with provenance |
| `calibration_opencv.yml` | the same parameters for `cv::FileStorage` — `fs["K1"] >> K1` |
| `diagnostics_*.png` | coverage, residual field, residual scatter, per-view error, pose spread |
| `rectified_preview.jpg` | one pair rectified, with rules — the crude check that catches sign errors |
| `calibration_report.pdf` | the whole diagnosis in one file, with `--pdf` — see below |
| `rectify_maps.npz` | remap LUTs, with `--maps` |
| `selection.json` | which views you excluded. An *input*, kept separate from the results |
| `detections.pkl` | the detection cache, keyed on board spec, settings, and image mtimes |

`R` and `T` map a point in the **left** camera frame into the **right** one.

### The PDF report

`--pdf` writes `calibration_report.pdf` alongside the rest

```
calib run recordings/recording-2026-08-26T13-36-12 --views all --pdf
calib run images/ --board-kind checkerboard --columns 9 --rows 6 --square 25 --mono --pdf
```

It works the same on `auto`, and the inspector has a **PDF report** button.

It leads with a verdict — clipping, coverage gaps, row error, whether the rig
moved — and then shows the evidence: exposure and clipping, coverage, sharpness
and motion, pose dependence, and the frame before and after undistortion at
both ends of the alpha range. A stereo run adds the mono residuals on their own
(each camera before the rig was imposed), the rig test, the alpha sweep, and
the rectified pair at full size. Mono gets seven pages, stereo thirteen.

Two things worth knowing. It costs a few seconds and re-reads one image pair,
so it is opt-in rather than part of every save. And the time-based tests — the
autocorrelation, the board-speed check, the time-versus-pose verdict — need
adjacent frames, so they are far sharper with `--views all` than with the
default thinned selection; the report says so on the page when the views are
too sparse to answer.

---

## Diagnostics worth reading

Beyond the error tables, two things in the report earn their space.

**Coverage.** `56/64 cells of the frame reached [WARNING: frame border not
covered]` means the distortion model is unconstrained at the periphery, and
undistorting out there is extrapolation. Get the board into the corners.

**Rig consistency** (stereo). Every view gets its own answer for where the two
cameras are, from two independent pose solves and *no stereo constraint*:

```
  rig consistency -- what each view alone says the geometry is,
  from two independent pose solves and no stereo constraint:
    rotation    [+0.009, -0.024, -0.597] deg  +- [0.145, 0.032, 0.245]
    translation [-108.23, -0.04, -2.59] mm  +- [0.68, 0.99, 0.45]
    the rotation spread alone forces 9.95 px of reprojection error
```

On a rigid rig with sound intrinsics those answers pile up on one value and the
spread is corner noise — for a board a thousand pixels across, thousandths of a
degree. A spread far above that means **no single `R, T` can satisfy every
view**, and the stereo RMS you are about to read is a floor set by the capture,
not by the solver. Always read the rotation spread in the pixels it forces:
a hundredth of a degree is nothing until you multiply it by a 2300 px focal
length.

---

## Capture notes

The calibration can only be as good as the poses you gave it.

- get the board into **all four corners** of the frame, not just the middle —
  this is the single most common defect, and the coverage line calls it out
- **tilt hard**, past 30–45°. Shallow tilts leave focal length and board
  distance trading off against each other, which shows up later as a scale
  error rather than a large RMS
- vary the **distance** as well as the position
- short exposure, and hold still at each pose. Motion blur biases corners in a
  direction, which is exactly what a solver cannot tell from a real pose
- 30–60 good views beat 400 mediocre ones, and solve two orders of magnitude
  faster
