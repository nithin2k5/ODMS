# ODMS – Optical Diameter Measurement System

A desktop application (Python, Tkinter/ttk, OpenCV, NumPy, Pillow) that measures
**operator-selected** circular diameters of mechanical parts — wheel outer
diameters, flange outer diameters, recesses, bores — from a live USB camera or
from image files. Everything runs locally with classical image processing; no
trained model or cloud service is used.

Key principles:

* **You choose the ring.** A part has many concentric edges. ODMS never assumes
  the largest circle is the one you want: each measurement is tied to the
  boundary you picked and confirmed. If that boundary is lost, the result is
  **Not detected**. ODMS never switches to a nearby ring.
* **No invented millimetres.** Without a valid calibration for the measured
  plane, results are shown in pixels with *Calibration required*.
* **Detection quality is not accuracy.** The 0–100 quality score describes how
  well the edge was found and fitted. You can only establish dimensional
  accuracy by validating against independently measured parts
  (see [Checking accuracy](#checking-accuracy-against-known-measurements)).

---

## Folder structure

```
ODMS/
├── odms/
│   ├── app.py, __main__.py      entry point (python -m odms)
│   ├── paths.py                 data locations (override with ODMS_DATA_DIR)
│   ├── geometry.py              preview<->image mapping, homography helpers
│   ├── processing.py            background measurement worker thread
│   ├── synthetic.py             synthetic parts with exact ground truth
│   ├── validation.py            validation CLI (bias, repeatability, RMSE)
│   ├── camera/source.py         USB capture thread, bounded queue, reconnect
│   ├── calibration/
│   │   ├── models.py            calibration set, planes, validity rules
│   │   ├── scale.py             straight-on reference-length scale, revalidation
│   │   ├── perspective.py       homography / rectification for tilted planes
│   │   ├── lens.py              checkerboard lens calibration, undistortion
│   │   └── store.py             JSON save/load (forces revalidation on load)
│   ├── measurement/
│   │   ├── models.py            MeasurementSpec, ProcessingParams, results
│   │   ├── preprocess.py        grey, CLAHE, denoise, glare check
│   │   ├── edges.py             sub-pixel radial edge extraction in the band
│   │   ├── circle_fit.py        constrained RANSAC + geometric least squares
│   │   ├── quality.py           quality score and rejection reasons
│   │   ├── measure.py           one boundary: edges -> fit -> checks -> lock
│   │   ├── candidates.py        Hough candidates to help ring picking
│   │   ├── tracking.py          short-window smoothing, stale reset
│   │   ├── engine.py            per-frame pipeline, PASS/FAIL, overlay
│   │   └── overlay.py           drawing of boundaries, centres, labels
│   ├── storage/
│   │   ├── profiles.py          part profiles (named measurements)
│   │   └── results.py           captures, annotated images, CSV export
│   └── ui/                      Tk main window, preview, panels, dialogs
├── tools/make_sample_images.py  generates samples/ for image-file mode
├── tests/                       pytest suite (unit, end-to-end, UI)
├── requirements.txt
└── README.md
```

At runtime the application writes to `data/` (git-ignored):
`data/calibration/`, `data/profiles/`, `data/results/` (with
`results_log.csv` and `images/`).

## Installation and launch

Python 3.10+ is required (developed with 3.12).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

python tools/make_sample_images.py   # optional: sample images for image-file mode
python -m odms                       # start the application
python -m odms samples/flange.png    # start and open an image directly
python -m pytest                     # run the test suite
```

On Linux you may need the Tk package (`sudo apt install python3-tk`).

## Quick start without a camera (image-file mode)

1. Run `python tools/make_sample_images.py`. It writes `samples/wheel.png`,
   `samples/flange.png`, `samples/flange_tilted.png`, `ground_truth.json`, an
   example profile and calibration, and a `validation/` set.
2. `python -m odms`, then **File ▸ Load Sample Image ▸ flange.png**.
3. **Calibration** tab ▸ *Scale: mark reference*. Click the centres of the
   two crosses on the ruler (zoom with the mouse wheel), then enter `100` mm.
4. **Measurements** tab ▸ *Add*. Drag an ROI around the flange, then press
   *3. Pick edge* and click on the outer edge. Check the cyan preview, enter a
   name, and press *Confirm & save*. Repeat for the recess and the bore.
5. *Save Result* stores an annotated image and log rows. *Export CSV* writes
   the session's results.

Alternatively, load `samples/flange_profile.json` (Measurements ▸ Open) and
`samples/flange_calibration.json` (Calibration ▸ Load). The loaded calibration
first requires revalidation: select the plane, press *Revalidate plane…*,
click the two ruler crosses and enter 100 mm.

---

## Camera mounting and lighting

* **Mount the camera rigidly**, ideally looking straight down at the part
  (optical axis perpendicular to the measured face). Any movement, zoom change
  or refocus after calibration invalidates the scale.
* **Fix zoom and focus manually.** Turn off autofocus and auto-zoom. If the
  camera driver allows it, also fix exposure and white balance so that edge
  contrast does not drift.
* **Use the highest resolution the processing rate allows**, and fill as much of
  the frame with the part as practical. Diameter resolution is set by pixels per
  millimetre.
* **Lighting:**
  * *Back-lighting* (part silhouetted on a light panel) gives the sharpest outer
    diameters and is the best choice for through-features.
  * *Diffuse front light* (dome, ring light with diffuser, or light tent) is
    better for recesses and steps. Avoid point sources that create specular
    highlights on machined metal.
  * Low-angle (dark-field) light can bring out chamfers and steps, but it
    shifts the apparent edge. Calibrate and validate with the same lighting.
* **Telecentric lenses** remove most magnification change with height and
  perspective. They are recommended when parts vary in height or when you need
  high accuracy.

## Calibration

The **Calibration** tab manages one *calibration set* per camera and
resolution. A set contains an optional lens calibration and one or more
**planes**. A plane is a physical surface at a specific height/distance from
the camera, for example `flange face` and `recess floor`. Every measurement
names the plane it lies on and gets millimetres **only** from that plane's
calibration. A scale measured on the flange face is never applied to a recess
floor at a different depth.

Before you calibrate, fill in the *Setup record* (camera distance, zoom, focus,
measured surface height, lighting). It is saved with each plane so the setup can
be reproduced later.

### Straight-on scale (reference length)

Use this when the camera faces the measured plane squarely.

1. Place a reference of known length **on the same plane and at the same height**
   as the boundary you will measure: a calibrated scale, gauge block, or a
   precision ring. A ring used as a diameter reference is ideal.
2. Choose the *Plane to calibrate* and press *Scale: mark reference*.
3. Click both ends of the reference. Zoom in with the mouse wheel; pan with the
   right or middle mouse button. Enter the true length in mm.
4. The dialog shows px/mm and how much a 1-pixel marking error changes the scale.
   Use a reference that spans most of the field of view.

`diameter_mm = diameter_px / px_per_mm`

### Lens distortion (checkerboard)

Use *Lens: checkerboard…* to correct radial and tangential distortion. Enter the
**inner** corner counts and the square size, then capture 12 or more views that
cover the whole image, including the corners, with the board at varied tilts.
Check the RMS reprojection error: below 0.5 px is good. After the lens
correction is applied, every displayed and measured image is undistorted.
Because it changes the image geometry, existing plane calibrations become
invalid and must be redone. Re-check the ring positions in your profiles too.

### Tilted planes (perspective correction)

When the camera cannot face the part squarely, circles appear as ellipses.
ODMS **never reports an ellipse's axis as a diameter**. To measure a tilted
part:

* *Perspective: mark points*: click 4 or more non-collinear reference points
  that lie on the measured plane (for example marks on a fixture plate). Press
  *Finish points* and enter each point's real X/Y position in mm. With exactly
  4 points the homography has no redundancy to detect a marking mistake, so use
  5 or more points when you can. The dialog then reports the fit residual.
* *Perspective: checkerboard…*: lay a flat checkerboard on the measured plane.
  All of its corners are used.

Measurements on a perspective plane are fitted in the rectified
(perspective-corrected) image, where the boundary is a true circle. The overlay
is mapped back onto the camera view.

### Validity and revalidation

A plane's state appears in the planes table:

| State | Meaning | Millimetres shown? |
| --- | --- | --- |
| Valid | calibrated and confirmed for this camera/resolution/lens | yes |
| Revalidation required | loaded from disk, or *Setup changed* was pressed, or a revalidation failed | no (pixels) |
| Invalid | different camera, different resolution, or the lens calibration changed | no (pixels) |
| Missing | the measurement's plane has no calibration | no (pixels) |

* Calibrations are stored with the camera identity (`usb:<index>`, or
  `image-file`), resolution, lens id and setup record. OpenCV exposes no
  camera serial number, so keep each camera on the same USB port/index.
* A calibration **loaded from disk always requires revalidation**, because the
  physical setup may have changed while the file sat on disk.
* After moving the camera or changing zoom, focus, or part height, press
  *Setup changed*. ODMS cannot detect these changes by itself.
* *Revalidate plane…*: click the two ends of a known reference on that plane and
  enter its length. If the measured length is within the tolerance (default
  0.5 %), the plane becomes valid again. Otherwise, recalibrate.
* Each change to a calibration creates a new calibration ID. Profiles and
  exported results record this ID for traceability.

## Selecting the correct ring

For each measurement (**Measurements ▸ Add**):

1. **Draw ROI**: drag a rectangle around the part. Edge points outside the ROI
   are ignored.
2. **Set centre** (optional): click the approximate centre. This helps when
   rings are interrupted, for example by wheel spokes.
3. **Pick edge**: click exactly on the boundary you want. ODMS proposes
   candidate circles (Hough), verifies each one with the robust edge fit, and
   keeps only a ring whose fitted circumference passes through your click. It
   then places a search band of ±*Band* px around that ring.
4. **Preview.** The draft boundary is drawn in cyan, with its search band and a
   live status line (diameter, quality, reason). Adjust *Inner/Outer radius* to
   narrow the band if a neighbouring ring falls inside it. *Re-centre on fit*
   centres the band on the fitted boundary.
5. **Edge polarity.** `rising` = dark inside, bright outside; `falling` = bright
   inside, dark outside. Leave it on `auto` during set-up. On confirmation it
   locks to the observed direction, so an adjacent edge with the opposite
   contrast can never be picked up.
6. **Limits** (optional): min/max mm for PASS/FAIL.
7. **Confirm & save** locks the measurement to the confirmed ring radius. If
   the boundary found in the band later differs from the confirmed radius by
   more than the *Ring lock tolerance* (Processing tab, default 5 %), the
   result is **Not detected**. If a part may legitimately differ by more than
   that, widen the tolerance deliberately.

Mouse positions are converted from the scaled, letterboxed and zoomed preview
back to original image pixels. Overlays are drawn at full resolution, so zooming
in shows the true sub-pixel fit.

## Measuring multiple diameters

* Add as many named measurements as you need, for example *Wheel outer
  diameter*, *Flange outer diameter* or *Recess diameter*. Each one has its own
  ROI, band, polarity, plane, limits and processing parameters.
* Assign each measurement to the **plane** it lies on. A recess floor at a
  different depth needs its own plane calibration. Calibrate it with a reference
  placed at that depth, or use a perspective calibration on that surface.
* Save the set as a **part profile** (Measurements ▸ Save). The profile stores
  ring geometry in the camera view, together with the calibration and lens id it
  was set up with. ODMS warns you when you open a profile under a different
  lens correction.
* **Results table:** name, diameter (mm), diameter (px), detection quality,
  status, and detail (fit RMS, coverage, inliers, limits, reason).
  * `PASS` / `FAIL` appear only when the plane calibration is valid and the
    detection passed all checks.
  * `MEASURED` means calibrated, with no limits set.
  * `Calibration required`, `Revalidation required` and `Calibration invalid`
    mean the reading is shown in pixels only.
  * `Rejected` means a boundary was found but failed a quality check. The
    reason is shown.
  * `Not detected` means no acceptable boundary was found in the locked band.
* **Smoothing:** valid readings are averaged over a short window (default 5
  frames). The window is cleared immediately when detection fails, the part
  moves more than the move tolerance, the diameter jumps, or the calibration or
  set-up changes. If the camera stops delivering frames, the readings are
  cleared.
* **Save Result** (Ctrl+S) writes a timestamped annotated PNG, the raw frame
  and one log row per measurement to `data/results/`. **Export CSV** writes the
  session's saved rows (or the current live readings). Columns include:
  diameter (mm and px), units, status, reason, limits, detection quality, fit
  RMS, inlier ratio, angular coverage, largest gap, edge width, smoothing
  count, measurement space, calibration ID, plane calibration ID, calibration
  state, px/mm, lens applied, camera ID, resolution, and image paths.

## Processing pipeline

For each measurement and frame:

1. Undistort the frame if a lens calibration is active. For a perspective
   plane, warp only the window around the band into the rectified plane.
2. Crop the window, convert it to grey, apply CLAHE contrast enhancement and a
   light Gaussian denoise.
3. Cast 360 rays from the centre through the search band, sample the profile
   bilinearly, differentiate it, and keep the strongest edge peaks with the
   locked polarity, refined to sub-pixel position with a parabolic fit.
4. Fit the circle with RANSAC (constrained radius range and centre offset, so
   bolt holes and small features cannot win), then geometric least squares on
   the inliers, re-selecting inliers after each refinement.
5. Check for glare (saturated pixels in the band), angular coverage and
   largest gap, inlier ratio, fit RMS, edge width (blur) and ellipticity
   (camera tilt), and compare the radius with the confirmed ring.

Hough circles are used **only** to propose a centre while you pick a ring. The
diameter never comes from a bounding box, a minimum enclosing circle, or a
Hough radius. Every parameter is adjustable per measurement on the
**Processing** tab. *Show edge points* draws inliers in green and rejected
points in red.

## Troubleshooting

**Glare / reflections** (*Glare*, *Edge points inconsistent*)
* Use diffuse lighting (dome, diffuser, cross-polarisation), or back-lighting
  for outer diameters. Lower the exposure until no part of the band is
  saturated.
* Narrow the band so that highlights next to the edge fall outside it. Set the
  polarity explicitly. Increase *Edge candidates per ray* to 2–3 so RANSAC can
  skip a highlight that is stronger than the real edge.

**Perspective** (*Boundary is elliptical*)
* The camera is not square to the plane. Re-mount it, or calibrate that plane
  with *Perspective* (marked points or checkerboard). Raise *Max ellipticity*
  only when you understand the resulting error. A 1 % axis ratio corresponds to
  roughly 8° of tilt.
* Strong barrel distortion near the image edges also makes circles elliptical.
  Run the lens calibration, and keep the parts near the image centre.

**Unstable or jumping readings**
* *Edge blurred*: refocus, reduce motion (shorter exposure, more light), or
  reduce *Denoise blur sigma*.
* *Poor circle fit*: the edge is chamfered, burred or rough. Narrow the band to
  the intended edge, or increase *RANSAC inlier threshold* slightly.
* *Boundary incomplete*: part of the ring is hidden or outside the ROI or
  image. Enlarge the ROI, or lower *Min angular coverage* only if you accept a
  less reliable fit from a partial arc.
* *Confirmed ring not found*: the band now contains a different ring, or the
  part has changed by more than the lock tolerance. Re-pick the ring, or
  re-place the part in the fixture.
* Readings that drift with temperature or vibration point to an unstable mount.
  Revalidate regularly.

**Camera**
* *Unavailable*: try another index (*Scan*) or backend (DirectShow / Media
  Foundation on Windows). Close other applications that use the camera.
* *Disconnected*: ODMS retries every 2 s and clears the readings meanwhile.
* The requested resolution is a hint. The camera may choose another
  resolution, and the status bar shows the actual one. A resolution change
  invalidates the calibration.

## Checking accuracy against known measurements

The application makes **no accuracy claim** (such as ±0.1 mm) for your setup.
Establish it as follows:

1. Collect reference parts or rings whose diameters were measured independently
   with a traceable instrument (CMM, calibrated ring gauges, micrometer). Cover
   the range of sizes you will measure.
2. With the production setup, calibration and profile, capture each reference
   part **at least 10 times, removing and re-placing it between captures**. Use
   *Save Result*. The raw frames are stored next to the annotated images.
3. Write `references.csv` with the columns
   `image,part_id,measurement,reference_mm`.
4. Run:

   ```bash
   python -m odms.validation --profile data/profiles/part.json \
       --calibration data/calibration/cal.json --images data/results/images \
       --references references.csv --setup-unchanged --out validation/report
   ```

   `--setup-unchanged` is your statement that the images were captured with the
   unchanged, calibrated setup. Without it, no millimetre values are produced.
5. The report (`*_summary.csv`, `*_samples.csv`) gives, for each measurement and
   part: **bias** (mean error), **repeatability** (standard deviation of the
   repeated readings), maximum absolute error, RMSE, and the number of
   captures that could not be measured, with reasons.
6. Decide whether the setup is fit for the tolerance. A common rule of thumb
   asks for the measurement uncertainty (bias plus about 2–3× repeatability) to
   be well under 1/10 to 1/4 of the tolerance band. Repeat the validation after
   any change to the camera, lens, lighting or fixture, and periodically.

A demonstration on synthetic images:

```bash
python tools/make_sample_images.py
python -m odms.validation --profile samples/flange_profile.json \
    --calibration samples/flange_calibration.json --images samples/validation \
    --references samples/validation/references.csv --setup-unchanged
```

The synthetic images have perfect geometry, so their small errors show the
algorithm's sub-pixel behaviour. **They say nothing about a real camera
setup.**

## Tests

`python -m pytest` runs:

* unit tests for geometry/coordinate mapping, circle fitting, calibration,
  storage and validation statistics;
* end-to-end engine tests on synthetic wheel and flange images. These cover
  scale and perspective calibration, bolt-hole rejection, PASS/FAIL, per-plane
  calibration, a removed part (Not detected), and the ring lock;
* camera-failure and worker tests, and a test that drives the real Tk window
  through the image-file workflow (skipped when no display is available).

## Limitations

* Camera identity is the OpenCV device index. Moving a camera to another port
  can change it.
* The perspective model assumes a flat measured plane. Measuring a boundary at
  a different height from the plane's reference points gives wrong values. Give
  each such surface its own plane.
* Edge position depends on lighting and edge geometry (chamfers, burrs).
  Calibrate and validate with the same lighting, on parts that are
  representative of production.
