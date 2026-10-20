"""Generate synthetic sample images for trying ODMS without a camera.

    python tools/make_sample_images.py [--out samples]

Writes:
  wheel.png, flange.png, flange_tilted.png   clean test images
  ground_truth.json                           true diameters and fiducial positions
  flange_profile.json, flange_calibration.json
                                              ready-made profile + calibration for flange.png
  validation/                                 repeated noisy captures + references.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odms.calibration.models import CalibrationSet  # noqa: E402
from odms.calibration.scale import make_scale_plane  # noqa: E402
from odms.calibration.store import save_calibration  # noqa: E402
from odms.geometry import apply_homography  # noqa: E402
from odms.measurement.models import MeasurementSpec  # noqa: E402
from odms.storage.profiles import PartProfile, save_profile  # noqa: E402
from odms.storage.results import imwrite  # noqa: E402
from odms.synthetic import fiducials_px, flange_scene, make_samples, render  # noqa: E402

CAMERA_ID = "image-file"


def flange_specs(mapping: np.ndarray) -> list[MeasurementSpec]:
    c = apply_homography(mapping, np.array([[0.0, 0.0]]))[0]
    ppm = mapping[0, 0]
    specs = []
    for name, r_mm, polarity, lo, hi in [
        ("Flange outer diameter", 45.0, "falling", 89.8, 90.2),
        ("Recess diameter", 25.0, "rising", 49.9, 50.1),
        ("Bore diameter", 12.0, "rising", None, None),
    ]:
        s = MeasurementSpec(name=name, center=(float(c[0]), float(c[1])),
                            r_inner=r_mm * ppm - 10, r_outer=r_mm * ppm + 10,
                            min_mm=lo, max_mm=hi)
        s.roi = (int(c[0] - 50 * ppm), int(c[1] - 50 * ppm), int(100 * ppm), int(100 * ppm))
        s.params.polarity = polarity
        s.locked_radius, s.locked_space = r_mm * ppm, "view"
        specs.append(s)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "samples"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truth = {}
    samples = make_samples()
    for name, (image, scene, mapping) in samples.items():
        imwrite(out / f"{name}.png", image)
        truth[name] = {
            "file": f"{name}.png",
            "diameters_mm": scene.truth_mm,
            "fiducials_mm": scene.fiducials_mm,
            "fiducials_px": fiducials_px(scene, mapping).round(3).tolist(),
            "plane_to_image_homography": mapping.tolist(),
        }
        print("wrote", out / f"{name}.png")
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    # Example calibration + profile for flange.png
    image, scene, mapping = samples["flange"]
    h, w = image.shape[:2]
    f = fiducials_px(scene, mapping)
    cal = CalibrationSet(camera_id=CAMERA_ID, resolution=(w, h),
                         notes="Synthetic example: ruler crosses 100 mm apart")
    cal.planes["default"] = make_scale_plane("default", f[0], f[1], 100.0, None)
    save_calibration(out / "flange_calibration.json", cal)
    profile = PartProfile(name="Example flange", description="Synthetic flange sample",
                          specs=flange_specs(mapping), calibration_id=cal.calibration_id,
                          camera_id=CAMERA_ID)
    save_profile(out / "flange_profile.json", profile)
    print("wrote example profile and calibration")

    # Validation set: repeated captures with different noise and small shifts.
    vdir = out / "validation"
    vdir.mkdir(exist_ok=True)
    rows = []
    for i in range(5):
        shift = (0.4 * i - 0.8, 0.3 * i - 0.6)
        m = mapping.copy()
        m[0, 2] += shift[0]
        m[1, 2] += shift[1]
        img = render(flange_scene(), (w, h), m, seed=100 + i)
        fname = f"flange_{i + 1:02d}.png"
        imwrite(vdir / fname, img)
        for meas, ref in [("Flange outer diameter", 90.0), ("Recess diameter", 50.0),
                          ("Bore diameter", 24.0)]:
            rows.append({"image": fname, "part_id": "flange-A", "measurement": meas,
                         "reference_mm": ref})
    with open(vdir / "references.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "part_id", "measurement",
                                                "reference_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote validation set in", vdir)


if __name__ == "__main__":
    main()
