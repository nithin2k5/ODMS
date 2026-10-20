"""Validate measurements against parts with independently measured diameters.

    python -m odms.validation --profile part.json --calibration cal.json \\
        --images captures/ --references references.csv --setup-unchanged \\
        [--out report]

``references.csv`` has the columns ``image, part_id, measurement,
reference_mm`` (one row per image and measurement).  Capture several images
of each reference part, re-placing the part between captures, so that the
spread of the results reflects real repeatability.

The report gives, per measurement and part: mean error (bias), standard
deviation of repeated readings (repeatability), maximum absolute error, RMSE
and the number of captures that could not be measured.  These figures apply
only to the camera, optics, lighting and parts used for the validation.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .calibration.store import load_calibration
from .measurement.engine import EngineConfig, MeasurementEngine
from .storage.profiles import load_profile
from .storage.results import imread


@dataclass
class Sample:
    image: str
    part_id: str
    measurement: str
    reference_mm: float
    measured_mm: float | None
    status: str
    reason: str
    quality: float | None

    @property
    def error_mm(self) -> float | None:
        return None if self.measured_mm is None else self.measured_mm - self.reference_mm


@dataclass
class Summary:
    measurement: str
    part_id: str
    reference_mm: float
    n: int
    n_measured: int
    mean_mm: float | None
    bias_mm: float | None
    repeatability_sd_mm: float | None
    max_abs_error_mm: float | None
    rmse_mm: float | None


def summarize(samples: list[Sample]) -> list[Summary]:
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for s in samples:
        groups[(s.measurement, s.part_id)].append(s)
    out = []
    for (meas, part), items in sorted(groups.items()):
        vals = [s.measured_mm for s in items if s.measured_mm is not None]
        errs = [s.error_mm for s in items if s.error_mm is not None]
        ref = sum(s.reference_mm for s in items) / len(items)
        n = len(vals)
        mean = sum(vals) / n if n else None
        sd = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None)
        out.append(Summary(
            measurement=meas, part_id=part, reference_mm=ref, n=len(items), n_measured=n,
            mean_mm=mean, bias_mm=(sum(errs) / n if n else None), repeatability_sd_mm=sd,
            max_abs_error_mm=(max(abs(e) for e in errs) if errs else None),
            rmse_mm=(math.sqrt(sum(e * e for e in errs) / n) if n else None)))
    return out


def run_validation(profile_path: Path, calibration_path: Path, images_dir: Path,
                   references_path: Path, setup_unchanged: bool,
                   camera_id: str | None = None) -> list[Sample]:
    profile = load_profile(profile_path)
    calibration = load_calibration(calibration_path, require_revalidation=not setup_unchanged)
    cam = camera_id or calibration.camera_id
    with open(references_path, newline="", encoding="utf-8-sig") as f:
        refs = list(csv.DictReader(f))
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in refs:
        by_image[row["image"]].append(row)

    samples: list[Sample] = []
    for image_name, rows in by_image.items():
        image = imread(images_dir / image_name)
        results = {}
        if image is not None:
            engine = MeasurementEngine()  # no smoothing across separate captures
            frame = engine.process(image, EngineConfig(specs=profile.specs,
                                                       calibration=calibration, camera_id=cam))
            results = {r.name: r for r in frame.results}
        for row in rows:
            r = results.get(row["measurement"])
            if image is None:
                status, reason, mm, q = "error", "image could not be read", None, None
            elif r is None:
                status, reason, mm, q = "error", "measurement not in profile", None, None
            else:
                status, reason = r.status.value, r.reason
                mm = r.diameter_mm
                q = r.quality.score if r.detected else None
            samples.append(Sample(image_name, row.get("part_id", "") or "", row["measurement"],
                                  float(row["reference_mm"]), mm, status, reason, q))
    return samples


def _f(v, d=4):
    return "" if v is None else f"{v:.{d}f}"


def write_report(samples: list[Sample], summaries: list[Summary], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}_samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "part_id", "measurement", "reference_mm", "measured_mm",
                    "error_mm", "status", "reason", "detection_quality"])
        for s in samples:
            w.writerow([s.image, s.part_id, s.measurement, _f(s.reference_mm), _f(s.measured_mm),
                        _f(s.error_mm), s.status, s.reason, _f(s.quality, 1)])
    with open(f"{out_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["measurement", "part_id", "reference_mm", "captures", "measured",
                    "mean_mm", "bias_mm", "repeatability_sd_mm", "max_abs_error_mm", "rmse_mm"])
        for s in summaries:
            w.writerow([s.measurement, s.part_id, _f(s.reference_mm), s.n, s.n_measured,
                        _f(s.mean_mm), _f(s.bias_mm), _f(s.repeatability_sd_mm),
                        _f(s.max_abs_error_mm), _f(s.rmse_mm)])


def print_summary(summaries: list[Summary]) -> None:
    header = (f"{'Measurement':28} {'Part':10} {'Ref mm':>9} {'n':>5} {'Bias':>9} "
              f"{'SD':>8} {'Max|e|':>8} {'RMSE':>8}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s.measurement[:28]:28} {s.part_id[:10]:10} {s.reference_mm:9.3f} "
              f"{s.n_measured:>2}/{s.n:<2} {_f(s.bias_mm):>9} {_f(s.repeatability_sd_mm):>8} "
              f"{_f(s.max_abs_error_mm):>8} {_f(s.rmse_mm):>8}")
    print("\nBias = mean(measured - reference); SD = sample standard deviation of repeated "
          "readings.\nThese results describe this setup only; they are not a general "
          "accuracy specification.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate ODMS against reference parts.")
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--references", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("validation_report"))
    ap.add_argument("--camera-id", default=None,
                    help="camera identity the images were captured with "
                         "(default: the calibration's camera)")
    ap.add_argument("--setup-unchanged", action="store_true",
                    help="confirm the images were captured with the calibrated, unchanged "
                         "setup; without it a loaded calibration yields no millimetres")
    args = ap.parse_args(argv)

    samples = run_validation(args.profile, args.calibration, args.images, args.references,
                             args.setup_unchanged, args.camera_id)
    if not args.setup_unchanged:
        print("Note: --setup-unchanged not given, so the calibration is treated as requiring "
              "revalidation and no millimetre values are produced.", file=sys.stderr)
    summaries = summarize(samples)
    write_report(samples, summaries, args.out)
    print_summary(summaries)
    print(f"\nWrote {args.out}_samples.csv and {args.out}_summary.csv")
    failed = sum(1 for s in samples if s.measured_mm is None)
    if failed:
        print(f"{failed} of {len(samples)} readings could not be measured; see the samples "
              "file for reasons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
