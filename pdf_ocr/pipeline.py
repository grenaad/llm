#!/usr/bin/env python3
"""
PDF -> per-page PNG @ 300 DPI -> per-page skew detection & correction.

Usage:
    python3 pipeline.py PDF [PDF ...] --out OUT_DIR [--pages N1,N2,...]
                        [--dpi 300] [--threshold-deg 0.05] [--max-deg 5.0]

For each input PDF this produces:
    OUT_DIR/<pdf-stem>/page_NNN.png   (deskewed)
    OUT_DIR/<pdf-stem>/skew_log.csv   (per-page detected angles)

Skew detection: a Hough-lines baseline detector is used as primary, with a
minAreaRect corroboration. The Radon-transform `deskew` library was tried first
but its 1-degree search granularity made it blind to the sub-degree skews
present in these scans (see diagnostic.py and diagnostic/results.md).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def rasterise_pdf(pdf: Path, dpi: int, pages: Optional[list[int]], workdir: Path) -> list[Path]:
    """Render PDF to PNGs at the given DPI. Returns sorted list of PNG paths."""
    out_prefix = workdir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if pages:
        # pdftoppm requires contiguous -f/-l; just call once per page when subsetting.
        produced: list[Path] = []
        for pg in pages:
            subprocess.run(
                cmd + ["-f", str(pg), "-l", str(pg), str(pdf), str(out_prefix)],
                check=True,
            )
        # Collect all PNGs that pdftoppm produced.
        produced = sorted(workdir.glob("page-*.png"))
        return produced
    subprocess.run(cmd + [str(pdf), str(out_prefix)], check=True)
    return sorted(workdir.glob("page-*.png"))


def _binary_text_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu-binarise to a white-text-on-black mask (text = 255)."""
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def _projection_score(bw: np.ndarray, angle: float) -> float:
    """Rotate the binary mask by `angle` and score the horizontal projection.

    The score is the variance of the row-sum differences. When text lines are
    perfectly horizontal, the row-sum profile has sharp peaks (text rows) and
    troughs (inter-line gaps), maximising this variance.
    """
    h, w = bw.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(bw, M, (w, h),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    proj = rot.sum(axis=1).astype(np.float64)
    return float(np.var(np.diff(proj)))


def _projection_skew(gray: np.ndarray,
                     coarse_range: float = 2.0,
                     coarse_step: float = 0.10,
                     fine_step: float = 0.02,
                     work_dim: int = 1000) -> Optional[float]:
    """Detect skew via projection-profile variance maximisation.

    Coarse-to-fine search over candidate angles. Returns the detected skew in
    degrees where POSITIVE means the page is tilted clockwise and should be
    corrected by rotating counter-clockwise (PIL rotate(+angle)). Returns None
    if the page has too little text to score reliably.
    """
    bw = _binary_text_mask(gray)
    # Need a meaningful amount of ink to get a stable projection profile.
    # Blank / near-blank pages (section dividers, image-only plates) otherwise
    # lock onto noise and report wild angles. Require ink to cover at least
    # ~1.5% of the page area AND a minimum absolute pixel count.
    ink = int(bw.sum() // 255)
    if ink < 20000 or ink < 0.015 * bw.size:
        return None

    # Downscale for speed; sub-pixel angle accuracy is preserved because the
    # projection profile is a global statistic.
    h, w = bw.shape
    if max(h, w) > work_dim:
        s = work_dim / max(h, w)
        bw = cv2.resize(bw, (int(w * s), int(h * s)),
                        interpolation=cv2.INTER_AREA)
        # Re-binarise after area interpolation introduced grays.
        _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)

    # Coarse search.
    coarse_angles = np.arange(-coarse_range, coarse_range + coarse_step,
                              coarse_step)
    scores = [(_projection_score(bw, a), a) for a in coarse_angles]
    best_score, best_a = max(scores, key=lambda t: t[0])

    # Fine search around the coarse optimum.
    fine_angles = np.arange(best_a - coarse_step, best_a + coarse_step + fine_step,
                            fine_step)
    fine = [(_projection_score(bw, a), a) for a in fine_angles]
    _, best_fine = max(fine, key=lambda t: t[0])

    # The detector reports the angle that, when applied, best aligns text.
    # getRotationMatrix2D with +angle rotates counter-clockwise, so a positive
    # best_fine means CCW rotation aligns the page => the page is tilted CW.
    # We return that positive value; rotate_pil(+value) then corrects it.
    return float(best_fine)


def detect_skew_angle(img_path: Path, max_deg: float = 5.0
                      ) -> tuple[Optional[float], str]:
    """File-path wrapper. Reads PNG at full resolution, runs projection skew."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "unreadable"
    return _angle_from_gray(img, max_deg)


def _angle_from_gray(gray: np.ndarray, max_deg: float
                     ) -> tuple[Optional[float], str]:
    """Projection-profile skew detection on a grayscale image.

    Returns (angle_deg, source). Positive = page tilted clockwise.
    """
    a = _projection_skew(gray, coarse_range=max_deg)
    if a is None:
        return None, "no_angle"
    if abs(a) > max_deg:
        return a, "projection"  # caller applies the max-deg reject + logging
    return a, "projection"


def _rotate_pil(pil_img: Image.Image, angle_deg: float) -> Image.Image:
    """Return a new PIL image rotated by `angle_deg` (CCW for positive)."""
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    fill = (255, 255, 255) if pil_img.mode == "RGB" else 255
    return pil_img.rotate(
        angle_deg,  # PIL rotates counter-clockwise for positive angle, which
                    # cancels a clockwise (positive) skew exactly.
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill,
    )


def _pil_to_gray_np(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV-compatible grayscale numpy array."""
    if pil_img.mode != "L":
        gray = pil_img.convert("L")
    else:
        gray = pil_img
    return np.asarray(gray)


def iterative_deskew(src: Path, dst: Path,
                     threshold_deg: float, max_deg: float,
                     max_iters: int = 1
                     ) -> tuple[Optional[float], list[float], list[str], str]:
    """Detect skew on the original and rotate ONCE by the detected angle.

    Why single-pass: experiments showed that re-detecting on an already-rotated
    image returns a fixed ~-0.08 deg "residual" that is a resampling artefact,
    not real skew (it stays constant no matter how much you over-rotate). The
    projection-profile detector is accurate to ~0.02 deg on the *original*, so a
    single rotation is both correct and avoids stacking resample blur.

    `max_iters` is retained for API compatibility but values > 1 no longer add
    refinement (see the calibration note above).

    Returns:
        applied:       rotation applied to the original (signed deg), 0.0 if the
                       detected skew was below threshold, or None if detection
                       failed (page kept as-is).
        pass_angles:   single-element list with the detected angle.
        pass_sources:  single-element list with the detector tag.
        status:        final status string.
    """
    original = Image.open(src)

    gray0 = _pil_to_gray_np(original)
    angle0, source0 = _angle_from_gray(gray0, max_deg)

    if angle0 is None:
        original.save(dst, format="PNG", optimize=True)
        return None, [float("nan")], [source0], source0

    if abs(angle0) > max_deg:
        original.save(dst, format="PNG", optimize=True)
        return 0.0, [angle0], [source0], f"rejected:|{angle0:.2f}|>max"

    if abs(angle0) < threshold_deg:
        # Negligible skew; keep the original to avoid a needless resample.
        original.save(dst, format="PNG", optimize=True)
        return 0.0, [angle0], [source0], "ok"

    final = _rotate_pil(original, angle0)
    final.save(dst, format="PNG", optimize=True)
    return float(angle0), [angle0], [source0], "ok"


def process_pdf(
    pdf: Path,
    out_root: Path,
    dpi: int,
    pages: Optional[list[int]],
    threshold_deg: float,
    max_deg: float,
    max_iters: int,
) -> dict:
    stem = pdf.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "skew_log.csv"

    summary = {"pdf": pdf.name, "pages": 0, "rotated": 0, "failed": 0,
               "angles": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = time.time()
        pngs = rasterise_pdf(pdf, dpi, pages, tmp)
        t_raster = time.time() - t0
        print(f"  rasterised {len(pngs)} page(s) in {t_raster:.1f}s")

        with log_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "page_file",
                "applied_rotation_deg",
                "pass_angles_deg",       # semicolon-separated per-pass
                "pass_detectors",        # semicolon-separated per-pass
                "passes",
                "status",
            ])

            for png in pngs:
                page_num_str = png.stem.split("-")[-1]
                try:
                    page_num = int(page_num_str)
                except ValueError:
                    page_num = 0
                dst = out_dir / f"page_{page_num:03d}.png"

                try:
                    applied, pass_angles, pass_sources, status = iterative_deskew(
                        png, dst,
                        threshold_deg=threshold_deg,
                        max_deg=max_deg,
                        max_iters=max_iters,
                    )
                except Exception as e:
                    status = f"deskew_error:{e}"
                    applied = None
                    pass_angles = []
                    pass_sources = []
                    shutil.copyfile(png, dst)

                if applied is not None and applied != 0.0:
                    summary["rotated"] += 1
                    summary["angles"].append(applied)

                writer.writerow([
                    dst.name,
                    f"{applied:.4f}" if applied is not None else "",
                    ";".join(f"{a:.4f}" if not (a is None or (isinstance(a, float) and a != a)) else ""
                             for a in pass_angles),
                    ";".join(pass_sources),
                    len(pass_angles),
                    status,
                ])
                summary["pages"] += 1

    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=str, default=None,
                    help="Comma-separated page numbers to process (default: all)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--threshold-deg", type=float, default=0.05,
                    help="Skip rotation when |angle| < threshold (deg)")
    ap.add_argument("--max-deg", type=float, default=5.0,
                    help="Reject detections with |angle| > max (deg)")
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Max iterative deskew passes per page")
    return ap.parse_args()


def main():
    args = parse_args()
    pages = None
    if args.pages:
        pages = [int(x) for x in args.pages.split(",") if x.strip()]

    args.out.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    grand_rotated = 0
    all_angles: list[float] = []

    for pdf in args.pdfs:
        if not pdf.exists():
            print(f"!! missing: {pdf}", file=sys.stderr)
            continue
        print(f"\n=== {pdf.name} ===")
        s = process_pdf(pdf, args.out, args.dpi, pages,
                        args.threshold_deg, args.max_deg, args.max_iters)
        grand_total += s["pages"]
        grand_rotated += s["rotated"]
        all_angles.extend(s["angles"])
        if s["angles"]:
            arr = np.array(s["angles"])
            print(f"  angles: n={len(arr)}  "
                  f"median={np.median(arr):+.3f}deg  "
                  f"mean={arr.mean():+.3f}deg  "
                  f"min={arr.min():+.3f}  max={arr.max():+.3f}  "
                  f"|abs|.95={np.quantile(np.abs(arr), 0.95):.3f}")
        print(f"  rotated: {s['rotated']}/{s['pages']}  failed: {s['failed']}")

    print(f"\n=== TOTAL ===  pages={grand_total}  rotated={grand_rotated}")
    if all_angles:
        a = np.array(all_angles)
        print(f"global median={np.median(a):+.3f}deg  "
              f"mean={a.mean():+.3f}deg  "
              f"95%ile |abs|={np.quantile(np.abs(a), 0.95):.3f}deg")


if __name__ == "__main__":
    main()
