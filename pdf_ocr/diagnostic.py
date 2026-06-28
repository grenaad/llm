#!/usr/bin/env python3
"""
Skew diagnostic: render N random interior pages across all input PDFs,
run four independent skew detectors on each, and emit a comparison table.

Outputs (under --out, default ./diagnostic):
    pages/<stem>_pNNN.png          one PNG per sampled page (300 DPI)
    results.csv                    machine-readable angles per method
    results.md                     human-readable Markdown table

Methods used:
    1. deskew_lo   - deskew.determine_skew on a 1500-px-downscaled image
    2. deskew_hi   - deskew.determine_skew on the full-resolution image
    3. hough_rect  - cv2.minAreaRect over horizontally dilated text mass
    4. hough_lines - median angle of near-horizontal Hough lines (sub-degree bins)
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deskew import determine_skew


# ---------- page count / sampling ----------

def pdf_page_count(pdf: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine page count for {pdf}")


def sample_pages(pdfs: list[Path], total: int, seed: int) -> list[tuple[Path, int]]:
    """Choose `total` random interior pages distributed across all PDFs,
    proportional to each PDF's page count. Excludes page 1 (cover) and the
    final page where possible."""
    rng = random.Random(seed)
    counts = {p: pdf_page_count(p) for p in pdfs}
    grand = sum(counts.values())
    quotas: dict[Path, int] = {}
    # Largest-remainder allocation so quotas sum to exactly `total`.
    raw = {p: total * c / grand for p, c in counts.items()}
    base = {p: int(v) for p, v in raw.items()}
    remainder = total - sum(base.values())
    # Distribute remainder by largest fractional part.
    frac_sorted = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for p, _ in frac_sorted[:remainder]:
        base[p] += 1
    quotas = base

    picks: list[tuple[Path, int]] = []
    for p, n in quotas.items():
        c = counts[p]
        candidates = list(range(2, c)) if c >= 3 else list(range(1, c + 1))
        n = min(n, len(candidates))
        chosen = rng.sample(candidates, n)
        for pg in sorted(chosen):
            picks.append((p, pg))
    return picks


# ---------- rendering ----------

def render_page(pdf: Path, page: int, dpi: int, dst: Path) -> None:
    """Render a single PDF page to `dst` (PNG) at the given DPI."""
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "p"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi),
             "-f", str(page), "-l", str(page),
             str(pdf), str(prefix)],
            check=True,
        )
        produced = sorted(Path(td).glob("p-*.png"))
        if not produced:
            raise RuntimeError(f"pdftoppm produced nothing for {pdf}:{page}")
        produced[0].replace(dst)


# ---------- skew detectors ----------

def m_deskew_lo(gray: np.ndarray) -> Optional[float]:
    h, w = gray.shape
    max_dim = 1500
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        gs = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        gs = gray
    try:
        a = determine_skew(gs)
    except Exception:
        return None
    return float(a) if a is not None else None


def m_deskew_hi(gray: np.ndarray) -> Optional[float]:
    try:
        a = determine_skew(gray)
    except Exception:
        return None
    return float(a) if a is not None else None


def _binary_text_mask(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def m_hough_rect(gray: np.ndarray) -> Optional[float]:
    bw = _binary_text_mask(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dil = cv2.dilate(bw, kernel, iterations=1)
    coords = np.column_stack(np.where(dil > 0))
    if len(coords) < 100:
        return None
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # Normalise: cv2's minAreaRect returns angle in [-90, 0).
    if angle < -45:
        angle = 90 + angle
    return float(-angle)


def m_hough_lines(gray: np.ndarray) -> Optional[float]:
    bw = _binary_text_mask(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    bw = cv2.dilate(bw, kernel, iterations=1)
    edges = cv2.Canny(bw, 50, 150)
    # angular resolution ~ 0.05 deg
    lines = cv2.HoughLinesP(edges, 1, np.pi / 3600, threshold=200,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -20 < a < 20:
            angles.append(a)
    if not angles:
        return None
    return float(np.median(angles))


METHODS = [
    ("deskew_lo", m_deskew_lo),
    ("deskew_hi", m_deskew_hi),
    ("hough_rect", m_hough_rect),
    ("hough_lines", m_hough_lines),
]


# ---------- main ----------

def fmt(v: Optional[float]) -> str:
    if v is None:
        return "  none"
    return f"{v:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("diagnostic"))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    picks = sample_pages(args.pdfs, args.samples, args.seed)
    print(f"Sampled {len(picks)} pages across {len(args.pdfs)} PDFs")

    rows = []
    for pdf, pg in picks:
        stem = pdf.stem
        png = pages_dir / f"{stem}_p{pg:03d}.png"
        if not png.exists():
            print(f"  rendering {stem} p.{pg} ...")
            render_page(pdf, pg, args.dpi, png)
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  !! cannot read {png}")
            continue

        result = {name: fn(gray) for name, fn in METHODS}
        rows.append({
            "pdf": stem,
            "page": pg,
            "png": png.name,
            **result,
        })
        line = (f"  {stem}:p{pg:03d}  "
                f"deskew_lo={fmt(result['deskew_lo'])}  "
                f"deskew_hi={fmt(result['deskew_hi'])}  "
                f"hough_rect={fmt(result['hough_rect'])}  "
                f"hough_lines={fmt(result['hough_lines'])}")
        print(line)

    # CSV
    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf", "page", "png",
                    "deskew_lo", "deskew_hi", "hough_rect", "hough_lines"])
        for r in rows:
            w.writerow([
                r["pdf"], r["page"], r["png"],
                "" if r["deskew_lo"] is None else f"{r['deskew_lo']:.4f}",
                "" if r["deskew_hi"] is None else f"{r['deskew_hi']:.4f}",
                "" if r["hough_rect"] is None else f"{r['hough_rect']:.4f}",
                "" if r["hough_lines"] is None else f"{r['hough_lines']:.4f}",
            ])
    print(f"\nwrote {csv_path}")

    # Markdown
    md_path = args.out / "results.md"
    with md_path.open("w") as fh:
        fh.write("# Skew diagnostic\n\n")
        fh.write(f"- Samples: {len(rows)}\n")
        fh.write(f"- DPI: {args.dpi}\n")
        fh.write(f"- Seed: {args.seed}\n\n")
        fh.write("| PDF | Page | deskew_lo | deskew_hi | hough_rect | hough_lines | PNG |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(
                f"| {r['pdf']} | {r['page']} | "
                f"{fmt(r['deskew_lo'])} | {fmt(r['deskew_hi'])} | "
                f"{fmt(r['hough_rect'])} | {fmt(r['hough_lines'])} | "
                f"`pages/{r['png']}` |\n"
            )
        # Per-method summary
        fh.write("\n## Per-method summary\n\n")
        fh.write("| Method | n non-null | median | mean | min | max | 95th %ile |abs| |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, _ in METHODS:
            arr = np.array([r[name] for r in rows if r[name] is not None])
            if len(arr) == 0:
                fh.write(f"| {name} | 0 | - | - | - | - | - |\n")
                continue
            fh.write(
                f"| {name} | {len(arr)} | {np.median(arr):+.3f} | "
                f"{arr.mean():+.3f} | {arr.min():+.3f} | {arr.max():+.3f} | "
                f"{np.quantile(np.abs(arr), 0.95):.3f} |\n"
            )
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
