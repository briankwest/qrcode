"""Multi-scanner ensemble for stylized QR codes.

Three decoders, ordered by speed-then-capability:
  1. cv2.QRCodeDetector — fastest, handles plain QRs, struggles with stylized.
  2. zxing-cpp — Google's reference QR/barcode lib (the same family iOS uses
     under the hood). Catches a meaningful fraction of stylized QRs that cv2
     misses without the cost of YOLO inference.
  3. qreader — YOLO-based detector + libzbar for decoding. Slowest but most
     forgiving on heavily-stylized AI outputs.

Each decoder is also tried over a small set of preprocessing variants
(equalizeHist, adaptive threshold, scale up/down) since the "almost-scanned"
candidates often need just a small tonal shift to decode.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


_qreader = None


def _get_qreader():
    global _qreader
    if _qreader is None:
        from qreader import QReader
        _qreader = QReader(model_size="m")
    return _qreader


def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _variants(img: Image.Image) -> list[np.ndarray]:
    """Generate preprocessing variants for the cv2/zxing decoders. The
    YOLO-based qreader does its own preprocessing so we feed it only the
    original image."""
    base = _to_cv(img)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    h, w = base.shape[:2]
    out: list[np.ndarray] = []

    # Scale variants — different scanners prefer different module densities.
    for s in (1.0, 0.75, 1.25, 0.5, 1.5, 2.0):
        out.append(cv2.resize(base, (max(1, int(w * s)), max(1, int(h * s)))))

    # Tonal variants.
    out.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    out.append(cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR))
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    out.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    return out


def _try_cv2(detector: cv2.QRCodeDetector, arr: np.ndarray) -> str | None:
    try:
        data, _, _ = detector.detectAndDecode(arr)
    except cv2.error:
        return None
    return data or None


def _scan_cv2(variants: list[np.ndarray]) -> str | None:
    detector = cv2.QRCodeDetector()
    for v in variants:
        result = _try_cv2(detector, v)
        if result:
            return result
    return None


def _scan_zxing(variants: list[np.ndarray]) -> str | None:
    try:
        import zxingcpp
    except ImportError:
        return None
    for v in variants:
        try:
            results = zxingcpp.read_barcodes(v)
        except Exception:
            continue
        for r in results:
            if r.text and r.format == zxingcpp.BarcodeFormat.QRCode:
                return r.text
    return None


def _scan_qreader(img: Image.Image) -> str | None:
    try:
        qr = _get_qreader()
        arr = np.array(img.convert("RGB"))
        results = qr.detect_and_decode(image=arr)
        for r in results:
            if r:
                return r
    except Exception:
        return None
    return None


def scan(img: Image.Image) -> str | None:
    """Decode a QR. Try cv2 → zxing → qreader. First success wins."""
    variants = _variants(img)
    return (
        _scan_cv2(variants)
        or _scan_zxing(variants)
        or _scan_qreader(img)
    )
