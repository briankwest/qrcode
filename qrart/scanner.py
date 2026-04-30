from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# qreader uses a YOLO model for QR detection + libzbar/cv2 for decoding. It
# matches iOS Camera-class scan capability for stylized AI-generated QRs that
# cv2's plain QRCodeDetector misses. Lazy-loaded — first use downloads ~55MB.
_qreader = None


def _get_qreader():
    global _qreader
    if _qreader is None:
        from qreader import QReader
        _qreader = QReader(model_size="m")
    return _qreader


def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _try_decode(detector: cv2.QRCodeDetector, arr: np.ndarray) -> str | None:
    try:
        data, _, _ = detector.detectAndDecode(arr)
    except cv2.error:
        return None
    return data or None


def _scan_cv2(img: Image.Image) -> str | None:
    detector = cv2.QRCodeDetector()
    base = _to_cv(img)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    variants: list[np.ndarray] = []
    for s in (1.0, 0.75, 1.25, 0.5, 1.5, 2.0):
        h, w = base.shape[:2]
        variants.append(cv2.resize(base, (max(1, int(w * s)), max(1, int(h * s)))))

    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    eq = cv2.equalizeHist(gray)
    variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    for v in variants:
        result = _try_decode(detector, v)
        if result:
            return result
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
    """Decode a QR. Tries cv2 first (fast, plain QRs), then qreader (YOLO-based,
    handles stylized AI QRs that iOS Camera reads but cv2 doesn't).
    """
    result = _scan_cv2(img)
    if result:
        return result
    return _scan_qreader(img)
