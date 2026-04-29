from __future__ import annotations

import io
import cv2
import numpy as np
from PIL import Image


def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _try_decode(detector: cv2.QRCodeDetector, arr: np.ndarray) -> str | None:
    try:
        data, _, _ = detector.detectAndDecode(arr)
    except cv2.error:
        return None
    return data or None


def scan(img: Image.Image) -> str | None:
    """Try to decode a QR from the image. Returns payload string or None.

    AI QR art often only decodes after some preprocessing — try several
    rescales, contrast boosts, and binarization variants before giving up.
    """
    detector = cv2.QRCodeDetector()
    base = _to_cv(img)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    variants: list[np.ndarray] = []
    # raw at multiple scales
    for s in (1.0, 0.75, 1.25, 0.5, 1.5, 2.0):
        h, w = base.shape[:2]
        variants.append(cv2.resize(base, (max(1, int(w * s)), max(1, int(h * s)))))

    # grayscale + contrast
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    eq = cv2.equalizeHist(gray)
    variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

    # adaptive threshold (good for low-contrast hidden QRs)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    # Otsu + Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    for v in variants:
        result = _try_decode(detector, v)
        if result:
            return result
    return None
