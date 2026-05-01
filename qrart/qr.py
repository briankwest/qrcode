from __future__ import annotations

import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image


def _build(data: str, border: int = 1) -> qrcode.QRCode:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=10,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    return qr


def make_qr(data: str, size: int = 768, border: int = 1) -> Image.Image:
    """Render a high-error-correction QR as a square PIL image at exactly `size` px.

    Level H gives ~30% error correction headroom — that's the slack the diffusion
    model uses to "lie" about pixels while keeping the code scannable.
    """
    qr = _build(data, border=border)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size), Image.NEAREST)


def qr_modules(data: str) -> np.ndarray:
    """Return the QR module grid as a (n, n) bool array (True = dark).

    No border included — the caller is responsible for accounting for it.
    Used by scannability scoring and finder-pattern reinforcement.
    """
    qr = _build(data, border=0)
    return np.array(qr.modules, dtype=bool)
