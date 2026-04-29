from __future__ import annotations

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image


def make_qr(data: str, size: int = 768, border: int = 1) -> Image.Image:
    """Render a high-error-correction QR as a square PIL image at exactly `size` px.

    Level H gives ~30% error correction headroom — that's the slack the diffusion
    model uses to "lie" about pixels while keeping the code scannable.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=10,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size), Image.NEAREST)
