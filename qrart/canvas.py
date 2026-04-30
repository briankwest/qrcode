"""Composition canvases — generate the QR as a feature in a larger scene by:

  1. Generating the scene at canvas dims via txt2img (no ControlNet).
  2. Generating the QR art separately at QR-region dims using the standalone
     ControlNet path (full-canvas QR pattern as control image — matches
     QR Monster's training distribution).
  3. Compositing the QR art into the scene with a feathered alpha mask.

The earlier two-stage inpaint approach failed because QR Monster ControlNet
was trained on full-canvas QR patterns, not partial ones — feeding it a
scene-sized canvas with QR-in-corner produced a weak conditioning signal
that the prompt + scene-init overrode, so the masked region rendered as
"more scene" instead of QR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFilter

from .qr import make_qr


class Composition(TypedDict):
    canvas_size: tuple[int, int]  # (width, height)
    qr_size: int                  # square QR side length, px
    qr_pos: tuple[int, int]       # top-left corner inside canvas
    scaffold: str                 # appended to user's prompt to bias both scene + QR-art


# Canvas dims sized for SD 1.5 finetunes. Going much past 1024 in any axis on
# MPS is slow and degrades quality.
COMPOSITIONS: dict[str, Composition] = {
    "standalone": {
        "canvas_size": (768, 768),
        "qr_size": 768,
        "qr_pos": (0, 0),
        "scaffold": "",
    },
    "subject-portrait": {
        # Subject above, QR-as-feature below. Astronaut on a patterned platform.
        "canvas_size": (768, 1024),
        "qr_size": 640,
        "qr_pos": (64, 320),
        "scaffold": ", with an intricate decorative patterned ornament featured below, cinematic photorealistic composition",
    },
    "scene-landscape": {
        # Wide scene with a QR-as-stone-monolith feature on the right.
        "canvas_size": (1024, 768),
        "qr_size": 600,
        "qr_pos": (400, 84),
        "scaffold": ", featuring an ancient stone monolith covered in intricate carved geometric patterns on the right, dramatic landscape composition, photorealistic",
    },
    "garment": {
        # Fashion shot — the QR is the patterned outfit centerpiece.
        "canvas_size": (768, 1024),
        "qr_size": 600,
        "qr_pos": (84, 320),
        "scaffold": ", wearing an intricate richly patterned ornate ceremonial garment with bold geometric design, fashion photograph",
    },
}


@dataclass
class CompositionInfo:
    canvas_w: int
    canvas_h: int
    qr_size: int
    qr_pos: tuple[int, int]
    qr_image: Image.Image  # QR pattern at qr_size × qr_size — used as ControlNet input
    scaffold: str


def build_composition(data: str, name: str) -> CompositionInfo:
    cfg = COMPOSITIONS.get(name, COMPOSITIONS["standalone"])
    cw, ch = cfg["canvas_size"]
    return CompositionInfo(
        canvas_w=cw,
        canvas_h=ch,
        qr_size=cfg["qr_size"],
        qr_pos=cfg["qr_pos"],
        qr_image=make_qr(data, size=cfg["qr_size"]),
        scaffold=cfg["scaffold"],
    )


def is_standalone(name: str) -> bool:
    return name == "standalone"


def composite_qr_into_scene(
    scene: Image.Image,
    qr_art: Image.Image,
    composition: str,
    feather_px: int = 4,
) -> Image.Image:
    """Paste qr_art into scene at the composition's QR position with a
    feathered alpha edge.

    The interior of the QR art is fully opaque (preserves the data modules
    exactly), only the outer feather_px ring fades into the scene to avoid a
    hard rectangular seam. feather_px must be small (4-6) — anything larger
    eats into the QR's quiet zone and may break decoding.
    """
    cfg = COMPOSITIONS.get(composition, COMPOSITIONS["standalone"])
    qx, qy = cfg["qr_pos"]
    qsz = cfg["qr_size"]

    # Inset-then-blur produces a feathered ring on the outer edge while the
    # interior stays at ~255. A flat all-white mask wouldn't feather (no
    # gradient for the blur kernel to act on).
    alpha = Image.new("L", (qsz, qsz), 0)
    if feather_px > 0:
        ImageDraw.Draw(alpha).rectangle(
            (feather_px, feather_px, qsz - feather_px - 1, qsz - feather_px - 1),
            fill=255,
        )
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_px))
    else:
        ImageDraw.Draw(alpha).rectangle((0, 0, qsz - 1, qsz - 1), fill=255)

    out = scene.copy().convert("RGB")
    qr_resized = qr_art.resize((qsz, qsz)).convert("RGB")
    out.paste(qr_resized, (qx, qy), alpha)
    return out
