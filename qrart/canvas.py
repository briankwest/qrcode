"""Composition canvases — generate the QR as a feature in a larger scene by:

  1. Generating the scene at canvas dims via txt2img (no ControlNet).
  2. Generating the QR art separately at QR-region dims using the standalone
     ControlNet path (full-canvas QR pattern as control image — matches
     QR Monster's training distribution).
  3. Compositing the QR art into the scene with a finder-aware alpha mask
     (corners hard, BR + interior edges feathered) plus an alpha-blended
     reinforcement of the three ground-truth finder patterns to lock in
     scannability.

The earlier two-stage inpaint approach failed because QR Monster ControlNet
was trained on full-canvas QR patterns, not partial ones — feeding it a
scene-sized canvas with QR-in-corner produced a weak conditioning signal
that the prompt + scene-init overrode, so the masked region rendered as
"more scene" instead of QR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .qr import make_qr, qr_modules


class Composition(TypedDict):
    canvas_size: tuple[int, int]  # (width, height)
    qr_size: int                  # square QR side length, px
    qr_pos: tuple[int, int]       # top-left corner inside canvas
    scaffold: str                 # appended to user's prompt to bias both scene + QR-art


# Canvas dims sized for SD 1.5 finetunes. Going much past 1024 in any axis on
# MPS is slow and degrades quality. qr_size bumped from 600 → 720 in
# scene-landscape and garment so module-pixel density doesn't fall below
# scanner thresholds.
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
        "qr_size": 720,
        "qr_pos": (288, 24),
        "scaffold": ", featuring an ancient stone monolith covered in intricate carved geometric patterns on the right, dramatic landscape composition, photorealistic",
    },
    "garment": {
        # Fashion shot — the QR is the patterned outfit centerpiece.
        "canvas_size": (768, 1024),
        "qr_size": 720,
        "qr_pos": (24, 280),
        "scaffold": ", wearing an intricate richly patterned ornate ceremonial garment with bold geometric design, fashion photograph",
    },
}


# B3 reinforcement: blend strength of the ground-truth finder pattern overlay.
# 0.85 keeps the texture readable underneath while giving scanners a pristine
# corner. Tuned empirically — anything < 0.7 doesn't reliably push borderline
# candidates over the threshold, anything > 0.92 starts looking pasted-on.
FINDER_BLEND_ALPHA = 0.85


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


def _finder_aware_mask(qsz: int, feather_px: int) -> Image.Image:
    """Alpha mask sized (qsz, qsz). The three finder-pattern corners
    (TL/TR/BL) are kept fully opaque with NO feather — those corners are
    the most fragile QR feature; even a 4-px blur can break detection.
    The bottom-right and the interior edges get the feather.

    Implementation: start with a fully-opaque rectangle, blur it (this
    feathers all four edges equally), then paint hard-opaque squares back
    over the three finder corners.
    """
    if feather_px <= 0:
        return Image.new("L", (qsz, qsz), 255)

    # Start with an inset rectangle so the blur produces a gradient.
    alpha = Image.new("L", (qsz, qsz), 0)
    ImageDraw.Draw(alpha).rectangle(
        (feather_px, feather_px, qsz - feather_px - 1, qsz - feather_px - 1),
        fill=255,
    )
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Re-impose hard opacity over the three finder-pattern regions. The
    # finder is 7 modules; a QR with default border=1 places it 1 module
    # in. A typical QR is 25-49 modules, so each module is qsz / (n+2)
    # pixels. Conservatively reserve 9 modules' worth around each corner
    # (7 finder + 2 buffer) so we cover the separator too.
    # We don't know n at this point but ~50 px is safe for qsz >= 600.
    margin = max(50, qsz // 8)
    draw = ImageDraw.Draw(alpha)
    # Top-left
    draw.rectangle((0, 0, margin - 1, margin - 1), fill=255)
    # Top-right
    draw.rectangle((qsz - margin, 0, qsz - 1, margin - 1), fill=255)
    # Bottom-left
    draw.rectangle((0, qsz - margin, margin - 1, qsz - 1), fill=255)
    return alpha


def _reinforce_finders(
    image: Image.Image,
    data: str,
    qr_pos: tuple[int, int],
    qr_size: int,
    alpha: float = FINDER_BLEND_ALPHA,
) -> Image.Image:
    """Alpha-blend the ground-truth finder patterns over the three corners.

    The QR finder is a 7×7-module square. We render those three regions
    from the ground-truth grid at the same scale as the QR art and blend
    them in at `alpha` opacity. This is mostly imperceptible at viewing
    distance but gives camera scanners a pristine 1:1:3:1:1 ratio to lock
    onto — the single most reliable feature for QR detection.
    """
    modules = qr_modules(data)
    n = modules.shape[0]
    border = 1
    total = n + 2 * border
    px_per_module = qr_size / total
    finder_modules = 7

    # Module-grid positions of the three finders (top-left module, in the
    # n×n grid without border). Ordered TL, TR, BL.
    finder_corners = [(0, 0), (0, n - finder_modules), (n - finder_modules, 0)]

    pixels_per_finder = int(round(finder_modules * px_per_module))
    qx, qy = qr_pos

    out = image.copy().convert("RGB")
    out_arr = np.array(out, dtype=np.float32)

    for fi, fj in finder_corners:
        # Pixel position of the finder's top-left within the QR region.
        py = int(round((border + fi) * px_per_module)) + qy
        px = int(round((border + fj) * px_per_module)) + qx

        # Render the 7×7 finder pattern at the right pixel size.
        finder_grid = modules[fi:fi + finder_modules, fj:fj + finder_modules]
        finder_img = Image.fromarray(
            np.where(finder_grid, 0, 255).astype(np.uint8)
        ).resize((pixels_per_finder, pixels_per_finder), Image.NEAREST)
        finder_arr = np.array(finder_img.convert("RGB"), dtype=np.float32)

        # Clip to image bounds (in case a corner is right at the edge).
        h, w = out_arr.shape[:2]
        y1 = min(h, py + pixels_per_finder)
        x1 = min(w, px + pixels_per_finder)
        fy = y1 - py
        fx = x1 - px
        if fy <= 0 or fx <= 0:
            continue

        # Blend: out = (1-alpha)*out + alpha*finder
        target = out_arr[py:y1, px:x1]
        finder_slice = finder_arr[:fy, :fx]
        out_arr[py:y1, px:x1] = (1 - alpha) * target + alpha * finder_slice

    return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))


def composite_qr_into_scene(
    scene: Image.Image,
    qr_art: Image.Image,
    composition: str,
    feather_px: int = 4,
    data: str | None = None,
    reinforce_finders: bool = True,
) -> Image.Image:
    """Paste qr_art into scene at the composition's QR position with a
    finder-aware alpha mask, then optionally reinforce the three corner
    finder patterns by alpha-blending the ground-truth pattern over them.

    feather_px is the soft-edge radius applied to BR + interior edges only.
    The TL/TR/BL finder-pattern corners are kept hard (no feathering) — they
    are the most fragile QR feature and even a 4-px blur can break detection.

    `data` is the QR payload — required when `reinforce_finders=True` so we
    can render the ground-truth finder pattern.
    """
    cfg = COMPOSITIONS.get(composition, COMPOSITIONS["standalone"])
    qx, qy = cfg["qr_pos"]
    qsz = cfg["qr_size"]

    out = scene.copy().convert("RGB")
    qr_resized = qr_art.resize((qsz, qsz)).convert("RGB")
    out.paste(qr_resized, (qx, qy), _finder_aware_mask(qsz, feather_px))

    if reinforce_finders and data is not None:
        out = _reinforce_finders(out, data, (qx, qy), qsz)

    return out
