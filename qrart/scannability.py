"""Per-candidate scannability score (0.0-1.0).

Compares the candidate's per-module luminance against the ground-truth QR
grid. Used to:
  - Rank candidates beyond binary scan/no-scan (which is hopeless when zero
    candidates pass).
  - Drive auto-escalation (A2) and the score-driven refine pass (C1).

Method:
  1. Crop the candidate to the QR region (compositions only have the QR in a
     sub-rectangle; standalone is the full image).
  2. Vectorized per-module mean luminance via reshape — for each of the n²
     ground-truth modules, take the mean over the center 70% of its pixel
     area (avoids module-edge bleed).
  3. Threshold = mean of the per-module means. This is robust to overall
     scene brightness (a dark photo has a low mean which still splits its
     own dark/light modules correctly), and unlike Otsu doesn't degenerate
     on perfectly-bimodal inputs.
  4. Score = fraction of modules on the correct side.

A candidate that scans typically scores ≥ 0.92. The interesting band is
0.80-0.92 — those are "almost scanned" and respond well to a small refine
nudge (C1) or a slight scale bump (A2).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .qr import qr_modules


def score(
    image: Image.Image,
    data: str,
    qr_pos: tuple[int, int] = (0, 0),
    qr_size: int | None = None,
    border: int = 1,
) -> float:
    """Compute the 0.0-1.0 module-correctness score.

    qr_size=None means the whole image is the QR (standalone). For
    compositions pass qr_pos and qr_size from CompositionInfo.
    """
    if qr_size is not None:
        x, y = qr_pos
        region = image.crop((x, y, x + qr_size, y + qr_size))
    else:
        region = image

    modules = qr_modules(data)  # (n, n) bool, no border
    n = modules.shape[0]
    total_modules = n + 2 * border

    gray = np.array(region.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < total_modules or w < total_modules:
        return 0.0

    px_per_module_y = h / total_modules
    px_per_module_x = w / total_modules

    # For each module, mean over the center 70% × 70% of its pixel area.
    # The 15% margin on each side avoids contamination from neighbor modules.
    means = np.empty((n, n), dtype=np.float32)
    inset_y = px_per_module_y * 0.15
    inset_x = px_per_module_x * 0.15
    for i in range(n):
        y0 = int((border + i) * px_per_module_y + inset_y)
        y1 = int((border + i + 1) * px_per_module_y - inset_y)
        if y1 <= y0:
            y1 = y0 + 1
        for j in range(n):
            x0 = int((border + j) * px_per_module_x + inset_x)
            x1 = int((border + j + 1) * px_per_module_x - inset_x)
            if x1 <= x0:
                x1 = x0 + 1
            means[i, j] = gray[y0:y1, x0:x1].mean()

    # Threshold = mean of per-module means. Robust to overall brightness
    # because we're averaging the module-level signal, not all pixels (which
    # would be biased by module-area variance).
    threshold = float(means.mean())
    observed_dark = means < threshold
    correct = int(np.sum(observed_dark == modules))
    return correct / (n * n)
