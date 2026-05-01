from __future__ import annotations

# Each preset is (positive suffix, default negative). The user's prompt is
# appended with the suffix; the negative prompt defaults to the listed one but
# can be overridden via GenerationRequest.negative_prompt.
#
# C3: each negative includes "low contrast, washed out, foggy, hazy" — these
# tokens push the model away from soft tonal regions where QR modules get
# mushy and start failing scans. Marginal but consistent improvement to
# scan rate without visible damage to realism.
_QR_CONTRAST_NEG = "low contrast, washed out, foggy, hazy"

STYLE_PRESETS: dict[str, tuple[str, str]] = {
    "photoreal": (
        "RAW photo, 8k uhd, DSLR, cinematic lighting, film grain, "
        "Kodak Portra 400, sharp focus, photographic, hyperdetailed, realistic skin texture",
        f"illustration, painting, drawing, art, sketch, cartoon, anime, render, "
        f"cgi, 3d, plastic, low quality, blurry, deformed, watermark, text, signature, "
        f"{_QR_CONTRAST_NEG}",
    ),
    "cinematic": (
        "cinematic, dramatic lighting, ultra detailed, octane render, masterpiece, 8k, vivid",
        f"low quality, blurry, deformed, watermark, text, ugly, jpeg artifacts, "
        f"{_QR_CONTRAST_NEG}",
    ),
    "illustration": (
        "digital illustration, concept art, trending on artstation, masterpiece, vivid colors",
        f"low quality, blurry, deformed, watermark, photo, photographic, jpeg artifacts, "
        f"{_QR_CONTRAST_NEG}",
    ),
    "custom": (
        "",
        f"low quality, blurry, deformed, watermark, text, {_QR_CONTRAST_NEG}",
    ),
}


def compose(prompt: str, style: str, negative_override: str | None) -> tuple[str, str]:
    suffix, default_negative = STYLE_PRESETS.get(style, STYLE_PRESETS["photoreal"])
    full_prompt = f"{prompt}, {suffix}" if suffix else prompt
    full_negative = negative_override if negative_override else default_negative
    return full_prompt, full_negative
