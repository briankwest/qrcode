from __future__ import annotations

# Each preset is (positive suffix, default negative). The user's prompt is
# appended with the suffix; the negative prompt defaults to the listed one but
# can be overridden via GenerationRequest.negative_prompt.
STYLE_PRESETS: dict[str, tuple[str, str]] = {
    "photoreal": (
        "RAW photo, 8k uhd, DSLR, cinematic lighting, film grain, "
        "Kodak Portra 400, sharp focus, photographic, hyperdetailed, realistic skin texture",
        "illustration, painting, drawing, art, sketch, cartoon, anime, render, "
        "cgi, 3d, plastic, low quality, blurry, deformed, watermark, text, signature",
    ),
    "cinematic": (
        "cinematic, dramatic lighting, ultra detailed, octane render, masterpiece, 8k, vivid",
        "low quality, blurry, deformed, watermark, text, ugly, jpeg artifacts",
    ),
    "illustration": (
        "digital illustration, concept art, trending on artstation, masterpiece, vivid colors",
        "low quality, blurry, deformed, watermark, photo, photographic, jpeg artifacts",
    ),
    "custom": (
        "",
        "low quality, blurry, deformed, watermark, text",
    ),
}


def compose(prompt: str, style: str, negative_override: str | None) -> tuple[str, str]:
    suffix, default_negative = STYLE_PRESETS.get(style, STYLE_PRESETS["photoreal"])
    full_prompt = f"{prompt}, {suffix}" if suffix else prompt
    full_negative = negative_override if negative_override else default_negative
    return full_prompt, full_negative
