"""Batch-generate the 10 test prompts via the running server.

Hits POST /api/generate for each prompt with its recommended model/composition
/finishing settings, then copies the winning candidate into ./batch_outputs/
under a labeled filename. Also writes a JSON sidecar with the full job result
so we can revisit metadata (seed, scale, decoded URL, etc.) later.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import urllib.request

ROOT = Path(__file__).parent
API = "http://127.0.0.1:8000/api/generate"
OUT_DIR = ROOT / "batch_outputs"
OUT_DIR.mkdir(exist_ok=True)

# (label, prompt, model, composition, tile_scale, hires_fix, adetailer)
# Baseline only: tile_scale=0, hires_fix=False. Tile ControlNet stacking with
# the QR pattern as input *reinforces* the QR — the opposite of what we want.
# Disabled until we can plumb in a non-QR tile reference. ADetailer kept on
# face prompts since it worked during dev tests.
PROMPTS: list[dict] = [
    {
        "label": "01_cosmic_galaxy",
        "prompt": "swirling spiral galaxy with luminous stardust clouds, deep purple and gold nebula, distant stars piercing the void, ultra-detailed astrophotography, Hubble deep field",
        "model": "photoreal",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "02_karst_mountains",
        "prompt": "ancient karst mountains rising from misty river at dawn, traditional fishing boat in silhouette, golden light cutting through fog, Guilin China, large format film",
        "model": "photon",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "03_eagle_hunter",
        "prompt": "weathered Mongolian eagle hunter in fur cloak, golden eagle perched on arm, snowy steppe at sunrise, intense gaze, Steve McCurry photograph, 85mm",
        "model": "photoreal",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": True,
    },
    {
        "label": "04_neon_tokyo",
        "prompt": "Tokyo Shibuya at midnight in rain, neon reflections on wet pavement, blade runner atmosphere, anamorphic lens flares, cinematic depth of field",
        "model": "epic",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "05_tiger_waterfall",
        "prompt": "Bengal tiger emerging from dense jungle waterfall, water droplets mid-air frozen, dappled afternoon light, National Geographic, telephoto compression",
        "model": "photoreal",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "06_sorceress",
        "prompt": "ethereal sorceress in flowing emerald robes, floating runes orbiting her hands, ancient stone circle at dusk, volumetric mist, painterly cinematic",
        "model": "dreamshaper",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "07_battle_mech",
        "prompt": "towering battle mech standing in ruined cyberpunk megacity, glowing reactor core, ash and embers drifting, dramatic chiaroscuro, Syd Mead concept art",
        "model": "epic",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "08_humpback_coral",
        "prompt": "humpback whale gliding through coral cathedral, shafts of sunlight piercing turquoise water, schools of silver fish, David Doubilet photograph",
        "model": "dreamlike",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
    {
        "label": "09_apollo_astronaut",
        "prompt": "1970s NASA astronaut in spacesuit on lunar surface, earthrise behind, Hasselblad film grain, faded color, Apollo program archival photograph",
        "model": "analog",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": True,
    },
    {
        "label": "10_aurora_cabin",
        "prompt": "lone wooden cabin under violent green aurora, frozen lake mirror reflection, snow-dusted pines, Iceland midwinter, long exposure astrophotography",
        "model": "photon",
        "composition": "standalone",
        "tile_scale": 0.0,
        "hires_fix": False,
        "adetailer": False,
    },
]


def build_body(p: dict, fast: bool, candidates: int, data: str) -> dict:
    body = {
        "data": data,
        "prompt": p["prompt"],
        "style": "photoreal",
        "model": p["model"],
        "composition": p["composition"],
        "candidates": candidates,
        "controlnet_scale": 1.35,
        "tile_scale": p["tile_scale"],
        "refine": True,
        "refine_strength": 0.30,
        "hires_fix": p["hires_fix"],
        "hires_target": 1024,
        "hires_strength": 0.20,
        "adetailer": p["adetailer"],
        "adetailer_strength": 0.35,
        "fast_mode": fast,
        "seed": None,
    }
    return body


def post(body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        API,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def warm(model: str) -> None:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/warm",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": model}).encode(),
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="LCM Fast mode (3-4x faster)")
    parser.add_argument("--candidates", type=int, default=2, help="Per-prompt candidates")
    parser.add_argument("--data", default="https://signalwire.com", help="QR payload")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated label substrings to filter")
    args = parser.parse_args()

    targets = PROMPTS
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        targets = [p for p in PROMPTS if any(k in p["label"] for k in keys)]
        if not targets:
            print(f"no prompts matched: {args.only}")
            return 1

    print(f"Generating {len(targets)} prompts · candidates={args.candidates} · fast={args.fast}")
    print(f"Output dir: {OUT_DIR}")
    print()

    t0 = time.time()
    results: list[dict] = []
    for i, p in enumerate(targets, 1):
        body = build_body(p, fast=args.fast, candidates=args.candidates, data=args.data)
        print(f"[{i:>2}/{len(targets)}] {p['label']:<22} {p['model']:<12} {p['composition']:<18} tile={p['tile_scale']} hires={p['hires_fix']} ad={p['adetailer']}")
        ts = time.time()
        try:
            j = post(body, timeout=900)
        except Exception as e:
            print(f"     ✗ FAILED: {e}")
            continue
        best = j["candidates"][j["best_index"]]
        src = ROOT / "outputs" / j["job_id"] / f"cand{best['index']}.png"
        dst = OUT_DIR / f"{p['label']}.png"
        if src.exists():
            shutil.copy(src, dst)
        # Copy the QR too in case we want to verify decoding
        qr_src = ROOT / "outputs" / j["job_id"] / "qr.png"
        if qr_src.exists() and i == 1:
            shutil.copy(qr_src, OUT_DIR / "_qr.png")
        sidecar = {
            "label": p["label"],
            "prompt": p["prompt"],
            "settings": {k: v for k, v in body.items() if k not in {"prompt"}},
            "elapsed_s": j["elapsed_s"],
            "best_seed": best["seed"],
            "scans": best["scans"],
            "decoded": best["decoded"],
            "controlnet_scale": best["controlnet_scale"],
            "refine_strength": best.get("refine_strength"),
        }
        (OUT_DIR / f"{p['label']}.json").write_text(json.dumps(sidecar, indent=2))
        elapsed = time.time() - ts
        scan_marker = "✓" if best["scans"] else "✗"
        print(f"     {scan_marker} {elapsed:>6.1f}s · seed={best['seed']} scale={best['controlnet_scale']:.2f} → {dst.name}")
        results.append(sidecar)

    total = time.time() - t0
    scanned = sum(1 for r in results if r["scans"])
    print()
    print(f"Done · {len(results)}/{len(targets)} succeeded · {scanned} scanned · {total/60:.1f} min total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
