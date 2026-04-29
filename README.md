# QR Art Studio

Local AI QR art generator — generates photo-realistic images that secretly encode a QR code, using Stable Diffusion 1.5 + the Monster Labs QR ControlNet, with an img2img refine pass for that "looks like a real photo" finish.

Runs on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

```
                ┌─────────┐    ┌──────────────┐     ┌─────────────┐
  URL  ────►    │  QR-H   │ ──►│  Pass 1      │ ──► │  Pass 2     │ ──► PNG
                │  level  │    │  ControlNet  │     │  img2img    │
                └─────────┘    │  (plant QR)  │     │  (refine)   │
                               └──────────────┘     └─────────────┘
```

Pass 1 plants the QR pattern. Pass 2 (no ControlNet, low denoising strength) smooths the QR-textured output into a photoreal image while preserving enough structure to still scan. Every output is verified — if scanning breaks, the generator retries automatically with adjusted parameters.

---

## Setup

Requires Python 3.12 (3.11 also works).

```bash
cd /Users/brian/workdir/qrcode
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### First-run model downloads (~6GB total, cached in `~/.cache/huggingface`)

| Model | Purpose | Size |
|---|---|---|
| `SG161222/Realistic_Vision_V6.0_B1_noVAE` | Photoreal SD 1.5 base | ~4 GB |
| `monster-labs/control_v1p_sd15_qrcode_monster` | QR ControlNet | ~1.4 GB |
| `stabilityai/sd-vae-ft-mse` | Sharper VAE | ~330 MB |

Downloads happen automatically on first generation.

---

## Web UI

```bash
source venv/bin/activate
python app.py
# open http://127.0.0.1:8000
```

The page auto-warms the model on load (~30s on M-series). After that, generations take ~30–50s for a 768×768 with the refine pass enabled.

**UI features:**
- **Style dropdown** — `photoreal` / `cinematic` / `illustration` / `custom`. Auto-applies a positive suffix and negative prompt scaffold tuned for that style.
- **Reference image (IP-Adapter)** — drag any image into the dropzone to influence the output's style, colors, and mood. Adjustable strength.
- **Refine toggle** — flip off for a single-pass generation (faster, more QR-textured). On by default.
- **QR strength slider** — controlnet conditioning scale. Sweet spot 1.3–1.5 with refine on.
- **Refine strength slider** — img2img denoising strength. Sweet spot 0.25–0.35.
- **Compare pass 1 ↔ refined** — see what the refine pass actually did.
- **Candidates gallery** — generate up to 6 in one go; the best-scanning is auto-selected.

---

## CLI

```bash
source venv/bin/activate

# Single image, photoreal, with refine
python -m qrart "https://signalwire.com" \
  --prompt "a majestic snow leopard standing on a cliff at sunrise, dramatic lighting" \
  --out art.png

# More candidates, save them all
python -m qrart "https://signalwire.com" \
  --prompt "a cinematic mountain landscape at golden hour" \
  --candidates 4 --save-all --save-pass1 --out art.png

# Skip refine (faster, ~30% less time)
python -m qrart "https://example.com" --prompt "..." --no-refine

# Fast mode — LCM-LoRA, 6 steps, ~3× faster
python -m qrart "https://example.com" --prompt "..." --fast

# Pick a different style
python -m qrart "https://example.com" --prompt "..." --style cinematic
python -m qrart "https://example.com" --prompt "..." --style illustration

# Use a reference image (IP-Adapter) to influence style/colors
python -m qrart "https://signalwire.com" \
  --prompt "a cinematic mountain landscape" \
  --reference ./inspiration.jpg --reference-scale 0.5 \
  --out art.png
```

### Tuning sweep

When dialing in a prompt, `--sweep` fans out across a grid of `controlnet_scale × refine_strength` with the same seed, so you can pick the best combo by eye:

```bash
python -m qrart "https://signalwire.com" \
  --prompt "a snow leopard on a cliff at sunrise, cinematic" \
  --sweep --seed 42 \
  --sweep-scales 1.2,1.3,1.4,1.5 \
  --sweep-strengths 0.20,0.30,0.40 \
  --out leopard.png
# Outputs land in ./leopard/scale-X__refine-Y.png plus a sweep.txt summary
```

### All flags

```
qrart DATA --prompt PROMPT [options]

Options:
  --out PATH                 output PNG (best candidate)
  --style {photoreal,cinematic,illustration,custom}
  --negative-prompt TEXT     override the style's default negative
  --candidates N             generate N, pick the best (default 4)
  --steps N                  pass-1 steps (default 35)
  --scale FLOAT              controlnet scale (default 1.4)
  --guidance FLOAT           CFG (default 7.5)
  --size N                   resolution (default 768)
  --seed N                   reproducibility
  --no-refine                skip the img2img refine pass
  --refine-strength FLOAT    refine denoising (default 0.30)
  --refine-steps N           refine steps (default 25)
  --no-require-scan          accept first candidate even if it doesn't scan
  --save-all                 save every candidate, not just the best
  --save-pass1               also save the pre-refine image for each candidate
  --model NAME-OR-HF-ID      base model alias or full HF id
  --reference PATH           reference image (IP-Adapter influence)
  --reference-scale FLOAT    IP-Adapter strength 0–1 (default 0.5)
  --sweep                    grid sweep across (scale, refine-strength)
  --sweep-scales A,B,C       scales to sweep
  --sweep-strengths A,B,C    refine strengths to sweep
```

Available model aliases:
- `photoreal` — Realistic Vision V6 (default)
- `photoreal-v51` — Realistic Vision V5.1
- `dreamshaper` — DreamShaper 8 (more stylized)

You can also pass any SD-1.5-compatible Hugging Face id directly.

---

## Tuning guide

### "I want a more photorealistic look"
- Use `--style photoreal` (default) — it auto-appends RAW photo / 8k / DSLR / film grain language and adds illustration/cartoon/render to the negative prompt.
- Keep refine **on** with strength 0.25–0.35. The refine pass is what makes the difference between "QR with photo texture" and "photo with hidden QR."
- Use simple, dramatic prompts with strong central subjects ("a snow leopard on a cliff at sunrise") rather than busy scenes.
- Resolution 768 is the sweet spot. 512 is too small for the QR ControlNet to work cleanly.

### "I want a more obvious QR / more reliable scanning"
- Bump `--scale` to 1.5–1.7
- Lower `--refine-strength` to 0.15–0.20, or `--no-refine` entirely
- `--style cinematic` is more permissive of QR artifacts than photoreal

### Reference image (IP-Adapter)
Drop an image into the UI dropzone (or pass `--reference path.jpg`). The model uses your image as a "visual prompt" — its style, colors, and overall mood influence the output without locking the composition.

- **Strength 0.3–0.5**: subtle hint, prompt still drives the image
- **Strength 0.6–0.8**: strong influence, output looks like a stylistic remix of your reference
- **Strength 1.0+**: reference dominates, prompt becomes background

Use cases:
- Match a brand color palette by uploading a brand image
- Get a specific photography style (vintage film, drone shot, etc.) from a reference photo
- Pin a mood/lighting that's hard to describe in words

First time you use it, ~50MB of IP-Adapter weights download (`h94/IP-Adapter`, cached after).

### "It's not scanning"
- The generator already auto-retries with bumped scale and lowered refine strength. If all retries fail, you'll see `scans=False` and the file is still saved — try running with `--candidates 4` to roll new seeds.
- Some prompts genuinely fight the QR (very dark scenes, busy textures). Try a brighter or simpler composition.
- Try `--style cinematic` instead of `photoreal` — the refine pass on photoreal can be aggressive.

### Speed
- ~30s per image with refine off, ~50s with refine on (768×768, M-series, 35 steps)
- `--steps 25 --refine-steps 18` shaves ~30% with minor quality cost
- `--size 512` is faster but quality drops noticeably

---

## Troubleshooting

**Black or garbage outputs on Apple Silicon**: known SD 1.5 + fp16 + MPS issue. We default to fp32 on MPS. Don't set `QRART_MPS_FP16=1` unless you're prepared to verify outputs.

**`zsh: no matches found: qrcode[pil]` during install**: use `pip install "qrcode[pil]"` (with quotes) or stick with `pip install -r requirements.txt`.

**Slow first run**: ~6GB of model weights download on first generation. They're cached after that.

**Models won't download**: check `~/.cache/huggingface/`. If you have HF token gating issues, run `huggingface-cli login`.

**Out-of-memory**: drop `--size` to 512 or set `QRART_MPS_FP16=1` (with caveats). The pipeline already uses attention slicing.

---

## Project layout

```
qrcode/
├── app.py                  # FastAPI server + JSON API
├── qrart/
│   ├── __init__.py
│   ├── __main__.py         # CLI entrypoint (python -m qrart)
│   ├── pipeline.py         # ControlNet + Img2Img diffusers pipelines
│   ├── generator.py        # Orchestration: candidates, retry, scan-validate
│   ├── styles.py           # photoreal / cinematic / illustration prompt presets
│   ├── qr.py               # QR-H rendering
│   └── scanner.py          # Multi-variant cv2 QR decoder
├── static/index.html       # Web UI
├── outputs/                # Generated images served at /outputs/<job-id>/
├── requirements.txt
└── README.md
```

---

## API

`POST /api/generate`

```json
{
  "data": "https://signalwire.com",
  "prompt": "a snow leopard on a cliff at sunrise",
  "style": "photoreal",
  "candidates": 2,
  "controlnet_scale": 1.4,
  "refine": true,
  "refine_strength": 0.30,
  "steps": 35,
  "seed": null
}
```

Returns:
```json
{
  "job_id": "ab12cd34ef56",
  "elapsed_s": 47.2,
  "best_index": 1,
  "scans": true,
  "decoded": "https://signalwire.com",
  "qr_url": "/outputs/ab12cd34ef56/qr.png",
  "candidates": [
    {
      "index": 0,
      "url": "/outputs/ab12cd34ef56/cand0.png",
      "pass1_url": "/outputs/ab12cd34ef56/cand0.pass1.png",
      "seed": 1234567,
      "scans": false,
      "controlnet_scale": 1.4,
      "refine_strength": 0.3
    }
  ]
}
```

`POST /api/warm` — load model into memory eagerly. Returns when ready.
`GET /api/health` — device, model, loaded state.
