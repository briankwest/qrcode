from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import COMPOSITIONS, Generator, GenerationRequest
from .qr import make_qr
from .scanner import scan
from .styles import STYLE_PRESETS, compose


def main() -> int:
    p = argparse.ArgumentParser(prog="qrart", description="Generate AI QR art locally.")
    p.add_argument("data", help="URL or text to encode")
    p.add_argument("--prompt", required=True, help="Image prompt")
    p.add_argument("--out", default="out.png", help="Output PNG path (best candidate)")
    p.add_argument("--style", default="photoreal", choices=list(STYLE_PRESETS.keys()))
    p.add_argument(
        "--composition",
        default="standalone",
        choices=list(COMPOSITIONS.keys()),
        help="standalone (768x768), subject-portrait, scene-landscape, garment",
    )
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--scale", type=float, default=1.35, help="ControlNet conditioning scale")
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-refine", action="store_true", help="Skip the img2img refine pass")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: load LCM-LoRA, run at 6 steps, ~3× faster (slight fidelity drop)",
    )
    p.add_argument("--refine-strength", type=float, default=0.30)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--no-require-scan", action="store_true")
    p.add_argument("--save-all", action="store_true", help="Save every candidate")
    p.add_argument("--save-pass1", action="store_true", help="Also save the pre-refine pass-1 image")
    p.add_argument(
        "--model",
        default=None,
        help="Model alias (photoreal, photoreal-xl, dreamshaper, sdxl-base) or any HF id",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Fan out across (scale × refine-strength) combos with the same seed",
    )
    p.add_argument(
        "--sweep-scales",
        default="1.2,1.3,1.4,1.5",
        help="Comma-separated controlnet scales for --sweep",
    )
    p.add_argument(
        "--sweep-strengths",
        default="0.20,0.30,0.40",
        help="Comma-separated refine strengths for --sweep",
    )
    args = p.parse_args()

    gen = Generator(base_model=args.model)

    if args.sweep:
        return _run_sweep(gen, args)

    # Fast mode swaps in LCM defaults if the user didn't override them.
    if args.fast:
        steps = args.steps if args.steps != 28 else 6
        refine_steps = args.refine_steps if args.refine_steps != 20 else 12
        guidance = args.guidance if args.guidance != 7.5 else 1.5
    else:
        steps = args.steps
        refine_steps = args.refine_steps
        guidance = args.guidance

    req = GenerationRequest(
        data=args.data,
        prompt=args.prompt,
        style=args.style,
        negative_prompt=args.negative_prompt,
        candidates=args.candidates,
        steps=steps,
        controlnet_scale=args.scale,
        guidance=guidance,
        size=args.size,
        composition=args.composition,
        seed=args.seed,
        refine=not args.no_refine,
        refine_strength=args.refine_strength,
        refine_steps=refine_steps,
        require_scan=not args.no_require_scan,
        fast_mode=args.fast,
    )
    print(
        f"Generating {args.candidates} candidate(s) on {gen.pipeline.device} "
        f"({gen.pipeline.base_model})...",
        file=sys.stderr,
    )
    result = gen.generate(req)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out)
    rs = f" refine={result.refine_strength:.2f}" if result.refine_strength else ""
    print(
        f"Best -> {out}  scans={result.scans}  seed={result.seed}  "
        f"scale={result.controlnet_scale:.2f}{rs}"
    )

    if args.save_all:
        for i, c in enumerate(result.candidates):
            cp = out.with_name(f"{out.stem}.cand{i}.seed{c.seed}.scans-{c.scans}{out.suffix}")
            c.image.save(cp)
            print(f"  cand{i} -> {cp}")
            if args.save_pass1 and c.pass1_image is not None:
                pp = out.with_name(f"{out.stem}.cand{i}.seed{c.seed}.pass1{out.suffix}")
                c.pass1_image.save(pp)
    return 0 if result.scans else 1


def _run_sweep(gen: Generator, args: argparse.Namespace) -> int:
    """Run a grid of (scale × refine-strength) with the same seed and prompt.

    Useful for finding the per-prompt sweet spot — the curves vary a lot by
    composition, so a single sweep is worth more than tuning one image at a time.
    """
    scales = [float(s) for s in args.sweep_scales.split(",") if s]
    strengths = [float(s) for s in args.sweep_strengths.split(",") if s]
    seed = args.seed if args.seed is not None else 42
    prompt, negative = compose(args.prompt, args.style, args.negative_prompt)

    out = Path(args.out)
    out_dir = out.parent / out.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep: {len(scales)} scales × {len(strengths)} strengths = {len(scales)*len(strengths)} images", file=sys.stderr)
    print(f"Saving to {out_dir}/", file=sys.stderr)

    qr = make_qr(args.data, size=args.size)
    rows: list[str] = []
    for scale in scales:
        pass1 = gen.pipeline.generate_pass1(
            qr_image=qr,
            prompt=prompt,
            negative_prompt=negative,
            steps=args.steps,
            guidance=args.guidance,
            controlnet_scale=scale,
            control_start=0.0,
            control_end=0.95,
            seed=seed,
            width=args.size,
            height=args.size,
        )
        pass1_path = out_dir / f"scale-{scale:.2f}__pass1.png"
        pass1.save(pass1_path)
        decoded1 = scan(pass1)
        rows.append(f"scale={scale:.2f} refine=---  scans={decoded1==args.data}  -> {pass1_path.name}")

        if args.no_refine:
            continue
        for strength in strengths:
            refined = gen.pipeline.refine(
                image=pass1,
                prompt=prompt,
                negative_prompt=negative,
                strength=strength,
                steps=args.refine_steps,
                guidance=args.guidance,
                seed=seed,
            )
            path = out_dir / f"scale-{scale:.2f}__refine-{strength:.2f}.png"
            refined.save(path)
            decoded = scan(refined)
            rows.append(
                f"scale={scale:.2f} refine={strength:.2f} scans={decoded==args.data}  -> {path.name}"
            )

    print("\nSweep complete:")
    for r in rows:
        print(" ", r)
    (out_dir / "sweep.txt").write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
