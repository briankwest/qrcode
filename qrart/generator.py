from __future__ import annotations

import random
from dataclasses import dataclass, field
from PIL import Image

from .canvas import build_composition, composite_qr_into_scene, is_standalone
from .pipeline import QRArtPipeline
from .scanner import scan
from .styles import compose


@dataclass
class GenerationRequest:
    data: str  # the URL/text to encode
    prompt: str
    style: str = "photoreal"
    negative_prompt: str | None = None  # overrides the style's default negative
    candidates: int = 5
    steps: int = 28
    guidance: float = 7.5
    # Photo-dominant band: 1.05–1.20. Below 1.05 the QR usually doesn't decode;
    # above 1.20 the grid pattern starts to dominate. With 5+ candidates, scan
    # rate at 1.10 is high enough to reliably get a winner.
    controlnet_scale: float = 1.10
    # Tile ControlNet stacked alongside QR Monster. 0 = off; 0.3-0.5 nudges
    # toward photo coherence at the cost of slightly weakened QR signal.
    tile_scale: float = 0.0
    control_end: float = 1.0  # full QR control through denoising for scannability
    refine: bool = True
    refine_strength: float = 0.30  # polishes pass-1 without erasing the QR
    refine_steps: int = 20
    size: int = 768  # ignored when composition != "standalone" (canvas drives size)
    composition: str = "standalone"
    seed: int | None = None
    require_scan: bool = True
    # Fast mode: swaps in LCM-LoRA + LCMScheduler. ~3–4x faster, slight fidelity drop.
    fast_mode: bool = False
    # Hi-res fix: upscale best candidate via Lanczos and run a low-strength
    # img2img pass for sharper detail. Only runs once on the winning candidate.
    hires_fix: bool = False
    hires_target: int = 1024
    hires_strength: float = 0.20
    hires_steps: int = 18
    # ADetailer: detect faces, re-render at 512x512 each, paste back. Same
    # "winner only" semantics as hi-res fix.
    adetailer: bool = False
    adetailer_strength: float = 0.35
    adetailer_steps: int = 20


@dataclass
class Candidate:
    image: Image.Image  # final image (refined if refine, else pass1)
    pass1_image: Image.Image | None  # pre-refine image, None when refine=False
    seed: int
    scans: bool
    decoded: str | None
    controlnet_scale: float
    refine_strength: float | None  # None when refine=False


@dataclass
class GenerationResult:
    image: Image.Image
    qr_image: Image.Image
    seed: int
    scans: bool
    decoded: str | None
    controlnet_scale: float
    refine_strength: float | None
    candidates: list[Candidate] = field(default_factory=list)


def _refine_strengths(target: float) -> list[float]:
    return [target] if target <= 0.18 else [target, max(0.15, target - 0.1)]


class Generator:
    def __init__(self, base_model: str | None = None):
        self.pipeline = QRArtPipeline(base_model=base_model)

    def warm(self) -> None:
        self.pipeline.load()

    def generate(self, req: GenerationRequest) -> GenerationResult:
        # Composition scaffold goes in BEFORE style preset so the style suffix
        # (RAW photo, 8k, etc.) lands at the very end where SD weighs it most.
        comp = build_composition(req.data, req.composition)
        full_prompt = req.prompt + comp.scaffold
        prompt, negative = compose(full_prompt, req.style, req.negative_prompt)

        # Apply Fast/Quality mode once per request so all candidates use the
        # same scheduler/LoRA state.
        self.pipeline.set_fast_mode(req.fast_mode)
        candidates: list[Candidate] = []
        rng = random.Random(req.seed)

        for _ in range(max(1, req.candidates)):
            seed = rng.randrange(2**31)
            candidates.append(
                self._make_candidate(req, comp, seed, prompt, negative)
            )

        # Best: scans first, then lowest controlnet_scale (= least visible QR).
        best = sorted(
            candidates, key=lambda c: (0 if c.scans else 1, c.controlnet_scale)
        )[0]
        best_idx = candidates.index(best)

        # Finishing passes — only run on the winner so we don't pay 3x for
        # them. Each pass re-scans; if it kills scannability we keep the
        # post-processed image but report scans=False so the UI can flag it.
        if req.hires_fix or req.adetailer:
            final = best.image
            if req.hires_fix:
                final = self.pipeline.hires_fix(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    target_size=req.hires_target,
                    strength=req.hires_strength,
                    steps=req.hires_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                )
            if req.adetailer:
                final = self.pipeline.adetailer_faces(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    strength=req.adetailer_strength,
                    steps=req.adetailer_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                )
            decoded = scan(final)
            best = Candidate(
                image=final,
                pass1_image=best.pass1_image,
                seed=best.seed,
                scans=decoded == req.data,
                decoded=decoded,
                controlnet_scale=best.controlnet_scale,
                refine_strength=best.refine_strength,
            )
            candidates[best_idx] = best

        return GenerationResult(
            image=best.image,
            qr_image=comp.qr_image,
            seed=best.seed,
            scans=best.scans,
            decoded=best.decoded,
            controlnet_scale=best.controlnet_scale,
            refine_strength=best.refine_strength,
            candidates=candidates,
        )

    def _make_candidate(
        self,
        req: GenerationRequest,
        comp,
        seed: int,
        prompt: str,
        negative: str,
    ) -> Candidate:
        # Generate the QR art using the standalone path. For non-standalone
        # compositions this runs at qr_size × qr_size — small canvas, full QR
        # control image, matches QR Monster's training distribution exactly.
        qr_pass1 = self.pipeline.generate_pass1(
            qr_image=comp.qr_image,
            prompt=prompt,
            negative_prompt=negative,
            steps=req.steps,
            guidance=req.guidance,
            controlnet_scale=req.controlnet_scale,
            tile_scale=req.tile_scale,
            control_start=0.0,
            control_end=req.control_end,
            seed=seed,
            width=comp.qr_size,
            height=comp.qr_size,
        )

        # For non-standalone, generate the scene independently. Different seed
        # offset so the scene RNG isn't correlated with the QR art RNG.
        scene: Image.Image | None = None
        if not is_standalone(req.composition):
            scene = self.pipeline.generate_scene(
                prompt=prompt,
                negative_prompt=negative,
                steps=req.steps,
                guidance=req.guidance,
                seed=seed + 9001,
                width=comp.canvas_w,
                height=comp.canvas_h,
            )

        def composite(qr_art: Image.Image) -> Image.Image:
            if scene is None:
                return qr_art
            return composite_qr_into_scene(scene, qr_art, req.composition)

        if not req.refine:
            final = composite(qr_pass1)
            decoded = scan(final)
            return Candidate(
                image=final,
                pass1_image=None,
                seed=seed,
                scans=decoded == req.data,
                decoded=decoded,
                controlnet_scale=req.controlnet_scale,
                refine_strength=None,
            )

        # Refine the QR art (not the composite — scene doesn't need it). Try
        # each strength in the ladder; first one that scans wins.
        last: Candidate | None = None
        pass1_composite = composite(qr_pass1)
        for strength in _refine_strengths(req.refine_strength):
            qr_refined = self.pipeline.refine(
                image=qr_pass1,
                prompt=prompt,
                negative_prompt=negative,
                strength=strength,
                steps=req.refine_steps,
                guidance=req.guidance,
                seed=seed,
            )
            final = composite(qr_refined)
            decoded = scan(final)
            ok = decoded == req.data
            cand = Candidate(
                image=final,
                pass1_image=pass1_composite,
                seed=seed,
                scans=ok,
                decoded=decoded,
                controlnet_scale=req.controlnet_scale,
                refine_strength=strength,
            )
            if ok:
                return cand
            last = cand

        assert last is not None
        return last
