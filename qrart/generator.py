from __future__ import annotations

import random
from dataclasses import dataclass, field
from PIL import Image

from .pipeline import QRArtPipeline
from .qr import make_qr
from .scanner import scan
from .styles import compose


@dataclass
class GenerationRequest:
    data: str  # the URL/text to encode
    prompt: str
    style: str = "photoreal"
    negative_prompt: str | None = None  # overrides the style's default negative
    candidates: int = 3
    steps: int = 28
    guidance: float = 7.5
    # Sweet spot for "photo + scans": 1.30–1.40. Lower = prettier but doesn't
    # decode; higher = QR dominates the composition.
    controlnet_scale: float = 1.35
    control_end: float = 1.0  # full QR control through denoising for scannability
    refine: bool = True
    refine_strength: float = 0.30  # polishes pass-1 without erasing the QR
    refine_steps: int = 20
    size: int = 768
    seed: int | None = None
    require_scan: bool = True
    # Fast mode: swaps in LCM-LoRA + LCMScheduler. ~3–4x faster, slight fidelity drop.
    fast_mode: bool = False
    # If a candidate doesn't scan, bump controlnet_scale up to N times. Each
    # retry costs another full pass-1 + refine.
    max_retries_per_candidate: int = 1


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
        prompt, negative = compose(req.prompt, req.style, req.negative_prompt)
        qr = make_qr(req.data, size=req.size)
        # Apply Fast/Quality mode once per request so all candidates use the
        # same scheduler/LoRA state.
        self.pipeline.set_fast_mode(req.fast_mode)
        candidates: list[Candidate] = []
        rng = random.Random(req.seed)

        for _ in range(max(1, req.candidates)):
            seed = rng.randrange(2**31)
            candidates.append(self._make_candidate(req, qr, seed, prompt, negative))

        # Best: scans first, then lowest controlnet_scale (= least visible QR).
        best = sorted(
            candidates, key=lambda c: (0 if c.scans else 1, c.controlnet_scale)
        )[0]
        return GenerationResult(
            image=best.image,
            qr_image=qr,
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
        qr: Image.Image,
        seed: int,
        prompt: str,
        negative: str,
    ) -> Candidate:
        scale = req.controlnet_scale
        last: Candidate | None = None

        for _ in range(req.max_retries_per_candidate + 1):
            pass1 = self.pipeline.generate_pass1(
                qr_image=qr,
                prompt=prompt,
                negative_prompt=negative,
                steps=req.steps,
                guidance=req.guidance,
                controlnet_scale=scale,
                control_start=0.0,
                control_end=req.control_end,
                seed=seed,
                width=req.size,
                height=req.size,
            )

            if not req.refine:
                decoded = scan(pass1)
                ok = decoded == req.data
                cand = Candidate(
                    image=pass1,
                    pass1_image=None,
                    seed=seed,
                    scans=ok,
                    decoded=decoded,
                    controlnet_scale=scale,
                    refine_strength=None,
                )
                if ok or not req.require_scan:
                    return cand
                last = cand
                scale = min(scale + 0.15, 2.0)
                continue

            for strength in _refine_strengths(req.refine_strength):
                refined = self.pipeline.refine(
                    image=pass1,
                    prompt=prompt,
                    negative_prompt=negative,
                    strength=strength,
                    steps=req.refine_steps,
                    guidance=req.guidance,
                    seed=seed,
                )
                decoded = scan(refined)
                ok = decoded == req.data
                cand = Candidate(
                    image=refined,
                    pass1_image=pass1,
                    seed=seed,
                    scans=ok,
                    decoded=decoded,
                    controlnet_scale=scale,
                    refine_strength=strength,
                )
                if ok:
                    return cand
                last = cand

            if not req.require_scan:
                assert last is not None
                return last
            scale = min(scale + 0.15, 2.0)

        assert last is not None
        return last
