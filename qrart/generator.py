from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable
from PIL import Image

from .canvas import build_composition, composite_qr_into_scene, is_standalone
from .pipeline import QRArtPipeline
from .scannability import score as scannability_score
from .scanner import scan
from .styles import compose


@dataclass
class Progress:
    """Per-job progress emitter. The worker passes one to Generator.generate()
    so pipeline step callbacks can emit phase-aware events and check for
    cancellation. publish(type, payload) is the only side-effect — the worker
    wires it to db.insert_event so SSE subscribers can poll it out.
    """
    publish: Callable[[str, dict[str, Any]], None] | None = None
    is_cancelled: Callable[[], bool] | None = None
    total_candidates: int = 1
    candidate_idx: int = 0

    def emit(self, type_: str, **payload: Any) -> None:
        if self.publish is None:
            return
        self.publish(type_, payload)

    def step_cb(self, phase: str, total: int) -> Callable[[int], None]:
        def cb(step: int) -> None:
            self.emit(
                "step",
                phase=phase,
                candidate=self.candidate_idx,
                total_candidates=self.total_candidates,
                step=step,
                total_steps=total,
            )
        return cb

    @property
    def cancel_check(self) -> Callable[[], bool] | None:
        return self.is_cancelled


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
    # A2: see app.GenerateBody. Persisted on the request so reruns inherit it.
    auto_escalate: bool = True
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
    scannability: float = 0.0       # 0.0-1.0, fraction of correctly-resolved QR modules


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


def _score_for(image: Image.Image, data: str, comp) -> float:
    """Compute scannability against the QR region defined by comp."""
    if is_standalone_comp(comp):
        return scannability_score(image, data)
    return scannability_score(
        image, data, qr_pos=comp.qr_pos, qr_size=comp.qr_size,
    )


def is_standalone_comp(comp) -> bool:
    """A standalone comp's qr_pos is (0,0) and qr_size matches canvas dims."""
    return comp.qr_pos == (0, 0) and comp.qr_size == comp.canvas_w == comp.canvas_h


class Generator:
    def __init__(self, base_model: str | None = None):
        self.pipeline = QRArtPipeline(base_model=base_model)

    def warm(self) -> None:
        self.pipeline.load()

    def generate(
        self,
        req: GenerationRequest,
        progress: Progress | None = None,
    ) -> GenerationResult:
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

        progress = progress or Progress()
        progress.total_candidates = max(1, req.candidates)

        for i in range(max(1, req.candidates)):
            seed = rng.randrange(2**31)
            progress.candidate_idx = i
            progress.emit("candidate_started", idx=i, seed=seed)
            cand = self._make_candidate(req, comp, seed, prompt, negative, progress)
            progress.emit(
                "candidate_done",
                idx=i,
                scans=cand.scans,
                decoded=cand.decoded,
                controlnet_scale=cand.controlnet_scale,
                refine_strength=cand.refine_strength,
                scannability=cand.scannability,
            )
            candidates.append(cand)

        # Best: scans first, then highest scannability score, then lowest
        # controlnet_scale (= least visible QR). Score breaks ties when
        # multiple candidates scan, and when none scan it picks the closest
        # one — which the C1 rescue pass nudges over the threshold.
        best = sorted(
            candidates,
            key=lambda c: (
                0 if c.scans else 1,
                -c.scannability,
                c.controlnet_scale,
            ),
        )[0]
        best_idx = candidates.index(best)

        # C1: cheap in-generation rescue. If no candidate scanned but the
        # best is close (score ≥ 0.70), do ONE more generation at scale
        # +0.10 with the best seed +1. Cheaper than A2's full re-run, and
        # often enough to push a borderline candidate over.
        if not best.scans and best.scannability >= 0.70 and req.controlnet_scale < 1.5:
            rescue_scale = round(req.controlnet_scale + 0.10, 2)
            progress.emit(
                "rescue_started",
                from_score=round(best.scannability, 3),
                from_scale=req.controlnet_scale,
                to_scale=rescue_scale,
            )
            rescue_req = GenerationRequest(**{
                **req.__dict__,
                "controlnet_scale": rescue_scale,
                "candidates": 1,
                "seed": best.seed + 1,
            })
            rescue_progress = Progress(
                publish=progress.publish,
                is_cancelled=progress.is_cancelled,
                # Show the rescue as one extra candidate beyond the original
                # set so the UI's progress bar makes sense.
                total_candidates=len(candidates) + 1,
                candidate_idx=len(candidates),
            )
            rescue = self._make_candidate(
                rescue_req, comp, best.seed + 1, prompt, negative, rescue_progress,
            )
            progress.emit(
                "rescue_done",
                scans=rescue.scans,
                score=round(rescue.scannability, 3),
            )
            candidates.append(rescue)
            # Re-pick best including the rescue candidate.
            best = sorted(
                candidates,
                key=lambda c: (
                    0 if c.scans else 1,
                    -c.scannability,
                    c.controlnet_scale,
                ),
            )[0]
            best_idx = candidates.index(best)

        # Finishing passes — only run on the winner so we don't pay 3x for
        # them. Each pass re-scans; if it kills scannability we keep the
        # post-processed image but report scans=False so the UI can flag it.
        if req.hires_fix or req.adetailer:
            final = best.image
            if req.hires_fix:
                progress.emit("phase", phase="hires", candidate=best_idx)
                final = self.pipeline.hires_fix(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    target_size=req.hires_target,
                    strength=req.hires_strength,
                    steps=req.hires_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                    step_callback=progress.step_cb("hires", req.hires_steps),
                    cancel_check=progress.cancel_check,
                )
            if req.adetailer:
                progress.emit("phase", phase="adetailer", candidate=best_idx)
                final = self.pipeline.adetailer_faces(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    strength=req.adetailer_strength,
                    steps=req.adetailer_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                    step_callback=progress.step_cb("adetailer", req.adetailer_steps),
                    cancel_check=progress.cancel_check,
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
                scannability=_score_for(final, req.data, comp),
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
        progress: Progress,
    ) -> Candidate:
        # Generate the QR art using the standalone path. For non-standalone
        # compositions this runs at qr_size × qr_size — small canvas, full QR
        # control image, matches QR Monster's training distribution exactly.
        progress.emit("phase", phase="pass1", candidate=progress.candidate_idx)
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
            step_callback=progress.step_cb("pass1", req.steps),
            cancel_check=progress.cancel_check,
        )

        # For non-standalone, generate the scene independently. Different seed
        # offset so the scene RNG isn't correlated with the QR art RNG.
        scene: Image.Image | None = None
        if not is_standalone(req.composition):
            progress.emit("phase", phase="scene", candidate=progress.candidate_idx)
            scene = self.pipeline.generate_scene(
                prompt=prompt,
                negative_prompt=negative,
                steps=req.steps,
                guidance=req.guidance,
                seed=seed + 9001,
                width=comp.canvas_w,
                height=comp.canvas_h,
                step_callback=progress.step_cb("scene", req.steps),
                cancel_check=progress.cancel_check,
            )

        def composite(qr_art: Image.Image) -> Image.Image:
            if scene is None:
                return qr_art
            return composite_qr_into_scene(
                scene, qr_art, req.composition, data=req.data,
            )

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
                scannability=_score_for(final, req.data, comp),
            )

        # Refine the QR art (not the composite — scene doesn't need it). Try
        # each strength in the ladder; first one that scans wins.
        last: Candidate | None = None
        pass1_composite = composite(qr_pass1)
        for strength in _refine_strengths(req.refine_strength):
            progress.emit(
                "phase", phase="refine", candidate=progress.candidate_idx,
                strength=strength,
            )
            qr_refined = self.pipeline.refine(
                image=qr_pass1,
                prompt=prompt,
                negative_prompt=negative,
                strength=strength,
                steps=req.refine_steps,
                guidance=req.guidance,
                seed=seed,
                step_callback=progress.step_cb("refine", req.refine_steps),
                cancel_check=progress.cancel_check,
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
                scannability=_score_for(final, req.data, comp),
            )
            if ok:
                return cand
            last = cand

        assert last is not None
        return last
