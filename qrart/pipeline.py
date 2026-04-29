from __future__ import annotations

import os
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
)
from PIL import Image

# QR Monster v1 — trained for hiding QR codes inside generated images.
CONTROLNET_ID = "monster-labs/control_v1p_sd15_qrcode_monster"

# External VAE that produces sharper, higher-contrast outputs than the bundled one.
VAE_ID = "stabilityai/sd-vae-ft-mse"

# Latent Consistency Model LoRA — turns SD 1.5 into a 4–8 step generator.
# ~3–4x speedup at the cost of some fidelity. Used by the Fast mode toggle.
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

MODELS = {
    "photoreal": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "photoreal-v51": "SG161222/Realistic_Vision_V5.1_noVAE",
    "dreamshaper": "Lykon/dreamshaper-8",
}
DEFAULT_BASE_MODEL = MODELS["photoreal"]


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


def pick_device() -> tuple[str, torch.dtype]:
    # MPS defaults to fp32 — Realistic Vision V6 produces NaN in fp16 on Apple
    # Silicon. Set QRART_MPS_FP16=1 to opt in if your model tolerates it.
    if torch.backends.mps.is_available():
        if os.environ.get("QRART_MPS_FP16") == "1":
            return "mps", torch.float16
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class QRArtPipeline:
    """Two-pass diffusion pipeline.

    Pass 1: ControlNet pipe plants the QR pattern into a generated image.
    Pass 2: Img2Img pipe refines that QR-textured output toward photorealism
    while preserving enough structure to keep the code scannable.
    """

    def __init__(
        self,
        base_model: str | None = None,
        use_external_vae: bool = True,
    ) -> None:
        self.device, self.dtype = pick_device()
        self.base_model = resolve_model(base_model) if base_model else DEFAULT_BASE_MODEL
        self.use_external_vae = use_external_vae
        self._pipe: StableDiffusionControlNetPipeline | None = None
        self._refiner: StableDiffusionImg2ImgPipeline | None = None
        self._lcm_loaded = False
        self._fast_mode = False
        self._default_scheduler_config: dict | None = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=self.dtype)

        kwargs: dict = {
            "torch_dtype": self.dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if self.use_external_vae:
            kwargs["vae"] = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=self.dtype)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnet, **kwargs
        )
        # Euler-Ancestral is numerically stable on MPS; SDE-DPM variants tend to
        # produce NaN outputs in lower precision on Apple Silicon.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # Save the original scheduler config so we can restore it when toggling
        # back from Fast mode (LCM) to Quality.
        self._default_scheduler_config = dict(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        if self.device != "cpu":
            pipe.enable_attention_slicing()
        self._pipe = pipe

        self._refiner = StableDiffusionImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=pipe.feature_extractor,
            requires_safety_checker=False,
        )

    def ensure_lcm(self) -> None:
        """Lazily download + apply the LCM LoRA the first time Fast mode is used."""
        self.load()
        if self._lcm_loaded:
            return
        assert self._pipe is not None
        # Load with an adapter name so we can toggle it on/off later without unloading.
        self._pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
        self._lcm_loaded = True

    def set_fast_mode(self, fast: bool) -> None:
        """Switch between Quality (Euler-Ancestral, full LoRA off) and Fast
        (LCM scheduler + LCM LoRA active). Idempotent — calling twice is a no-op."""
        if fast == self._fast_mode and self._pipe is not None:
            return
        self.load()
        assert self._pipe is not None and self._refiner is not None
        if fast:
            self.ensure_lcm()
            self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
            self._refiner.scheduler = self._pipe.scheduler
            self._pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        else:
            assert self._default_scheduler_config is not None
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._default_scheduler_config
            )
            self._refiner.scheduler = self._pipe.scheduler
            if self._lcm_loaded:
                # Disable the adapter without unloading — fast to flip back later.
                try:
                    self._pipe.set_adapters([], adapter_weights=[])
                except Exception:
                    self._pipe.disable_lora()
        self._fast_mode = fast

    def generate_pass1(
        self,
        qr_image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        steps: int,
        guidance: float,
        controlnet_scale: float,
        control_start: float,
        control_end: float,
        seed: int | None,
        width: int,
        height: int,
    ) -> Image.Image:
        self.load()
        assert self._pipe is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        out = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=qr_image.resize((width, height)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=controlnet_scale,
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            generator=gen,
            width=width,
            height=height,
        )
        return out.images[0]

    def refine(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> Image.Image:
        """img2img pass with no ControlNet — smooths QR-textured pass-1 output
        into photorealism while preserving enough structure to still scan.
        """
        self.load()
        assert self._refiner is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed + 7919)
            if seed is not None
            else None
        )
        out = self._refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        return out.images[0]
