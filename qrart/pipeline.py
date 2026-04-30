from __future__ import annotations

import os
import torch
from diffusers import (
    StableDiffusionPipeline,
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

# Tile ControlNet — when stacked alongside QR Monster, it adds a coherence /
# detail-preservation signal that pushes outputs toward photo (less QR-noisy)
# at no significant runtime cost. Default tile_scale=0 keeps it dormant.
CONTROLNET_TILE_ID = "lllyasviel/control_v11f1e_sd15_tile"

# External VAE that produces sharper, higher-contrast outputs than the bundled one.
VAE_ID = "stabilityai/sd-vae-ft-mse"

# Latent Consistency Model LoRA — turns SD 1.5 into a 4–8 step generator.
# ~3–4x speedup at the cost of some fidelity. Used by the Fast mode toggle.
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

MODELS = {
    # All SD 1.5 finetunes — same VAE + QR Monster ControlNet. Differences are
    # aesthetic: warm/cool palette, soft/sharp focus, photo-y vs stylized.
    "photoreal": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "photoreal-v51": "SG161222/Realistic_Vision_V5.1_noVAE",
    "photon": "digiplay/Photon_v1",
    "epic": "emilianJR/epiCRealism",
    "absolute": "Lykon/AbsoluteReality",
    "analog": "wavymulder/Analog-Diffusion",
    "dreamlike": "dreamlike-art/dreamlike-photoreal-2.0",
    "openjourney": "prompthero/openjourney-v4",
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

    For non-standalone compositions, the scene_pipe (txt2img, no ControlNet)
    generates a full-canvas scene, the QR art is generated via the standalone
    Pass 1 path at QR-region size, and the two are composited in canvas.py.
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
        # Scene pipe (txt2img, no ControlNet) for stage A of paste-composite
        # compositions — shares weights with the main pipe; created lazily.
        self._scene_pipe: StableDiffusionPipeline | None = None
        self._lcm_loaded = False
        self._fast_mode = False
        self._default_scheduler_config: dict | None = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        qr_controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=self.dtype)
        tile_controlnet = ControlNetModel.from_pretrained(CONTROLNET_TILE_ID, torch_dtype=self.dtype)
        # Multi-ControlNet: QR Monster does the heavy lifting, Tile rides along
        # at low scale to bias toward coherent photo structure.
        controlnets = [qr_controlnet, tile_controlnet]

        kwargs: dict = {
            "torch_dtype": self.dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if self.use_external_vae:
            kwargs["vae"] = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=self.dtype)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnets, **kwargs
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

        # Scene generation pipe — txt2img with no ControlNet. Used for stage A
        # of paste-composite compositions to produce the surrounding scene.
        self._scene_pipe = StableDiffusionPipeline(
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
            self._pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        else:
            assert self._default_scheduler_config is not None
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._default_scheduler_config
            )
            if self._lcm_loaded:
                try:
                    self._pipe.set_adapters([], adapter_weights=[])
                except Exception:
                    self._pipe.disable_lora()
        # Keep all sibling pipes (refiner, scene, inpaint) on the same scheduler.
        self._refiner.scheduler = self._pipe.scheduler
        if self._scene_pipe is not None:
            self._scene_pipe.scheduler = self._pipe.scheduler
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
        tile_scale: float,
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
        qr = qr_image.resize((width, height))
        out = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[qr, qr],
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=[controlnet_scale, tile_scale],
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            generator=gen,
            width=width,
            height=height,
        )
        return out.images[0]

    def generate_scene(
        self,
        prompt: str,
        *,
        negative_prompt: str,
        steps: int,
        guidance: float,
        seed: int | None,
        width: int,
        height: int,
    ) -> Image.Image:
        """Stage A: clean scene generation with no ControlNet. Used as the
        base for inpaint compositions. Output is what the canvas would look
        like before any QR is added."""
        self.load()
        assert self._scene_pipe is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        out = self._scene_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
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

    def hires_fix(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        target_size: int,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> Image.Image:
        """Lanczos-upscale to target_size on the longest dim, then img2img at
        low strength. Adds detail/sharpness without redrawing — strength must
        stay low (~0.15-0.25) or the QR pattern washes out.
        """
        self.load()
        assert self._refiner is not None

        w, h = image.size
        if w >= h:
            new_w = target_size
            new_h = int(round(h * (target_size / w)))
        else:
            new_h = target_size
            new_w = int(round(w * (target_size / h)))
        # SD requires multiple-of-8 dims.
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        if new_w <= w and new_h <= h:
            return image

        upscaled = image.resize((new_w, new_h), Image.LANCZOS)
        gen = (
            torch.Generator(device="cpu").manual_seed(seed + 31337)
            if seed is not None
            else None
        )
        out = self._refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        return out.images[0]

    def adetailer_faces(
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
        """Detect faces with cv2 Haar cascade, re-render each at 512x512 via
        img2img, paste back. No-op if no faces detected. Uses lower strength
        than full refine so the underlying QR structure isn't disrupted.
        """
        import cv2
        import numpy as np

        self.load()
        assert self._refiner is not None

        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )
        if len(faces) == 0:
            return image

        result = image.copy()
        face_prompt = (
            prompt
            + ", detailed realistic face, sharp focus, photorealistic skin texture, "
              "natural eyes, symmetric features"
        )
        face_negative = (
            (negative_prompt + ", ") if negative_prompt else ""
        ) + "deformed, asymmetric eyes, extra eyes, distorted face, blurry face"

        for (x, y, w, h) in faces:
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.width, x + w + pad_x)
            y2 = min(image.height, y + h + pad_y)
            face_crop = result.crop((x1, y1, x2, y2))
            orig_size = face_crop.size
            face_resized = face_crop.resize((512, 512), Image.LANCZOS)

            face_seed = (seed + int(x) * 7 + int(y) * 13) if seed is not None else None
            gen = (
                torch.Generator(device="cpu").manual_seed(face_seed)
                if face_seed is not None
                else None
            )
            out = self._refiner(
                prompt=face_prompt,
                negative_prompt=face_negative,
                image=face_resized,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            )
            face_rendered = out.images[0].resize(orig_size, Image.LANCZOS)
            result.paste(face_rendered, (x1, y1))

        return result
