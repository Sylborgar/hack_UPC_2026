"""Florence-2 model wrapper for OCR and phrase grounding tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass
class FlorenceConfig:
    model_name: str
    device: torch.device
    max_new_tokens: int = 1024
    num_beams: int = 3


class FlorenceRunner:
    """Thin wrapper around Florence-2 generation + task post-processing."""

    def __init__(self, config: FlorenceConfig):
        self.config = config

        dtype = self._resolve_dtype(config.device)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        model_kwargs["attn_implementation"] = "eager"
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

        # Some Florence remote-code variants do not define these flags, but newer
        # transformers internals expect them to exist.
        if not hasattr(self.model, "_supports_sdpa"):
            self.model._supports_sdpa = False
        if not hasattr(self.model, "_supports_flash_attn_2"):
            self.model._supports_flash_attn_2 = False

        self.model.to(config.device)
        self.model.eval()

    @staticmethod
    def _resolve_dtype(device: torch.device) -> torch.dtype | None:
        if device.type == "cuda":
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return None

    def run_task(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> dict[str, Any]:
        prompt = task_prompt if not text_input else f"{task_prompt} {text_input}"

        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            input_ids = inputs["input_ids"].to(self.config.device)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                pixel_values = inputs["pixel_values"].to(device=self.config.device, dtype=model_dtype)
            else:
                pixel_values = inputs["pixel_values"].to(self.config.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=False,
                    use_cache=False,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        except Exception as exc:
            return {"error": str(exc), "task": task_prompt}

        try:
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )
        except Exception:
            parsed = {"raw_text": generated_text, "task": task_prompt}

        return parsed

    def run_ocr_with_region(self, image: Image.Image) -> dict[str, Any]:
        return self.run_task(image=image, task_prompt="<OCR_WITH_REGION>")

    def run_phrase_grounding(self, image: Image.Image, text: str) -> dict[str, Any]:
        return self.run_task(
            image=image,
            task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
            text_input=text,
        )
