"""HydraLLM adapter: multimodal HydraGemma 3 with M3 visual-token pooling and head-count width.

How the budget reaches HydraLLM without modifying its code:

* ``W`` -> ``current_num_heads=W`` passed to HydraLLM's own ``generate``. HydraLLM derives the
  embedding, FFN and projector output widths from it.
* ``N`` -> (1) the run of ``mm_tokens_per_image`` image placeholder tokens produced by the Gemma 3
  processor is shortened to ``N``; (2) a forward hook on HydraLLM's multimodal projector applies
  M3 pooling to its output (M3 pools after the projector), so exactly ``N`` image features are
  scattered into the ``N`` placeholders.
* The SigLIP encoder runs once, before the budget decision (:meth:`encode_vision`). During
  ``generate`` the vision tower is temporarily replaced by a stub that returns those cached
  features, so the encoder is not run twice. Both the hook and the stub are removed in ``finally``.

HydraLLM is not a pip package; its repository root is put on ``sys.path`` (see ``from_config``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from ..types import GenerationOptions, OperatingPoints
from .m3 import M3TokenPooler, nested_token_counts

DEFAULT_PROMPT_TEMPLATE = "<start_of_image>\n{prompt}"
DEFAULT_IMAGE_PLACEHOLDER = "<start_of_image>"


@dataclass
class PreparedInputs:
    """Processor output for one request (batch size 1), on the model device."""

    image: Any
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor]
    pixel_values: torch.Tensor


@dataclass(frozen=True)
class BackendOutput:
    text: str
    generated_token_ids: Sequence[int]
    prompt_length: int
    encoder_tokens: int
    actual_visual_tokens: int
    observed_visual_embed_dim: int


def load_image(image: Any) -> Any:
    """Return an RGB ``PIL.Image`` from a path or an image object."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"image must be a path or PIL.Image.Image, got {type(image).__name__}")


def shrink_image_token_runs(
    tensors: Dict[str, Optional[torch.Tensor]],
    image_token_id: int,
    expected_run: int,
    new_run: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """Shorten every run of ``image_token_id`` in ``tensors["input_ids"]`` to ``new_run``.

    All tensors in ``tensors`` must be aligned with ``input_ids`` along dim 1 (attention mask,
    token type ids); ``None`` entries are passed through. Batch size must be 1. Every run must
    have length ``expected_run`` (the processor's ``mm_tokens_per_image``), otherwise the
    prompt/processor layout is not what the adapter assumes and a ``ValueError`` is raised.
    """
    ids = tensors["input_ids"]
    assert ids is not None
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError(f"expected input_ids of shape (1, T), got {tuple(ids.shape)}")
    if not 1 <= new_run <= expected_run:
        raise ValueError(f"new_run must be in [1, {expected_run}], got {new_run}")
    positions = (ids[0] == image_token_id).nonzero().flatten().tolist()
    if not positions:
        raise ValueError("no image placeholder tokens found in input_ids")
    runs, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p != prev + 1:
            runs.append((start, prev + 1))
            start = p
        prev = p
    runs.append((start, prev + 1))
    keep = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
    for s, e in runs:
        if e - s != expected_run:
            raise ValueError(f"image placeholder run of length {e - s}, expected {expected_run}")
        keep[s + new_run : e] = False
    return {k: (v[:, keep] if v is not None else None) for k, v in tensors.items()}


class _CachedVisionTower(nn.Module):
    """Stands in for the SigLIP tower during generate and returns precomputed features."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        super().__init__()
        self._features = last_hidden_state
        self.calls = 0

    def forward(self, pixel_values: Any = None, **_: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self._features)


class HydraM3Backend:
    """Multimodal HydraGemma 3 + M3 pooling, behind the interface the pipeline expects."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        tokenizer: Any,
        device: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
        supported_visual_tokens: Optional[Sequence[int]] = None,
        supported_widths: Optional[Sequence[int]] = None,
    ) -> None:
        core = getattr(model, "model", None)
        if (
            core is None
            or not hasattr(core, "vision_tower")
            or not hasattr(core, "multi_modal_projector")
        ):
            raise TypeError(
                "expected a HydraLLM multimodal model (HydraGemma3ForConditionalGeneration) with "
                ".model.vision_tower and .model.multi_modal_projector"
            )
        self.model: Any = model
        self._core = core
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        if prompt_template.count("{prompt}") != 1:
            raise ValueError("prompt_template must contain '{prompt}' exactly once")
        self.prompt_template = prompt_template
        self.image_placeholder = image_placeholder

        cfg: Any = model.config
        vc, tc = cfg.vision_config, cfg.text_config
        encoder_tokens = int(cfg.mm_tokens_per_image)
        self.image_token_id = int(cfg.image_token_id)
        self.pooler = M3TokenPooler(encoder_tokens)
        num_heads = int(tc.num_attention_heads)
        self._operating_points = OperatingPoints(
            encoder_patches=(int(vc.image_size) // int(vc.patch_size)) ** 2,
            encoder_tokens=encoder_tokens,
            supported_visual_tokens=_validated(
                supported_visual_tokens, nested_token_counts(encoder_tokens), "visual tokens"
            ),
            num_heads=num_heads,
            supported_widths=_validated(
                supported_widths, tuple(range(1, num_heads + 1)), "widths (active heads)"
            ),
            hidden_size=int(tc.hidden_size),
            intermediate_size=int(tc.intermediate_size),
        )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, cfg: Any) -> HydraM3Backend:
        """Load from a :class:`hydram3.config.PipelineConfig` (needs a real checkpoint)."""
        hc = cfg.hydrallm
        if not hc.model_path:
            raise ValueError("hydrallm.model_path is required (multimodal HydraGemma checkpoint)")
        import_hydrallm(hc.repo_path)
        from common_helpers import parse_dtype  # type: ignore[import-not-found]
        from model_implementations.hydra_gemma_from_flex import (  # type: ignore[import-not-found]
            get_mm_model,
        )
        from transformers import AutoProcessor

        dtype = parse_dtype(hc.dtype, hc.device)
        model = get_mm_model(
            hc.model_path, dtype=dtype, attn_implementation=hc.attn_implementation, device=hc.device
        )
        model.to(hc.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(hc.processor_path or hc.model_path)
        return cls(
            model,
            processor,
            processor.tokenizer,
            device=hc.device,
            prompt_template=cfg.prompt_template,
            image_placeholder=cfg.image_placeholder,
            supported_visual_tokens=cfg.budget.supported_visual_tokens,
            supported_widths=cfg.budget.supported_widths,
        )

    # ------------------------------------------------------------------ backend interface

    @property
    def operating_points(self) -> OperatingPoints:
        return self._operating_points

    @property
    def is_async_device(self) -> bool:
        return str(self.device).startswith("cuda")

    def synchronize(self) -> None:
        if self.is_async_device:
            torch.cuda.synchronize(self.device)

    def preprocess(self, image: Any, prompt: str) -> PreparedInputs:
        img = load_image(image)
        text = self.prompt_template.replace("{prompt}", prompt)
        if text.count(self.image_placeholder) != 1:
            raise ValueError(
                f"the formatted prompt must contain '{self.image_placeholder}' exactly once"
            )
        enc = self.processor(images=[img], text=text, return_tensors="pt")
        pixel_values = enc["pixel_values"]
        if pixel_values.shape[0] != 1:
            raise ValueError(
                f"expected 1 image crop, got {pixel_values.shape[0]} (pan-and-scan unsupported)"
            )
        vision_dtype = next(self._core.vision_tower.parameters()).dtype
        tti = enc.get("token_type_ids")
        return PreparedInputs(
            image=img,
            prompt=prompt,
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
            token_type_ids=tti.to(self.device) if tti is not None else None,
            pixel_values=pixel_values.to(self.device, dtype=vision_dtype),
        )

    @torch.inference_mode()
    def encode_vision(self, prepared: PreparedInputs) -> torch.Tensor:
        """SigLIP ``last_hidden_state`` (1, encoder_patches, d_vision); independent of N and W."""
        return self._core.vision_tower(pixel_values=prepared.pixel_values).last_hidden_state

    @torch.inference_mode()
    def generate(
        self,
        prepared: PreparedInputs,
        vision_features: torch.Tensor,
        visual_tokens: int,
        width: int,
        options: GenerationOptions,
    ) -> BackendOutput:
        ops = self._operating_points
        if visual_tokens not in ops.supported_visual_tokens:
            raise ValueError(f"unsupported visual token count {visual_tokens}")
        if width not in ops.supported_widths:
            raise ValueError(f"unsupported width {width}")

        inputs = shrink_image_token_runs(
            {
                "input_ids": prepared.input_ids,
                "attention_mask": prepared.attention_mask,
                "token_type_ids": prepared.token_type_ids,
            },
            self.image_token_id,
            ops.encoder_tokens,
            visual_tokens,
        )
        record: Dict[str, int] = {"calls": 0}

        def pool_hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
            record["calls"] += 1
            record["encoder_tokens"] = int(output.shape[1])
            pooled = self.pooler.pool(output, visual_tokens)
            record["retained_tokens"] = int(pooled.shape[1])
            record["embed_dim"] = int(pooled.shape[-1])
            return pooled

        gen_kwargs = options.to_generate_kwargs()
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            pad = self.tokenizer.pad_token_id
            gen_kwargs["pad_token_id"] = pad if pad is not None else self.tokenizer.eos_token_id

        original_tower = self._core.vision_tower
        stub = _CachedVisionTower(vision_features)
        handle = self._core.multi_modal_projector.register_forward_hook(pool_hook)
        self._core.vision_tower = stub
        try:
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                pixel_values=prepared.pixel_values,
                current_num_heads=int(width),
                **gen_kwargs,
            )
        finally:
            handle.remove()
            self._core.vision_tower = original_tower

        if record["calls"] != 1 or stub.calls != 1:
            raise RuntimeError(
                f"expected one projector call and one (cached) vision call during generate, got "
                f"projector={record['calls']} vision={stub.calls}"
            )
        final_ids = inputs["input_ids"]
        assert final_ids is not None
        prompt_length = int(final_ids.shape[1])
        new_ids = out[0, prompt_length:].tolist()
        return BackendOutput(
            text=self.tokenizer.decode(new_ids, skip_special_tokens=True),
            generated_token_ids=new_ids,
            prompt_length=prompt_length,
            encoder_tokens=record["encoder_tokens"],
            actual_visual_tokens=record["retained_tokens"],
            observed_visual_embed_dim=record["embed_dim"],
        )


def import_hydrallm(repo_path: Any) -> Path:
    """Put the HydraLLM repository root on ``sys.path`` so its top-level modules import."""
    root = Path(repo_path).expanduser().resolve()
    if not (root / "model_implementations").is_dir() or not (root / "common_helpers").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like a HydraLLM checkout (model_implementations/ and "
            "common_helpers/ not found); set hydrallm.repo_path"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _validated(configured: Optional[Sequence[int]], achievable: Sequence[int], what: str) -> tuple:
    if configured is None:
        return tuple(achievable)
    values = tuple(sorted({int(v) for v in configured}))
    bad = [v for v in values if v not in achievable]
    if not values or bad:
        raise ValueError(
            f"configured supported {what} {list(configured)} invalid for this checkpoint; "
            f"achievable values: {list(achievable)}"
        )
    return values
