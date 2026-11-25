"""Pipeline orchestrator: preprocess, vision encoder, budget decision, M3 + HydraLLM generate."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .budget import BudgetProvider, DecisionContext, resolve_budget
from .timing import StageTimer
from .types import GenerationOptions, InferenceRequest, InferenceResult, OperatingPoints


class PipelineStageError(RuntimeError):
    """Raised when a stage fails; ``stage`` names it and ``__cause__`` holds the original error."""

    def __init__(self, stage: str, request_id: Optional[str], message: str) -> None:
        super().__init__(f"[{stage}] request_id={request_id}: {message}")
        self.stage = stage
        self.request_id = request_id


class VLMBackend(Protocol):
    """What the orchestrator needs from a backend (implemented by ``HydraM3Backend``)."""

    @property
    def operating_points(self) -> OperatingPoints: ...

    @property
    def is_async_device(self) -> bool: ...

    def synchronize(self) -> None: ...

    def preprocess(self, image: Any, prompt: str) -> Any: ...

    def encode_vision(self, prepared: Any) -> Any: ...

    def generate(
        self,
        prepared: Any,
        vision_features: Any,
        visual_tokens: int,
        width: int,
        options: GenerationOptions,
    ) -> Any: ...


class ElasticVLMPipeline:
    """Runs one request end to end and reports what was actually executed.

    Order (fixed; the provider cannot influence earlier stages):
      1. ``preprocess``      processor + placeholder expansion (CPU) and host-to-device copy
      2. ``vision_encoder``  SigLIP forward, once per request
      3. ``budget_decision`` provider.decide + resolve_budget (controller overhead)
      4. ``generate``        projector at width W, M3 pooling to N, prefill and decode at width W
    """

    def __init__(
        self,
        backend: VLMBackend,
        provider: BudgetProvider,
        default_generation: Optional[GenerationOptions] = None,
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.default_generation = default_generation or GenerationOptions()
        self._calls = 0

    def run(self, request: InferenceRequest) -> InferenceResult:
        rid = request.request_id
        timer = StageTimer(self.backend.synchronize if self.backend.is_async_device else None)
        ops = self.backend.operating_points
        first_call = self._calls == 0
        self._calls += 1

        def fail(stage: str, exc: BaseException) -> PipelineStageError:
            return PipelineStageError(stage, rid, f"{type(exc).__name__}: {exc}")

        try:
            with timer.stage(
                "preprocess", "image load, prompt template, processor, copy to device"
            ):
                prepared = self.backend.preprocess(request.image, request.prompt)
        except Exception as exc:
            raise fail("preprocess", exc) from exc

        try:
            with timer.stage("vision_encoder", "vision tower forward only", sync=True):
                vision = self.backend.encode_vision(prepared)
        except Exception as exc:
            raise fail("vision_encoder", exc) from exc

        try:
            with timer.stage("budget_decision", "provider.decide + resolve_budget", sync=True):
                context = DecisionContext(
                    request=request,
                    image=getattr(prepared, "image", request.image),
                    prompt=request.prompt,
                    pixel_values=getattr(prepared, "pixel_values", None),
                    vision_features=vision,
                    operating_points=ops,
                )
                decision = self.provider.decide(context)
                budget = resolve_budget(decision, ops)
        except Exception as exc:
            raise fail("budget_decision", exc) from exc

        options = request.generation or self.default_generation
        try:
            with timer.stage(
                "generate",
                "projector + M3 pooling + LLM prefill + decode (vision features reused, not rerun)",
                sync=True,
            ):
                out = self.backend.generate(
                    prepared, vision, budget.visual_tokens, budget.width, options
                )
        except Exception as exc:
            raise fail("generate", exc) from exc

        width_point = ops.width_point(budget.width)
        # Never report a budget the model did not use.
        if out.actual_visual_tokens != budget.visual_tokens:
            raise PipelineStageError(
                "generate",
                rid,
                f"backend consumed {out.actual_visual_tokens} visual tokens, resolved budget "
                f"was {budget.visual_tokens}",
            )
        if out.observed_visual_embed_dim != width_point.projector_out_dim:
            raise PipelineStageError(
                "generate",
                rid,
                f"visual token width {out.observed_visual_embed_dim} != expected "
                f"{width_point.projector_out_dim} for {budget.width} heads",
            )

        return InferenceResult(
            request_id=rid,
            text=out.text,
            budget=budget,
            width_point=width_point,
            encoder_patches=ops.encoder_patches,
            encoder_tokens=out.encoder_tokens,
            actual_visual_tokens=out.actual_visual_tokens,
            observed_visual_embed_dim=out.observed_visual_embed_dim,
            prompt_length=out.prompt_length,
            generated_tokens=len(out.generated_token_ids),
            timings=tuple(timer.records),
            first_call=first_call,
        )
