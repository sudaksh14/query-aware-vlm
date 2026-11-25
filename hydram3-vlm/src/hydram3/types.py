"""Request, budget and result types shared by the pipeline, the budget providers and the adapters.

Vocabulary used throughout the package:

* ``N`` (visual-token budget): number of visual tokens per image that the LLM consumes after
  M3-style nested average pooling.
* ``W`` (width budget): HydraLLM's native width control, the number of active attention heads
  ``k`` (``current_num_heads``). HydraLLM derives the embedding, FFN and projector output widths
  from ``k / num_attention_heads``; see :class:`WidthOperatingPoint`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GenerationOptions:
    """Decoding settings forwarded to ``model.generate``. ``None`` fields are not forwarded."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None

    def to_generate_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": bool(self.do_sample),
        }
        optional = {
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        if self.do_sample:  # sampling knobs are ignored (and warned about) by HF under greedy
            optional.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        return kwargs


@dataclass(frozen=True)
class InferenceRequest:
    """One image plus one prompt.

    ``image`` is a path or a ``PIL.Image.Image``. ``prompt`` is the user text; the configured
    prompt template adds HydraLLM's image placeholder. ``generation=None``: pipeline default.
    """

    image: Any
    prompt: str
    generation: Optional[GenerationOptions] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class BudgetDecision:
    """Raw ``(N, W)`` pair produced by a budget provider, before validation/resolution."""

    visual_tokens: int
    width: int
    source: str


@dataclass(frozen=True)
class WidthOperatingPoint:
    """HydraLLM dimensions implied by ``heads`` active attention heads (mirrors flex_gemma.py)."""

    heads: int
    num_heads_total: int
    ratio: float
    embed_dim: int
    ffn_dim: int
    projector_out_dim: int


@dataclass(frozen=True)
class OperatingPoints:
    """Supported operating points of a loaded backend.

    ``encoder_patches``: vision-tower sequence length (SigLIP patches per image).
    ``encoder_tokens``: visual tokens per image produced by HydraLLM's projector, i.e. before
    M3 pooling. This is the "full" token count (N_max).
    """

    encoder_patches: int
    encoder_tokens: int
    supported_visual_tokens: Tuple[int, ...]
    num_heads: int
    supported_widths: Tuple[int, ...]
    hidden_size: int
    intermediate_size: int

    def width_point(self, heads: int) -> WidthOperatingPoint:
        # Same arithmetic as FlexGemma3ForConditionalGeneration.forward in HydraLLM:
        # head_ratio = k / H; dim = safe_int(full_dim * head_ratio) with safe_int = max(1, int(x)).
        ratio = float(heads) / float(self.num_heads)
        embed = max(1, int(self.hidden_size * ratio))
        return WidthOperatingPoint(
            heads=heads,
            num_heads_total=self.num_heads,
            ratio=ratio,
            embed_dim=embed,
            ffn_dim=max(1, int(self.intermediate_size * ratio)),
            projector_out_dim=embed,
        )


@dataclass(frozen=True)
class ResolvedBudget:
    """Requested and resolved budgets. Resolved values are what the upstream models actually use."""

    requested_visual_tokens: int
    requested_width: int
    visual_tokens: int
    width: int
    source: str
    policy: str

    @property
    def visual_tokens_adjusted(self) -> bool:
        return self.visual_tokens != self.requested_visual_tokens

    @property
    def width_adjusted(self) -> bool:
        return self.width != self.requested_width


@dataclass(frozen=True)
class StageTiming:
    """Wall-clock duration of one stage, measured with ``time.perf_counter`` (monotonic).

    ``device_synchronized`` is True when the accelerator was synchronized before the start and the
    end of the stage, so queued asynchronous kernels are counted in the stage that launched them.
    ``includes`` states in words what the interval covers.
    """

    name: str
    seconds: float
    device_synchronized: bool
    includes: str


@dataclass(frozen=True)
class InferenceResult:
    """Pipeline output: generated text and non-sensitive metadata (no image or prompt content)."""

    request_id: Optional[str]
    text: str
    budget: ResolvedBudget
    width_point: WidthOperatingPoint
    encoder_patches: int
    encoder_tokens: int
    actual_visual_tokens: int
    observed_visual_embed_dim: int
    prompt_length: int
    generated_tokens: int
    timings: Tuple[StageTiming, ...]
    first_call: bool
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["budget"]["visual_tokens_adjusted"] = self.budget.visual_tokens_adjusted
        out["budget"]["width_adjusted"] = self.budget.width_adjusted
        return out
