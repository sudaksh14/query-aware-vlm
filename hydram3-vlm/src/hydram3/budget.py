"""Budget providers, the QBSM extension point, and budget resolution.

Control point: the pipeline calls a provider **after** the vision encoder has run once and
**before** the projector, M3 pooling and the LLM (decision point D1 of the research plan).
Encoder features are therefore available to the provider at no extra cost, and nothing the
provider returns can feed back into the encoder (no circular dependency).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, Tuple, Union, runtime_checkable

from .types import BudgetDecision, InferenceRequest, OperatingPoints, ResolvedBudget

RESOLUTION_POLICY = "floor"


class BudgetError(ValueError):
    """A budget is malformed or cannot be mapped to a supported operating point."""


@dataclass(frozen=True)
class DecisionContext:
    """Everything available at decision time.

    ``image`` is the RGB ``PIL.Image``; ``pixel_values`` the processor output fed to SigLIP;
    ``vision_features`` the SigLIP ``last_hidden_state`` of shape ``(1, encoder_patches, d_vision)``
    already computed for this request (it is reused by the LLM pass, not recomputed).
    """

    request: InferenceRequest
    image: Any
    prompt: str
    pixel_values: Any
    vision_features: Any
    operating_points: OperatingPoints


@runtime_checkable
class BudgetProvider(Protocol):
    """Returns the logical pair ``(N, W)`` for one request."""

    name: str

    def decide(self, context: DecisionContext) -> BudgetDecision: ...


class FixedBudgetProvider:
    """Returns the same configured ``(N, W)`` for every request. Not query-aware, not learned."""

    name = "fixed"

    def __init__(self, visual_tokens: int, width: int) -> None:
        self.visual_tokens = _as_int(visual_tokens, "visual_tokens")
        self.width = _as_int(width, "width")

    def decide(self, context: DecisionContext) -> BudgetDecision:
        return BudgetDecision(self.visual_tokens, self.width, source=self.name)


# --------------------------------------------------------------------------- QBSM seam


@dataclass(frozen=True)
class QBSMInputs:
    """Inputs handed to the student's Query Budget Selection Module.

    ``vision_features``: SigLIP ``last_hidden_state``, ``(1, encoder_patches, d_vision)`` torch
    tensor on the model device and dtype. ``supported_*`` list the valid operating points; a
    predictor may return other values, which are then resolved with the ``floor`` policy.
    """

    image: Any
    prompt: str
    vision_features: Any
    supported_visual_tokens: Tuple[int, ...]
    supported_widths: Tuple[int, ...]


class QBSMPredictor(Protocol):
    def predict(self, inputs: QBSMInputs) -> Tuple[int, int]: ...


PredictorLike = Union[QBSMPredictor, Callable[[QBSMInputs], Sequence[Any]]]


class QBSMBudgetProviderAdapter:
    """Wraps a student predictor returning ``(N, W)`` as a :class:`BudgetProvider`.

    Accepts an object with ``predict(inputs)`` or a plain callable. The output must be a pair of
    integers (Python, NumPy or 0-d torch integers); anything else raises :class:`BudgetError`.
    """

    def __init__(self, predictor: PredictorLike, name: str = "qbsm") -> None:
        self.predictor = predictor
        self.name = name

    def decide(self, context: DecisionContext) -> BudgetDecision:
        ops = context.operating_points
        inputs = QBSMInputs(
            image=context.image,
            prompt=context.prompt,
            vision_features=context.vision_features,
            supported_visual_tokens=ops.supported_visual_tokens,
            supported_widths=ops.supported_widths,
        )
        predictor: Any = self.predictor
        predict: Callable[[QBSMInputs], Any] = getattr(predictor, "predict", predictor)
        out = predict(inputs)
        if isinstance(out, (str, bytes)) or not hasattr(out, "__len__") or len(out) != 2:
            raise BudgetError(f"{self.name} must return a pair (N, W), got {out!r}")
        n, w = out
        return BudgetDecision(_as_int(n, "N"), _as_int(w, "W"), source=self.name)


# --------------------------------------------------------------------------- resolution


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool):
        raise BudgetError(f"{what} must be an integer, got bool {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        pass
    item = getattr(value, "item", None)  # 0-d torch / numpy scalars
    if callable(item) and getattr(value, "ndim", 1) == 0:
        return _as_int(item(), what)
    raise BudgetError(f"{what} must be an integer, got {type(value).__name__} {value!r}")


def _floor(value: int, supported: Sequence[int], what: str) -> int:
    if value < 1:
        raise BudgetError(f"{what}={value} must be >= 1")
    candidates = [s for s in supported if s <= value]
    if not candidates:
        raise BudgetError(f"{what}={value} is below the smallest supported value {min(supported)}")
    return max(candidates)


def resolve_budget(decision: BudgetDecision, ops: OperatingPoints) -> ResolvedBudget:
    """Map a raw decision onto supported operating points.

    Policy ``floor``: each value becomes the largest supported operating point that does not
    exceed it, so the budget is an upper bound that is never exceeded. Values below the smallest
    supported point, non-integers and values < 1 raise :class:`BudgetError`. Requested and
    resolved values are both kept in the returned :class:`ResolvedBudget`.
    """
    n = _as_int(decision.visual_tokens, "N")
    w = _as_int(decision.width, "W")
    return ResolvedBudget(
        requested_visual_tokens=n,
        requested_width=w,
        visual_tokens=_floor(n, ops.supported_visual_tokens, "N (visual tokens)"),
        width=_floor(w, ops.supported_widths, "W (active heads)"),
        source=decision.source,
        policy=RESOLUTION_POLICY,
    )
