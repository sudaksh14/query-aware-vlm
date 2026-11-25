"""hydram3: joint visual-token (M3) and LLM-width (HydraLLM) budgets for one VLM inference path."""

from .budget import (
    BudgetError,
    BudgetProvider,
    DecisionContext,
    FixedBudgetProvider,
    QBSMBudgetProviderAdapter,
    QBSMInputs,
    QBSMPredictor,
    resolve_budget,
)
from .pipeline import ElasticVLMPipeline, PipelineStageError
from .types import (
    BudgetDecision,
    GenerationOptions,
    InferenceRequest,
    InferenceResult,
    OperatingPoints,
    ResolvedBudget,
    StageTiming,
    WidthOperatingPoint,
)

__all__ = [
    "BudgetDecision",
    "BudgetError",
    "BudgetProvider",
    "DecisionContext",
    "ElasticVLMPipeline",
    "FixedBudgetProvider",
    "GenerationOptions",
    "InferenceRequest",
    "InferenceResult",
    "OperatingPoints",
    "PipelineStageError",
    "QBSMBudgetProviderAdapter",
    "QBSMInputs",
    "QBSMPredictor",
    "ResolvedBudget",
    "StageTiming",
    "WidthOperatingPoint",
    "resolve_budget",
]
__version__ = "0.1.0"
