from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hydram3 import (
    ElasticVLMPipeline,
    FixedBudgetProvider,
    GenerationOptions,
    InferenceRequest,
    OperatingPoints,
    PipelineStageError,
    QBSMBudgetProviderAdapter,
)
from hydram3.budget import BudgetError

OPS = OperatingPoints(4096, 256, (1, 4, 16, 64, 256), 4, (1, 2, 3, 4), 640, 2048)


@dataclass
class Out:
    text: str
    generated_token_ids: list
    prompt_length: int
    encoder_tokens: int
    actual_visual_tokens: int
    observed_visual_embed_dim: int


class FakeBackend:
    def __init__(self, async_device=False, lie_tokens=None, fail_stage=None):
        self.calls = []
        self.async_device = async_device
        self.lie_tokens = lie_tokens
        self.fail_stage = fail_stage
        self.syncs = 0

    operating_points = property(lambda self: OPS)
    is_async_device = property(lambda self: self.async_device)

    def synchronize(self):
        self.syncs += 1

    def preprocess(self, image, prompt):
        self.calls.append(("preprocess", image, prompt))
        if self.fail_stage == "preprocess":
            raise FileNotFoundError("no such image")
        return SimpleNamespace(image="PIL", pixel_values="pv")

    def encode_vision(self, prepared):
        self.calls.append(("encode_vision",))
        return "vision-features"

    def generate(self, prepared, vision, n, w, options):
        self.calls.append(("generate", vision, n, w, options))
        if self.fail_stage == "generate":
            raise RuntimeError("CUDA out of memory")
        n_used = self.lie_tokens if self.lie_tokens is not None else n
        return Out("answer", [5, 6, 7], 20 + n, 256, n_used, OPS.width_point(w).embed_dim)


def test_order_values_and_metadata():
    backend = FakeBackend()
    seen = []

    def student(inputs):
        seen.append(inputs)
        return (100, 3)  # resolved by floor to (64, 3)

    pipe = ElasticVLMPipeline(backend, QBSMBudgetProviderAdapter(student))
    req = InferenceRequest("img.jpg", "what is this?", request_id="r1")
    res = pipe.run(req)

    assert [c[0] for c in backend.calls] == ["preprocess", "encode_vision", "generate"]
    assert seen[0].vision_features == "vision-features"  # provider runs after the encoder
    assert seen[0].image == "PIL" and seen[0].prompt == "what is this?"
    _, vision, n, w, options = backend.calls[2]
    assert (vision, n, w) == ("vision-features", 64, 3)
    assert options == GenerationOptions()

    b = res.budget
    assert (b.requested_visual_tokens, b.requested_width, b.visual_tokens, b.width) == (
        100,
        3,
        64,
        3,
    )
    assert b.source == "qbsm" and b.visual_tokens_adjusted and not b.width_adjusted
    assert res.actual_visual_tokens == 64 and res.encoder_tokens == 256
    assert res.width_point.embed_dim == 480 and res.observed_visual_embed_dim == 480
    assert res.generated_tokens == 3 and res.prompt_length == 84
    assert res.request_id == "r1" and res.first_call
    assert [t.name for t in res.timings] == [
        "preprocess",
        "vision_encoder",
        "budget_decision",
        "generate",
    ]
    assert not any(t.device_synchronized for t in res.timings)
    d = res.to_dict()
    assert d["budget"]["visual_tokens_adjusted"] is True
    assert "what is this" not in str(d)  # no prompt content in metadata
    assert not pipe.run(req).first_call


def test_request_generation_overrides_default():
    backend = FakeBackend()
    pipe = ElasticVLMPipeline(backend, FixedBudgetProvider(16, 2), GenerationOptions(8))
    pipe.run(InferenceRequest("i", "p", generation=GenerationOptions(max_new_tokens=3)))
    pipe.run(InferenceRequest("i", "p"))
    assert backend.calls[2][4].max_new_tokens == 3
    assert backend.calls[5][4].max_new_tokens == 8


def test_async_device_stages_are_synchronized():
    backend = FakeBackend(async_device=True)
    res = ElasticVLMPipeline(backend, FixedBudgetProvider(16, 2)).run(InferenceRequest("i", "p"))
    flags = {t.name: t.device_synchronized for t in res.timings}
    assert flags == {
        "preprocess": False,
        "vision_encoder": True,
        "budget_decision": True,
        "generate": True,
    }
    assert backend.syncs == 6


def test_budget_error_stops_before_generate():
    backend = FakeBackend()
    pipe = ElasticVLMPipeline(backend, QBSMBudgetProviderAdapter(lambda _: (0, 4)))
    with pytest.raises(PipelineStageError) as ei:
        pipe.run(InferenceRequest("i", "p", request_id="r9"))
    assert ei.value.stage == "budget_decision" and ei.value.request_id == "r9"
    assert isinstance(ei.value.__cause__, BudgetError)
    assert "generate" not in [c[0] for c in backend.calls]


@pytest.mark.parametrize(
    "stage,exc", [("preprocess", FileNotFoundError), ("generate", RuntimeError)]
)
def test_backend_errors_carry_stage(stage, exc):
    pipe = ElasticVLMPipeline(FakeBackend(fail_stage=stage), FixedBudgetProvider(16, 2))
    with pytest.raises(PipelineStageError) as ei:
        pipe.run(InferenceRequest("i", "p"))
    assert ei.value.stage == stage and isinstance(ei.value.__cause__, exc)


def test_mismatched_token_count_is_not_reported_as_success():
    pipe = ElasticVLMPipeline(FakeBackend(lie_tokens=256), FixedBudgetProvider(16, 2))
    with pytest.raises(PipelineStageError, match="consumed 256 visual tokens"):
        pipe.run(InferenceRequest("i", "p"))
