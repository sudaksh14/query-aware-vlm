import pytest
import torch

from hydram3 import (
    BudgetDecision,
    BudgetError,
    FixedBudgetProvider,
    OperatingPoints,
    QBSMBudgetProviderAdapter,
    resolve_budget,
)
from hydram3.budget import DecisionContext
from hydram3.types import InferenceRequest

OPS = OperatingPoints(
    encoder_patches=4096,
    encoder_tokens=256,
    supported_visual_tokens=(1, 4, 16, 64, 256),
    num_heads=4,
    supported_widths=(1, 2, 3, 4),
    hidden_size=640,
    intermediate_size=2048,
)


def ctx(ops=OPS):
    return DecisionContext(InferenceRequest("img", "q"), "img", "q", None, "feats", ops)


@pytest.mark.parametrize(
    "n,w,exp_n,exp_w",
    [(256, 4, 256, 4), (100, 3, 64, 3), (63, 4, 16, 4), (1, 1, 1, 1), (10_000, 9, 256, 4)],
)
def test_floor_policy(n, w, exp_n, exp_w):
    r = resolve_budget(BudgetDecision(n, w, "t"), OPS)
    assert (r.visual_tokens, r.width) == (exp_n, exp_w)
    assert (r.requested_visual_tokens, r.requested_width) == (n, w)
    assert r.visual_tokens_adjusted == (n != exp_n)
    assert r.width_adjusted == (w != exp_w)
    assert r.policy == "floor"


def test_floor_with_sparse_supported_set():
    ops = OperatingPoints(4096, 256, (16, 64), 4, (2, 4), 640, 2048)
    r = resolve_budget(BudgetDecision(255, 3, "t"), ops)
    assert (r.visual_tokens, r.width) == (64, 2)
    with pytest.raises(BudgetError, match="below the smallest"):
        resolve_budget(BudgetDecision(8, 4, "t"), ops)
    with pytest.raises(BudgetError, match="below the smallest"):
        resolve_budget(BudgetDecision(64, 1, "t"), ops)


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_rejected(bad):
    with pytest.raises(BudgetError, match=">= 1"):
        resolve_budget(BudgetDecision(bad, 4, "t"), OPS)


@pytest.mark.parametrize("bad", [16.0, 0.5, True, "16", None])
def test_non_integer_rejected(bad):
    with pytest.raises(BudgetError, match="integer"):
        resolve_budget(BudgetDecision(bad, 4, "t"), OPS)


def test_width_point_matches_hydrallm_arithmetic():
    p = OPS.width_point(3)
    assert p.ratio == 0.75
    assert p.embed_dim == int(640 * 0.75) == 480
    assert p.ffn_dim == int(2048 * 0.75) == 1536
    assert p.projector_out_dim == p.embed_dim


def test_fixed_provider_is_constant():
    prov = FixedBudgetProvider(64, 2)
    assert prov.decide(ctx()) == BudgetDecision(64, 2, "fixed")
    with pytest.raises(BudgetError):
        FixedBudgetProvider(64.0, 2)


def test_qbsm_adapter_accepts_tuple_list_and_tensor_scalars():
    seen = {}

    def predictor(inputs):
        seen["inputs"] = inputs
        return torch.tensor(64), [3][0]

    d = QBSMBudgetProviderAdapter(predictor).decide(ctx())
    assert d == BudgetDecision(64, 3, "qbsm")
    inp = seen["inputs"]
    assert inp.vision_features == "feats" and inp.prompt == "q"
    assert inp.supported_visual_tokens == OPS.supported_visual_tokens
    assert inp.supported_widths == OPS.supported_widths


def test_qbsm_adapter_object_with_predict():
    class Student:
        def predict(self, inputs):
            return (16, 4)

    assert QBSMBudgetProviderAdapter(Student(), name="s").decide(ctx()) == BudgetDecision(
        16, 4, "s"
    )


@pytest.mark.parametrize("out", [64, (64,), (64, 2, 1), "64,2", (64.5, 2), (64, torch.ones(2))])
def test_qbsm_adapter_rejects_malformed_output(out):
    with pytest.raises(BudgetError):
        QBSMBudgetProviderAdapter(lambda _: out).decide(ctx())
