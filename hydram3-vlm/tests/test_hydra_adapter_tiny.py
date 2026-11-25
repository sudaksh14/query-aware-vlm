"""HydraLLM adapter tests on HydraLLM's real model classes with a tiny random-weight config.

Nothing is downloaded: the Gemma 3 + SigLIP config is built in code. Skips when transformers or
the HydraLLM checkout is unavailable. Outputs are meaningless text; these tests check plumbing.
"""

from types import SimpleNamespace

import pytest
import torch

from hydram3 import ElasticVLMPipeline, FixedBudgetProvider, GenerationOptions, InferenceRequest
from hydram3.adapters.hydra import (
    HydraM3Backend,
    PreparedInputs,
    import_hydrallm,
    shrink_image_token_runs,
)
from hydram3.adapters.m3 import matryoshka_pool

pytestmark = pytest.mark.hydrallm

VOCAB, BOI, EOI, IMG = 300, 297, 298, 299
GRID_TOKENS = 16  # 4x4 visual tokens after HydraLLM's projector pooling
HEADS = 4


def tiny_model(hydrallm_repo):
    import_hydrallm(hydrallm_repo)
    from model_implementations.hydra_gemma_from_flex import HydraGemma3ForConditionalGeneration
    from transformers import Gemma3Config

    cfg = Gemma3Config(
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=HEADS,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=256,
            sliding_window=64,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=64,
            patch_size=4,  # 16x16 = 256 patches, pooled by HydraLLM's projector to 4x4
        ),
        mm_tokens_per_image=GRID_TOKENS,
        boi_token_index=BOI,
        eoi_token_index=EOI,
        image_token_index=IMG,
    )
    cfg.text_config.attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = HydraGemma3ForConditionalGeneration(cfg).eval()
    with torch.no_grad():  # random but non-zero projector (its init is zeros)
        model.model.multi_modal_projector.mm_input_projection_weight.normal_(0, 0.5)
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    return model


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class FakeProcessor:
    """Mimics Gemma3Processor output layout: text, BOI, mm_tokens_per_image x IMG, EOI, text."""

    def __call__(self, images, text, return_tensors):
        ids = [2, 10, 11, BOI] + [IMG] * GRID_TOKENS + [EOI, 12, 13, 14]
        t = torch.tensor([ids])
        return {
            "input_ids": t,
            "attention_mask": torch.ones_like(t),
            "token_type_ids": (t == IMG).long(),
            "pixel_values": torch.randn(len(images), 3, 64, 64),
        }


def make_backend(model):
    return HydraM3Backend(model, FakeProcessor(), FakeTokenizer(), device="cpu")


def prepared_for(backend):
    from PIL import Image

    return backend.preprocess(Image.new("RGB", (32, 32)), "describe")


def test_operating_points_from_checkpoint_config(hydrallm_repo):
    ops = make_backend(tiny_model(hydrallm_repo)).operating_points
    assert ops.encoder_patches == 256 and ops.encoder_tokens == GRID_TOKENS
    assert ops.supported_visual_tokens == (1, 4, 16)
    assert ops.supported_widths == (1, 2, 3, 4)


def test_configured_points_are_validated(hydrallm_repo):
    model = tiny_model(hydrallm_repo)
    with pytest.raises(ValueError, match="achievable"):
        HydraM3Backend(model, FakeProcessor(), FakeTokenizer(), "cpu", supported_visual_tokens=[9])
    with pytest.raises(ValueError, match="achievable"):
        HydraM3Backend(model, FakeProcessor(), FakeTokenizer(), "cpu", supported_widths=[5])


def test_full_budget_is_identical_to_plain_hydrallm_generate(hydrallm_repo):
    """At N = full grid and W = all heads, the stub + hook must not change HydraLLM's output."""
    model = tiny_model(hydrallm_repo)
    backend = make_backend(model)
    prep = prepared_for(backend)
    opts = GenerationOptions(max_new_tokens=5)
    out = backend.generate(prep, backend.encode_vision(prep), GRID_TOKENS, HEADS, opts)

    with torch.inference_mode():
        ref = model.generate(
            input_ids=prep.input_ids,
            attention_mask=prep.attention_mask,
            token_type_ids=prep.token_type_ids,
            pixel_values=prep.pixel_values,
            current_num_heads=HEADS,
            max_new_tokens=5,
            do_sample=False,
        )
    assert list(out.generated_token_ids) == ref[0, prep.input_ids.shape[1] :].tolist()
    assert out.actual_visual_tokens == GRID_TOKENS == out.encoder_tokens
    assert out.observed_visual_embed_dim == 64


@pytest.mark.parametrize("n,w", [(4, 2), (1, 1), (16, 3)])
def test_reduced_budget_reaches_both_upstreams(hydrallm_repo, n, w):
    model = tiny_model(hydrallm_repo)
    backend = make_backend(model)
    prep = prepared_for(backend)
    tower = model.model.vision_tower
    out = backend.generate(prep, backend.encode_vision(prep), n, w, GenerationOptions(3))
    assert out.actual_visual_tokens == n and out.encoder_tokens == GRID_TOKENS
    assert out.observed_visual_embed_dim == 64 * w // HEADS
    assert out.prompt_length == prep.input_ids.shape[1] - (GRID_TOKENS - n)
    assert len(out.generated_token_ids) == 3
    # Everything restored: real tower back in place, no hooks left on the projector.
    assert model.model.vision_tower is tower
    assert not model.model.multi_modal_projector._forward_hooks


def test_hooked_features_equal_m3_pooling_of_hydrallm_projector(hydrallm_repo):
    model = tiny_model(hydrallm_repo)
    backend = make_backend(model)
    prep = prepared_for(backend)
    vision = backend.encode_vision(prep)
    captured = {}

    def capture(_m, _a, out):  # must return None, or the hook replaces the module output
        if out.image_hidden_states is not None:
            captured.setdefault("img", out.image_hidden_states)

    handle = model.model.register_forward_hook(capture)
    try:
        backend.generate(prep, vision, 4, 2, GenerationOptions(1))
    finally:
        handle.remove()
    with torch.inference_mode():
        full = model.model.multi_modal_projector(vision, out_cols=32, final_embed_dim=32)
    torch.testing.assert_close(captured["img"], matryoshka_pool(full, 4))


def test_vision_tower_runs_once_per_request(hydrallm_repo):
    model = tiny_model(hydrallm_repo)
    calls = []
    model.model.vision_tower.register_forward_hook(lambda *a: calls.append(1))
    pipe = ElasticVLMPipeline(make_backend(model), FixedBudgetProvider(4, 2), GenerationOptions(2))
    from PIL import Image

    res = pipe.run(InferenceRequest(Image.new("RGB", (8, 8)), "p", request_id="t"))
    assert len(calls) == 1
    assert res.actual_visual_tokens == 4 and res.width_point.embed_dim == 32


def test_shrink_image_token_runs():
    ids = torch.tensor([[1, IMG, IMG, IMG, IMG, 2, IMG, IMG, IMG, IMG, 3]])
    out = shrink_image_token_runs(
        {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": None},
        IMG,
        4,
        1,
    )
    assert out["input_ids"].tolist() == [[1, IMG, 2, IMG, 3]]
    assert out["attention_mask"].shape == (1, 5) and out["token_type_ids"] is None
    with pytest.raises(ValueError, match="expected 3"):
        shrink_image_token_runs({"input_ids": ids}, IMG, 3, 1)
    with pytest.raises(ValueError, match="no image placeholder"):
        shrink_image_token_runs({"input_ids": torch.tensor([[1, 2]])}, IMG, 4, 1)


def test_prompt_placeholder_checked(hydrallm_repo):
    backend = HydraM3Backend(
        tiny_model(hydrallm_repo),
        FakeProcessor(),
        FakeTokenizer(),
        "cpu",
        prompt_template="{prompt}",
    )
    from PIL import Image

    with pytest.raises(ValueError, match="exactly once"):
        backend.preprocess(Image.new("RGB", (8, 8)), "no placeholder here")


def test_rejects_non_hydra_model():
    with pytest.raises(TypeError, match="HydraLLM multimodal model"):
        HydraM3Backend(SimpleNamespace(config=None), None, None, "cpu")


def test_prepared_inputs_dataclass_fields():
    assert set(PreparedInputs.__dataclass_fields__) == {
        "image",
        "prompt",
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "pixel_values",
    }
