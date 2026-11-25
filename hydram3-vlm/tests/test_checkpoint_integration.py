"""End-to-end run on a real multimodal HydraGemma checkpoint. Opt-in: set

  HYDRAM3_CONFIG=configs/hydra_gemma3_270m_mm.yaml   (with hydrallm.model_path resolvable)
  HYDRAM3_TEST_IMAGE=/path/to/image.jpg

Checks plumbing and metadata at the smallest and largest operating points, not answer quality.
"""

import os

import pytest

pytestmark = pytest.mark.checkpoint

CONFIG = os.environ.get("HYDRAM3_CONFIG")
IMAGE = os.environ.get("HYDRAM3_TEST_IMAGE")


@pytest.fixture(scope="module")
def backend():
    if not CONFIG or not IMAGE:
        pytest.skip("set HYDRAM3_CONFIG and HYDRAM3_TEST_IMAGE to run checkpoint tests")
    pytest.importorskip("transformers")
    from hydram3.adapters.hydra import HydraM3Backend
    from hydram3.config import load_config

    return HydraM3Backend.from_config(load_config(CONFIG))


def test_extreme_operating_points(backend):
    from hydram3 import ElasticVLMPipeline, FixedBudgetProvider, GenerationOptions, InferenceRequest

    ops = backend.operating_points
    for n in (min(ops.supported_visual_tokens), max(ops.supported_visual_tokens)):
        for w in (min(ops.supported_widths), max(ops.supported_widths)):
            pipe = ElasticVLMPipeline(backend, FixedBudgetProvider(n, w), GenerationOptions(8))
            res = pipe.run(InferenceRequest(IMAGE, "Describe this image.", request_id=f"{n}-{w}"))
            assert res.actual_visual_tokens == n
            assert res.width_point.heads == w
            assert res.observed_visual_embed_dim == res.width_point.projector_out_dim
            assert 0 < res.generated_tokens <= 8
