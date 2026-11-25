"""Numerical parity with M3's own implementation. Runs only where `llava` (matryoshka-mm) imports,
e.g. in an environment built from matryoshka-mm/pyproject.toml with matryoshka-mm on PYTHONPATH."""

import pytest
import torch

from hydram3.adapters.m3 import matryoshka_pool

llava_arch = pytest.importorskip(
    "llava.model.llava_arch", reason="matryoshka-mm (llava) not importable"
)


@pytest.mark.parametrize(
    "length,scale",
    [(576, 576), (576, 144), (576, 36), (576, 9), (576, 1), (256, 64), (256, 16), (256, 4)],
)
def test_matches_matryoshka_vis_token_process(length, scale):
    x = torch.randn(2, length, 8)
    upstream = llava_arch.LlavaMetaForCausalLM.matryoshka_vis_token_process(None, x, scale)
    torch.testing.assert_close(matryoshka_pool(x, scale), upstream)
