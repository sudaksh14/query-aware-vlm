# Portions adapted from matryoshka-mm (M3), llava/model/llava_arch.py,
# LlavaMetaForCausalLM.matryoshka_vis_token_process.
#   Copyright 2023 Haotian Liu; M3 modifications by Mu Cai et al.
#   Licensed under the Apache License, Version 2.0. See ../../../NOTICE.
# The operation is re-implemented here (instead of imported) because the `llava` package pins
# torch==2.1.2 / transformers==4.36.2, which conflict with HydraLLM's torch 2.8 / transformers 4.56.
# tests/test_m3_upstream_parity.py checks numerical equality against the upstream function
# whenever `llava` is importable.
"""M3 (Matryoshka Multimodal Models) visual-token budgeting.

M3 has no learned selector: a budget ``N`` is realised by 2D average pooling of the square grid
of *projected* visual tokens (pooling runs after ``mm_projector`` in M3), with
``kernel = stride = int(sqrt(L / N))``. The nested operating points are the grids whose side
evenly divides the full grid side; M3's released LLaVA-1.5 checkpoints were trained at
``1, 9, 36, 144, 576`` on a 24x24 grid. HydraLLM's Gemma 3 projector emits a 16x16 grid
(``mm_tokens_per_image=256``), whose exact nested points are ``1, 4, 16, 64, 256``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def grid_side(num_tokens: int) -> int:
    """Side of the square token grid; raises if ``num_tokens`` is not a perfect square."""
    side = math.isqrt(num_tokens)
    if side * side != num_tokens or side == 0:
        raise ValueError(f"M3 pooling needs a square token grid; got {num_tokens} tokens")
    return side


def m3_pool_size(num_tokens: int, target_tokens: int) -> int:
    """Kernel/stride M3 uses for ``target_tokens`` (same formula as upstream)."""
    if not 1 <= target_tokens <= num_tokens:
        raise ValueError(f"target_tokens must be in [1, {num_tokens}], got {target_tokens}")
    return int(math.sqrt(num_tokens / target_tokens))


def m3_output_tokens(num_tokens: int, target_tokens: int) -> int:
    """Number of tokens M3 pooling really produces for ``target_tokens``.

    Equals ``target_tokens`` only on the nested points; e.g. on a 16x16 grid a request of 100
    yields pool size 1 and 256 tokens, and 25 yields pool 3 and 25 tokens with the last
    row/column dropped. :func:`nested_token_counts` lists the exact, border-preserving points.
    """
    side = grid_side(num_tokens)
    return (side // m3_pool_size(num_tokens, target_tokens)) ** 2


def nested_token_counts(num_tokens: int) -> Tuple[int, ...]:
    """Exact nested M3 operating points: ``(side / p)^2`` for every ``p`` dividing the side."""
    side = grid_side(num_tokens)
    return tuple(sorted((side // p) ** 2 for p in range(1, side + 1) if side % p == 0))


def matryoshka_pool(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Pool ``(B, L, C)`` grid tokens (raster order) to the M3 scale ``target_tokens``."""
    if features.dim() != 3:
        raise ValueError(f"expected (batch, tokens, channels), got shape {tuple(features.shape)}")
    batch, length, channels = features.shape
    side = grid_side(length)
    pool = m3_pool_size(length, target_tokens)
    grid = features.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, kernel_size=pool, stride=pool)
    return pooled.permute(0, 2, 3, 1).reshape(batch, -1, channels)


class M3TokenPooler:
    """M3 adapter: maps a resolved visual-token budget onto M3 pooling for one grid size."""

    def __init__(self, encoder_tokens: int) -> None:
        self.encoder_tokens = encoder_tokens
        self.supported = nested_token_counts(encoder_tokens)

    def pool(self, features: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Pool to exactly ``target_tokens`` tokens; refuses budgets M3 cannot realise exactly."""
        if features.shape[1] != self.encoder_tokens:
            raise ValueError(
                f"pooler built for {self.encoder_tokens} tokens, got {features.shape[1]}"
            )
        if target_tokens not in self.supported:
            raise ValueError(
                f"{target_tokens} visual tokens is not an exact nested M3 point for a "
                f"{self.encoder_tokens}-token grid; supported: {list(self.supported)}"
            )
        pooled = matryoshka_pool(features, target_tokens)
        assert pooled.shape[1] == target_tokens  # guaranteed by membership in self.supported
        return pooled
