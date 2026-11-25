import pytest
import torch

from hydram3.adapters.m3 import (
    M3TokenPooler,
    m3_output_tokens,
    matryoshka_pool,
    nested_token_counts,
)


def test_nested_points():
    assert nested_token_counts(256) == (1, 4, 16, 64, 256)
    pts = nested_token_counts(576)
    assert pts == (1, 4, 9, 16, 36, 64, 144, 576)
    assert {1, 9, 36, 144, 576} <= set(pts)  # M3's released LLaVA-1.5 training scales


def test_non_square_grid_rejected():
    with pytest.raises(ValueError, match="square"):
        nested_token_counts(255)


def test_upstream_formula_off_grid_values():
    # What M3's int(sqrt(L/N)) kernel really yields for budgets that are not nested points.
    assert m3_output_tokens(256, 100) == 256
    assert m3_output_tokens(256, 25) == 25  # kernel 3, last row/column dropped
    assert m3_output_tokens(256, 63) == 64


def test_pool_is_block_mean_in_raster_order():
    side, c = 4, 3
    x = torch.arange(side * side * c, dtype=torch.float32).reshape(1, side * side, c)
    out = matryoshka_pool(x, 4)
    grid = x.reshape(side, side, c)
    expected = torch.stack(
        [grid[i : i + 2, j : j + 2].mean(dim=(0, 1)) for i in (0, 2) for j in (0, 2)]
    ).unsqueeze(0)
    torch.testing.assert_close(out, expected)


def test_pool_is_nested():
    x = torch.randn(2, 256, 8)
    torch.testing.assert_close(matryoshka_pool(matryoshka_pool(x, 64), 16), matryoshka_pool(x, 16))
    assert matryoshka_pool(x, 256).equal(x)
    assert matryoshka_pool(x, 1).shape == (2, 1, 8)


def test_pooler_rejects_inexact_and_wrong_grid():
    pooler = M3TokenPooler(256)
    with pytest.raises(ValueError, match="not an exact nested"):
        pooler.pool(torch.zeros(1, 256, 4), 25)
    with pytest.raises(ValueError, match="built for 256"):
        pooler.pool(torch.zeros(1, 64, 4), 16)
    assert pooler.pool(torch.zeros(1, 256, 4), 16).shape == (1, 16, 4)
