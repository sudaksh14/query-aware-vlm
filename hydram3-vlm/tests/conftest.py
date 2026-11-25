import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # allow running without `pip install -e .`

# HydraLLM checkout: env override, else the sibling directory in this workspace.
HYDRALLM_REPO = Path(os.environ.get("HYDRALLM_REPO", REPO.parent / "hydrallm")).resolve()


@pytest.fixture
def hydrallm_repo() -> Path:
    if not (HYDRALLM_REPO / "model_implementations").is_dir():
        pytest.skip(f"HydraLLM checkout not found at {HYDRALLM_REPO} (set HYDRALLM_REPO)")
    pytest.importorskip("transformers")
    return HYDRALLM_REPO
