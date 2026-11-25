from pathlib import Path

import pytest

from hydram3 import FixedBudgetProvider, QBSMBudgetProviderAdapter
from hydram3.config import BudgetConfig, load_config, make_provider

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "hydra_gemma3_270m_mm.yaml"


def test_example_config_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRAM3_MODEL_PATH", str(tmp_path))
    cfg = load_config(EXAMPLE)
    assert Path(cfg.hydrallm.repo_path).name == "hydrallm"
    assert cfg.hydrallm.model_path == str(tmp_path.resolve())
    assert cfg.budget.supported_visual_tokens == [1, 4, 16, 64, 256]
    assert cfg.budget.supported_widths == [1, 2, 3, 4]
    assert cfg.generation.max_new_tokens == 64 and cfg.generation.do_sample is False
    assert isinstance(make_provider(cfg.budget), FixedBudgetProvider)


def test_unset_env_var_is_an_error(monkeypatch):
    monkeypatch.delenv("HYDRAM3_MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="unexpanded environment variable"):
        load_config(EXAMPLE)


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("budget:\n  supported_width: [1]\n")
    with pytest.raises(ValueError, match="supported_width"):
        load_config(p)


def test_hub_id_left_untouched(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("hydrallm:\n  model_path: some-org/some-model\n")
    assert load_config(p).hydrallm.model_path == "some-org/some-model"


def build_student(scale=1):
    return lambda inputs: (16 * scale, 2)


def test_qbsm_factory_import():
    prov = make_provider(
        BudgetConfig(
            provider="qbsm", qbsm_factory=f"{__name__}:build_student", qbsm_kwargs={"scale": 4}
        )
    )
    assert isinstance(prov, QBSMBudgetProviderAdapter)
    assert prov.predictor(None) == (64, 2)


def test_bad_provider_config():
    with pytest.raises(ValueError, match="fixed_visual_tokens"):
        make_provider(BudgetConfig(provider="fixed"))
    with pytest.raises(ValueError, match="unknown budget.provider"):
        make_provider(BudgetConfig(provider="learned"))
