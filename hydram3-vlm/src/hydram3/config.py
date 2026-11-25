"""YAML configuration and pipeline factory.

Relative paths in the YAML are resolved against the directory of the YAML file; ``${VAR}`` and
``~`` are expanded, so machine-specific paths can come from the environment.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .budget import BudgetProvider, FixedBudgetProvider, QBSMBudgetProviderAdapter
from .types import GenerationOptions


@dataclass
class HydraLLMConfig:
    repo_path: str = "../hydrallm"
    model_path: Optional[str] = None
    processor_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "auto"
    attn_implementation: str = "auto"


@dataclass
class BudgetConfig:
    supported_visual_tokens: Optional[List[int]] = None
    supported_widths: Optional[List[int]] = None
    provider: str = "fixed"  # "fixed" | "qbsm"
    fixed_visual_tokens: Optional[int] = None
    fixed_width: Optional[int] = None
    qbsm_factory: Optional[str] = None  # "package.module:callable" returning a predictor
    qbsm_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    hydrallm: HydraLLMConfig = field(default_factory=HydraLLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    prompt_template: str = "<start_of_image>\n{prompt}"
    image_placeholder: str = "<start_of_image>"


def _build(cls: Any, data: Optional[Dict[str, Any]], where: str) -> Any:
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {unknown}; allowed: {sorted(known)}")
    return cls(**data)


def _resolve_path(value: Optional[str], base: Path) -> Optional[str]:
    if value is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        raise ValueError(f"unexpanded environment variable in path {value!r}; is it set?")
    path = Path(expanded)
    # Hugging Face hub IDs ("org/name") stay untouched unless they exist relative to the config.
    candidate = path if path.is_absolute() else base / path
    if path.is_absolute() or candidate.exists() or expanded.startswith("."):
        return str(candidate.resolve())
    return expanded


def load_config(path: Union[str, Path]) -> PipelineConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    top = {f.name for f in fields(PipelineConfig)}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"unknown top-level key(s) {unknown}; allowed: {sorted(top)}")
    base = path.parent
    cfg = PipelineConfig(
        hydrallm=_build(HydraLLMConfig, raw.get("hydrallm"), "hydrallm"),
        budget=_build(BudgetConfig, raw.get("budget"), "budget"),
        generation=_build(GenerationOptions, raw.get("generation"), "generation"),
        **{k: raw[k] for k in ("prompt_template", "image_placeholder") if k in raw},
    )
    hc = cfg.hydrallm
    hc.repo_path = _resolve_path(hc.repo_path, base) or hc.repo_path
    hc.model_path = _resolve_path(hc.model_path, base)
    hc.processor_path = _resolve_path(hc.processor_path, base)
    return cfg


def make_provider(cfg: BudgetConfig) -> BudgetProvider:
    if cfg.provider == "fixed":
        if cfg.fixed_visual_tokens is None or cfg.fixed_width is None:
            raise ValueError("budget.fixed_visual_tokens and budget.fixed_width are required")
        return FixedBudgetProvider(cfg.fixed_visual_tokens, cfg.fixed_width)
    if cfg.provider == "qbsm":
        if not cfg.qbsm_factory or ":" not in cfg.qbsm_factory:
            raise ValueError("budget.qbsm_factory must look like 'package.module:callable'")
        module_name, attr = cfg.qbsm_factory.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attr)
        return QBSMBudgetProviderAdapter(factory(**cfg.qbsm_kwargs))
    raise ValueError(f"unknown budget.provider {cfg.provider!r}; use 'fixed' or 'qbsm'")


def build_pipeline(cfg: PipelineConfig) -> Any:
    """Load the HydraLLM checkpoint and return a ready :class:`ElasticVLMPipeline`."""
    from .adapters.hydra import HydraM3Backend
    from .pipeline import ElasticVLMPipeline

    backend = HydraM3Backend.from_config(cfg)
    return ElasticVLMPipeline(backend, make_provider(cfg.budget), cfg.generation)
