"""Command line: one image + one prompt -> JSON result. Needs a real HydraLLM checkpoint."""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .config import build_pipeline, load_config
from .types import InferenceRequest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hydram3", description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="YAML config (see configs/)")
    parser.add_argument("--image", required=True, help="path to an image file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--visual-tokens", type=int, help="override: fixed N (implies provider=fixed)"
    )
    parser.add_argument(
        "--width", type=int, help="override: fixed W, active heads (provider=fixed)"
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--request-id")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.visual_tokens is not None or args.width is not None:
        cfg.budget.provider = "fixed"
        if args.visual_tokens is not None:
            cfg.budget.fixed_visual_tokens = args.visual_tokens
        if args.width is not None:
            cfg.budget.fixed_width = args.width
    if args.max_new_tokens is not None:
        from dataclasses import replace

        cfg.generation = replace(cfg.generation, max_new_tokens=args.max_new_tokens)

    pipeline = build_pipeline(cfg)
    result = pipeline.run(
        InferenceRequest(image=args.image, prompt=args.prompt, request_id=args.request_id)
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
