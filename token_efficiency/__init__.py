"""Utilities for generating token efficiency and legibility plots."""

from .data_generation import (
    DEFAULT_TOKENIZER_SPECS,
    GenerationConfig,
    TokenizerSpec,
    generate_and_write_dataset,
    generate_token_efficiency_dataset,
    write_dataset,
)
from .plots import (
    LIGHT_THEME,
    DARK_THEME,
    TokenEfficiencyData,
    LegibilityData,
    LegibilityResult,
    load_token_efficiency_data,
    load_legibility_data,
    generate_token_efficiency_plots,
    generate_legibility_plots,
    mean_tokens_per_node,
)
from .legibility import (
    LegibilityBenchmarkConfig,
    LegibilityPromptSample,
    OpenRouterResponder,
    build_legibility_prompt_sample,
    run_legibility_benchmark,
    write_legibility_dataset,
)

__all__ = [
    "LIGHT_THEME",
    "DARK_THEME",
    "TokenEfficiencyData",
    "LegibilityData",
    "LegibilityResult",
    "load_token_efficiency_data",
    "load_legibility_data",
    "generate_token_efficiency_plots",
    "generate_legibility_plots",
    "mean_tokens_per_node",
    "GenerationConfig",
    "TokenizerSpec",
    "DEFAULT_TOKENIZER_SPECS",
    "generate_token_efficiency_dataset",
    "generate_and_write_dataset",
    "write_dataset",
    "LegibilityBenchmarkConfig",
    "LegibilityPromptSample",
    "run_legibility_benchmark",
    "write_legibility_dataset",
    "OpenRouterResponder",
    "build_legibility_prompt_sample",
]
