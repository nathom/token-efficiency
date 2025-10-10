from __future__ import annotations

import numpy as np

from pathlib import Path

from token_efficiency.plots import (
    generate_legibility_plots,
    generate_token_efficiency_plots,
    load_legibility_data,
    load_token_efficiency_data,
    mean_tokens_per_node,
)


DATA_DIR = Path("data")


def test_token_efficiency_loading() -> None:
    data = load_token_efficiency_data(DATA_DIR / "token_efficiency.json")
    assert data.shapes == ["nested", "tabular", "sparse"]
    assert data.formats == ["json_min", "yaml_block", "toml", "xml", "csv"]
    assert data.tokenizers == ["gpt-oss", "qwen3", "llama3.2"]
    nested_json_min = mean_tokens_per_node(data, "nested", "json_min")
    assert np.isfinite(nested_json_min)
    assert nested_json_min > 0.0


def test_generate_token_efficiency_plots(tmp_path) -> None:
    data = load_token_efficiency_data(DATA_DIR / "token_efficiency.json")
    outputs = generate_token_efficiency_plots(data, tmp_path)
    expected_files = {
        "tokens_per_node_matrix_light.svg",
        "tokens_per_node_matrix_dark.svg",
        "tokens_per_node_by_format_light.svg",
        "tokens_per_node_by_format_dark.svg",
        "tokens_per_node_by_shape_light.svg",
        "tokens_per_node_by_shape_dark.svg",
        "tokens_per_node_by_tokenizer_light.svg",
        "tokens_per_node_by_tokenizer_dark.svg",
    }
    produced = {path.name for path in outputs}
    assert produced == expected_files
    for name in expected_files:
        assert (tmp_path / name).exists()


def test_generate_legibility_plots(tmp_path) -> None:
    data = load_legibility_data(DATA_DIR / "legibility.json")
    outputs = generate_legibility_plots(data, tmp_path)
    expected_files = {
        "json_min_accuracy_matrix_light.svg",
        "json_min_accuracy_matrix_dark.svg",
        "yaml_block_accuracy_matrix_light.svg",
        "yaml_block_accuracy_matrix_dark.svg",
        "toml_accuracy_matrix_light.svg",
        "toml_accuracy_matrix_dark.svg",
        "json_min_jaccard_matrix_light.svg",
        "json_min_jaccard_matrix_dark.svg",
        "yaml_block_jaccard_matrix_light.svg",
        "yaml_block_jaccard_matrix_dark.svg",
        "toml_jaccard_matrix_light.svg",
        "toml_jaccard_matrix_dark.svg",
        "json_min_dictdiff_matrix_light.svg",
        "json_min_dictdiff_matrix_dark.svg",
        "yaml_block_dictdiff_matrix_light.svg",
        "yaml_block_dictdiff_matrix_dark.svg",
        "toml_dictdiff_matrix_light.svg",
        "toml_dictdiff_matrix_dark.svg",
        "accuracy_vs_output_nodes_light.svg",
        "accuracy_vs_output_nodes_dark.svg",
        "accuracy_vs_input_nodes_light.svg",
        "accuracy_vs_input_nodes_dark.svg",
        "jaccard_vs_output_nodes_light.svg",
        "jaccard_vs_output_nodes_dark.svg",
        "jaccard_vs_input_nodes_light.svg",
        "jaccard_vs_input_nodes_dark.svg",
        "dictdiff_vs_output_nodes_light.svg",
        "dictdiff_vs_output_nodes_dark.svg",
        "dictdiff_vs_input_nodes_light.svg",
        "dictdiff_vs_input_nodes_dark.svg",
    }
    produced = {path.name for path in outputs}
    assert produced == expected_files
    for name in expected_files:
        assert (tmp_path / name).exists()
