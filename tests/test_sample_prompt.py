from token_efficiency.cli import build_parser, run_sample_prompt
from token_efficiency.data_generation import count_nodes
from token_efficiency.legibility import build_legibility_prompt_sample


def test_build_legibility_prompt_sample_deterministic() -> None:
    sample = build_legibility_prompt_sample(
        fmt="json_min",
        input_nodes=12,
        output_nodes=4,
        seed=12345,
    )
    duplicate = build_legibility_prompt_sample(
        fmt="json_min",
        input_nodes=12,
        output_nodes=4,
        seed=12345,
    )
    assert sample.prompt.user_message == duplicate.prompt.user_message
    assert sample.prompt.format == "json_min"
    assert sample.prompt.input_node_count == 12
    assert sample.prompt.output_node_count == count_nodes(sample.expected_output)
    assert sample.serialized_input
    assert "Dataset:" in sample.prompt.user_message


def test_sample_prompt_subcommand_prints_prompt(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["sample-prompt", "12", "4", "--seed", "42", "--format", "json_min"])
    run_sample_prompt(args)
    captured = capsys.readouterr()
    assert "Format: json_min" in captured.out
    assert "Input nodes observed: 12" in captured.out
