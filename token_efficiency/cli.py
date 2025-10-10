"""Command-line interface for data generation and plotting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data_generation import (
    DEFAULT_SIZES,
    DEFAULT_TOKENIZER_SPECS,
    GenerationConfig,
    TokenizerSpec,
    generate_and_write_dataset,
)
from .plots import (
    generate_legibility_plots,
    generate_token_efficiency_plots,
    load_legibility_data,
    load_token_efficiency_data,
)
from .legibility import (
    LegibilityBenchmarkConfig,
    build_legibility_prompt_sample,
    run_legibility_benchmark,
    write_legibility_dataset,
)


def parse_sizes(values: Iterable[str] | None) -> Sequence[int]:
    if not values:
        return tuple(DEFAULT_SIZES)
    parsed = []
    for item in values:
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    if not parsed:
        raise ValueError("at least one size must be provided when overriding sizes")
    return tuple(parsed)


def parse_tokenizer_overrides(values: Iterable[str] | None) -> Mapping[str, TokenizerSpec]:
    overrides: dict[str, TokenizerSpec] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError(f"tokenizer override must use NAME=repo[@revision] format: {item}")
        name, target = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"invalid tokenizer name in override: {item}")
        if "@" in target:
            pretrained, revision = target.split("@", 1)
        else:
            pretrained, revision = target, None
        overrides[name] = TokenizerSpec(name=name, pretrained=pretrained.strip(), revision=revision)
    return overrides


def parse_int_sequence(values: Iterable[str] | None, *, default: Sequence[int]) -> Sequence[int]:
    if not values:
        return tuple(default)
    parsed = []
    for chunk in values:
        parts = (part.strip() for part in str(chunk).split(","))
        for item in parts:
            if not item:
                continue
            try:
                parsed.append(int(item))
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"invalid integer value: {item}") from exc
    if not parsed:
        raise ValueError("at least one integer value must be provided when overriding defaults")
    return tuple(parsed)


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    sizes = parse_sizes(args.size)
    tokenizer_overrides = parse_tokenizer_overrides(args.tokenizer)

    tokenizer_specs = list(DEFAULT_TOKENIZER_SPECS)
    known_names = {spec.name for spec in tokenizer_specs}
    for index, spec in enumerate(tokenizer_specs):
        if spec.name in tokenizer_overrides:
            tokenizer_specs[index] = tokenizer_overrides[spec.name]
    for name, spec in tokenizer_overrides.items():
        if name not in known_names:
            tokenizer_specs.append(spec)

    return GenerationConfig(
        seed=args.seed,
        samples_per_size=args.samples_per_size,
        sizes=sizes,
        tokenizer_specs=tuple(tokenizer_specs),
        cache_dir=args.cache_dir,
        wordlist_path=args.wordlist,
    )


def run_generate(args: argparse.Namespace) -> Path:
    config = build_generation_config(args)
    dataset = generate_and_write_dataset(
        config=config,
        output_path=args.output,
        force=args.force,
        cache_dir=args.cache_dir,
    )
    produced = Path(args.output).resolve()
    print(f"Wrote token efficiency dataset: {produced} ({len(dataset['observations'])} observations)")
    return produced


def run_plot(args: argparse.Namespace) -> None:
    token_data = load_token_efficiency_data(args.token_efficiency_data)
    legibility_data = load_legibility_data(args.legibility_data)
    token_dir = args.output_dir / "token_efficiency"
    legibility_dir = args.output_dir / "legibility"
    generate_token_efficiency_plots(token_data, token_dir)
    generate_legibility_plots(legibility_data, legibility_dir)
    print(f"Wrote plots to {args.output_dir.resolve()}")


def run_generate_and_plot(args: argparse.Namespace) -> None:
    dataset_path = run_generate(args)
    plot_args = argparse.Namespace(
        token_efficiency_data=dataset_path,
        legibility_data=args.legibility_data,
        output_dir=args.output_dir,
    )
    run_plot(plot_args)


def run_legibility(args: argparse.Namespace) -> None:
    defaults = LegibilityBenchmarkConfig()
    formats = tuple(args.format) if args.format else tuple(defaults.formats)
    input_nodes = parse_int_sequence(args.input_nodes, default=defaults.input_nodes)
    output_nodes = parse_int_sequence(args.output_nodes, default=defaults.output_nodes)
    config = LegibilityBenchmarkConfig(
        seed=args.seed,
        formats=formats,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        num_trials=args.num_trials,
        terminals_only=args.terminals_only,
        model=args.model,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        system_prompt=defaults.system_prompt,
        wordlist_path=args.wordlist,
    )
    dataset = run_legibility_benchmark(config)
    write_legibility_dataset(dataset, args.output)
    produced = Path(args.output).resolve()
    print(
        f"Wrote legibility benchmark dataset: {produced} "
        f"({len(dataset['results'])} entries, {config.num_trials} trials each)"
    )


def run_sample_prompt(args: argparse.Namespace) -> None:
    sample = build_legibility_prompt_sample(
        fmt=args.format,
        input_nodes=args.input_nodes,
        output_nodes=args.output_nodes,
        seed=args.seed,
        terminals_only=args.terminals_only,
        wordlist_path=args.wordlist,
    )
    print(sample.prompt.user_message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token efficiency utilities.")
    subparsers = parser.add_subparsers(dest="command")
    leg_defaults = LegibilityBenchmarkConfig()

    def add_generate_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--output",
            type=Path,
            default=Path("data") / "token_efficiency.json",
            help="Where to write the generated dataset.",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=GenerationConfig().seed,
            help="Random seed used for generation.",
        )
        subparser.add_argument(
            "--samples-per-size",
            type=int,
            default=GenerationConfig().samples_per_size,
            help="Number of samples per shape/size bucket.",
        )
        subparser.add_argument(
            "--size",
            action="append",
            metavar="N",
            help="Specify a size (node target). Repeat to provide multiple shared sizes.",
        )
        subparser.add_argument(
            "--tokenizer",
            action="append",
            metavar="NAME=repo[@revision]",
            help="Override or add tokenizer definitions. Can be repeated.",
        )
        subparser.add_argument(
            "--cache-dir",
            type=Path,
            default=GenerationConfig().cache_dir,
            help="Directory used to cache generated samples.",
        )
        subparser.add_argument(
            "--wordlist",
            type=Path,
            default=GenerationConfig().wordlist_path,
            help="Path to the dictionary used for generating identifiers.",
        )
        subparser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate data even if an existing dataset matches the metadata.",
        )

    generate_parser = subparsers.add_parser("generate", help="Generate token efficiency data.")
    add_generate_options(generate_parser)

    plot_parser = subparsers.add_parser("plot", help="Render plots from existing data.")
    plot_parser.add_argument(
        "--token-efficiency-data",
        type=Path,
        default=Path("data") / "token_efficiency.json",
        help="Path to the token efficiency dataset JSON file.",
    )
    plot_parser.add_argument(
        "--legibility-data",
        type=Path,
        default=Path("data") / "legibility.json",
        help="Path to the legibility dataset JSON file.",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write the generated plots.",
    )

    gap_parser = subparsers.add_parser(
        "generate-and-plot",
        help="Generate token efficiency data and then render plots.",
    )
    add_generate_options(gap_parser)
    gap_parser.add_argument(
        "--legibility-data",
        type=Path,
        default=Path("data") / "legibility.json",
        help="Path to the legibility dataset JSON file.",
    )

    leg_parser = subparsers.add_parser(
        "legibility",
        help="Run the legibility benchmark using OpenRouter.",
    )
    leg_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "legibility.json",
        help="Where to write the benchmark results.",
    )
    leg_parser.add_argument(
        "--seed",
        type=int,
        default=leg_defaults.seed,
        help="Random seed used for sample generation.",
    )
    leg_parser.add_argument(
        "--format",
        action="append",
        dest="format",
        help="Specify a serialization format to include. Repeat to provide multiple values.",
    )
    leg_parser.add_argument(
        "--input-nodes",
        action="append",
        help="Target input node counts. Repeat to provide multiple values.",
    )
    leg_parser.add_argument(
        "--output-nodes",
        action="append",
        help="Target output node counts. Repeat to provide multiple values.",
    )
    leg_parser.add_argument(
        "--num-trials",
        type=int,
        default=leg_defaults.num_trials,
        help="Number of trials per format/input/output bucket.",
    )
    leg_parser.add_argument(
        "--model",
        type=str,
        default=leg_defaults.model,
        help="OpenRouter model identifier used for evaluation.",
    )
    leg_parser.add_argument(
        "--temperature",
        type=float,
        default=leg_defaults.temperature,
        help="Sampling temperature for the model.",
    )
    leg_parser.add_argument(
        "--timeout",
        type=float,
        default=leg_defaults.timeout_seconds,
        help="HTTP timeout (seconds) for OpenRouter requests.",
    )
    leg_parser.add_argument(
        "--wordlist",
        type=Path,
        default=leg_defaults.wordlist_path,
        help="Dictionary used for generating identifiers.",
    )
    leg_parser.add_argument(
        "--terminals-only",
        action="store_true",
        help="Restrict benchmark targets to lists of terminal values (no nested structures).",
    )

    sample_parser = subparsers.add_parser(
        "sample-prompt",
        help="Preview the prompt sent to the model for the legibility benchmark.",
    )
    sample_parser.add_argument(
        "input_nodes",
        type=int,
        help="Target input node count for the generated dataset.",
    )
    sample_parser.add_argument(
        "output_nodes",
        type=int,
        help="Target output node count extracted by the Python snippet.",
    )
    sample_parser.add_argument(
        "--format",
        choices=tuple(leg_defaults.formats),
        default=leg_defaults.formats[0],
        help="Serialization format for the dataset.",
    )
    sample_parser.add_argument(
        "--seed",
        type=int,
        default=leg_defaults.seed,
        help="Random seed used for deterministic sample generation.",
    )
    sample_parser.add_argument(
        "--terminals-only",
        action="store_true",
        help="Restrict target values to terminals (scalars only).",
    )
    sample_parser.add_argument(
        "--wordlist",
        type=Path,
        default=leg_defaults.wordlist_path,
        help="Word list used when constructing identifiers.",
    )
    gap_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write the generated plots.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command is None:
        parser.print_help()
        return
    if command == "generate":
        run_generate(args)
    elif command == "plot":
        run_plot(args)
    elif command == "generate-and-plot":
        run_generate_and_plot(args)
    elif command == "legibility":
        run_legibility(args)
    elif command == "sample-prompt":
        run_sample_prompt(args)
    else:
        parser.error(f"unknown command: {command}")


__all__ = [
    "main",
    "build_parser",
    "run_generate",
    "run_plot",
    "run_generate_and_plot",
    "run_legibility",
    "run_sample_prompt",
]
