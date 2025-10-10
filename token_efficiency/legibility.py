"""Legibility benchmark powered by OpenRouter."""

from __future__ import annotations

import asyncio
import difflib
import functools
import inspect
import json
import logging
import os
import random
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dictdiffer import diff as dictdiffer_diff
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)
import yaml
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .data_generation import (
    SERIALIZERS,
    GenerationConfig,
    count_nodes,
    generate_nested,
    load_wordlist,
)

logger = logging.getLogger(__name__)


# --- Configuration ---------------------------------------------------------


DEFAULT_FORMATS: Sequence[str] = ("json_min", "yaml_block", "toml")
DEFAULT_INPUT_NODES: Sequence[int] = (50,)
DEFAULT_OUTPUT_NODES: Sequence[int] = (5,)


@dataclass(frozen=True)
class LegibilityBenchmarkConfig:
    """Runtime configuration for the legibility benchmark."""

    seed: int = 42097
    formats: Sequence[str] = field(default_factory=lambda: DEFAULT_FORMATS)
    input_nodes: Sequence[int] = field(default_factory=lambda: DEFAULT_INPUT_NODES)
    output_nodes: Sequence[int] = field(default_factory=lambda: DEFAULT_OUTPUT_NODES)
    num_trials: int = 25
    terminals_only: bool = False
    model: str = "deepseek/deepseek-chat"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    system_prompt: str = (
        "You evaluate structured data legibility. Respond with a single fenced code block "
        "containing only the requested serialized payload—no commentary or extra text."
    )
    wordlist_path: Path = GenerationConfig().wordlist_path

    def ensure_valid(self) -> None:
        if not self.formats:
            raise ValueError("at least one serialization format must be provided")
        if not self.input_nodes:
            raise ValueError("at least one input_nodes target must be provided")
        if not self.output_nodes:
            raise ValueError("at least one output_nodes target must be provided")
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")


def _config_log_payload(config: LegibilityBenchmarkConfig) -> Dict[str, object]:
    return {
        "seed": config.seed,
        "formats": list(config.formats),
        "input_nodes": list(config.input_nodes),
        "output_nodes": list(config.output_nodes),
        "num_trials": config.num_trials,
        "terminals_only": config.terminals_only,
        "model": config.model,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
        "wordlist_path": str(config.wordlist_path),
    }


def _log_configuration(config: LegibilityBenchmarkConfig) -> None:
    payload = _config_log_payload(config)
    message = f"Starting legibility benchmark with configuration: {payload}"
    if logger.isEnabledFor(logging.INFO):
        logger.info(message)
    else:
        print(message, file=sys.stderr)


def _render_summary_table(results: Sequence[Mapping[str, object]]) -> None:
    if not results:
        return
    console = Console()
    table = Table(title="Legibility Benchmark Summary")
    table.add_column("Format")
    table.add_column("Input Nodes", justify="right")
    table.add_column("Output Nodes", justify="right")
    table.add_column("Trials", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Jaccard", justify="right")
    table.add_column("DictDiff", justify="right")
    table.add_column("Mean Input", justify="right")
    table.add_column("Mean Output", justify="right")
    table.add_column("Prompt Chars", justify="right")
    table.add_column("Response Chars", justify="right")
    table.add_column("Terminals Only", justify="center")

    total_trials = 0
    total_prompt_chars = 0
    total_response_chars = 0
    total_successes = 0.0
    total_jaccard = 0.0
    total_dictdiff = 0.0

    for entry in results:
        num_trials = int(entry.get("num_trials", 0))
        accuracy = float(entry.get("accuracy", 0.0))
        total_trials += num_trials
        total_prompt_chars += int(entry.get("total_prompt_chars", 0))
        total_response_chars += int(entry.get("total_response_chars", 0))
        total_successes += accuracy * num_trials
        jaccard = float(entry.get("jaccard", 0.0))
        total_jaccard += jaccard * num_trials
        dictdiff = float(entry.get("dictdiff", 0.0))
        total_dictdiff += dictdiff * num_trials
        table.add_row(
            str(entry.get("format", "")),
            str(entry.get("input_nodes", "")),
            str(entry.get("output_nodes", "")),
            str(num_trials),
            f"{accuracy * 100:.1f}%",
            f"{jaccard * 100:.1f}%",
            f"{dictdiff * 100:.1f}%",
            f"{float(entry.get('mean_input_nodes', 0.0)):.1f}",
            f"{float(entry.get('mean_output_nodes', 0.0)):.1f}",
            f"{int(entry.get('total_prompt_chars', 0)):,}",
            f"{int(entry.get('total_response_chars', 0)):,}",
            "✓" if entry.get("terminals_only") else "",
        )

    if total_trials:
        overall_accuracy = total_successes / total_trials
        overall_jaccard = total_jaccard / total_trials
        overall_dictdiff = total_dictdiff / total_trials
    else:
        overall_accuracy = 0.0
        overall_jaccard = 0.0
        overall_dictdiff = 0.0
    table.add_section()
    table.add_row(
        "TOTAL",
        "-",
        "-",
        str(total_trials),
        f"{overall_accuracy * 100:.1f}%",
        f"{overall_jaccard * 100:.1f}%",
        f"{overall_dictdiff * 100:.1f}%",
        "-",
        "-",
        f"{total_prompt_chars:,}",
        f"{total_response_chars:,}",
        "",
    )
    console.print(table)


@dataclass(frozen=True)
class LegibilityPrompt:
    """Container describing a single benchmark prompt."""

    format: str
    user_message: str
    expected_object: object
    input_node_count: int
    output_node_count: int


@dataclass(frozen=True)
class LegibilityPromptSample:
    """Artifacts describing a deterministic legibility prompt sample."""

    prompt: LegibilityPrompt
    input_data: object
    expected_output: object
    python_snippet: str
    serialized_input: str


@dataclass(frozen=True)
class ExampleCase:
    """Static example comprised of dataset and executable snippet."""

    dataset: object
    snippet: str
    expected: object


# --- Helpers ----------------------------------------------------------------


def _format_code_block_language(fmt: str) -> str:
    if fmt == "json_min":
        return "json"
    if fmt == "yaml_block":
        return "yaml"
    if fmt == "toml":
        return "toml"
    return ""


def _serialize(format_name: str, value: object) -> str:
    serializer = SERIALIZERS.get(format_name)
    if serializer is None:
        raise ValueError(f"unsupported serialization format: {format_name}")
    return serializer("nested", value)


def _parse(format_name: str, text: str) -> object:
    stripped = text.strip()
    if "```" in stripped:
        lines = stripped.splitlines()
        fence_indices = [idx for idx, line in enumerate(lines) if line.startswith("```")]
        if fence_indices:
            start = fence_indices[0]
            end = next((idx for idx in fence_indices[1:] if idx > start), None)
            if end is not None:
                stripped = "\n".join(lines[start + 1 : end]).strip()
            else:
                stripped = "\n".join(lines[start + 1 :]).strip()
    if format_name == "json_min":
        data = json.loads(stripped)
        if not isinstance(data, Mapping) or "results" not in data:
            raise ValueError("JSON response must include top-level 'results'")
        return data["results"]
    if format_name == "yaml_block":
        data = yaml.safe_load(stripped)
        if not isinstance(data, Mapping) or "results" not in data:
            raise ValueError("YAML response must include top-level 'results'")
        return data["results"]
    if format_name == "toml":
        data = tomllib.loads(stripped)
        if not isinstance(data, Mapping) or "results" not in data:
            raise ValueError("TOML response must include top-level 'results'")
        return data["results"]
    raise ValueError(f"unsupported format for parsing: {format_name}")


PathComponent = Tuple[str | int, ...]


def _iter_paths(value: object, prefix: Tuple[str | int, ...] = ()) -> Iterable[Tuple[str | int, ...]]:
    yield prefix
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_paths(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_paths(child, prefix + (index,))


def _read_path(value: object, path: Tuple[str | int, ...]) -> object:
    current = value
    for component in path:
        if isinstance(component, int):
            if not isinstance(current, list) or component >= len(current):
                raise KeyError(f"list index {component} unavailable in path {path}")
            current = current[component]
        else:
            if not isinstance(current, Mapping) or component not in current:
                raise KeyError(f"key {component} unavailable in path {path}")
            current = current[component]
    return current


def _collect_nodes_by_path(value: object) -> Dict[Tuple[str | int, ...], Tuple[str, object]]:
    """Return a mapping from structural path to a tuple describing the node."""
    nodes: Dict[Tuple[str | int, ...], Tuple[str, object]] = {}
    for path in _iter_paths(value):
        node = _read_path(value, path)
        if isinstance(node, Mapping):
            nodes[path] = ("mapping", None)
        elif isinstance(node, list):
            nodes[path] = ("list", None)
        else:
            nodes[path] = ("scalar", node)
    return nodes


def _dictdiff_score(
    expected: object,
    actual: object | None,
) -> Tuple[float, Dict[str, int], Dict[str, List[str]]]:
    changes = list(
        dictdiffer_diff(
            {"result": expected},
            {"result": actual},
            dot_notation=False,
        )
    )

    counts: Dict[str, int] = {"add": 0, "remove": 0, "change": 0, "total": len(changes)}
    mismatches: Dict[str, List[str]] = {
        "missing": [],
        "extra": [],
        "value": [],
        "order": [],
    }
    to_label = functools.partial(_path_to_expr, root="result")

    for action, base_path, _ in changes:
        if isinstance(base_path, (list, tuple)):
            path_tuple: Tuple[str | int, ...] = tuple(base_path)
        elif base_path in (None, "", ()):
            path_tuple = ()
        else:
            path_tuple = (base_path,)
        if path_tuple and path_tuple[0] == "result":
            path_tuple = path_tuple[1:]
        label = to_label(path_tuple)

        if action == "add":
            counts["add"] += 1
            mismatches["extra"].append(label)
        elif action == "remove":
            counts["remove"] += 1
            mismatches["missing"].append(label)
        elif action == "change":
            counts["change"] += 1
            mismatches["value"].append(label)

    counts["total"] = counts["add"] + counts["remove"] + counts["change"]

    expected_nodes = _collect_nodes_by_path(expected)
    actual_nodes = _collect_nodes_by_path(actual) if actual is not None else {}
    shared_count = sum(
        1
        for path in expected_nodes.keys() & actual_nodes.keys()
        if expected_nodes[path] == actual_nodes[path]
    )

    total_ops = counts["total"]
    denominator = shared_count + total_ops
    score = 1.0 if denominator == 0 else shared_count / denominator
    score = max(0.0, min(1.0, score))

    return score, counts, mismatches


def _jaccard_index(expected: object, actual: Optional[object]) -> float:
    """Compute the Jaccard index between two nested structures keyed by path."""
    expected_nodes = _collect_nodes_by_path(expected)
    actual_nodes = _collect_nodes_by_path(actual) if actual is not None else {}

    expected_paths = set(expected_nodes)
    actual_paths = set(actual_nodes)
    union_paths = expected_paths | actual_paths
    if not union_paths:
        return 1.0

    intersection = sum(
        1
        for path in expected_paths & actual_paths
        if expected_nodes[path] == actual_nodes[path]
    )
    score = intersection / len(union_paths)
    return max(0.0, min(1.0, score))


def _is_terminal(value: object) -> bool:
    return not isinstance(value, (Mapping, list))


def _path_to_expr(path: Tuple[str | int, ...], *, root: str = "data") -> str:
    expr = root
    for component in path:
        if isinstance(component, int):
            expr += f"[{component}]"
        else:
            expr += f"[{json.dumps(component)}]"
    return expr


def _render_python_snippet(paths: Sequence[Tuple[str | int, ...]]) -> str:
    lines = ["target = ["]
    for path in paths:
        value_expr = _path_to_expr(path)
        lines.append(f"    {value_expr},")
    lines.append("]")
    return "\n".join(lines)


def _build_target(
    data: object,
    rng: random.Random,
    target_nodes: int,
    *,
    terminals_only: bool = False,
) -> Tuple[object, List[Tuple[str | int, ...]]]:
    paths = [path for path in _iter_paths(data) if path]
    if not paths:
        entry_value = deepcopy(data)
        if terminals_only and not _is_terminal(entry_value):
            raise RuntimeError("terminals_only=True but dataset contains no terminal values")
        return [entry_value], [()]
    rng.shuffle(paths)
    entries: List[Tuple[Tuple[str | int, ...], object]] = []
    result: List[object] = []
    used: set[Tuple[str | int, ...]] = set()
    for path in paths:
        if path in used:
            continue
        raw_value = _read_path(data, path)
        if terminals_only and not _is_terminal(raw_value):
            continue
        value = deepcopy(raw_value)
        entries.append((path, value))
        result.append(value)
        used.add(path)
        if count_nodes(result) >= target_nodes:
            break
    if not result:
        entry_value = deepcopy(data)
        if terminals_only and not _is_terminal(entry_value):
            raise RuntimeError("terminals_only=True but no terminal nodes available for selection")
        entries.append(((), entry_value))
        result.append(entry_value)
    return result, [path for path, _ in entries]


_EXAMPLE_CONFIGS: Mapping[str, Sequence[Tuple[int, int, int]]] = {
    "json_min": (
        (101001, 50, 10),
        (202002, 100, 20),
    ),
    "yaml_block": (
        (303003, 50, 10),
        (404004, 100, 20),
    ),
    "toml": (
        (505005, 50, 10),
        (606006, 100, 20),
    ),
}


@functools.lru_cache(maxsize=None)
def _example_words() -> Sequence[str]:
    return load_wordlist(GenerationConfig().wordlist_path)


def _generate_example_case(
    fmt: str,
    seed: int,
    input_nodes: int,
    output_nodes: int,
    *,
    terminals_only: bool = False,
) -> ExampleCase:
    words = _example_words()
    rng = random.Random(seed)
    data = generate_nested(input_nodes, rng, words)
    actual_input_nodes = count_nodes(data)
    if actual_input_nodes != input_nodes:
        raise RuntimeError(f"example data input nodes {actual_input_nodes} != requested {input_nodes}")
    target_object, paths = _build_target(
        data,
        rng,
        output_nodes,
        terminals_only=terminals_only,
    )
    snippet = _render_python_snippet(paths)
    namespace: Dict[str, object] = {"data": deepcopy(data)}
    exec(snippet, {}, namespace)
    expected = namespace.get("target")
    if expected != target_object:
        raise RuntimeError("example snippet did not reproduce generated target")
    return ExampleCase(dataset=deepcopy(data), snippet=snippet, expected=deepcopy(expected))


@functools.lru_cache(maxsize=None)
def _example_cases(fmt: str, terminals_only: bool) -> Sequence[ExampleCase]:
    configs = _EXAMPLE_CONFIGS.get(fmt)
    if not configs:
        return ()
    return tuple(
        _generate_example_case(
            fmt,
            seed,
            input_nodes,
            output_nodes,
            terminals_only=terminals_only,
        )
        for seed, input_nodes, output_nodes in configs
    )


def _example_blocks(fmt: str, *, terminals_only: bool) -> str:
    cases = _example_cases(fmt, terminals_only)
    if not cases:
        return "Example not available for this format."
    fence = _format_code_block_language(fmt)
    blocks: List[str] = []
    for index, case in enumerate(cases, start=1):
        serialized_input = _serialize(fmt, case.dataset)
        serialized_target = _serialize(fmt, case.expected)
        data_block = f"```{fence}\n{serialized_input}\n```" if fence else f"```\n{serialized_input}\n```"
        python_block = f"```python\n{case.snippet}\n```"
        response_block = (
            f"```{fence}\n{serialized_target}\n```" if fence else f"```\n{serialized_target}\n```"
        )
        sections = [
            f"Example {index}:",
            "Dataset:",
            data_block,
            "",
            "Python snippet:",
            python_block,
            "",
            "Response:",
            response_block,
        ]
        blocks.append("\n".join(sections))
    return "\n\n".join(blocks)


def _build_prompt(
    fmt: str,
    serialized_input: str,
    snippet: str,
    input_nodes: int,
    output_nodes: int,
    *,
    terminals_only: bool,
) -> str:
    fence = _format_code_block_language(fmt)
    data_block = f"```{fence}\n{serialized_input}\n```" if fence else f"```\n{serialized_input}\n```"
    python_block = f"```python\n{snippet}\n```"
    example_section = _example_blocks(fmt, terminals_only=terminals_only)
    fence_label = fence or fmt
    template = textwrap.dedent(
        """
        Format: {format_name}
        Input nodes observed: {input_nodes}
        Target output nodes: {output_nodes}

        Instructions:
        1. Parse the dataset into a Python variable named `data`.
        2. Execute the Python snippet below to populate a variable named `target`.
        3. Serialize `target` using the original format ({format_name}) and place the result inside a fenced code block tagged `{fence_label}`.
        4. The code block must contain only the serialized data.
        5. Be very careful to make sure the format and structure match exactly.
        {terminal_note}

        Examples:
        {example_section}

        Dataset:
        {data_block}

        Python snippet:
        {python_block}
        """
    ).strip()
    return template.format(
        format_name=fmt,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        data_block=data_block,
        python_block=python_block,
        fence_label=fence_label,
        example_section=example_section,
        terminal_note=(
            "Additional constraint: `target` must be a Python list containing only terminal values "
            "(no nested dictionaries or lists; use scalars such as strings, numbers, booleans, or null)."
            if terminals_only
            else ""
        ),
    )


def build_legibility_prompt_sample(
    *,
    fmt: str,
    input_nodes: int,
    output_nodes: int,
    seed: int,
    terminals_only: bool = False,
    wordlist_path: Path | None = None,
    words: Sequence[str] | None = None,
) -> LegibilityPromptSample:
    """Construct a reproducible prompt sample without contacting any model."""

    if input_nodes <= 0:
        raise ValueError("input_nodes must be positive")
    if output_nodes <= 0:
        raise ValueError("output_nodes must be positive")

    if words is None:
        path = wordlist_path or GenerationConfig().wordlist_path
        words = load_wordlist(path)

    rng = random.Random(seed)
    data = generate_nested(input_nodes, rng, words)
    actual_input_nodes = count_nodes(data)
    if actual_input_nodes != input_nodes:
        raise RuntimeError(
            f"generated dataset input nodes {actual_input_nodes} != requested {input_nodes}"
        )
    target_object, selected_paths = _build_target(
        data,
        rng,
        output_nodes,
        terminals_only=terminals_only,
    )
    snippet = _render_python_snippet(selected_paths)

    namespace = {"data": deepcopy(data)}
    exec(snippet, {}, namespace)
    expected_output = namespace.get("target")
    if expected_output != target_object:
        raise RuntimeError("generated target does not match snippet output")

    serialized_input = _serialize(fmt, data)
    expected_output_nodes = count_nodes(expected_output)
    user_message = _build_prompt(
        fmt,
        serialized_input,
        snippet,
        actual_input_nodes,
        expected_output_nodes,
        terminals_only=terminals_only,
    )
    prompt = LegibilityPrompt(
        format=fmt,
        user_message=user_message,
        expected_object=deepcopy(expected_output),
        input_node_count=actual_input_nodes,
        output_node_count=expected_output_nodes,
    )
    return LegibilityPromptSample(
        prompt=prompt,
        input_data=deepcopy(data),
        expected_output=deepcopy(expected_output),
        python_snippet=snippet,
        serialized_input=serialized_input,
    )


# --- OpenRouter integration -------------------------------------------------


class OpenRouterResponder:
    """Thin client for OpenRouter chat completions using the OpenAI SDK."""

    def __init__(self, *, model: str, temperature: float, timeout: float, system_prompt: str):
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._system_prompt = system_prompt
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable must be defined")

        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if referer:
            default_headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_X_TITLE")
        if title:
            default_headers["X-Title"] = title

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers or None,
        )

    async def complete(self, prompt: LegibilityPrompt) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt.user_message},
            ],
            timeout=self._timeout,
        )
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError) as exc:  # pragma: no cover
            raise OpenAIError(f"Malformed OpenRouter response: {response}") from exc

    async def aclose(self) -> None:
        await self._client.close()


# --- Benchmark runner -------------------------------------------------------


@dataclass
class TrialStats:
    successes: int = 0
    jaccard_sum: float = 0.0
    dictdiff_sum: float = 0.0
    attempts: int = 0
    input_nodes: List[int] = field(default_factory=list)
    output_nodes: List[int] = field(default_factory=list)

    def record(
        self,
        success: bool,
        jaccard: float,
        dictdiff: float,
        actual_input: int,
        actual_output: int,
    ) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        self.jaccard_sum += jaccard
        self.dictdiff_sum += dictdiff
        self.input_nodes.append(actual_input)
        self.output_nodes.append(actual_output)

    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def jaccard(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.jaccard_sum / self.attempts

    def dictdiff(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.dictdiff_sum / self.attempts


async def _run_single_trial(
    fmt: str,
    input_target: int,
    output_target: int,
    *,
    seed: int,
    words: Sequence[str],
    responder: OpenRouterResponder,
    terminals_only: bool,
) -> Tuple[bool, float, float, int, int, Optional[Mapping[str, object]], Mapping[str, object]]:
    sample = build_legibility_prompt_sample(
        fmt=fmt,
        input_nodes=input_target,
        output_nodes=output_target,
        seed=seed,
        terminals_only=terminals_only,
        words=words,
    )
    prompt = sample.prompt
    expected_object = sample.expected_output
    actual_input_nodes = prompt.input_node_count
    expected_output_nodes = prompt.output_node_count

    serialized_input = sample.serialized_input
    user_message = prompt.user_message
    observation: Optional[Mapping[str, object]] = None
    record: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "format": fmt,
        "input_target": input_target,
        "output_target": output_target,
        "input_nodes": actual_input_nodes,
        "output_nodes": expected_output_nodes,
        "seed": seed,
        "input_data": sample.input_data,
        "expected_output": expected_object,
        "python_snippet": sample.python_snippet,
        "serialized_input": serialized_input,
        "prompt": user_message,
        "prompt_length": len(user_message),
        "terminals_only": terminals_only,
    }
    try:
        response = responder.complete(prompt)
        response_text = await response if inspect.isawaitable(response) else response
        record["model_response"] = response_text
    except APITimeoutError as exc:
        observation = {
            "kind": "timeout",
            "message": f"{type(exc).__name__}: {exc}",
            "seed": seed,
        }
        parsed = None
        record["model_response"] = None
        record["response_length"] = 0
    except APIConnectionError as exc:
        observation = {
            "kind": "connection_error",
            "message": f"{type(exc).__name__}: {exc}",
            "seed": seed,
        }
        parsed = None
        record["model_response"] = None
        record["response_length"] = 0
    except APIStatusError as exc:
        observation = {
            "kind": "http_error",
            "status_code": exc.status_code,
            "message": f"{type(exc).__name__}: {exc}",
            "seed": seed,
        }
        parsed = None
        record["model_response"] = None
        record["response_length"] = 0
    except OpenAIError as exc:
        observation = {
            "kind": "openai_error",
            "message": f"{type(exc).__name__}: {exc}",
            "seed": seed,
        }
        parsed = None
        record["model_response"] = None
        record["response_length"] = 0
    except Exception as exc:  # pragma: no cover
        observation = {
            "kind": "request_error",
            "message": f"{type(exc).__name__}: {exc}",
            "seed": seed,
        }
        parsed = None
        record["model_response"] = None
        record["response_length"] = 0
    else:
        try:
            parsed = _parse(fmt, response_text)
        except Exception as exc:
            observation = {
                "kind": "parse_error",
                "message": f"{type(exc).__name__}: {exc}",
                "seed": seed,
            }
            parsed = None
            record["response_length"] = len(response_text)
        else:
            record["response_length"] = len(response_text)
    dictdiff_score, dictdiff_counts, dictdiff_mismatches = _dictdiff_score(expected_object, parsed)
    jaccard = _jaccard_index(expected_object, parsed)
    success = parsed == expected_object
    record["jaccard"] = jaccard
    record["dictdiff_score"] = dictdiff_score
    record["dictdiff_counts"] = dictdiff_counts
    record["dictdiff_mismatches"] = dictdiff_mismatches
    record["success"] = success
    record["observation"] = observation
    record["parsed_output"] = parsed
    return (
        success,
        jaccard,
        dictdiff_score,
        actual_input_nodes,
        expected_output_nodes,
        observation,
        record,
    )


async def _run_trials_for_combo(
    fmt: str,
    input_target: int,
    output_target: int,
    *,
    index: int,
    seeds: Sequence[int],
    words: Sequence[str],
    responder: OpenRouterResponder,
    terminals_only: bool,
) -> Tuple[int, Mapping[str, object], List[Mapping[str, object]]]:
    trial_results = await asyncio.gather(
        *(
            _run_single_trial(
                fmt,
                input_target,
                output_target,
                seed=seed,
                words=words,
                responder=responder,
                terminals_only=terminals_only,
            )
            for seed in seeds
        )
    )
    stats = TrialStats()
    observations: List[Mapping[str, object]] = []
    records: List[Mapping[str, object]] = []
    for trial_index, (
        success,
        jaccard,
        dictdiff_score,
        actual_input,
        actual_output,
        observation,
        record,
    ) in enumerate(trial_results):
        stats.record(success, jaccard, dictdiff_score, actual_input, actual_output)
        if observation:
            observations.append(
                {
                    "seed": observation.get("seed"),
                    "kind": observation.get("kind"),
                    "message": observation.get("message"),
                    "status_code": observation.get("status_code"),
                    "prompt_length": record.get("prompt_length"),
                    "response_length": record.get("response_length"),
                }
            )
        record = dict(record)
        record["trial_index"] = trial_index
        record["combo_index"] = index
        records.append(record)
    total_prompt_chars = sum(record.get("prompt_length", 0) for record in records)
    total_response_chars = sum(record.get("response_length", 0) for record in records)
    result = {
        "format": fmt,
        "input_nodes": input_target,
        "output_nodes": output_target,
        "num_trials": stats.attempts,
        "accuracy": stats.accuracy(),
        "jaccard": stats.jaccard(),
        "dictdiff": stats.dictdiff(),
        "mean_input_nodes": sum(stats.input_nodes) / max(len(stats.input_nodes), 1),
        "mean_output_nodes": sum(stats.output_nodes) / max(len(stats.output_nodes), 1),
        "total_prompt_chars": total_prompt_chars,
        "total_response_chars": total_response_chars,
        "terminals_only": terminals_only,
    }
    if observations:
        result["observations"] = observations
    return index, result, records


async def _run_legibility_benchmark_async(
    config: LegibilityBenchmarkConfig,
    *,
    responder: OpenRouterResponder,
    words: Sequence[str],
) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]]]:
    rng = random.Random(config.seed)
    seed_groups: Dict[Tuple[int, int], Sequence[int]] = {}
    for input_target in config.input_nodes:
        for output_target in config.output_nodes:
            seeds = tuple(rng.randint(0, 2**63 - 1) for _ in range(config.num_trials))
            seed_groups[(input_target, output_target)] = seeds

    combos: List[Tuple[str, int, int, Sequence[int]]] = []
    for fmt in config.formats:
        for input_target in config.input_nodes:
            for output_target in config.output_nodes:
                combos.append((fmt, input_target, output_target, seed_groups[(input_target, output_target)]))
    tasks: List[asyncio.Task[Tuple[int, Mapping[str, object], List[Mapping[str, object]]]]] = []
    for index, (fmt, input_target, output_target, seeds) in enumerate(combos):
        tasks.append(
            asyncio.create_task(
                _run_trials_for_combo(
                    fmt,
                    input_target,
                    output_target,
                    index=index,
                    seeds=seeds,
                    words=words,
                    responder=responder,
                    terminals_only=config.terminals_only,
                )
            )
        )
    results: List[Mapping[str, object] | None] = [None] * len(tasks)
    all_records: List[Mapping[str, object]] = []
    disable_progress = not sys.stderr.isatty()
    with tqdm(
        total=len(tasks),
        desc="Legibility Trials",
        unit="combo",
        disable=disable_progress,
    ) as progress:
        for completed in asyncio.as_completed(tasks):
            index, result, records = await completed
            results[index] = result
            all_records.extend(records)
            progress.update()
    ordered_results = [result for result in results if result is not None]
    return ordered_results, all_records


def _write_trial_records(records: Sequence[Mapping[str, object]]) -> Path:
    log_dir = Path("data") / "legibility_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"trials_{timestamp}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in records:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
    return path


def _render_failure_diffs(records: Sequence[Mapping[str, object]]) -> None:
    failures = [record for record in records if record.get("success") is False]
    if not failures:
        return

    console = Console()
    console.print("\n[bold red]Failed Trial Diffs[/bold red]")

    def _format_payload(payload: object) -> list[str]:
        try:
            serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialized = repr(payload)
        return serialized.splitlines()

    for record in failures:
        header = (
            f"format={record.get('format')} seed={record.get('seed')} "
            f"trial={record.get('trial_index')} combo={record.get('combo_index')} "
            f"input_nodes={record.get('input_nodes')} output_nodes={record.get('output_nodes')}"
        )
        console.print(f"[bold]{header}[/bold]")

        observation = record.get("observation") or {}
        message = observation.get("message")
        parsed = record.get("parsed_output")
        expected = record.get("expected_output")

        if parsed is None:
            if message:
                console.print(f"  Reason: {message}")
            else:
                console.print("  No parsed output available.")
            response = record.get("model_response")
            if response:
                console.print("  Model response:")
                console.print(textwrap.indent(response, "    "))
            console.print("")
            continue

        expected_lines = _format_payload(expected)
        actual_lines = _format_payload(parsed)
        diff_lines = list(
            difflib.unified_diff(
                expected_lines,
                actual_lines,
                fromfile="expected",
                tofile="actual",
                lineterm="",
            )
        )
        if not diff_lines:
            console.print("  No textual diff – check data structures for subtle differences.\n")
            continue
        console.print("  Diff:")
        console.print(textwrap.indent("\n".join(diff_lines), "    "))
        console.print("")


async def arun_legibility_benchmark(
    config: LegibilityBenchmarkConfig,
    *,
    responder: Optional[OpenRouterResponder] = None,
) -> Mapping[str, object]:
    """Asynchronously execute the legibility benchmark and return a dataset dictionary."""
    config.ensure_valid()
    _log_configuration(config)
    words = load_wordlist(config.wordlist_path)
    if responder is None:
        responder = OpenRouterResponder(
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            system_prompt=config.system_prompt,
        )

    results, trial_records = await _run_legibility_benchmark_async(
        config,
        responder=responder,
        words=words,
    )
    trial_log_path = _write_trial_records(trial_records)

    dataset = {
        "seed": config.seed,
        "formats": list(config.formats),
        "input_nodes": list(config.input_nodes),
        "output_nodes": list(config.output_nodes),
        "num_trials": config.num_trials,
        "terminals_only": config.terminals_only,
        "model": config.model,
        "temperature": config.temperature,
        "results": results,
        "trial_log_path": str(trial_log_path),
        "trial_record_count": len(trial_records),
    }
    _render_failure_diffs(trial_records)
    _render_summary_table(results)
    return dataset


def run_legibility_benchmark(
    config: LegibilityBenchmarkConfig,
    *,
    responder: Optional[OpenRouterResponder] = None,
) -> Mapping[str, object]:
    """Execute the legibility benchmark and return a dataset dictionary."""

    return asyncio.run(arun_legibility_benchmark(config, responder=responder))


def write_legibility_dataset(dataset: Mapping[str, object], path: Path) -> None:
    """Persist benchmark output to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")


__all__ = [
    "LegibilityBenchmarkConfig",
    "LegibilityPrompt",
    "LegibilityPromptSample",
    "OpenRouterResponder",
    "build_legibility_prompt_sample",
    "arun_legibility_benchmark",
    "run_legibility_benchmark",
    "write_legibility_dataset",
]
