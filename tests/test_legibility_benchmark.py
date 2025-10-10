from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from token_efficiency.legibility import (
    LegibilityBenchmarkConfig,
    LegibilityPrompt,
    run_legibility_benchmark,
)
from token_efficiency.data_generation import SERIALIZERS


class EchoResponder:
    async def complete(self, prompt):
        serializer = SERIALIZERS[prompt.format]
        return serializer("nested", prompt.expected_object)


class AssertingResponder(EchoResponder):
    def __init__(self, validator: Callable[[LegibilityPrompt], None]):
        self._validator = validator

    async def complete(self, prompt):
        self._validator(prompt)
        return await super().complete(prompt)


def test_legibility_benchmark_offline(tmp_path) -> None:
    config = LegibilityBenchmarkConfig(
        seed=123,
        formats=("json_min", "yaml_block"),
        input_nodes=(12,),
        output_nodes=(6,),
        num_trials=3,
    )
    dataset = run_legibility_benchmark(config, responder=EchoResponder())
    assert dataset["formats"] == ["json_min", "yaml_block"]
    assert dataset["model"] == config.model

    for entry in dataset["results"]:
        assert entry["accuracy"] == 1.0
        assert entry["jaccard"] == 1.0
        assert entry["dictdiff"] == 1.0
        assert entry["num_trials"] == config.num_trials
        assert entry["format"] in {"json_min", "yaml_block"}
        assert entry["input_nodes"] == 12
        assert entry["output_nodes"] == 6


def test_legibility_benchmark_terminals_only(tmp_path) -> None:
    def validate(prompt: LegibilityPrompt) -> None:
        assert isinstance(prompt.expected_object, list)
        assert prompt.expected_object
        for value in prompt.expected_object:
            assert not isinstance(value, (dict, list))

    config = LegibilityBenchmarkConfig(
        seed=321,
        formats=("json_min",),
        input_nodes=(15,),
        output_nodes=(5,),
        num_trials=2,
        terminals_only=True,
    )
    dataset = run_legibility_benchmark(config, responder=AssertingResponder(validate))
    assert dataset["terminals_only"] is True
    for entry in dataset["results"]:
        assert entry["terminals_only"] is True
        assert entry["jaccard"] == 1.0
        assert entry["dictdiff"] == 1.0


def test_legibility_benchmark_toml_terminals_only(tmp_path) -> None:
    config = LegibilityBenchmarkConfig(
        seed=777,
        formats=("toml",),
        input_nodes=(20,),
        output_nodes=(4,),
        num_trials=2,
        terminals_only=True,
    )
    dataset = run_legibility_benchmark(config, responder=EchoResponder())
    assert dataset["terminals_only"] is True
    for entry in dataset["results"]:
        assert entry["format"] == "toml"
        assert entry["accuracy"] == 1.0
        assert entry["jaccard"] == 1.0
        assert entry["dictdiff"] == 1.0


def test_legibility_shared_inputs_across_formats(tmp_path) -> None:
    config = LegibilityBenchmarkConfig(
        seed=999,
        formats=("json_min", "yaml_block", "toml"),
        input_nodes=(12,),
        output_nodes=(4,),
        num_trials=3,
    )
    dataset = run_legibility_benchmark(config, responder=EchoResponder())

    log_path = Path(dataset["trial_log_path"])
    assert log_path.exists()

    by_trial: dict[tuple[int, int, int], list[dict]] = {}
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw_line)
        key = (record["input_nodes"], record["output_nodes"], record["trial_index"])
        by_trial.setdefault(key, []).append(record)

    for records in by_trial.values():
        assert len(records) == len(config.formats)
        baseline = records[0]
        for record in records[1:]:
            assert record["input_data"] == baseline["input_data"]
            assert record["expected_output"] == baseline["expected_output"]
            assert "jaccard" in record
            assert 0.0 <= record["jaccard"] <= 1.0
            assert "dictdiff_score" in record
            assert 0.0 <= record["dictdiff_score"] <= 1.0
