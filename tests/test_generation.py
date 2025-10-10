from __future__ import annotations

import random
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from token_efficiency.data_generation import (
    GenerationConfig,
    TokenizerSpec,
    count_nodes,
    generate_nested,
    generate_token_efficiency_dataset,
)


class StubCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def test_generate_dataset_with_stub_tokenizer(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    config = GenerationConfig(
        seed=123,
        samples_per_size=1,
        shapes=("nested", "tabular"),
        formats=("json_min", "csv"),
        sizes=(7,),
        tokenizer_specs=(TokenizerSpec(name="stub", pretrained="stub"),),
        cache_dir=cache_dir,
        wordlist_path=Path("resources") / "system_words.txt",
    )
    counters = {"stub": StubCounter()}
    dataset = generate_token_efficiency_dataset(
        config=config,
        tokenizers=counters,
        cache_dir=cache_dir,
    )
    assert dataset["tokenizers"] == ["stub"]
    assert dataset["sizes"] == [7]
    assert len(dataset["observations"]) >= 1
    observation = dataset["observations"][0]
    assert observation["tokens_per_node"] > 0
    assert (cache_dir / "samples_seed_123.json").exists()

    repeat = generate_token_efficiency_dataset(
        config=config,
        tokenizers=counters,
        cache_dir=cache_dir,
    )
    assert dataset == repeat


def test_generate_nested_respects_requested_size() -> None:
    words = ["alpha", "beta", "gamma", "delta"]
    rng = random.Random(123)
    for size in range(1, 32):
        data = generate_nested(size, rng, words)
        assert count_nodes(data) == size
