"""Random data generation and token counting for token efficiency experiments."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence
from xml.etree import ElementTree as ET

import tomli_w
import yaml

# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerSpec:
    """Describe a tokenizer to load from Hugging Face."""

    name: str
    pretrained: str
    revision: str | None = None


DEFAULT_TOKENIZER_SPECS: Sequence[TokenizerSpec] = (
    TokenizerSpec(name="gpt-oss", pretrained="openai-community/gpt2"),
    TokenizerSpec(name="qwen3", pretrained="Qwen/Qwen2-0.5B-Instruct"),
    TokenizerSpec(name="llama3.2", pretrained="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
)

DEFAULT_SHAPES: Sequence[str] = ("nested", "tabular", "sparse")
DEFAULT_FORMATS: Sequence[str] = ("json_min", "yaml_block", "toml", "xml", "csv")
DEFAULT_SIZES: Sequence[int] = (31, 63, 127)


@dataclass
class GenerationConfig:
    """Runtime configuration for dataset generation."""

    seed: int = 54873
    samples_per_size: int = 4
    shapes: Sequence[str] = field(default_factory=lambda: DEFAULT_SHAPES)
    formats: Sequence[str] = field(default_factory=lambda: DEFAULT_FORMATS)
    sizes: Sequence[int] = field(default_factory=lambda: DEFAULT_SIZES)
    tokenizer_specs: Sequence[TokenizerSpec] = field(
        default_factory=lambda: DEFAULT_TOKENIZER_SPECS
    )
    cache_dir: Path = Path("data") / "cache"
    wordlist_path: Path = Path("resources") / "system_words.txt"

    def ensure_valid(self) -> None:
        if not self.sizes:
            raise ValueError("at least one size must be provided")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_wordlist(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"word list not found: {path}")
    words = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not words:
        raise ValueError(f"word list {path} is empty")
    return words


def random_identifier(rng: random.Random, words: Sequence[str], *, min_parts: int = 1,
                      max_parts: int = 3) -> str:
    parts = [rng.choice(words) for _ in range(rng.randint(min_parts, max_parts))]
    return "_".join(part.lower() for part in parts)


def random_scalar(rng: random.Random, words: Sequence[str]) -> object:
    choice = rng.random()
    if choice < 0.4:
        return random_identifier(rng, words, min_parts=1, max_parts=2)
    if choice < 0.7:
        return rng.randint(-200, 200)
    if choice < 0.9:
        return round(rng.uniform(-500.0, 500.0), 3)
    return bool(rng.randint(0, 1))


def count_nodes(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(count_nodes(v) for v in value.values())
    if isinstance(value, list):
        return 1 + sum(count_nodes(v) for v in value)
    return 1


def integer_partition(total: int, parts: int, rng: random.Random) -> List[int]:
    buckets = [1] * parts
    remaining = total - parts
    for _ in range(remaining):
        buckets[rng.randrange(parts)] += 1
    return buckets


# ---------------------------------------------------------------------------
# Shape generators
# ---------------------------------------------------------------------------


def build_nested_node(remaining: int, rng: random.Random, words: Sequence[str]) -> object:
    if remaining <= 1:
        return random_scalar(rng, words)

    max_children = min(5, remaining - 1)
    num_children = rng.randint(1, max_children)
    partitions = integer_partition(remaining - 1, num_children, rng)

    choice = rng.random()
    if choice < 0.5:
        result: Dict[str, object] = {}
        used_keys: set[str] = set()
        for share in partitions:
            key = random_identifier(rng, words, min_parts=1, max_parts=3)
            while key in used_keys:
                key = random_identifier(rng, words, min_parts=1, max_parts=3)
            used_keys.add(key)
            result[key] = build_nested_node(share, rng, words)
        return result

    return [build_nested_node(share, rng, words) for share in partitions]


def generate_nested(size: int, rng: random.Random, words: Sequence[str]) -> object:
    return build_nested_node(max(size, 1), rng, words)


def generate_tabular(size: int, rng: random.Random, words: Sequence[str]) -> Mapping[str, object]:
    columns = max(2, min(8, rng.randint(3, 6)))
    rows = max(2, (size // (columns + 1)) or 2)
    column_names = []
    while len(column_names) < columns:
        candidate = random_identifier(rng, words, min_parts=1, max_parts=2)
        if candidate not in column_names:
            column_names.append(candidate)
    rows_data = []
    for idx in range(rows):
        row = {}
        for col in column_names:
            row[col] = random_scalar(rng, words)
        row["row_id"] = idx
        rows_data.append(row)
    return {"columns": column_names + ["row_id"], "rows": rows_data}


def generate_sparse(size: int, rng: random.Random, words: Sequence[str]) -> Mapping[str, object]:
    dim_rows = rng.randint(4, 10)
    dim_cols = rng.randint(4, 10)
    max_entries = dim_rows * dim_cols
    desired = max(3, min(size, max_entries // 2))

    used_positions: set[tuple[int, int]] = set()
    entries = []
    while len(entries) < desired:
        position = (rng.randrange(dim_rows), rng.randrange(dim_cols))
        if position in used_positions:
            continue
        used_positions.add(position)
        entries.append(
            {
                "coordinate": {"row": position[0], "col": position[1]},
                "value": random_scalar(rng, words),
            }
        )
    default_value = 0 if rng.random() < 0.5 else ""
    return {
        "dimensions": {"rows": dim_rows, "cols": dim_cols},
        "default": default_value,
        "non_zero": entries,
        "description": random_identifier(rng, words, min_parts=2, max_parts=4),
    }


SHAPE_BUILDERS: Mapping[str, Callable[[int, random.Random, Sequence[str]], object]] = {
    "nested": generate_nested,
    "tabular": generate_tabular,
    "sparse": generate_sparse,
}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class SerializationError(RuntimeError):
    """Raised when a data structure cannot be serialized for a format."""


def serialize_json_min(data: object) -> str:
    payload = {"results": data}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def serialize_yaml_block(data: object) -> str:
    payload = {"results": data}
    return yaml.safe_dump(payload, sort_keys=False)


def serialize_toml(shape: str, data: object) -> str:
    payload = {"results": data}
    return tomli_w.dumps(payload)


def sanitize_tag(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    if not cleaned:
        cleaned = "item"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def build_xml_element(name: str, value: object) -> ET.Element:
    elem = ET.Element(name)
    if isinstance(value, dict):
        for key, child in value.items():
            child_name = sanitize_tag(str(key))
            elem.append(build_xml_element(child_name, child))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_elem = build_xml_element("item", child)
            child_elem.set("index", str(index))
            elem.append(child_elem)
    else:
        elem.text = str(value)
    return elem


def serialize_xml(shape: str, data: object) -> str:
    if shape == "tabular" and isinstance(data, Mapping):
        root = ET.Element("table")
        columns_elem = ET.SubElement(root, "columns")
        for column in data.get("columns", []):
            col_elem = ET.SubElement(columns_elem, "column")
            col_elem.set("name", str(column))
        rows_elem = ET.SubElement(root, "rows")
        for row in data.get("rows", []):
            row_elem = ET.SubElement(rows_elem, "row")
            if isinstance(row, Mapping):
                for key, value in row.items():
                    cell = build_xml_element("cell", value)
                    cell.set("column", str(key))
                    row_elem.append(cell)
            else:
                row_elem.text = str(row)
        return ET.tostring(root, encoding="unicode")
    if shape == "sparse" and isinstance(data, Mapping):
        root = ET.Element("sparse_matrix")
        dims = data.get("dimensions", {})
        dims_elem = ET.SubElement(root, "dimensions")
        if isinstance(dims, Mapping):
            for key, value in dims.items():
                dim_elem = ET.SubElement(dims_elem, "dimension")
                dim_elem.set("name", str(key))
                dim_elem.text = str(value)
        default_elem = ET.SubElement(root, "default")
        default_elem.text = str(data.get("default", ""))
        entries_elem = ET.SubElement(root, "entries")
        for entry in data.get("non_zero", []):
            entry_elem = ET.SubElement(entries_elem, "entry")
            if isinstance(entry, Mapping):
                coord = entry.get("coordinate", {})
                if isinstance(coord, Mapping):
                    for key, value in coord.items():
                        entry_elem.set(str(key), str(value))
                value_elem = ET.SubElement(entry_elem, "value")
                value_elem.text = str(entry.get("value", ""))
            else:
                entry_elem.text = str(entry)
        description = data.get("description")
        if description is not None:
            desc_elem = ET.SubElement(root, "description")
            desc_elem.text = str(description)
        return ET.tostring(root, encoding="unicode")

    root = build_xml_element("root", data)
    return ET.tostring(root, encoding="unicode")


def serialize_csv(shape: str, data: object) -> str:
    if shape != "tabular":
        raise SerializationError("CSV is only supported for tabular shape")
    table = data
    if not isinstance(table, Mapping):
        raise SerializationError("tabular data must be a mapping with rows")
    rows = table.get("rows")
    columns = table.get("columns")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise SerializationError("tabular rows/columns missing")
    header = ",".join(columns)
    body_lines = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SerializationError("tabular row must be mapping")
        line = ",".join(str(row.get(col, "")) for col in columns)
        body_lines.append(line)
    return "\n".join([header, *body_lines])


SERIALIZERS: Mapping[str, Callable[[str, object], str]] = {
    "json_min": lambda shape, data: serialize_json_min(data),
    "yaml_block": lambda shape, data: serialize_yaml_block(data),
    "toml": serialize_toml,
    "xml": serialize_xml,
    "csv": serialize_csv,
}


# ---------------------------------------------------------------------------
# Tokenizer wrappers
# ---------------------------------------------------------------------------


class TokenCounter:
    """Lightweight wrapper for Hugging Face tokenizers."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            # Some tokenizers return batched results even for single strings.
            input_ids = input_ids[0]
        return len(input_ids)


def load_tokenizers(specs: Sequence[TokenizerSpec], *, cache_dir: Path | None = None
                    ) -> Mapping[str, TokenCounter]:
    from transformers import AutoTokenizer

    counters: Dict[str, TokenCounter] = {}
    for spec in specs:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.pretrained,
            revision=spec.revision,
            use_fast=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        # Avoid spurious warnings when counting tokens longer than model context.
        try:
            tokenizer.model_max_length = sys.maxsize
        except Exception:
            pass
        counters[spec.name] = TokenCounter(tokenizer)
    return counters


# ---------------------------------------------------------------------------
# Sample caching
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    data: object
    node_count: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Sample":
        return cls(data=raw["data"], node_count=int(raw["node_count"]))

    def to_dict(self) -> Mapping[str, object]:
        return {"data": self.data, "node_count": self.node_count}


class SampleCache:
    """Persist randomly generated samples keyed by shape and size."""

    def __init__(self, path: Path):
        self._path = path
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {}
        self._store: Dict[str, Dict[str, List[Mapping[str, object]]]] = {
            shape: {size: list(entries) for size, entries in sizes.items()}
            for shape, sizes in raw.items()
        }
        self._dirty = False

    def ensure_samples(
        self,
        *,
        shape: str,
        size: int,
        count: int,
        generator: Callable[[], object],
    ) -> List[Sample]:
        shape_bucket = self._store.setdefault(shape, {})
        size_key = str(size)
        samples = shape_bucket.setdefault(size_key, [])
        while len(samples) < count:
            value = generator()
            sample = Sample(data=value, node_count=count_nodes(value))
            samples.append(sample.to_dict())
            self._dirty = True
        selected = samples[:count]
        return [Sample.from_dict(entry) for entry in selected]

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            shape: {size: entries for size, entries in sizes.items()}
            for shape, sizes in self._store.items()
        }
        self._path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        self._dirty = False


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_samples(config: GenerationConfig, rng: random.Random, words: Sequence[str],
                     cache: SampleCache) -> Dict[str, List[Sample]]:
    samples_by_shape: Dict[str, List[Sample]] = {}
    for shape in config.shapes:
        builder = SHAPE_BUILDERS.get(shape)
        if builder is None:
            raise ValueError(f"unsupported shape: {shape}")
        collected: List[Sample] = []
        for size in config.sizes:
            target = config.samples_per_size
            generated = cache.ensure_samples(
                shape=shape,
                size=size,
                count=target,
                generator=lambda b=builder, s=size: b(s, rng, words),
            )
            collected.extend(generated)
        samples_by_shape[shape] = collected
    cache.save()
    return samples_by_shape


def generate_token_efficiency_dataset(
    *,
    config: GenerationConfig,
    tokenizers: Mapping[str, TokenCounter] | None = None,
    cache_dir: Path | None = None,
) -> Mapping[str, object]:
    config.ensure_valid()
    rng = random.Random(config.seed)
    words = load_wordlist(config.wordlist_path)

    cache_path = (cache_dir or config.cache_dir) / f"samples_seed_{config.seed}.json"
    cache = SampleCache(cache_path)
    samples_by_shape = generate_samples(config, rng, words, cache)

    if tokenizers is None:
        tokenizers = load_tokenizers(config.tokenizer_specs)

    observations: List[Mapping[str, object]] = []
    raw_data: List[str] = []
    for shape, samples in samples_by_shape.items():
        for fmt in config.formats:
            serializer = SERIALIZERS.get(fmt)
            if serializer is None:
                raise ValueError(f"unsupported serialization format: {fmt}")
            total_nodes = 0
            token_totals = {name: 0 for name in tokenizers}
            used = 0
            for sample in samples:
                try:
                    rendered = serializer(shape, sample.data)
                except SerializationError:
                    continue
                total_nodes += sample.node_count
                used += 1
                for name, counter in tokenizers.items():
                    token_totals[name] += counter.count(rendered)
            if used == 0 or total_nodes == 0:
                continue
            for name, total_tokens in token_totals.items():
                tokens_per_node = total_tokens / total_nodes
                observations.append(
                    {
                        "shape": shape,
                        "format": fmt,
                        "tokenizer": name,
                        "tokens_per_node": tokens_per_node,
                        "samples": used,
                        "total_nodes": total_nodes,
                        "total_tokens": total_tokens,
                    }
                )
    observations.sort(key=lambda entry: (entry["shape"], entry["format"], entry["tokenizer"]))

    dataset = {
        "seed": config.seed,
        "samples_per_size": config.samples_per_size,
        "shapes": list(config.shapes),
        "formats": list(config.formats),
        "tokenizers": [spec.name for spec in config.tokenizer_specs],
        "sizes": list(config.sizes),
        "observations": observations,
    }
    return dataset


def write_dataset(dataset: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def generate_and_write_dataset(
    *,
    config: GenerationConfig,
    output_path: Path,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Mapping[str, object]:
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        metadata_matches = (
            existing.get("seed") == config.seed
            and existing.get("samples_per_size") == config.samples_per_size
            and existing.get("sizes") == list(config.sizes)
            and existing.get("tokenizers") == [spec.name for spec in config.tokenizer_specs]
        )
        if metadata_matches:
            return existing

    dataset = generate_token_efficiency_dataset(config=config, cache_dir=cache_dir)
    write_dataset(dataset, output_path)
    return dataset


__all__ = [
    "GenerationConfig",
    "TokenizerSpec",
    "generate_token_efficiency_dataset",
    "generate_and_write_dataset",
    "write_dataset",
    "DEFAULT_TOKENIZER_SPECS",
]
