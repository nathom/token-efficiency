import pytest

from token_efficiency.cli import parse_int_sequence


def test_parse_int_sequence_defaults() -> None:
    assert parse_int_sequence(None, default=(1, 2)) == (1, 2)


def test_parse_int_sequence_simple_values() -> None:
    assert parse_int_sequence(["3", "4"], default=(0,)) == (3, 4)


def test_parse_int_sequence_comma_separated_values() -> None:
    assert parse_int_sequence(["5, 6,7"], default=(0,)) == (5, 6, 7)


def test_parse_int_sequence_mixed_values() -> None:
    assert parse_int_sequence(["8", "9, 10"], default=(0,)) == (8, 9, 10)


def test_parse_int_sequence_invalid_value() -> None:
    with pytest.raises(ValueError):
        parse_int_sequence(["oops"], default=(0,))
