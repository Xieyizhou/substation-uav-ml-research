"""Bounded, non-mutating readers for JSON and text artifacts."""

from __future__ import annotations

from collections import deque
import json
from itertools import islice
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def iter_jsonl(path: Path):
    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    with source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed {path.name} line {number}") from error
            if isinstance(value, dict):
                yield value


def jsonl_page(path: Path, offset: int, limit: int) -> tuple[dict, ...]:
    return tuple(islice(iter_jsonl(path), offset, offset + limit))


def jsonl_count(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def bounded_tail(path: Path, limit: int) -> tuple[tuple[int, str], ...]:
    if limit < 1 or limit > 1000:
        raise ValueError("line limit must be between 1 and 1000")
    try:
        with path.open(encoding="utf-8", errors="replace") as source:
            lines = deque(enumerate(source, 1), maxlen=limit)
    except FileNotFoundError:
        return ()
    except OSError as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    return tuple((number, text.rstrip("\r\n")) for number, text in lines)
