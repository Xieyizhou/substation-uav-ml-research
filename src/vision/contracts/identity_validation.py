"""Shared validation primitives for visual artifact identities."""

from pathlib import PurePath
import re


VISUAL_IDENTITY_SCHEMA_VERSION = 1
DATASET_ROLES = frozenset(
    {
        "pilot",
        "development",
        "validation",
        "held_out_test",
        "formal",
        "external_real_image",
    }
)
PRECISIONS = frozenset({"fp32", "fp16", "int8"})
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def required_text(value, name):
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if (
        PurePath(value).is_absolute()
        or value.startswith("~/")
        or _WINDOWS_ABSOLUTE_PATTERN.match(value)
    ):
        raise ValueError(f"{name} must not contain an absolute path")
    return value


def optional_text(value, name):
    return None if value is None else required_text(value, name)


def sha256(value, name, *, optional=False):
    if value is None and optional:
        return None
    value = str(value)
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def non_negative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def stable_tuple(values, name, *, cast=str, allow_empty=False):
    result = tuple(cast(value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if cast is str:
            required_text(value, name)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result
