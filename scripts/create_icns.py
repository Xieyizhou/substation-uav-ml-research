#!/usr/bin/env python3
"""Create a modern ICNS container from a standard macOS iconset."""

from __future__ import annotations

from pathlib import Path
import struct
import sys


ICNS_ENTRIES = (
    (b"icp4", "icon_16x16.png"),
    (b"icp5", "icon_32x32.png"),
    (b"icp6", "icon_32x32@2x.png"),
    (b"ic07", "icon_128x128.png"),
    (b"ic08", "icon_256x256.png"),
    (b"ic09", "icon_512x512.png"),
    (b"ic10", "icon_512x512@2x.png"),
)


def create_icns(iconset: Path, destination: Path) -> None:
    body = bytearray()
    for entry_type, name in ICNS_ENTRIES:
        payload = (iconset / name).read_bytes()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"icon entry is not PNG: {name}")
        body.extend(entry_type)
        body.extend(struct.pack(">I", len(payload) + 8))
        body.extend(payload)
    destination.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: create_icns.py ICONSET OUTPUT.icns", file=sys.stderr)
        return 2
    create_icns(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
