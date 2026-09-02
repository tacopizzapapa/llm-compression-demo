"""
Minimal container format for a compressed file:

    [4 bytes: big-endian uint32 header length N]
    [N bytes: UTF-8 JSON header]
    [remaining bytes: the arithmetic-coded payload]

The header carries everything decompress.py needs that isn't recoverable
from the payload itself: which checkpoint produced the pdfs, the coder's
base/precision, and how many tokens to decode before stopping.
"""

from __future__ import annotations

import json
import struct

_LEN_STRUCT = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix


def write_container(path: str, meta: dict, payload: bytes) -> None:
    header = json.dumps(meta).encode("utf-8")
    with open(path, "wb") as f:
        f.write(_LEN_STRUCT.pack(len(header)))
        f.write(header)
        f.write(payload)


def read_container(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        (header_len,) = _LEN_STRUCT.unpack(f.read(_LEN_STRUCT.size))
        meta = json.loads(f.read(header_len).decode("utf-8"))
        payload = f.read()
    return meta, payload