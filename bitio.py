"""
arithmetic_coder.Encoder writes one base-`base` digit at a time via a
callback; arithmetic_coder.Decoder reads them back the same way. With
base=2 those digits are bits. These two classes are that callback, packing
bits into bytes on write and unpacking them back out on read.
"""

from __future__ import annotations


class BitWriter:
    """Collects individual bits, MSB-first, into a bytes buffer."""

    def __init__(self):
        self._bytes = bytearray()
        self._current = 0
        self._num_bits = 0

    def write_bit(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._num_bits += 1
        if self._num_bits == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._num_bits = 0

    def getvalue(self) -> bytes:
        """Flush any partial final byte (zero-padded) and return all bytes."""
        if self._num_bits > 0:
            padded = self._current << (8 - self._num_bits)
            return bytes(self._bytes) + bytes([padded])
        return bytes(self._bytes)


class BitReader:
    """Yields individual bits, MSB-first, from a bytes buffer.

    Returns None once exhausted -- arithmetic_coder.Decoder relies on this
    to detect end-of-stream and pad internally, per its own docstring.
    """

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0  # 0 = most significant bit of current byte

    def read_bit(self) -> int | None:
        if self._byte_pos >= len(self._data):
            return None
        byte = self._data[self._byte_pos]
        bit = (byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit