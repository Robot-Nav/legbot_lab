"""CAN-ID conversion helpers for RS02 serial USB-CAN framing."""

from __future__ import annotations


def ext_raw_to_can29(ext_raw_4b: bytes, shift_right: int = 3) -> int:
    """Convert serial 4-byte extended field to true 29-bit CAN ID."""
    if len(ext_raw_4b) != 4:
        raise ValueError("ext_raw_4b must be exactly 4 bytes")
    ext_u32 = int.from_bytes(ext_raw_4b, byteorder="big", signed=False)
    can29 = (ext_u32 >> shift_right) & 0x1FFFFFFF
    return can29


def can29_to_ext_raw(can29: int, shift_left: int = 3) -> bytes:
    """Convert true 29-bit CAN ID to serial 4-byte extended field."""
    can29 &= 0x1FFFFFFF
    ext_u32 = (can29 << shift_left) & 0xFFFFFFFF
    return ext_u32.to_bytes(4, byteorder="big", signed=False)
