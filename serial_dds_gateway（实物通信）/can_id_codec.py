# 用途：RS02 USB-CAN 串口帧中 29 位 CAN ID 与 4 字节扩展原始字段的互转辅助函数。
# 说明：灵足旧工具暴露的是移位后的 raw 字段，右移 shift_right 位后取低 29 位即为真实 CAN ID。
"""CAN-ID 转换辅助函数（RS02 串口 USB-CAN 成帧）。"""

from __future__ import annotations


def ext_raw_to_can29(ext_raw_4b: bytes, shift_right: int = 3) -> int:
    """将串口 4 字节扩展字段转换为真实 29 位 CAN ID。"""
    if len(ext_raw_4b) != 4:
        raise ValueError("ext_raw_4b 必须为 4 字节")
    ext_u32 = int.from_bytes(ext_raw_4b, byteorder="big", signed=False)
    can29 = (ext_u32 >> shift_right) & 0x1FFFFFFF
    return can29


def can29_to_ext_raw(can29: int, shift_left: int = 3) -> bytes:
    """将真实 29 位 CAN ID 转换为串口 4 字节扩展字段。"""
    can29 &= 0x1FFFFFFF
    ext_u32 = (can29 << shift_left) & 0xFFFFFFFF
    return ext_u32.to_bytes(4, byteorder="big", signed=False)
