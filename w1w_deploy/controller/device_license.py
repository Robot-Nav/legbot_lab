#!/usr/bin/env python3
"""Device binding and in-memory policy decryption for sealed W1W builds."""

from __future__ import annotations

import hashlib
import hmac
import re
import sys
from pathlib import Path


# The sealed builder replaces both placeholders before compiling this module.
# The source tree remains usable for unit tests, but this file is never shipped.
AUTHORIZED_CPU_DIGEST_HEX = "__W1W_AUTHORIZED_CPU_DIGEST_HEX__"
MODEL_MASTER_KEY_HEX = "__W1W_MODEL_MASTER_KEY_HEX__"

_DEVICE_DOMAIN = b"w1w-device-v1\0"
_MODEL_DOMAIN = b"w1w-model-key-v1\0"
_MODEL_MAGIC = b"W1WENC1\0"
_MODEL_AAD = b"w1w-policy-v1"
_SERIAL_PATTERN = re.compile(r"[0-9a-f]{16}")


def _is_sealed() -> bool:
    return not AUTHORIZED_CPU_DIGEST_HEX.startswith("__W1W_")


def _read_cpu_serial() -> str | None:
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name.strip() == "Serial":
            serial = value.strip().lower()
            if _SERIAL_PATTERN.fullmatch(serial) and serial != "0000000000000000":
                return serial
            return None
    return None


def require_authorized(component: str) -> str:
    """Return the CPU serial or terminate before any robot I/O."""
    if not _is_sealed():
        return "development"
    serial = _read_cpu_serial()
    try:
        expected = bytes.fromhex(AUTHORIZED_CPU_DIGEST_HEX)
    except ValueError:
        expected = b""
    actual = hashlib.sha256(_DEVICE_DOMAIN + (serial or "").encode("ascii")).digest()
    if serial is None or len(expected) != 32 or not hmac.compare_digest(expected, actual):
        print(f"{component}: unauthorized device", file=sys.stderr, flush=True)
        raise SystemExit(77)
    return serial


def decrypt_model(model_path: str) -> bytes:
    """Decrypt a sealed policy directly into memory for ONNX Runtime."""
    path = Path(model_path)
    blob = path.read_bytes()
    if not _is_sealed():
        return blob

    serial = require_authorized("w1w-controller")
    if len(blob) < len(_MODEL_MAGIC) + 12 + 16 or not blob.startswith(_MODEL_MAGIC):
        raise RuntimeError("sealed policy has an invalid format")
    try:
        master_key = bytes.fromhex(MODEL_MASTER_KEY_HEX)
    except ValueError as error:
        raise RuntimeError("sealed policy key is invalid") from error
    if len(master_key) != 32:
        raise RuntimeError("sealed policy key is invalid")

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce_start = len(_MODEL_MAGIC)
    nonce = blob[nonce_start:nonce_start + 12]
    ciphertext = blob[nonce_start + 12:]
    key = hashlib.sha256(
        _MODEL_DOMAIN + master_key + b"\0" + serial.encode("ascii")
    ).digest()
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, _MODEL_AAD)
    except Exception as error:
        raise RuntimeError("sealed policy authentication failed") from error
