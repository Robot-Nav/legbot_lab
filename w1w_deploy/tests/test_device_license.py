import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controller"
sys.path.insert(0, str(CONTROLLER_DIR))

import device_license  # noqa: E402


class DeviceLicenseTests(unittest.TestCase):
    def test_source_tree_reads_plain_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.write_bytes(b"plain-policy")
            self.assertEqual(device_license.decrypt_model(str(path)), b"plain-policy")

    def test_wrong_cpu_is_rejected(self):
        expected = hashlib.sha256(
            b"w1w-device-v1\0" + b"1111111111111111"
        ).hexdigest()
        with mock.patch.object(device_license, "AUTHORIZED_CPU_DIGEST_HEX", expected), \
                mock.patch.object(device_license, "_read_cpu_serial", return_value="2222222222222222"), \
                self.assertRaises(SystemExit) as raised:
            device_license.require_authorized("test")
        self.assertEqual(raised.exception.code, 77)

    def test_sealed_model_decrypts_only_on_authorized_cpu(self):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            self.skipTest("cryptography is not installed")

        serial = "1234567890abcdef"
        master = bytes(range(32))
        expected = hashlib.sha256(
            b"w1w-device-v1\0" + serial.encode("ascii")
        ).hexdigest()
        key = hashlib.sha256(
            b"w1w-model-key-v1\0" + master + b"\0" + serial.encode("ascii")
        ).digest()
        nonce = bytes(range(12))
        plaintext = b"encrypted-policy"
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, b"w1w-policy-v1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.write_bytes(b"W1WENC1\0" + nonce + ciphertext)
            with mock.patch.object(device_license, "AUTHORIZED_CPU_DIGEST_HEX", expected), \
                    mock.patch.object(device_license, "MODEL_MASTER_KEY_HEX", master.hex()), \
                    mock.patch.object(device_license, "_read_cpu_serial", return_value=serial):
                self.assertEqual(device_license.decrypt_model(str(path)), plaintext)


if __name__ == "__main__":
    unittest.main()
