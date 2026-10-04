import json
import math
import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
import server  # noqa: E402


class WebProtocolTests(unittest.TestCase):
    def test_w190_status_uses_v12_bits(self):
        self.assertEqual(server.decode_faults(0x08, wheel=True), ["过温故障"])

    def test_v2_feedback_contains_wheel_voltage(self):
        bridge = object.__new__(server.MotorBridge)
        bridge._lock = threading.Lock()
        bridge._latest = {}
        bridge._last_feedback_monotonic = 0.0
        status = server.STATUS_READY | server.STATUS_RELEASE_SUPPORTED
        packet = bytearray(server.FB_HEADER.pack(
            server.FEEDBACK_MAGIC, server.PROTOCOL_VERSION, 16,
            1, 0, status, 1429.0, 0.0, 6000.0, 700.0,
        ))
        for index in range(16):
            voltage = 48.5 if index in server.WHEEL_INDICES else math.nan
            packet.extend(server.FB_MOTOR_V2.pack(
                0.0, 0.0, 0.0, 30.0, voltage, 0,
                1 if index in server.WHEEL_INDICES else 2, 0,
            ))
        bridge._parse_feedback(bytes(packet))
        self.assertEqual(bridge._latest["motors"][3]["voltage"], 48.5)
        self.assertIsNone(bridge._latest["motors"][0]["voltage"])
        json.dumps(bridge._latest, allow_nan=False)
        self.assertEqual(bridge._latest["safety_state"], "ready")


if __name__ == "__main__":
    unittest.main()
