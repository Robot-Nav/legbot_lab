import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SystemdSafetyTests(unittest.TestCase):
    def read_unit(self, name):
        return (ROOT / "systemd" / name).read_text(encoding="utf-8")

    def test_motor_service_is_local_and_never_auto_restarts(self):
        unit = self.read_unit("w1w-motor.service")
        self.assertIn("--bind-host 127.0.0.1", unit)
        self.assertIn("--cycle-us 1429 --wheel-cycle-us 1429", unit)
        self.assertIn("--timeout-ms 100", unit)
        self.assertRegex(unit, r"(?m)^Restart=no$")

    def test_imu_and_sealed_services_are_local(self):
        self.assertIn("--bind-host 127.0.0.1", self.read_unit("w1w-imu.service"))
        sealed = ROOT / "sealed_build" / "systemd"
        for name in ("w1w-motor.service", "w1w-imu.service"):
            self.assertIn("--bind-host 127.0.0.1", (sealed / name).read_text(encoding="utf-8"))

    def test_controller_requires_authorization_and_never_auto_restarts(self):
        unit = self.read_unit("w1w-controller.service")
        self.assertIn("ConditionPathExists=/run/w1w/controller-authorized", unit)
        self.assertIn("ExecStartPre=/usr/bin/test -f /run/w1w/controller-authorized", unit)
        self.assertIn("ExecStopPost=+/usr/bin/rm -f /run/w1w/controller-authorized", unit)
        self.assertRegex(unit, r"(?m)^Restart=no$")
        self.assertIn("w1w-elrs.service", unit)

    def test_elrs_service_uses_configured_crsf_serial(self):
        unit = self.read_unit("w1w-elrs.service")
        self.assertIn("SupplementaryGroups=dialout", unit)
        self.assertIn("--device ${W1W_ELRS_DEVICE}", unit)
        self.assertIn("--baud ${W1W_ELRS_BAUD}", unit)
        self.assertIn("--port ${W1W_ELRS_PORT}", unit)

    def test_installer_contains_no_service_start(self):
        script = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"systemctl\s+(start|restart|enable --now)\b", script))


if __name__ == "__main__":
    unittest.main()
