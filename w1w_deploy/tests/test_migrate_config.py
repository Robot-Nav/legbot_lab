import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.migrate_config import migrate


class ConfigMigrationTests(unittest.TestCase):
    def test_wifi_configuration_is_migrated_without_losing_robot_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.yaml"
            path.write_text(
                "network:\n"
                "  motor: {host: 127.0.0.1, port: 55100}\n"
                "  remote: {bind_host: 0.0.0.0, port: 55210}\n"
                "safety:\n"
                "  remote_timeout_ms: 400\n"
                "robot:\n"
                "  marker: keep-me\n"
                "  default_angles: [0, -0.6, 1, 0, 0, 0.6, -1, 0, 0, 0.6, -1, 0, 0, -0.6, 1, 0]\n"
                "  kp: [80, 80, 80, 0, 80, 80, 80, 0, 80, 80, 80, 0, 80, 80, 80, 0]\n"
                "max_cmd: [2, 1, 2]\n",
                encoding="utf-8",
            )
            self.assertTrue(migrate(path))
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("remote", config["network"])
            self.assertEqual(config["network"]["elrs"]["port"], 55201)
            self.assertEqual(config["safety"]["elrs_timeout_ms"], 400.0)
            self.assertNotIn("remote_timeout_ms", config["safety"])
            self.assertEqual(config["robot"]["marker"], "keep-me")
            self.assertEqual(config["robot"]["default_angles"], [
                0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0,
                0.0, -0.7, 1.5, 0.0, 0.0, 0.7, -1.5, 0.0,
            ])
            self.assertEqual(config["robot"]["kp"], [100, 100, 100, 0] * 4)
            self.assertEqual(config["max_cmd"], [3.0, 1.0, 3.14])
            self.assertTrue(path.with_name("controller.yaml.pre-elrs").is_file())
            self.assertFalse(migrate(path))


if __name__ == "__main__":
    unittest.main()
