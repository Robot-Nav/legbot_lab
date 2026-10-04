import struct
import unittest
from pathlib import Path

import numpy as np

from controller.config import Config
from controller import imu_client


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_packaged_configuration(self):
        config = Config(str(ROOT / "controller" / "config.yaml"))
        self.assertEqual(config.num_actions, 16)
        self.assertEqual(config.num_obs, 57)
        self.assertEqual(config.wheel_sim_indices, (3, 7, 11, 15))
        self.assertEqual(config.motor_host, "127.0.0.1")
        self.assertEqual(config.motor_port, 55100)
        self.assertEqual(config.elrs_host, "127.0.0.1")
        self.assertEqual(config.elrs_port, 55201)
        self.assertFalse(config.elrs_emergency_active_high)
        self.assertEqual(config.elrs_timeout_ms, 500.0)
        self.assertEqual(config.controller_send_deadline_ms, 80.0)
        self.assertEqual(config.active_acquire_timeout_ms, 50.0)
        joint_indices = [i for i in range(16) if i not in config.wheel_sim_indices]
        np.testing.assert_allclose(
            config.default_angles,
            [0.0, -0.68, 1.4, 0.0, 0.0, 0.68, -1.4, 0.0,
             0.0, -0.68, 1.4, 0.0, 0.0, 0.68, -1.4, 0.0],
        )
        np.testing.assert_allclose(config.max_cmd, [3.0, 1.0, 3.14])
        self.assertEqual([config.kps[i] for i in config.wheel_sim_indices], [0.0] * 4)
        self.assertEqual([config.kps[i] for i in joint_indices], [100.0] * 12)
        self.assertEqual([config.kds[i] for i in config.wheel_sim_indices], [1.0] * 4)
        self.assertEqual([config.kds[i] for i in joint_indices], [2.0] * 12)

    def test_imu_packet_layout_matches_cpp(self):
        self.assertEqual(imu_client.COMMAND_STRUCT.size, 16)
        self.assertEqual(imu_client.FEEDBACK_STRUCT.size, 72)
        packet = imu_client.FEEDBACK_STRUCT.pack(
            imu_client.FEEDBACK_MAGIC, 1, 0, 1, 2, 3, 0,
            0.0, 0.0, 9.8, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0,
            1.0, 1000.0, 1000.0,
        )
        self.assertEqual(len(packet), 72)


if __name__ == "__main__":
    unittest.main()
