import unittest

from scripts import base_health


class BaseHealthTests(unittest.TestCase):
    def test_exact_damping_values(self):
        packet = base_health.damping_packet(9)
        header = base_health.MOTOR_COMMAND_HEADER.unpack_from(packet)
        self.assertEqual(
            header[:4],
            (base_health.COMMAND_MAGIC, base_health.VERSION, 16, 9),
        )
        offset = base_health.MOTOR_COMMAND_HEADER.size
        for index in range(16):
            command = base_health.MOTOR_COMMAND.unpack_from(
                packet, offset + index * base_health.MOTOR_COMMAND.size
            )
            self.assertEqual(command[:4], (0.0, 0.0, 0.0, 0.0))
            self.assertEqual(
                command[4], 1.0 if index in base_health.WHEEL_INDICES else 2.0
            )

    def test_state_names(self):
        self.assertEqual(base_health.state_name(base_health.STATUS_READY), "READY")
        self.assertEqual(base_health.state_name(base_health.STATUS_RUNNING), "RUNNING")
        self.assertEqual(
            base_health.state_name(base_health.STATUS_SAFETY_LATCHED), "LATCHED"
        )


if __name__ == "__main__":
    unittest.main()
