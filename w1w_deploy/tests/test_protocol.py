import struct
import unittest

import numpy as np

from controller import motor_client as protocol
from scripts import base_health


class MotorProtocolTests(unittest.TestCase):
    def test_packet_sizes(self):
        self.assertEqual(protocol.COMMAND_PACKET_SIZE, 336)
        self.assertEqual(protocol.FEEDBACK_PACKET_SIZE, 356)
        self.assertEqual(len(base_health.damping_packet(0)), 336)
        self.assertEqual(base_health.IMU_FEEDBACK.size, 72)

    def test_damping_packet(self):
        torque, position, velocity, kp, kd = protocol.damping_arrays()
        packet = protocol.build_command_packet(7, torque, position, velocity, kp, kd)
        self.assertEqual(len(packet), 336)
        self.assertEqual(
            protocol.COMMAND_HEADER.unpack_from(packet)[:4],
            (protocol.COMMAND_MAGIC, protocol.PROTOCOL_VERSION, 16, 7),
        )
        offset = protocol.COMMAND_HEADER.size
        for index in range(16):
            command = protocol.COMMAND_MOTOR.unpack_from(
                packet, offset + index * protocol.COMMAND_MOTOR.size
            )
            self.assertEqual(command[:4], (0.0, 0.0, 0.0, 0.0))
            self.assertEqual(command[4], 1.0 if index in protocol.WHEEL_INDICES else 2.0)

    def test_rejects_non_finite_command(self):
        arrays = [np.zeros(16, dtype=np.float32) for _ in range(5)]
        arrays[0][4] = np.nan
        with self.assertRaises(ValueError):
            protocol.build_command_packet(0, *arrays)

    def test_release_flag_uses_reserved_header_field(self):
        arrays = protocol.damping_arrays()
        packet = protocol.build_command_packet(
            3, *arrays, flags=protocol.COMMAND_FLAG_RELEASE_CONTROL
        )
        self.assertEqual(protocol.COMMAND_HEADER.unpack_from(packet)[4], 1)

    def test_feedback_status_and_motor_order(self):
        status = (
            protocol.STATUS_SAFETY_LATCHED
            | protocol.STATUS_LATCH_MOTOR_OFFLINE
            | (1 << (10 + 1))
        )
        packet = bytearray(protocol.FEEDBACK_HEADER.pack(
            protocol.FEEDBACK_MAGIC, 1, 16, 4, 3, status, 2000.0, 0.0, 6000.0, 500.0
        ))
        for index in range(16):
            packet.extend(protocol.FEEDBACK_MOTOR.pack(
                float(index), float(index + 1), 0.0, 30.0, 0, 2, index
            ))
        feedback = protocol.parse_feedback_packet(bytes(packet))
        self.assertTrue(feedback.safety_latched)
        self.assertEqual(feedback.offline_indices, (10,))
        self.assertEqual(feedback.latch_reasons, ("motor feedback loss",))
        self.assertEqual(feedback.motors[15].position, 15.0)

    def test_active_confirmation_requires_exact_echo(self):
        client = protocol.MotorClient("127.0.0.1", 9)
        motors = tuple(protocol.MotorFeedback(0, 0, 0, 30, 0, 2, 0) for _ in range(16))
        try:
            client.last_feedback = protocol.FeedbackPacket(
                1, 42, protocol.STATUS_RUNNING, 2000, 0, 6000, 500, motors
            )
            client.update_motor_states = lambda: False
            self.assertEqual(client.confirm_active_command(42, 2).command_sequence, 42)
            with self.assertRaises(protocol.MotorSafetyError):
                client.confirm_active_command(43, 2)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
