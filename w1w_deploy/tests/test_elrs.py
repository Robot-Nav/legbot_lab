import time
import unittest

from controller import elrs_client as protocol


def feedback_fields(
    feedback_sequence=1,
    command_sequence=7,
    channels=None,
    status=0,
    data_age_ms=2.0,
):
    if channels is None:
        channels = [992] * 16
    packet = protocol.FEEDBACK_STRUCT.pack(
        protocol.FEEDBACK_MAGIC,
        protocol.PROTOCOL_VERSION,
        0,
        feedback_sequence,
        1,
        command_sequence,
        status,
        *channels,
        0x16,
        0,
        data_age_ms,
        496.0,
        1000.0,
    )
    return protocol.FEEDBACK_STRUCT.unpack(packet)


class ElrsProtocolTests(unittest.TestCase):
    def test_packet_sizes_and_wrapping_sequence(self):
        self.assertEqual(protocol.LEGACY_COMMAND_STRUCT.size, 16)
        self.assertEqual(protocol.COMMAND_STRUCT.size, 48)
        self.assertEqual(protocol.FEEDBACK_STRUCT.size, 70)
        self.assertTrue(protocol.sequence_is_newer(0, 0xFFFFFFFF))
        self.assertFalse(protocol.sequence_is_newer(9, 9))
        self.assertFalse(protocol.sequence_is_newer(8, 9))

    def test_motor_temperatures_are_encoded_in_degrees_tenths(self):
        client = protocol.ElrsClient(port=9)
        try:
            client.set_motor_temperatures([30.04 + i for i in range(16)])
            self.assertTrue(client.motor_temperatures_valid)
            self.assertEqual(int(client.motor_temperatures_deci_c[0]), 300)
            self.assertEqual(int(client.motor_temperatures_deci_c[15]), 450)
        finally:
            client.close()

    def test_normalization_is_clamped(self):
        self.assertEqual(protocol.normalize_channel(172), -1.0)
        self.assertEqual(protocol.normalize_channel(1811), 1.0)
        self.assertEqual(protocol.normalize_channel(0), -1.0)
        self.assertEqual(protocol.normalize_channel(2047), 1.0)

    def test_channels_map_to_controller_contract(self):
        client = protocol.ElrsClient(port=9, peer_timeout_ms=500.0)
        try:
            client._recent_commands.append(7)
            channels = [992] * 16
            channels[0] = 172   # wz becomes +1
            channels[2] = 1811  # vx +1
            channels[3] = 172   # vy +1
            channels[5] = 1811  # RL
            channels[6] = 172   # low speed
            channels[7] = 1811  # emergency released (active low)
            channels[8] = 1811  # speed lock
            self.assertTrue(client._accept_feedback(feedback_fields(channels=channels), 1.0))
            self.assertEqual(client.mode_level, 1)
            self.assertEqual(client.speed_level, 0)
            self.assertFalse(client.emergency)
            self.assertTrue(client.speed_lock)
            self.assertEqual(client.get_velocity_command(), (1.0, 1.0, 1.0))
        finally:
            client.close()

    def test_emergency_requires_explicit_released_switch_position(self):
        for active_high, released, active in ((False, 1.0, -1.0), (True, -1.0, 1.0)):
            client = protocol.ElrsClient(port=9, emergency_active_high=active_high)
            try:
                client._ever_fresh = True
                client.channels[7] = released
                self.assertFalse(client.emergency)
                client.channels[7] = 0.0
                self.assertTrue(client.emergency)
                client.channels[7] = active
                self.assertTrue(client.emergency)
            finally:
                client.close()

    def test_startup_defaults_are_safe(self):
        client = protocol.ElrsClient(port=9)
        try:
            self.assertEqual(client.mode_level, -1)
            self.assertEqual(client.speed_level, 0)
            self.assertTrue(client.emergency)
            self.assertTrue(client.speed_lock)
            self.assertTrue(client.data_stale)
        finally:
            client.close()

    def test_echo_and_feedback_sequences_are_checked(self):
        client = protocol.ElrsClient(port=9)
        try:
            client._recent_commands.append(7)
            self.assertFalse(client._accept_feedback(feedback_fields(command_sequence=6), 1.0))
            self.assertTrue(client._accept_feedback(feedback_fields(feedback_sequence=10), 1.0))
            self.assertFalse(client._accept_feedback(feedback_fields(feedback_sequence=10), 1.1))
            self.assertFalse(client._accept_feedback(feedback_fields(feedback_sequence=9), 1.1))
        finally:
            client.close()

    def test_server_stale_and_local_timeout_latch_permanently(self):
        client = protocol.ElrsClient(port=9, peer_timeout_ms=500.0)
        try:
            client._recent_commands.append(7)
            self.assertTrue(client._accept_feedback(feedback_fields(), 1.0))
            self.assertFalse(client.data_stale)
            client._refresh_age(1.501)
            self.assertTrue(client.data_stale)
            self.assertTrue(client.link_timeout_latched)
            client._recent_commands.append(8)
            self.assertTrue(
                client._accept_feedback(
                    feedback_fields(feedback_sequence=2, command_sequence=8), 1.6
                )
            )
            self.assertFalse(client.data_stale)
            self.assertTrue(client.link_timeout_latched)

            client2 = protocol.ElrsClient(port=9)
            try:
                client2._recent_commands.append(7)
                self.assertTrue(client2._accept_feedback(feedback_fields(status=1), 1.0))
                self.assertTrue(client2.data_stale)
            finally:
                client2.close()
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
