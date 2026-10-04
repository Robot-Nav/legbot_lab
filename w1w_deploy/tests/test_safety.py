import unittest

from controller.safety import PermanentSafetyLatch, command_deadline_missed


class ControllerSafetyTests(unittest.TestCase):
    def test_latch_keeps_first_reason(self):
        latch = PermanentSafetyLatch()
        self.assertFalse(latch.latched)
        self.assertTrue(latch.latch("ELRS stale"))
        self.assertFalse(latch.latch("ELRS recovered"))
        self.assertTrue(latch.latched)
        self.assertEqual(latch.reason, "ELRS stale")

    def test_deadline_applies_only_after_active_control(self):
        self.assertFalse(command_deadline_missed(False, 10.0, 0.0, 80.0))
        self.assertFalse(command_deadline_missed(True, 10.079, 10.0, 80.0))
        self.assertTrue(command_deadline_missed(True, 10.081, 10.0, 80.0))


if __name__ == "__main__":
    unittest.main()
