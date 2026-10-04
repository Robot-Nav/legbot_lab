import unittest
import sys
from pathlib import Path

import numpy as np

CONTROLLER = Path(__file__).resolve().parents[1] / "controller"
sys.path.insert(0, str(CONTROLLER))

from deploy import W1WController  # noqa: E402


class ObservationTests(unittest.TestCase):
    def test_projected_gravity_matches_training_deployment(self):
        controller = object.__new__(W1WController)
        controller.gravity = np.zeros(3, dtype=np.float32)
        identity = controller._gravity_from_quaternion(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(identity, [0.0, 0.0, -1.0])

        half = np.sqrt(0.5)
        pitched = controller._gravity_from_quaternion(
            np.array([half, 0.0, half, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(pitched, [1.0, 0.0, 0.0], atol=1e-6)
        rolled = controller._gravity_from_quaternion(
            np.array([half, half, 0.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(rolled, [0.0, -1.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
