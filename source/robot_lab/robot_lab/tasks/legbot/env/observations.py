"""Keep the CTS current frame identical to the latest policy history frame."""
import torch
from isaaclab.managers import ObservationManager


def align_policy_observations(observations):
    """Apply the deployed 57 x 4 term-major layout without changing dimensions."""
    policy = observations["policy"]
    if policy.shape[-1] != 228 or observations["single_obs"].shape[-1] != 57:
        raise ValueError("W1W deployment requires 57 observation values and 4 history frames.")
    frames = []
    start = 0
    for index, width in enumerate((3, 3, 3, 16, 16, 16)):
        history = policy[:, start:start + 4 * width].reshape(-1, 4, width)
        if index == 3:
            # Wheels have no position observation, including observation noise.
            history[:, :, 3::4] = 0.0
        frames.append(history[:, -1])
        start += 4 * width
    observations["single_obs"] = torch.cat(frames, dim=-1)
    return observations


class LegbotObservationManager(ObservationManager):
    def compute(self, update_history=False):
        observations = super().compute(update_history=update_history)
        # super() caches this same dictionary, so both wrappers and env.step see
        # the aligned frame, including the first observation after reset.
        return align_policy_observations(observations)
