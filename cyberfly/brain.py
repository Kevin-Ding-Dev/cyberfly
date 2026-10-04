"""Functional sensory navigation plus a trainable short-history neural encoder.

This is not a connectome or a claim about named biological brain circuits.
The reflex baseline consumes sensor readings only, never world food coordinates.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

EYE_ANGLES = np.stack((np.linspace(-0.35, 2.8, 24),
                       np.linspace(-2.8, 0.35, 24)))
# 48 angular visual contrast channels, contact taste, internal/proprioceptive state.
SLICES = {
    "vision": slice(0, 48), "odor": slice(48, 50),
    "feet_taste": slice(50, 56), "mouth_taste": slice(56, 57),
    "energy": slice(57, 58), "velocity": slice(58, 61),
    "angular_velocity": slice(61, 64), "up": slice(64, 67),
    "cpg": slice(67, 85), "mouth_extension": slice(85, 86),
    "last_action": slice(86, 89), "recent_food": slice(89, 90),
}
FRAME_SIZE = 90


class SensoryBrain:
    """Search/approach/feeding state machine: an explicitly engineered baseline."""
    def reset(self):
        self.mode = "search"
        self.time = 0.0

    def __init__(self):
        self.reset()

    def act(self, sensors, dt):
        self.time += dt
        visual = sensors[SLICES["vision"]].reshape(2, 24)
        peak = float(visual.max())
        mouth = float(sensors[56])
        feet = float(sensors[50:56].max())
        if sensors[57] >= 0.98:
            self.mode = "satiated"
            return np.array([0.0, 0.0, -1.0])
        if mouth > 0.5:
            self.mode = "feed"
            return np.array([0.0, 0.0, 1.0])
        if peak > 0.015:
            self.mode = "approach"
            strongest = np.unravel_index(visual.argmax(), visual.shape)
            center = EYE_ANGLES[strongest]
            difference = np.arctan2(np.sin(EYE_ANGLES-center), np.cos(EYE_ANGLES-center))
            weights = visual * (np.abs(difference) < 0.4)
            bearing = float(np.arctan2((weights*np.sin(EYE_ANGLES)).sum(),
                                       (weights*np.cos(EYE_ANGLES)).sum()))
            # Retinal apparent size grows with proximity. It is not exact distance.
            forward = 0.75 if peak < 0.20 else 0.35
            forward *= max(0.0, np.cos(bearing))
            turn = np.clip(1.2 * bearing, -0.65, 0.65)
            if abs(bearing) > 0.6:
                forward = 0.2
            mouth_command = 1.0 if (peak > 0.20 or feet > 0.5) else -1.0
            return np.array([forward-turn, forward+turn, mouth_command])
        self.mode = "search"
        turn = 0.28 * np.sin(self.time * 1.5) + 0.16
        return np.array([0.65-turn, 0.65+turn, -1.0])


class SensoryMemoryEncoder(BaseFeaturesExtractor):
    """GRU over four observed frames (~80 ms), trained by PPO.

Hidden state is recomputed from the finite window on each call. This is not
whole-episode recurrent PPO or long-term memory.
"""
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.project = nn.Sequential(nn.Linear(FRAME_SIZE, 64), nn.Tanh())
        self.gru = nn.GRU(64, features_dim, batch_first=True)

    def forward(self, observations):
        _, state = self.gru(self.project(observations))
        return state[-1]
