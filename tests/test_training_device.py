"""Workload selection, explicit overrides and calibration isolation."""
from io import BytesIO
import random
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cyberfly.training_device import (
    _benchmark_policy, choose_from_timings, select_training_device,
)


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0., False, False, {}


class TrainingDeviceTests(unittest.TestCase):
    def test_accounts_for_inference_and_physics_not_just_update_speed(self):
        times = {"cpu": {"inference_seconds_per_vector_step": .001, "ppo_update_seconds_per_rollout": 2.},
                 "mps": {"inference_seconds_per_vector_step": .003, "ppo_update_seconds_per_rollout": .2}}
        self.assertEqual(choose_from_timings(times, .001, 64)[0], "mps")
        # Many small inference calls negate the faster GPU update.
        self.assertEqual(choose_from_timings(times, .001, 2048)[0], "cpu")
        # A physics-dominated task does not justify MPS for a tiny total gain.
        self.assertEqual(choose_from_timings(times, 1., 64)[0], "cpu")

    def test_near_ties_and_invalid_measurements(self):
        times = {d: {"inference_seconds_per_vector_step": .001, "ppo_update_seconds_per_rollout": 1.}
                 for d in ("cpu", "mps")}
        self.assertEqual(choose_from_timings(times, .01, 64)[0], "cpu")
        times["mps"]["ppo_update_seconds_per_rollout"] = float("nan")
        with self.assertRaises(ValueError):
            choose_from_timings(times, .01, 64)

    def test_auto_without_mps_uses_cpu_without_running_calibration(self):
        with patch("torch.backends.mps.is_available", return_value=False), \
                patch("cyberfly.training_device._sample_physics") as sample:
            result = select_training_device(None, None, "auto")
            self.assertEqual(result["policy_device"], "cpu")
            self.assertEqual(result["requested_device"], "auto")
            self.assertIn("mps_error", result)
            sample.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "No CPU fallback"):
                select_training_device(None, None, "mps")

    def test_explicit_cpu_bypasses_benchmark_and_fallback_flag_disables_auto_mps(self):
        with patch("cyberfly.training_device._benchmark_policy") as bench:
            self.assertEqual(select_training_device(None, None, "cpu")["selection_reason"], "explicit_override")
            with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
                self.assertEqual(select_training_device(None, None)["policy_device"], "cpu")
            bench.assert_not_called()

    def test_real_ppo_benchmark_keeps_source_weights_optimizer_and_steps(self):
        env = DummyVecEnv([TinyEnv])
        try:
            torch.set_num_threads(1)
            model = PPO("MlpPolicy", env, device="cpu", n_steps=8, batch_size=8, n_epochs=1, seed=7)
            model.learn(8)
            weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
            moments = [s["exp_avg"].clone() for s in model.policy.optimizer.state.values()]
            snapshot = BytesIO()
            model.save(snapshot)
            report = _benchmark_policy(snapshot.getvalue(), "cpu", np.ones((3, 1, 4), dtype=np.float32))
            self.assertGreater(report["ppo_update_seconds_per_rollout"], 0)
            self.assertGreater(report["device_audit"]["gradient_tensors"], 0)
            self.assertEqual(model.num_timesteps, 8)
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
            for state, original in zip(model.policy.optimizer.state.values(), moments):
                torch.testing.assert_close(state["exp_avg"], original, rtol=0, atol=0)
        finally:
            env.close()

    def test_mps_kernel_failure_falls_back_and_restores_random_streams(self):
        env = DummyVecEnv([TinyEnv])
        try:
            model = PPO("MlpPolicy", env, device="cpu", n_steps=8, batch_size=8, n_epochs=1, seed=7)
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()

            def benchmark(snapshot, device, frames):
                random.random()
                np.random.random()
                torch.rand(1)
                if device == "mps":
                    raise NotImplementedError("Unsupported test kernel")
                return {"inference_seconds_per_vector_step": .001, "ppo_update_seconds_per_rollout": .1}

            with patch("cyberfly.training_device.require_device", return_value={"policy_device": "cpu", "backward_probe_passed": True}), \
                    patch("cyberfly.training_device._sample_physics", return_value=(np.zeros((3, 1, 4)), .01)), \
                    patch("cyberfly.training_device._benchmark_policy", side_effect=benchmark), \
                    patch("torch.mps.get_rng_state", return_value=torch_state), patch("torch.mps.set_rng_state"):
                report = select_training_device(model, env)
            self.assertEqual(report["policy_device"], "cpu")
            self.assertEqual(report["selection_reason"], "mps_policy_benchmark_failed")
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
