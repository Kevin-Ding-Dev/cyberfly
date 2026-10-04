"""Integration checks exercise real MuJoCo transitions, not mocks."""
import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as check_sb3

from .env import CyberflyEnv, TaskConfig


def run_checks():
    env = CyberflyEnv(TaskConfig(episode_seconds=0.2))
    try:
        check_env(env, skip_render_check=True)
        check_sb3(env, warn=True, skip_render_check=True)
        trajectories = []
        for _ in range(2):
            obs, _ = env.reset(seed=123)
            trajectory = [obs]
            for index in range(env.max_steps):
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([0.1, -0.1], dtype=np.float32))
                assert np.isfinite(obs).all() and np.isfinite(reward)
                assert env.observation_space.contains(obs)
                trajectory.append(obs)
                if terminated or truncated:
                    break
            assert index + 1 == env.max_steps and truncated and not terminated
            trajectories.append(np.array(trajectory))
        np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-6, rtol=0)
        env.reset(seed=123)
        try:
            env.step([np.nan, 0])
            raise AssertionError("NaN action was accepted")
        except ValueError:
            pass
        print("PASS: Gymnasium/SB3 contracts, deterministic seeds, finite state, "
              "time-limit truncation, invalid-action rejection.", flush=True)
    finally:
        env.close()
