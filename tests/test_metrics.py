"""Speed reports must agree with displacement and actual elapsed time."""
import unittest

import numpy as np

from cyberfly.behavior import describe
from cyberfly.foraging import ForageConfig, ForagingEnv


class SpeedMetricTests(unittest.TestCase):
    def test_irregular_samples_are_time_weighted(self):
        rows = [dict(time_s=t, x_mm=x, y_mm=0, heading_rad=0, feeding=0,
                     speed_mm_s=999, filtered_speed_mm_s=888)
                for t, x in ((0, 0), (1, 2), (4, 8), (5, 8))]
        result = describe(rows)
        self.assertAlmostEqual(result["mean_speed_mm_s"], 8/5)
        self.assertAlmostEqual(result["stopped_fraction"], 1/5)

    def test_physical_rollout_summary_matches_trajectory(self):
        env = ForagingEnv(ForageConfig(episode_seconds=0.4, food_min=1, food_max=1))
        try:
            env.reset(seed=12)
            for _ in range(env.max_steps):
                _, _, terminated, truncated, info = env.step(np.zeros(3))
                if terminated or truncated:
                    break
            metrics = info["metrics"]
            rows = env.trajectory
            xy = np.array([[r["x_mm"], r["y_mm"]] for r in rows])
            dt = np.diff([r["time_s"] for r in rows])
            distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            self.assertEqual(metrics["schema_version"], 2)
            self.assertAlmostEqual(metrics["path_length_mm"], distance.sum())
            self.assertAlmostEqual(metrics["mean_speed_mm_s"], distance.sum()/dt.sum())
            self.assertAlmostEqual(metrics["mean_speed_mm_s"], describe(rows)["mean_speed_mm_s"])
            np.testing.assert_allclose([r["speed_mm_s"] for r in rows[1:]], distance/dt)
            self.assertAlmostEqual(metrics["mean_filtered_speed_mm_s"],
                                   np.average([r["filtered_speed_mm_s"] for r in rows[1:]], weights=dt))
            self.assertFalse(np.isclose(metrics["mean_speed_mm_s"], metrics["mean_filtered_speed_mm_s"]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
