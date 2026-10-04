"""Exercise the CLI frame loop without a graphics context or video encoder."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cyberfly.forage_cli import rollout_forage
from cyberfly.foraging import ForageConfig


class RecordingTests(unittest.TestCase):
    def record(self, steps, *, early_stop=None):
        completed = early_stop or steps
        env = Mock()
        env.metadata = {"render_fps":25}
        env.config.control_dt = 0.02
        env.max_steps = steps
        env.food_count = 1
        env.food_positions = np.zeros((1, 3))
        env.camera = Namespace(distance=30, azimuth=100, elevation=-65, lookat=np.zeros(3))
        env.camera_follow_body = False
        env.trajectory = []
        env.events = []
        env.reset.return_value = (np.zeros((4, 90)), {})
        env.render.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        env.step.side_effect = [
            (None, 0, i == completed and bool(early_stop), i == steps,
             {"time_s":i*0.02, "mode":"search", "consumed":0, "food_count":1,
              "energy":0.45, **({"metrics":{"duration_s":i*0.02}} if i==completed else {})})
            for i in range(1, completed+1)]
        writer = Mock()
        with TemporaryDirectory() as folder:
            args = Namespace(model=None, config=None, seconds=steps*0.02, seed=12,
                             viewer=False, video=folder+"/clip.mp4", output=folder)
            with patch("cyberfly.forage_cli.ForagingEnv", return_value=env), \
                    patch("imageio.v2.get_writer", return_value=writer), redirect_stdout(StringIO()):
                result = rollout_forage(args, show_summary=False)
        writer.close.assert_called_once()
        self.assertEqual(writer.append_data.call_count, result["video"]["frames"])
        self.assertFalse(result["default_camera"]["follow_body"])
        self.assertEqual(result["default_camera"]["lookat"], [0, 0, 0])
        env.close.assert_called_once()
        return result["video"]

    def test_full_fifteen_seconds_has_375_frames(self):
        video = self.record(750)
        self.assertEqual(video["frames"], 375)
        self.assertEqual(video["duration_s"], 15)
        self.assertEqual(video["duration_error_s"], 0)
        self.assertEqual(video["frame_times_s"][0], 0.04)
        self.assertEqual(video["frame_times_s"][-1], 15)

    def test_final_partial_frame_preserved_once(self):
        for steps, early in ((1, None), (3, None), (750, 3), (750, 4)):
            with self.subTest(steps=steps, early_stop=early):
                video = self.record(steps, early_stop=early)
                completed = early or steps
                self.assertEqual(video["frames"], (completed+1)//2)
                self.assertEqual(video["frame_times_s"][-1], completed*0.02)
                self.assertEqual(len(video["frame_times_s"]), len(set(video["frame_times_s"])))
                self.assertAlmostEqual(video["duration_error_s"], 0.02 if completed%2 else 0)


if __name__ == "__main__":
    unittest.main()
